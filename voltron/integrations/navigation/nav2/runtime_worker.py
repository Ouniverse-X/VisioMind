"""Persistent ROS worker that hosts Nav2 planner/controller servers for Voltron."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from lifecycle_msgs.srv import GetState
from map_msgs.msg import OccupancyGridUpdate
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.srv import GetCostmap
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

R1PRO_NAV_FOOTPRINT = "[[0.24, 0.34], [0.24, -0.34], [-0.40, -0.34], [-0.40, 0.34]]"
R1PRO_NAV_FOOTPRINT_PADDING = 0.02
# Keep a broad, non-lethal cost gradient around walls so Navfn prefers the
# middle of traversable corridors instead of optimizing only geometric path
# length and clipping inside corners.  The footprint remains the source of
# hard collision geometry; this outer radius is deliberately a soft planning
# preference so narrow but footprint-valid doors remain traversable.
NAV2_COSTMAP_INFLATION_RADIUS = 0.75
NAV2_COSTMAP_COST_SCALING_FACTOR = 3.0
NAV2_COSTMAP_UPDATE_BARRIER_TIMEOUT_S = 3.0
NAV2_COSTMAP_FALLBACK_BARRIER_S = 0.6


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _wait_future(future: Any, timeout_s: float) -> bool:
    deadline = time.time() + max(0.1, timeout_s)
    while time.time() < deadline:
        if future.done():
            return True
        time.sleep(0.02)
    return future.done()


def _stamp_nanoseconds(stamp: Any) -> int | None:
    if stamp is None:
        return None
    try:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None


class Nav2RuntimeWorker(Node):
    def __init__(self) -> None:
        super().__init__("voltron_nav2_runtime_worker")
        self._lock = threading.Lock()
        self._frame_id = "map"
        self._odom_frame = "odom"
        self._base_frame = "base_link"
        self._action_name = "compute_path_to_pose"
        self._scene_id: str | None = None
        self._map_msg: OccupancyGrid | None = None
        self._map_revision = 0
        self._current_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._current_twist = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._latest_cmd_vel = {"linear": {"x": 0.0, "y": 0.0}, "angular": {"z": 0.0}}
        self._follow_status = "idle"
        self._follow_goal_handle: Any = None
        self._follow_result_future: Any = None
        log_root = (
            Path(os.environ.get("VOLTRON_HOME", Path(__file__).resolve().parents[3]))
            / "logs"
        )
        log_root.mkdir(parents=True, exist_ok=True)
        self._runtime_dir = Path(
            tempfile.mkdtemp(prefix="nav2_runtime_", dir=str(log_root))
        )
        self._params_path = self._runtime_dir / "nav2_params.yaml"
        self._log_paths = {
            "planner_server": self._runtime_dir / "planner_server.log",
            "controller_server": self._runtime_dir / "controller_server.log",
            "lifecycle_manager": self._runtime_dir / "lifecycle_manager.log",
        }
        self._log_files: dict[str, Any] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._map_pub = self.create_publisher(OccupancyGrid, "map", map_qos)
        self._map_update_pub = self.create_publisher(
            OccupancyGridUpdate, "map_updates", 10
        )
        self._global_costmap_client = self.create_client(
            GetCostmap, "/global_costmap/get_costmap"
        )
        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._cmd_vel_sub = self.create_subscription(
            Twist, "cmd_vel", self._on_cmd_vel, 10
        )
        self._tf_pub = TransformBroadcaster(self)
        self._state_timer = self.create_timer(0.05, self._publish_state)
        self._path_client = ActionClient(self, ComputePathToPose, self._action_name)
        self._follow_client = ActionClient(self, FollowPath, "follow_path")

    def close(self) -> None:
        self._cancel_follow_path()
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self._processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        self._processes.clear()
        for handle in self._log_files.values():
            try:
                handle.close()
            except Exception:
                pass
        self._log_files.clear()
        # Keep planner/controller logs under VOLTRON_HOME/logs for post-run diagnosis.

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._scene_id = str(payload.get("scene_id") or "").strip() or None
            self._frame_id = str(payload.get("frame_id") or "map")
            self._action_name = str(
                payload.get("action_name") or "compute_path_to_pose"
            )
            self._map_msg = self._build_map_message(payload["map"])
            self._map_revision += 1
            map_publish_stamp_ns = self._publish_map()
            self._write_params_file()
            if not self._processes:
                self._launch_processes()

        self._path_client = ActionClient(self, ComputePathToPose, self._action_name)
        self._follow_client = ActionClient(self, FollowPath, "follow_path")
        if not self._path_client.wait_for_server(timeout_sec=20.0):
            return {
                "status": "error",
                "error": "planner_server_unavailable",
                "logs": self._collect_log_tails(),
            }
        if not self._follow_client.wait_for_server(timeout_sec=20.0):
            return {
                "status": "error",
                "error": "controller_server_unavailable",
                "logs": self._collect_log_tails(),
            }
        if not self._wait_for_node_active("planner_server", timeout_s=20.0):
            return {
                "status": "error",
                "error": "planner_server_inactive",
                "logs": self._collect_log_tails(),
            }
        if not self._wait_for_node_active("controller_server", timeout_s=20.0):
            return {
                "status": "error",
                "error": "controller_server_inactive",
                "logs": self._collect_log_tails(),
            }
        barrier = self._wait_for_global_costmap_update(
            map_publish_stamp_ns,
            timeout_s=NAV2_COSTMAP_UPDATE_BARRIER_TIMEOUT_S,
        )
        if barrier is None:
            return {
                "status": "error",
                "error": "global_costmap_initialization_timeout",
                "logs": self._collect_log_tails(),
            }
        return {
            "status": "ok",
            "scene_id": self._scene_id,
            "runtime_dir": str(self._runtime_dir),
            "map_revision": self._map_revision,
            "costmap_sync": barrier,
        }

    def update_map(self, payload: dict[str, Any]) -> dict[str, Any]:
        update = payload.get("update")
        if not isinstance(update, dict):
            return {"status": "error", "error": "map_update_missing"}
        with self._lock:
            if self._map_msg is None:
                return {"status": "error", "error": "map_not_configured"}
            x_coord = int(update.get("x", 0))
            y_coord = int(update.get("y", 0))
            width = int(update.get("width", 0))
            height = int(update.get("height", 0))
            map_width = int(self._map_msg.info.width)
            map_height = int(self._map_msg.info.height)
            data = [int(value) for value in update.get("data", [])]
            if width <= 0 or height <= 0 or len(data) != width * height:
                return {"status": "error", "error": "map_update_shape_invalid"}
            if (
                x_coord < 0
                or y_coord < 0
                or x_coord + width > map_width
                or y_coord + height > map_height
            ):
                return {"status": "error", "error": "map_update_out_of_bounds"}
            map_data = list(self._map_msg.data)
            for local_row in range(height):
                source_start = local_row * width
                target_start = (y_coord + local_row) * map_width + x_coord
                map_data[target_start : target_start + width] = data[
                    source_start : source_start + width
                ]
            self._map_msg.data = map_data
            self._map_revision += 1

            update_msg = OccupancyGridUpdate()
            update_msg.header.frame_id = self._frame_id
            update_msg.header.stamp = self.get_clock().now().to_msg()
            update_msg.x = x_coord
            update_msg.y = y_coord
            update_msg.width = width
            update_msg.height = height
            update_msg.data = data
            self._map_update_pub.publish(update_msg)
            update_stamp_ns = _stamp_nanoseconds(update_msg.header.stamp)
        barrier = self._wait_for_global_costmap_update(
            update_stamp_ns,
            timeout_s=NAV2_COSTMAP_UPDATE_BARRIER_TIMEOUT_S,
        )
        if barrier is None:
            return {
                "status": "error",
                "error": "global_costmap_update_timeout",
                "map_revision": self._map_revision,
                "dirty_bounds": update.get("dirty_bounds"),
            }
        return {
            "status": "ok",
            "scene_id": self._scene_id,
            "map_revision": self._map_revision,
            "changed_cell_count": int(update.get("changed_cell_count", len(data))),
            "dirty_bounds": update.get("dirty_bounds"),
            "costmap_sync": barrier,
        }

    def set_pose(self, payload: dict[str, Any]) -> dict[str, Any]:
        pose = payload.get("pose", {})
        twist = payload.get("twist", {})
        with self._lock:
            self._current_pose = {
                "x": float(pose.get("x", 0.0)),
                "y": float(pose.get("y", 0.0)),
                "yaw": float(pose.get("yaw", 0.0)),
            }
            self._current_twist = {
                "vx": float(twist.get("vx", 0.0)),
                "vy": float(twist.get("vy", 0.0)),
                "wz": float(twist.get("wz", 0.0)),
            }
        return {"status": "ok"}

    def compute_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout_s = float(payload.get("timeout_s", 8.0))
        goal_msg = ComputePathToPose.Goal()
        goal_msg.start = self._build_pose(
            payload["start"], frame_id=str(payload.get("frame_id") or self._frame_id)
        )
        goal_msg.goal = self._build_pose(
            payload["goal"], frame_id=str(payload.get("frame_id") or self._frame_id)
        )
        goal_msg.use_start = True
        goal_msg.planner_id = str(payload.get("planner_id") or "")

        goal_future = self._path_client.send_goal_async(goal_msg)
        if not _wait_future(goal_future, timeout_s):
            return {"status": "error", "error": "compute_path_send_goal_timeout"}
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"status": "error", "error": "compute_path_goal_rejected"}

        result_future = goal_handle.get_result_async()
        if not _wait_future(result_future, timeout_s):
            return {"status": "error", "error": "compute_path_result_timeout"}
        result = result_future.result()
        if result is None or result.result is None:
            return {"status": "error", "error": "compute_path_result_missing"}
        return {
            "status": "ok",
            "points": [
                {
                    "x": float(item.pose.position.x),
                    "y": float(item.pose.position.y),
                }
                for item in getattr(result.result.path, "poses", [])
            ],
        }

    def follow_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._cancel_follow_path()
        timeout_s = float(payload.get("timeout_s", 8.0))
        points = payload.get("points") or []
        if not points:
            return {"status": "error", "error": "follow_path_points_missing"}

        goal_msg = FollowPath.Goal()
        goal_msg.path = self._build_path(
            points=points, frame_id=str(payload.get("frame_id") or self._frame_id)
        )
        goal_msg.controller_id = "FollowPath"
        goal_msg.goal_checker_id = "general_goal_checker"

        goal_future = self._follow_client.send_goal_async(goal_msg)
        if not _wait_future(goal_future, timeout_s):
            return {"status": "error", "error": "follow_path_send_goal_timeout"}
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"status": "error", "error": "follow_path_goal_rejected"}

        self._follow_goal_handle = goal_handle
        self._follow_result_future = goal_handle.get_result_async()
        self._follow_status = "active"
        return {"status": "ok", "follow_status": self._follow_status}

    def get_cmd_vel(self) -> dict[str, Any]:
        if self._follow_result_future is not None and self._follow_result_future.done():
            result = self._follow_result_future.result()
            status = getattr(result, "status", GoalStatus.STATUS_UNKNOWN)
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._follow_status = "succeeded"
            elif status in {
                GoalStatus.STATUS_ABORTED,
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_UNKNOWN,
            }:
                self._follow_status = "failed"
            self._follow_goal_handle = None
            self._follow_result_future = None
            self._latest_cmd_vel = {
                "linear": {"x": 0.0, "y": 0.0},
                "angular": {"z": 0.0},
            }

        return {
            "status": "ok",
            "cmd_vel": self._latest_cmd_vel,
            "follow_status": self._follow_status,
            "scene_id": self._scene_id,
        }

    def cancel_follow_path(self) -> dict[str, Any]:
        self._cancel_follow_path()
        return {"status": "ok"}

    def _cancel_follow_path(self) -> None:
        if self._follow_goal_handle is None:
            self._follow_status = "idle"
            return
        cancel_future = self._follow_goal_handle.cancel_goal_async()
        _wait_future(cancel_future, 2.0)
        self._follow_goal_handle = None
        self._follow_result_future = None
        self._follow_status = "idle"

    def _wait_for_node_active(self, node_name: str, *, timeout_s: float) -> bool:
        service_name = f"/{node_name}/get_state"
        client = self.create_client(GetState, service_name)
        try:
            deadline = time.time() + max(0.1, timeout_s)
            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                if not client.wait_for_service(timeout_sec=min(0.5, remaining)):
                    continue
                future = client.call_async(GetState.Request())
                if not _wait_future(future, min(0.5, remaining)):
                    continue
                response = future.result()
                current_state = getattr(response, "current_state", None)
                label = str(getattr(current_state, "label", "") or "").strip().lower()
                if label == "active":
                    return True
                time.sleep(0.1)
            return False
        finally:
            self.destroy_client(client)

    def _wait_for_global_costmap_update(
        self,
        minimum_stamp_ns: int | None,
        *,
        timeout_s: float,
    ) -> str | None:
        """Wait until the global costmap reports a revision after the map publish."""
        if minimum_stamp_ns is None:
            time.sleep(NAV2_COSTMAP_FALLBACK_BARRIER_S)
            return "timed_fallback"
        deadline = time.time() + max(0.1, timeout_s)
        not_before = time.time() + NAV2_COSTMAP_FALLBACK_BARRIER_S
        if not self._global_costmap_client.wait_for_service(timeout_sec=0.5):
            time.sleep(NAV2_COSTMAP_FALLBACK_BARRIER_S)
            return "timed_fallback"
        while time.time() < deadline:
            future = self._global_costmap_client.call_async(GetCostmap.Request())
            if not _wait_future(future, min(0.5, max(0.1, deadline - time.time()))):
                continue
            response = future.result()
            costmap = getattr(response, "map", None)
            header = getattr(costmap, "header", None)
            stamp_ns = _stamp_nanoseconds(getattr(header, "stamp", None))
            if (
                stamp_ns is not None
                and stamp_ns >= minimum_stamp_ns
                and time.time() >= not_before
            ):
                return "global_costmap_stamp"
            time.sleep(0.05)
        return None

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._latest_cmd_vel = {
            "linear": {"x": float(msg.linear.x), "y": float(msg.linear.y)},
            "angular": {"z": float(msg.angular.z)},
        }

    def _publish_state(self) -> None:
        with self._lock:
            pose = dict(self._current_pose)
            twist = dict(self._current_twist)
        now = self.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = self._frame_id
        transform.child_frame_id = self._odom_frame
        transform.transform.rotation.w = 1.0
        self._tf_pub.sendTransform(transform)

        base_transform = TransformStamped()
        base_transform.header.stamp = now
        base_transform.header.frame_id = self._odom_frame
        base_transform.child_frame_id = self._base_frame
        base_transform.transform.translation.x = float(pose["x"])
        base_transform.transform.translation.y = float(pose["y"])
        base_transform.transform.translation.z = 0.0
        quat = _quaternion_from_yaw(float(pose["yaw"]))
        base_transform.transform.rotation.x = quat[0]
        base_transform.transform.rotation.y = quat[1]
        base_transform.transform.rotation.z = quat[2]
        base_transform.transform.rotation.w = quat[3]
        self._tf_pub.sendTransform(base_transform)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = float(pose["x"])
        odom.pose.pose.position.y = float(pose["y"])
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        odom.twist.twist.linear.x = float(twist["vx"])
        odom.twist.twist.linear.y = float(twist["vy"])
        odom.twist.twist.angular.z = float(twist["wz"])
        self._odom_pub.publish(odom)

    def _publish_map(self) -> int | None:
        if self._map_msg is not None:
            self._map_msg.header.stamp = self.get_clock().now().to_msg()
            self._map_pub.publish(self._map_msg)
            return _stamp_nanoseconds(self._map_msg.header.stamp)
        return None

    def _build_map_message(self, payload: dict[str, Any]) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = self._frame_id
        msg.info.resolution = float(payload["resolution"])
        msg.info.width = int(payload["width"])
        msg.info.height = int(payload["height"])
        origin = payload.get("origin", {})
        msg.info.origin.position.x = float(origin.get("x", 0.0))
        msg.info.origin.position.y = float(origin.get("y", 0.0))
        quat = _quaternion_from_yaw(float(origin.get("yaw", 0.0)))
        msg.info.origin.orientation.x = quat[0]
        msg.info.origin.orientation.y = quat[1]
        msg.info.origin.orientation.z = quat[2]
        msg.info.origin.orientation.w = quat[3]
        msg.data = [int(value) for value in payload.get("data", [])]
        return msg

    def _build_pose(self, payload: dict[str, Any], *, frame_id: str) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(payload["x"])
        pose.pose.position.y = float(payload["y"])
        pose.pose.position.z = 0.0
        quat = _quaternion_from_yaw(float(payload.get("yaw", 0.0)))
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def _build_path(self, *, points: list[dict[str, Any]], frame_id: str) -> NavPath:
        path = NavPath()
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        headings: list[float] = []
        for index, point in enumerate(points):
            if index + 1 < len(points):
                nxt = points[index + 1]
                headings.append(
                    math.atan2(
                        float(nxt["y"]) - float(point["y"]),
                        float(nxt["x"]) - float(point["x"]),
                    )
                )
            elif headings:
                headings.append(headings[-1])
            else:
                headings.append(0.0)

        for point, yaw in zip(points, headings):
            pose = self._build_pose(
                {"x": point["x"], "y": point["y"], "yaw": yaw}, frame_id=frame_id
            )
            path.poses.append(pose)
        return path

    def _launch_processes(self) -> None:
        planner_exec = (
            Path(get_package_prefix("nav2_planner"))
            / "lib"
            / "nav2_planner"
            / "planner_server"
        )
        controller_exec = (
            Path(get_package_prefix("nav2_controller"))
            / "lib"
            / "nav2_controller"
            / "controller_server"
        )
        lifecycle_exec = (
            Path(get_package_prefix("nav2_lifecycle_manager"))
            / "lib"
            / "nav2_lifecycle_manager"
            / "lifecycle_manager"
        )
        executables = {
            "planner_server": planner_exec,
            "controller_server": controller_exec,
            "lifecycle_manager": lifecycle_exec,
        }
        for name, executable in executables.items():
            if not executable.is_file():
                raise RuntimeError(f"Required Nav2 executable missing: {executable}")

        for name, executable in executables.items():
            log_file = self._log_paths[name].open("a", encoding="utf-8")
            self._log_files[name] = log_file
            command = [
                str(executable),
                "--ros-args",
                "--params-file",
                str(self._params_path),
            ]
            if name == "lifecycle_manager":
                command.extend(["-r", "__node:=lifecycle_manager"])
            self._processes[name] = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=dict(os.environ),
            )
            time.sleep(0.4)

    def _write_params_file(self) -> None:
        params = {
            "planner_server": {
                "ros__parameters": {
                    "use_sim_time": False,
                    "expected_planner_frequency": 5.0,
                    "planner_plugins": ["GridBased"],
                    "GridBased": {
                        "plugin": "nav2_navfn_planner/NavfnPlanner",
                        "tolerance": 0.25,
                        "use_astar": True,
                        "allow_unknown": True,
                    },
                }
            },
            "controller_server": {
                "ros__parameters": {
                    "use_sim_time": False,
                    "controller_frequency": 10.0,
                    "odom_topic": "odom",
                    "progress_checker_plugin": "progress_checker",
                    "goal_checker_plugins": ["general_goal_checker"],
                    "controller_plugins": ["FollowPath"],
                    "min_x_velocity_threshold": 0.001,
                    "min_y_velocity_threshold": 0.001,
                    "min_theta_velocity_threshold": 0.001,
                    "progress_checker": {
                        "plugin": "nav2_controller::SimpleProgressChecker",
                        "required_movement_radius": 0.05,
                        "movement_time_allowance": 10.0,
                    },
                    "general_goal_checker": {
                        "plugin": "nav2_controller::SimpleGoalChecker",
                        "xy_goal_tolerance": 0.25,
                        "yaw_goal_tolerance": 0.35,
                        "stateful": True,
                    },
                    "FollowPath": {
                        "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
                        "desired_linear_vel": 0.45,
                        "lookahead_dist": 0.5,
                        "min_lookahead_dist": 0.25,
                        "max_lookahead_dist": 0.9,
                        "lookahead_time": 1.0,
                        "rotate_to_heading_angular_vel": 0.9,
                        "transform_tolerance": 0.2,
                        "use_velocity_scaled_lookahead_dist": False,
                        "min_approach_linear_velocity": 0.05,
                        "approach_velocity_scaling_dist": 0.6,
                        "use_collision_detection": False,
                        "allow_reversing": False,
                        "use_rotate_to_heading": True,
                    },
                }
            },
            "local_costmap": {
                "local_costmap": {
                    "ros__parameters": {
                        "use_sim_time": False,
                        "update_frequency": 10.0,
                        "publish_frequency": 2.0,
                        "global_frame": self._odom_frame,
                        "robot_base_frame": self._base_frame,
                        "rolling_window": True,
                        "width": 6,
                        "height": 6,
                        "resolution": 0.1,
                        "track_unknown_space": False,
                        "footprint": R1PRO_NAV_FOOTPRINT,
                        "footprint_padding": R1PRO_NAV_FOOTPRINT_PADDING,
                        "plugins": ["static_layer", "inflation_layer"],
                        "static_layer": {
                            "plugin": "nav2_costmap_2d::StaticLayer",
                            "map_subscribe_transient_local": True,
                            "subscribe_to_updates": True,
                        },
                        "inflation_layer": {
                            "plugin": "nav2_costmap_2d::InflationLayer",
                            "inflation_radius": NAV2_COSTMAP_INFLATION_RADIUS,
                            "cost_scaling_factor": NAV2_COSTMAP_COST_SCALING_FACTOR,
                        },
                        "always_send_full_costmap": False,
                    }
                }
            },
            "global_costmap": {
                "global_costmap": {
                    "ros__parameters": {
                        "use_sim_time": False,
                        "update_frequency": 5.0,
                        "publish_frequency": 1.0,
                        "global_frame": self._frame_id,
                        "robot_base_frame": self._base_frame,
                        "track_unknown_space": False,
                        "footprint": R1PRO_NAV_FOOTPRINT,
                        "footprint_padding": R1PRO_NAV_FOOTPRINT_PADDING,
                        "resolution": 0.1,
                        "plugins": ["static_layer", "inflation_layer"],
                        "static_layer": {
                            "plugin": "nav2_costmap_2d::StaticLayer",
                            "map_subscribe_transient_local": True,
                            "subscribe_to_updates": True,
                        },
                        "inflation_layer": {
                            "plugin": "nav2_costmap_2d::InflationLayer",
                            "inflation_radius": NAV2_COSTMAP_INFLATION_RADIUS,
                            "cost_scaling_factor": NAV2_COSTMAP_COST_SCALING_FACTOR,
                        },
                        "always_send_full_costmap": False,
                    }
                }
            },
            "lifecycle_manager": {
                "ros__parameters": {
                    "use_sim_time": False,
                    "autostart": True,
                    "node_names": ["planner_server", "controller_server"],
                }
            },
        }
        self._params_path.write_text(
            yaml.safe_dump(params, sort_keys=False), encoding="utf-8"
        )

    def _collect_log_tails(self) -> dict[str, str]:
        tails: dict[str, str] = {}
        for name, path in self._log_paths.items():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            tails[name] = "\n".join(text.splitlines()[-20:])
        return tails


def main() -> int:
    rclpy.init(args=None)
    worker = Nav2RuntimeWorker()
    executor = MultiThreadedExecutor()
    executor.add_node(worker)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
                cmd = str(payload.get("cmd") or "").strip()
                if cmd == "configure":
                    response = worker.configure(payload)
                elif cmd == "update_map":
                    response = worker.update_map(payload)
                elif cmd == "set_pose":
                    response = worker.set_pose(payload)
                elif cmd == "compute_path":
                    response = worker.compute_path(payload)
                elif cmd == "follow_path":
                    response = worker.follow_path(payload)
                elif cmd == "get_cmd_vel":
                    response = worker.get_cmd_vel()
                elif cmd == "cancel_follow_path":
                    response = worker.cancel_follow_path()
                elif cmd == "shutdown":
                    response = {"status": "ok"}
                    print(json.dumps(response), flush=True)
                    break
                else:
                    response = {"status": "error", "error": f"unknown_command:{cmd}"}
            except Exception as exc:
                response = {"status": "error", "error": str(exc)}
            print(json.dumps(response), flush=True)
    finally:
        worker.close()
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        worker.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
