from __future__ import annotations

import json
import math
import sys
from typing import Any


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def main() -> int:
    request = json.load(sys.stdin)

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import ComputePathToPose
        from rclpy.action import ActionClient
        from rclpy.node import Node
    except Exception as exc:
        print(json.dumps({"error": f"nav2_import_failed: {exc}"}))
        return 2

    class _ComputePathNode(Node):
        def __init__(self, action_name: str) -> None:
            super().__init__("voltron_nav2_compute_path_worker")
            self._client = ActionClient(self, ComputePathToPose, action_name)

        def build_pose(self, xy: dict[str, Any], frame_id: str) -> PoseStamped:
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(xy["x"])
            pose.pose.position.y = float(xy["y"])
            pose.pose.position.z = 0.0
            quat = _quaternion_from_yaw(float(xy.get("yaw", 0.0)))
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            return pose

        def compute(self, payload: dict[str, Any]) -> dict[str, Any]:
            timeout_s = float(payload.get("timeout_s", 8.0))
            if not self._client.wait_for_server(timeout_sec=timeout_s):
                return {"error": "nav2_action_server_unavailable"}

            goal_msg = ComputePathToPose.Goal()
            goal_msg.goal = self.build_pose(payload["goal"], payload["frame_id"])
            goal_msg.start = self.build_pose(payload["start"], payload["frame_id"])
            goal_msg.use_start = True
            goal_msg.planner_id = str(payload.get("planner_id") or "")

            goal_future = self._client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, goal_future, timeout_sec=timeout_s)
            if not goal_future.done():
                return {"error": "nav2_send_goal_timeout"}

            goal_handle = goal_future.result()
            if goal_handle is None or not goal_handle.accepted:
                return {"error": "nav2_goal_rejected"}

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_s)
            if not result_future.done():
                return {"error": "nav2_result_timeout"}

            result = result_future.result()
            if result is None:
                return {"error": "nav2_result_missing"}

            poses = getattr(result.result.path, "poses", [])
            return {
                "points": [
                    {
                        "x": float(pose.pose.position.x),
                        "y": float(pose.pose.position.y),
                    }
                    for pose in poses
                ],
                "pose_count": len(poses),
            }

    rclpy.init(args=None)
    node = _ComputePathNode(str(request.get("action_name") or "compute_path_to_pose"))
    try:
        payload = node.compute(request)
        print(json.dumps(payload))
        return 0 if "error" not in payload else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
