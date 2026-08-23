"""Environment bootstrap and wrapper traversal helpers for BEHAVIOR."""

from __future__ import annotations

import importlib
import sys
from types import MethodType
from pathlib import Path
from typing import Any, Callable


def ensure_env(
    *,
    current_env: Any,
    env_factory: Callable[[], Any] | None,
    env_id: str,
    env_kwargs: dict[str, Any],
    auto_register: bool,
    import_gymnasium: Callable[[], Any],
    register_behavior_envs_if_needed: Callable[[Any], None],
    is_behavior_env: Callable[[str], bool],
    install_behavior_rgb_wrapper_fallback: Callable[[], None],
) -> Any:
    if current_env is not None:
        return current_env

    if env_factory is not None:
        return env_factory()

    gym = import_gymnasium()
    if auto_register:
        register_behavior_envs_if_needed(gym)
    if is_behavior_env(env_id):
        install_behavior_rgb_wrapper_fallback()

    env_make_kwargs = dict(env_kwargs)
    for runtime_only_key in (
        "nav2_trav_map_filename",
        "scene_state_include_aabb",
        "scene_state_navigation_role_overrides",
        "scene_vertical_axis",
        "simulator_vertical_axis",
        "scene_from_simulator_transform",
        "portal_annotations",
        "transition_portals",
        "portals",
        "post_reset_robot_position",
        "post_reset_robot_orientation",
        "post_reset_object_states",
        "post_reset_robot_joint_positions",
        "post_reset_robot_joint_velocities",
        "post_reset_refresh_observation",
        "post_reset_settle_steps",
        "recording_third_person_local_offset",
        "recording_third_person_look_at_offset",
        "recording_third_person_prefer_live_capture",
    ):
        env_make_kwargs.pop(runtime_only_key, None)
    return gym.make(env_id, **env_make_kwargs)


def configure_recording_third_person_pose(
    env: Any,
    *,
    local_offset: Any,
    look_at_offset: Any,
) -> bool:
    """Override the GR00T recording camera with a robot-relative look-at pose."""
    wrapper = env
    seen: set[int] = set()
    while wrapper is not None and id(wrapper) not in seen:
        seen.add(id(wrapper))
        if callable(getattr(wrapper, "_recording_third_person_sensor", None)) and hasattr(
            wrapper, "robot"
        ):
            break
        wrapper = getattr(wrapper, "env", None)
    else:
        return False

    module = importlib.import_module(type(wrapper).__module__)
    th = getattr(module, "th", None)
    transform_utils = getattr(module, "T", None)
    if th is None or transform_utils is None:
        return False

    configured_offset = th.as_tensor(local_offset, dtype=th.float32)
    configured_target = th.as_tensor(look_at_offset, dtype=th.float32)

    def _sync_recording_third_person_sensor_pose(self: Any) -> None:
        sensor = self._recording_third_person_sensor()
        if sensor is None:
            return
        try:
            robot_pos, robot_quat = self.robot.get_position_orientation()
            robot_pos = th.as_tensor(robot_pos, dtype=th.float32).detach().cpu()
            robot_quat = th.as_tensor(robot_quat, dtype=th.float32).detach().cpu()
            robot_rotation = transform_utils.quat2mat(robot_quat)
            camera_pos = robot_pos + robot_rotation @ configured_offset
            target_pos = robot_pos + robot_rotation @ configured_target

            forward = target_pos - camera_pos
            forward = forward / th.linalg.norm(forward).clamp_min(1e-6)
            world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
            right = th.linalg.cross(forward, world_up)
            if float(th.linalg.norm(right)) < 1e-6:
                world_up = th.tensor([0.0, 1.0, 0.0], dtype=th.float32)
                right = th.linalg.cross(forward, world_up)
            right = right / th.linalg.norm(right).clamp_min(1e-6)
            backward = -forward
            up = th.linalg.cross(backward, right)
            camera_rotation = th.stack((right, up, backward), dim=1)
            camera_quat = transform_utils.mat2quat(camera_rotation)
            sensor.set_position_orientation(
                position=camera_pos,
                orientation=camera_quat,
                frame="scene",
            )
        except Exception:
            return

    wrapper._sync_recording_third_person_sensor_pose = MethodType(
        _sync_recording_third_person_sensor_pose, wrapper
    )
    return True


def register_behavior_envs_if_needed(
    *,
    registered: bool,
    gym: Any,
    env_id: str,
    load_behavior_register_fn: Callable[[], Callable[[], None]],
) -> bool:
    if registered:
        return True

    try:
        gym.spec(env_id)
        return True
    except Exception:
        pass

    register_fn = load_behavior_register_fn()
    try:
        register_fn()
    except Exception:
        pass

    gym.spec(env_id)
    return True


def import_gymnasium() -> Any:
    try:
        return importlib.import_module("gymnasium")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gymnasium is required for BehaviorRuntimeEnvironment. "
            "Install Isaac-GR00T simulator dependencies first."
        ) from exc


def load_behavior_register_fn(
    *, load_behavior_module: Callable[[], Any]
) -> Callable[[], None]:
    module = load_behavior_module()
    return getattr(module, "register_behavior_envs")


def load_behavior_module(
    *,
    runtime_bridge_file: str,
    prepend_local_gr00t_repo: Callable[[Path], None],
) -> Any:
    try:
        return importlib.import_module("gr00t.eval.sim.BEHAVIOR.behavior_env")
    except ModuleNotFoundError:
        repo_root = Path(runtime_bridge_file).resolve().parents[2]
        local_gr00t_root = repo_root / "isaac_gr00t_learn"
        if local_gr00t_root.exists():
            prepend_local_gr00t_repo(local_gr00t_root)
        return importlib.import_module("gr00t.eval.sim.BEHAVIOR.behavior_env")


def prepend_local_gr00t_repo(local_gr00t_root: Path) -> None:
    preferred_paths: list[str] = []
    try:
        import bddl

        bddl_file = getattr(bddl, "__file__", None)
        if isinstance(bddl_file, str) and bddl_file:
            preferred_paths.append(str(Path(bddl_file).resolve().parents[1]))
    except Exception:
        pass

    preferred_paths.append(str(local_gr00t_root))
    for candidate in reversed(preferred_paths):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)


def install_behavior_rgb_wrapper_fallback(
    *, load_behavior_module: Callable[[], Any]
) -> None:
    try:
        module = load_behavior_module()
    except Exception:
        return

    wrapper_cls = getattr(module, "RGBLowResWrapper", None)
    if wrapper_cls is None or getattr(wrapper_cls, "_voltron_safe_patch", False):
        return

    original_init = wrapper_cls.__init__
    camera_names = dict(getattr(module, "ROBOT_CAMERA_NAMES", {}).get("R1Pro", {}))
    gym_module = getattr(module, "gym", None)
    wrapper_base = getattr(gym_module, "Wrapper", None)

    def _safe_init(self, env: Any) -> None:
        try:
            original_init(self, env)
            return
        except TypeError as exc:
            if "Invalid NodeObj object in Py_Node in getAttributes" not in str(exc):
                raise

        if wrapper_base is not None and getattr(self, "env", None) is None:
            wrapper_base.__init__(self, env)
        elif getattr(self, "env", None) is None:
            self.env = env

        try:
            robot = env.robots[0]
            for camera_id, camera_name in camera_names.items():
                sensor_name = str(camera_name).split("::")[1]
                sensor = robot.sensors.get(sensor_name)
                if sensor is None:
                    continue
                if camera_id == "head":
                    sensor.horizontal_aperture = 40.0
            env.load_observation_space()
        except Exception:
            pass

    wrapper_cls.__init__ = _safe_init
    wrapper_cls._voltron_safe_patch = True


def call_env_method(env: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    if env is None:
        return None

    candidates: list[Any] = []
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        candidates.append(current)
        seen.add(id(current))
        current = getattr(current, "env", None)

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and id(unwrapped) not in seen:
        candidates.append(unwrapped)

    for candidate in candidates:
        method = getattr(candidate, method_name, None)
        if callable(method):
            return method(*args, **kwargs)
    return None
