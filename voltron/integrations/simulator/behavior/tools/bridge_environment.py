"""Environment/bootstrap bindings for the BEHAVIOR runtime facade."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np

from voltron.integrations.manipulation.openpi_comet.observation_adapter import (
    OpenPICometObservationAdapter,
)
from voltron.integrations.simulator.behavior.environment import client as behavior_environment_client
from voltron.integrations.simulator.behavior.observation import robot_state as behavior_robot_state
from voltron.integrations.simulator.behavior.tools import bridge_localization as behavior_bridge_localization
from voltron.shared.action_semantics import is_open_state_action, is_toggle_state_action, normalize_action_name
from voltron.shared.context import Subtask
from voltron.shared.errors import AdapterError


def load_behavior_module(runtime_bridge_file: str) -> Any:
    return behavior_environment_client.load_behavior_module(
        runtime_bridge_file=runtime_bridge_file,
        prepend_local_gr00t_repo=behavior_environment_client.prepend_local_gr00t_repo,
    )


def install_behavior_rgb_wrapper_fallback(*, runtime_bridge_file: str) -> None:
    behavior_environment_client.install_behavior_rgb_wrapper_fallback(
        load_behavior_module=lambda: load_behavior_module(runtime_bridge_file)
    )


def register_behavior_envs_if_needed(runtime: Any, gym: Any, *, runtime_bridge_file: str) -> None:
    runtime._registered = behavior_environment_client.register_behavior_envs_if_needed(
        registered=runtime._registered,
        gym=gym,
        env_id=runtime.env_id,
        load_behavior_register_fn=lambda: behavior_environment_client.load_behavior_register_fn(
            load_behavior_module=lambda: load_behavior_module(runtime_bridge_file)
        ),
    )


def ensure_env(runtime: Any, *, runtime_bridge_file: str) -> Any:
    runtime._env = behavior_environment_client.ensure_env(
        current_env=runtime._env,
        env_factory=runtime.env_factory,
        env_id=runtime.env_id,
        env_kwargs=runtime.env_kwargs,
        auto_register=runtime.auto_register,
        import_gymnasium=behavior_environment_client.import_gymnasium,
        register_behavior_envs_if_needed=lambda gym: register_behavior_envs_if_needed(
            runtime,
            gym,
            runtime_bridge_file=runtime_bridge_file,
        ),
        is_behavior_env=lambda env_id: str(env_id).startswith("sim_behavior"),
        install_behavior_rgb_wrapper_fallback=lambda: install_behavior_rgb_wrapper_fallback(
            runtime_bridge_file=runtime_bridge_file
        ),
    )
    env_kwargs = runtime.env_kwargs if isinstance(runtime.env_kwargs, dict) else {}
    local_offset = env_kwargs.get("recording_third_person_local_offset")
    look_at_offset = env_kwargs.get("recording_third_person_look_at_offset")
    if local_offset is not None and look_at_offset is not None:
        behavior_environment_client.configure_recording_third_person_pose(
            runtime._env,
            local_offset=local_offset,
            look_at_offset=look_at_offset,
        )
    return runtime._env


def apply_post_reset_state(runtime: Any, *, env: Any, obs: Any, info: Any) -> dict[str, Any]:
    env_kwargs = runtime.env_kwargs if isinstance(runtime.env_kwargs, dict) else {}
    position = env_kwargs.get("post_reset_robot_position")
    orientation = env_kwargs.get("post_reset_robot_orientation")
    object_states = env_kwargs.get("post_reset_object_states")
    joint_positions = env_kwargs.get("post_reset_robot_joint_positions")
    joint_velocities = env_kwargs.get("post_reset_robot_joint_velocities")
    if all(
        value is None
        for value in (
            position,
            orientation,
            object_states,
            joint_positions,
            joint_velocities,
        )
    ):
        return {"obs": obs, "info": info}

    applied = _apply_post_reset_overrides(
        env,
        position=position,
        orientation=orientation,
        object_states=object_states,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
    )
    settle_steps = _post_reset_settle_steps(runtime)
    last_obs, last_info = obs, info
    for _ in range(settle_steps):
        last_obs, last_info = _step_env_zero_action(env)

    exact_snapshot = any(
        value is not None for value in (object_states, joint_positions, joint_velocities)
    )
    if settle_steps and exact_snapshot:
        applied = _apply_post_reset_overrides(
            env,
            position=position,
            orientation=orientation,
            object_states=object_states,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
        )

    refresh_diagnostics = None
    if bool(env_kwargs.get("post_reset_refresh_observation", True)):
        last_obs, last_info, refresh_diagnostics = _refresh_post_reset_observation(
            env,
            previous_info=last_info,
        )

    return {
        "obs": last_obs,
        "info": last_info,
        "event_payload": {
            "settle_steps": settle_steps,
            "reapplied_after_settle": bool(settle_steps and exact_snapshot),
            "observation_refresh": refresh_diagnostics,
            **applied,
        },
    }


def _refresh_post_reset_observation(
    env: Any,
    *,
    previous_info: Any,
    render_count: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    wrapper, base_env, preprocess_obs = _resolve_behavior_observation_pipeline(env)
    renderer_name, kinematics_updater = _render_without_physics(
        wrapper,
        render_count=render_count,
    )
    camera_pose_order, camera_relative_poses, camera_pose_source = (
        _camera_relative_pose_diagnostics(wrapper)
    )
    get_obs = getattr(base_env, "get_obs", None)
    if not callable(get_obs):
        raise RuntimeError("post-reset observation refresh requires base_env.get_obs()")
    raw_obs, raw_info = get_obs()
    refreshed_obs = preprocess_obs(wrapper, raw_obs)
    if not isinstance(refreshed_obs, dict):
        raise RuntimeError("post-reset observation preprocessor returned a non-dict observation")

    diagnostics = _validate_refreshed_policy_observation(refreshed_obs, raw_obs)
    if camera_relative_poses is not None:
        diagnostics.update(
            {
                "camera_relative_pose_order": camera_pose_order,
                "camera_relative_poses": camera_relative_poses,
                "camera_relative_pose_source": camera_pose_source,
            }
        )
    merged_info = dict(previous_info) if isinstance(previous_info, dict) else {}
    if isinstance(raw_info, dict):
        merged_info.update(raw_info)
    wrapper.obs = refreshed_obs
    wrapper.info = merged_info
    return refreshed_obs, merged_info, {
        "method": "base_env.get_obs+behavior.preprocess_obs",
        "renderer": renderer_name,
        "articulation_kinematics_updater": kinematics_updater,
        "render_count": render_count,
        **diagnostics,
    }


def _resolve_behavior_observation_pipeline(env: Any) -> tuple[Any, Any, Any]:
    errors: list[str] = []
    for candidate in _env_candidates(env):
        base_env_getter = getattr(candidate, "_base_env", None)
        if not callable(base_env_getter):
            continue
        try:
            base_env = base_env_getter()
        except Exception as exc:
            errors.append(f"{type(candidate).__name__}._base_env: {exc}")
            continue
        candidate_preprocessor = getattr(candidate, "preprocess_obs", None)
        if callable(candidate_preprocessor):
            def preprocess_obs(
                _wrapper: Any,
                raw_obs: Any,
                preprocessor: Any = candidate_preprocessor,
            ) -> Any:
                return preprocessor(raw_obs)
        else:
            try:
                module = importlib.import_module(type(candidate).__module__)
                preprocess_obs = getattr(module, "preprocess_obs", None)
            except Exception as exc:
                errors.append(f"{type(candidate).__name__}.preprocess_obs: {exc}")
                continue
        if callable(preprocess_obs) and callable(getattr(base_env, "get_obs", None)):
            return candidate, base_env, preprocess_obs
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise RuntimeError(
        "post-reset observation refresh could not find the BEHAVIOR GR00T wrapper pipeline"
        f"{detail}"
    )


def _render_without_physics(wrapper: Any, *, render_count: int) -> tuple[str, str]:
    kinematics_updater, kinematics_updater_name = _resolve_articulation_kinematics_updater(wrapper)
    kinematics_updater()

    wrapper_renderer = getattr(wrapper, "render_without_physics", None)
    if callable(wrapper_renderer):
        for _ in range(render_count):
            wrapper_renderer()
        return f"{type(wrapper).__name__}.render_without_physics", kinematics_updater_name

    module = importlib.import_module(type(wrapper).__module__)
    sim = getattr(getattr(module, "og", None), "sim", None)
    renderer = getattr(sim, "render", None)
    if not callable(renderer):
        raise RuntimeError("post-reset observation refresh requires og.sim.render()")
    for _ in range(render_count):
        renderer()
    return "og.sim.render", kinematics_updater_name


def _resolve_articulation_kinematics_updater(wrapper: Any) -> tuple[Any, str]:
    wrapper_updater = getattr(wrapper, "update_articulations_kinematic", None)
    if callable(wrapper_updater):
        return wrapper_updater, f"{type(wrapper).__name__}.update_articulations_kinematic"

    module = importlib.import_module(type(wrapper).__module__)
    sim = getattr(getattr(module, "og", None), "sim", None)
    physics_sim_view = getattr(sim, "physics_sim_view", None)
    updater = getattr(physics_sim_view, "update_articulations_kinematic", None)
    if not callable(updater):
        raise RuntimeError(
            "post-reset observation refresh requires "
            "og.sim.physics_sim_view.update_articulations_kinematic()"
        )
    return updater, "og.sim.physics_sim_view.update_articulations_kinematic"


def _camera_relative_pose_diagnostics(
    wrapper: Any,
) -> tuple[list[str] | None, list[float] | None, str | None]:
    module = importlib.import_module(type(wrapper).__module__)
    camera_names_by_robot = getattr(module, "ROBOT_CAMERA_NAMES", None)
    transforms = getattr(module, "T", None)
    torch = getattr(module, "th", None)
    robot = _first_robot(wrapper)
    if not isinstance(camera_names_by_robot, dict) or transforms is None or torch is None or robot is None:
        return None, None, None

    camera_names = camera_names_by_robot.get("R1Pro")
    sensors = getattr(robot, "sensors", None)
    if not isinstance(camera_names, dict) or not isinstance(sensors, dict):
        return None, None, None

    base_pose_getter = getattr(robot, "get_position_orientation", None)
    if not callable(base_pose_getter):
        raise RuntimeError("post-reset camera diagnostic requires robot pose readback")
    base_pose = base_pose_getter()
    order: list[str] = []
    relative_poses: list[float] = []
    sources: list[str] = []
    for camera_id, camera_name in camera_names.items():
        name_parts = str(camera_name).split("::")
        if len(name_parts) < 2 or name_parts[1] not in sensors:
            raise RuntimeError(f"post-reset camera diagnostic could not resolve {camera_name!r}")
        camera = sensors[name_parts[1]]
        camera_parameters = getattr(camera, "camera_parameters", {})
        view_transform = np.asarray(
            camera_parameters.get("cameraViewTransform", np.zeros(16)),
            dtype=np.float64,
        ).reshape(-1)
        if view_transform.size != 16 or np.allclose(view_transform, np.zeros(16)):
            camera_pose_getter = getattr(camera, "get_position_orientation", None)
            if not callable(camera_pose_getter):
                raise RuntimeError(
                    f"post-reset camera diagnostic requires camera {camera_id} pose readback"
                )
            camera_pose = camera_pose_getter()
            source = "camera.get_position_orientation"
        else:
            world_from_camera = np.linalg.inv(view_transform.reshape(4, 4).T)
            camera_pose = transforms.mat2pose(torch.tensor(world_from_camera, dtype=torch.float32))
            source = "camera.camera_parameters.cameraViewTransform"
        relative_pose = transforms.relative_pose_transform(*camera_pose, *base_pose)
        order.append(str(camera_id))
        relative_poses.extend(_float_list(relative_pose[0]))
        relative_poses.extend(_float_list(relative_pose[1]))
        sources.append(source)
    return order, relative_poses, ",".join(sources)


def _validate_refreshed_policy_observation(
    observation: dict[str, Any], raw_observation: Any
) -> dict[str, Any]:
    required_rgb_suffixes = (
        "rgb.head_256_256",
        "rgb.left_wrist_256_256",
        "rgb.right_wrist_256_256",
    )
    rgb_shapes: dict[str, list[int]] = {}
    for suffix in required_rgb_suffixes:
        matches = [key for key in observation if str(key).endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"post-reset observation refresh expected one RGB key ending in {suffix!r}, "
                f"found {matches}"
            )
        key = matches[0]
        shape = list(np.asarray(observation[key]).shape)
        if len(shape) < 3 or shape[-1] < 3:
            raise RuntimeError(f"post-reset RGB observation {key!r} has invalid shape {shape}")
        rgb_shapes[key] = shape

    try:
        policy_observation = OpenPICometObservationAdapter.convert(observation)
    except AdapterError as exc:
        raise RuntimeError(
            "post-reset observation refresh could not produce a valid OpenPI policy observation: "
            f"{exc}"
        ) from exc
    policy_proprio = policy_observation[OpenPICometObservationAdapter.PROPRIO_KEY]
    proprio_diagnostics = OpenPICometObservationAdapter.proprio_layout_diagnostics(
        observation, policy_proprio=policy_proprio
    )
    if proprio_diagnostics["policy_proprio_size"] != OpenPICometObservationAdapter.POLICY_PROPRIO_SIZE:
        raise RuntimeError(
            "post-reset observation refresh produced an invalid normalized policy proprio size: "
            f"expected {OpenPICometObservationAdapter.POLICY_PROPRIO_SIZE}, "
            f"got {proprio_diagnostics['policy_proprio_size']}"
        )
    raw_proprio = _find_nested_value(raw_observation, "proprio")
    camera_relative_poses = _find_flattened_value(observation, "cam_rel_poses")
    return {
        "rgb_shapes": rgb_shapes,
        "camera_relative_poses": (
            _float_list(camera_relative_poses) if camera_relative_poses is not None else None
        ),
        **proprio_diagnostics,
        "raw_proprio_size": int(np.asarray(raw_proprio).size) if raw_proprio is not None else None,
    }


def _find_nested_value(value: Any, key_name: str) -> Any | None:
    if not isinstance(value, dict):
        return None
    if key_name in value:
        return value[key_name]
    for child in value.values():
        result = _find_nested_value(child, key_name)
        if result is not None:
            return result
    return None


def _find_flattened_value(value: dict[str, Any], key_suffix: str) -> Any | None:
    matches = [item for key, item in value.items() if str(key).endswith(key_suffix)]
    if len(matches) > 1:
        raise RuntimeError(
            f"post-reset observation refresh found multiple keys ending in {key_suffix!r}"
        )
    return matches[0] if matches else None


def _apply_post_reset_overrides(
    env: Any,
    *,
    position: Any,
    orientation: Any,
    object_states: Any,
    joint_positions: Any,
    joint_velocities: Any,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"robot": None, "objects": []}
    if object_states is not None:
        diagnostics["objects"] = _apply_post_reset_object_states(env, object_states)

    if any(value is not None for value in (position, orientation, joint_positions, joint_velocities)):
        robot = _first_robot(env)
        if robot is None:
            raise RuntimeError("post-reset robot state override requested but no robot was found")
        diagnostics["robot"] = _apply_post_reset_robot_state(
            robot,
            position=position,
            orientation=orientation,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
        )
    return diagnostics


def _apply_post_reset_robot_state(
    robot: Any,
    *,
    position: Any,
    orientation: Any,
    joint_positions: Any,
    joint_velocities: Any,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    requested_position = None
    requested_orientation = None
    pose_setter = None
    pose_reapplied_after_joint_state = False
    if position is not None or orientation is not None:
        requested_position, requested_orientation = position, orientation
        if position is None or orientation is None:
            current_position, current_orientation = _current_robot_pose(robot)
            requested_position = current_position if position is None else position
            requested_orientation = current_orientation if orientation is None else orientation

    requested_joint_positions = None
    if joint_positions is not None:
        requested_joint_positions = _float_list(joint_positions)
        _set_robot_joint_positions(robot, requested_joint_positions, drive=False)
        readback = _required_joint_readback(robot, "get_joint_positions", "joint positions")
        error = _max_abs_error(requested_joint_positions, readback)
        _require_readback_close("robot joint positions", error)
        _set_robot_joint_positions(robot, requested_joint_positions, drive=True)

    if requested_position is not None:
        if requested_joint_positions is None:
            pose_setter = _set_pose(robot, position=requested_position, orientation=requested_orientation)
        else:
            pose_setter = _set_root_pose_for_robot_pose(
                robot,
                position=requested_position,
                orientation=requested_orientation,
            )
            pose_reapplied_after_joint_state = True

    requested_joint_velocities = None
    if joint_velocities is not None:
        requested_joint_velocities = _float_list(joint_velocities)
        _set_robot_joint_velocities(robot, requested_joint_velocities)

    if pose_setter is not None:
        read_position, read_orientation = _required_pose_readback(robot, "robot")
        position_error = _max_abs_error(requested_position, read_position)
        orientation_error = _quaternion_max_abs_error(requested_orientation, read_orientation)
        _require_readback_close("final robot position", position_error)
        _require_readback_close("final robot orientation", orientation_error)
        diagnostics["pose"] = {
            "requested_position": _float_list(requested_position),
            "requested_orientation": _float_list(requested_orientation),
            "read_position": read_position,
            "read_orientation": read_orientation,
            "position_max_abs_error": position_error,
            "orientation_max_abs_error": orientation_error,
            "setter": pose_setter,
            "reapplied_after_joint_state": pose_reapplied_after_joint_state,
        }

    if requested_joint_positions is not None:
        readback = _required_joint_readback(robot, "get_joint_positions", "final joint positions")
        error = _max_abs_error(requested_joint_positions, readback)
        _require_readback_close("final robot joint positions", error)
        diagnostics["joint_positions"] = {
            "count": len(requested_joint_positions),
            "readback": readback,
            "max_abs_error": error,
            "hold_targets": "current_joint_positions",
        }

    if requested_joint_velocities is not None:
        readback = _required_joint_readback(robot, "get_joint_velocities", "joint velocities")
        error = _max_abs_error(requested_joint_velocities, readback)
        _require_readback_close("final robot joint velocities", error)
        diagnostics["joint_velocities"] = {
            "count": len(requested_joint_velocities),
            "readback": readback,
            "max_abs_error": error,
        }
    return diagnostics


def _apply_post_reset_object_states(env: Any, object_states: Any) -> list[dict[str, Any]]:
    if not isinstance(object_states, dict) or not object_states:
        raise ValueError("post_reset_object_states must be a non-empty object keyed by exact object name")
    objects_by_name: dict[str, Any] = {}
    for obj in _collect_scene_objects(env):
        name = str(getattr(obj, "name", "")).strip()
        if name:
            objects_by_name[name] = obj

    missing = sorted(set(object_states) - set(objects_by_name))
    if missing:
        raise RuntimeError(f"post-reset objects were not found by exact name: {missing}")

    diagnostics: list[dict[str, Any]] = []
    for object_name, requested_state in object_states.items():
        if not isinstance(requested_state, dict):
            raise ValueError(f"post-reset state for {object_name!r} must be an object")
        unknown = set(requested_state) - {"position", "orientation", "states"}
        if unknown:
            raise ValueError(f"post-reset state for {object_name!r} has unknown keys: {sorted(unknown)}")
        obj = objects_by_name[object_name]
        position = requested_state.get("position")
        orientation = requested_state.get("orientation")
        item: dict[str, Any] = {"name": object_name}
        if position is not None or orientation is not None:
            if position is None or orientation is None:
                current_position, current_orientation = _required_pose_readback(obj, object_name)
                position = current_position if position is None else position
                orientation = current_orientation if orientation is None else orientation
            setter = _set_pose(obj, position=position, orientation=orientation)
            read_position, read_orientation = _required_pose_readback(obj, object_name)
            position_error = _max_abs_error(position, read_position)
            orientation_error = _quaternion_max_abs_error(orientation, read_orientation)
            _require_readback_close(f"object {object_name} position", position_error)
            _require_readback_close(f"object {object_name} orientation", orientation_error)
            item["pose"] = {
                "requested_position": _float_list(position),
                "requested_orientation": _float_list(orientation),
                "read_position": read_position,
                "read_orientation": read_orientation,
                "position_max_abs_error": position_error,
                "orientation_max_abs_error": orientation_error,
                "setter": setter,
            }
        named_states = requested_state.get("states")
        if named_states is not None:
            item["states"] = _apply_named_object_states(obj, object_name, named_states)
        diagnostics.append(item)
    return diagnostics


def _apply_named_object_states(obj: Any, object_name: str, requested: Any) -> dict[str, Any]:
    if not isinstance(requested, dict):
        raise ValueError(f"post-reset named states for {object_name!r} must be an object")
    available = getattr(obj, "states", None)
    if not isinstance(available, dict):
        raise RuntimeError(f"object {object_name!r} does not expose named states")
    states_by_name = {_normalized_state_name(key): state for key, state in available.items()}
    diagnostics: dict[str, Any] = {}
    for state_name, value in requested.items():
        state = states_by_name.get(_normalized_state_name(state_name))
        if state is None:
            raise RuntimeError(f"object {object_name!r} has no state named {state_name!r}")
        setter = getattr(state, "set_value", None)
        getter = getattr(state, "get_value", None)
        if not callable(setter) or not callable(getter):
            raise RuntimeError(f"object state {object_name}.{state_name} is not readable and writable")
        setter(value)
        readback = getter()
        matches = bool(readback) == value if isinstance(value, bool) else readback == value
        if not matches:
            raise RuntimeError(
                f"post-reset object state readback mismatch for {object_name}.{state_name}: "
                f"requested={value!r}, read={readback!r}"
            )
        diagnostics[str(state_name)] = readback
    return diagnostics


def _normalized_state_name(value: Any) -> str:
    name = value if isinstance(value, str) else getattr(value, "__name__", str(value))
    return "".join(character for character in str(name).lower() if character.isalnum())


def _set_robot_joint_positions(robot: Any, positions: list[float], *, drive: bool) -> None:
    setter = getattr(robot, "set_joint_positions", None)
    if not callable(setter):
        raise RuntimeError("post-reset joint positions requested but robot has no set_joint_positions")
    setter_positions = _joint_values_for_setter(robot, "get_joint_positions", positions)
    try:
        setter(positions=setter_positions, drive=drive)
    except TypeError:
        setter(setter_positions, drive=drive)


def _set_robot_joint_velocities(robot: Any, velocities: list[float]) -> None:
    setter = getattr(robot, "set_joint_velocities", None)
    if not callable(setter):
        raise RuntimeError("post-reset joint velocities requested but robot has no set_joint_velocities")
    setter_velocities = _joint_values_for_setter(robot, "get_joint_velocities", velocities)
    try:
        setter(velocities=setter_velocities, drive=False)
    except TypeError:
        setter(setter_velocities, drive=False)


def _joint_values_for_setter(robot: Any, getter_name: str, requested: list[float]) -> Any:
    getter = getattr(robot, getter_name, None)
    if not callable(getter):
        raise RuntimeError(f"post-reset robot state requested but robot has no {getter_name}")
    current = getter()
    current_values = _float_list(current)
    if len(current_values) != len(requested):
        raise RuntimeError(
            "post-reset joint state length mismatch: "
            f"robot has {len(current_values)} DOF, requested {len(requested)}"
        )
    new_tensor = getattr(current, "new_tensor", None)
    if callable(new_tensor):
        return new_tensor(requested)
    if isinstance(current, np.ndarray):
        return np.asarray(requested, dtype=current.dtype)
    return requested


def _required_joint_readback(robot: Any, method_name: str, label: str) -> list[float]:
    getter = getattr(robot, method_name, None)
    if not callable(getter):
        raise RuntimeError(f"post-reset {label} requested but robot has no {method_name}")
    return _float_list(getter())


def _required_pose_readback(obj: Any, label: str) -> tuple[list[float], list[float]]:
    getter = getattr(obj, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError(f"post-reset pose requested but {label} has no get_position_orientation")
    position, orientation = getter()
    return _float_list(position), _float_list(orientation)


def _float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _max_abs_error(expected: Any, actual: Any) -> float:
    expected_arr = np.asarray(_float_list(expected), dtype=np.float64)
    actual_arr = np.asarray(_float_list(actual), dtype=np.float64)
    if expected_arr.shape != actual_arr.shape:
        raise RuntimeError(
            f"post-reset readback shape mismatch: requested={expected_arr.shape}, read={actual_arr.shape}"
        )
    return float(np.max(np.abs(expected_arr - actual_arr))) if expected_arr.size else 0.0


def _quaternion_max_abs_error(expected: Any, actual: Any) -> float:
    expected_arr = np.asarray(_float_list(expected), dtype=np.float64)
    actual_arr = np.asarray(_float_list(actual), dtype=np.float64)
    if expected_arr.shape != (4,) or actual_arr.shape != (4,):
        raise RuntimeError("post-reset orientation must be a four-element XYZW quaternion")
    return float(min(np.max(np.abs(expected_arr - actual_arr)), np.max(np.abs(expected_arr + actual_arr))))


def _require_readback_close(label: str, error: float, *, tolerance: float = 1e-4) -> None:
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(f"post-reset {label} readback error {error:.6g} exceeds tolerance {tolerance:.6g}")


def _post_reset_settle_steps(runtime: Any) -> int:
    try:
        return max(0, int(runtime.env_kwargs.get("post_reset_settle_steps", 5)))
    except Exception:
        return 5


def _first_robot(env: Any) -> Any | None:
    for candidate in _env_candidates(env):
        robots = getattr(candidate, "robots", None)
        if robots:
            return robots[0]
    return None


def _set_pose(obj: Any, *, position: Any, orientation: Any) -> str:
    for method_name in ("set_position_orientation", "set_position_and_orientation"):
        setter = getattr(obj, method_name, None)
        if not callable(setter):
            continue
        try:
            setter(position=position, orientation=orientation)
            return method_name
        except TypeError:
            setter(position, orientation)
            return method_name
    raise RuntimeError("post-reset pose override requested but object has no supported pose setter")


def _set_root_pose_for_robot_pose(robot: Any, *, position: Any, orientation: Any) -> str:
    """Move an articulation root without changing holonomic virtual base joints."""
    root_link = getattr(robot, "root_link", None)
    if root_link is None or root_link is robot:
        raise RuntimeError(
            "post-reset exact robot pose plus joint positions requires a distinct robot.root_link"
        )

    current_robot_pose = _required_pose_readback(robot, "robot before root reanchor")
    current_root_pose = _required_pose_readback(root_link, "robot root_link before reanchor")
    robot_to_target = _compose_pose(
        (_float_list(position), _float_list(orientation)),
        _invert_pose(current_robot_pose),
    )
    target_root_position, target_root_orientation = _compose_pose(
        robot_to_target,
        current_root_pose,
    )
    setter = _set_pose(
        root_link,
        position=target_root_position,
        orientation=target_root_orientation,
    )
    return f"root_link.{setter}"


def _compose_pose(
    left: tuple[Any, Any],
    right: tuple[Any, Any],
) -> tuple[list[float], list[float]]:
    left_position = np.asarray(_float_list(left[0]), dtype=np.float64)
    right_position = np.asarray(_float_list(right[0]), dtype=np.float64)
    if left_position.shape != (3,) or right_position.shape != (3,):
        raise RuntimeError("post-reset pose positions must contain exactly three elements")
    left_orientation = _normalized_quaternion(left[1])
    right_orientation = _normalized_quaternion(right[1])
    position = left_position + _rotate_vector(left_orientation, right_position)
    orientation = _quaternion_multiply(left_orientation, right_orientation)
    return position.tolist(), _normalized_quaternion(orientation).tolist()


def _invert_pose(pose: tuple[Any, Any]) -> tuple[list[float], list[float]]:
    position = np.asarray(_float_list(pose[0]), dtype=np.float64)
    if position.shape != (3,):
        raise RuntimeError("post-reset pose positions must contain exactly three elements")
    orientation = _normalized_quaternion(pose[1])
    inverse_orientation = np.asarray(
        [-orientation[0], -orientation[1], -orientation[2], orientation[3]],
        dtype=np.float64,
    )
    inverse_position = _rotate_vector(inverse_orientation, -position)
    return inverse_position.tolist(), inverse_orientation.tolist()


def _normalized_quaternion(value: Any) -> np.ndarray:
    quaternion = np.asarray(_float_list(value), dtype=np.float64)
    if quaternion.shape != (4,):
        raise RuntimeError("post-reset orientation must be a four-element XYZW quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("post-reset orientation must be a finite non-zero quaternion")
    return quaternion / norm


def _quaternion_multiply(left: Any, right: Any) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _rotate_vector(quaternion: Any, vector: Any) -> np.ndarray:
    orientation = _normalized_quaternion(quaternion)
    xyz = orientation[:3]
    scalar = orientation[3]
    value = np.asarray(vector, dtype=np.float64)
    return value + 2.0 * np.cross(xyz, np.cross(xyz, value) + scalar * value)


def _current_robot_pose(robot: Any) -> tuple[Any, Any]:
    getter = getattr(robot, "get_position_orientation", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]


def _step_env_zero_action(env: Any) -> tuple[Any, Any]:
    result = env.step(_zero_action_for_env(env))
    if isinstance(result, tuple) and len(result) >= 5:
        return result[0], result[4]
    if isinstance(result, tuple) and len(result) >= 4:
        return result[0], result[3]
    raise RuntimeError("post-reset settle step returned an unsupported env.step result")


def _zero_action_for_env(env: Any) -> Any:
    action_space = getattr(env, "action_space", None)
    return _zero_action_for_space(action_space)


def _zero_action_for_space(space: Any) -> Any:
    if space is None:
        return {}
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        return {key: _zero_action_for_space(value) for key, value in spaces.items()}
    shape = tuple(getattr(space, "shape", ()) or ())
    dtype = getattr(space, "dtype", np.float32)
    if shape:
        return np.zeros(shape, dtype=dtype)
    try:
        sample = space.sample()
    except Exception:
        return 0.0
    arr = np.asarray(sample)
    return np.zeros_like(arr)


def format_behavior_action(
    runtime: Any,
    action: dict[str, Any],
    *,
    runtime_bridge_file: str | None = None,
    hold_grippers_closed: bool = False,
) -> dict[str, Any]:
    env = runtime._env
    if env is None:
        if runtime_bridge_file is None:
            raise ValueError("runtime_bridge_file is required when environment is not initialized")
        env = ensure_env(runtime, runtime_bridge_file=runtime_bridge_file)
    action_space = getattr(env, "action_space", None)
    return behavior_bridge_localization.format_behavior_action(
        action=action,
        action_spaces=getattr(action_space, "spaces", {}),
        reference_observation=runtime._last_obs,
        hold_grippers_closed=hold_grippers_closed,
    )


def build_hovsg_localizer(runtime: Any) -> Any | None:
    runtime._hovsg_localizer = behavior_bridge_localization.build_hovsg_localizer(
        existing_localizer=runtime._hovsg_localizer,
        last_info=runtime._last_info,
        last_obs=runtime._last_obs,
        scene_id=runtime._scene_id,
        hovsg_graph_path=runtime._hovsg_graph_path,
        hovsg_graph_root=runtime._hovsg_graph_root,
        hovsg_nav_graph_type=runtime._hovsg_nav_graph_type,
    )
    return runtime._hovsg_localizer


def localize_runtime_state_snapshot(
    runtime: Any,
    last_obs: dict[str, Any],
    last_info: dict[str, Any],
    resolved_metadata: dict[str, str | None],
) -> dict[str, Any]:
    runtime._hovsg_localizer, localized_state = behavior_bridge_localization.localize_runtime_state_snapshot(
        existing_localizer=runtime._hovsg_localizer,
        last_info=last_info,
        last_obs=last_obs,
        scene_id=runtime._scene_id,
        hovsg_graph_path=runtime._hovsg_graph_path,
        hovsg_graph_root=runtime._hovsg_graph_root,
        hovsg_nav_graph_type=runtime._hovsg_nav_graph_type,
        resolved_metadata=resolved_metadata,
        frame_config=runtime.env_kwargs,
    )
    return localized_state


def apply_navigation_success_override(
    runtime: Any,
    *,
    subtask: Subtask,
    last_info: dict[str, Any],
    task_success: bool,
) -> dict[str, Any]:
    return behavior_bridge_localization.apply_navigation_success_override(
        subtask=subtask,
        last_info=last_info,
        task_success=task_success,
        nav_state=(
            runtime._navigation_runtime_state.get(subtask.runtime_id)
            or runtime._navigation_runtime_state.get(subtask.subtask_id, {})
        ),
        task_type=runtime._task_type,
        localizer=build_hovsg_localizer(runtime),
        last_obs=runtime._last_obs,
        scene_id=behavior_bridge_localization.extract_scene_id(
            last_info=runtime._last_info,
            last_obs=runtime._last_obs,
            scene_id=runtime._scene_id,
        ),
        object_goal_distance_tolerance_m=runtime.object_goal_distance_tolerance_m,
        object_goal_heading_tolerance_rad=runtime.object_goal_heading_tolerance_rad,
        frame_config=runtime.env_kwargs,
    )


def apply_action_completion_override(
    runtime: Any,
    *,
    subtask: Subtask,
    last_info: dict[str, Any],
    task_success: bool,
) -> dict[str, Any]:
    updated_last_info = dict(last_info)
    updated_task_success = bool(task_success)
    agent_name = str(getattr(getattr(subtask, "agent", None), "value", getattr(subtask, "agent", ""))).upper()
    if agent_name != "ACTION" or updated_last_info.get("subtask_completed"):
        return {"last_info": updated_last_info, "task_success": updated_task_success}

    status = _evaluate_action_subtask_completion(runtime, subtask=subtask)
    if not isinstance(status, dict) or not status.get("completed"):
        if isinstance(status, dict) and status.get("diagnostics"):
            updated_last_info["action_completion_diagnostics"] = dict(status["diagnostics"])
        return {"last_info": updated_last_info, "task_success": updated_task_success}

    updated_last_info.update(
        {
            "subtask_completed": True,
            "subtask_succeeded": True,
            "subtask_completion_reason": status.get("reason") or "action_state_reached",
            "action_completion_diagnostics": dict(status.get("diagnostics") or {}),
        }
    )
    return {"last_info": updated_last_info, "task_success": updated_task_success}


def _evaluate_action_subtask_completion(runtime: Any, *, subtask: Subtask) -> dict[str, Any] | None:
    raw_action = str(getattr(subtask, "action", "") or "").strip().lower()
    action = normalize_action_name(raw_action)
    if not is_open_state_action(action) and not is_toggle_state_action(action):
        return None

    target = _target_text(subtask)
    objects = _collect_scene_objects(getattr(runtime, "_env", None))
    diagnostics: dict[str, Any] = {
        "action": action,
        "raw_action": raw_action,
        "target": target,
        "candidate_count": len(objects),
    }
    if not objects:
        diagnostics["reason"] = "no_scene_objects"
        return {"completed": False, "diagnostics": diagnostics}

    robot_xy = _robot_xy(getattr(runtime, "_env", None), getattr(runtime, "_last_info", {}))
    ranked = _rank_target_objects(objects, target=target, robot_xy=robot_xy, action=action)
    diagnostics["ranked_candidates"] = [item[1] for item in ranked[:5]]
    if not ranked:
        diagnostics["reason"] = "no_stateful_target_candidates"
        return {"completed": False, "diagnostics": diagnostics}

    best_object, best_diag = ranked[0]
    if not best_diag.get("lexical_match") and best_diag.get("distance_m") is None:
        diagnostics["reason"] = "target_not_identified"
        return {"completed": False, "diagnostics": diagnostics}
    if not best_diag.get("lexical_match"):
        nearby = [item for item in ranked if item[1].get("distance_m") is not None and item[1]["distance_m"] <= 0.8]
        if len(nearby) != 1:
            diagnostics["reason"] = "ambiguous_nearby_target"
            return {"completed": False, "diagnostics": diagnostics}
        best_object, best_diag = nearby[0]
        best_diag = {**best_diag, "target_evidence": "unique_nearby_stateful_object"}
    else:
        best_diag = {**best_diag, "target_evidence": "lexical_match"}

    diagnostics["selected_candidate"] = dict(best_diag)
    state_key = _completion_state_key(subtask=subtask, action=action, candidate=best_diag)
    initial_states = getattr(runtime, "_action_completion_initial_states", None)
    if not isinstance(initial_states, dict):
        initial_states = {}
        setattr(runtime, "_action_completion_initial_states", initial_states)

    if is_open_state_action(action):
        open_state = _object_open_state(best_object)
        diagnostics["open_state"] = open_state
        if open_state is None:
            diagnostics["reason"] = "open_state_unavailable"
            return {"completed": False, "diagnostics": diagnostics}
        desired_open = action == "open"
        if state_key not in initial_states:
            initial_states[state_key] = bool(open_state)
            diagnostics["initial_open_state"] = bool(open_state)
            diagnostics["reason"] = "awaiting_open_state_transition"
            return {"completed": False, "diagnostics": diagnostics}
        diagnostics["initial_open_state"] = bool(initial_states[state_key])
        completed = bool(open_state) is desired_open and bool(initial_states[state_key]) is not desired_open
        if not completed:
            diagnostics["reason"] = "open_state_not_transitioned"
        return {
            "completed": completed,
            "reason": "object_opened" if desired_open else "object_closed",
            "diagnostics": diagnostics,
        }

    toggled = _object_toggle_state(best_object)
    diagnostics["toggle_state"] = toggled
    if toggled is None:
        diagnostics["reason"] = "toggle_state_unavailable"
        return {"completed": False, "diagnostics": diagnostics}
    desired_on = action in {"toggle_on", "turn_on"}
    if state_key not in initial_states:
        initial_states[state_key] = bool(toggled)
        diagnostics["initial_toggle_state"] = bool(toggled)
        diagnostics["reason"] = "awaiting_toggle_state_transition"
        return {"completed": False, "diagnostics": diagnostics}
    diagnostics["initial_toggle_state"] = bool(initial_states[state_key])
    completed = bool(toggled) is desired_on and bool(initial_states[state_key]) is not desired_on
    if not completed:
        diagnostics["reason"] = "toggle_state_not_transitioned"
    return {
        "completed": completed,
        "reason": "object_toggled_on" if desired_on else "object_toggled_off",
        "diagnostics": diagnostics,
    }


def _completion_state_key(*, subtask: Subtask, action: str, candidate: dict[str, Any]) -> tuple[str, str, str]:
    object_name = str(candidate.get("name") or candidate.get("category") or candidate.get("model") or "")
    return (str(getattr(subtask, "runtime_id", subtask.subtask_id)), action, object_name)


def _target_text(subtask: Subtask) -> str:
    target = getattr(subtask, "target", {})
    if not isinstance(target, dict):
        return ""
    for key in ("object", "object_id", "category", "name"):
        value = target.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _collect_scene_objects(env: Any) -> list[Any]:
    objects: list[Any] = []
    seen: set[int] = set()
    for candidate in _env_candidates(env):
        scene = getattr(candidate, "scene", None)
        for container in (getattr(scene, "objects", None), getattr(scene, "object_registry", None)):
            if isinstance(container, dict):
                iterable = container.values()
            elif isinstance(container, (list, tuple, set)):
                iterable = container
            else:
                continue
            for item in iterable:
                if item is None or id(item) in seen:
                    continue
                seen.add(id(item))
                objects.append(item)
    return objects


def _rank_target_objects(
    objects: list[Any],
    *,
    target: str,
    robot_xy: tuple[float, float] | None,
    action: str,
) -> list[tuple[Any, dict[str, Any]]]:
    ranked: list[tuple[tuple[int, float, str], Any, dict[str, Any]]] = []
    for obj in objects:
        descriptor = _object_descriptor(obj)
        has_state = _object_open_state(obj) is not None if action in {"open", "close"} else _object_toggle_state(obj) is not None
        if not has_state:
            continue
        lexical_match = _matches_target(descriptor, target)
        distance = _distance_to_robot(obj, robot_xy)
        if not lexical_match and distance is None:
            continue
        lexical_rank = 0 if lexical_match else 1
        distance_rank = distance if distance is not None else 999.0
        diag = {
            **descriptor,
            "lexical_match": lexical_match,
            "distance_m": distance,
        }
        ranked.append(((lexical_rank, distance_rank, descriptor.get("name") or ""), obj, diag))
    ranked.sort(key=lambda item: item[0])
    return [(obj, diag) for _, obj, diag in ranked]


def _object_descriptor(obj: Any) -> dict[str, Any]:
    name = str(getattr(obj, "name", "") or "")
    category = str(getattr(obj, "category", "") or getattr(obj, "class_name", "") or "")
    model = str(getattr(obj, "model", "") or "")
    return {"name": name, "category": category, "model": model}


def _matches_target(descriptor: dict[str, Any], target: str) -> bool:
    if not target:
        return False
    haystack = " ".join(str(value).lower().replace("_", " ") for value in descriptor.values())
    target_norm = target.lower().replace("_", " ")
    target_tokens = [token for token in target_norm.split() if token]
    if target_norm in haystack:
        return True
    return bool(target_tokens) and all(token in haystack for token in target_tokens)


def _object_open_state(obj: Any) -> bool | None:
    joint_positions = _object_joint_positions(obj)
    if joint_positions is not None:
        return any(abs(value) >= 0.1 for value in joint_positions)
    state_value = _object_named_state(obj, "Open")
    if state_value is not None:
        return bool(state_value)
    return None


def _object_toggle_state(obj: Any) -> bool | None:
    return _object_named_state(obj, "ToggledOn")


def _object_named_state(obj: Any, state_name: str) -> bool | None:
    states = getattr(obj, "states", None)
    if not isinstance(states, dict):
        return None
    for key, state in states.items():
        key_name = getattr(key, "__name__", str(key))
        if key_name != state_name and str(key_name).split(".")[-1] != state_name:
            continue
        getter = getattr(state, "get_value", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                return None
        value = getattr(state, "value", None)
        if isinstance(value, bool):
            return value
    return None


def _object_joint_positions(obj: Any) -> list[float] | None:
    getter = getattr(obj, "get_joint_positions", None)
    value = None
    if callable(getter):
        try:
            value = getter()
        except TypeError:
            try:
                value = getter(normalized=False)
            except Exception:
                value = None
        except Exception:
            value = None
    if value is None:
        value = getattr(obj, "joint_pos", None)
    try:
        values = [float(item) for item in value]
    except Exception:
        return None
    return values or None


def _robot_xy(env: Any, last_info: dict[str, Any]) -> tuple[float, float] | None:
    for candidate in _env_candidates(env):
        robots = getattr(candidate, "robots", None)
        if not robots:
            continue
        robot = robots[0]
        position = _object_position(robot)
        if position is not None:
            return position
    pose = behavior_robot_state.extract_simulator_pose(
        last_info=last_info if isinstance(last_info, dict) else {},
        last_obs={},
    )
    if isinstance(pose, dict):
        try:
            return float(pose["x"]), float(pose["y"])
        except Exception:
            pass
    return None


def _distance_to_robot(obj: Any, robot_xy: tuple[float, float] | None) -> float | None:
    if robot_xy is None:
        return None
    position = _object_position(obj)
    if position is None:
        return None
    return math.hypot(float(position[0]) - robot_xy[0], float(position[1]) - robot_xy[1])


def _object_position(obj: Any) -> tuple[float, float] | None:
    getter = getattr(obj, "get_position_orientation", None)
    if callable(getter):
        try:
            position, _ = getter()
            return float(position[0]), float(position[1])
        except Exception:
            pass
    getter = getattr(obj, "get_position", None)
    if callable(getter):
        try:
            position = getter()
            return float(position[0]), float(position[1])
        except Exception:
            pass
    for attr in ("position", "pos"):
        value = getattr(obj, attr, None)
        try:
            return float(value[0]), float(value[1])
        except Exception:
            continue
    return None


def read_environment_vlm_heartbeat(runtime: Any) -> dict[str, Any]:
    env = runtime._env
    for candidate in _env_candidates(env):
        vlm_client = getattr(candidate, "vlm_client", None)
        if vlm_client is None:
            continue
        detector_enabled = bool(getattr(candidate, "vlm", False))
        heartbeat = {
            "available": True,
            "enabled": detector_enabled,
            "source": "behavior_vlm_detector",
        }
        if hasattr(vlm_client, "is_busy"):
            heartbeat["request_in_flight"] = bool(getattr(vlm_client, "is_busy"))
        if hasattr(vlm_client, "last_result") and getattr(vlm_client, "last_result") not in (None, ""):
            heartbeat["last_result"] = getattr(vlm_client, "last_result")
        return heartbeat

    return {
        "available": False,
        "enabled": False,
        "source": "behavior_vlm_detector",
    }


def merge_environment_vlm_heartbeat(
    runtime: Any,
    last_info: dict[str, Any],
    *,
    subtask: Any | None = None,
) -> dict[str, Any]:
    del subtask
    heartbeat = read_environment_vlm_heartbeat(runtime)
    merged = dict(last_info)
    merged["environment_vlm_heartbeat"] = heartbeat
    return merged


def _env_candidates(env: Any) -> list[Any]:
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
    return candidates


__all__ = [
    "apply_action_completion_override",
    "apply_navigation_success_override",
    "apply_post_reset_state",
    "build_hovsg_localizer",
    "ensure_env",
    "format_behavior_action",
    "install_behavior_rgb_wrapper_fallback",
    "load_behavior_module",
    "localize_runtime_state_snapshot",
    "merge_environment_vlm_heartbeat",
    "read_environment_vlm_heartbeat",
    "register_behavior_envs_if_needed",
]
