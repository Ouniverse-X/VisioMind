from __future__ import annotations

from typing import Any

import numpy as np


_ROBOT_R1_ACTION_SLICES: tuple[tuple[str, int, int], ...] = (
    ("action.base", 0, 3),
    ("action.torso", 3, 7),
    ("action.left_arm", 7, 14),
    ("action.left_gripper", 14, 15),
    ("action.right_arm", 15, 22),
    ("action.right_gripper", 22, 23),
)


def format_behavior_action(
    *,
    action: dict[str, Any],
    action_spaces: dict[str, Any] | None,
    reference_observation: dict[str, Any] | None = None,
    hold_grippers_closed: bool = False,
) -> dict[str, Any]:
    if not isinstance(action_spaces, dict) or not action_spaces:
        return normalize_action_dict(action)

    source_action = expand_robot_r1_action(action, action_spaces=action_spaces)
    formatted: dict[str, Any] = {}
    for key, space in action_spaces.items():
        value = source_action.get(key)
        if value is None and key.startswith("action."):
            value = action.get(key.replace("action.", "", 1))
        if value is None and not key.startswith("action."):
            value = action.get(f"action.{key}")

        expected_shape = tuple(getattr(space, "shape", ()) or ())
        if value is None:
            held_value = held_action_value_from_observation(
                key,
                reference_observation=reference_observation,
                expected_shape=expected_shape,
                hold_grippers_closed=hold_grippers_closed,
            )
            formatted[key] = (
                held_value if held_value is not None else np.zeros(expected_shape, dtype=np.float32)
            )
            continue

        arr = to_numpy(value)
        if arr is None:
            formatted[key] = np.zeros(expected_shape, dtype=np.float32)
            continue

        arr = select_first_action_step(arr, expected_shape=expected_shape)
        arr = arr.astype(np.float32, copy=False)
        if expected_shape and arr.shape != expected_shape:
            target_size = int(np.prod(expected_shape))
            if arr.size == target_size:
                arr = arr.reshape(expected_shape)
            else:
                arr = np.resize(arr, expected_shape)
        formatted[key] = arr

    return formatted


def expand_robot_r1_action(
    action: dict[str, Any], *, action_spaces: dict[str, Any]
) -> dict[str, Any]:
    if "robot_r1" not in action or "robot_r1" in action_spaces:
        return action
    if not any(key.startswith("action.") for key in action_spaces):
        return action

    arr = to_numpy(action["robot_r1"])
    if arr is None:
        return action
    arr = (
        select_first_action_step(arr, expected_shape=()).astype(np.float32, copy=False).reshape(-1)
    )
    if arr.shape != (23,):
        return action

    expanded = dict(action)
    for key, start, end in _ROBOT_R1_ACTION_SLICES:
        expanded.setdefault(key, arr[start:end])
    return expanded


def held_action_value_from_observation(
    action_key: str,
    *,
    reference_observation: dict[str, Any] | None,
    expected_shape: tuple[int, ...],
    hold_grippers_closed: bool = False,
) -> np.ndarray | None:
    if hold_grippers_closed and action_key in {"action.left_gripper", "action.right_gripper"}:
        return np.array([-1.0], dtype=np.float32)
    if not isinstance(reference_observation, dict):
        return None
    state_key = {
        "action.torso": "state.trunk_qpos",
        "action.left_arm": "state.arm_left_qpos",
        "action.right_arm": "state.arm_right_qpos",
        "action.left_gripper": "state.gripper_left_qpos",
        "action.right_gripper": "state.gripper_right_qpos",
    }.get(action_key)
    if state_key is None:
        return None

    value = reference_observation.get(state_key)
    arr = to_numpy(value)
    if arr is None:
        return None
    arr = (
        select_first_action_step(arr, expected_shape=()).astype(np.float32, copy=False).reshape(-1)
    )
    if action_key in {"action.left_gripper", "action.right_gripper"}:
        return np.array([1.0 if float(np.mean(arr)) >= 0.01 else -1.0], dtype=np.float32)
    if expected_shape and arr.shape != expected_shape:
        target_size = int(np.prod(expected_shape))
        if arr.size != target_size:
            return None
        arr = arr.reshape(expected_shape)
    return arr.astype(np.float32, copy=False)


def normalize_action_dict(action: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in action.items():
        normalized_key = key if key == "robot_r1" or key.startswith("action.") else f"action.{key}"
        arr = to_numpy(value)
        if arr is None:
            continue
        normalized[normalized_key] = select_first_action_step(
            arr,
            expected_shape=(),
        ).astype(
            np.float32,
            copy=False,
        )
    return normalized


def extract_action(runtime_artifacts: dict[str, Any]) -> dict[str, Any] | None:
    projected = runtime_artifacts.get("projected_action")
    if isinstance(projected, dict) and projected:
        return projected
    full_action = runtime_artifacts.get("full_action")
    if isinstance(full_action, dict) and full_action:
        return full_action
    return None


def to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def select_first_action_step(arr: np.ndarray, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    out = arr
    target_ndim = len(expected_shape) if expected_shape else 1
    while out.ndim > target_ndim:
        out = out[0]
    if out.ndim == 0:
        out = out.reshape(1)
    if not expected_shape and out.ndim > 1:
        out = out.reshape(-1)
    return out


def build_policy_observation_source(
    *,
    last_obs: dict[str, Any],
    root_task_instruction: str,
    language_instruction: str | None = None,
) -> dict[str, Any]:
    source = dict(last_obs)
    instruction = (
        str(language_instruction).strip()
        if isinstance(language_instruction, str) and language_instruction.strip()
        else root_task_instruction or str(source.get("annotation.human.coarse_action", "")).strip()
    )
    if instruction:
        source["annotation.human.coarse_action"] = instruction
    return source
