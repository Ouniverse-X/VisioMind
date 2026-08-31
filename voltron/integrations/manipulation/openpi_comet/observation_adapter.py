from __future__ import annotations

from typing import Any

import numpy as np

from voltron.shared.errors import AdapterError

_HEAD_RGB_KEY = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
_LEFT_WRIST_RGB_KEY = "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
_RIGHT_WRIST_RGB_KEY = "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
_PROPRIO_KEY = "robot_r1::proprio"
_POLICY_PROPRIO_SIZE = 256
_WRAPPER_PROPRIO_SIZE = 258
_DROPPED_PROPRIO_INDICES = (193, 233)
_DROPPED_STATE_KEYS = ("state.grasp_left", "state.grasp_right")

_REQUIRED_RGB_KEYS = frozenset({_HEAD_RGB_KEY, _LEFT_WRIST_RGB_KEY, _RIGHT_WRIST_RGB_KEY})
_PROPRIO_STATE_KEYS = (
    "state.joint_qpos",
    "state.joint_qpos_sin",
    "state.joint_qpos_cos",
    "state.joint_qvel",
    "state.joint_qeffort",
    "state.robot_pos",
    "state.robot_ori_cos",
    "state.robot_ori_sin",
    "state.robot_2d_ori",
    "state.robot_2d_ori_cos",
    "state.robot_2d_ori_sin",
    "state.robot_lin_vel",
    "state.robot_ang_vel",
    "state.arm_left_qpos",
    "state.arm_left_qpos_sin",
    "state.arm_left_qpos_cos",
    "state.arm_left_qvel",
    "state.eef_left_pos",
    "state.eef_left_quat",
    "state.gripper_left_qpos",
    "state.gripper_left_qvel",
    "state.arm_right_qpos",
    "state.arm_right_qpos_sin",
    "state.arm_right_qpos_cos",
    "state.arm_right_qvel",
    "state.eef_right_pos",
    "state.eef_right_quat",
    "state.gripper_right_qpos",
    "state.gripper_right_qvel",
    "state.trunk_qpos",
    "state.trunk_qvel",
    "state.base_qpos",
    "state.base_qpos_sin",
    "state.base_qpos_cos",
    "state.base_qvel",
)
_VOLTRON_RGB_ALIASES = {
    "video.observation.images.rgb.head": _HEAD_RGB_KEY,
    "video.observation.images.rgb.head_256_256": _HEAD_RGB_KEY,
    "video.observation.images.rgb.left_wrist": _LEFT_WRIST_RGB_KEY,
    "video.observation.images.rgb.left_wrist_256_256": _LEFT_WRIST_RGB_KEY,
    "video.observation.images.rgb.right_wrist": _RIGHT_WRIST_RGB_KEY,
    "video.observation.images.rgb.right_wrist_256_256": _RIGHT_WRIST_RGB_KEY,
}


class OpenPICometObservationAdapter:
    REQUIRED_RGB_KEYS = _REQUIRED_RGB_KEYS
    PROPRIO_KEY = _PROPRIO_KEY
    POLICY_PROPRIO_SIZE = _POLICY_PROPRIO_SIZE

    @staticmethod
    def convert(
        observation: dict[str, Any],
        *,
        task_id: int | None = None,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = OpenPICometObservationAdapter._source_observation(observation)
        payload: dict[str, Any] = {}

        for key, value in source.items():
            if not isinstance(key, str):
                continue
            if key in _REQUIRED_RGB_KEYS:
                payload[key] = OpenPICometObservationAdapter._rgb(value)
            elif key in _VOLTRON_RGB_ALIASES:
                payload[_VOLTRON_RGB_ALIASES[key]] = OpenPICometObservationAdapter._rgb(value)
            elif key == _PROPRIO_KEY:
                payload[key] = OpenPICometObservationAdapter._proprio(value)

        if _PROPRIO_KEY not in payload:
            proprio_parts = OpenPICometObservationAdapter._collect_state_proprio(source)
            if proprio_parts:
                payload[_PROPRIO_KEY] = OpenPICometObservationAdapter._proprio(
                    np.concatenate(proprio_parts)
                )

        missing = sorted(_REQUIRED_RGB_KEYS - set(payload))
        if missing:
            raise AdapterError(
                f"OpenPI Comet observation missing required RGB keys: {missing}. "
                f"Available keys: {sorted(str(key) for key in source)}"
            )
        if _PROPRIO_KEY not in payload:
            raise AdapterError("OpenPI Comet observation missing robot_r1::proprio")
        if payload[_PROPRIO_KEY].shape != (_POLICY_PROPRIO_SIZE,):
            raise AdapterError(
                "OpenPI Comet observation expected robot_r1::proprio shape "
                f"({_POLICY_PROPRIO_SIZE},), got {payload[_PROPRIO_KEY].shape}"
            )

        resolved_prompt = OpenPICometObservationAdapter._resolve_prompt(
            observation=observation,
            source=source,
            options=options or {},
            explicit_prompt=prompt,
        )
        if resolved_prompt:
            payload["prompt"] = resolved_prompt
        resolved_task_id = (options or {}).get("task_id", task_id)
        if resolved_task_id is not None:
            payload["task_id"] = np.array([int(resolved_task_id)], dtype=np.int64)
        return payload

    @staticmethod
    def proprio_layout_diagnostics(
        observation: dict[str, Any], *, policy_proprio: Any
    ) -> dict[str, Any]:
        source = OpenPICometObservationAdapter._source_observation(observation)
        direct_proprio = source.get(_PROPRIO_KEY)
        if direct_proprio is not None:
            wrapper_proprio_size = int(np.asarray(direct_proprio).size)
            proprio_layout = "direct_proprio"
            dropped_state_keys: list[str] = []
        else:
            wrapper_state_keys = tuple(
                key for key in (*_PROPRIO_STATE_KEYS, *_DROPPED_STATE_KEYS) if key in source
            )
            wrapper_proprio_size = sum(
                int(np.asarray(source[key]).size) for key in wrapper_state_keys
            )
            proprio_layout = "behavior_state_fields"
            dropped_state_keys = [key for key in _DROPPED_STATE_KEYS if key in source]

        return {
            "wrapper_proprio_size": wrapper_proprio_size,
            "policy_proprio_size": int(np.asarray(policy_proprio).size),
            "proprio_layout": proprio_layout,
            "dropped_proprio_indices": (
                list(_DROPPED_PROPRIO_INDICES)
                if wrapper_proprio_size == _WRAPPER_PROPRIO_SIZE
                else []
            ),
            "dropped_state_keys": dropped_state_keys,
        }

    @staticmethod
    def diagnostics(observation: dict[str, Any]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "rgb_keys": {key: key in observation for key in sorted(_REQUIRED_RGB_KEYS)},
            "proprio": _PROPRIO_KEY in observation,
            "non_rgb_keys": sorted(
                str(key) for key in observation if not str(key).endswith("::rgb")
            ),
        }
        if _PROPRIO_KEY in observation:
            diagnostics["proprio_array"] = array_summary(observation[_PROPRIO_KEY])
        diagnostics["rgb_arrays"] = {
            key: array_summary(observation[key])
            for key in sorted(_REQUIRED_RGB_KEYS)
            if key in observation
        }
        return diagnostics

    @staticmethod
    def _source_observation(observation: dict[str, Any]) -> dict[str, Any]:
        source: dict[str, Any] = {}
        source.update(observation)
        raw = observation.get("raw_observation")
        if isinstance(raw, dict):
            source.update(OpenPICometObservationAdapter._flatten_raw_observation(raw))
        return source

    @staticmethod
    def _flatten_raw_observation(raw: dict[str, Any]) -> dict[str, Any]:
        robot_obs = raw.get("robot_r1")
        if not isinstance(robot_obs, dict):
            return dict(raw)
        flattened: dict[str, Any] = dict(raw)
        proprio = robot_obs.get("proprio")
        if proprio is not None:
            flattened[_PROPRIO_KEY] = proprio
        for camera_name, camera_obs in robot_obs.items():
            if not isinstance(camera_name, str) or not isinstance(camera_obs, dict):
                continue
            rgb = camera_obs.get("rgb")
            if rgb is not None:
                flattened[f"robot_r1::{camera_name}::rgb"] = rgb
        return flattened

    @staticmethod
    def _rgb(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim != 3:
            raise AdapterError(f"OpenPI Comet RGB image expected 3-D array, got shape {arr.shape}")
        if arr.shape[-1] < 3:
            raise AdapterError(
                f"OpenPI Comet RGB image expected at least 3 channels, got shape {arr.shape}"
            )
        return arr[..., :3].astype(np.uint8, copy=False)

    @staticmethod
    def _proprio(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape == (_WRAPPER_PROPRIO_SIZE,):
            arr = np.delete(arr, _DROPPED_PROPRIO_INDICES)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _collect_state_proprio(source: dict[str, Any]) -> list[np.ndarray]:
        parts: list[np.ndarray] = []
        for key in _PROPRIO_STATE_KEYS:
            if key not in source:
                return []
            arr = np.asarray(source[key], dtype=np.float32)
            while arr.ndim > 1:
                arr = arr[0]
            parts.append(arr.reshape(-1))
        return parts

    @staticmethod
    def _resolve_prompt(
        *,
        observation: dict[str, Any],
        source: dict[str, Any],
        options: dict[str, Any],
        explicit_prompt: str | None,
    ) -> str:
        for value in (options.get("instruction"), observation.get("instruction")):
            if isinstance(value, str) and value.strip():
                return value.strip()
        for container in (observation, source):
            annotation = container.get("annotation.human.coarse_action")
            if isinstance(annotation, str) and annotation.strip():
                return annotation.strip()
            if isinstance(annotation, (list, tuple)) and annotation:
                return str(annotation[0]).strip()
        if isinstance(explicit_prompt, str) and explicit_prompt.strip():
            return explicit_prompt.strip()
        return ""


def array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    summary: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size == 0:
        summary["empty"] = True
        return summary
    if arr.dtype.kind in "uifb":
        numeric = arr.astype(np.float32, copy=False)
        summary.update(
            {
                "min": round(float(np.nanmin(numeric)), 4),
                "max": round(float(np.nanmax(numeric)), 4),
                "mean": round(float(np.nanmean(numeric)), 4),
            }
        )
    return summary


__all__ = ["OpenPICometObservationAdapter", "array_summary"]
