from __future__ import annotations

import functools
import logging
from typing import Any

import msgpack
import numpy as np
import requests
import websockets.exceptions
import websockets.sync.client

from visiomind.action.shared.errors import AdapterError

logger = logging.getLogger(__name__)


def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    summary: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size == 0:
        summary["empty"] = True
        return summary
    numeric = arr.astype(np.float32, copy=False) if arr.dtype.kind in "uifb" else None
    if numeric is not None:
        summary.update(
            {
                "min": round(float(np.nanmin(numeric)), 4),
                "max": round(float(np.nanmax(numeric)), 4),
                "mean": round(float(np.nanmean(numeric)), 4),
            }
        )
    return summary


def _summarize_named_arrays(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(key): _array_summary(value) for key, value in values.items()}


class ActionConverter:
    _SLICES: tuple[tuple[str, int, int], ...] = (
        ("action.base", 0, 3),
        ("action.torso", 3, 7),
        ("action.left_arm", 7, 14),
        ("action.left_gripper", 14, 15),
        ("action.right_arm", 15, 22),
        ("action.right_gripper", 22, 23),
    )
    EXPECTED_DIM: int = 23

    @staticmethod
    def convert(action: np.ndarray) -> dict[str, np.ndarray]:
        if action.ndim != 1 or action.shape[0] != ActionConverter.EXPECTED_DIM:
            raise AdapterError(
                f"Expected 1-D action of length {ActionConverter.EXPECTED_DIM}, got shape {action.shape}"
            )
        return {
            name: action[start:end].astype(np.float32, copy=True).reshape(1, 1, -1)
            for name, start, end in ActionConverter._SLICES
        }


class ObservationConverter:
    _REQUIRED_IMAGE_VIEWS: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
    _EXPECTED_PROPRIO_DIMS: frozenset[int] = frozenset({256, 258})

    _IMAGE_KEY_MAP: dict[str, str] = {
        "video.observation.images.rgb.head_256_256": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        "video.observation.images.rgb.left_wrist_256_256": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
        "video.observation.images.rgb.right_wrist_256_256": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
        "video.observation.images.rgb.head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        "video.observation.images.rgb.left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
        "video.observation.images.rgb.right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    }

    _REQUIRED_OG_IMAGE_KEYS: frozenset[str] = frozenset(
        {
            "robot_r1::robot_r1:zed_link:Camera:0::rgb",
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
        }
    )

    _PROPRIO_KEY_ORDER: tuple[str, ...] = (
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

    @staticmethod
    def needs_raw_observation_fallback(observation: dict[str, Any]) -> bool:
        missing_image = any(
            not ObservationConverter._has_image_view(observation, view)
            for view in ObservationConverter._REQUIRED_IMAGE_VIEWS
        )
        has_proprio = "robot_r1::proprio" in observation or any(
            key in observation for key in ObservationConverter._PROPRIO_KEY_ORDER
        )
        if not has_proprio:
            return True
        proprio_dim = ObservationConverter._proprio_dim(observation)
        return missing_image or (
            proprio_dim is not None
            and proprio_dim not in ObservationConverter._EXPECTED_PROPRIO_DIMS
        )

    @staticmethod
    def _has_image_view(observation: dict[str, Any], view: str) -> bool:
        return any(
            str(key).startswith("video.observation.images.rgb") and view in str(key)
            for key in observation
        )

    @staticmethod
    def _proprio_dim(observation: dict[str, Any]) -> int | None:
        if "robot_r1::proprio" in observation:
            return int(np.asarray(observation["robot_r1::proprio"]).reshape(-1).shape[0])
        size = 0
        found = False
        for key in ObservationConverter._PROPRIO_KEY_ORDER:
            if key not in observation:
                continue
            arr = np.asarray(observation[key])
            while arr.ndim > 1:
                arr = arr[0]
            size += int(arr.reshape(-1).shape[0])
            found = True
        return size if found else None

    @staticmethod
    def request_diagnostics(observation: dict[str, Any]) -> dict[str, Any]:
        image_arrays: dict[str, dict[str, Any]] = {}
        for view in ObservationConverter._REQUIRED_IMAGE_VIEWS:
            value = ObservationConverter._og_image_value(observation, view)
            if value is not None:
                image_arrays[view] = _array_summary(value)
        diagnostics = {
            "images": {
                view: ObservationConverter._has_og_image_view(observation, view)
                for view in ObservationConverter._REQUIRED_IMAGE_VIEWS
            },
            "image_arrays": image_arrays,
            "proprio": "robot_r1::proprio" in observation,
            "non_image_keys": sorted(
                str(key) for key in observation if not str(key).endswith("::rgb")
            ),
        }
        if "robot_r1::proprio" in observation:
            diagnostics["proprio_array"] = _array_summary(observation["robot_r1::proprio"])
        return diagnostics

    @staticmethod
    def _og_image_value(observation: dict[str, Any], view: str) -> Any | None:
        if view == "head":
            markers = ("zed_link",)
        elif view == "left_wrist":
            markers = ("left_realsense_link",)
        elif view == "right_wrist":
            markers = ("right_realsense_link",)
        else:
            markers = (view,)
        for key, value in observation.items():
            if str(key).endswith("::rgb") and any(marker in str(key) for marker in markers):
                return value
        return None

    @staticmethod
    def _has_og_image_view(observation: dict[str, Any], view: str) -> bool:
        if view == "head":
            markers = ("zed_link",)
        elif view == "left_wrist":
            markers = ("left_realsense_link",)
        elif view == "right_wrist":
            markers = ("right_realsense_link",)
        else:
            markers = (view,)
        return any(
            str(key).endswith("::rgb") and any(marker in str(key) for marker in markers)
            for key in observation
        )

    @staticmethod
    def convert(
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        default_prompt: str = "",
    ) -> dict[str, Any]:
        options = options or {}
        raw_observation = observation.get("raw_observation")
        if isinstance(raw_observation, dict) and raw_observation:
            return ObservationConverter._from_raw_observation(
                raw_observation=raw_observation,
                options=options,
                default_prompt=default_prompt,
                fallback_observation=observation,
            )

        out: dict[str, Any] = {}
        for groot_key, og_key in ObservationConverter._IMAGE_KEY_MAP.items():
            if groot_key not in observation:
                continue
            arr = np.asarray(observation[groot_key])
            while arr.ndim > 3:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[-1] > 3:
                arr = arr[..., :3]
            out[og_key] = arr

        missing = ObservationConverter._REQUIRED_OG_IMAGE_KEYS - set(out)
        if missing:
            raise AdapterError(
                f"Observation missing required image keys: {sorted(missing)}. "
                f"Available GR00T keys: {sorted(k for k in observation if k.startswith('video.'))}"
            )

        if "robot_r1::proprio" in observation:
            out["robot_r1::proprio"] = ObservationConverter._normalize_proprio_layout(
                observation["robot_r1::proprio"]
            )
        else:
            proprio_parts = ObservationConverter._collect_proprio_parts(observation)
            if proprio_parts:
                out["robot_r1::proprio"] = np.concatenate(proprio_parts)

        ObservationConverter._apply_prompt_and_task_id(
            out=out,
            source=observation,
            options=options,
            default_prompt=default_prompt,
        )
        return out

    @staticmethod
    def _from_raw_observation(
        *,
        raw_observation: dict[str, Any],
        options: dict[str, Any],
        default_prompt: str,
        fallback_observation: dict[str, Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        proprio_parts: list[np.ndarray] = []
        for key, value in raw_observation.items():
            if not isinstance(key, str):
                continue
            mapped_key = ObservationConverter._IMAGE_KEY_MAP.get(key)
            if mapped_key is not None:
                arr = np.asarray(value)
                while arr.ndim > 3:
                    arr = arr[0]
                if arr.ndim == 3 and arr.shape[-1] > 3:
                    arr = arr[..., :3]
                out[mapped_key] = arr
                continue
            if key.endswith("::rgb"):
                arr = np.asarray(value)
                while arr.ndim > 3:
                    arr = arr[0]
                if arr.ndim == 3 and arr.shape[-1] > 3:
                    arr = arr[..., :3]
                out[key] = arr
                continue
            if key == "robot_r1::proprio":
                out[key] = ObservationConverter._normalize_proprio_layout(value)
                continue
            if key in ObservationConverter._PROPRIO_KEY_ORDER:
                arr = np.asarray(value, dtype=np.float32)
                while arr.ndim > 1:
                    arr = arr[0]
                proprio_parts.append(arr)

        if "robot_r1::proprio" not in out and proprio_parts:
            out["robot_r1::proprio"] = np.concatenate(proprio_parts)

        missing = ObservationConverter._REQUIRED_OG_IMAGE_KEYS - set(out)
        if missing:
            raise AdapterError(
                f"Raw observation missing required image keys: {sorted(missing)}. "
                f"Available raw keys: {sorted(k for k in raw_observation if isinstance(k, str))}"
            )

        ObservationConverter._apply_prompt_and_task_id(
            out=out,
            source=fallback_observation,
            options=options,
            default_prompt=default_prompt,
        )
        return out

    @staticmethod
    def _collect_proprio_parts(observation: dict[str, Any]) -> list[np.ndarray]:
        proprio_parts: list[np.ndarray] = []
        for key in ObservationConverter._PROPRIO_KEY_ORDER:
            if key not in observation:
                continue
            arr = np.asarray(observation[key], dtype=np.float32)
            while arr.ndim > 1:
                arr = arr[0]
            proprio_parts.append(arr)
        return proprio_parts

    @staticmethod
    def _normalize_proprio_layout(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 258:
            return np.delete(arr, [193, 233]).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _apply_prompt_and_task_id(
        *,
        out: dict[str, Any],
        source: dict[str, Any],
        options: dict[str, Any],
        default_prompt: str,
    ) -> None:
        prompt = ""
        if "instruction" in options and isinstance(options["instruction"], str):
            prompt = options["instruction"].strip()
        if not prompt:
            annotation = source.get("annotation.human.coarse_action")
            if isinstance(annotation, (list, tuple)) and annotation:
                prompt = str(annotation[0]).strip()
            elif isinstance(annotation, str):
                prompt = annotation.strip()
        if not prompt:
            prompt = default_prompt
        if prompt:
            out["prompt"] = prompt

        task_id = options.get("task_id")
        if task_id is not None:
            out["task_id"] = np.array([int(task_id)])


class Pi05PolicyAdapter:
    _DEFAULT_CHUNK_SIZE: int = 20

    def __init__(
        self,
        endpoint: str = "ws://127.0.0.1:9000",
        timeout_s: float = 15.0,
        default_prompt: str = "",
        task_id: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.default_prompt = default_prompt
        self.task_id = task_id
        self.chunk_size = chunk_size or self._DEFAULT_CHUNK_SIZE

        self._ws: websockets.sync.client.ClientConnection | None = None
        self._server_metadata: dict[str, Any] = {}
        self._packer = Packer()
        self._call_count = 0
        self._action_buffer: list[dict[str, np.ndarray]] = []
        self._info_buffer: list[dict[str, Any]] = []
        self._last_logged_prompt: str | None = None

    def _http_base(self) -> str:
        url = self.endpoint.replace("wss://", "https://").replace("ws://", "http://")
        return url.rstrip("/")

    def _connect(self) -> None:
        try:
            self._ws = websockets.sync.client.connect(
                self.endpoint,
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=300,
                close_timeout=self.timeout_s,
            )
            raw = self._ws.recv(timeout=self.timeout_s)
            self._server_metadata = unpackb(raw)
            logger.info("Pi05PolicyAdapter connected to %s", self.endpoint)
        except Exception as exc:
            self._ws = None
            raise AdapterError(
                f"Pi05 WebSocket connection failed ({self.endpoint}): {exc}"
            ) from exc

    def _ensure_connected(self) -> None:
        if self._ws is None:
            self._connect()

    def _send_recv(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        assert self._ws is not None
        try:
            self._ws.send(self._packer.pack(obs))
            raw = self._ws.recv(timeout=self.timeout_s)
        except websockets.exceptions.ConnectionClosedError:
            self._ws = None
            raise AdapterError("Pi05 WebSocket connection closed")
        except Exception as exc:
            self._ws = None
            raise AdapterError(f"Pi05 communication error: {exc}") from exc
        if isinstance(raw, str):
            raise AdapterError(f"Pi05 server error: {raw}")
        return unpackb(raw)

    def ping(self) -> bool:
        try:
            resp = requests.get(f"{self._http_base()}/healthz", timeout=min(self.timeout_s, 5.0))
            return resp.ok
        except Exception as exc:
            raise AdapterError(f"Pi05 ping failed: {exc}") from exc

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._action_buffer:
            return self._action_buffer.pop(0), self._info_buffer.pop(0)

        normalized_observation = dict(observation)
        raw_observation = normalized_observation.pop("raw_observation", None)
        if isinstance(raw_observation, dict) and raw_observation:
            normalized_observation["raw_observation"] = raw_observation

        og_obs = ObservationConverter.convert(
            normalized_observation, options, default_prompt=self.default_prompt
        )
        if self.task_id is not None and "task_id" not in og_obs:
            og_obs["task_id"] = np.array([self.task_id])

        self._call_count += 1
        self._log_first_request(og_obs)
        first_resp = self._send_recv(og_obs)
        if "action" not in first_resp:
            raise AdapterError(f"Pi05 response missing 'action'. Keys: {sorted(first_resp.keys())}")

        actions_raw = [np.asarray(first_resp["action"]).flatten()]
        infos = [self._extract_info(first_resp)]

        lightweight: dict[str, Any] = {}
        if "robot_r1::proprio" in og_obs:
            lightweight["robot_r1::proprio"] = og_obs["robot_r1::proprio"]
        if "task_id" in og_obs:
            lightweight["task_id"] = og_obs["task_id"]

        for _ in range(self.chunk_size - 1):
            try:
                resp = self._send_recv(lightweight)
                if "action" not in resp:
                    break
                actions_raw.append(np.asarray(resp["action"]).flatten())
                infos.append(self._extract_info(resp))
            except AdapterError:
                logger.warning(
                    "Pi05 prefetch interrupted, using %d/%d actions",
                    len(actions_raw),
                    self.chunk_size,
                )
                break

        for raw, info in zip(actions_raw, infos):
            converted = ActionConverter.convert(raw)
            info["action_summary"] = _summarize_named_arrays(converted)
            self._action_buffer.append(converted)
            self._info_buffer.append(info)

        return self._action_buffer.pop(0), self._info_buffer.pop(0)

    def _log_first_request(self, og_obs: dict[str, Any]) -> None:
        prompt = str(og_obs.get("prompt") or "")
        if len(prompt) > 160:
            prompt = f"{prompt[:157]}..."
        task_id_status = "omitted"
        if "task_id" in og_obs:
            task_id_value = np.asarray(og_obs["task_id"]).reshape(-1)
            task_id_status = str(int(task_id_value[0])) if task_id_value.size else "present"
        diagnostics = ObservationConverter.request_diagnostics(og_obs)
        logger.warning(
            "Pi05PolicyAdapter request modalities count=%d prompt=%r task_id=%s diagnostics=%s",
            self._call_count,
            prompt,
            task_id_status,
            diagnostics,
        )
        logger.info(
            "Pi05PolicyAdapter request count=%d prompt=%r task_id=%s keys=%s",
            self._call_count,
            prompt,
            task_id_status,
            sorted(key for key in og_obs.keys() if not str(key).endswith("::rgb")),
        )
        if prompt != self._last_logged_prompt:
            logger.warning(
                "Pi05PolicyAdapter active prompt count=%d prompt=%r task_id=%s",
                self._call_count,
                prompt,
                task_id_status,
            )
            self._last_logged_prompt = prompt

    def _extract_info(self, response: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if "server_timing" in response:
            info["server_timing"] = response["server_timing"]
        info["server_metadata"] = self._server_metadata
        return info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._action_buffer.clear()
        self._info_buffer.clear()
        self._call_count = 0
        self._last_logged_prompt = None

        self._ensure_connected()
        assert self._ws is not None
        try:
            self._ws.send(self._packer.pack({"reset": True}))
        except websockets.exceptions.ConnectionClosedError:
            self._ws = None
            raise AdapterError("Pi05 WebSocket connection closed during reset")
        except Exception as exc:
            self._ws = None
            raise AdapterError(f"Pi05 reset failed: {exc}") from exc
        logger.info("Pi05 policy reset (server + local buffer)")
        return {"status": "reset", "endpoint": self.endpoint}

    def close(self) -> None:
        self._action_buffer.clear()
        self._info_buffer.clear()
        self._call_count = 0
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception as exc:
            logger.warning("Pi05 websocket close failed: %s", exc)

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "backend": "pi05",
            "protocol": "websocket+msgpack",
            "chunk_size": self.chunk_size,
            "input_modalities": {
                "images": ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
                "state": ["proprio"],
                "language": ["prompt"],
            },
            "output_modalities": {
                "action_dim": ActionConverter.EXPECTED_DIM,
                "action_keys": [name for name, _, _ in ActionConverter._SLICES],
            },
            "server_metadata": self._server_metadata,
        }
