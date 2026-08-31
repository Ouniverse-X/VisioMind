from __future__ import annotations

import logging
from typing import Any

import numpy as np

from visiomind.action.integrations.manipulation.openpi_comet.action_adapter import (
    OpenPICometActionAdapter,
    OpenPICometActionMode,
)
from visiomind.action.integrations.manipulation.openpi_comet.client import OpenPICometClient
from visiomind.action.integrations.manipulation.openpi_comet.observation_adapter import (
    OpenPICometObservationAdapter,
    array_summary,
)
from visiomind.action.shared.errors import AdapterError

logger = logging.getLogger(__name__)

OPENPI_COMET_REQUEST_DIAGNOSTICS_LOG_EVERY = 250


class OpenPICometPolicyAdapter:
    def __init__(
        self,
        endpoint: str = "ws://127.0.0.1:9000",
        timeout_s: float = 60.0,
        task_name: str | None = None,
        task_id: int | None = None,
        prompt: str | None = None,
        action_mode: OpenPICometActionMode = "raw",
        request_diagnostics_enabled: bool = True,
    ) -> None:
        if action_mode not in {"raw", "dict"}:
            raise ValueError(f"Unsupported OpenPI Comet action mode: {action_mode!r}")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.task_name = task_name
        self.task_id = task_id
        self.prompt = prompt
        self.action_mode: OpenPICometActionMode = action_mode
        self.request_diagnostics_enabled = bool(request_diagnostics_enabled)
        self.client = OpenPICometClient(endpoint=endpoint, timeout_s=timeout_s)
        self._call_count = 0
        self._last_logged_prompt: str | None = None

    def ping(self) -> bool:
        return self.client.ping()

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = dict(options or {})
        request = OpenPICometObservationAdapter.convert(
            observation,
            task_id=self.task_id,
            prompt=self.prompt,
            options=options,
        )
        self._call_count += 1
        self._log_request(request)
        response = self.client.request(request)
        if "action" not in response:
            raise AdapterError(
                f"OpenPI Comet response missing 'action'. Keys: {sorted(response.keys())}"
            )

        converted = OpenPICometActionAdapter.convert(response["action"], mode=self.action_mode)
        return converted, self._extract_info(response=response, converted_action=converted)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._call_count = 0
        self._last_logged_prompt = None
        result = self.client.reset()
        logger.info("openpi_comet policy reset")
        return result

    def close(self) -> None:
        self._call_count = 0
        self._last_logged_prompt = None
        self.client.close()

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "backend": "openpi_comet",
            "protocol": "websocket+msgpack",
            "task_name": self.task_name,
            "task_id": self.task_id,
            "action_mode": self.action_mode,
            "input_modalities": {
                "images": [
                    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
                    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
                    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
                ],
                "state": ["robot_r1::proprio[256]"],
                "language": ["prompt"],
            },
            "output_modalities": {
                "action_dim": OpenPICometActionAdapter.EXPECTED_DIM,
                "action_keys": [OpenPICometActionAdapter.RAW_ACTION_KEY]
                if self.action_mode == "raw"
                else [name for name, _, _ in OpenPICometActionAdapter.SLICES],
            },
            "server_metadata": self.client.server_metadata,
        }

    def _log_request(self, request: dict[str, Any]) -> None:
        prompt = str(request.get("prompt") or "")
        if len(prompt) > 160:
            prompt = f"{prompt[:157]}..."
        task_id_status = "omitted"
        if "task_id" in request:
            task_id_value = np.asarray(request["task_id"]).reshape(-1)
            task_id_status = str(int(task_id_value[0])) if task_id_value.size else "present"
        should_log_diagnostics = self.request_diagnostics_enabled and (
            self._call_count == 1
            or OPENPI_COMET_REQUEST_DIAGNOSTICS_LOG_EVERY > 0
            and self._call_count % OPENPI_COMET_REQUEST_DIAGNOSTICS_LOG_EVERY == 0
        )
        if should_log_diagnostics:
            logger.warning(
                "openpi_comet request count=%d task_name=%r prompt=%r task_id=%s action_mode=%s diagnostics=%s",
                self._call_count,
                self.task_name,
                prompt,
                task_id_status,
                self.action_mode,
                OpenPICometObservationAdapter.diagnostics(request),
            )
        else:
            logger.debug(
                "openpi_comet request count=%d task_name=%r prompt=%r task_id=%s action_mode=%s",
                self._call_count,
                self.task_name,
                prompt,
                task_id_status,
                self.action_mode,
            )
        if prompt != self._last_logged_prompt:
            logger.warning(
                "openpi_comet active prompt count=%d task_name=%r prompt=%r task_id=%s",
                self._call_count,
                self.task_name,
                prompt,
                task_id_status,
            )
            self._last_logged_prompt = prompt

    def _extract_info(
        self, *, response: dict[str, Any], converted_action: dict[str, Any]
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": "openpi_comet",
            "task_name": self.task_name,
            "action_mode": self.action_mode,
            "server_metadata": self.client.server_metadata,
            "action_summary": {
                key: array_summary(value) for key, value in converted_action.items()
            },
        }
        if "server_timing" in response:
            info["server_timing"] = response["server_timing"]
        return info


__all__ = ["OpenPICometPolicyAdapter"]
