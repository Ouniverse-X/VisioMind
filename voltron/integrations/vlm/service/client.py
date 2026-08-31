from __future__ import annotations

import time
from typing import Any, Callable

import requests

from voltron.shared.errors import AdapterError
from voltron.shared.models import PerceptionObject, PerceptionRelation, PerceptionReport


class VLMHttpAdapter:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8081/process",
        timeout_s: float = 60.0,
        max_retries: int = 0,
        retry_backoff_s: float = 1.0,
        custom_parser: Callable[[dict[str, Any]], PerceptionReport] | None = None,
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.custom_parser = custom_parser

    def analyze(
        self,
        images_b64: list[str],
        instruction: str,
        task_name: str,
        image_view_order: list[str] | None = None,
        image_detail: str | None = None,
    ) -> PerceptionReport:
        payload = {
            "images": images_b64,
            "instruction": instruction,
            "task_name": task_name,
        }
        if image_view_order:
            payload["image_view_order"] = list(image_view_order)
        if image_detail:
            payload["image_detail"] = str(image_detail)

        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.Timeout as exc:
                if attempt < attempts:
                    time.sleep(self.retry_backoff_s)
                    continue
                raise AdapterError(
                    f"VLM service timeout after {self.timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts}) calling {self.endpoint}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                detail = self._extract_http_error_detail(exc.response)
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                if attempt < attempts and status_code in {408, 429, 500, 502, 503, 504}:
                    time.sleep(self.retry_backoff_s)
                    continue
                raise AdapterError(
                    f"VLM service HTTP {status_code} on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                if attempt < attempts:
                    time.sleep(self.retry_backoff_s)
                    continue
                raise AdapterError(
                    f"VLM service connection error on attempt {attempt}/{attempts}: {exc}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise AdapterError(f"VLM request failed: {exc}") from exc
            except Exception as exc:
                raise AdapterError(f"VLM request failed: {exc}") from exc

        if self.custom_parser is not None:
            return self.custom_parser(data)

        return self._default_parse(data)

    def _default_parse(self, data: dict[str, Any]) -> PerceptionReport:
        objects: list[PerceptionObject] = []
        for item in data.get("objects", []):
            objects.append(
                PerceptionObject(
                    name=str(item.get("name", "unknown_object")),
                    confidence=float(item.get("confidence", 0.0)),
                    attributes=dict(item.get("attributes", {})),
                    position=item.get("position"),
                    node_id=item.get("node_id"),
                )
            )

        relations: list[PerceptionRelation] = []
        for item in data.get("relations", []):
            relations.append(
                PerceptionRelation(
                    source=str(item.get("source", "")),
                    target=str(item.get("target", "")),
                    relation=str(item.get("relation", "near")),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )

        raw_text = str(data.get("raw_text", data.get("result", "")))
        task_complete = bool(data.get("task_complete", data.get("is_success", False)))

        return PerceptionReport(
            objects=objects,
            relations=relations,
            task_complete=task_complete,
            raw_text=raw_text,
            metadata={"raw_response": data},
        )

    @staticmethod
    def _extract_http_error_detail(response: requests.Response | None) -> str:
        if response is None:
            return "empty error response"

        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text or "empty error response"

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
        return str(payload)
