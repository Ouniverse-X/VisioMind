from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from visiomind.action.shared.telemetry import EventRecord


def build_event_record(
    *,
    event: str,
    payload: dict[str, Any],
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    return EventRecord.create(event=event, payload=payload, now=now).to_payload()


def write_event_record(
    *,
    record_file: Any,
    event: str,
    payload: dict[str, Any],
    now: Callable[[], datetime] | None = None,
) -> None:
    if record_file is None:
        return
    record = build_event_record(event=event, payload=payload, now=now)
    try:
        record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        record_file.flush()
    except Exception:
        pass
