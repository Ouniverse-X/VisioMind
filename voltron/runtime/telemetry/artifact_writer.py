"""File-writing helpers for runtime artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_artifact_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_artifact_file(path: Path, mode: str = "a", encoding: str = "utf-8") -> Any:
    ensure_artifact_parent(path)
    if "b" in mode:
        return path.open(mode)
    return path.open(mode, encoding=encoding)


def write_json_artifact(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> Path:
    ensure_artifact_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent), encoding="utf-8")
    return path
