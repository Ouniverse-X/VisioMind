from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


def load_maps_payload(maps_path: Path) -> dict[str, dict[str, Any]]:
    if not maps_path.exists():
        return {}
    with maps_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return {
        str(scene_id): {
            "map_payload": deepcopy(entry.get("map_payload", {})),
            "metadata": deepcopy(entry.get("metadata", {})),
        }
        for scene_id, entry in payload.items()
        if isinstance(entry, dict)
    }


def write_maps_payload(
    *,
    maps_path: Path,
    maps: dict[str, dict[str, Any]],
    serializer: Callable[[Any], Any],
) -> None:
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    with maps_path.open("w", encoding="utf-8") as handle:
        json.dump(serializer(maps), handle, ensure_ascii=False, indent=2)


def quarantine_persistence_dir(
    *,
    persistence_dir: Path,
    exc: Exception,
    source: str,
    logger: Any | None,
) -> Path:
    backup_dir = persistence_dir.with_name(f"{persistence_dir.name}.corrupt.{uuid.uuid4().hex[:8]}")
    if persistence_dir.exists():
        persistence_dir.rename(backup_dir)
    persistence_dir.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.warning(
            "Quarantined corrupted HEMS persistence. source=%s backup_dir=%s error=%s",
            source,
            backup_dir,
            exc,
        )
    return backup_dir
