from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def load_persistent_state(
    *,
    auto_persist: bool,
    persistence_dir: Path,
    memory: Any,
    maps_path: Path,
    maps: dict[str, dict[str, Any]],
    load_maps_payload: Callable[[Path], dict[str, dict[str, Any]]],
    quarantine_persistence_dir: Callable[..., Path],
    reset_runtime_memory: Callable[[], None],
    logger: Any,
) -> dict[str, dict[str, Any]]:
    if not auto_persist:
        return maps

    persistence_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(memory, "load"):
        try:
            memory.load(str(persistence_dir))
        except json.JSONDecodeError as exc:
            backup_dir = quarantine_persistence_dir(
                persistence_dir=persistence_dir,
                exc=exc,
                source="memory.load",
                logger=logger,
            )
            maps.clear()
            reset_runtime_memory()
            logger.warning(
                "Reset HEMS persistence after corrupted memory payload. backup_dir=%s error=%s",
                backup_dir,
                exc,
            )

    if maps_path.exists():
        try:
            loaded_maps = load_maps_payload(maps_path)
        except json.JSONDecodeError as exc:
            backup_dir = quarantine_persistence_dir(
                persistence_dir=persistence_dir,
                exc=exc,
                source="maps.json",
                logger=logger,
            )
            maps.clear()
            logger.warning(
                "Reset HEMS persistence after corrupted map payload. backup_dir=%s error=%s",
                backup_dir,
                exc,
            )
            return maps

        maps.clear()
        maps.update(loaded_maps)
    return maps


def persist_state(
    *,
    auto_persist: bool,
    persistence_dir: Path,
    memory: Any,
    maps_path: Path,
    maps: dict[str, dict[str, Any]],
    serializer: Callable[[Any], Any],
    write_maps_payload: Callable[..., None],
) -> None:
    if not auto_persist:
        return
    persistence_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(memory, "save"):
        memory.save(str(persistence_dir))
    write_maps_payload(
        maps_path=maps_path,
        maps=maps,
        serializer=serializer,
    )


__all__ = ["load_persistent_state", "persist_state"]
