"""Bootstrap and runtime reset helpers for the HEMS backend."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable


def load_hems_symbols(
    *,
    import_symbols: Callable[[], dict[str, Any]],
    repo_root: Path,
    sys_path: list[str],
    adapter_error_cls: type[Exception],
) -> dict[str, Any]:
    try:
        return import_symbols()
    except (AttributeError, ModuleNotFoundError):
        local_hems_root = _find_local_hems_root(repo_root)
        if local_hems_root is not None:
            _purge_imported_hems_modules()
            sys_path.insert(0, str(local_hems_root))
            return import_symbols()

        raise adapter_error_cls(
            "Cannot import HEMS modules. Install hems package or ensure <repo>/hems exists."
        )


def import_symbols(import_module: Callable[[str], Any] = importlib.import_module) -> dict[str, Any]:
    return {
        "HEMSConfig": import_module("hems").HEMSConfig,
        "UnifiedMemorySystem": import_module("hems.memory").UnifiedMemorySystem,
        "RetrievalAPI": import_module("hems.retrieval").RetrievalAPI,
        "TaskType": import_module("hems.core.types").TaskType,
        "Outcome": import_module("hems.core.types").Outcome,
        "KGNode": import_module("hems.core.types").KGNode,
        "KGEdge": import_module("hems.core.types").KGEdge,
        "NodeType": import_module("hems.core.types").NodeType,
        "RelationType": import_module("hems.core.types").RelationType,
        "Position": import_module("hems.core.types").Position,
        "ActionRecord": import_module("hems.core.types").ActionRecord,
    }


def _find_local_hems_root(repo_root: Path) -> Path | None:
    for root in (repo_root, *repo_root.parents):
        candidate = root / "hems"
        if (candidate / "hems" / "__init__.py").exists():
            return candidate
    return None


def _purge_imported_hems_modules() -> None:
    import sys

    for name in list(sys.modules):
        if name == "hems" or name.startswith("hems."):
            del sys.modules[name]


def reset_runtime_memory(
    *,
    deps: dict[str, Any],
    auto_initialize: bool,
    owns_memory_backend: bool,
    owns_retrieval_api: bool,
    memory: Any,
    retrieval: Any,
) -> dict[str, Any]:
    if not owns_memory_backend:
        return {
            "memory": memory,
            "retrieval": retrieval,
        }

    config = deps["HEMSConfig"]()
    memory_system = deps["UnifiedMemorySystem"](config)
    if auto_initialize and not memory_system.is_initialized:
        memory_system.initialize()

    updated_retrieval = retrieval
    if owns_retrieval_api:
        updated_retrieval = deps["RetrievalAPI"](memory_system)
    elif hasattr(updated_retrieval, "memory"):
        updated_retrieval.memory = memory_system

    return {
        "memory": memory_system,
        "retrieval": updated_retrieval,
    }


__all__ = ["import_symbols", "load_hems_symbols", "reset_runtime_memory"]
