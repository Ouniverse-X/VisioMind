"""Small mutable runtime context store used by control-plane helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def merge_runtime_context(
    current: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(current or {})
    if updates:
        merged.update(dict(updates))
    return merged


@dataclass
class RuntimeContextStore:
    """A small dict-backed store for runtime-scoped context updates."""

    initial: Mapping[str, Any] | None = None
    _values: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._values = dict(self.initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def merge(self, updates: Mapping[str, Any] | None) -> dict[str, Any]:
        self._values = merge_runtime_context(self._values, updates)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)
