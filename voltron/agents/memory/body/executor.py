"""Execution helpers for the Memory agent facade."""

from __future__ import annotations

from typing import Any, Callable


def build_backend(
    *,
    backend: Any | None,
    backend_factory: Callable[..., Any],
    backend_kwargs: dict[str, Any],
) -> Any:
    if backend is not None:
        return backend
    return backend_factory(**backend_kwargs)


def call_backend_method(backend: Any, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
    return getattr(backend, method_name)(*args, **kwargs)
