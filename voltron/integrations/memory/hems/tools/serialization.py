"""Serialization helpers for the HEMS backend."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable


def serialize_retrieval(*, retrieval_result: Any, serializer: Callable[[Any], Any]) -> dict[str, Any]:
    return {
        "query_type": retrieval_result.query_type,
        "query": serializer(retrieval_result.query),
        "results": [serializer(item) for item in retrieval_result.results],
        "scores": list(retrieval_result.scores),
        "explanation": retrieval_result.explanation,
        "timestamp": str(retrieval_result.timestamp),
        "metadata": serializer(retrieval_result.metadata),
    }


def serialize_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: serialize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_obj(v) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float)):
        return value.value
    if is_dataclass(value):
        return {k: serialize_obj(v) for k, v in asdict(value).items()}
    if hasattr(value, "__dict__"):
        return {k: serialize_obj(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


__all__ = ["serialize_obj", "serialize_retrieval"]
