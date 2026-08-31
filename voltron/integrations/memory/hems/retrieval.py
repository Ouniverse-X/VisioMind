from __future__ import annotations

from typing import Any, Callable

from .tools import query_runtime


def find_object(
    *,
    retrieval_api: Any,
    name: str,
    attributes: dict[str, Any] | None = None,
    top_k: int = 5,
    serialize_retrieval: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return serialize_retrieval(retrieval_api.find_object(name, attributes=attributes, top_k=top_k))


def find_objects_near(
    *,
    retrieval_api: Any,
    position: tuple[float, float, float],
    radius: float = 2.0,
    serialize_retrieval: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return serialize_retrieval(retrieval_api.find_objects_near(position=position, radius=radius))


def find_similar_episodes(
    *,
    retrieval_api: Any,
    description: str,
    top_k: int = 5,
    serialize_retrieval: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return serialize_retrieval(retrieval_api.find_similar_episodes(description, top_k=top_k))


def find_applicable_skills(
    *,
    retrieval_api: Any,
    current_state: dict[str, Any],
    top_k: int = 5,
    serialize_retrieval: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return serialize_retrieval(retrieval_api.find_applicable_skills(current_state, top_k=top_k))


def predict_action_effects(
    *,
    retrieval_api: Any,
    action: str,
    target: str,
    conditions: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    match_mode: str = "strict",
    serialize_retrieval: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return serialize_retrieval(
        retrieval_api.predict_action_effects(
            action=action,
            target=target,
            conditions=conditions,
            parameters=parameters,
            match_mode=match_mode,
        )
    )


def diagnose_effect_cause(
    *,
    retrieval_api: Any,
    effect: str,
    value: Any = None,
    serialize_retrieval: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return serialize_retrieval(retrieval_api.diagnose_effect_cause(effect=effect, value=value))


def query_semantic_region(
    *,
    semantic_memory: Any,
    name: str,
    attributes: dict[str, Any] | None = None,
    top_k: int = 5,
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    return query_runtime.query_semantic_region(
        semantic_memory=semantic_memory,
        name=name,
        attributes=attributes,
        top_k=top_k,
        serializer=serializer,
    )
