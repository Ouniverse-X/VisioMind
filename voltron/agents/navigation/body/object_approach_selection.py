from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from http import HTTPStatus

import requests

from voltron.agents.navigation.body.json_response import extract_json_object

_SYSTEM_PROMPT = """You are the Navigation agent's object-approach anchor selector.
Return valid JSON only. Do not include markdown fences or extra text.

Pick exactly one candidate from the provided list. Do not invent new coordinates.

Return a top-level JSON object with this schema:
{
  "candidate_id": "one candidate_id from the list",
  "reason": "short explanation"
}

Selection rules:
- Prefer anchors that improve the chance of successful downstream manipulation.
- Avoid anchors with repeated recent failures when a reasonable alternative exists.
- Treat manipulation readiness and collision clearance as primary, and path length as secondary.
- `candidate_geometry_score` is an error from the preferred interaction distance: lower is better.
- Do not maximize `candidate_geometry_score`.
- Nearby-object distances represent clearance: higher is better.
- `path_cost` is travel effort: lower is better, but only after the anchor is suitable for the follow-up interaction.
- `handoff_distance_m` must stay close to the graph handoff point so the local controller can actually reach the final stance.
"""
_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}
_OPENPI_COMET_REFERENCE_HANDOFFS = {
    "turning_on_radio": {
        "object_names": {"radio", "radio receiver"},
        "candidate_id": "openpi_comet_turning_on_radio_start_pose",
        "x": 4.783253,
        "y": 4.789430,
        "z": 0.004995,
        "desired_heading": 0.575359,
    }
}


class HeuristicNavigationApproachPointSelector:
    min_candidate_clearance_m = 0.5
    min_path_clearance_m = 0.5
    max_handoff_distance_m = 1.0

    def select_candidate(
        self,
        *,
        subtask,
        context,
        goal: dict,
        prepared_payload: dict,
    ) -> dict:
        del subtask
        candidates = [dict(item) for item in prepared_payload.get("candidates") or []]
        if not candidates:
            return {
                "candidate": None,
                "reason": "no object-approach candidates available",
                "source": "heuristic_navigation_approach_point_selector",
            }

        usable_candidates, rejected_candidates = self._partition_candidates(candidates)
        if not usable_candidates:
            return {
                "candidate": None,
                "reason": "all object-approach candidates violated hard constraints",
                "source": "heuristic_navigation_approach_point_selector",
                "ranked_candidate_ids": [
                    item.get("candidate_id") for item in sorted(candidates, key=self._sort_key)
                ],
                "rejected_candidate_ids": [
                    item.get("candidate_id") for item in rejected_candidates
                ],
                "rejection_reasons": {
                    str(item.get("candidate_id")): self._constraint_violations(item)
                    for item in rejected_candidates
                },
            }

        ranked = sorted(usable_candidates, key=self._sort_key)
        forced_candidate_id = str(
            os.getenv("VOLTRON_FORCE_OBJECT_APPROACH_CANDIDATE_ID") or ""
        ).strip()
        if forced_candidate_id:
            forced_candidate = next(
                (
                    item
                    for item in ranked
                    if str(item.get("candidate_id") or "") == forced_candidate_id
                ),
                None,
            )
            if forced_candidate is not None:
                selected = deepcopy(forced_candidate)
                result = {
                    "candidate": selected,
                    "reason": f"forced object-approach candidate {forced_candidate_id}",
                    "source": "heuristic_navigation_approach_point_selector_forced",
                    "ranked_candidate_ids": [item.get("candidate_id") for item in ranked],
                }
                if rejected_candidates:
                    result["rejected_candidate_ids"] = [
                        item.get("candidate_id") for item in rejected_candidates
                    ]
                    result["rejection_reasons"] = {
                        str(item.get("candidate_id")): self._constraint_violations(item)
                        for item in rejected_candidates
                    }
                return result

        reference_candidate = self._openpi_comet_reference_candidate(
            context=context,
            goal=goal,
            candidates=ranked,
        )
        if reference_candidate is not None:
            result = {
                "candidate": reference_candidate,
                "reason": "selected OpenPI Comet reference handoff pose for policy start distribution",
                "source": "openpi_comet_reference_handoff_selector",
                "ranked_candidate_ids": [item.get("candidate_id") for item in ranked],
            }
            if rejected_candidates:
                result["rejected_candidate_ids"] = [
                    item.get("candidate_id") for item in rejected_candidates
                ]
                result["rejection_reasons"] = {
                    str(item.get("candidate_id")): self._constraint_violations(item)
                    for item in rejected_candidates
                }
            return result

        selected = deepcopy(ranked[0])
        result = {
            "candidate": selected,
            "reason": "selected best constraint-satisfying candidate after geometry, clearance, and history penalties",
            "source": "heuristic_navigation_approach_point_selector",
            "ranked_candidate_ids": [item.get("candidate_id") for item in ranked],
        }
        if rejected_candidates:
            result["rejected_candidate_ids"] = [
                item.get("candidate_id") for item in rejected_candidates
            ]
            result["rejection_reasons"] = {
                str(item.get("candidate_id")): self._constraint_violations(item)
                for item in rejected_candidates
            }
        return result

    @classmethod
    def _openpi_comet_reference_candidate(
        cls,
        *,
        context,
        goal: dict,
        candidates: list[dict],
    ) -> dict | None:
        if not cls._uses_openpi_comet(context) or not cls._openpi_comet_reference_handoff_enabled(
            context
        ):
            return None
        if not candidates:
            return None
        reference = cls._openpi_comet_reference(
            goal
        ) or cls._openpi_comet_reference_from_candidates(candidates)
        if reference is None:
            return None
        nearest = min(candidates, key=lambda item: cls._squared_xy_distance(item, reference))
        candidate = deepcopy(nearest)
        candidate.update(
            {
                "candidate_id": reference["candidate_id"],
                "x": reference["x"],
                "y": reference["y"],
                "z": reference["z"],
                "desired_heading": reference["desired_heading"],
                "waypoint_type": "object_approach",
                "selection_source": "openpi_comet_reference_handoff",
                "reference_handoff_backend": "openpi_comet",
            }
        )
        object_position = candidate.get("object_position")
        if isinstance(object_position, dict):
            dx = float(candidate["x"]) - float(object_position.get("x", candidate["x"]))
            dy = float(candidate["y"]) - float(object_position.get("y", candidate["y"]))
            candidate["approach_distance_m"] = (dx * dx + dy * dy) ** 0.5
        candidate["candidate_geometry_score"] = 0.0
        candidate["history_penalty"] = 0.0
        candidate["blocked_by_history"] = False
        return candidate

    @staticmethod
    def _uses_openpi_comet(context) -> bool:
        metadata = getattr(getattr(context, "task_request", None), "metadata", {})
        if not isinstance(metadata, dict):
            return False
        return str(metadata.get("policy_backend") or "").strip().lower() == "openpi_comet"

    @staticmethod
    def _openpi_comet_reference_handoff_enabled(context) -> bool:
        metadata = getattr(getattr(context, "task_request", None), "metadata", {})
        if isinstance(metadata, dict) and metadata.get("openpi_comet_reference_handoff") is True:
            return True
        return str(
            os.getenv("VOLTRON_ENABLE_OPENPI_COMET_REFERENCE_HANDOFF") or ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @classmethod
    def _openpi_comet_reference_from_candidates(cls, candidates: list[dict]) -> dict | None:
        for candidate in candidates:
            reference = cls._openpi_comet_reference(candidate)
            if reference is not None:
                return reference
        return None

    @staticmethod
    def _openpi_comet_reference(goal: dict) -> dict | None:
        object_name = (
            str(
                goal.get("object_name")
                or goal.get("object")
                or goal.get("target_object")
                or goal.get("target")
                or ""
            )
            .strip()
            .lower()
            .replace("_", " ")
        )
        object_id = str(goal.get("object_id") or "").strip().lower().replace("_", " ")
        for task_name, reference in _OPENPI_COMET_REFERENCE_HANDOFFS.items():
            if object_name in reference["object_names"] or any(
                name in object_id for name in reference["object_names"]
            ):
                return dict(reference, task_name=task_name)
        return None

    @staticmethod
    def _squared_xy_distance(candidate: dict, reference: dict) -> float:
        dx = float(candidate.get("x", 0.0)) - float(reference["x"])
        dy = float(candidate.get("y", 0.0)) - float(reference["y"])
        return dx * dx + dy * dy

    @classmethod
    def _partition_candidates(cls, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        usable: list[dict] = []
        rejected: list[dict] = []
        for candidate in candidates:
            if cls._constraint_violations(candidate):
                rejected.append(candidate)
            else:
                usable.append(candidate)
        return usable, rejected

    @classmethod
    def _constraint_violations(cls, candidate: dict) -> list[str]:
        violations: list[str] = []
        if candidate.get("path_found") is False or candidate.get("found") is False:
            violations.append("unreachable_path")
        evidence = cls._clearance_evidence(candidate)
        if evidence:
            nearest = cls._effective_candidate_clearance_m(evidence)
            if nearest is not None and nearest < cls.min_candidate_clearance_m:
                violations.append("insufficient_candidate_clearance")

        handoff_distance = cls._float_or_none(candidate.get("handoff_distance_m"))
        if handoff_distance is not None and handoff_distance > cls.max_handoff_distance_m:
            violations.append("unstable_navigation_handoff")

        return violations

    @staticmethod
    def _clearance_evidence(candidate: dict) -> dict:
        evidence = candidate.get("nearby_object_evidence")
        merged = dict(evidence) if isinstance(evidence, dict) else {}
        for key in (
            "nearest_object_id",
            "nearest_object_name",
            "nearest_object_distance_m",
            "path_nearest_object_id",
            "path_nearest_object_name",
            "path_nearest_object_distance_m",
            "nearby_objects",
        ):
            if key not in merged and key in candidate:
                merged[key] = candidate[key]
        return merged

    @staticmethod
    def _is_portal_like_object_name(value: object) -> bool:
        normalized = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
        if not normalized:
            return False
        tokens = set(normalized.split())
        return bool(tokens & {"door", "doorway", "gate", "gateway", "opening", "portal"})

    @classmethod
    def _effective_candidate_clearance_m(cls, evidence: dict) -> float | None:
        nearest = cls._float_or_none(evidence.get("nearest_object_distance_m"))
        explicit_candidate_distance = cls._nearest_candidate_distance_m(evidence)
        if explicit_candidate_distance is not None:
            return explicit_candidate_distance
        return nearest

    @classmethod
    def _nearest_candidate_distance_m(cls, evidence: dict) -> float | None:
        nearby_objects = evidence.get("nearby_objects")
        if not isinstance(nearby_objects, list):
            return None
        distances: list[float] = []
        for item in nearby_objects:
            if not isinstance(item, dict):
                continue
            distance = cls._float_or_none(item.get("distance_to_candidate_m"))
            if distance is not None:
                distances.append(distance)
        return min(distances) if distances else None

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _sort_key(candidate: dict) -> tuple[float, float, float, float, float, str]:
        blocked_penalty = 1000.0 if candidate.get("blocked_by_history") else 0.0
        obstacle_penalty = HeuristicNavigationApproachPointSelector._obstacle_penalty(candidate)
        target_part_score = HeuristicNavigationApproachPointSelector._target_part_score(candidate)
        return (
            target_part_score,
            blocked_penalty
            + obstacle_penalty
            + float(candidate.get("history_penalty", 0.0))
            + 6.0 * float(candidate.get("candidate_geometry_score", 0.0)),
            float(candidate.get("approach_distance_m", 0.0)),
            float(candidate.get("path_cost", 0.0)),
            float(candidate.get("candidate_geometry_score", 0.0)),
            str(candidate.get("candidate_id")),
        )

    @staticmethod
    def _target_part_score(candidate: dict) -> float:
        value = candidate.get("target_part_score")
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    @staticmethod
    def _has_target_part_score(candidate: dict) -> bool:
        return isinstance(candidate.get("target_part_score"), (int, float))

    @staticmethod
    def _obstacle_penalty(candidate: dict) -> float:
        evidence = HeuristicNavigationApproachPointSelector._clearance_evidence(candidate)
        if not evidence:
            return 0.0
        nearest = HeuristicNavigationApproachPointSelector._effective_candidate_clearance_m(
            evidence
        )
        distances = [value for value in (nearest,) if isinstance(value, (int, float))]
        path_nearest = evidence.get("path_nearest_object_distance_m")
        if isinstance(
            path_nearest, (int, float)
        ) and not HeuristicNavigationApproachPointSelector._is_portal_like_object_name(
            evidence.get("path_nearest_object_name")
        ):
            distances.append(path_nearest)
        if not distances:
            return 0.0
        nearest = min(float(value) for value in distances)
        if nearest >= 1.0:
            return 0.0
        return 8.0 * (1.0 - nearest)


@dataclass(frozen=True)
class OpenAINavigationApproachPointSelectorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 10.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleNavigationApproachPointSelector:
    def __init__(self, config: OpenAINavigationApproachPointSelectorConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.fallback_selector = HeuristicNavigationApproachPointSelector()

    def select_candidate(
        self,
        *,
        subtask,
        context,
        goal: dict,
        prepared_payload: dict,
    ) -> dict:
        candidates = list(prepared_payload.get("candidates") or [])
        if not candidates:
            return self.fallback_selector.select_candidate(
                subtask=subtask,
                context=context,
                goal=goal,
                prepared_payload=prepared_payload,
            )
        usable_candidates, rejected_candidates = self.fallback_selector._partition_candidates(
            [dict(candidate) for candidate in candidates]
        )
        if not usable_candidates:
            return self.fallback_selector.select_candidate(
                subtask=subtask,
                context=context,
                goal=goal,
                prepared_payload=prepared_payload,
            )
        forced_candidate_id = str(
            os.getenv("VOLTRON_FORCE_OBJECT_APPROACH_CANDIDATE_ID") or ""
        ).strip()
        if forced_candidate_id:
            forced_candidate = next(
                (
                    item
                    for item in usable_candidates
                    if str(item.get("candidate_id") or "") == forced_candidate_id
                ),
                None,
            )
            if forced_candidate is not None:
                result = {
                    "candidate": dict(forced_candidate),
                    "reason": f"forced object-approach candidate {forced_candidate_id}",
                    "source": "openai_compatible_navigation_approach_point_selector_forced",
                }
                if rejected_candidates:
                    result["rejected_candidate_ids"] = [
                        item.get("candidate_id") for item in rejected_candidates
                    ]
                    result["rejection_reasons"] = {
                        str(
                            item.get("candidate_id")
                        ): self.fallback_selector._constraint_violations(item)
                        for item in rejected_candidates
                    }
                return result

        reference_candidate = self.fallback_selector._openpi_comet_reference_candidate(
            context=context,
            goal=goal,
            candidates=usable_candidates,
        )
        if reference_candidate is not None:
            result = {
                "candidate": reference_candidate,
                "reason": "selected OpenPI Comet reference handoff pose for policy start distribution",
                "source": "openpi_comet_reference_handoff_selector",
            }
            if rejected_candidates:
                result["rejected_candidate_ids"] = [
                    item.get("candidate_id") for item in rejected_candidates
                ]
                result["rejection_reasons"] = {
                    str(item.get("candidate_id")): self.fallback_selector._constraint_violations(
                        item
                    )
                    for item in rejected_candidates
                }
            return result

        if any(
            self.fallback_selector._has_target_part_score(candidate)
            for candidate in usable_candidates
        ):
            selection = self.fallback_selector.select_candidate(
                subtask=subtask,
                context=context,
                goal=goal,
                prepared_payload=prepared_payload,
            )
            selection["reason"] = (
                "selected deterministic target-part-aligned candidate before LLM point selection; "
                f"{selection.get('reason', '')}"
            ).strip()
            selection["source"] = "target_part_navigation_approach_point_selector"
            return selection

        filtered_payload = dict(prepared_payload)
        filtered_payload["candidates"] = usable_candidates
        try:
            prompt = self._build_prompt(
                subtask=subtask, goal=goal, prepared_payload=filtered_payload
            )
            content = self._request_chat_completion(prompt)
            payload = extract_json_object(content, label="Navigation point selector")
            selected_id = str(payload.get("candidate_id", "")).strip()
            for candidate in usable_candidates:
                if str(candidate.get("candidate_id")) == selected_id:
                    result = {
                        "candidate": dict(candidate),
                        "reason": str(payload.get("reason", "")).strip()
                        or "selected by navigation llm point selector",
                        "source": "openai_compatible_navigation_approach_point_selector",
                    }
                    if rejected_candidates:
                        result["rejected_candidate_ids"] = [
                            item.get("candidate_id") for item in rejected_candidates
                        ]
                        result["rejection_reasons"] = {
                            str(
                                item.get("candidate_id")
                            ): self.fallback_selector._constraint_violations(item)
                            for item in rejected_candidates
                        }
                    return result
            raise ValueError(f"Unknown or constraint-rejected candidate_id {selected_id!r}")
        except Exception as exc:
            fallback = self.fallback_selector.select_candidate(
                subtask=subtask,
                context=context,
                goal=goal,
                prepared_payload=prepared_payload,
            )
            fallback["reason"] = (
                f"fallback after openai point selector error: {exc}; {fallback['reason']}"
            )
            fallback["source"] = "heuristic_navigation_approach_point_selector_fallback"
            return fallback

    def _request_chat_completion(self, user_prompt: str) -> str:
        url = self._completion_url(self.config.base_url)
        api_key = (
            self.config.api_key or os.getenv(self.config.api_key_env) or os.getenv("OPENAI_API_KEY")
        )
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
        }
        attempts = max(1, int(self.config.max_retries) + 1)
        response = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout_s
                )
                if response.status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise
        if response is None:
            raise RuntimeError("Navigation point selector request produced no response")
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Navigation point selector response did not include choices")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Navigation point selector response content is empty")
        return content

    @staticmethod
    def _completion_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _build_prompt(*, subtask, goal: dict, prepared_payload: dict) -> str:
        metric_semantics = {
            "candidate_geometry_score": "distance error from the preferred manipulation standoff; lower is better",
            "approach_boundary_distance_m": "distance from the target object's physical boundary",
            "approach_distance_m": "horizontal distance from the target object's center",
            "handoff_distance_m": "distance from the nearest graph node to the continuous anchor; lower is better",
            "path_cost": "estimated travel effort to the candidate; lower is better but secondary to interaction readiness",
            "nearby_object_evidence.nearest_object_distance_m": "clearance around the candidate; higher is better",
            "nearby_object_evidence.path_nearest_object_distance_m": "clearance along the route; higher is better",
            "history_penalty": "recent failure penalty; lower is better",
            "target_part_score": "target-part alignment score for part-specific goals such as car trunk; lower is better",
            "target_part_alignment_m": "signed alignment with the requested part direction; higher is better",
        }
        payload = {
            "subtask_id": subtask.subtask_id,
            "action": subtask.action,
            "target": subtask.target,
            "goal": goal,
            "metric_semantics": metric_semantics,
            "hard_constraints": [
                "Candidate must be reachable by the navigation backend; reject explicit path_found=false/found=false candidates.",
                "Candidate must be physically plausible common-sense free space, not inside or directly against furniture.",
                "Candidate must leave enough free space for the robot footprint and downstream arm/body motion.",
                "Continuous anchors must stay close to the graph handoff point; reject large handoff_distance_m values.",
                "For switches, handles, buttons, and appliances, prefer an open stance beside/in front of the object, not a pose aimed into nearby furniture.",
            ],
            "selection_priorities": [
                "The chosen anchor must support the follow-up interaction with the named object.",
                "Prefer clear candidates and clear routes over candidates close to obstacles.",
                "Treat ordinary-object proximity along the semantic path as a soft penalty; Nav2 validates the dynamic route.",
                "Prefer low candidate_geometry_score; it is an error metric, not a quality score.",
                "For part-specific targets, prefer low target_part_score before path_cost.",
                "Use path_cost as a secondary tie-breaker after interaction readiness and clearance.",
            ],
            "selection_context": prepared_payload.get("selection_context"),
            "history": prepared_payload.get("history"),
            "candidates": prepared_payload.get("candidates"),
        }
        return (
            "Choose the best discrete object-approach anchor.\n"
            "Metric directionality: candidate_geometry_score lower is better. "
            "Do not maximize candidate_geometry_score. "
            "nearby_object_evidence clearance distances higher is better. "
            "target_part_score lower is better for part-specific targets. "
            "path_cost lower is better but secondary to interaction readiness.\n"
            f"Selection context JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "Return JSON only."
        )


__all__ = [
    "HeuristicNavigationApproachPointSelector",
    "OpenAICompatibleNavigationApproachPointSelector",
    "OpenAINavigationApproachPointSelectorConfig",
    "time",
]
