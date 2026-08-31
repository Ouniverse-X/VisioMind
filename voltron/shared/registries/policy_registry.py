from __future__ import annotations

from typing import Any


class PolicyRegistry:
    def __init__(self, *, default_policy_id: str | None = None) -> None:
        self._policies: dict[str, Any] = {}
        self._default_policy_id = default_policy_id

    def register(self, policy_id: str, policy: Any) -> None:
        self._policies[policy_id] = policy

    def get(self, policy_id: str | None = None) -> Any:
        resolved_id = policy_id or self._default_policy_id
        if resolved_id is None:
            raise KeyError("No policy id was provided and no default policy is configured.")
        return self._policies[resolved_id]

    def policy_ids(self) -> list[str]:
        return sorted(self._policies.keys())
