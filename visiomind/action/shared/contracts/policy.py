from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PolicyRequest:
    observation: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    action: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(Protocol):
    def ping(self) -> bool:
        pass

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pass

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        pass

    def get_modality_config(self) -> dict[str, Any]:
        pass
