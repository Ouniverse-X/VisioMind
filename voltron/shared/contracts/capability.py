from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.enums import AgentName


@dataclass(frozen=True)
class AgentCapability:
    capability_id: str
    agent: AgentName
    kind: str
    action_names: tuple[str, ...]
    description: str
    intent_examples: tuple[str, ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "agent": self.agent.value,
            "kind": self.kind,
            "action_names": list(self.action_names),
            "description": self.description,
            "intent_examples": list(self.intent_examples),
            "input_schema": deepcopy(self.input_schema),
            "output_schema": deepcopy(self.output_schema),
        }


class AgentCapabilityProvider(Protocol):
    def capability_manifest(self) -> list[AgentCapability]:
        pass


def serialize_agent_capabilities(
    capabilities: list[AgentCapability],
) -> list[dict[str, Any]]:
    return [capability.to_dict() for capability in capabilities]
