from __future__ import annotations

from typing import Any

from voltron.shared.contracts import AgentCapability


def collect_agent_capabilities(agents: list[Any]) -> list[AgentCapability]:
    capabilities: list[AgentCapability] = []
    for agent in agents:
        manifest = getattr(agent, "capability_manifest", None)
        if not callable(manifest):
            continue
        for capability in manifest():
            if isinstance(capability, AgentCapability):
                capabilities.append(capability)
    return capabilities


def inject_agent_capabilities(brain_agent: Any, agents: list[Any]) -> list[AgentCapability]:
    capabilities = collect_agent_capabilities(agents)
    setter = getattr(brain_agent, "set_agent_capabilities", None)
    if callable(setter):
        setter(capabilities)
    return capabilities
