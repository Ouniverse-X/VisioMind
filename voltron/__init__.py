"""Voltron multi-agent orchestration package."""

try:
    from .runtime.orchestrator.closed_loop import ClosedLoopOrchestrator
    from .runtime.orchestrator.open_loop import VoltronOrchestrator
except ImportError:  # pragma: no cover - fallback for direct module execution during test collection
    from runtime.orchestrator.closed_loop import ClosedLoopOrchestrator
    from runtime.orchestrator.open_loop import VoltronOrchestrator

__all__ = ["VoltronOrchestrator", "ClosedLoopOrchestrator"]
