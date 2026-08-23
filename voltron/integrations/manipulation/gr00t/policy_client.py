"""Adapter for GR00T PolicyClient backend."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from voltron.shared.errors import AdapterError


class Gr00tPolicyAdapter:
    """Thin adapter over GR00T PolicyClient.

    This isolates ZeroMQ client and import path details from agent logic.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15000,
        strict: bool = False,
        api_token: str | None = None,
        policy_client: Any | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.strict = strict
        self.api_token = api_token

        if policy_client is not None:
            self._client = policy_client
        else:
            client_cls = self._load_policy_client_class()
            self._client = client_cls(
                host=self.host,
                port=self.port,
                timeout_ms=self.timeout_ms,
                strict=self.strict,
                api_token=self.api_token,
            )

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception as exc:  # pragma: no cover - backend/network dependent
            raise AdapterError(f"Policy ping failed: {exc}") from exc

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return self._client.get_action(observation, options=options)
        except Exception as exc:
            raise AdapterError(f"Policy get_action failed: {exc}") from exc

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self._client.reset(options=options)
        except Exception as exc:
            raise AdapterError(f"Policy reset failed: {exc}") from exc

    def get_modality_config(self) -> dict[str, Any]:
        try:
            return self._client.get_modality_config()
        except Exception as exc:
            raise AdapterError(f"Policy get_modality_config failed: {exc}") from exc

    @staticmethod
    def _load_policy_client_class() -> type:
        """Load PolicyClient even when Isaac-GR00T package is not pip-installed."""
        try:
            module = importlib.import_module("gr00t.policy.server_client")
            return getattr(module, "PolicyClient")
        except ModuleNotFoundError:
            # Fallback to local repo layout: <repo>/isaac_gr00t_learn
            repo_root = Path(__file__).resolve().parents[3]
            local_gr00t_root = repo_root / "isaac_gr00t_learn"
            if local_gr00t_root.exists():
                Gr00tPolicyAdapter._prepend_local_gr00t_repo(local_gr00t_root)
                module = importlib.import_module("gr00t.policy.server_client")
                return getattr(module, "PolicyClient")

            raise AdapterError(
                "Cannot import gr00t.policy.server_client.PolicyClient. "
                "Install Isaac-GR00T or ensure <repo>/isaac_gr00t_learn exists."
            )

    @staticmethod
    def _prepend_local_gr00t_repo(local_gr00t_root: Path) -> None:
        preferred_paths: list[str] = []
        try:
            import bddl

            bddl_file = getattr(bddl, "__file__", None)
            if isinstance(bddl_file, str) and bddl_file:
                preferred_paths.append(str(Path(bddl_file).resolve().parents[1]))
        except Exception:
            pass

        preferred_paths.append(str(local_gr00t_root))
        for candidate in reversed(preferred_paths):
            if candidate and candidate not in sys.path:
                sys.path.insert(0, candidate)
