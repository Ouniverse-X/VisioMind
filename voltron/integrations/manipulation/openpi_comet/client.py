"""OpenPI Comet websocket client."""

from __future__ import annotations

import logging
from typing import Any

import requests
import websockets.exceptions
import websockets.sync.client

from voltron.integrations.manipulation.openpi_comet.protocol import Packer, unpackb
from voltron.shared.errors import AdapterError

logger = logging.getLogger(__name__)


class OpenPICometClient:
    def __init__(self, endpoint: str, timeout_s: float = 60.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.server_metadata: dict[str, Any] = {}
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._packer = Packer()

    def http_base(self) -> str:
        return self.endpoint.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")

    def ping(self) -> bool:
        try:
            response = requests.get(f"{self.http_base()}/healthz", timeout=min(self.timeout_s, 5.0))
            return response.ok
        except Exception as exc:
            raise AdapterError(f"OpenPI Comet ping failed: {exc}") from exc

    def ensure_connected(self) -> None:
        if self._ws is not None:
            return
        try:
            self._ws = websockets.sync.client.connect(
                self.endpoint,
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=300,
                close_timeout=self.timeout_s,
            )
            raw = self._ws.recv(timeout=self.timeout_s)
            self.server_metadata = unpackb(raw)
            logger.info("OpenPICometClient connected to %s", self.endpoint)
        except Exception as exc:
            self._ws = None
            raise AdapterError(f"OpenPI Comet WebSocket connection failed ({self.endpoint}): {exc}") from exc

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_connected()
        assert self._ws is not None
        try:
            self._ws.send(self._packer.pack(payload))
            raw = self._ws.recv(timeout=self.timeout_s)
        except websockets.exceptions.ConnectionClosedError:
            self._ws = None
            raise AdapterError("OpenPI Comet WebSocket connection closed")
        except Exception as exc:
            self._ws = None
            raise AdapterError(f"OpenPI Comet communication error: {exc}") from exc
        if isinstance(raw, str):
            raise AdapterError(f"OpenPI Comet server error: {raw}")
        return unpackb(raw)

    def reset(self) -> dict[str, Any]:
        try:
            self.ensure_connected()
            assert self._ws is not None
            self._ws.send(self._packer.pack({"reset": True}))
        except Exception as exc:
            self.close()
            raise AdapterError(f"OpenPI Comet reset failed: {exc}") from exc
        return {"status": "reset_sent", "endpoint": self.endpoint}

    def close(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception as exc:
            logger.warning("OpenPI Comet websocket close failed: %s", exc)


__all__ = ["OpenPICometClient"]
