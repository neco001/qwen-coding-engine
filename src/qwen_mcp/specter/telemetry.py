import asyncio
import json
from typing import Set, Dict, Any, Optional

from fastapi import WebSocket
from websockets.exceptions import ConnectionClosed

# Global instance for easy access across the package
_global_broadcaster: Optional["TelemetryBroadcaster"] = None


def get_broadcaster() -> "TelemetryBroadcaster":
    global _global_broadcaster
    if _global_broadcaster is None:
        _global_broadcaster = TelemetryBroadcaster()
    return _global_broadcaster


class TelemetryBroadcaster:
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._last_state: Dict[str, Any] = {
            "active_model": "None",
            "session_tokens": {"prompt": 0, "completion": 0},
            "loop_iteration": 0,
            "role_mapping": {},
        }

    async def add_client(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)
            # Send current state to late joiners
            try:
                await websocket.send_text(json.dumps(self._last_state))
            except Exception:
                pass

    async def remove_client(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast_state(self, payload: dict) -> None:
        """
        Broadcasts the current state to all connected clients in parallel.
        Updates the local cache for late joiners.
        """
        self._last_state.update(payload)

        if not self._clients:
            return

        message = json.dumps(self._last_state)
        disconnected = set()

        async with self._lock:
            clients = list(self._clients)

        # Send to all clients in parallel
        tasks = [client.send_text(message) for client in clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for client, result in zip(clients, results):
            if isinstance(result, (ConnectionClosed, Exception)):
                disconnected.add(client)

        if disconnected:
            async with self._lock:
                self._clients -= disconnected

    async def report_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ):
        """Helper to broadcast model usage specifically."""
        await self.broadcast_state(
            {
                "active_model": model,
                "session_tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                },
            }
        )
