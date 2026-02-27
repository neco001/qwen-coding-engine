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
            "request_tokens": {"prompt": 0, "completion": 0},
            "session_tokens": {"prompt": 0, "completion": 0},
            "loop_iteration": 0,
            "role_mapping": {},
            "thinking": "",
            "streaming_content": "",
        }
        # Accumulation state - Session
        self._session_prompt_total: int = 0
        self._session_completion_total: int = 0
        
        # Accumulation state - Request
        self._request_prompt_total: int = 0
        self._request_completion_total: int = 0
        self._thinking_buffer: str = ""
        self._content_buffer: str = ""
        self._last_interaction_time: float = 0

    async def add_client(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)
            # Send current state to late joiners
            try:
                await websocket.send_text(json.dumps(self._last_state, ensure_ascii=False))
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

        message = json.dumps(self._last_state, ensure_ascii=False)
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

    async def start_request(self):
        """Resets counters and buffers for a new high-level user request."""
        self._request_prompt_total = 0
        self._request_completion_total = 0
        self._thinking_buffer = ""
        self._content_buffer = ""
        await self.broadcast_state({
            "request_tokens": {"prompt": 0, "completion": 0},
            "thinking": "",
            "streaming_content": "",
            "loop_iteration": 0
        })

    async def update_stream(self, thinking: str = "", content: str = ""):
        """Appends to the current stream buffers and broadcasts."""
        if thinking:
            self._thinking_buffer += thinking
        if content:
            self._content_buffer += content
            
        await self.broadcast_state({
            "thinking": self._thinking_buffer,
            "streaming_content": self._content_buffer
        })

    async def report_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ):
        """Accumulates token usage. Resets 'Request' level if inactivity > 60s."""
        import time
        now = time.time()
        
        # SMART RESET: If more than 60s passed since last LLM activity, 
        # assume it's a new user prompt and reset request tokens.
        if self._last_interaction_time > 0 and (now - self._last_interaction_time > 60):
            self._request_prompt_total = 0
            self._request_completion_total = 0
            self._thinking_buffer = ""
            self._content_buffer = ""

        self._last_interaction_time = now
        
        self._session_prompt_total += prompt_tokens
        self._session_completion_total += completion_tokens
        
        self._request_prompt_total += prompt_tokens
        self._request_completion_total += completion_tokens
        
        await self.broadcast_state(
            {
                "active_model": model,
                "request_tokens": {
                    "prompt": self._request_prompt_total,
                    "completion": self._request_completion_total,
                },
                "session_tokens": {
                    "prompt": self._session_prompt_total,
                    "completion": self._session_completion_total,
                },
            }
        )
