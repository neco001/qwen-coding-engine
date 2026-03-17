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
        self._clients: Dict[str, Set[WebSocket]] = {"default": set()}
        self._lock = asyncio.Lock()
        self._project_states: Dict[str, Dict[str, Any]] = {}

    def _get_init_state(self) -> Dict[str, Any]:
        return {
            "active_model": "Standby...",
            "request_tokens": {"prompt": 0, "completion": 0},
            "session_tokens": {"prompt": 0, "completion": 0},
            "loop_iteration": 0,
            "role_mapping": {},
            "thinking": "",
            "streaming_content": "",
            "session_images": 0,
            "request_images": 0,
            "session_prompt_total": 0,
            "session_completion_total": 0,
            "request_prompt_total": 0,
            "request_completion_total": 0,
            "session_images_total": 0,
            "request_images_total": 0,
            "thinking_buffer": "",
            "content_buffer": "",
            "last_interaction_time": 0
        }

    def get_state(self, project_id: str = "default") -> Dict[str, Any]:
        if project_id not in self._project_states:
            self._project_states[project_id] = self._get_init_state()
        return self._project_states[project_id]

    async def add_client(self, websocket: WebSocket, project_id: str = "default") -> None:
        async with self._lock:
            if project_id not in self._clients:
                self._clients[project_id] = set()
            self._clients[project_id].add(websocket)
            
            # Send current state to late joiners
            state = self.get_state(project_id)
            try:
                await websocket.send_text(json.dumps(state, ensure_ascii=False))
            except Exception:
                pass

    async def remove_client(self, websocket: WebSocket) -> None:
        async with self._lock:
            for project_id in self._clients:
                self._clients[project_id].discard(websocket)

    async def broadcast_state(self, payload: dict, project_id: str = "default") -> None:
        """
        Broadcasts the current state to all connected clients for a specific project.
        """
        state = self.get_state(project_id)
        state.update(payload)

        clients = self._clients.get(project_id, set())
        if not clients:
            return

        message = json.dumps(state, ensure_ascii=False)
        disconnected = set()

        async with self._lock:
            target_clients = list(clients)

        # Send to all clients of this project in parallel
        tasks = [client.send_text(message) for client in target_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for client, result in zip(target_clients, results):
            if isinstance(result, (ConnectionClosed, Exception)):
                disconnected.add(client)

        if disconnected:
            async with self._lock:
                self._clients[project_id] -= disconnected

    async def start_request(self, project_id: str = "default"):
        """Resets counters and buffers for a new high-level user request."""
        state = self.get_state(project_id)
        state["request_prompt_total"] = 0
        state["request_completion_total"] = 0
        state["thinking_buffer"] = ""
        state["content_buffer"] = ""
        state["request_images_total"] = 0
        
        await self.broadcast_state({
            "request_tokens": {"prompt": 0, "completion": 0},
            "thinking": "",
            "streaming_content": "",
            "loop_iteration": 0,
            "request_images": 0
        }, project_id=project_id)

    async def update_stream(self, thinking: str = "", content: str = "", project_id: str = "default"):
        """Appends to the current stream buffers and broadcasts."""
        state = self.get_state(project_id)
        
        if thinking:
            state["thinking_buffer"] += thinking
        if content:
            state["content_buffer"] += content
            
        await self.broadcast_state({
            "thinking": state["thinking_buffer"],
            "streaming_content": state["content_buffer"]
        }, project_id=project_id)

    async def report_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int, 
        is_image: bool = False, project_id: str = "default"
    ):
        """Accumulates token usage. Resets 'Request' level if inactivity > 60s."""
        import time
        now = time.time()
        state = self.get_state(project_id)
        
        # SMART RESET: If more than 60s passed since last LLM activity
        if state["last_interaction_time"] > 0 and (now - state["last_interaction_time"] > 60):
            state["request_prompt_total"] = 0
            state["request_completion_total"] = 0
            state["thinking_buffer"] = ""
            state["content_buffer"] = ""

        state["last_interaction_time"] = now
        
        if is_image:
            state["session_images_total"] += 1
            state["request_images_total"] += 1

        state["session_prompt_total"] += prompt_tokens
        state["session_completion_total"] += completion_tokens
        
        state["request_prompt_total"] += prompt_tokens
        state["request_completion_total"] += completion_tokens
        
        await self.broadcast_state(
            {
                "active_model": model,
                "request_tokens": {
                    "prompt": state["request_prompt_total"],
                    "completion": state["request_completion_total"],
                },
                "session_tokens": {
                    "prompt": state["session_prompt_total"],
                    "completion": state["session_completion_total"],
                },
                "session_images": state["session_images_total"],
                "request_images": state["request_images_total"],
            },
            project_id=project_id
        )
