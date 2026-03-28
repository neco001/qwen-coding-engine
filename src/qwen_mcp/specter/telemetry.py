import asyncio
import json
import time
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, field

from fastapi import WebSocket
from websockets.exceptions import ConnectionClosed

# Global instance for easy access across the package
_global_broadcaster: Optional["TelemetryBroadcaster"] = None
_heartbeat_task: Optional[asyncio.Task] = None


def get_broadcaster() -> "TelemetryBroadcaster":
    global _global_broadcaster
    if _global_broadcaster is None:
        _global_broadcaster = TelemetryBroadcaster()
    return _global_broadcaster


async def start_heartbeat_loop():
    """Background task that sends heartbeat to all clients every 5 seconds."""
    global _heartbeat_task
    broadcaster = get_broadcaster()
    while True:
        try:
            await asyncio.sleep(5)
            await broadcaster._send_heartbeat()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Telemetry] Heartbeat error: {e}")


def ensure_heartbeat_running():
    """Ensure heartbeat background task is running."""
    global _heartbeat_task
    if _heartbeat_task is None or _heartbeat_task.done():
        try:
            loop = asyncio.get_event_loop()
            _heartbeat_task = loop.create_task(start_heartbeat_loop())
        except Exception:
            pass


class TelemetryBroadcaster:
    def __init__(self):
        self._clients: Dict[str, Set[WebSocket]] = {"default": set()}
        self._lock = asyncio.Lock()
        self._project_states: Dict[str, Dict[str, Any]] = {}
        self._heartbeat_counter = 0

    def _get_init_state(self) -> Dict[str, Any]:
        return {
            "active_model": "Standby...",
            "status": "idle",  # "idle", "live", "processing"
            "operation": "",  # Current operation description
            "progress_percent": 0,  # 0-100 for long operations
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
            "last_interaction_time": time.time(),
            "heartbeat": 0
        }

    def get_state(self, project_id: str = "default") -> Dict[str, Any]:
        if project_id not in self._project_states:
            self._project_states[project_id] = self._get_init_state()
        return self._project_states[project_id]

    async def add_client(self, websocket: WebSocket, project_id: str = "default") -> None:
        """Add client and send current state immediately. Also starts heartbeat if needed."""
        async with self._lock:
            if project_id not in self._clients:
                self._clients[project_id] = set()
            self._clients[project_id].add(websocket)
        
        # CRITICAL: Send state immediately on connect (outside lock to avoid deadlock)
        state = self.get_state(project_id)
        try:
            await websocket.send_text(json.dumps(state, ensure_ascii=False))
        except Exception:
            async with self._lock:
                self._clients[project_id].discard(websocket)
        
        # Ensure heartbeat is running for long operations
        ensure_heartbeat_running()

    async def remove_client(self, websocket: WebSocket) -> None:
        async with self._lock:
            for project_id in self._clients:
                self._clients[project_id].discard(websocket)

    async def broadcast_state(self, payload: dict, project_id: str = "default") -> None:
        """
        Broadcasts the current state to ALL connected clients (HUD compatibility).
        CRITICAL: Updates ALL project states to ensure consistent data across all clients.
        """
        # Step 1: Update states and prepare message under lock
        async with self._lock:
            # Ensure the specified project_id exists
            if project_id not in self._project_states:
                self._project_states[project_id] = self._get_init_state()
            
            # Update ALL project states with the payload
            for proj_id in self._project_states:
                state = self._project_states[proj_id]
                state.update(payload)
            
            # Collect all clients across all projects
            all_clients = []
            for proj_clients in self._clients.values():
                all_clients.extend(list(proj_clients))

            if not all_clients:
                return

            # Get state and prepare message while holding lock
            state = self._project_states[project_id]
            message = json.dumps(state, ensure_ascii=False)

        # Step 2: Send messages OUTSIDE lock to avoid blocking other operations
        disconnected = set()
        tasks = [client.send_text(message) for client in all_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for client, result in zip(all_clients, results):
            if isinstance(result, (ConnectionClosed, Exception)):
                disconnected.add(client)

        # Step 3: Remove disconnected clients under lock
        if disconnected:
            async with self._lock:
                for proj_id in self._clients:
                    self._clients[proj_id] -= disconnected

    async def start_request(self, project_id: str = "default"):
        """Resets counters and buffers for a new high-level user request."""
        state = self.get_state(project_id)
        state["request_prompt_total"] = 0
        state["request_completion_total"] = 0
        state["thinking_buffer"] = ""  # FIX: Must be string (concatenated in update_stream)
        state["content_buffer"] = ""
        state["request_images_total"] = 0
        state["status"] = "live"
        state["operation"] = "Processing request..."
        state["progress_percent"] = 0
        
        await self.broadcast_state({
            "request_tokens": {"prompt": 0, "completion": 0},
            "thinking": "",
            "streaming_content": "",
            "loop_iteration": 0,
            "request_images": 0,
            "status": "live",
            "operation": "Processing request...",
            "progress_percent": 0
        }, project_id=project_id)

    async def update_stream(self, thinking: str = "", content: str = "", project_id: str = "default"):
        """Appends to the current stream buffers and broadcasts."""
        state = self.get_state(project_id)
        
        if thinking:
            state["thinking_buffer"] += thinking
        if content:
            state["content_buffer"] += content
            # Update status to live when content is streaming
            state["status"] = "live"
            state["operation"] = "Generating response..."
        
        # Fire-and-forget broadcast to avoid blocking streaming
        # Use create_task to avoid waiting for WebSocket sends
        asyncio.create_task(self.broadcast_state({
            "thinking": state["thinking_buffer"],
            "streaming_content": state["content_buffer"],
            "status": state["status"],
            "operation": state["operation"]
        }, project_id=project_id))

    async def update_progress(self, percent: int, operation: str = "", project_id: str = "default"):
        """Update progress for long operations (0-100%)."""
        state = self.get_state(project_id)
        state["progress_percent"] = min(100, max(0, percent))
        if operation:
            state["operation"] = operation
        state["status"] = "processing"
        
        await self.broadcast_state({
            "progress_percent": state["progress_percent"],
            "operation": state["operation"],
            "status": "processing"
        }, project_id=project_id)

    async def end_request(self, project_id: str = "default"):
        """Mark request as complete and reset status."""
        state = self.get_state(project_id)
        state["status"] = "idle"
        state["operation"] = ""
        state["progress_percent"] = 100
        
        await self.broadcast_state({
            "status": "idle",
            "operation": "",
            "progress_percent": 100
        }, project_id=project_id)

    async def _send_heartbeat(self):
        """Internal method to send heartbeat to all clients."""
        self._heartbeat_counter += 1
        for project_id in list(self._project_states.keys()):
            state = self.get_state(project_id)
            state["heartbeat"] = self._heartbeat_counter
            # Only send if clients are connected
            if project_id in self._clients and self._clients[project_id]:
                await self.broadcast_state({"heartbeat": self._heartbeat_counter}, project_id=project_id)

    async def report_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int, 
        is_image: bool = False, project_id: str = "default"
    ):
        """Accumulates token usage. Resets 'Request' level if inactivity > 60s."""
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
