import asyncio
import json
import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

# We expect this import to fail initially (RED phase)
from qwen_mcp.specter.telemetry import TelemetryBroadcaster

app = FastAPI()
broadcaster = TelemetryBroadcaster()


@app.websocket("/ws/telemetry")
async def telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    await broadcaster.add_client(websocket)
    try:
        while True:
            # Keep connection open; no need to read
            await asyncio.sleep(1)
            # Add a small yield to allow other tasks to run
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        await broadcaster.remove_client(websocket)


@pytest.mark.asyncio
async def test_telemetry_broadcaster_pushes_updates_to_websocket():
    # TestClient block connection, so we shouldn't use classic TestClient asyncio background tasks straightforwardly.
    # But let's follow the AI's exact instruction and see it fail.

    # Actually, TestClient is synchronous in usage context:
    client = TestClient(app)
    received_messages = []

    with client.websocket_connect("/ws/telemetry") as websocket:
        # First, receive and discard the initial state sent by add_client
        initial_msg = websocket.receive_json()
        assert initial_msg["active_model"] == "Standby..."  # Initial state
        
        # We simulate the broadcast that would come from the MCP server.
        payload = {
            "session_tokens": {"test": 123},
            "active_model": "gpt-4o",
            "loop_iteration": 42,
        }

        # Because we're in a sync block of TestClient but inside an async test,
        # this pattern from AI might hang. Let's execute it directly:
        loop = asyncio.get_running_loop()
        broadcast_task = loop.create_task(broadcaster.broadcast_state(payload))

        # Wait for the task to schedule
        await asyncio.sleep(0.1)

        try:
            msg = websocket.receive_json()
            received_messages.append(msg)
        except Exception as e:
            pytest.fail(f"Expected to receive broadcast message, got: {e}")

    # Assertions
    assert len(received_messages) == 1, "Expected exactly one broadcast message"
    msg = received_messages[0]
    assert "session_tokens" in msg
    assert "active_model" in msg
    assert "loop_iteration" in msg
    assert msg["active_model"] == "gpt-4o"
    assert msg["loop_iteration"] == 42
