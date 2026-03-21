"""
Unit tests for HUD Telemetry Broadcaster
Tests the telemetry.py TelemetryBroadcaster class
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


class MockWebSocket:
    """Mock WebSocket for testing."""
    def __init__(self):
        self.sent_messages = []
        self.closed = False
    
    async def send_text(self, message):
        if not self.closed:
            self.sent_messages.append(message)
    
    async def close(self):
        self.closed = True


@pytest.fixture
def broadcaster():
    """Create a fresh TelemetryBroadcaster instance."""
    from qwen_mcp.specter.telemetry import TelemetryBroadcaster
    return TelemetryBroadcaster()


@pytest.fixture
def mock_ws():
    """Create a mock WebSocket."""
    return MockWebSocket()


@pytest.mark.asyncio
async def test_init_state_has_status_field(broadcaster):
    """Test that initial state includes status field."""
    state = broadcaster._get_init_state()
    assert "status" in state
    assert state["status"] == "idle"
    assert "operation" in state
    assert "progress_percent" in state
    assert "heartbeat" in state


@pytest.mark.asyncio
async def test_add_client_sends_state_immediately(broadcaster, mock_ws):
    """Test that adding a client sends the current state immediately."""
    await broadcaster.add_client(mock_ws, project_id="test")
    
    assert len(mock_ws.sent_messages) == 1
    data = json.loads(mock_ws.sent_messages[0])
    assert "active_model" in data
    assert "status" in data


@pytest.mark.asyncio
async def test_broadcast_state_updates_clients(broadcaster, mock_ws):
    """Test that broadcasting state updates connected clients."""
    await broadcaster.add_client(mock_ws, project_id="test")
    mock_ws.sent_messages.clear()
    
    await broadcaster.broadcast_state({"active_model": "test-model"}, project_id="test")
    
    assert len(mock_ws.sent_messages) == 1
    data = json.loads(mock_ws.sent_messages[0])
    assert data["active_model"] == "test-model"


@pytest.mark.asyncio
async def test_start_request_sets_live_status(broadcaster, mock_ws):
    """Test that start_request sets status to 'live'."""
    await broadcaster.add_client(mock_ws, project_id="test")
    mock_ws.sent_messages.clear()
    
    await broadcaster.start_request(project_id="test")
    
    data = json.loads(mock_ws.sent_messages[0])
    assert data["status"] == "live"
    assert data["operation"] == "Processing request..."
    assert data["progress_percent"] == 0


@pytest.mark.asyncio
async def test_update_progress_sets_processing_status(broadcaster, mock_ws):
    """Test that update_progress sets status to 'processing'."""
    await broadcaster.add_client(mock_ws, project_id="test")
    mock_ws.sent_messages.clear()
    
    await broadcaster.update_progress(50, "Testing...", project_id="test")
    
    data = json.loads(mock_ws.sent_messages[0])
    assert data["status"] == "processing"
    assert data["progress_percent"] == 50
    assert data["operation"] == "Testing..."


@pytest.mark.asyncio
async def test_end_request_resets_status(broadcaster, mock_ws):
    """Test that end_request resets status to 'idle'."""
    await broadcaster.add_client(mock_ws, project_id="test")
    
    # First start a request
    await broadcaster.start_request(project_id="test")
    mock_ws.sent_messages.clear()
    
    # Then end it
    await broadcaster.end_request(project_id="test")
    
    data = json.loads(mock_ws.sent_messages[0])
    assert data["status"] == "idle"
    assert data["operation"] == ""


@pytest.mark.asyncio
async def test_multi_session_state_isolation(broadcaster, mock_ws):
    """Test that different project_ids have isolated STATE but broadcast goes to all clients (HUD compatibility)."""
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    
    await broadcaster.add_client(ws1, project_id="session1")
    await broadcaster.add_client(ws2, project_id="session2")
    
    # Update session1's state
    await broadcaster.broadcast_state({"active_model": "model1"}, project_id="session1")
    
    # Check that session1's STATE has model1 (but both clients receive the broadcast)
    state1 = broadcaster.get_state("session1")
    assert state1["active_model"] == "model1"
    
    # Update session2's state
    await broadcaster.broadcast_state({"active_model": "model2"}, project_id="session2")
    
    # Check that session2's STATE has model2
    state2 = broadcaster.get_state("session2")
    assert state2["active_model"] == "model2"
    
    # Verify states are isolated (this is the key test)
    assert broadcaster.get_state("session1")["active_model"] == "model1"
    assert broadcaster.get_state("session2")["active_model"] == "model2"


@pytest.mark.asyncio
async def test_report_usage_updates_tokens(broadcaster, mock_ws):
    """Test that report_usage updates token counts."""
    await broadcaster.add_client(mock_ws, project_id="test")
    mock_ws.sent_messages.clear()
    
    await broadcaster.report_usage("test-model", 100, 50, project_id="test")
    
    data = json.loads(mock_ws.sent_messages[0])
    assert data["request_tokens"]["prompt"] == 100
    assert data["request_tokens"]["completion"] == 50
    assert data["session_tokens"]["prompt"] == 100
    assert data["session_tokens"]["completion"] == 50


@pytest.mark.asyncio
async def test_heartbeat_increments_counter(broadcaster):
    """Test that heartbeat increments the counter."""
    initial_counter = broadcaster._heartbeat_counter
    await broadcaster._send_heartbeat()
    assert broadcaster._heartbeat_counter == initial_counter + 1
