import pytest
import json
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen_mcp.specter.telemetry import TelemetryBroadcaster


@pytest.fixture
def telemetry_broadcaster():
    """Create a minimal TelemetryBroadcaster instance for testing."""
    broadcaster = TelemetryBroadcaster()
    # Mock internal state
    broadcaster._clients = {"proj123": set()}
    broadcaster._clients["proj123"].add(AsyncMock())
    broadcaster._clients["proj123"].add(AsyncMock())
    broadcaster._project_states = {"proj123": broadcaster._get_init_state()}
    broadcaster._heartbeat_counter = 0
    return broadcaster


@pytest.mark.asyncio(loop_scope="function")
async def test_heartbeat_message_format(telemetry_broadcaster):
    """Test that _send_heartbeat sends properly formatted heartbeat messages."""
    # Mock broadcast_state to detect if it's called (it shouldn't be)
    telemetry_broadcaster.broadcast_state = AsyncMock()
    
    # Mock the send_text method on WebSocket clients
    for client in telemetry_broadcaster._clients["proj123"]:
        client.send_text = AsyncMock()
    
    # Call the method under test
    await telemetry_broadcaster._send_heartbeat()
    
    # Verify broadcast_state was NOT called (we expect direct client sending)
    telemetry_broadcaster.broadcast_state.assert_not_called()
    
    # Verify each client received a properly formatted heartbeat message
    expected_message = json.dumps({"type": "heartbeat", "count": 1}, ensure_ascii=False)
    
    for client in telemetry_broadcaster._clients["proj123"]:
        client.send_text.assert_called_once_with(expected_message)


@pytest.mark.asyncio(loop_scope="function")
async def test_heartbeat_counter_increments(telemetry_broadcaster):
    """Test heartbeat counter increments correctly across multiple calls."""
    # Mock send_text on all clients
    for client in telemetry_broadcaster._clients["proj123"]:
        client.send_text = AsyncMock()
    
    # Call the method under test twice to verify counter increments
    await telemetry_broadcaster._send_heartbeat()
    await telemetry_broadcaster._send_heartbeat()
    
    # Verify counter incremented
    assert telemetry_broadcaster._heartbeat_counter == 2
    
    # Verify each client received the correct messages with incrementing count
    expected_messages = [
        json.dumps({"type": "heartbeat", "count": 1}, ensure_ascii=False),
        json.dumps({"type": "heartbeat", "count": 2}, ensure_ascii=False)
    ]
    
    for client in telemetry_broadcaster._clients["proj123"]:
        assert client.send_text.call_count == 2
        calls = client.send_text.call_args_list
        assert calls[0][0][0] == expected_messages[0]
        assert calls[1][0][0] == expected_messages[1]


@pytest.mark.asyncio(loop_scope="function")
async def test_heartbeat_handles_disconnected_clients(telemetry_broadcaster):
    """Test that disconnected clients are removed from client set."""
    # Create a client that will raise a connection error on send
    bad_client = AsyncMock()
    bad_client.send_text = AsyncMock(side_effect=ConnectionError("Connection lost"))
    
    # Create a good client
    good_client = AsyncMock()
    good_client.send_text = AsyncMock()
    
    # Set up clients
    telemetry_broadcaster._clients["proj123"] = {bad_client, good_client}
    
    # Call the method under test
    await telemetry_broadcaster._send_heartbeat()
    
    # Verify bad client was removed, good client remains
    assert bad_client not in telemetry_broadcaster._clients["proj123"]
    assert good_client in telemetry_broadcaster._clients["proj123"]
    
    # Verify good client received the message
    expected_message = json.dumps({"type": "heartbeat", "count": 1}, ensure_ascii=False)
    good_client.send_text.assert_called_once_with(expected_message)