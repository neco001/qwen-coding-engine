"""
Unit Tests for Sparring Engine v2

Tests verify:
- Flash mode execution
- Discovery mode with session creation
- Red/Blue/White step execution
- Session checkpointing integration
- Guided UX responses
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from qwen_mcp.engines.sparring_v2 import SparringEngineV2, SparringResponse, TIMEOUTS
from qwen_mcp.engines.session_store import SessionStore


@pytest.fixture
def mock_client():
    """Create a mock DashScopeClient."""
    client = MagicMock()
    client.generate_completion = AsyncMock()
    return client


@pytest.fixture
def engine(mock_client):
    """Create SparringEngineV2 with mock client and temp session store."""
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="sparring_v2_test_")
    store = SessionStore(storage_dir=temp_dir)
    return SparringEngineV2(client=mock_client, session_store=store)


# =============================================================================
# Test: Invalid Mode
# =============================================================================

class TestInvalidMode:
    """Tests for invalid mode handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self, engine):
        """Invalid mode should return error response."""
        response = await engine.execute(mode="invalid")
        
        assert response.success is False
        assert "Unknown mode" in response.error
        assert response.next_step is None


# =============================================================================
# Test: Flash Mode
# =============================================================================

class TestFlashMode:
    """Tests for flash mode execution."""
    
    @pytest.mark.asyncio
    async def test_flash_executes_analyst_and_drafter(self, engine, mock_client):
        """Flash mode should execute analyst then drafter."""
        mock_client.generate_completion.side_effect = [
            "Analysis result",  # Analyst
            "<thought>Reasoning</thought>Final strategy"  # Drafter
        ]
        
        response = await engine.execute(
            mode="flash",
            topic="Test Topic",
            context="Test Context"
        )
        
        assert response.success is True
        assert response.step_completed == "flash"
        assert response.next_step is None  # Flash is complete, no next step
        assert "strategy" in response.result
        assert mock_client.generate_completion.call_count == 2
    
    @pytest.mark.asyncio
    async def test_flash_response_format(self, engine, mock_client):
        """Flash response should have proper format."""
        mock_client.generate_completion.side_effect = [
            "Analysis",
            "<thought>Reasoning</thought>Strategy"
        ]
        
        response = await engine.execute(
            mode="flash",
            topic="Test",
            context=""
        )
        
        assert response.success is True
        assert response.session_id is None  # Flash doesn't create session
        assert response.step_completed == "flash"
        assert response.next_command is None


# =============================================================================
# Test: Discovery Mode
# =============================================================================

class TestDiscoveryMode:
    """Tests for discovery mode execution."""
    
    @pytest.mark.asyncio
    async def test_discovery_creates_session(self, engine, mock_client):
        """Discovery should create a new session."""
        mock_client.generate_completion.return_value = '''
```json
{
    "red_role": "Test Red",
    "red_profile": "Red Profile",
    "blue_role": "Test Blue",
    "blue_profile": "Blue Profile",
    "white_role": "Test White",
    "white_profile": "White Profile"
}
```
'''
        
        response = await engine.execute(
            mode="discovery",
            topic="Test Topic",
            context="Test Context"
        )
        
        assert response.success is True
        assert response.session_id is not None
        assert response.step_completed == "discovery"
        assert response.next_step == "red"
        assert "roles" in response.result
    
    @pytest.mark.asyncio
    async def test_discovery_uses_default_roles_on_parse_failure(self, engine, mock_client):
        """Discovery should use default roles if JSON parse fails."""
        mock_client.generate_completion.return_value = "Invalid JSON response"
        
        response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        assert response.success is True
        assert response.session_id is not None
        assert "red_role" in response.result["roles"]
    
    @pytest.mark.asyncio
    async def test_discovery_guided_ux(self, engine, mock_client):
        """Discovery response should include guided UX hints."""
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "P", "blue_role": "B", "blue_profile": "P", "white_role": "W", "white_profile": "P"}
'''
        
        response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        assert response.next_step == "red"
        assert response.next_command is not None
        assert f"session_id='{response.session_id}'" in response.next_command


# =============================================================================
# Test: Red Cell Mode
# =============================================================================

class TestRedCellMode:
    """Tests for Red Cell mode execution."""
    
    @pytest.mark.asyncio
    async def test_red_requires_session_id(self, engine):
        """Red mode should fail without session_id."""
        response = await engine.execute(mode="red")
        
        assert response.success is False
        assert "session_id" in response.error
    
    @pytest.mark.asyncio
    async def test_red_fails_for_nonexistent_session(self, engine):
        """Red mode should fail for nonexistent session."""
        response = await engine.execute(
            mode="red",
            session_id="sp_nonexistent"
        )
        
        assert response.success is False
        assert "not found" in response.error
    
    @pytest.mark.asyncio
    async def test_red_executes_critique(self, engine, mock_client):
        """Red mode should execute critique and update session."""
        # First create a session via discovery
        mock_client.generate_completion.return_value = '''
{"red_role": "Test Red", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        discovery_response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        # Now execute red
        mock_client.generate_completion.return_value = "<thought>Red reasoning</thought>Red critique content"
        
        red_response = await engine.execute(
            mode="red",
            session_id=discovery_response.session_id
        )
        
        assert red_response.success is True
        assert red_response.step_completed == "red"
        assert red_response.next_step == "blue"
        assert "critique" in red_response.result


# =============================================================================
# Test: Blue Cell Mode
# =============================================================================

class TestBlueCellMode:
    """Tests for Blue Cell mode execution."""
    
    @pytest.mark.asyncio
    async def test_blue_requires_session_id(self, engine):
        """Blue mode should fail without session_id."""
        response = await engine.execute(mode="blue")
        
        assert response.success is False
        assert "session_id" in response.error
    
    @pytest.mark.asyncio
    async def test_blue_fails_without_red_critique(self, engine, mock_client):
        """Blue mode should fail if red critique is missing."""
        # Create session but skip red step
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        discovery_response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        # Try blue without red
        response = await engine.execute(
            mode="blue",
            session_id=discovery_response.session_id
        )
        
        assert response.success is False
        assert "red" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_blue_executes_defense(self, engine, mock_client):
        """Blue mode should execute defense after red."""
        # Discovery
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        discovery_response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        # Red
        mock_client.generate_completion.return_value = "Red critique"
        await engine.execute(
            mode="red",
            session_id=discovery_response.session_id
        )
        
        # Blue
        mock_client.generate_completion.return_value = "<thought>Blue reasoning</thought>Blue defense"
        
        blue_response = await engine.execute(
            mode="blue",
            session_id=discovery_response.session_id
        )
        
        assert blue_response.success is True
        assert blue_response.step_completed == "blue"
        assert blue_response.next_step == "white"
        assert "defense" in blue_response.result


# =============================================================================
# Test: White Cell Mode
# =============================================================================

class TestWhiteCellMode:
    """Tests for White Cell mode execution."""
    
    @pytest.mark.asyncio
    async def test_white_requires_session_id(self, engine):
        """White mode should fail without session_id."""
        response = await engine.execute(mode="white")
        
        assert response.success is False
        assert "session_id" in response.error
    
    @pytest.mark.asyncio
    async def test_white_fails_without_prerequisites(self, engine, mock_client):
        """White mode should fail without red/blue results."""
        # Create session but skip red/blue
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        discovery_response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        # Try white without red/blue
        response = await engine.execute(
            mode="white",
            session_id=discovery_response.session_id
        )
        
        assert response.success is False
        assert "prerequisites" in response.error
    
    @pytest.mark.asyncio
    async def test_white_executes_synthesis(self, engine, mock_client):
        """White mode should execute synthesis after red/blue."""
        # Discovery
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        discovery_response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        # Red
        mock_client.generate_completion.return_value = "Red critique"
        await engine.execute(mode="red", session_id=discovery_response.session_id)
        
        # Blue
        mock_client.generate_completion.return_value = "Blue defense"
        await engine.execute(mode="blue", session_id=discovery_response.session_id)
        
        # White
        mock_client.generate_completion.return_value = "<thought>White reasoning</thought>White consensus"
        
        white_response = await engine.execute(
            mode="white",
            session_id=discovery_response.session_id
        )
        
        assert white_response.success is True
        assert white_response.step_completed == "white"
        assert white_response.next_step is None  # Final step
        assert "consensus" in white_response.result
        assert "report" in white_response.result
    
    @pytest.mark.asyncio
    async def test_white_handles_regeneration(self, engine, mock_client):
        """White mode should handle [REGENERATE] requests."""
        # Discovery
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        discovery_response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        # Red
        mock_client.generate_completion.return_value = "Red critique"
        await engine.execute(mode="red", session_id=discovery_response.session_id)
        
        # Blue (initial)
        mock_client.generate_completion.return_value = "Blue defense v1"
        await engine.execute(mode="blue", session_id=discovery_response.session_id)
        
        # White (requests regeneration)
        # Then Blue (regenerates)
        # Then White (final)
        mock_client.generate_completion.side_effect = [
            "[REGENERATE: Too weak] Needs improvement",  # White loop 1
            "Blue defense v2",  # Blue regen
            "Final consensus"  # White loop 2
        ]
        
        white_response = await engine.execute(
            mode="white",
            session_id=discovery_response.session_id
        )
        
        assert white_response.success is True
        assert white_response.result["loops"] == 2
        assert "Final consensus" in white_response.result["consensus"]


# =============================================================================
# Test: Full Session Flow
# =============================================================================

class TestFullSessionFlow:
    """Tests for complete sparring session flow."""
    
    @pytest.mark.asyncio
    async def test_complete_flow_discovery_to_white(self, engine, mock_client):
        """Test complete flow: discovery → red → blue → white."""
        # Setup mocks
        mock_client.generate_completion.side_effect = [
            # Discovery
            '{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}',
            # Red
            '<thought>Red thought</thought>Red critique',
            # Blue
            '<thought>Blue thought</thought>Blue defense',
            # White
            '<thought>White thought</thought>White consensus'
        ]
        
        # Execute flow
        discovery = await engine.execute(
            mode="discovery",
            topic="Test Topic",
            context="Test Context"
        )
        assert discovery.success is True
        
        red = await engine.execute(
            mode="red",
            session_id=discovery.session_id
        )
        assert red.success is True
        assert red.next_step == "blue"
        
        blue = await engine.execute(
            mode="blue",
            session_id=discovery.session_id
        )
        assert blue.success is True
        assert blue.next_step == "white"
        
        white = await engine.execute(
            mode="white",
            session_id=discovery.session_id
        )
        assert white.success is True
        assert white.next_step is None
        assert "report" in white.result
        
        # Verify session state
        session = engine.session_store.load(discovery.session_id)
        assert session.status == "completed"
        # Note: discovery mode doesn't add itself to steps_completed, only red/blue/white
        assert "red" in session.steps_completed
        assert "blue" in session.steps_completed
        assert "white" in session.steps_completed
    
    @pytest.mark.asyncio
    async def test_response_to_markdown(self, engine, mock_client):
        """Test SparringResponse.to_markdown() output."""
        mock_client.generate_completion.return_value = '''
{"red_role": "R", "red_profile": "RP", "blue_role": "B", "blue_profile": "BP", "white_role": "W", "white_profile": "WP"}
'''
        
        response = await engine.execute(
            mode="discovery",
            topic="Test",
            context=""
        )
        
        markdown = response.to_markdown()
        
        assert "✅" in markdown
        assert "Discovery completed" in markdown
        assert response.session_id in markdown
        assert "➡️" in markdown
        assert "Next step" in markdown
