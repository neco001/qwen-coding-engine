"""
Tests for SessionMapper - simplified session naming with "Sesja 1", "Sesja 2" etc.
"""
import pytest
import asyncio
from qwen_mcp.specter.telemetry import SessionMapper


class TestSessionMapper:
    """Test suite for SessionMapper class."""

    @pytest.mark.asyncio
    async def test_first_project_gets_session_1(self):
        """First unique project_id should get session number 1."""
        mapper = SessionMapper()
        result = await mapper.get_or_create_session_number("project_abc123")
        assert result == 1

    @pytest.mark.asyncio
    async def test_second_project_gets_session_2(self):
        """Second unique project_id should get session number 2."""
        mapper = SessionMapper()
        await mapper.get_or_create_session_number("project_abc123")
        result = await mapper.get_or_create_session_number("project_xyz789")
        assert result == 2

    @pytest.mark.asyncio
    async def test_same_project_returns_same_number(self):
        """Same project_id should always return the same session number."""
        mapper = SessionMapper()
        first = await mapper.get_or_create_session_number("project_abc123")
        second = await mapper.get_or_create_session_number("project_abc123")
        assert first == second == 1

    @pytest.mark.asyncio
    async def test_session_count_increments(self):
        """Session count should match number of unique projects."""
        mapper = SessionMapper()
        await mapper.get_or_create_session_number("project_1")
        await mapper.get_or_create_session_number("project_2")
        await mapper.get_or_create_session_number("project_3")
        count = await mapper.get_session_count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_thread_safety(self):
        """Concurrent calls should not produce duplicate session numbers."""
        mapper = SessionMapper()

        async def assign_session(project_id: str):
            # Simulate concurrent access
            await asyncio.sleep(0.001)
            return await mapper.get_or_create_session_number(project_id)

        tasks = [
            assign_session(f"project_{i}")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        # All results should be unique (no duplicates)
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique
        assert set(results) == set(range(1, 11))  # 1 through 10

    @pytest.mark.asyncio
    async def test_get_display_name(self):
        """Should return formatted display name like 'Sesja 1'."""
        mapper = SessionMapper()
        session_num = await mapper.get_or_create_session_number("project_abc")
        display_name = mapper.get_display_name(session_num)
        assert display_name == "Sesja 1"

    @pytest.mark.asyncio
    async def test_get_display_name_polish(self):
        """Display name should use Polish 'Sesja' prefix."""
        mapper = SessionMapper()
        await mapper.get_or_create_session_number("project_1")
        await mapper.get_or_create_session_number("project_2")
        assert mapper.get_display_name(1) == "Sesja 1"
        assert mapper.get_display_name(2) == "Sesja 2"

    @pytest.mark.asyncio
    async def test_remove_session(self):
        """Should be able to remove a session when client disconnects."""
        mapper = SessionMapper()
        await mapper.get_or_create_session_number("project_1")
        await mapper.get_or_create_session_number("project_2")
        
        # Remove first session
        await mapper.remove_session("project_1")
        
        # Count should decrease
        count = await mapper.get_session_count()
        assert count == 1
        
        # New project should still get next number (3, not reuse 1)
        result = await mapper.get_or_create_session_number("project_3")
        assert result == 3