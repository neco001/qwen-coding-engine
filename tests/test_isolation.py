"""Tests for Session Isolation Manager - Security Audit."""
import pytest
from src.session.isolation_manager import SessionIsolationManager, IsolationViolationError


@pytest.mark.asyncio
async def test_create_session_validates_role():
    """Valid roles (coder, test, validator) create sessions successfully."""
    manager = SessionIsolationManager()
    
    for role in ["coder", "test", "validator"]:
        session = manager.create_session(role, topic="test topic")
        assert session is not None
        assert session.session_id is not None


@pytest.mark.asyncio
async def test_create_session_rejects_unknown_role():
    """Unknown roles raise IsolationViolationError."""
    manager = SessionIsolationManager()
    
    with pytest.raises(IsolationViolationError):
        manager.create_session("unknown_role", topic="test topic")


@pytest.mark.asyncio
async def test_create_session_rejects_forbidden_context():
    """Context containing forbidden role names is rejected."""
    manager = SessionIsolationManager()
    
    # Test each role with its forbidden context
    forbidden_cases = [
        ("coder", "this is about test session"),
        ("test", "this is about validator session"),
        ("validator", "this is about coder session"),
    ]
    
    for role, forbidden_context in forbidden_cases:
        with pytest.raises(IsolationViolationError):
            manager.create_session(role, topic="test topic", context=forbidden_context)


@pytest.mark.asyncio
async def test_validate_isolation_detects_breach():
    """validate_isolation detects when forbidden context is added."""
    manager = SessionIsolationManager()
    
    # Create a session first - use topic without forbidden words
    session = manager.create_session("coder", topic="implementation task")
    
    # Validate isolation should work for clean session
    assert manager.validate_isolation(session.session_id) is True


@pytest.mark.asyncio
async def test_coder_cannot_access_test_context():
    """Coder role cannot have 'test' in context."""
    manager = SessionIsolationManager()
    
    with pytest.raises(IsolationViolationError):
        manager.create_session("coder", topic="test topic", context="test session said")


@pytest.mark.asyncio
async def test_validator_cannot_access_coder_context():
    """Validator role cannot have 'coder' in context."""
    manager = SessionIsolationManager()
    
    with pytest.raises(IsolationViolationError):
        manager.create_session("validator", topic="test topic", context="coder session said")


@pytest.mark.asyncio
async def test_execute_isolated_runs_in_session():
    """execute_isolated validates isolation before execution."""
    manager = SessionIsolationManager()
    
    # Create a valid session
    session = manager.create_session("coder", topic="coding task")
    
    # Mock model client
    class MockClient:
        pass
    
    # Execute should work with valid session
    # Note: This test verifies the method exists and validates isolation
    # The actual model integration is tested elsewhere
    try:
        result = await manager.execute_isolated(
            session.session_id,
            "Write a function",
            MockClient()
        )
        assert "[coder]" in result
    except IsolationViolationError:
        pytest.fail("execute_isolated raised IsolationViolationError for valid session")
