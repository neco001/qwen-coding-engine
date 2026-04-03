"""End-to-End Isolation Tests for AI-Driven Testing System.

These tests verify that the complete isolation system works correctly:
1. Session creation with role-specific prompts
2. Context validation and forbidden context rejection
3. Isolation breach detection
4. Integration with validator triggers
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.session.isolation_manager import SessionIsolationManager, IsolationViolationError
from src.session.prompts import Role, ROLE_CONFIG
from src.validator.trigger_logic import TriggerLogic, TriggerThresholds
from src.validator.integration import ValidatorIntegration


@pytest.fixture
def isolation_manager():
    """Create a SessionIsolationManager instance."""
    return SessionIsolationManager()


@pytest.fixture
def validator_integration():
    """Create a ValidatorIntegration instance with mocked dependencies."""
    mock_trigger = MagicMock(spec=TriggerLogic)
    mock_trigger.thresholds = TriggerThresholds()
    mock_trigger.evaluate_triggers = MagicMock()
    
    mock_metrics = MagicMock()
    mock_metrics.collect_metrics = AsyncMock()
    
    mock_isolation = MagicMock(spec=SessionIsolationManager)
    mock_isolation.create_session = MagicMock()
    
    return ValidatorIntegration(
        trigger_logic=mock_trigger,
        metrics_collector=mock_metrics,
        isolation_manager=mock_isolation,
    )


@pytest.mark.asyncio
async def test_e2e_coder_session_isolation(isolation_manager):
    """E2E: Coder session maintains isolation from test context."""
    # Create coder session
    checkpoint = isolation_manager.create_session(
        role="coder",
        topic="Implement user authentication",
        context="Need to add login and logout endpoints",
    )
    
    assert checkpoint.session_id is not None
    
    # Validate isolation
    is_valid = isolation_manager.validate_isolation(checkpoint.session_id)
    assert is_valid is True
    
    # Verify role is stored in isolation manager
    sessions = isolation_manager.list_active_sessions()
    coder_sessions = [s for s in sessions if s.get("role") == "coder"]
    assert len(coder_sessions) >= 1


@pytest.mark.asyncio
async def test_e2e_test_session_isolation(isolation_manager):
    """E2E: Test session maintains isolation from coder context."""
    # Create test session
    checkpoint = isolation_manager.create_session(
        role="test",
        topic="Write tests for authentication",
        context="Need to cover login, logout, and token refresh",
    )
    
    assert checkpoint.session_id is not None
    
    # Validate isolation
    is_valid = isolation_manager.validate_isolation(checkpoint.session_id)
    assert is_valid is True
    
    # Verify role is stored in isolation manager
    sessions = isolation_manager.list_active_sessions()
    test_sessions = [s for s in sessions if s.get("role") == "test"]
    assert len(test_sessions) >= 1


@pytest.mark.asyncio
async def test_e2e_validator_session_isolation(isolation_manager):
    """E2E: Validator session maintains isolation from coder and test contexts."""
    # Create validator session
    checkpoint = isolation_manager.create_session(
        role="validator",
        topic="Validate authentication changes",
        context="Review code for security issues and regressions",
    )
    
    assert checkpoint.session_id is not None
    
    # Validate isolation
    is_valid = isolation_manager.validate_isolation(checkpoint.session_id)
    assert is_valid is True
    
    # Verify role is stored in isolation manager
    sessions = isolation_manager.list_active_sessions()
    validator_sessions = [s for s in sessions if s.get("role") == "validator"]
    assert len(validator_sessions) >= 1


@pytest.mark.asyncio
async def test_e2e_coder_rejected_test_context(isolation_manager):
    """E2E: Coder session rejects context containing test information."""
    # Attempt to create coder session with test context
    with pytest.raises(IsolationViolationError) as exc_info:
        isolation_manager.create_session(
            role="coder",
            topic="Implement feature",
            context="The test expects this function to return True",
        )
    
    assert "test" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_e2e_test_rejected_coder_context(isolation_manager):
    """E2E: Test session rejects context containing coder implementation."""
    # Attempt to create test session with coder context
    with pytest.raises(IsolationViolationError) as exc_info:
        isolation_manager.create_session(
            role="test",
            topic="Write tests",
            context="The coder implementation uses a private method _validate",
        )
    
    assert "coder" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_e2e_validator_triggered_on_high_risk(validator_integration):
    """E2E: Validator session is triggered when risk exceeds threshold."""
    # Mock trigger evaluation to return trigger
    validator_integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=True,
        reason="lines_changed >= 500",
        metrics={"lines_changed": 600}
    )
    
    validator_integration.isolation_manager.create_session.return_value = MagicMock(
        session_id="val_e2e_123"
    )
    
    # Simulate high-risk change
    change_info = {
        "lines_changed": 600,
        "files_modified": 5,
        "dependencies_affected": 10,
        "risk_score": 0.8,
    }
    
    triggered, session_id, result = await validator_integration.evaluate_and_trigger_validator(
        Path("/tmp/test_project"),
        change_info=change_info
    )
    
    assert triggered is True
    assert session_id == "val_e2e_123"
    assert result.should_trigger is True


@pytest.mark.asyncio
async def test_e2e_validator_not_triggered_on_low_risk(validator_integration):
    """E2E: Validator session is not triggered when risk is below threshold."""
    # Mock trigger evaluation to not trigger
    validator_integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=False,
        reason="All metrics below thresholds",
        metrics={}
    )
    
    # Simulate low-risk change
    change_info = {
        "lines_changed": 50,
        "files_modified": 1,
        "dependencies_affected": 2,
        "risk_score": 0.2,
    }
    
    triggered, session_id, result = await validator_integration.evaluate_and_trigger_validator(
        Path("/tmp/test_project"),
        change_info=change_info
    )
    
    assert triggered is False
    assert session_id is None
    assert result.should_trigger is False


@pytest.mark.asyncio
async def test_e2e_multiple_isolated_sessions(isolation_manager):
    """E2E: Multiple sessions can coexist while maintaining isolation."""
    # Create sessions for all roles
    coder_session = isolation_manager.create_session(
        role="coder",
        topic="Feature implementation",
        context="Add new API endpoint",
    )
    
    test_session = isolation_manager.create_session(
        role="test",
        topic="Feature testing",
        context="Write integration tests",
    )
    
    validator_session = isolation_manager.create_session(
        role="validator",
        topic="Feature validation",
        context="Review for regressions",
    )
    
    # All sessions should have unique IDs
    session_ids = [coder_session.session_id, test_session.session_id, validator_session.session_id]
    assert len(set(session_ids)) == 3
    
    # All sessions should maintain isolation
    for session_id in session_ids:
        is_valid = isolation_manager.validate_isolation(session_id)
        assert is_valid is True
    
    # List all sessions
    sessions = isolation_manager.list_active_sessions()
    assert len(sessions) >= 3
