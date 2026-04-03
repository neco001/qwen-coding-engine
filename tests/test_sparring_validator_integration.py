"""Tests for Sparring Engine Integration with Validator Triggers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.validator.trigger_logic import TriggerLogic, TriggerResult


@pytest.fixture
def mock_trigger_logic():
    """Create a mock TriggerLogic instance."""
    mock = MagicMock(spec=TriggerLogic)
    return mock


@pytest.fixture
def mock_sparring_engine():
    """Create a mock SparringEngineV2."""
    with patch('src.qwen_mcp.engines.sparring_v2.engine.SparringEngineV2') as mock:
        yield mock


def test_trigger_result_integration():
    """TriggerResult integrates with sparring engine response."""
    trigger_result = TriggerResult(
        should_trigger=True,
        reason="lines_changed >= 500",
        metrics={"lines_changed": 600, "files_modified": 3},
    )
    
    assert trigger_result.should_trigger is True
    assert "lines_changed" in trigger_result.reason
    assert trigger_result.metrics["lines_changed"] == 600


def test_evaluate_triggers_with_change_info():
    """TriggerLogic evaluates change info correctly."""
    logic = TriggerLogic()
    change_info = {
        "lines_changed": 600,
        "files_modified": 6,
        "dependencies_affected": 5,
        "risk_score": 0.5,
    }
    
    result = logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "lines_changed" in result.reason
    assert "files_modified" in result.reason


def test_evaluate_triggers_no_trigger():
    """TriggerLogic does not trigger when below thresholds."""
    logic = TriggerLogic()
    change_info = {
        "lines_changed": 100,
        "files_modified": 2,
        "dependencies_affected": 3,
        "risk_score": 0.3,
    }
    
    result = logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is False
    assert "below" in result.reason.lower()


def test_trigger_with_structural_change():
    """TriggerLogic triggers on structural change flag."""
    logic = TriggerLogic()
    change_info = {
        "lines_changed": 50,
        "files_modified": 1,
        "dependencies_affected": 0,
        "risk_score": 0.2,
        "structural_change_detected": True,
    }
    
    result = logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "structural" in result.reason.lower()


@pytest.mark.asyncio
async def test_validator_session_triggered_on_high_changes():
    """Validator session is triggered when changes exceed thresholds."""
    from src.validator.trigger_logic import TriggerLogic, TriggerThresholds
    
    # Custom low thresholds for testing
    thresholds = TriggerThresholds(
        lines_changed=100,
        files_modified=2,
        dependencies_affected=5,
        risk_score=0.5,
    )
    logic = TriggerLogic(thresholds=thresholds)
    
    change_info = {
        "lines_changed": 150,
        "files_modified": 3,
        "dependencies_affected": 2,
        "risk_score": 0.3,
    }
    
    result = logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True


@pytest.mark.asyncio
async def test_validator_session_not_triggered_on_low_changes():
    """Validator session is not triggered when changes are below thresholds."""
    logic = TriggerLogic()
    
    change_info = {
        "lines_changed": 50,
        "files_modified": 1,
        "dependencies_affected": 2,
        "risk_score": 0.3,
    }
    
    result = logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is False
