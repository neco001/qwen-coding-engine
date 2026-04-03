"""Tests for Trigger Logic."""

import pytest
from src.validator.trigger_logic import TriggerLogic, TriggerResult, TriggerThresholds


@pytest.fixture
def trigger_logic():
    """Create a TriggerLogic instance with default thresholds."""
    return TriggerLogic()


@pytest.fixture
def custom_thresholds():
    """Create custom trigger thresholds."""
    return TriggerThresholds(
        lines_changed=100,
        files_modified=3,
        dependencies_affected=5,
        risk_score=0.5,
    )


@pytest.fixture
def trigger_with_custom_thresholds(custom_thresholds):
    """Create a TriggerLogic instance with custom thresholds."""
    return TriggerLogic(thresholds=custom_thresholds)


def test_trigger_result_has_required_fields():
    """TriggerResult has all required fields."""
    result = TriggerResult(
        should_trigger=True,
        reason="lines_changed >= 500",
        metrics={"lines_changed": 600},
    )
    
    assert hasattr(result, "should_trigger")
    assert hasattr(result, "reason")
    assert hasattr(result, "metrics")


def test_default_thresholds():
    """Default thresholds match specification."""
    logic = TriggerLogic()
    
    assert logic.thresholds.lines_changed == 500
    assert logic.thresholds.files_modified == 5
    assert logic.thresholds.dependencies_affected == 10
    assert logic.thresholds.risk_score == 0.7


def test_evaluate_triggers_returns_false_below_thresholds(trigger_logic):
    """evaluate_triggers returns False when below all thresholds."""
    change_info = {
        "lines_changed": 100,
        "files_modified": 2,
        "dependencies_affected": 3,
        "risk_score": 0.3,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is False
    assert "below" in result.reason.lower()


def test_evaluate_triggers_lines_threshold(trigger_logic):
    """evaluate_triggers triggers when lines_changed >= 500."""
    change_info = {
        "lines_changed": 500,
        "files_modified": 2,
        "dependencies_affected": 3,
        "risk_score": 0.3,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "lines" in result.reason.lower()


def test_evaluate_triggers_files_threshold(trigger_logic):
    """evaluate_triggers triggers when files_modified >= 5."""
    change_info = {
        "lines_changed": 100,
        "files_modified": 5,
        "dependencies_affected": 3,
        "risk_score": 0.3,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "files" in result.reason.lower()


def test_evaluate_triggers_dependencies_threshold(trigger_logic):
    """evaluate_triggers triggers when dependencies_affected >= 10."""
    change_info = {
        "lines_changed": 100,
        "files_modified": 2,
        "dependencies_affected": 10,
        "risk_score": 0.3,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "dependencies" in result.reason.lower()


def test_evaluate_triggers_risk_score_threshold(trigger_logic):
    """evaluate_triggers triggers when risk_score >= 0.7."""
    change_info = {
        "lines_changed": 100,
        "files_modified": 2,
        "dependencies_affected": 3,
        "risk_score": 0.7,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "risk" in result.reason.lower()


def test_evaluate_triggers_multiple_thresholds(trigger_logic):
    """evaluate_triggers reports all triggered thresholds."""
    change_info = {
        "lines_changed": 600,
        "files_modified": 6,
        "dependencies_affected": 3,
        "risk_score": 0.3,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "lines" in result.reason.lower()
    assert "files" in result.reason.lower()


def test_evaluate_triggers_custom_thresholds(trigger_with_custom_thresholds):
    """evaluate_triggers uses custom thresholds when provided."""
    change_info = {
        "lines_changed": 100,
        "files_modified": 3,
        "dependencies_affected": 3,
        "risk_score": 0.3,
    }
    
    result = trigger_with_custom_thresholds.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "lines" in result.reason.lower()


def test_evaluate_triggers_with_structural_change(trigger_logic):
    """evaluate_triggers triggers on structural_change_detected flag."""
    change_info = {
        "lines_changed": 50,
        "files_modified": 1,
        "dependencies_affected": 0,
        "risk_score": 0.2,
        "structural_change_detected": True,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True
    assert "structural" in result.reason.lower()


def test_evaluate_triggers_exactly_at_threshold(trigger_logic):
    """evaluate_triggers triggers when exactly at threshold."""
    change_info = {
        "lines_changed": 500,
        "files_modified": 5,
        "dependencies_affected": 10,
        "risk_score": 0.7,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is True


def test_evaluate_triggers_just_below_threshold(trigger_logic):
    """evaluate_triggers does not trigger when just below threshold."""
    change_info = {
        "lines_changed": 499,
        "files_modified": 4,
        "dependencies_affected": 9,
        "risk_score": 0.69,
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    assert result.should_trigger is False
