"""Tests for Validator Integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from src.validator.integration import ValidatorIntegration
from src.validator.trigger_logic import TriggerLogic, TriggerThresholds
from src.validator.metrics_collector import MetricsCollector, ChangeMetrics


@pytest.fixture
def integration():
    """Create a ValidatorIntegration instance with mocked dependencies."""
    mock_trigger = MagicMock(spec=TriggerLogic)
    mock_trigger.thresholds = TriggerThresholds()
    mock_trigger.evaluate_triggers.return_value = MagicMock(
        should_trigger=False,
        reason="All metrics below thresholds",
        metrics={}
    )
    
    mock_metrics = MagicMock(spec=MetricsCollector)
    mock_metrics.collect_metrics = AsyncMock(
        return_value=ChangeMetrics(
            total_files=0,
            total_lines=0,
            total_functions=0,
            total_classes=0,
            total_imports=0,
        )
    )
    
    mock_isolation = MagicMock()
    mock_isolation.create_session.return_value = MagicMock(session_id="mock_session")
    
    return ValidatorIntegration(
        trigger_logic=mock_trigger,
        metrics_collector=mock_metrics,
        isolation_manager=mock_isolation,
    )


@pytest.fixture
def sample_project_dir(tmp_path):
    """Create a sample project directory."""
    main_file = tmp_path / "main.py"
    main_file.write_text("def func(): pass")
    return tmp_path


@pytest.mark.asyncio
async def test_evaluate_and_trigger_validator_no_trigger(integration, sample_project_dir):
    """evaluate_and_trigger_validator returns False when triggers not met."""
    integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=False,
        reason="All metrics below thresholds",
        metrics={}
    )
    
    triggered, session_id, result = await integration.evaluate_and_trigger_validator(
        sample_project_dir
    )
    
    assert triggered is False
    assert session_id is None
    assert result.should_trigger is False


@pytest.mark.asyncio
async def test_evaluate_and_trigger_validator_triggers(integration, sample_project_dir):
    """evaluate_and_trigger_validator creates session when triggers met."""
    integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=True,
        reason="lines_changed >= 500",
        metrics={"lines_changed": 600}
    )
    
    integration.isolation_manager.create_session.return_value = MagicMock(
        session_id="val_123"
    )
    
    triggered, session_id, result = await integration.evaluate_and_trigger_validator(
        sample_project_dir
    )
    
    assert triggered is True
    assert session_id == "val_123"
    assert result.should_trigger is True


@pytest.mark.asyncio
async def test_evaluate_and_trigger_validator_with_change_info(integration, sample_project_dir):
    """evaluate_and_trigger_validator uses provided change_info."""
    change_info = {
        "lines_changed": 600,
        "files_modified": 5,
        "dependencies_affected": 10,
        "risk_score": 0.8,
    }
    
    integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=True,
        reason="lines_changed >= 500",
        metrics=change_info
    )
    
    integration.isolation_manager.create_session.return_value = MagicMock(
        session_id="val_456"
    )
    
    triggered, session_id, result = await integration.evaluate_and_trigger_validator(
        sample_project_dir,
        change_info=change_info
    )
    
    assert triggered is True
    assert session_id == "val_456"
    integration.trigger_logic.evaluate_triggers.assert_called_once_with(change_info)


@pytest.mark.asyncio
async def test_evaluate_and_trigger_validator_collects_metrics(integration, sample_project_dir):
    """evaluate_and_trigger_validator collects metrics when change_info not provided."""
    integration.metrics_collector.collect_metrics = AsyncMock(
        return_value=ChangeMetrics(
            total_files=5,
            total_lines=600,
            total_functions=20,
            total_classes=5,
            total_imports=10,
        )
    )
    
    integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=False,
        reason="All metrics below thresholds",
        metrics={}
    )
    
    await integration.evaluate_and_trigger_validator(sample_project_dir)
    
    integration.metrics_collector.collect_metrics.assert_called_once_with(sample_project_dir)


@pytest.mark.asyncio
async def test_evaluate_and_trigger_validator_session_creation_error(integration, sample_project_dir):
    """evaluate_and_trigger_validator handles session creation errors."""
    integration.trigger_logic.evaluate_triggers.return_value = MagicMock(
        should_trigger=True,
        reason="lines_changed >= 500",
        metrics={}
    )
    
    integration.isolation_manager.create_session.side_effect = Exception("Session creation failed")
    
    triggered, session_id, result = await integration.evaluate_and_trigger_validator(
        sample_project_dir
    )
    
    assert triggered is False
    assert session_id is None


def test_get_trigger_thresholds(integration):
    """get_trigger_thresholds returns current thresholds."""
    thresholds = integration.get_trigger_thresholds()
    
    assert isinstance(thresholds, TriggerThresholds)
    assert thresholds.lines_changed == 500
    assert thresholds.files_modified == 5


def test_update_thresholds(integration):
    """update_thresholds updates trigger thresholds."""
    new_thresholds = TriggerThresholds(
        lines_changed=100,
        files_modified=3,
        dependencies_affected=5,
        risk_score=0.5,
    )
    
    integration.update_thresholds(new_thresholds)
    
    assert integration.trigger_logic.thresholds.lines_changed == 100
    assert integration.trigger_logic.thresholds.files_modified == 3


def test_build_validator_context(integration):
    """_build_validator_context creates proper context string."""
    change_info = {
        "lines_changed": 600,
        "files_modified": 5,
        "dependencies_affected": 10,
        "risk_score": 0.75,
        "structural_change_detected": True,
    }
    
    context = integration._build_validator_context(change_info)
    
    assert "Lines changed: 600" in context
    assert "Files modified: 5" in context
    assert "Dependencies affected: 10" in context
    assert "Risk score: 0.75" in context
    assert "Structural change detected: True" in context
    assert "Validation Tasks" in context
