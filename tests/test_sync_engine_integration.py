import pytest
from unittest.mock import patch, MagicMock

from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine


class TestDecisionLogSyncEngineIntegration:
    @patch('qwen_mcp.engines.decision_log_sync.DecisionLogOrchestrator')
    def test_mark_task_completed_calls_orchestrator_archive(self, mock_orchestrator_class):
        """
        RED phase test: Verifies that _mark_task_completed delegates to
        DecisionLogOrchestrator.archive_task instead of naive string replacement.
        """
        # Arrange
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator_instance
        
        engine = DecisionLogSyncEngine()
        decision_id = "some-id"

        # Act
        engine._mark_task_completed(decision_id)

        # Assert
        mock_orchestrator_class.assert_called_once()
        mock_orchestrator_instance.archive_task.assert_called_once_with(decision_id)
