import pytest
from unittest.mock import MagicMock, patch

from qwen_mcp.engines.orchestrator import DecisionLogOrchestrator


@pytest.fixture
def mock_path_resolver():
    """Create a mock PathResolver with backlog_path and changelog_path."""
    resolver = MagicMock()
    resolver.backlog_path = "/path/to/backlog.md"
    resolver.changelog_path = "/path/to/changelog.md"
    return resolver


@pytest.fixture
def orchestrator(mock_path_resolver):
    """Create a DecisionLogOrchestrator instance with mocked PathResolver."""
    return DecisionLogOrchestrator(path_resolver=mock_path_resolver)


@patch('qwen_mcp.engines.orchestrator.SectionManager')
@patch('qwen_mcp.engines.orchestrator.FileHandler')
class TestDecisionLogOrchestrator:

    def test_archive_task_success(
        self,
        mock_file_handler,
        mock_section_manager_class,
        orchestrator,
        mock_path_resolver
    ):
        mock_file_handler.read_text.return_value = "# Backlog\n- Task 1"
        mock_section_manager_instance = MagicMock()
        mock_section_manager_class.return_value = mock_section_manager_instance
        mock_section_manager_instance.archive_task.return_value = "# Backlog\n- Task 1 [archived]"
        
        decision_id = "DEC-001"
        result = orchestrator.archive_task(decision_id)
        
        assert result is True
        mock_file_handler.read_text.assert_called_once_with(mock_path_resolver.backlog_path)
        mock_section_manager_instance.archive_task.assert_called_once_with("# Backlog\n- Task 1", decision_id)
        mock_file_handler.write_atomic.assert_called_once_with(
            mock_path_resolver.backlog_path,
            "# Backlog\n- Task 1 [archived]"
        )

    def test_archive_task_failure(
        self,
        mock_file_handler,
        mock_section_manager_class,
        orchestrator,
        mock_path_resolver
    ):
        mock_file_handler.read_text.return_value = "# Backlog\n- Task 1"
        mock_section_manager_instance = MagicMock()
        mock_section_manager_class.return_value = mock_section_manager_instance
        mock_section_manager_instance.archive_task.side_effect = ValueError("Not found")
        
        decision_id = "DEC-999"
        result = orchestrator.archive_task(decision_id)
        
        assert result is False
        mock_file_handler.read_text.assert_called_once_with(mock_path_resolver.backlog_path)
        mock_section_manager_instance.archive_task.assert_called_once_with("# Backlog\n- Task 1", decision_id)
        mock_file_handler.write_atomic.assert_not_called()

    def test_archive_task_file_read_error(
        self,
        mock_file_handler,
        mock_section_manager_class,
        orchestrator,
        mock_path_resolver
    ):
        mock_file_handler.read_text.return_value = None
        decision_id = "DEC-001"
        result = orchestrator.archive_task(decision_id)
        
        assert result is False
        mock_file_handler.read_text.assert_called_once_with(mock_path_resolver.backlog_path)
        mock_file_handler.write_atomic.assert_not_called()

    def test_append_to_changelog_success(
        self,
        mock_file_handler,
        mock_section_manager_class,
        orchestrator,
        mock_path_resolver
    ):
        mock_file_handler.read_text.return_value = "# Changelog\n"
        mock_section_manager_instance = MagicMock()
        mock_section_manager_class.return_value = mock_section_manager_instance
        mock_section_manager_instance.append_to_changelog.return_value = "# Changelog\n- New entry"
        
        entry = "Added new feature"
        orchestrator.append_to_changelog(entry)
        
        mock_file_handler.read_text.assert_called_once_with(mock_path_resolver.changelog_path)
        mock_section_manager_instance.append_to_changelog.assert_called_once_with("# Changelog\n", entry)
        mock_file_handler.write_atomic.assert_called_once_with(
            mock_path_resolver.changelog_path,
            "# Changelog\n- New entry"
        )
