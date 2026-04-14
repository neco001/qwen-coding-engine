import pytest
from qwen_mcp.engines.models.task import Task


class TestTask:
    def test_task_initialization(self):
        task = Task(decision_id="dec_123", description="Test task", state="pending")
        assert task.decision_id == "dec_123"
        assert task.description == "Test task"
        assert task.state == "pending"

    def test_is_complete_returns_false_when_pending(self):
        task = Task(decision_id="dec_123", description="Test task", state="pending")
        assert task.is_complete() is False

    def test_mark_complete_changes_state(self):
        task = Task(decision_id="dec_123", description="Test task", state="pending")
        task.mark_complete()
        assert task.state == "completed"

    def test_is_complete_returns_true_after_mark_complete(self):
        task = Task(decision_id="dec_123", description="Test task", state="pending")
        task.mark_complete()
        assert task.is_complete() is True

    def test_task_initialized_as_completed(self):
        task = Task(decision_id="dec_456", description="Already done", state="completed")
        assert task.is_complete() is True
