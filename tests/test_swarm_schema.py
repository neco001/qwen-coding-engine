import pytest
from qwen_mcp.orchestrator import SubTask, SwarmResult


def test_subtask_validation():
    # Test valid SubTask creation
    valid_subtask = SubTask(id="1", task="Do something")
    assert valid_subtask.id == "1"
    assert valid_subtask.task == "Do something"
    assert valid_subtask.priority == 1
    assert valid_subtask.context_keys == []

    # Test invalid priority (negative)
    with pytest.raises(ValueError):
        SubTask(id="1", task="Do something", priority=-1)

    # Test missing id
    with pytest.raises(ValueError):
        SubTask(task="Do something")

    # Test missing task
    with pytest.raises(ValueError):
        SubTask(id="1")


def test_swarm_result_parsing():
    # Valid dictionary to parse into SwarmResult
    valid_data = {
        "intent_validation": True,
        "sub_tasks": [
            {"id": "1", "task": "First task", "priority": 2},
            {"id": "2", "task": "Second task", "context_keys": ["key1", "key2"]}
        ],
        "estimated_total_tokens": 150
    }

    result = SwarmResult(**valid_data)

    assert result.intent_validation is True
    assert len(result.sub_tasks) == 2
    assert result.estimated_total_tokens == 150

    # Check first subtask
    assert result.sub_tasks[0].id == "1"
    assert result.sub_tasks[0].task == "First task"
    assert result.sub_tasks[0].priority == 2
    assert result.sub_tasks[0].context_keys == []

    # Check second subtask
    assert result.sub_tasks[1].id == "2"
    assert result.sub_tasks[1].task == "Second task"
    assert result.sub_tasks[1].priority == 1  # Default value
    assert result.sub_tasks[1].context_keys == ["key1", "key2"]
