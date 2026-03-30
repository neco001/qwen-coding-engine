import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from qwen_mcp.orchestrator import SwarmOrchestrator, SwarmResult
from qwen_mcp.prompts.swarm import DECOMPOSE_SYSTEM_PROMPT

@pytest.mark.asyncio
async def test_swarm_orchestrator_decompose_success():
    # Mock CompletionHandler
    mock_handler = AsyncMock()
    mock_response = {
        "intent_validation": True,
        "sub_tasks": [
            {"id": "1", "task": "Subtask 1", "priority": 5},
            {"id": "2", "task": "Subtask 2", "context_keys": ["key1"]}
        ],
        "estimated_total_tokens": 100
    }
    mock_handler.generate_completion.return_value = json.dumps(mock_response)
    
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
    result = await orchestrator.decompose("Analyze this complex request")
    
    assert isinstance(result, SwarmResult)
    assert result.intent_validation is True
    assert len(result.sub_tasks) == 2
    assert result.sub_tasks[0].task == "Subtask 1"
    
    # Verify handler call
    mock_handler.generate_completion.assert_called_once()
    args, kwargs = mock_handler.generate_completion.call_args
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DECOMPOSE_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Analyze this complex request"

@pytest.mark.asyncio
async def test_swarm_orchestrator_decompose_invalid_json():
    """Invalid JSON should gracefully fallback to single-task result instead of crash."""
    mock_handler = AsyncMock()
    mock_handler.generate_completion.return_value = "Invalid JSON string"
    
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
    
    # FIX: Should return fallback SwarmResult instead of raising
    result = await orchestrator.decompose("blah")
    assert isinstance(result, SwarmResult)
    assert result.intent_validation is False  # Fallback mode
    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].task == "blah"


@pytest.mark.asyncio
async def test_swarm_orchestrator_decompose_empty_response():
    """Empty model response should gracefully fallback to single-task result."""
    mock_handler = AsyncMock()
    mock_handler.generate_completion.return_value = ""
    
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
    
    result = await orchestrator.decompose("test prompt")
    assert isinstance(result, SwarmResult)
    assert result.intent_validation is False
    assert len(result.sub_tasks) == 1
    assert result.sub_tasks[0].task == "test prompt"


@pytest.mark.asyncio
async def test_swarm_orchestrator_decompose_whitespace_only():
    """Whitespace-only response should gracefully fallback."""
    mock_handler = AsyncMock()
    mock_handler.generate_completion.return_value = "   \n\t  "
    
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
    
    result = await orchestrator.decompose("another test")
    assert isinstance(result, SwarmResult)
    assert result.intent_validation is False
    assert result.sub_tasks[0].task == "another test"
