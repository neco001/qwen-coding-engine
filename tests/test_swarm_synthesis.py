import pytest
from unittest.mock import AsyncMock
from qwen_mcp.orchestrator import SwarmOrchestrator
from qwen_mcp.prompts.swarm import SYNTHESIZE_SYSTEM_PROMPT

@pytest.mark.asyncio
async def test_swarm_orchestrator_synthesize():
    mock_handler = AsyncMock()
    mock_handler.generate_completion.return_value = "This is the final synthesized answer."
    
    orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
    
    original_prompt = "Compare X and Y"
    results = {
        "1": "X is fast.",
        "2": "Y is cheap."
    }
    
    final_output = await orchestrator.synthesize(original_prompt, results)
    
    assert final_output == "This is the final synthesized answer."
    
    # Verify the synthesis call
    mock_handler.generate_completion.assert_called_once()
    args, kwargs = mock_handler.generate_completion.call_args
    messages = args[0]
    
    # Check if results are in the prompt sent to the model
    user_message = messages[1]["content"]
    assert messages[0]["content"] == SYNTHESIZE_SYSTEM_PROMPT
    assert original_prompt in user_message
    assert "X is fast." in user_message
    assert "Y is cheap." in user_message
