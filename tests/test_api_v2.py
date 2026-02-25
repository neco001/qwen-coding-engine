import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from qwen_mcp.api import DashScopeClient


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_completion_reasoning_fallback():
    # Mocking choices to return empty content but filled reasoning_content
    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.reasoning_content = "THINKING: I should use reasoning."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    with patch("qwen_mcp.api.AsyncOpenAI") as MockAsyncOpenAI:
        mock_instance = MockAsyncOpenAI.return_value
        # Mocking the async call
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        # Set environment variables for client init
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            client = DashScopeClient()

            # Non-streaming mode (standard)
            result = await client.generate_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

            # Assertion: It should return the reasoning if content is empty
            assert result == "THINKING: I should use reasoning."
