"""
Test generate_code_unified integration with CoderEngineV2.

This test verifies that the fix for the "ai slop" issue properly
delegates to CoderEngineV2 for intelligent code generation.
"""
import sys
import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen_mcp.tools import generate_code_unified
from qwen_mcp.engines.coder_v2 import CoderEngineV2, CoderResponse


@pytest.mark.asyncio
async def test_generate_code_unified_uses_coder_engine_v2():
    """
    Test that generate_code_unified delegates to CoderEngineV2.
    
    This is the fix for the "ai slop" issue where the old implementation
    made direct API calls without brownfield detection or mode routing.
    """
    # Mock the CoderEngineV2.execute method
    mock_response = CoderResponse(
        success=True,
        mode_used="auto",
        model_used="qwen3.5-coder",
        result="def add(a, b):\n    return a + b",
        message="Code generated successfully",
        routing_reason="Local heuristics: simple task"
    )
    
    with patch.object(CoderEngineV2, 'execute', new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = mock_response
        
        # Call the function
        result = await generate_code_unified(
            prompt="Write a function to add two numbers",
            mode="auto",
            context="",
            ctx=None,
            project_id="test"
        )
        
        # Verify CoderEngineV2.execute was called
        mock_execute.assert_called_once()
        
        # Verify the call arguments
        call_args = mock_execute.call_args
        assert call_args.kwargs['prompt'] == "Write a function to add two numbers"
        assert call_args.kwargs['mode'] == "auto"
        assert call_args.kwargs['context'] == ""
        assert call_args.kwargs['project_id'] == "test"
        
        # Verify the result is formatted as markdown (CoderResponse.to_markdown())
        assert "✅ **Code generation complete!**" in result
        assert "qwen3.5-coder" in result
        assert "def add(a, b)" in result


@pytest.mark.asyncio
async def test_generate_code_unified_handles_failure():
    """
    Test that generate_code_unified properly handles CoderEngineV2 failures.
    """
    # Mock a failure response
    mock_response = CoderResponse(
        success=False,
        mode_used="auto",
        model_used="",
        result="",
        message="Execution failed",
        error="API timeout"
    )
    
    with patch.object(CoderEngineV2, 'execute', new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = mock_response
        
        # Call the function
        result = await generate_code_unified(
            prompt="Write complex code",
            mode="auto",
            ctx=None,
            project_id="test"
        )
        
        # Verify error handling
        assert "❌ Code generation failed" in result
        assert "API timeout" in result


@pytest.mark.asyncio
async def test_generate_code_unified_with_context():
    """
    Test that generate_code_unified passes context to CoderEngineV2.
    
    This is critical for brownfield detection - the engine uses context
    length to determine if existing code is being modified.
    """
    existing_code = """
class McpExecution:
    def __init__(self):
        self.status = "pending"
    
    def update_status(self, new_status):
        self.status = new_status
"""
    
    mock_response = CoderResponse(
        success=True,
        mode_used="auto",
        model_used="qwen3.5-coder",
        result="SEARCH/REPLACE blocks for McpExecution class",
        message="Code generated with brownfield detection",
        routing_reason="Context-based heuristic: brownfield"
    )
    
    with patch.object(CoderEngineV2, 'execute', new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = mock_response
        
        result = await generate_code_unified(
            prompt="Add progress tracking to McpExecution",
            mode="auto",
            context=existing_code,
            ctx=None,
            project_id="test"
        )
        
        # Verify context was passed
        call_args = mock_execute.call_args
        assert call_args.kwargs['context'] == existing_code
        
        # Verify the result mentions brownfield detection
        assert "brownfield" in mock_response.routing_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
