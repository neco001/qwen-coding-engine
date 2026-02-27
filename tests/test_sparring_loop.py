import pytest
from qwen_mcp.tools import generate_sparring
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_sparring_internal_loop():
    """
    Test that generate_sparring in PRO mode triggers a loop when White Cell says [REGENERATE].
    """
    topic = "Test loop"
    context = "Context"
    
    with patch("qwen_mcp.tools.DashScopeClient") as MockClient:
        instance = MockClient.return_value
        instance.generate_completion = AsyncMock()
        
        # Mock responses: 
        # 1. Discovery
        # 2. Red Turn
        # 3. Blue Turn (Trial 1)
        # 4. White Turn (Trial 1 -> Trigger Regen)
        # 5. Blue Turn (Trial 2)
        # 6. White Turn (Trial 2 -> Final)
        instance.generate_completion.side_effect = [
            '{"red_role": "R", "red_profile": "P", "blue_role": "B", "blue_profile": "P", "white_role": "W", "white_profile": "P"}', # 1
            "Red Critique", # 2
            "Blue Defense 1", # 3
            "[REGENERATE: Too weak] Reason", # 4 (Trigger)
            "Blue Defense 2", # 5
            "White Final" # 6
        ]
        
        with patch("qwen_mcp.tools.ContentValidator") as MockValidator:
            MockValidator.validate_response.side_effect = lambda x: x
            MockValidator.sanitize_input.side_effect = lambda x: x

            result = await generate_sparring(topic, context, mode="pro")
            
            # Assert that Blue Defense 2 is in the result (meaning loop happened)
            assert "Blue Defense 2" in result
            assert "White Final" in result
            assert "optimization cycles" in result
            assert "[REGENERATE" not in result # Should be cleaned
