import pytest
from qwen_mcp.tools import generate_sparring
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_sparring_dynamic_role_discovery():
    """
    Test that generate_sparring in PRO mode performs role discovery.
    We check if the internal flow calls generate_completion for a classifier/discovery step.
    """
    topic = "Post o ankiecie LinkedIn dla C-Level"
    context = "Zasięgi, autorytet, LinkedIn, behawioryzm"
    
    with patch("qwen_mcp.tools.DashScopeClient") as MockClient:
        instance = MockClient.return_value
        instance.generate_completion = AsyncMock()
        
        # Mock responses: 1. Discovery, 2. Red, 3. Blue, 4. White
        instance.generate_completion.side_effect = [
            '{"red_role": "Sceptyczny Behawiorysta", "red_profile": "Cynik", "blue_role": "Strateg", "blue_profile": "Adwokat", "white_role": "Arbiter", "white_profile": "Sędzia"}', # Discovery
            "Red Critique", 
            "Blue Defense", 
            "White Consensus"
        ]
        
        with patch("qwen_mcp.tools.ContentValidator") as MockValidator:
            MockValidator.validate_response.side_effect = lambda x: x
            MockValidator.sanitize_input.side_effect = lambda x: x

            result = await generate_sparring(topic, context, mode="pro")
            
            # Assert that Discovery happened
            assert "Selected Roles:" in result
            assert "Sceptyczny Behawiorysta" in result
            assert "Strateg" in result
            assert "Turn 2: Sceptyczny Behawiorysta" in result
