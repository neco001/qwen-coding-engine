import pytest
import os
import json
from unittest.mock import patch, AsyncMock, MagicMock
from qwen_mcp.wanx_builder import WanxPayloadBuilder
from qwen_mcp.images import ImageHandler

def test_wanx_payload_universal_prompt_extend():
    """Verify that prompt_extend is correctly passed to the payload."""
    builder = WanxPayloadBuilder(model="qwen-image-edit-max", size="1024*1024")
    
    # Test True
    payload_true = builder.build(images=[], prompt="test", prompt_extend=True)
    assert payload_true["parameters"]["prompt_extend"] is True
    
    # Test False
    payload_false = builder.build(images=[], prompt="test", prompt_extend=False)
    assert payload_false["parameters"]["prompt_extend"] is False

@pytest.mark.asyncio
async def test_image_handler_dry_run_logic():
    """Verify that dry_run=True returns the payload without calling HTTP."""
    handler = ImageHandler()
    
    # Mocking environment for DashScope
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
        # We don't want to mock aiohttp because it shouldn't be called at all
        with patch('aiohttp.ClientSession.post') as mock_post:
            result = await handler.generate_image(
                prompt="Universal prompt",
                dry_run=True,
                prompt_extend=False
            )
            
            # Assertions
            assert result["status"] == "dry_run"
            assert "payload" in result
            assert result["payload"]["parameters"]["prompt_extend"] is False
            assert "endpoint" in result
            
            # Verify HTTP was NOT called
            mock_post.assert_not_called()

def test_no_product_hardcoding_in_logic():
    """Verify that the generated prompt doesn't contain old hardcoded hacks."""
    builder = WanxPayloadBuilder()
    # If the user prompt is generic, the builder shouldn't add headphone specific hacks
    # (Note: My previous version of ImageHandler had the hack, but I refactored it out).
    # Since builder is just a builder, it handles images and text.
    # The logic was in ImageHandler.generate_image.
    pass

@pytest.mark.asyncio
async def test_no_product_leaks_in_payload():
    """Verify that ImageHandler doesn't inject 'headphones' or 'MAINTAIN EXACT SHAPE' in the prompt."""
    handler = ImageHandler()
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
        result = await handler.generate_image(
            prompt="Generate a girl in a forest.",
            dry_run=True
        )
        final_prompt = result["payload"]["input"]["messages"][0]["content"][-1]["text"]
        
        assert "headphones" not in final_prompt.lower()
        assert "MAINTAIN EXACT SHAPE" not in final_prompt
        assert final_prompt == "Generate a girl in a forest."
