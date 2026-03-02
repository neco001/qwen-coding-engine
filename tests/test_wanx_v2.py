import asyncio
import json
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from qwen_mcp.wanx_builder import WanxPayloadBuilder
from qwen_mcp.wanx_client import WanxClient


@pytest.fixture
def mock_aiohttp_session():
    with patch("aiohttp.ClientSession") as mock_session_class:
        # ClientSession itself is handled as an async context manager in some codes, 
        # but here we use it as a normal class that's instantiated.
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # We need __aexit__ for the session itself if it's used in 'async with'
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        def setup_response(method, json_data=None, status=200, read_data=None):
            mock_resp = MagicMock() # The response object
            mock_resp.status = status
            if json_data:
                mock_resp.json = AsyncMock(return_value=json_data)
            if read_data:
                mock_resp.read = AsyncMock(return_value=read_data)
            
            # The context manager returned by post/get
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_cm.__aexit__ = AsyncMock()
            
            getattr(mock_session, method).return_value = mock_cm
            return mock_resp

        mock_session.setup_response = setup_response
        yield mock_session


@pytest.mark.asyncio
async def test_wanx_payload_builder():
    builder = WanxPayloadBuilder()
    
    # Test basic payload construction (using fluent interface)
    payload = builder.set_prompt("A beautiful landscape").build()
    
    assert "input" in payload
    assert "messages" in payload["input"]
    content = payload["input"]["messages"][0]["content"]
    assert any(c.get("text") == "A beautiful landscape" for c in content)
    
    # Test additional parameters
    builder2 = WanxPayloadBuilder()
    payload_with_params = (
        builder2.set_prompt("A cat sitting on a chair")
        .set_size("1280*720")
        .set_n(2)
        .build()
    )
    
    assert payload_with_params["parameters"]["n"] == 2
    assert payload_with_params["parameters"]["size"] == "1280*720"


@pytest.mark.asyncio
async def test_wanx_client_generate_image(mock_aiohttp_session):
    client = WanxClient(api_key="test-key")
    
    # Mock the POST response for image generation
    mock_aiohttp_session.setup_response("post", json_data={"output": {"task_id": "test-task-123"}})
    
    # Build payload using the builder
    builder = WanxPayloadBuilder()
    payload = builder.set_prompt("A futuristic city").build()
    
    # Call generate_image
    task_id = await client.generate_task(payload)
    
    # Verify the call was made correctly
    mock_aiohttp_session.post.assert_called_once()
    args, kwargs = mock_aiohttp_session.post.call_args
    assert "multimodal-generation" in str(args[0])
    
    # Verify returned task_id
    assert task_id == "test-task-123"


@pytest.mark.asyncio
async def test_wanx_client_poll_task_success(mock_aiohttp_session):
    client = WanxClient(api_key="test-key")
    
    # Mock the GET response for polling - return completed status
    mock_aiohttp_session.setup_response("get", json_data={
        "output": {
            "task_status": "SUCCEEDED",
            "results": [{"url": "https://example.com/results/test-result.jpg"}]
        }
    })
    
    # Poll the task
    urls = await client.poll_task("test-task-123", interval=0.1, max_attempts=5)
    
    # Verify returned result
    assert len(urls) == 1
    assert urls[0] == "https://example.com/results/test-result.jpg"


@pytest.mark.asyncio
async def test_wanx_client_download_results(mock_aiohttp_session):
    # Ensure .inbox exists for test
    os.makedirs(".inbox", exist_ok=True)
    
    client = WanxClient(api_key="test-key")
    
    # Mock the GET response for downloading results
    mock_aiohttp_session.setup_response("get", read_data=b"fake image data")
    
    # Download results
    output_path = await client.download_image(
        "https://example.com/results/test-result.jpg",
        "test_download"
    )
    
    # Verify file was saved
    assert ".inbox" in output_path
    assert os.path.exists(output_path)
    
    # Clean up test file
    if os.path.exists(output_path):
        os.remove(output_path)
