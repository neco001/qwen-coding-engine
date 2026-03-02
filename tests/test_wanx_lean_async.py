import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qwen_mcp.wanx_client import WanxClient
from qwen_mcp.wanx_builder import WanxPayloadBuilder

@pytest.fixture
def mock_aiohttp():
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        def setup_response(method, json_data=None, status=200, read_data=None):
            mock_resp = MagicMock()
            mock_resp.status = status
            if json_data:
                mock_resp.json = AsyncMock(return_value=json_data)
            if read_data:
                mock_resp.read = AsyncMock(return_value=read_data)
            
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_cm.__aexit__ = AsyncMock()
            
            getattr(mock_session, method).return_value = mock_cm
            return mock_resp

        mock_session.setup_response = setup_response
        yield mock_session

@pytest.mark.asyncio
async def test_wanx_client_generate_image_full_sync_flow(mock_aiohttp):
    """
    Tests the new 'full' flow where the API returns the image URL directly (sync mode)
    and the client automatically downloads it.
    """
    client = WanxClient(api_key="test-key")
    
    # 1. Mock the POST response for image generation (Synchronous result)
    mock_aiohttp.setup_response("post", json_data={
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"image": "https://example.com/sync-result.png"}
                        ]
                    }
                }
            ]
        }
    })
    
    # 2. Mock the GET response for downloading the image
    mock_aiohttp.setup_response("get", read_data=b"fake image data")
    
    payload = WanxPayloadBuilder().set_prompt("A beautiful sunset").build()
    
    # This method DOES NOT EXIST YET -> RED PHASE
    # It should:
    # - Call the API
    # - Detect sync/async
    # - Download the image(s)
    # - Return local paths
    result = await client.generate_image_full(payload)
    
    # Verify results
    assert "local_paths" in result
    assert len(result["local_paths"]) == 1
    assert os.path.exists(result["local_paths"][0])
    
    # Clean up
    if os.path.exists(result["local_paths"][0]):
        os.remove(result["local_paths"][0])

@pytest.mark.asyncio
async def test_wanx_client_generate_image_full_async_flow(mock_aiohttp):
    """
    Tests the flow where the API returns a task_id, the client polls it, 
    and then downloads the result.
    """
    client = WanxClient(api_key="test-key")
    
    # 1. Mock POST -> returns task_id
    mock_aiohttp.post.return_value.__aenter__.return_value.status = 200
    mock_aiohttp.post.return_value.__aenter__.return_value.json = AsyncMock(return_value={
        "output": {"task_id": "async-123"}
    })
    
    # 2. Mock GET (poll) -> returns SUCCESS
    mock_aiohttp.get.side_effect = [
        # First GET for polling
        MagicMock(__aenter__=AsyncMock(return_value=MagicMock(
            status=200, 
            json=AsyncMock(return_value={
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://example.com/async-result.png"}]
                }
            })
        ))),
        # Second GET for downloading
        MagicMock(__aenter__=AsyncMock(return_value=MagicMock(
            status=200,
            read=AsyncMock(return_value=b"fake image data")
        )))
    ]
    
    payload = WanxPayloadBuilder().set_prompt("A futuristic city").build()
    
    # Calling the new method
    result = await client.generate_image_full(payload, poll_interval=0.1)
    
    assert "local_paths" in result
    assert len(result["local_paths"]) == 1
    assert "async-result" in result["local_paths"][0]
    
    if os.path.exists(result["local_paths"][0]):
        os.remove(result["local_paths"][0])
