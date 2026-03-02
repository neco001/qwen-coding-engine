import pytest
import base64
from unittest.mock import patch, mock_open

# We will import the class from the file we are about to create
try:
    from qwen_mcp.wanx_builder import WanxPayloadBuilder
except ImportError:
    # Fallback to make the test run and inevitably fail if it doesn't exist
    WanxPayloadBuilder = None

def test_payload_builder_exists():
    assert WanxPayloadBuilder is not None, "WanxPayloadBuilder logic not implemented yet."

@patch('os.path.exists')
def test_build_payload_urls(mock_exists):
    """Test payload construction with simple URLs (no base64 needed)"""
    # Assuming URL doesn't trigger os.path.exists, but just in case
    builder = WanxPayloadBuilder(model="qwen-image-edit-max", size="1024*1536")
    
    images = ["https://example.com/img1.png", "https://example.com/img2.jpg"]
    prompt = "Make the girl from Image 1 wear the dress from Image 2."
    
    payload = builder.build(images, prompt)
    
    # Verify exact JSON shape
    assert payload["model"] == "qwen-image-edit-max"
    assert "input" in payload
    assert "messages" in payload["input"]
    
    content = payload["input"]["messages"][0]["content"]
    assert len(content) == 3, "Content should have 2 images and 1 text instruction"
    
    # First item must be image 1
    assert "image" in content[0]
    assert content[0]["image"] == images[0]
    
    # Second item must be image 2
    assert "image" in content[1]
    assert content[1]["image"] == images[1]
    
    # Third must be text
    assert "text" in content[2]
    assert content[2]["text"] == prompt

@patch('os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data=b"fake_image_data")
def test_build_payload_local_base64(mock_file, mock_exists):
    """Test payload construction with local files that need Base64 encoding"""
    builder = WanxPayloadBuilder(model="qwen-image-edit-max", size="768*1152")
    
    images = ["/local/path/img1.png"]
    prompt = "Describe Image 1."
    
    payload = builder.build(images, prompt)
    content = payload["input"]["messages"][0]["content"]
    
    assert len(content) == 2 # 1 image, 1 text
    assert "image" in content[0]
    
    b64_str = base64.b64encode(b"fake_image_data").decode('utf-8')
    expected_image_node = f"data:image/png;base64,{b64_str}"
    
    assert content[0]["image"] == expected_image_node

def test_parameters_inclusion():
    """Ensure n, negative_prompt, and watermark are passed correctly."""
    if WanxPayloadBuilder is None:
        pytest.skip("Class not implemented")
        
    builder = WanxPayloadBuilder()
    payload = builder.build(["http://url.com/1.png"], "prompt text", n=2, negative_prompt="ugly")
    
    params = payload["parameters"]
    assert params["n"] == 2
    assert params["negative_prompt"] == "ugly"
    assert params["prompt_extend"] is True
    assert params["watermark"] is False
