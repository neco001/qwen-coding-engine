import asyncio
import os
import json
from qwen_mcp.api import DashScopeClient

async def test_dry_run():
    client = DashScopeClient()
    
    # Mock API Key if not present
    if not os.getenv("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = "sk-fake-key"
        
    prompt = "A girl wearing the dress from Image 2, holding the object from Image 1."
    image_paths = ["image1.png", "image2.jpg"]
    
    # Mock file existence
    from unittest.mock import patch
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', create=True):
            result = await client.generate_image(
                prompt=prompt,
                image_paths=image_paths,
                aspect_ratio="16:9",
                dry_run=True,
                prompt_extend=False
            )
            
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_dry_run())
