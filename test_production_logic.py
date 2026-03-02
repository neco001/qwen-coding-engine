
import asyncio
import os
import sys
from qwen_mcp.images import ImageHandler
from qwen_mcp.wanx_builder import WanxPayloadBuilder

async def test_internal_logic():
    print(">>> [ANIA] Testing Internal ImageHandler logic without MCP transport")
    
    # Setup environment
    # The API key should already be in the environment, but let's be safe
    api_key = os.getenv("ALIBABA_AI_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: No API key found.")
        return
    
    os.environ["DASHSCOPE_API_KEY"] = api_key
    os.environ["DASHSCOPE_WANX_BASE"] = "https://dashscope.aliyuncs.com/api/v1"
    
    handler = ImageHandler()
    
    img1 = r".inbox\image_1.tmp.jpg"
    img2 = r".inbox\image_2.tmp.jpg"
    
    prompt = "A simple test: A girl with headphones in a cafe."
    
    print("Calling handler.generate_image (Sync mode implicitly)...")
    result = await handler.generate_image(
        prompt=prompt,
        image_paths=[img1, img2],
        model_override="qwen-image-edit-max",
        dry_run=False
    )
    
    print("\nRESULT:")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_internal_logic())
