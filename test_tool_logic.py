import sys
import os
import asyncio
import json

# Add src to path
sys.path.append(os.getcwd() + "/src")

from qwen_mcp.tools import generate_qwen_image
from mcp.server.fastmcp import Context

async def test_tool_direct():
    print("Starting manual tool execution test...")
    try:
        os.environ["DASHSCOPE_API_KEY"] = "fake-key"
        
        # Test parameters
        prompt = "test prompt"
        image_paths = ["c:/Users/pawel/OneDrive/python_reps/_Toolbox/mcp_servers/Qwen-mcp/qwen-coding-local/.inbox/image_1.tmp.jpg"]
        
        # Ensure file exists
        os.makedirs(os.path.dirname(image_paths[0]), exist_ok=True)
        with open(image_paths[0], "wb") as f:
            f.write(b"fake image data")

        print("Executing generate_qwen_image(dry_run=True)...")
        res = await generate_qwen_image(
            prompt=prompt,
            image_paths=image_paths,
            dry_run=True,
            aspect_ratio="1:1"
        )
        print("\n=== TOOL OUTPUT ===")
        print(res)
        print("====================")
        
    except KeyError as ek:
        print(f"FAILED WITH KeyError: {str(ek)}")
    except Exception as e:
        print(f"FAILED WITH {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tool_direct())
