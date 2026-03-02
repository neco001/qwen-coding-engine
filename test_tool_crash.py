
import asyncio
from qwen_mcp.tools import generate_qwen_image
from mcp.server.fastmcp import Context

async def test_tool_direct():
    print("Testing generate_qwen_image tool directly...")
    ctx = Context()
    try:
        res = await generate_qwen_image(
            prompt="test",
            dry_run=True,
            image_paths=[".inbox/image_1.tmp.jpg"]
        )
        print("Success!")
        print(res)
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tool_direct())
