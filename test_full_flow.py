import sys
import asyncio
import time
sys.path.append('src')

from qwen_mcp.server import qwen_coder
from mcp.server.fastmcp import Context

async def test():
    print("Starting full qwen_coder flow test...")
    start = time.time()
    
    try:
        # Simulate MCP tool call without context (ctx=None)
        result = await qwen_coder(
            prompt="Write a simple Python function that adds two numbers",
            mode="standard",
            context=None,
            ctx=None
        )
        elapsed = time.time() - start
        print(f"SUCCESS in {elapsed:.2f}s")
        print(f"Result: {result[:200]}...")
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test())
