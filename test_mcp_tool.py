import sys
import asyncio
sys.path.append('src')

from qwen_mcp.tools import generate_code_unified
from mcp.server.fastmcp import Context

async def test():
    # Simulate MCP tool call without context
    result = await generate_code_unified(
        prompt="Write a simple Python function that adds two numbers",
        mode="standard",
        context=None,
        ctx=None,
        project_id="test_project"
    )
    print(result[:500])

if __name__ == '__main__':
    asyncio.run(test())
