import sys
import asyncio
sys.path.append('src')

from qwen_mcp.engines.coder_v2 import CoderEngineV2
from qwen_mcp.api import DashScopeClient

async def test():
    c = DashScopeClient()
    e = CoderEngineV2(c)
    r = await e.execute(prompt='add two numbers', mode='standard')
    print(r.to_markdown())

if __name__ == '__main__':
    asyncio.run(test())
