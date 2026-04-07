import sys
import asyncio
sys.path.append('src')

from qwen_mcp.api import DashScopeClient

async def test():
    c = DashScopeClient()
    r = await c.generate_completion(
        messages=[{'role': 'user', 'content': 'hi'}],
        task_type='coder',
        complexity='low'
    )
    print('API Response:', r[:100] if r else 'EMPTY')

if __name__ == '__main__':
    asyncio.run(test())
