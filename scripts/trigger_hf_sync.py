import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from qwen_mcp.registry import registry


async def main():
    print("Starting manual HF sync...")
    res = await registry.sync_with_hf()
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
