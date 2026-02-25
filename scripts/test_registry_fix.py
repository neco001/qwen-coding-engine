import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from qwen_mcp.registry import registry


async def test_sync():
    print("Starting HF sync...")
    res = await registry.sync_with_hf()
    print(f"Sync result: {res}")

    print("\nTesting Router:")
    for task in ["discovery", "coder", "audit", "strategist"]:
        model = registry.route_request(task)
        print(f"Task: {task:12} -> Model: {model}")


if __name__ == "__main__":
    asyncio.run(test_sync())
