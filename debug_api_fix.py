
import os
import asyncio
import sys

# Dodaj src do path, żeby impory działały
sys.path.append(os.path.join(os.getcwd(), "src"))

from qwen_mcp.api import DashScopeClient

async def test():
    try:
        client = DashScopeClient()
        print(f"✅ Success! API Key found: {client.api_key[:5]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
