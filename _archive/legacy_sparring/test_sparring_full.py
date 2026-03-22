import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

import logging

# Configure logging to see API insights
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from qwen_mcp.tools import generate_sparring


async def test_sparring_integration():
    print("Testing Sparring Integration (Flash Mode)...")
    topic = "Should we focus on feature A or B?"
    context = "Minimal context for testing."

    try:
        # Note: This will actually call the Alibaba API.
        # Using flash mode to minimize tokens.
        res = await generate_sparring(topic=topic, context=context, mode="flash")
        print("\n--- SPARRING RESULT ---")
        print(res)
        print("-----------------------")
        print("\nSUCCESS: Sparring Engine is online.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_sparring_integration())
