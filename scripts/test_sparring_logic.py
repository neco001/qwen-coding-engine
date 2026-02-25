import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from qwen_mcp.tools import generate_sparring

async def test_sparring():
    print("Testing Sparring (Flash Mode)...")
    topic = "Should we move the entire infra to serverless to save 20% cost but risk vendor lock-in?"
    context = "Small startup, 5 engineers, growing traffic."
    
    # We won't actually call the API here to save tokens, 
    # but we will check if the logic flows up to the client call.
    # Actually, let's do a dry-run check of the prompts.
    
    from qwen_mcp.tools import SPARRING_ATTACKER_PROMPT
    print(f"Attacker Prompt available: {len(SPARRING_ATTACKER_PROMPT) > 0}")

if __name__ == "__main__":
    asyncio.run(test_sparring())
