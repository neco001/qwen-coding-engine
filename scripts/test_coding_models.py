import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def check_coding_plan_models():
    # Attempt to use the Coding Plan endpoint
    coding_base = "https://coding-intl.dashscope.aliyuncs.com/v1"
    api_key = "sk-020a31d4785c4ad68602c9df4d18af38" # From .env
    
    print(f"Checking models at: {coding_base}")
    client = AsyncOpenAI(api_key=api_key, base_url=coding_base)
    
    try:
        response = await client.models.list()
        models = [m.id for m in response.data]
        print(f"Found {len(models)} models:")
        for m in sorted(models):
            print(f"- {m}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_coding_plan_models())
