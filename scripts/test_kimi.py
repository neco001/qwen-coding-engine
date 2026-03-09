import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def test_kimi_connectivity():
    coding_base = "https://coding-intl.dashscope.aliyuncs.com/v1"
    api_key = "sk-020a31d4785c4ad68602c9df4d18af38"
    
    print(f"Pinging kimi-k2.5 at: {coding_base}")
    client = AsyncOpenAI(api_key=api_key, base_url=coding_base)
    
    try:
        response = await client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "Say hello from Kimi!"}],
            max_tokens=20,
            extra_body={"enable_thinking": True}
        )
        print("Success!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_kimi_connectivity())
