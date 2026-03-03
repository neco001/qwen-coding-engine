
import os
import asyncio
import aiohttp
import json
import base64

# Configuration
API_KEY = os.getenv("ALIBABA_AI_KEY") or os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def test_async_wrapper_logic():
    print(">>> [ANIA] Testing Async Python Shell for Synchronous API (Singapore)")
    
    img1_path = r"C:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\img4.jpg"
    img2_path = r"C:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\img5.jpg"
    
    img1_b64 = encode_image(img1_path)
    img2_b64 = encode_image(img2_path)
    
    # Cleaning the prompt from "episodes" - focused on geometry and identity.
    prompttt = """ 
        **The goal** of this task is to get **product packshot in new perspective**. 
        - show product from image 1 and image 2.
        - *Position:* The product is lying flat with the top edge pointing towards the upper right and the bottom edge towards the lower left.
        - *Perspective:* The photo is taken from a high-angle, slightly elevated perspective (roughly 45 degrees), looking down at the device. 
        - prohibited is to change color or change of the look of any part of the product
        - product should be on white, plain background.
        - you have to keep colors, and product details on very high level.
        - Prohibited is to change color of displayed `100%` or `22.5W` text.
        - keep high level od similarity of the braided lanyard, including the "cube" shape of the lanyard.
    """
    payload = {
        "model": "qwen-image-edit-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{img1_b64}"},
                        {"image": f"data:image/jpeg;base64,{img2_b64}"},
                        {"text": f"{prompttt}"}
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "size": "1024*1024",
            "negative_prompt": "low quality, bad proportions, blurry, deformed text, stand-up position",
            "prompt_extend": False,
            "watermark": False
        }
    }

    # IMPORTANT: No "X-DashScope-Async": "enable" here.
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    print("--- Sending long-lived async request via aiohttp...")
    async with aiohttp.ClientSession() as session:
        # This await will take ~60-90s, but it's non-blocking for the event loop
        async with session.post(BASE_URL, headers=headers, json=payload) as resp:
            print(f"--- Response Status: {resp.status}")
            data = await resp.json()
            
            # Show output
            if resp.status == 200:
                print("✅ SUCCESS (Direct Sync Result)")
                print(json.dumps(data, indent=2))
            else:
                print("❌ FAILED")
                print(data)

if __name__ == "__main__":
    if not API_KEY:
        print("Error: Missing API KEY")
    else:
        asyncio.run(test_async_wrapper_logic())
