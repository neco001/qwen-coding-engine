
import os
import dashscope
from dashscope import MultiModalConversation
from http import HTTPStatus
import json

# Setup
api_key = os.getenv("ALIBABA_AI_KEY") or os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def test_sync_call():
    print(">>> Testing Synchronous Call to qwen-image-edit-max (Singapore)")
    
    # Simple prompt for testing
    prompt = "A high-quality photo of a cup of coffee on a wooden table, soft morning light."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]
    
    # We will try a synchronous call without setting any async headers
    response = MultiModalConversation.call(
        model="qwen-image-edit-max",
        messages=messages,
        n=1,
        size="1024*1024"
    )
    
    if response.status_code == HTTPStatus.OK:
        print("✅ SUCCESS!")
        print(json.dumps(response.output, indent=2))
    else:
        print(f"❌ FAILED! Status: {response.status_code}")
        print(f"Code: {response.code}")
        print(f"Message: {response.message}")
        print(f"Request ID: {response.request_id}")

if __name__ == "__main__":
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found in environment.")
    else:
        test_sync_call()
