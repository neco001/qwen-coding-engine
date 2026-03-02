
import os
import dashscope
from dashscope import MultiModalConversation
from http import HTTPStatus
import json
import base64

# Setup
api_key = os.getenv("ALIBABA_AI_KEY") or os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_sync_call_with_images():
    print(">>> Testing Synchronous Call with 2 Images (Singapore)")
    
    img1_path = r"c:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\image_1.tmp.jpg"
    img2_path = r"c:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\image_2.tmp.jpg"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: Images not found.")
        return

    img1_base64 = encode_image(img1_path)
    img2_base64 = encode_image(img2_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{img1_base64}"},
                {"image": f"data:image/jpeg;base64,{img2_base64}"},
                {"text": "A girl with headphones (from image 1) sitting in a cafe, ultra realistic, detailed skin."}
            ]
        }
    ]
    
    print("Sending request... (Synchronous call to Singapore might take 30-90s)")
    # MultiModalConversation.call is synchronous by default
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
        test_sync_call_with_images()
