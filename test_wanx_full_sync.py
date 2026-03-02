
import os
import dashscope
from dashscope import MultiModalConversation
from http import HTTPStatus
import json
import base64

# Setup
api_key = os.getenv("ALIBABA_AI_KEY") or os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key
# Explicitly set to Singapore endpoint
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def run_production_test():
    print(">>> [ANIA] Field Test: Synchronous Generation via SDK (Singapore)")
    print(">>> Target Model: qwen-image-edit-max")
    
    # Using official Alibaba sample URLs for stable testing
    img1_uri = "https://img.alicdn.com/imgextra/i3/O1CN0157XGE51l6iL9441yX_!!6000000004770-49-tps-1104-1472.webp"
    img2_uri = "https://img.alicdn.com/imgextra/i3/O1CN01SfG4J41UYn9WNt4X1_!!6000000002530-49-tps-1696-960.webp"
    
    print(f"--- Using remote assets:\n    1: {img1_uri}\n    2: {img2_uri}")
    
    prompt = """High-resolution lifestyle photograph, shallow depth of field: A young Central European woman (Polish features — fair skin, light brown wavy hair tied in a low bun, soft facial structure) seated at a marble-topped café table, torso upright but relaxed, hands resting gently on her thighs. She wears over-ear headphones (reference: image 1 and image 2 - keep high level of references details) — snugly fitted, ear pads fully covering ears, head slightly tilted down in quiet immersion. Her eyes are softly closed, lips faintly curved upward, jaw muscles relaxed, subtle smile lines around eyes — conveying deep calm and private satisfaction. Background is a vibrant, bustling Warsaw-style café: blurred barista pouring espresso, warm pendant lights, chalkboard menu with Polish script, ceramic mugs and scattered croissants — all rendered as creamy bokeh with warm amber/cream tonal palette. Shot from eye-level, f/1.4, natural window light from left highlighting cheekbone and headphone texture, skin texture and fabric weave (linen blouse, wool-blend scarf) rendered in ultra-sharp focus."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": img1_uri},
                {"image": img2_uri},
                {"text": prompt}
            ]
        }
    ]
    
    print("--- Sending Synchronous Request to Singapore...")
    print("--- Note: This usually takes 60-120 seconds for the 'max' model. Patience requested.")
    
    # We use .call() which is synchronous. No X-DashScope-Async header is added by default here.
    response = MultiModalConversation.call(
        model="qwen-image-edit-max",
        messages=messages,
        n=1,
        # size is specified as a parameter in SDK
        size="720*1280" 
    )
    
    if response.status_code == HTTPStatus.OK:
        print("\n✅ MISSION ACCOMPLISHED!")
        print(json.dumps(response.output, indent=2))
        
        # Save results for verification
        with open("last_success_output.json", "w") as f:
            json.dump(response.output, f, indent=2)
    else:
        print(f"\n🛑 MISSION FAILED! Status: {response.status_code}")
        print(f"Code: {response.code}")
        print(f"Message: {response.message}")
        print(f"Request ID: {response.request_id}")

if __name__ == "__main__":
    if not api_key:
        print("🛑 Missing API Key!")
    else:
        run_production_test()
