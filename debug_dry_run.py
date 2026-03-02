import sys
import os
import asyncio
import json

# Add src to path
sys.path.append(os.getcwd() + "/src")

from qwen_mcp.tools import generate_qwen_image

async def test_dry_run():
    print("Starting test...")
    try:
        # Mocking setup
        os.environ["DASHSCOPE_API_KEY"] = "fake-key"
        
        prompt = "test prompt"
        image_paths = ["c:/Users/pawel/OneDrive/python_reps/_Toolbox/mcp_servers/Qwen-mcp/qwen-coding-local/.inbox/image_1.tmp.jpg"]
        
        # Ensure file exists for builder test
        if not os.path.exists(image_paths[0]):
             with open(image_paths[0], "wb") as f:
                 f.write(b"fake data")

        print(f"Calling generate_qwen_image with dry_run=True...")
        res = await generate_qwen_image(
            prompt=prompt,
            image_paths=image_paths,
            dry_run=True
        )
        print("\n--- RESULT CONTENT ---")
        print(res)
        print("--- END RESULT ---")
        
    except Exception as e:
        print(f"FAILED WITH EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dry_run())
