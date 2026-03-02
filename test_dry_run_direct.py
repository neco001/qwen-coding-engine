"""Direct test of generate_qwen_image dry_run — bypasses MCP entirely."""
import sys, os, asyncio
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "fake-for-dry-run")

from qwen_mcp.tools import generate_qwen_image

async def main():
    print("=" * 60)
    print("TEST: generate_qwen_image(dry_run=True)")
    print("=" * 60)
    
    img1 = r"c:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\.inbox\image_1.tmp.jpg"
    img2 = r"c:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\.inbox\image_2.tmp.jpg"
    
    for p in [img1, img2]:
        print(f"  File exists: {os.path.exists(p)} -> {p}")
    
    try:
        result = await generate_qwen_image(
            prompt="Girl in cafe with headphones. Ultra realistic.",
            image_paths=[img1, img2],
            aspect_ratio="9:16",
            model_id="qwen-image-edit-max",
            dry_run=True
        )
        print("\n--- RESULT (first 2000 chars) ---")
        print(str(result)[:2000])
        print("--- END ---")
    except Exception as e:
        print(f"\nEXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
