
import os
import json
import sys
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any

def generate_qwen_image_robust(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    aspect_ratio: str = "1:1",
    dry_run: bool = False,
) -> str:
    """
    SYNCHRONOUS ROBUST VERSION: Uses subprocess.run to avoid asyncio loop issues on Windows.
    """
    os.makedirs(".inbox", exist_ok=True)
    req_file = os.path.abspath(f".inbox/req_{datetime.now().strftime('%H%M%S')}.json")
    res_file = req_file.replace("req_", "res_")
    
    config = {
        "prompt": prompt,
        "image_paths": image_paths,
        "aspect_ratio": aspect_ratio,
        "model_id": "qwen-image-edit-max",
        "prompt_extend": False,
        "negative_prompt": "low quality, bad anatomy",
        "dry_run": dry_run
    }
    
    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(config, f)

    # Simple wrapper to run logic isolated
    wrapper_code = f"""
import os
import json
import asyncio
import sys

# Add current dir to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from qwen_mcp.images import ImageHandler

async def main():
    try:
        with open(r'{req_file}', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        handler = ImageHandler()
        result = await handler.generate_image(
            prompt=cfg['prompt'],
            image_paths=cfg['image_paths'],
            aspect_ratio=cfg['aspect_ratio'],
            model_override=cfg['model_id'],
            prompt_extend=cfg['prompt_extend'],
            negative_prompt=cfg['negative_prompt'],
            dry_run=cfg['dry_run']
        )
        with open(r'{res_file}', 'w', encoding='utf-8') as f:
            json.dump(result, f)
    except Exception as e:
        with open(r'{res_file}', 'w', encoding='utf-8') as f:
            json.dump({{"status": "error", "message": str(e)}}, f)

if __name__ == "__main__":
    asyncio.run(main())
"""
    wrapper_script = req_file.replace(".json", ".py")
    with open(wrapper_script, "w", encoding="utf-8") as f:
        f.write(wrapper_code)

    try:
        # Run SYNCHRONOUSLY to be safe from async loop crashes
        subprocess.run([sys.executable, wrapper_script], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return f"🛑 Subprocess failed: {e.stderr}"

    if not os.path.exists(res_file):
        return "🛑 Result file not found."

    with open(res_file, "r", encoding="utf-8") as f:
        result = json.load(f)

    if result.get("status") == "success":
        urls = result.get("urls", [])
        paths = result.get("local_paths", [])
        return f"✅ SUCCESS!\nPaths:\n" + "\n".join(paths) + "\n\nURLs:\n" + "\n".join(urls)
    elif result.get("status") == "dry_run":
        return f"🔍 DRY RUN OK.\nPayload ready."
    else:
        return f"🛑 Error: {result.get('message', 'Unknown')}"
