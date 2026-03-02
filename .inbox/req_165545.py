
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
        with open(r'C:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\req_165545.json', 'r', encoding='utf-8') as f:
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
        with open(r'C:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\res_165545.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
    except Exception as e:
        with open(r'C:\Users\pawel\OneDrive\python_reps\_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.inbox\res_165545.json', 'w', encoding='utf-8') as f:
            json.dump({"status": "error", "message": str(e)}, f)

if __name__ == "__main__":
    asyncio.run(main())
