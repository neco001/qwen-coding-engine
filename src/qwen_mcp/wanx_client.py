import os
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import json

from .base import DASHSCOPE_WANX_BASE_URL

logger = logging.getLogger(__name__)

class WanxClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = DASHSCOPE_WANX_BASE_URL

    async def generate_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Starts an image generation task. 
        Returns a dict: {"task_id": str} OR {"urls": list[str]} if synchronous.
        """
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        # We REMOVE "X-DashScope-Async": "enable" to allow synchronous response
        # on International endpoints which often block (403) async tasks.
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable"
        }
        
        # DEBUG LOGGING (to file only, avoiding stderr to prevent MCP issues)
        try:
            os.makedirs(".inbox", exist_ok=True)
            with open(".inbox/debug_headers.log", "a", encoding="utf-8") as f:
                masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key else "NONE"
                f.write(f"[{datetime.now()}] URL: {url}\n")
                f.write(f"[{datetime.now()}] MASKED_KEY: {masked_key}\n")
                f.write(f"[{datetime.now()}] HEADERS: {json.dumps({k:v for k,v in headers.items() if k!='Authorization'}, indent=2)}\n")
        except:
            pass
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"DashScope API Error ({resp.status}): {error_text}")
                
                data = await resp.json()
                
                # Check for synchronous results (choices/content)
                choices = data.get("output", {}).get("choices") or data.get("choices")
                if choices:
                    urls = []
                    for choice in choices:
                        content = choice.get("message", {}).get("content", [])
                        for item in content:
                            if "image" in item:
                                urls.append(item["image"])
                    if urls:
                        return {"urls": urls}

                # Check for asynchronous task_id
                task_id = data.get("output", {}).get("task_id") or data.get("task_id")
                if task_id:
                    return {"task_id": task_id}
                
                raise Exception(f"Unexpected response format (no task_id or choices): {data}")

    async def poll_task(self, task_id: str, interval: float = 2.0, max_attempts: int = 30) -> List[str]:
        """Polls for task completion and returns a list of result URLs."""
        url = f"{self.base_url}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            for _ in range(max_attempts):
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json()
                    output = data.get("output", {})
                    status = output.get("task_status")
                    
                    if status == "SUCCEEDED":
                        results = output.get("results", [])
                        return [r["url"] for r in results if "url" in r]
                    elif status == "FAILED":
                        message = output.get("message", "Unknown error")
                        raise Exception(f"Task {task_id} failed: {message}")
                    
                    await asyncio.sleep(interval)
            
            raise TimeoutError(f"Task {task_id} timed out after {max_attempts} attempts.")

    async def download_image(self, url: str, prefix: str = "wanx") -> str:
        """Downloads an image from URL and saves it to .inbox."""
        os.makedirs(".inbox", exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{prefix}_{unique_id}.png"
        filepath = os.path.join(".inbox", filename)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to download image: {resp.status}")
                content = await resp.read()
                
                with open(filepath, "wb") as f:
                    f.write(content)
        
        return filepath
