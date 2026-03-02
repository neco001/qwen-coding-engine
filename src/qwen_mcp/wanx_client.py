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

    async def generate_task(self, payload: Dict[str, Any], async_mode: bool = True) -> Dict[str, Any]:
        """
        Starts an image generation task. 
        Returns a dict: {"task_id": str} OR {"urls": list[str]} if synchronous result.
        """
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # Detection: Singapore (intl) DOES NOT support X-DashScope-Async header
        # for qwen-image-edit series. It results in 403 AccessDenied.
        is_intl = "intl" in self.base_url.lower()
        if async_mode and not is_intl:
            headers["X-DashScope-Async"] = "enable"
        
        # DEBUG LOGGING (to file only)
        try:
            os.makedirs(".inbox", exist_ok=True)
            with open(".inbox/debug_headers.log", "a", encoding="utf-8") as f:
                masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key else "NONE"
                f.write(f"[{datetime.now()}] URL: {url}\n")
                f.write(f"[{datetime.now()}] ASYNC: {async_mode}\n")
                f.write(f"[{datetime.now()}] HEADERS: {json.dumps({k:v for k,v in headers.items() if k!='Authorization'}, indent=2)}\n")
        except:
            pass
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"DashScope API Error ({resp.status}): {error_text}")
                
                data = await resp.json()
                
                # Check for synchronous results (output.choices or output.results)
                output = data.get("output", {})
                choices = output.get("choices")
                if choices:
                    urls = []
                    for choice in choices:
                        content = choice.get("message", {}).get("content", [])
                        for item in content:
                            if isinstance(item, dict) and "image" in item:
                                urls.append(item["image"])
                    if urls:
                        return {"urls": urls}
                
                # Some endpoints use results directly
                results = output.get("results")
                if results:
                    urls = [r["url"] for r in results if "url" in r]
                    if urls:
                        return {"urls": urls}

                # Check for asynchronous task_id
                task_id = output.get("task_id") or data.get("task_id")
                if task_id:
                    return {"task_id": task_id}
                
                raise Exception(f"Unexpected response format: {data}")

    async def poll_task(self, task_id: str, interval: float = 2.0, max_attempts: int = 45) -> List[str]:
        """Polls for task completion and returns a list of result URLs."""
        url = f"{self.base_url}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            for i in range(max_attempts):
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Polling error ({resp.status}): {error_text}")
                        await asyncio.sleep(interval)
                        continue

                    data = await resp.json()
                    output = data.get("output", {})
                    status = output.get("task_status")
                    
                    if status == "SUCCEEDED":
                        results = output.get("results", [])
                        urls = [r["url"] for r in results if "url" in r]
                        if not urls:
                            # Fallback for some models that use results field differently
                            urls = [output.get("url")] if output.get("url") else []
                        return [u for u in urls if u]
                    elif status == "FAILED":
                        message = output.get("message", "Unknown error")
                        raise Exception(f"Task {task_id} failed: {message}")
                    
                    # Log progress if available
                    progress = data.get("request_id") # just for context
                    logger.debug(f"Polling task {task_id}, status: {status} (attempt {i+1}/{max_attempts})")
                    
                    await asyncio.sleep(interval)
            
            raise TimeoutError(f"Task {task_id} timed out after {max_attempts} attempts.")

    async def download_image(self, url: str, prefix: str = "wanx") -> str:
        """Downloads an image from URL and saves it to .inbox."""
        os.makedirs(".inbox", exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        # Clean prefix and ensure filename
        safe_prefix = "".join([c for c in prefix if c.isalnum() or c in ("_", "-")]).strip("_")
        filename = f"{safe_prefix}_{unique_id}.png"
        filepath = os.path.join(".inbox", filename)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to download image: {resp.status} for URL: {url}")
                content = await resp.read()
                
                # Use aiofiles if available, or just standard for now
                with open(filepath, "wb") as f:
                    f.write(content)
        
        return filepath

    async def generate_image_full(self, payload: Dict[str, Any], poll_interval: float = 2.0) -> Dict[str, Any]:
        """
        Coordinates the full cycle: request, optional poll, and immediate download.
        This provides the 'Surgical' experience with high stability.
        """
        # STEP 1: Start Task
        # Based on documentation and testing: 
        # Singapore (Intl) requires SYNC call (long request). Async pooling is NOT supported.
        is_intl = "intl" in self.base_url.lower()
        async_mode = not is_intl # Force sync for Intl, Async only for CN
        
        result_info = await self.generate_task(payload, async_mode=async_mode)
        
        urls = []
        if "urls" in result_info:
            urls = result_info["urls"]
        elif "task_id" in result_info:
            urls = await self.poll_task(result_info["task_id"], interval=poll_interval)
        
        if not urls:
            raise Exception("No image URLs were returned from the API.")
        
        # STEP 2: Immediate Download (CRITICAL: before URL signature expires)
        local_paths = []
        for url in urls:
            local_path = await self.download_image(url)
            local_paths.append(local_path)
            
        return {
            "local_paths": local_paths,
            "remote_urls": urls,
            "status": "SUCCEEDED"
        }

