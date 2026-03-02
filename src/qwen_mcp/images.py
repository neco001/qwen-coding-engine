import os
import re
import sys
import json
import aiohttp
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base import BaseDashScopeClient, DASHSCOPE_WANX_BASE_URL
from .wanx_builder import WanxPayloadBuilder
from .wanx_client import WanxClient
from .specter.telemetry import get_broadcaster
from .billing import billing_tracker

logger = logging.getLogger(__name__)

def _dbg(msg: str):
    """Debug logger — writes to .inbox/debug_payload.log AND stderr."""
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{ts}] [IMG] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        os.makedirs('.inbox', exist_ok=True)
        with open('.inbox/debug_payload.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

# Image generation constants
DEFAULT_IMAGE_MODEL = os.getenv("DASHSCOPE_IMAGE_MODEL", "qwen-image-plus")
MAX_IMAGE_MODEL = "qwen-image-edit-max"
DEFAULT_IMAGE_SIZE = "1024*1024"

class ImageHandler(BaseDashScopeClient):
    """Handles image generation with WanX payload building and DashScope polling."""

    async def generate_image(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        size: str = DEFAULT_IMAGE_SIZE,
        n: int = 1,
        model_override: Optional[str] = None,
        ctx: Optional[Any] = None,
        prompt_extend: bool = True,
        negative_prompt: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Generates image via WanxClient logic with full lifecycle management."""
        target_model = model_override or MAX_IMAGE_MODEL
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return {"status": "error", "message": "DASHSCOPE_API_KEY environment variable is not set."}
            
        sanitized_prompt = self.sanitizer_cls.redact(prompt)
        
        # Default high-fidelity negative prompt
        default_neg = "low resolution, error, worst quality, low quality, disfigured, extra fingers, bad proportions, ui, text, letters, blurry"
        final_neg = negative_prompt if negative_prompt is not None else default_neg

        if ctx and hasattr(ctx, "report_progress"):
            await ctx.report_progress(progress=0, total=None, message=f"Preparing {target_model}...")

        try:
            _dbg(f">>> generate_image START | model={target_model} | dry_run={dry_run} | images={image_paths}")
            
            # Use fluent builder
            builder = WanxPayloadBuilder(model=target_model)
            _dbg(f"    builder created OK")
            
            builder.set_prompt(sanitized_prompt) \
                   .set_size(size) \
                   .set_n(n) \
                   .set_negative_prompt(final_neg) \
                   .set_prompt_extend(prompt_extend)
            _dbg(f"    setters chained OK")
            
            if image_paths:
                builder.set_images(image_paths)
                _dbg(f"    set_images OK | count={len(image_paths)}")
            
            payload = builder.build()
            _dbg(f"    build() OK | payload keys={list(payload.keys())} | content_len={len(payload.get('input',{}).get('messages',[{}])[0].get('content',[]))}")
            
            if dry_run:
                result = {
                    "status": "dry_run",
                    "payload": payload,
                    "endpoint": f"{DASHSCOPE_WANX_BASE_URL}/services/aigc/multimodal-generation/generation",
                    "model": target_model
                }
                _dbg(f"    DRY_RUN returning dict with keys={list(result.keys())}")
                return result

            if ctx and hasattr(ctx, "report_progress"):
                await ctx.report_progress(progress=5, total=100, message=f"Sending {target_model} request...")

            if ctx and hasattr(ctx, "report_progress"):
                await ctx.report_progress(progress=5, total=100, message=f"Creating {target_model} task...")

            client = WanxClient(api_key=api_key)
            gen_resp = await client.generate_task(payload)
            _dbg(f"    client.generate_task returned keys: {list(gen_resp.keys())}")

            urls = []
            if "urls" in gen_resp:
                urls = gen_resp["urls"]
                _dbg(f"    Detected SYNCHRONOUS response | urls_count={len(urls)}")
            elif "task_id" in gen_resp:
                task_id = gen_resp["task_id"]
                _dbg(f"    Detected ASYNCHRONOUS task | task_id={task_id}")
                logger.info(f"WanX Task Created: {task_id}")

                # Polling and Downloading
                max_attempts = 60 # ~120 seconds
                for i in range(max_attempts):
                    if ctx and hasattr(ctx, "report_progress"):
                        prog = min(10 + (i * 2), 95)
                        await ctx.report_progress(progress=prog, total=100, message=f"Synthesizing... ({i*2}s)")
                    
                    try:
                        urls = await client.poll_task(task_id, interval=0.1, max_attempts=1)
                        if urls: 
                            _dbg(f"    Polling success! Got {len(urls)} URLs")
                            break
                    except TimeoutError:
                        await asyncio.sleep(2.0)
                        continue
                    except Exception as e:
                        err_str = str(e).lower()
                        if "pending" in err_str or "running" in err_str or "waiting" in err_str:
                            await asyncio.sleep(2.0)
                            continue
                        _dbg(f"    Polling error: {e}")
                        raise e
            else:
                _dbg(f"    ERROR: Unexpected response format from generate_task: {gen_resp}")
                return {"status": "error", "message": f"Unexpected response format: {gen_resp}"}
            
            if not urls:
                return {"status": "error", "message": "Generation timed out."}

            if ctx and hasattr(ctx, "report_progress"):
                await ctx.report_progress(progress=95, total=100, message="Saving files to .inbox...")

            local_paths = []
            for url in urls:
                path = await client.download_image(url, prefix=f"wanx_{target_model.split('-')[-1]}")
                local_paths.append(path)

            await self._log_image_usage(target_model)
            return {"status": "success", "urls": urls, "local_paths": local_paths, "model": target_model}
                    
        except Exception as e:
            _dbg(f"    EXCEPTION: {type(e).__name__}: {e}")
            _dbg(f"    TRACEBACK: {traceback.format_exc()}")
            logger.error(f"Image generation error: {e}")
            return {"status": "error", "message": str(e)}

    async def _log_image_usage(self, model: str):
        """Standardized image telemetry."""
        project_name = os.getenv("QWEN_PROJECT_NAME", "adhoc")
        try:
            billing_tracker.log_usage(project_name, model, 0, 0, image_count=1)
            await get_broadcaster().report_usage(model, 0, 0, is_image=True)
        except Exception as e:
            logger.error(f"Telemetry/Billing logging failed: {e}")
