import logging
import json
import re
import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime
from mcp.server.fastmcp import Context
from qwen_mcp.api import DashScopeClient
from qwen_mcp.registry import registry
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.specter.telemetry import get_broadcaster

# New modular imports
from qwen_mcp.prompts.system import AUDIT_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT
from qwen_mcp.prompts.lachman import LP_DISCOVERY_PROMPT, LP_ARCHITECT_PROMPT, LP_VERIFIER_PROMPT
from qwen_mcp.prompts.image import IMAGE_PROMPT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def _dbg(msg: str):
    """Debug logger — writes to .inbox/debug_payload.log AND stderr."""
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{ts}] [TOOL] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        os.makedirs('.inbox', exist_ok=True)
        with open('.inbox/debug_payload.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

from .utils import extract_json_from_text

async def generate_audit(content: str, context: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    client = DashScopeClient()
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context or 'None'}\n\nContent to audit:\n{content}"}
    ]
    return await client.generate_completion(
        messages=messages,
        task_type="audit",
        complexity="high",
        tags=["audit"],
        progress_callback=ctx.report_progress if ctx else None
    )

async def generate_code(prompt: str, context: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    client = DashScopeClient()
    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context or 'None'}\n\nPrompt: {prompt}"}
    ]
    return await client.generate_completion(
        messages=messages, 
        task_type="coding", 
        tags=["coder"],
        progress_callback=ctx.report_progress if ctx else None
    )

async def generate_code_25(prompt: str, context: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    client = DashScopeClient()
    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context or 'None'}\n\nPrompt: {prompt}"}
    ]
    return await client.generate_completion(
        messages=messages, 
        model_override="qwen2.5-72b-instruct", 
        task_type="coding", 
        tags=["coder25"],
        progress_callback=ctx.report_progress if ctx else None
    )

async def generate_lp_blueprint(goal: str, context: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    # This remains complex, but keeps the modular structure
    client = DashScopeClient()
    
    # 1. Discovery
    discovery_msg = [
        {"role": "system", "content": LP_DISCOVERY_PROMPT},
        {"role": "user", "content": f"Goal: {goal}\nContext: {context or ''}"}
    ]
    discovery_raw = await client.generate_completion(messages=discovery_msg)
    discovery = extract_json_from_text(discovery_raw) or {"hired_squad": []}
    squad_str = ", ".join([s.get("role", "Expert") for s in discovery.get("hired_squad", [])])

    # 2. Architect
    arch_msg = [
        {"role": "system", "content": LP_ARCHITECT_PROMPT.format(squad=squad_str)},
        {"role": "user", "content": f"Goal: {goal}"}
    ]
    blueprint = await client.generate_completion(messages=arch_msg, task_type="strategist")
    return blueprint

async def read_repo_file(path: str) -> str:
    import os
    if not os.path.exists(path):
        return f"Error: File not found at {path}"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

async def list_repo_files(directory: str = ".", pattern: str = "**/*") -> str:
    import glob
    import os
    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    return "\n".join([f for f in files if os.path.isfile(f)][:100])

async def generate_usage_report() -> str:
    from qwen_mcp.api import billing_tracker
    return billing_tracker.get_aggregated_report()

async def list_available_models() -> str:
    from qwen_mcp.api import DashScopeClient
    client = DashScopeClient()
    models = await client.list_models()
    return json.dumps(models, indent=2) if models else "[]"

async def set_model_in_registry(role: str, model_id: str) -> str:
    registry.models[role] = model_id
    await registry.save_cache()
    return f"Success: Role '{role}' set to '{model_id}'."

async def generate_sparring(
    topic: str, context: str = "", mode: str = "flash", ctx: Optional[Context] = None
) -> str:
    """
    Executes the 5D Sparring Engine via SparringEngine module.
    """
    from qwen_mcp.engines.sparring import SparringEngine
    client = DashScopeClient()
    engine = SparringEngine(client)
    
    if mode == "flash":
        return await engine.run_flash(topic, context, ctx)
    else:
        return await engine.run_pro(topic, context, ctx)


async def heal_registry() -> str:
    client = DashScopeClient()
    return await client.heal_registry()

async def generate_qwen_image(
    prompt: str, 
    image_paths: List[str] = None, 
    size: str = "1:1",
    model: Optional[str] = None,
    prompt_extend: bool = True,
    dry_run: bool = False,
    ctx: Optional[Context] = None
) -> str:
    """Executes WanX image generation via direct async API calls."""
    from qwen_mcp.wanx_client import WanxClient
    from qwen_mcp.wanx_builder import WanxPayloadBuilder
    from qwen_mcp.api import DashScopeClient
    
    # Get API key from DashScopeClient - fallback to environment
    ds_client = DashScopeClient()
    api_key = getattr(ds_client, 'api_key', None) or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_AI_KEY")
    
    if not api_key:
        return "Error: DashScope API key not found in environment or client."

    if dry_run:
        return f"Dry Run: Would generate image with prompt '{prompt}' and images {image_paths}"

    # Build payload using the smart builder
    builder = WanxPayloadBuilder()
    builder.set_model(model).set_prompt(prompt).set_size(size).set_prompt_extend(prompt_extend)
    
    if image_paths:
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        builder.set_images(valid_paths)
        for p in image_paths:
            if not os.path.exists(p):
                _dbg(f"Warning: Image path not found: {p}")

    payload = builder.build()
    
    # Initialize WanxClient
    wanx = WanxClient(api_key=api_key)
    
    try:
        if ctx:
            ctx.info(f"🎨 Starting WanX Generation (Lean Async)...")
        
        # Execute full cycle (Request -> Poll -> Download)
        result = await wanx.generate_image_full(payload)
        
        local_paths = result.get("local_paths", [])
        if not local_paths:
            return "Error: Generation succeeded but no local files were saved."
        
        paths_str = "\n".join([f"- `{p}`" for p in local_paths])
        return f"✅ **Image Generated Successfully!**\nLocal files saved at:\n{paths_str}\n\nYou can now view these files in your workspace."
        
    except Exception as e:
        error_msg = f"❌ **Generation Failed:** {str(e)}"
        _dbg(error_msg)
        return error_msg



async def refine_image_prompt(raw_prompt: str, ctx: Optional[Context] = None) -> str:
    """Uses qwen-plus to expand a raw idea into 3 balanced WanX-optimized prompts."""
    client = DashScopeClient()
    
    # The Architect's system prompt now handles Language Transition and Anatomy
    messages = [
        {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Architect the following prompt: {raw_prompt}"}
    ]
    
    return await client.generate_completion(
        messages=messages,
        model_override="qwen-plus",
        task_type="strategist",
        tags=["image_prompt"],
        progress_callback=ctx.report_progress if ctx else None
    )

async def prepare_visual_reference(image_paths: List[str]) -> str:
    """Collates up to 4 reference images into a single grid for Identity Preservation."""
    from PIL import Image
    import os
    
    if not image_paths:
        return "Error: No images provided."
    
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        return "Error: None of the provided image paths exist."
    
    # Limit to 4 images for the grid
    valid_paths = valid_paths[:4]
    images = [Image.open(p) for p in valid_paths]
    
    # Standardize size (e.g., 512x512 per cell)
    size = (512, 512)
    images = [img.resize(size) for img in images]
    
    # Calculate grid size
    n = len(images)
    cols = 2 if n > 1 else 1
    rows = (n + 1) // 2
    
    grid = Image.new('RGB', (cols * 512, rows * 512))
    
    for i, img in enumerate(images):
        col = i % 2
        row = i // 2
        grid.paste(img, (col * 512, row * 512))
    
    output_dir = ".inbox"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "temp_ref.png")
    grid.save(output_path)
    
    return f"✅ **Identity Preservation Grid Ready at:** `{output_path}`\nZłożyłam {n} zdjęć w jedno – teraz model zobaczy produkt z każdej strony. Możesz tego użyć jako `base_image_path` w `qwen_generate_image`."
