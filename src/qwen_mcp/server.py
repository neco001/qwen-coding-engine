from typing import Optional, List, Dict, Any, Union
from mcp.server.fastmcp import FastMCP, Context
from qwen_mcp.tools import (
    generate_audit,
    generate_code,
    generate_code_25,
    generate_lp_blueprint,
    read_repo_file,
    list_repo_files,
    generate_usage_report,
    list_available_models,
    set_model_in_registry,
    generate_sparring,
    heal_registry,
    refine_image_prompt,
    prepare_visual_reference,
    generate_qwen_image,
)
from qwen_mcp.specter.telemetry import get_broadcaster
from qwen_mcp.registry import registry
import asyncio
import sys
import threading
import uvicorn
from fastapi import FastAPI, WebSocket

# Force UTF-8 encoding for stdout/stderr on Windows to prevent 'krzaki'
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

# Initialize FastMCP Server
mcp = FastMCP("Qwen MCP Server (DashScope)")


@mcp.tool()
async def qwen_audit(
    content: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Audits the provided code or terminal logs using Qwen models.
    """
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_state({
        "active_model": registry.get_best_model("strategist")
    })
    return await generate_audit(content, context, ctx)


@mcp.tool()
async def qwen_coder(
    prompt: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Generates or completes code using Qwen 3.5 Plus.
    """
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_state({
        "active_model": registry.get_best_model("coding")
    })
    return await generate_code(prompt, context, ctx)


@mcp.tool()
async def qwen_coder_25(
    prompt: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Generates or completes code using specialized Qwen-2.5-Coder-32B.
    """
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_state({
        "active_model": registry.get_best_model("coding")
    })
    return await generate_code_25(prompt, context, ctx)


@mcp.tool()
async def qwen_architect(
    goal: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Initiates 'The Lachman Protocol' (LP).
    The server hires a dynamic expert squad to audit your goal and generate a high-precision Blueprint.
    """
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_state({
        "active_model": registry.get_best_model("strategist")
    })
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Initiating Lachman Protocol..."
        )
    return await generate_lp_blueprint(goal, context, ctx)


@mcp.tool()
async def qwen_sparring_flash(
    topic: str, context: str = "", ctx: Context = None
) -> str:
    """
    ⚡ FLASH MODE: High-speed strategic analysis and reasoning-only deep dive.
    Best for: Quick tactical decisions, content refinement, and logic checks.
    """
    return await generate_sparring(topic, context, "flash", ctx)


@mcp.tool()
async def qwen_sparring_pro(
    topic: str, context: str = "", ctx: Context = None
) -> str:
    """
    🔥 PRO MODE: Full adversarial multi-agent debate (Lachman Protocol for Strategy).
    Best for: High-stakes dilemmas, critical audit of plans, and stress-testing moves.
    """
    return await generate_sparring(topic, context, "pro", ctx)


@mcp.tool()
async def qwen_refresh_models() -> str:
    """
    Checks for the latest SOTA models from Alibaba DashScope and identifies candidates.
    Also synchronizes metadata from Hugging Face.
    """
    from qwen_mcp.api import DashScopeClient
    from qwen_mcp.registry import registry

    client = DashScopeClient()
    ds_res = await client.refresh_registry()
    hf_res = await registry.sync_with_hf()
    return f"DashScope: {ds_res}\nHugging Face: {hf_res}"


@mcp.tool()
async def qwen_list_available_models() -> str:
    """
    Fetches and displays the full list of models available via your DashScope API key.
    Use this to find IDs for qwen_set_model.
    """
    return await list_available_models()


@mcp.tool()
async def qwen_set_model(role: str, model_id: str) -> str:
    """
    Manually sets a specific model ID for a role.
    Roles: 'strategist' (audit/LP), 'coder' (qwen_coder_25), 'scout' (internal/discovery).
    """
    return await set_model_in_registry(role, model_id)


@mcp.tool()
async def qwen_read_file(path: str) -> str:
    """
    Reads a file from the local repository to use as context for your request.
    """
    return await read_repo_file(path)


@mcp.tool()
async def qwen_list_files(directory: str = ".", pattern: str = "**/*") -> str:
    """
    Lists files in the repository to discover context.
    """
    return await list_repo_files(directory, pattern)


@mcp.tool()
async def qwen_usage_report() -> str:
    """
    Retrieves the DuckDB token billing usage report and formats it as an aggregated table
    (Grouped by Date, Project, and Model).
    """
    return await generate_usage_report()


def run_telemetry_server():
    """Starts a lightweight FastAPI server for the HUD telemetry."""
    app = FastAPI()
    broadcaster = get_broadcaster()

    @app.get("/")
    async def root():
        return {"status": "SPECTER LENS SIDECAR ACTIVE", "uplink": "ws://127.0.0.1:8878/ws/telemetry"}

    @app.websocket("/ws/telemetry")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await broadcaster.add_client(websocket)
        try:
            while True:
                # Keep alive and signal handling
                try:
                    # Non-blocking check for messages + heartbeat
                    await asyncio.wait_for(websocket.receive_text(), timeout=10)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat"})
        except Exception:
            pass
        finally:
            await broadcaster.remove_client(websocket)

    # Run on fixed port 8878
    try:
        config = uvicorn.Config(app, host="127.0.0.1", port=8878, log_level="warning")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        print(f"❌ Telemetry Sidecar failed: {e}")

async def sync_hud_state():
    """Broadcaster update for role mapping and basic state."""
    await asyncio.sleep(2) # Wait for sidecar thread to start
    await get_broadcaster().broadcast_state({"role_mapping": registry.models})

def main():
    """Main entrypoint for the MCP server."""
    print("🚀 [SPECTER] Starting Qwen Engineering Engine Context...")
    print("📡 [SPECTER] Sidecar Uplink on port 8878/ws")
    
    # Start telemetry in dedicated thread
    sidecar = threading.Thread(target=run_telemetry_server, daemon=True)
    sidecar.start()
    
    # Schedule initial state sync
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(sync_hud_state())
    except Exception:
        pass

    mcp.run()


@mcp.tool()
async def qwen_init_request() -> str:
    """
    ⚡ INITIALIZE NEW REQUEST: Resets 'This Prompt' token counter and buffers in the HUD.
    MANDATORY: Call this as your FIRST tool at the start of EVERY new user prompt/task.
    Failure to call this results in incorrect token accumulation in telemetry.
    """
    broadcaster = get_broadcaster()
    await broadcaster.start_request()
    return "✅ Specter HUD: 'This Prompt' counters reset. Ready for new engagement."


@mcp.tool()
async def qwen_refine_image_prompt(raw_prompt: str, ctx: Context = None) -> str:
    """
    Uses qwen-plus to expand a raw idea into 3 balanced WanX-optimized prompts.
    Provides realistic, artistic, and 'vibe' variations.
    """
    return await refine_image_prompt(raw_prompt, ctx)


@mcp.tool()
async def qwen_prepare_visual_reference(image_paths: List[str]) -> str:
    """
    Collates up to 4 reference images into a single grid (cheatsheet) for identity preservation.
    Saves a temporary grid for use in WanX generation at .inbox/wanx_cheatsheet.png.
    """
    from typing import List
    return await prepare_visual_reference(image_paths)


@mcp.tool()
async def wanx_gen_isolated(
    prompt: str, 
    image_paths: List[str] = None, 
    aspect_ratio: str = "1:1",
    dry_run: bool = False
) -> str:
    """
    ULTRA-STABLE Image Generation. 
    Renamed and simplified to bypass transport errors.
    """
    from qwen_mcp.tools import generate_qwen_image
    return await generate_qwen_image(
        prompt=prompt,
        image_paths=image_paths,
        aspect_ratio=aspect_ratio,
        dry_run=dry_run
    )


@mcp.tool()
async def qwen_heal_registry() -> str:
    """
    ⚡ SELF-HEALING: Analyzes available models and maps them to roles based on ROI and SOTA status.
    Use this if tools report 'Model not found' or if you want to upgrade to the latest versions.
    """
    from qwen_mcp.api import DashScopeClient
    client = DashScopeClient()
    return await client.heal_registry()


if __name__ == "__main__":
    main()
