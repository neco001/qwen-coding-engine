from typing import Optional, List, Dict, Any, Union
from mcp.server.fastmcp import FastMCP, Context
from qwen_mcp.tools import (
    generate_audit,
    generate_code_unified,
    generate_lp_blueprint,
    read_repo_file,
    list_repo_files,
    generate_usage_report,
    list_available_models,
    set_model_in_registry,
    generate_sparring,
    generate_swarm,
    heal_registry,
    set_billing_mode,
    get_current_billing_mode,
    qwen_init_context,
    qwen_update_session_context,
)
from qwen_mcp.specter.telemetry import get_broadcaster
from qwen_mcp.specter.identity import get_current_project_id, get_session_id, get_or_create_instance_id
from qwen_mcp.registry import registry
import asyncio
import sys
import threading
import socket
import os
import uvicorn
from fastapi import FastAPI, WebSocket

# Global flag to track if telemetry server is already running
_TELEMETRY_SERVER_RUNNING = False

# Force UTF-8 encoding for stdout/stderr on Windows to prevent 'krzaki'
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

# Initialize FastMCP Server
mcp = FastMCP("Qwen MCP Server (DashScope)")


def _get_tool_session_id(ctx: Context = None, default_source: str = "mcp") -> str:
    """
    Extract client_source from MCP context and generate proper session ID.
    
    This ensures Gemini and Roo Code sessions are properly isolated.
    
    Args:
        ctx: MCP context (may contain X-Client-Source header)
        default_source: Fallback client source if not detected
    
    Returns:
        Session ID in format: {instanceId}_{clientSource}_{workspaceHash}
    """
    client_source = default_source
    
    # Try to extract client_source from MCP context headers
    if ctx is not None:
        try:
            if hasattr(ctx, 'request_context') and ctx.request_context:
                headers = getattr(ctx.request_context, 'headers', {})
                if headers:
                    client_source = headers.get('X-Client-Source', default_source)
        except (AttributeError, KeyError):
            pass
    
    instance_id = get_or_create_instance_id()
    cwd = os.getcwd()
    return get_session_id(instance_id, client_source, cwd)


@mcp.tool()
async def qwen_audit(
    content: str,
    context: Optional[str] = None,
    use_swarm: bool = True,
    ctx: Context = None
) -> str:
    """
    Audits the provided code or terminal logs using Qwen models.
    
    For multi-file content, automatically uses Swarm for parallel analysis.
    Set use_swarm=False to disable parallel processing.
    
    Args:
        content: The code or logs to audit
        context: Additional context for the audit
        use_swarm: Enable automatic parallel file analysis (default: True)
        ctx: MCP context for progress reporting
    
    Returns:
        Audit report with findings and recommendations
    """
    project_id = _get_tool_session_id(ctx, default_source="audit")
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("strategist"),
        "role_mapping": registry.models,
        "is_live": True
    }, project_id=project_id)
    return await generate_audit(content, context, ctx, use_swarm=use_swarm)


@mcp.tool()
async def qwen_coder(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Unified code generation tool with mode-based routing.
    
    MODES:
    - auto: Intelligent routing based on prompt complexity (default)
    - standard: Fast generation using qwen3-coder-next
    - pro: Heavy-duty generation using qwen3-coder-plus
    - expert: Maximum capability for complex refactors/architecture
    
    EXAMPLES:
    1. Simple task: qwen_coder(prompt="Write a function to add two numbers")
    2. Complex task: qwen_coder(prompt="...", mode="pro")
    3. Expert refactor: qwen_coder(prompt="...", mode="expert")
    
    DEPRECATED TOOLS (still available but use unified internally):
    - qwen_coder (old) → now calls qwen_coder(mode="standard")
    - qwen_coder_pro (old) → now calls qwen_coder(mode="pro")
    """
    project_id = _get_tool_session_id(ctx, default_source="coder")
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("coding"),
        "role_mapping": registry.models,
        "is_live": True
    }, project_id=project_id)
    return await generate_code_unified(prompt, mode, context, ctx)


@mcp.tool()
async def qwen_architect(
    goal: str, context: Optional[str] = None, ctx: Context = None
) -> str:
    """
    Initiates 'The Lachman Protocol' (LP).
    The server hires a dynamic expert squad to audit your goal and generate a high-precision Blueprint.
    """
    project_id = _get_tool_session_id(ctx, default_source="architect")
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("strategist"),
        "role_mapping": registry.models,
        "is_live": True
    }, project_id=project_id)
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Initiating Lachman Protocol..."
        )
    return await generate_lp_blueprint(goal, context, ctx)


@mcp.tool()
async def qwen_sparring(
    mode: str = "flash",
    topic: str = "",
    context: str = "",
    session_id: str = "",
    ctx: Context = None
) -> str:
    """
    Adwersarialna analiza z sesyjnym checkpointingiem.
    
    MODES:
    - flash: Szybka analiza + draft (pojedyncze wywołanie, bez sesji)
    - discovery: Utwórz sesję + zdefiniuj role (zwraca session_id)
    - red: Krytyka Red Cell (wymaga session_id z discovery)
    - blue: Obrona Blue Cell (wymaga session_id + red)
    - white: Synteza White Cell (wymaga session_id + red + blue)
    - full: Cała sesja (discovery→red→blue→white) w jednym wywołaniu
    
    EXAMPLES:
    1. qwen_sparring(mode="flash", topic="Czy użyć mikroserwisów?")
    2. qwen_sparring(mode="discovery", topic="Strategia migracji")
    3. qwen_sparring(mode="red", session_id="sp_abc123")
    4. qwen_sparring(mode="blue", session_id="sp_abc123")
    5. qwen_sparring(mode="white", session_id="sp_abc123")
    6. qwen_sparring(mode="full", topic="Decyzja architektoniczna")
    """
    project_id = _get_tool_session_id(ctx, default_source="sparring")
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("strategist"),
        "role_mapping": registry.models,
        "is_live": True
    }, project_id=project_id)
    return await generate_sparring(topic, context, mode, session_id, ctx)


@mcp.tool()
async def qwen_swarm(
    prompt: str,
    task_type: str = "general",
    ctx: Context = None
) -> str:
    """
    Executes the Swarm Orchestrator for parallel task decomposition and execution.
    
    The Swarm decomposes complex prompts into atomic sub-tasks, executes them
    in parallel, and synthesizes the results into a coherent response.
    
    Use cases:
    - Multi-file analysis (decompose into per-file tasks)
    - Multi-expert review (QA, Security, ROI in parallel)
    - Complex implementations (analyze, plan, implement phases)
    
    Args:
        prompt: The complex prompt to decompose and execute
        task_type: Type hint for decomposition (e.g., "coding", "audit", "general")
        ctx: MCP context for progress reporting
    
    Returns:
        Synthesized response from all parallel sub-tasks
    """
    project_id = _get_tool_session_id(ctx, default_source="swarm")
    await get_broadcaster().broadcast_state({
        "active_model": "swarm-orchestrator",
        "role_mapping": {"swarm": "parallel"},
        "is_live": True
    }, project_id=project_id)
    return await generate_swarm(prompt, task_type, ctx)


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
    Roles: 'strategist' (audit/LP), 'coder' (qwen_coder_pro), 'scout' (internal/discovery).
    """
    return await set_model_in_registry(role, model_id)


@mcp.tool()
async def qwen_set_billing_mode(mode: str) -> str:
    """
    Dynamically switches the billing mode of the Qwen Engine.
    Valid modes: 'coding_plan' (Strict Plan), 'payg' (Strict Pay-As-You-Go), 'hybrid' (Plan preferred, PAYG fallback).
    """
    return await set_billing_mode(mode)


@mcp.tool()
async def qwen_get_billing_mode() -> str:
    """
    Returns the currently active billing mode ('coding_plan', 'payg', or 'hybrid').
    """
    return await get_current_billing_mode()


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


def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is already in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True

def run_telemetry_server():
    """Starts a lightweight FastAPI server for the HUD telemetry."""
    global _TELEMETRY_SERVER_RUNNING
    
    # Check if port is already in use (another MCP instance is running)
    if is_port_in_use(8878):
        print("ℹ️ [SPECTER] Telemetry server already running on port 8878, skipping startup")
        _TELEMETRY_SERVER_RUNNING = True
        return
    
    if _TELEMETRY_SERVER_RUNNING:
        print("ℹ️ [SPECTER] Telemetry server already started in this process")
        return
    
    app = FastAPI()
    broadcaster = get_broadcaster()

    @app.get("/")
    async def root():
        project_id = get_current_project_id()
        return {
            "status": "SPECTER LENS SIDECAR ACTIVE", 
            "project_id": project_id,
            "uplink": f"ws://127.0.0.1:8878/ws/telemetry?project_id={project_id}"
        }

    @app.websocket("/ws/telemetry")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        # FIX: Extract project_id from query string manually (FastAPI doesn't auto-parse WebSocket query params)
        project_id = websocket.query_params.get("project_id", "default")
        await broadcaster.add_client(websocket, project_id=project_id)
        try:
            while True:
                # Keep alive and signal handling
                try:
                    # Non-blocking check for messages + heartbeat
                    await asyncio.wait_for(websocket.receive_text(), timeout=10)
                except asyncio.TimeoutError:
                    # Optional: send project-specific heartbeat if needed
                    await websocket.send_json({"type": "heartbeat", "project_id": project_id})
        except Exception:
            pass
        finally:
            await broadcaster.remove_client(websocket)

    # Run on fixed port 8878
    try:
        config = uvicorn.Config(app, host="127.0.0.1", port=8878, log_level="warning")
        server = uvicorn.Server(config)
        _TELEMETRY_SERVER_RUNNING = True
        server.run()
    except Exception as e:
        print(f"❌ Telemetry Sidecar failed: {e}")
        _TELEMETRY_SERVER_RUNNING = False

async def sync_hud_state():
    """Broadcaster update for role mapping and basic state."""
    await asyncio.sleep(2) # Wait for sidecar thread to start
    await get_broadcaster().broadcast_state({
        "role_mapping": registry.models,
        "is_live": False
    })

def main():
    """Main entrypoint for the MCP server."""
    global _TELEMETRY_SERVER_RUNNING
    
    print("🚀 [SPECTER] Starting Qwen Engineering Engine Context...")
    
    # Check if telemetry server is already running (shared across MCP instances)
    if is_port_in_use(8878):
        print("ℹ️ [SPECTER] Connecting to existing telemetry server on port 8878")
        _TELEMETRY_SERVER_RUNNING = True
    else:
        print("📡 [SPECTER] Starting telemetry sidecar on port 8878/ws")
        # Start telemetry in dedicated thread
        sidecar = threading.Thread(target=run_telemetry_server, daemon=True)
        sidecar.start()
        # Wait briefly for server to start
        threading.Event().wait(0.5)
    
    # Schedule initial state sync
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(sync_hud_state())
    except Exception:
        pass

    mcp.run()


@mcp.tool()
async def qwen_init_request(ctx: Context = None) -> str:
    """
    ⚡ INITIALIZE NEW REQUEST: Resets 'This Prompt' token counter and buffers in the HUD.
    MANDATORY: Call this as your FIRST tool at the start of EVERY new user prompt/task.
    Failure to call this results in incorrect token accumulation in telemetry.
    """
    broadcaster = get_broadcaster()
    project_id = _get_tool_session_id(ctx, default_source="init")
    await broadcaster.start_request(project_id=project_id)
    return f"✅ Specter HUD: 'This Prompt' counters reset for {project_id}. Ready for new engagement."


@mcp.tool()
async def qwen_heal_registry() -> str:
    """
    ⚡ SELF-HEALING: Analyzes available models and maps them to roles based on ROI and SOTA status.
    Use this if tools report 'Model not found' or if you want to upgrade to the latest versions.
    """
    from qwen_mcp.api import DashScopeClient
    client = DashScopeClient()
    return await client.heal_registry()


@mcp.tool()
async def qwen_init_context_tool(
    workspace_root: str = ".",
    ctx: Context = None
) -> str:
    """
    Initialize project context files using Swarm analysis.
    
    Generates:
    - .context/_PROJECT_CONTEXT.md: Tech stack, structure, conventions
    - .context/_DATA_CONTEXT.md: Data sources, schemas, pipelines
    
    Uses Swarm Orchestrator for parallel analysis of:
    1. Tech Stack (runtime, frameworks, libraries)
    2. Structure Mapping (directories, entry points, configs)
    3. Data Sources (databases, files, APIs)
    4. Documentation (conventions, workflows)
    
    Args:
        workspace_root: Path to workspace root (default: current directory)
        ctx: MCP context for progress reporting
    
    Returns:
        Summary of generated files with paths
    
    Example:
        qwen_init_context_tool(workspace_root=".")
    """
    project_id = _get_tool_session_id(ctx, default_source="context_init")
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("scout"),
        "role_mapping": registry.models,
        "is_live": True
    }, project_id=project_id)
    return await qwen_init_context(workspace_root, ctx)


@mcp.tool()
async def qwen_update_session_context_tool(
    session_summary: str,
    workspace_root: str = ".",
    ctx: Context = None
) -> str:
    """
    Update session supplement with current session insights.
    
    Call this at the END of each session to capture:
    - Key decisions made
    - Changes implemented
    - Open questions
    - Recommendations for next session
    
    Args:
        session_summary: Summary of work done in this session
        workspace_root: Path to workspace root
        ctx: MCP context for progress reporting
    
    Returns:
        Confirmation of update with session highlights
    
    Example:
        qwen_update_session_context_tool(
            session_summary="Implemented user auth with JWT",
            workspace_root="."
        )
    """
    project_id = _get_tool_session_id(ctx, default_source="context_update")
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("scout"),
        "role_mapping": registry.models,
        "is_live": True
    }, project_id=project_id)
    return await qwen_update_session_context(session_summary, workspace_root, ctx)


if __name__ == "__main__":
    main()
