from typing import Optional, List, Dict, Any, Union
from qwen_mcp.tools import add_task_to_backlog
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
    generate_sos_sync,
    qwen_list_tasks,
    qwen_get_task,
    qwen_update_task,
)
from qwen_mcp.diff_audit import (
    qwen_diff_audit,
    qwen_diff_audit_staged,
    qwen_create_baseline,
    qwen_compare_snapshots,
    qwen_audit_history,
)
from qwen_mcp.specter.telemetry import get_broadcaster
from qwen_mcp.specter.identity import get_current_project_id, get_session_id, get_or_create_instance_id
from qwen_mcp.registry import registry
import asyncio
import logging
import sys
import threading
import socket
import os
import hashlib
import uvicorn
from fastapi import FastAPI, WebSocket

logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("Qwen MCP Server (DashScope)")

async def _auto_init_request(ctx: Context = None, project_id: str = "default") -> None:
    """
    Automatically initialize request state for HUD telemetry.
    
    This eliminates the dependency on agent awareness by ensuring
    every tool call starts with a properly initialized request state.
    
    Args:
        ctx: MCP context for session ID extraction
        project_id: Optional explicit project ID (uses ctx if not provided)
    """
    broadcaster = get_broadcaster()
    if project_id == "default" and ctx is not None:
        project_id = _get_tool_session_id(ctx, default_source="auto")
    await broadcaster.start_request(project_id=project_id)


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
    await _auto_init_request(ctx, "audit")
    project_id = _get_tool_session_id(ctx, default_source="audit")
    
    # Report progress FIRST (before any broadcast_state to avoid double-response)
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Starting code audit..."
        )
    
    # Broadcast state AFTER report_progress (telemetry only, for Roo Code HUD visibility)
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("audit"),
        "role_mapping": registry.models,
        "is_live": True,
        "operation": "Code audit in progress..."
    }, project_id=project_id)
    
    return await generate_audit(content, context, ctx, use_swarm=use_swarm, project_id=project_id)

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
    await _auto_init_request(ctx, "coder")
    project_id = _get_tool_session_id(ctx, default_source="coder")
    
    # Extract workspace_root from MCP context
    workspace_root = None
    if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
        session = ctx.request_context.session
        workspace_uri = getattr(session, 'root_uri', None)
        if workspace_uri and workspace_uri.startswith('file:///'):
            workspace_root = workspace_uri[8:]
    
    # Report progress FIRST (before any broadcast_state to avoid double-response)
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Starting code generation..."
        )
    
    # Broadcast state AFTER report_progress (telemetry only, for Roo Code HUD visibility)
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("coder"),
        "role_mapping": registry.models,
        "is_live": True,
        "operation": "Code generation in progress..."
    }, project_id=project_id)
    
    return await generate_code_unified(prompt, mode, context, ctx, project_id=project_id, workspace_root=workspace_root)

@mcp.tool()
async def qwen_architect(
    goal: str,
    context: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Initiates 'The Lachman Protocol' (LP).
    
    LP is a multi-expert blueprinting system that generates a comprehensive
    implementation plan, manifest, and swarm tasks for complex features.
    
    Args:
        goal: The architectural goal or feature to plan
        context: Additional context or codebase reference
        ctx: MCP context for progress reporting
    
    GREENFIELD OUTPUT:
    - Full LP blueprint with swarm_tasks, manifest, roadmap
    """
    await _auto_init_request(ctx, "architect")
    project_id = _get_tool_session_id(ctx, default_source="architect")
    
    # Report progress FIRST (before any broadcast_state to avoid double-response)
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Initiating Lachman Protocol..."
        )
    
    # Broadcast state AFTER report_progress (telemetry only, not MCP response)
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("strategist"),
        "role_mapping": registry.models,
        "is_live": True,
        "operation": "Architectural planning in progress..."
    }, project_id=project_id)
    
    from qwen_mcp.tools import generate_lp_blueprint
    # Add timeout wrapper to prevent indefinite hangs
    try:
        return await asyncio.wait_for(
            generate_lp_blueprint(goal, context, ctx),
            timeout=240.0  # 4 minute timeout for architecture planning
        )
    except asyncio.TimeoutError:
        logger.error(f"qwen_architect timed out after 240s for goal: {goal}")
        return "❌ Error: Architect request timed out after 4 minutes. The task may be too complex or the API is unresponsive."

@mcp.tool()
async def qwen_sparring(
    mode: str = "sparring2",
    topic: str = "",
    context: str = "",
    session_id: str = "",
    ctx: Context = None
) -> str:
    """
    Sparring session for intellectual debate and strategic analysis.
    
    MODES:
    - sparring1: Quick 2-step analysis (flash mode)
    - sparring2 (default): Full sparring session with Red/Blue/White cell analysis
    - sparring3: Step-by-step with checkpointing (pro mode)
    - discovery: Initial topic exploration
    - red: Critical analysis and counter-arguments
    - blue: Defense and supporting arguments
    - white: Synthesis and conclusions
    
    Args:
        mode: Sparring mode (default: sparring2). Also accepts: sparring1, sparring3, flash, full, pro
        topic: The topic or thesis to debate
        context: Additional context or background
        session_id: Session identifier for continuity
        ctx: MCP context for progress reporting
    
    Returns:
        Sparring session transcript with analysis
    """
    await _auto_init_request(ctx, "sparring")
    project_id = _get_tool_session_id(ctx, default_source="sparring")
    
    # Extract workspace_root from MCP context
    workspace_root = None
    if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
        session = ctx.request_context.session
        workspace_uri = getattr(session, 'root_uri', None)
        if workspace_uri and workspace_uri.startswith('file:///'):
            workspace_root = workspace_uri[8:]
    
    # Report progress FIRST (before any broadcast_state to avoid double-response)
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message=f"Starting sparring session (mode: {mode})..."
        )
    
    # Broadcast state AFTER report_progress (telemetry only, for Roo Code HUD visibility)
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("strategist"),
        "role_mapping": registry.models,
        "is_live": True,
        "operation": f"Sparring session in progress (mode: {mode})..."
    }, project_id=project_id)
    
    return await generate_sparring(topic, context, mode, session_id, ctx, project_id=project_id, workspace_root=workspace_root)

@mcp.tool()
async def qwen_swarm(
    prompt: str,
    task_type: str = "general",
    ctx: Context = None
) -> str:
    """
    Swarm orchestrator for parallel multi-agent analysis.
    
    Automatically decomposes complex tasks into parallel subtasks
    executed by specialized agents, then synthesizes results.
    
    Args:
        prompt: The task or question for swarm analysis
        task_type: Type of task (general, audit, research, etc.)
        ctx: MCP context for progress reporting
    
    Returns:
        Synthesized swarm analysis report
    """
    await _auto_init_request(ctx, "swarm")
    project_id = _get_tool_session_id(ctx, default_source="swarm")
    
    # Report progress FIRST (before any broadcast_state to avoid double-response)
    if ctx:
        await ctx.report_progress(
            progress=0, total=None, message="Starting swarm analysis..."
        )
    
    # Broadcast state AFTER report_progress (telemetry only, for Roo Code HUD visibility)
    await get_broadcaster().broadcast_state({
        "active_model": registry.get_best_model("strategist"),
        "role_mapping": registry.models,
        "is_live": True,
        "operation": "Swarm analysis in progress..."
    }, project_id=project_id)
    
    return await generate_swarm(prompt, task_type, ctx, project_id=project_id)

@mcp.tool()
async def qwen_sync_state(
    apply: bool = False,
    decision_id: str = None,
    apply_all: bool = False,
    workspace_root: str = ".",
    ctx: Context = None
) -> str:
    """
    SOS (State of Session) synchronization tool.
    
    Synchronizes decision log state with filesystem.
    Use to materialize pending decisions or review session history.
    
    Args:
        apply: Apply pending decisions to filesystem
        decision_id: Specific decision ID to process
        apply_all: Apply all pending decisions
        workspace_root: Path to workspace root
    
    Returns:
        Synchronization status and results
    """
    await _auto_init_request(ctx, "sos_sync")
    project_id = _get_tool_session_id(ctx, default_source="sos_sync")
    
    # Report progress FIRST
    if ctx:
        await ctx.report_progress(progress=0, total=None, message="Synchronizing SOS state...")
    
    # Broadcast for UI visibility
    await get_broadcaster().broadcast_state({
        "operation": "SOS synchronization in progress...",
        "progress": 0.0,
        "is_live": True
    }, project_id=project_id)
    
    return await generate_sos_sync(apply, decision_id, apply_all, workspace_root)


@mcp.tool()
async def qwen_add_task(
    task_name: str,
    advice: str,
    workspace_root: str = ".",
    session_id: str = "sos_manual",
    decision_type: str = "manual_task",
    complexity: str = "medium",
    tags: Optional[List[str]] = None,
    risk_score: float = 0.0,
    ctx: Context = None
) -> str:
    """
    Add a new task from natural language to BACKLOG.md and decision_log.parquet.
    
    This is the "Files → Parquet" direction of SOS sync.
    User says "dodaj do backloga" → Agent creates task in BACKLOG.md + Parquet.
    
    Args:
        task_name: Human-readable task name (e.g., "Naprawić sparring3")
        advice: The agentic advice/recommendation
        workspace_root: Path to workspace root (default: ".")
        session_id: Session identifier (default: "sos_manual")
        decision_type: Type of decision (default: "manual_task")
        complexity: Task complexity (default: "medium")
        tags: Optional tags list
        risk_score: Risk assessment (default: 0.0)
        ctx: MCP context for progress reporting
    
    Returns:
        Confirmation message with decision_id
    """
    await _auto_init_request(ctx, "add_task")
    project_id = _get_tool_session_id(ctx, default_source="add_task")
    
    # Report progress FIRST
    if ctx:
        await ctx.report_progress(progress=0, total=None, message=f"Adding task to backlog: {task_name}...")
    
    # Broadcast for UI visibility
    await get_broadcaster().broadcast_state({
        "operation": "Adding task to backlog...",
        "progress": 0.0,
        "is_live": True
    }, project_id=project_id)
    
    return await add_task_to_backlog(
        task_name=task_name,
        advice=advice,
        workspace_root=workspace_root,
        session_id=session_id,
        decision_type=decision_type,
        complexity=complexity,
        tags=tags,
        risk_score=risk_score
    )


@mcp.tool()
async def qwen_add_tasks(
    tasks: List[Dict[str, Any]],
    workspace_root: str = ".",
    session_id: str = "sos_manual",
    decision_type: str = "manual_task",
    chunk_size: int = 20,
    ctx: Context = None
) -> str:
    """
    Add multiple tasks from natural language to BACKLOG.md and decision_log.parquet.
    
    This is the batch version of qwen_add_task for handling large task lists.
    Processes tasks in chunks to avoid MCP timeout.
    
    Args:
        tasks: List of task dictionaries with keys:
            - task_name (required): Human-readable task name
            - advice (required): The agentic advice/recommendation
            - complexity (optional): Task complexity (default: "medium")
            - tags (optional): Tags list
            - risk_score (optional): Risk assessment (default: 0.0)
        workspace_root: Path to workspace root (default: ".")
        session_id: Session identifier (default: "sos_manual")
        decision_type: Type of decision (default: "manual_task")
        chunk_size: Number of tasks per batch write (default: 20)
        ctx: MCP context for progress reporting
    
    Returns:
        Confirmation message with count of added tasks
    """
    await _auto_init_request(ctx, "add_tasks")
    project_id = _get_tool_session_id(ctx, default_source="add_tasks")
    
    # Report progress FIRST
    if ctx:
        await ctx.report_progress(
            progress=0,
            total=len(tasks) if tasks else 0,
            message=f"Adding {len(tasks) if tasks else 0} tasks to backlog..."
        )
    
    # Broadcast for UI visibility
    await get_broadcaster().broadcast_state({
        "operation": f"Adding {len(tasks) if tasks else 0} tasks to backlog...",
        "progress": 0.0,
        "is_live": True
    }, project_id=project_id)
    
    from qwen_mcp.tools import add_tasks_to_backlog_batch
    
    result = await add_tasks_to_backlog_batch(
        tasks=tasks,
        workspace_root=workspace_root,
        session_id=session_id,
        decision_type=decision_type,
        chunk_size=chunk_size
    )
    
    # Update progress on completion
    if ctx:
        await ctx.report_progress(
            progress=len(tasks) if tasks else 0,
            total=len(tasks) if tasks else 0,
            message="Tasks added successfully"
        )
    
    await get_broadcaster().broadcast_state({
        "operation": "Tasks added to backlog",
        "progress": 100.0,
        "is_live": False
    }, project_id=project_id)
    
    return result


@mcp.tool()
async def qwen_refresh_models(ctx: Context = None) -> str:
    """
    🔄 REFRESH MODELS: Syncs model registry with DashScope and HuggingFace.
    Updates available models and refreshes role mappings based on latest SOTA.
    
    Returns:
        Status of registry refresh from both sources
    """
    await _auto_init_request(ctx, "refresh")
    project_id = _get_tool_session_id(ctx, default_source="refresh")
    
    if ctx:
        await ctx.report_progress(progress=0, total=None, message="Refreshing model registry...")
    
    await get_broadcaster().broadcast_state({
        "operation": "Refreshing model registry...",
        "progress": 0.0,
        "is_live": True
    }, project_id=project_id)
    
    from qwen_mcp.api import DashScopeClient
    client = DashScopeClient()
    ds_res = await client.refresh_registry()
    hf_res = await registry.sync_with_hf()
    return f"DashScope: {ds_res}\nHugging Face: {hf_res}"

@mcp.tool()
async def qwen_list_available_models() -> str:
    """
    📋 LIST MODELS: Returns all available models from the registry.
    Shows model IDs, capabilities, and current role assignments.
    
    Returns:
        JSON-formatted list of available models
    """
    return await list_available_models()

@mcp.tool()
async def qwen_set_model(
    role: str,
    model_id: str
) -> str:
    """
    ⚙️ SET MODEL: Manually assigns a model to a specific role.
    Use to override automatic role-to-model mapping.
    
    Args:
        role: The role to assign (e.g., coder, strategist, audit)
        model_id: The model ID to assign (e.g., qwen3.5-plus)
    
    Returns:
        Confirmation of role assignment
    """
    return await set_model_in_registry(role, model_id)

@mcp.tool()
async def qwen_set_billing_mode(
    mode: str
) -> str:
    """
    💰 SET BILLING MODE: Switches between billing configurations.
    
    MODES:
    - coding_plan: Use Coding Plan API (flat monthly fee)
    - payg: Use PAYG API (pay-per-token)
    - hybrid: Intelligent routing between both
    
    Args:
        mode: Billing mode to activate
    
    Returns:
        Confirmation of mode change
    """
    return await set_billing_mode(mode)

@mcp.tool()
async def qwen_get_billing_mode() -> str:
    """
    💰 GET BILLING MODE: Returns current billing configuration.
    
    Returns:
        Current billing mode (coding_plan, payg, or hybrid)
    """
    return await get_current_billing_mode()

@mcp.tool()
async def qwen_read_file(
    path: str
) -> str:
    """
    📄 READ FILE: Reads content from a file in the workspace.
    
    Args:
        path: Relative or absolute path to the file
    
    Returns:
        File contents as string
    """
    return await read_repo_file(path)

@mcp.tool()
async def qwen_list_files(
    directory: str = ".",
    pattern: str = "**/*"
) -> str:
    """
    📁 LIST FILES: Lists files in a directory matching a pattern.
    
    Args:
        directory: Directory to search (default: current directory)
        pattern: Glob pattern for filtering (default: **/*)
    
    Returns:
        Newline-separated list of file paths
    """
    return await list_repo_files(directory, pattern)

@mcp.tool()
async def qwen_usage_report() -> str:
    """
    📊 USAGE REPORT: Generates token usage and cost summary.
    
    Returns:
        Usage statistics and cost breakdown
    """
    return await generate_usage_report()

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
    return f"OK: {project_id}"

@mcp.tool()
async def qwen_heal_registry(ctx: Context = None) -> str:
    """
    ⚡ SELF-HEALING: Analyzes available models and maps them to roles based on ROI and SOTA status.
    Use this if tools report 'Model not found' or if you want to upgrade to the latest versions.
    """
    await _auto_init_request(ctx, "heal")
    project_id = _get_tool_session_id(ctx, default_source="heal")
    
    if ctx:
        await ctx.report_progress(progress=0, total=None, message="Healing model registry...")
    
    await get_broadcaster().broadcast_state({
        "operation": "Self-healing registry...",
        "progress": 0.0,
        "is_live": True
    }, project_id=project_id)
    
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
    await _auto_init_request(ctx, "context_init")
    project_id = _get_tool_session_id(ctx, default_source="context_init")
    
    if ctx:
        await ctx.report_progress(progress=0, total=None, message="Initializing project context...")
    
    await get_broadcaster().broadcast_state({
        "operation": "Initializing project context...",
        "progress": 0.0,
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
    await _auto_init_request(ctx, "session_context")
    project_id = _get_tool_session_id(ctx, default_source="session_context")
    
    if ctx:
        await ctx.report_progress(progress=0, total=None, message="Updating session context...")
    
    await get_broadcaster().broadcast_state({
        "operation": "Updating session context...",
        "progress": 0.0,
        "is_live": True
    }, project_id=project_id)
    
    return await qwen_update_session_context(session_summary, workspace_root, ctx)

@mcp.tool()
async def qwen_list_tasks_tool(
    status: str = "pending",
    tags: str = None,
    workspace_root: str = "."
) -> str:
    """
    List tasks from BACKLOG.md with optional filtering.
    
    Args:
        status: Filter by status - "pending", "completed", or "all" (default: "pending")
        tags: Optional comma-separated list of tags to filter by
        workspace_root: Path to workspace root (default: current directory)
        
    Returns:
        Formatted list of tasks with their details
    """
    tags_list = [t.strip() for t in tags.split(",")] if tags else None
    return await qwen_list_tasks(status=status, tags=tags_list, workspace_root=workspace_root)

@mcp.tool()
async def qwen_get_task_tool(
    decision_id: str,
    workspace_root: str = "."
) -> str:
    """
    Get detailed information about a specific task by decision_id.
    
    Args:
        decision_id: The unique identifier of the task
        workspace_root: Path to workspace root (default: current directory)
        
    Returns:
        Detailed task information from decision_log.parquet
    """
    return await qwen_get_task(decision_id=decision_id, workspace_root=workspace_root)

@mcp.tool()
async def qwen_update_task_tool(
    decision_id: str,
    new_status: str,
    workspace_root: str = "."
) -> str:
    """
    Update the status of a task in both BACKLOG.md and decision_log.parquet.
    
    Args:
        decision_id: The unique identifier of the task
        new_status: New status - "pending", "in_progress", or "completed"
        workspace_root: Path to workspace root (default: current directory)
        
    Returns:
        Confirmation message with updated task details
    """
    return await qwen_update_task(decision_id=decision_id, new_status=new_status, workspace_root=workspace_root)

@mcp.tool()
async def qwen_diff_audit_tool(
    from_ref: str = "HEAD~1",
    to_ref: str = "HEAD",
    baseline_snapshot: str = None,
    shadow_mode: bool = False,
    workspace_root: str = "."
) -> str:
    """
    Audit git diff for potential regressions using Anti-Degradation System.
    
    Args:
        from_ref: Source ref (commit, branch, or HEAD~N)
        to_ref: Target ref
        baseline_snapshot: Name of baseline snapshot (default: "latest")
        shadow_mode: If True, warnings only - no blocking
        workspace_root: Path to workspace root
        
    Returns:
        Audit result with regression detection and risk assessment
    """
    import json
    result = await qwen_diff_audit(
        from_ref=from_ref,
        to_ref=to_ref,
        baseline_snapshot=baseline_snapshot,
        shadow_mode=shadow_mode,
        workspace_root=workspace_root
    )
    return json.dumps(result, indent=2)

@mcp.tool()
async def qwen_diff_audit_staged_tool(
    baseline_snapshot: str = None,
    shadow_mode: bool = False,
    workspace_root: str = "."
) -> str:
    """
    Audit staged changes for potential regressions (for pre-commit hook).
    
    Args:
        baseline_snapshot: Name of baseline snapshot (default: "latest")
        shadow_mode: If True, warnings only - no blocking
        workspace_root: Path to workspace root
        
    Returns:
        Audit result with blocking decision
    """
    import json
    result = await qwen_diff_audit_staged(
        baseline_snapshot=baseline_snapshot,
        shadow_mode=shadow_mode,
        workspace_root=workspace_root
    )
    return json.dumps(result, indent=2)

@mcp.tool()
async def qwen_create_baseline_tool(
    name: Optional[str] = "auto",
    workspace_root: str = "."
) -> str:
    """
    Create a new baseline snapshot for Anti-Degradation System.
    
    Args:
        name: Snapshot name. If "auto" or None, generates timestamped name (baseline-YYYYMMDD_HHMMSS).
        workspace_root: Path to workspace root
        
    Returns:
        Path to saved snapshot file
    """
    return await qwen_create_baseline(name=name, workspace_root=workspace_root)

@mcp.tool()
async def qwen_compare_snapshots_tool(
    snapshot1_name: Optional[str] = "auto",
    snapshot2_name: Optional[str] = "auto",
    workspace_root: str = "."
) -> str:
    """
    Compare two snapshots for regression detection.
    
    Args:
        snapshot1_name: First snapshot name. If "auto", selects newest snapshot.
        snapshot2_name: Second snapshot name. If "auto", selects second newest snapshot.
        workspace_root: Path to workspace root
        
    Returns:
        Comparison result with regression alerts
    """
    import json
    result = await qwen_compare_snapshots(
        snapshot1_name=snapshot1_name,
        snapshot2_name=snapshot2_name,
        workspace_root=workspace_root
    )
    return json.dumps(result, indent=2)

@mcp.tool()
async def qwen_audit_history_tool(
    limit: int = 100,
    workspace_root: str = "."
) -> str:
    """
    Get recent audit history from Anti-Degradation System.
    
    Args:
        limit: Maximum number of audits to return
        workspace_root: Path to workspace root
        
    Returns:
        List of recent audit results
    """
    import json
    result = await qwen_audit_history(limit=limit, workspace_root=workspace_root)
    return json.dumps(result, indent=2)

def main():
    mcp.run()

if __name__ == "__main__":
    main()
