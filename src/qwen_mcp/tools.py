import logging
import json
import re
import os
import glob
import pandas as pd
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import Context
from qwen_mcp.api import DashScopeClient
from qwen_mcp.base import set_billing_mode, get_billing_mode as get_current_billing_mode
from qwen_mcp.registry import registry
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.specter.telemetry import get_broadcaster
from qwen_mcp.config.sos_paths import DEFAULT_SOS_PATHS

# Re-export billing mode functions for server.py
__all__ = [
    'set_billing_mode',
    'get_current_billing_mode',
    'generate_audit',
    'generate_code_unified',
    'generate_lp_blueprint',
    'generate_sparring',
    'generate_swarm',
    'generate_sos_sync',
    'generate_usage_report',
    'heal_registry',
    'list_available_models',
    'list_repo_files',
    'qwen_init_context',
    'qwen_update_session_context',
    'read_repo_file',
    'set_model_in_registry',
    'qwen_list_tasks',
    'qwen_get_task',
    'qwen_update_task',
]

# New modular imports
from qwen_mcp.prompts.system import AUDIT_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT
from qwen_mcp.prompts.lachman import LP_DISCOVERY_PROMPT, LP_ARCHITECT_PROMPT, LP_VERIFIER_PROMPT

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Utility to pull JSON blocks from LLM markdown responses."""
    if not text:
        return None
    try:
        # Try finding a ```json block first
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        if not json_str: return None
        return json.loads(json_str.strip())
    except Exception:
        # Fallback to finding anything that looks like a JSON object
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
    return None

async def generate_lp_blueprint(goal: str, context: Optional[str] = None, ctx: Optional[Context] = None, auto_add_tasks: bool = False) -> str:
    """Generate blueprint with heavy defense against NoneType errors and timeout protection."""
    from qwen_mcp.engines.scout import ScoutEngine
    from qwen_mcp.prompts.lachman import LP_BROWNFIELD_PROMPT
    from qwen_mcp.specter.telemetry import get_broadcaster
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # Extract project_id for telemetry
    project_id = "default"
    if ctx:
        try:
            from mcp.server.fastmcp import Context
            # Extract from ctx if available
            pass
        except Exception:
            pass
    
    # 0. Initial Sizing with heartbeat protection
    try:
        scout_res = await scout.analyze_task(
            goal, context or "", task_hint="strategy/architecture",
            progress_callback=ctx.report_progress if ctx else None
        )
    except Exception as e:
        logger.error(f"Scout failed: {e}")
        scout_res = {"complexity": "high", "is_brownfield": False}
    
    if not isinstance(scout_res, dict):
        scout_res = {"complexity": "high", "is_brownfield": False}
        
    complexity = scout_res.get("complexity", "high")
    is_brownfield = scout_res.get("is_brownfield", False)
    
    if is_brownfield:
        brownfield_msg = [
            {"role": "system", "content": LP_BROWNFIELD_PROMPT},
            {"role": "user", "content": f"Goal: {goal}\nContext: {context or ''}"}
        ]
        # Heartbeat during API wait - both MCP progress and WebSocket broadcast
        if ctx:
            await ctx.report_progress(progress=10, total=None, message="Analyzing brownfield context...")
        # Broadcast via WebSocket for Roo Code HUD visibility
        await get_broadcaster().broadcast_state({
            "operation": "Analyzing brownfield context...",
            "progress": 10.0,
            "is_live": True
        }, project_id=project_id)
        result = await client.generate_completion(
            messages=brownfield_msg,
            task_type="strategist",
            complexity=complexity,
            project_id=project_id,
            timeout=60.0,
            progress_callback=ctx.report_progress if ctx else None
        )
        return f"## 🏗️ Brownfield Analysis\n\n{result}"
    
    # GREENFIELD MODE - Discovery phase with heartbeat
    discovery_msg = [
        {"role": "system", "content": LP_DISCOVERY_PROMPT},
        {"role": "user", "content": f"Goal: {goal}\nContext: {context or ''}"}
    ]
    # Report progress via MCP protocol AND broadcast via WebSocket for UI visibility
    if ctx:
        await ctx.report_progress(progress=25, total=None, message="Discovering squad requirements...")
    await get_broadcaster().broadcast_state({
        "operation": "Discovering squad requirements...",
        "progress": 25.0,
        "is_live": True
    }, project_id=project_id)
    discovery_raw = await client.generate_completion(
        messages=discovery_msg,
        complexity="medium",
        timeout=60.0,
        progress_callback=ctx.report_progress if ctx else None
    )
    discovery = extract_json_from_text(discovery_raw) or {"hired_squad": []}
    
    squad_list = discovery.get("hired_squad", [])
    if not isinstance(squad_list, list):
        squad_list = []
    
    squad_roles = []
    for s in squad_list:
        if isinstance(s, dict):
            squad_roles.append(s.get("role", "Expert"))
        else:
            squad_roles.append("Expert")
    
    squad_str = ", ".join(squad_roles)

    # Architect phase with heartbeat - both MCP progress and WebSocket broadcast
    arch_msg = [
        {"role": "system", "content": LP_ARCHITECT_PROMPT.format(squad=squad_str)},
        {"role": "user", "content": f"Goal: {goal}"}
    ]
    if ctx:
        await ctx.report_progress(progress=50, total=None, message="Generating architecture blueprint...")
    # Broadcast via WebSocket for Roo Code HUD visibility
    await get_broadcaster().broadcast_state({
        "operation": "Generating architecture blueprint...",
        "progress": 50.0,
        "is_live": True
    }, project_id=project_id)
    blueprint_raw = await client.generate_completion(
        messages=arch_msg,
        task_type="strategist",
        complexity=complexity,
        project_id=project_id,
        timeout=120.0,
        progress_callback=ctx.report_progress if ctx else None
    )
    
    blueprint_data = extract_json_from_text(blueprint_raw)
    swarm_tasks = []
    if blueprint_data and isinstance(blueprint_data, dict) and "swarm_tasks" in blueprint_data:
        swarm_tasks = blueprint_data.get("swarm_tasks", [])
        if isinstance(swarm_tasks, list):
            swarm_section = "\n\n## 🎯 Swarm Execution Tasks\n\n"
            for task in swarm_tasks:
                if isinstance(task, dict):
                    task_id = task.get("id", "unknown")
                    task_desc = task.get("task", "")
                    priority = task.get("priority", 5)
                    target_files = task.get("target_files", [])
                    exec_hint = task.get("execution_hint", "qwen_coder")
                    
                    swarm_section += f"### {task_id} (Priority: {priority})\n"
                    swarm_section += f"- **Task**: {task_desc}\n"
                    swarm_section += f"- **Target Files**: {', '.join(target_files) if target_files else 'N/A'}\n"
                    swarm_section += f"- **Execution**: `{exec_hint}`\n\n"
            # Auto-add tasks to backlog if enabled
            if auto_add_tasks:
                try:
                    from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
                    from qwen_mcp.config.sos_paths import DEFAULT_SOS_PATHS
                    from pathlib import Path
                    from urllib.parse import urlparse

                    # Determine workspace root from context
                    workspace_root = str(Path.cwd())
                    if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
                        session = ctx.request_context.session
                        workspace_uri = getattr(session, 'root_uri', None)
                        if workspace_uri:
                            parsed = urlparse(workspace_uri)
                            if parsed.scheme == 'file':
                                workspace_root = Path(parsed.path)

                    # Initialize decision log sync engine
                    decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace_root)
                    backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace_root)
                    sync_engine = DecisionLogSyncEngine(decision_log_path)

                    # Convert swarm tasks to compatible format
                    tasks_to_add = []
                    for task in swarm_tasks:
                        if isinstance(task, dict):
                            tasks_to_add.append({
                                "task_name": task.get("task", f"Unnamed task {task.get('id', 'unknown')}"),
                                "advice": f"Auto-added from LP blueprint: {task.get('task', '')}",
                                "complexity": str(task.get("priority", 5)),
                                "tags": ["auto-generated", "lp-blueprint"],
                                "risk_score": 0.0
                            })

                    # Add tasks to backlog
                    if tasks_to_add:
                        await sync_engine.add_tasks(tasks=tasks_to_add,backlog_path=backlog_path,workspace_root=workspace_root,session_id=project_id,decision_type="auto_task")
                except Exception as e:
                    logger.warning(f"auto_add_tasks failed: {e}. Blueprint returned without tasks.")
            return blueprint_raw + swarm_section
    
    return blueprint_raw

async def heal_registry() -> str:
    from qwen_mcp.api import DashScopeClient
    client = DashScopeClient()
    return await client.heal_registry()

async def qwen_init_context(workspace_root: str, ctx: Optional[Context] = None) -> str:
    """
    Initialize project context files using Swarm analysis.
    
    Generates:
    - .context/_PROJECT_CONTEXT.md: Tech stack, structure, conventions
    - .context/_DATA_CONTEXT.md: Data sources, schemas, pipelines
    """
    from pathlib import Path
    from qwen_mcp.engines.context_builder import ContextBuilderEngine
    
    # Input validation
    if workspace_root and not Path(workspace_root).exists():
        logger.error(f"Invalid workspace_root: {workspace_root} does not exist")
        return f"## Error\n\nWorkspace path does not exist: {workspace_root}"
    
    try:
        if ctx:
            await ctx.report_progress(
                progress=0,
                total=3,
                message="Initializing context builder..."
            )
        
        client = DashScopeClient()
        engine = ContextBuilderEngine(client=client)
        
        if ctx:
            await ctx.report_progress(
                progress=1,
                total=3,
                message="Analyzing tech stack and data sources..."
            )
        
        # Generate both contexts
        project_context, data_context = await engine.generate_project_context(
            workspace_root
        )
        
        if ctx:
            await ctx.report_progress(
                progress=2,
                total=3,
                message="Saving context files..."
            )
        
        # Save files
        saved = engine.save_context_files(
            project_context,
            data_context,
            workspace_root
        )
        
        result = f"## Context Files Generated\n\n"
        result += f"Successfully created {len(saved)} context files:\n\n"
        
        for ctx_type, path in saved.items():
            result += f"- **{ctx_type}**: `{path}`\n"
        
        result += f"\n### Next Steps\n\n"
        result += f"1. Review `_PROJECT_CONTEXT.md` for tech stack summary\n"
        result += f"2. Review `_DATA_CONTEXT.md` for data infrastructure\n"
        result += f"3. Use `qwen_update_session_context` at end of session\n"
        
        return result
        
    except Exception as e:
        logger.error(f"qwen_init_context failed: {e}", exc_info=True)
        return f"## Error\n\nFailed to initialize context: {str(e)}"

async def qwen_update_session_context(summary: str, workspace_root: str, ctx: Optional[Context] = None) -> str:
    """
    Update session supplement with current session insights.
    
    Call this at the END of each session to capture:
    - Key decisions made
    - Changes implemented
    - Open questions
    - Recommendations for next session
    """
    from pathlib import Path
    from qwen_mcp.engines.context_builder import ContextBuilderEngine
    
    # Input validation
    if not summary or not summary.strip():
        logger.error("Empty session_summary provided")
        return "## Error\n\nSession summary cannot be empty"
    
    if workspace_root and not Path(workspace_root).exists():
        logger.error(f"Invalid workspace_root: {workspace_root} does not exist")
        return f"## Error\n\nWorkspace path does not exist: {workspace_root}"
    
    try:
        if ctx:
            await ctx.report_progress(
                progress=0,
                total=2,
                message="Processing session summary..."
            )
        
        client = DashScopeClient()
        engine = ContextBuilderEngine(client=client)
        
        if ctx:
            await ctx.report_progress(
                progress=1,
                total=2,
                message="Updating session supplement..."
            )
        
        # Generate/update session context
        session_content = await engine.update_session_context(
            summary,
            workspace_root
        )
        
        # Save file
        saved_path = engine.save_session_context(
            session_content,
            workspace_root
        )
        
        # Extract key highlights from session summary
        highlights = summary.split("\n")[:5]  # First 5 lines as highlights
        
        result = f"## Session Context Updated\n\n"
        result += f"**File**: `{saved_path}`\n\n"
        result += f"### Session Highlights\n\n"
        for highlight in highlights:
            if highlight.strip():
                result += f"- {highlight.strip()}\n"
        
        result += f"\n### Recommendation\n\n"
        result += f"Review `_SESSION_SUPPLEMENT.md` before next session for continuity.\n"
        
        return result
        
    except Exception as e:
        logger.error(f"qwen_update_session_context failed: {e}", exc_info=True)
        return f"## Error\n\nFailed to update session context: {str(e)}"

async def generate_audit(content: str, context: Optional[str] = None, ctx: Optional[Context] = None, use_swarm: bool = True, project_id: str = "default") -> str:
    client = DashScopeClient()
    
    # Report progress and broadcast for UI visibility
    if ctx:
        await ctx.report_progress(progress=10, total=None, message="Analyzing code for audit...")
    await get_broadcaster().broadcast_state({
        "operation": "Analyzing code for audit...",
        "progress": 10.0,
        "is_live": True
    }, project_id=project_id)
    
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context}\nContent: {content}"}
    ]
    return await client.generate_completion(
        messages=messages,
        task_type="audit",
        project_id=project_id,
        progress_callback=ctx.report_progress if ctx else None
    )

async def generate_code_unified(prompt: str, mode: str = "auto", context: Optional[str] = None, ctx: Optional[Context] = None, project_id: str = "default", workspace_root: Optional[str] = None, require_plan: bool = False, require_test: bool = False) -> str:
    """
    Unified code generation with intelligent routing via CoderEngineV2.
    
    Features:
    - Brownfield detection (diffs vs. full files)
    - Mode-based model routing (standard/pro/expert)
    - Swarm orchestration for complex tasks
    - Context-aware code generation
    - Auto-invokes DecisionLogSync after successful completion
    
    Args:
        prompt: The code generation request
        mode: One of 'auto', 'standard', 'pro', 'expert'
        context: Additional context (existing code, requirements, etc.)
        ctx: MCP context for progress reporting
        project_id: Project/session ID for telemetry isolation
        workspace_root: Path to workspace root (default: current directory)
        
    Returns:
        Generated code or error message
    """
    from qwen_mcp.engines.coder_v2 import CoderEngineV2
    from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
    from pathlib import Path
    
    # Report progress FIRST
    if ctx:
        await ctx.report_progress(progress=0, total=None, message="Starting code generation...")
    
    # Broadcast state for UI visibility
    await get_broadcaster().broadcast_state({
        "operation": "Code generation in progress...",
        "progress": 10.0,
        "is_live": True
    }, project_id=project_id)
    
    # Delegate to CoderEngineV2 for intelligent routing
    engine = CoderEngineV2()
    response = await engine.execute(
        prompt=prompt,
        mode=mode,
        context=context,
        ctx=ctx,
        project_id=project_id,
        require_plan=require_plan,
        require_test=require_test
    )
    
    if response.success:
        # Auto-invoke DecisionLogSync after successful code generation
        try:
            # CRITICAL: Use provided workspace_root, NOT Path.cwd()
            # This ensures files are saved in the CLIENT's project directory, not server directory
            if workspace_root:
                workspace = Path(workspace_root)
            else:
                # Fallback to MCP context's workspace folder if available
                if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
                    session = ctx.request_context.session
                    workspace_uri = getattr(session, 'root_uri', None)
                    if workspace_uri and workspace_uri.startswith('file:///'):
                        workspace = Path(workspace_uri[8:])
                    else:
                        workspace = Path.cwd()
                else:
                    workspace = Path.cwd()
            
            decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace)
            backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace)
            changelog_path = DEFAULT_SOS_PATHS.get_changelog_path(workspace)
            
            sync_engine = DecisionLogSyncEngine(decision_log_path)
            
            # 1. Complete task: mark as done in BACKLOG.md, log to parquet, append to CHANGELOG.md
            await sync_engine.complete_task(
                task_description=prompt[:200],  # Truncate long prompts
                backlog_path=backlog_path,
                changelog_path=changelog_path,
                session_id=project_id,
                tokens_used=0,  # Would need to extract from response if available
                files_changed=[]  # Could extract from response.diff if available
            )
            
            # 2. Also check for pending advices and apply them
            advices = await sync_engine.scan_advices()
            if advices:
                await sync_engine.apply_all_advices(backlog_path, changelog_path)
        except Exception as e:
            logger.warning(f"DecisionLogSync auto-invoke failed: {e}")
        
        return response.to_markdown()
    else:
        return f"❌ Code generation failed: {response.error}\n\n{response.message}"

# Mode aliases: sparring1/2/3 → internal modes
MODE_ALIASES = {
    "sparring1": "flash",    # Quick 2-step analysis
    "sparring2": "full",     # Full session in one call (DEFAULT)
    "sparring3": "pro",      # Step-by-step with checkpointing
    # Short aliases
    "nor": "full",           # "normal" shortcut
    # Legacy aliases (passthrough)
    "flash": "flash",
    "full": "full",
    "pro": "pro",
    "discovery": "discovery",
    "red": "red",
    "blue": "blue",
    "white": "white",
}

def resolve_sparring_mode(mode: str) -> str:
    """
    Resolve sparring mode alias to internal mode name.
    
    Sparring Levels:
    - sparring1 (flash): 2 steps (analyst→drafter), 180s total
    - sparring2 (normal): 4 steps in one call (full), 180s total - DEFAULT
    - sparring3 (pro): 4 steps separately (step-by-step), 100s per step
    
    Args:
        mode: User-provided mode (sparring1, sparring2, sparring3, flash, full, pro, nor, etc.)
              Case-insensitive: "SPARRING1", "Sparring1", "sparring1" all work.
    
    Returns:
        Internal mode name (flash, full, pro, discovery, red, blue, white)
    """
    normalized_mode = mode.lower().strip() if mode else ""
    if normalized_mode in MODE_ALIASES:
        return MODE_ALIASES[normalized_mode]
    logger.warning(f"Unknown sparring mode '{mode}'. Using 'flash' as fallback.")
    return "flash"

async def generate_sparring(
    topic: str,
    context: str,
    session_id: str,
    mode: str = "sparring2",
    user_input: Optional[str] = None,  # New: multi-turn conversation input
    ctx: Optional[Context] = None,
    project_id: str = "default",
    workspace_root: Optional[str] = None,
    force_mode: Optional[str] = None,  # Manual override to bypass auto-detection
) -> str:
    """
    Execute sparring session using SparringEngineV2.
    
    Supports both full sessions (sparring1/2) and step-by-step (sparring3).
    
    Multi-turn support:
    - user_input: Append a new message to the conversation history
    - Automatically filters reasoning_content for security
    - Applies rolling summary when context exceeds limit
    
    Mode routing:
    - Auto-detects mode based on topic complexity and context length
    - force_mode: Manual override to bypass auto-detection
    
    Defensive parameter handling: ensures all parameters are strings even if MCP passes dicts.
    
    Args:
        topic: Sparring topic
        context: Additional context
        mode: Sparring mode (flash/full/pro)
        session_id: Session identifier for multi-turn
        user_input: Multi-turn conversation input
        ctx: MCP context for progress reporting
        project_id: Project identifier
        workspace_root: Path to workspace root (default: current directory)
        force_mode: Manual mode override (flash/full/pro) to bypass auto-detection
    """
    from qwen_mcp.engines.sparring_v2 import SparringEngineV2, SparringResponse
    from qwen_mcp.engines.session_store import SessionStore
    from pathlib import Path
    
    # Defensive: Ensure parameters are strings (MCP sometimes passes dicts for complex prompts)
    if isinstance(topic, dict):
        topic = str(topic.get("topic", topic))
    if isinstance(context, dict):
        context = str(context.get("context", context))
    if isinstance(mode, dict):
        mode = str(mode.get("mode", "sparring2"))
    if isinstance(session_id, dict):
        session_id = str(session_id.get("session_id", ""))
    if isinstance(user_input, dict):
        user_input = str(user_input.get("user_input", ""))
    if isinstance(workspace_root, dict):
        workspace_root = str(workspace_root.get("workspace_root", ""))
    
    # Ensure string types
    topic = str(topic) if topic else ""
    context = str(context) if context else ""
    mode = str(mode) if mode else "sparring2"
    session_id = str(session_id) if session_id else ""
    user_input = str(user_input) if user_input else ""
    workspace_root = str(workspace_root) if workspace_root else None
    
    # CRITICAL: Use provided workspace_root, NOT Path.cwd()
    # This ensures session files are saved in the CLIENT's project directory, not server directory
    if workspace_root:
        workspace = Path(workspace_root)
    else:
        # Fallback to MCP context's workspace folder if available
        if ctx and hasattr(ctx, 'request_context') and ctx.request_context:
            session = ctx.request_context.session
            workspace_uri = getattr(session, 'root_uri', None)
            if workspace_uri and workspace_uri.startswith('file:///'):
                workspace = Path(workspace_uri[8:])
            else:
                workspace = Path.cwd()
        else:
            workspace = Path.cwd()
    
    # Initialize session store for multi-turn support
    # Uses default storage directory resolution:
    # 1. QWEN_SPARRING_SESSIONS_DIR env variable
    # 2. %APPDATA%\qwen-mcp\sparring_sessions (Windows) / ~/.qwen-mcp/sparring_sessions (Unix)
    # 3. Fallback to .sparring_sessions in current directory
    session_store = SessionStore()
    
    client = DashScopeClient()
    engine = SparringEngineV2(client, session_store)
    
    # Resolve mode alias to internal mode
    # Use force_mode if provided (bypass auto-detection)
    if force_mode:
        # force_mode overrides everything - use it directly
        internal_mode = force_mode.lower().strip()
        logger.info(f"Using force_mode override: {internal_mode}")
    else:
        # Auto-detect mode based on topic complexity and context length
        # ONLY if user didn't specify an explicit mode (mode is empty/default)
        combined_text = f"{topic} {context}".strip()
        text_length = len(combined_text)

        # Check if user specified an explicit mode
        explicit_mode = mode and mode.lower().strip() not in ("", "auto", "default")
        
        if explicit_mode:
            # User specified an explicit mode - respect their choice
            internal_mode = resolve_sparring_mode(mode)
            logger.info(f"Using explicit mode: {internal_mode} (context length: {text_length} chars)")
        else:
            # No explicit mode - auto-detect based on context length
            internal_mode = resolve_sparring_mode(mode) or "flash"
            if text_length > 5000:
                # Very long context - use pro mode for higher token limits
                internal_mode = "pro"
                logger.info(f"Auto-detected mode: pro (context length: {text_length} chars)")
            elif text_length > 1500:
                # Medium context - use full mode
                internal_mode = "full"
                logger.info(f"Auto-detected mode: full (context length: {text_length} chars)")
            else:
                # Short context - use flash mode
                internal_mode = "flash"
                logger.info(f"Auto-detected mode: flash (context length: {text_length} chars)")
    
    # Multi-turn support: Append user input to conversation history BEFORE execution
    messages_appended = 0
    context_truncated = False
    
    if user_input:
        # Append user message to session history (sanitized automatically)
        user_msg = {"role": "user", "content": user_input}
        context_truncated = session_store.add_message(session_id, user_msg)
        messages_appended = 1
        
        # Also append the topic as context if it's a continuation
        if topic and session_id:
            # Topic becomes part of the conversation context
            pass
    
    # Report progress and broadcast for UI visibility
    if ctx:
        progress_msg = f"Starting sparring (mode: {internal_mode})"
        if user_input:
            progress_msg += f" | Multi-turn: {messages_appended} message(s) added"
        if context_truncated:
            progress_msg += " | Context truncated"
        await ctx.report_progress(progress=10, total=None, message=progress_msg + "...")
    
    await get_broadcaster().broadcast_state({
        "operation": f"Sparring session in progress (mode: {internal_mode})...",
        "progress": 10.0,
        "is_live": True,
        "multi_turn": {
            "messages_appended": messages_appended,
            "context_truncated": context_truncated
        }
    }, project_id=project_id)
    
    # Execute via engine
    try:
        response: SparringResponse = await engine.execute(
            mode=internal_mode,
            topic=topic,
            context=context,
            session_id=session_id,
            ctx=ctx
        )
        # Update response with multi-turn tracking info
        response.messages_appended = messages_appended
        response.context_truncated = context_truncated
    except Exception as e:
        logger.error(f"Sparring engine execution failed: {e}", exc_info=True)
        return f"❌ Sparring engine error: {type(e).__name__}: {e}"
    
    # Format response for MCP tool output
    if response.success:
        # Use the updated to_markdown() which includes multi-turn info and session file path
        # Pass the session store's storage directory so agent knows where to find the session file
        return response.to_markdown(storage_dir=str(session_store.storage_dir), session_store=session_store)
    else:
        return f"❌ Sparring failed: {response.error or response.message}"

async def generate_swarm(prompt: str, task_type: str, ctx: Optional[Context] = None, project_id: str = "default") -> str:
    """
    Swarm orchestrator for parallel multi-agent analysis.
    
    Decomposes complex tasks into parallel subtasks executed by specialized agents.
    """
    from qwen_mcp.orchestrator import SwarmOrchestrator
    from qwen_mcp.engines.scout import ScoutEngine
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # Scout analysis to determine complexity - with progress broadcast
    if ctx:
        await ctx.report_progress(progress=10, total=None, message="Analyzing task complexity...")
    await get_broadcaster().broadcast_state({
        "operation": "Swarm analyzing task complexity...",
        "progress": 10.0,
        "is_live": True
    }, project_id=project_id)
    
    try:
        scout_result = await scout.analyze_task(
            prompt,
            task_hint=task_type,
            progress_callback=ctx.report_progress if ctx else None
        )
        complexity = scout_result.get("complexity", "high")
        logger.info(f"Scout analysis for swarm: complexity={complexity}")
    except Exception as e:
        logger.warning(f"Scout analysis failed: {e}")
        complexity = None
    
    # Broadcast during orchestration
    if ctx:
        await ctx.report_progress(progress=50, total=None, message="Running swarm agents...")
    await get_broadcaster().broadcast_state({
        "operation": "Swarm agents executing in parallel...",
        "progress": 50.0,
        "is_live": True
    }, project_id=project_id)
    
    orchestrator = SwarmOrchestrator(completion_handler=client)
    result = await orchestrator.run_swarm(prompt, task_type=task_type)
    return result

async def generate_sos_sync(apply: bool, decision_id: str, apply_all: bool, workspace_root: str) -> str:
    """
    Decision Log Sync synchronization tool.
    
    Synchronizes decision_log.parquet state with BACKLOG.md and CHANGELOG.md.
    """
    import os
    import sys
    from pathlib import Path
    
    # Add src directory to Python path for imports
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
    
    decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace_root)
    backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace_root)
    
    if not decision_log_path.exists():
        return f"No decision log found at {decision_log_path}"
    
    engine = DecisionLogSyncEngine(decision_log_path)
    
    if apply_all:
        result = await engine.apply_all_advices(backlog_path)
        if result:
            return "✅ All pending Decision Log insights applied to BACKLOG.md"
        return "No pending insights to apply"
    
    if apply and decision_id:
        result = await engine.apply_advice(decision_id, backlog_path)
        if result:
            return f"✅ Decision Log insight {decision_id} applied to BACKLOG.md"
        return f"No pending insight found for {decision_id}"
    
    # Just scan and report
    advices = await engine.scan_advices()
    if not advices:
        return "No pending Decision Log insights"
    return f"Found {len(advices)} pending Decision Log insights. Use apply_all=True to apply."


async def add_task_to_backlog(
    task_name: str,
    advice: str,
    workspace_root: str,
    session_id: str = "decision_log_manual",
    decision_type: str = "manual_task",
    complexity: str = "medium",
    tags: Optional[List[str]] = None,
    risk_score: float = 0.0
) -> str:
    """
    Add a new task from natural language to BACKLOG.md and decision_log.parquet.
    
    This is the "Files → Parquet" direction of Decision Log sync.
    
    Args:
        task_name: Human-readable task name
        advice: The agentic advice/recommendation
        workspace_root: Workspace root path
        session_id: Session identifier (default: "decision_log_manual")
        decision_type: Type of decision (default: "manual_task")
        complexity: Task complexity (default: "medium")
        tags: Optional tags list
        risk_score: Risk assessment (default: 0.0)
        
    Returns:
        Confirmation message with decision_id
    """
    from pathlib import Path
    from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
    
    decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace_root)
    backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace_root)
    
    engine = DecisionLogSyncEngine(decision_log_path)
    
    try:
        decision_id = await engine.add_task(
            task_name=task_name,
            advice=advice,
            backlog_path=backlog_path,
            workspace_root=workspace_root,
            session_id=session_id,
            decision_type=decision_type,
            complexity=complexity,
            tags=tags,
            risk_score=risk_score
        )
        return f"✅ Task added to BACKLOG.md with decision_id: `{decision_id}`\n\nTask: {task_name}\nAdvice: {advice}"
    except ValueError as e:
        return f"❌ Invalid input: {e}"
    except Exception as e:
        return f"❌ Failed to add task: {e}"


async def add_tasks_to_backlog_batch(
    tasks: List[Dict[str, Any]],
    workspace_root: str,
    session_id: str = "sos_manual",
    decision_type: str = "manual_task",
    chunk_size: int = 20
) -> str:
    """
    Add multiple tasks from natural language to BACKLOG.md and decision_log.parquet.
    
    This is the batch version of add_task_to_backlog for handling large task lists.
    Processes tasks in chunks to avoid MCP timeout.
    
    Args:
        tasks: List of task dictionaries with keys:
            - task_name (required): Human-readable task name
            - advice (required): The agentic advice/recommendation
            - complexity (optional): Task complexity (default: "medium")
            - tags (optional): Tags list
            - risk_score (optional): Risk assessment (default: 0.0)
        workspace_root: Workspace root path
        session_id: Session identifier (default: "sos_manual")
        decision_type: Type of decision (default: "manual_task")
        chunk_size: Number of tasks per batch (default: 20)
    
    Returns:
        Confirmation message with count of added tasks
    """
    from pathlib import Path
    from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
    
    if not tasks:
        return "❌ No tasks provided"
    
    decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace_root)
    backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace_root)
    
    engine = DecisionLogSyncEngine(decision_log_path)
    
    try:
        decision_ids = await engine.add_tasks(
            tasks=tasks,
            backlog_path=backlog_path,
            workspace_root=workspace_root,
            session_id=session_id,
            decision_type=decision_type,
            chunk_size=chunk_size
        )
        
        # Format response
        ids_preview = decision_ids[:5]
        ids_str = ", ".join(ids_preview)
        if len(decision_ids) > 5:
            ids_str += f"... (+{len(decision_ids) - 5} more)"
        
        return f"✅ Added {len(decision_ids)} tasks to BACKLOG.md\n\nDecision IDs: {ids_str}"
    except ValueError as e:
        return f"❌ Invalid input: {e}"
    except Exception as e:
        return f"❌ Failed to add tasks: {e}"

async def list_available_models() -> str:
    client = DashScopeClient()
    models = await client.list_models()
    return json.dumps(models, indent=2)

async def set_model_in_registry(role: str, model_id: str) -> str:
    registry.models[role] = model_id
    await registry.save_cache()
    return f"Role {role} set to {model_id}"

async def generate_usage_report() -> str:
    """
    Generates token usage and cost summary from billing tracker.
    """
    from qwen_mcp.billing import billing_tracker
    
    report = billing_tracker.get_summary()
    
    output = "## 📊 Usage Report\n\n"
    output += f"**Total Requests**: {report.get('total_requests', 0)}\n"
    output += f"**Total Prompt Tokens**: {report.get('total_prompt_tokens', 0)}\n"
    output += f"**Total Completion Tokens**: {report.get('total_completion_tokens', 0)}\n\n"
    
    if report.get('by_model'):
        output += "### By Model\n\n"
        for model, stats in report.get('by_model', {}).items():
            output += f"- **{model}**: {stats.get('requests', 0)} requests, {stats.get('prompt', 0)} prompt, {stats.get('completion', 0)} completion\n"
    
    return output

async def read_repo_file(path: str) -> str:
    if not os.path.exists(path):
        return f"Error: File not found at {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

async def list_repo_files(directory: str = ".", pattern: str = "**/*") -> str:
    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    return "\n".join([f for f in files if os.path.isfile(f)][:100])


async def qwen_list_tasks(
    status: str = "pending",
    tags: Optional[List[str]] = None,
    workspace_root: str = "."
) -> str:
    """
    List tasks from BACKLOG.md with optional filtering.
    
    Args:
        status: Filter by status - "pending", "completed", or "all" (default: "pending")
        tags: Optional list of tags to filter by
        workspace_root: Path to workspace root (default: current directory)
        
    Returns:
        Formatted list of tasks with their details
    """
    from pathlib import Path
    
    backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace_root)
    
    if not backlog_path.exists():
        return f"❌ BACKLOG.md not found at {backlog_path}"
    
    tasks = []
    with open(backlog_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse tasks from BACKLOG.md
    # Format: "- [ ] TaskName - decision_id" or "- [x] TaskName - decision_id"
    import re
    task_pattern = r"- \[([ x])\] (.+?) - ([a-f0-9\-]{36})"
    matches = re.findall(task_pattern, content)
    
    for match in matches:
        task_status = "completed" if match[0] == "x" else "pending"
        task_name = match[1].strip()
        decision_id = match[2]
        
        # Filter by status
        if status != "all" and task_status != status:
            continue
        
        tasks.append({
            "status": task_status,
            "name": task_name,
            "decision_id": decision_id
        })
    
    if not tasks:
        return f"No tasks found with status: {status}"
    
    output = f"## 📋 Tasks ({status})\n\n"
    for i, task in enumerate(tasks, 1):
        status_icon = "✅" if task["status"] == "completed" else "⬜"
        output += f"{i}. {status_icon} **{task['name']}**\n"
        output += f"   - ID: `{task['decision_id']}`\n"
    
    output += f"\n**Total: {len(tasks)} tasks**"
    return output


async def qwen_get_task(
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
    from pathlib import Path
    
    decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace_root)
    
    if not decision_log_path.exists():
        return f"❌ decision_log.parquet not found at {decision_log_path}"
    
    try:
        df = pd.read_parquet(decision_log_path)
        
        # Find the task by decision_id
        matching = df[df['decision_id'] == decision_id]
        
        if matching.empty:
            return f"❌ Task with decision_id `{decision_id}` not found"
        
        row = matching.iloc[0]
        
        output = f"## 📋 Task Details\n\n"
        output += f"**Decision ID**: `{row.get('decision_id', 'N/A')}`\n"
        output += f"**Task Name**: {row.get('task_name', 'N/A')}\n"
        output += f"**Status**: {row.get('status', 'pending')}\n"
        output += f"**Type**: {row.get('decision_type', 'N/A')}\n"
        output += f"**Complexity**: {row.get('complexity', 'N/A')}\n"
        output += f"**Risk Score**: {row.get('risk_score', 0)}\n"
        output += f"**Session ID**: {row.get('session_id', 'N/A')}\n"
        output += f"**Created**: {row.get('timestamp', 'N/A')}\n\n"
        
        if 'tags' in row and row['tags']:
            output += f"**Tags**: {row['tags']}\n\n"
        
        if 'agentic_advice' in row:
            output += f"### Advice\n\n{row['agentic_advice']}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error reading decision_log.parquet: {e}"


async def qwen_update_task(
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
    from pathlib import Path
    
    backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace_root)
    decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace_root)
    
    # Update BACKLOG.md
    if not backlog_path.exists():
        return f"❌ BACKLOG.md not found at {backlog_path}"
    
    with open(backlog_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find and update the task in BACKLOG.md
    import re
    task_pattern = rf"(- \[) ([ x]) (\] .+? - {re.escape(decision_id)})"
    
    def replace_status(match):
        checkbox = "[x]" if new_status == "completed" else "[ ]"
        return f"- {checkbox} {match.group(3)[3:]}"  # Keep task name and id
    
    new_content, count = re.subn(task_pattern, replace_status, content)
    
    if count == 0:
        return f"❌ Task with decision_id `{decision_id}` not found in BACKLOG.md"
    
    with open(backlog_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    # Update decision_log.parquet
    if decision_log_path.exists():
        try:
            df = pd.read_parquet(decision_log_path)
            
            mask = df['decision_id'] == decision_id
            if mask.any():
                df.loc[mask, 'status'] = new_status
                df.to_parquet(decision_log_path, index=False)
        except Exception as e:
            return f"⚠️ Updated BACKLOG.md but failed to update decision_log.parquet: {e}"
    
    status_icon = "✅" if new_status == "completed" else "⬜" if new_status == "pending" else "🔄"
    return f"{status_icon} Task `{decision_id}` updated to **{new_status}** in both BACKLOG.md and decision_log.parquet"
