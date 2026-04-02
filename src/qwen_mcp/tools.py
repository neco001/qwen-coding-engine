import logging
import json
import re
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import Context
from qwen_mcp.api import DashScopeClient
from qwen_mcp.base import set_billing_mode as apply_billing_mode, get_billing_mode
from qwen_mcp.registry import registry
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.specter.telemetry import get_broadcaster

# New modular imports
from qwen_mcp.prompts.system import AUDIT_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT
from qwen_mcp.prompts.lachman import LP_DISCOVERY_PROMPT, LP_ARCHITECT_PROMPT, LP_VERIFIER_PROMPT

# Sparring mode configuration
from qwen_mcp.engines.sparring_v2.config import MODE_ALIASES, DEFAULT_SPARRING_MODE

logger = logging.getLogger(__name__)


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
    # Normalize to lowercase for case-insensitive matching
    normalized_mode = mode.lower().strip() if mode else ""
    
    # Check if mode is in aliases
    if normalized_mode in MODE_ALIASES:
        return MODE_ALIASES[normalized_mode]
    
    # If not found, return flash as fallback (safest option)
    logger.warning(f"Unknown sparring mode '{mode}'. Using 'flash' as fallback.")
    return "flash"

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Utility to pull JSON blocks from LLM markdown responses."""
    try:
        # Try finding a ```json block first
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        json_str = match.group(1) if match else text
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

async def generate_audit(
    content: str,
    context: Optional[str] = None,
    ctx: Optional[Context] = None,
    use_swarm: bool = True,
    project_id: str = "default"
) -> str:
    """
    Audits the provided code or terminal logs using Qwen models.
    
    For multi-file content, automatically uses Swarm for parallel analysis.
    Set use_swarm=False to disable parallel processing.
    
    Args:
        content: The code or logs to audit
        context: Additional context for the audit
        ctx: MCP context for progress reporting
        use_swarm: Enable automatic parallel file analysis (default: True)
        project_id: Project/session ID for telemetry isolation (format: {instance}_{source}_{hash})
    
    Returns:
        Audit report with findings and recommendations
    """
    from qwen_mcp.engines.scout import ScoutEngine
    from qwen_mcp.orchestrator import SwarmOrchestrator
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # 1. Scout Analysis
    scout_res = await scout.analyze_task(
        content, context, task_hint="audit",
        progress_callback=ctx.report_progress if ctx else None
    )
    use_swarm_recommendation = scout_res.get("use_swarm", False)
    complexity = scout_res.get("complexity", "high")
    reason = scout_res.get("reason", "Standard audit")
    
    # 2. Execution Choice
    if use_swarm and use_swarm_recommendation:
        orchestrator = SwarmOrchestrator(completion_handler=client)
        
        swarm_prompt = f"""Audit the following code or logs for issues, bugs, and improvements.
Context: {context or 'General code audit'}
Content to audit:
{content}

Provide a comprehensive audit report with summary, critical issues, and recommendations."""
        
        return await orchestrator.run_swarm(swarm_prompt, task_type="audit")
    
    # Standard completion with intelligent complexity limit
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context or 'None'}\n\nContent to audit:\n{content}"}
    ]
    return await client.generate_completion(
        messages=messages,
        task_type="audit",
        complexity=complexity,
        tags=["audit"],
        progress_callback=ctx.report_progress if ctx else None,
        project_id=project_id
    )

async def generate_code(
    prompt: str,
    context: Optional[str] = None,
    ctx: Optional[Context] = None,
    project_id: str = "default"
) -> str:
    """
    Simple code generation using standard coder model.
    
    Args:
        prompt: The code generation request
        context: Additional context (optional)
        ctx: MCP context for progress reporting
        project_id: Project/session ID for telemetry isolation (format: {instance}_{source}_{hash})
    
    Returns:
        Generated code as markdown text
    """
    client = DashScopeClient()
    messages = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context or 'None'}\n\nPrompt: {prompt}"}
    ]
    return await client.generate_completion(
        messages=messages,
        task_type="coding",
        tags=["coder"],
        progress_callback=ctx.report_progress if ctx else None,
        project_id=project_id
    )

async def generate_code_unified(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    ctx: Optional[Context] = None,
    project_id: str = "default"
) -> str:
    """
    Unified code generation with mode-based routing.
    
    Modes:
    - auto: Intelligent routing based on prompt complexity (default)
    - standard: Fast generation using qwen3-coder-next
    - pro: Heavy-duty generation using qwen3-coder-plus
    - expert: Maximum capability for complex refactors/architecture
    
    Args:
        prompt: The code generation request
        mode: One of 'auto', 'standard', 'pro', 'expert'
        context: Additional context (optional)
        ctx: MCP context for progress reporting
        project_id: Project/session ID for telemetry isolation (format: {instance}_{source}_{hash})
    
    Returns:
        Markdown-formatted response with generated code
    """
    from qwen_mcp.engines.coder_v2 import CoderEngineV2
    
    client = DashScopeClient()
    engine = CoderEngineV2(client)
    
    response = await engine.execute(
        prompt=prompt,
        mode=mode,
        context=context or "",
        ctx=ctx,
        project_id=project_id
    )
    
    return response.to_markdown()

async def generate_lp_blueprint(goal: str, context: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    # This remains complex, but keeps the modular structure
    from qwen_mcp.engines.scout import ScoutEngine
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # 0. Scout for Sizing (Architect blueprints are often large)
    scout_res = await scout.analyze_task(
        goal, context, task_hint="strategy/architecture",
        progress_callback=ctx.report_progress if ctx else None
    )
    complexity = scout_res.get("complexity", "high")
    
    # 1. Discovery
    discovery_msg = [
        {"role": "system", "content": LP_DISCOVERY_PROMPT},
        {"role": "user", "content": f"Goal: {goal}\nContext: {context or ''}"}
    ]
    discovery_raw = await client.generate_completion(messages=discovery_msg, complexity="medium")
    discovery = extract_json_from_text(discovery_raw) or {"hired_squad": []}
    squad_str = ", ".join([s.get("role", "Expert") for s in discovery.get("hired_squad", [])])

    # 2. Architect
    arch_msg = [
        {"role": "system", "content": LP_ARCHITECT_PROMPT.format(squad=squad_str)},
        {"role": "user", "content": f"Goal: {goal}"}
    ]
    # Blueprints use scouted complexity (often high/critical) to unlock 4k/8k tokens
    blueprint_raw = await client.generate_completion(
        messages=arch_msg,
        task_type="strategist",
        complexity=complexity,
        progress_callback=ctx.report_progress if ctx else None
    )
    
    # 3. Parse and enhance blueprint with swarm-ready task formatting
    blueprint_data = extract_json_from_text(blueprint_raw)
    if blueprint_data and "swarm_tasks" in blueprint_data:
        # Format swarm_tasks for downstream execution
        swarm_section = "\n\n## 🎯 Swarm Execution Tasks\n\n"
        swarm_section += "The following atomic tasks are ready for parallel execution:\n\n"
        
        for task in blueprint_data.get("swarm_tasks", []):
            task_id = task.get("id", "unknown")
            task_desc = task.get("task", "")
            priority = task.get("priority", 5)
            target_files = task.get("target_files", [])
            exec_hint = task.get("execution_hint", "qwen_coder")
            
            swarm_section += f"### {task_id} (Priority: {priority})\n"
            swarm_section += f"- **Task**: {task_desc}\n"
            swarm_section += f"- **Target Files**: {', '.join(target_files) if target_files else 'N/A'}\n"
            swarm_section += f"- **Execution**: `{exec_hint}`\n\n"
        
        # Append swarm section to raw blueprint
        return blueprint_raw + swarm_section
    
    return blueprint_raw

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

async def set_billing_mode(mode: str) -> str:
    """Changes the global billing mode: 'payg', 'coding_plan', or 'hybrid'."""
    success = apply_billing_mode(mode)
    if success:
        return f"✅ Billing Mode changed to: {mode.upper()}"
    return f"❌ Invalid mode: {mode}. Use 'payg', 'coding_plan', or 'hybrid'."

async def get_current_billing_mode() -> str:
    """Returns the current billing mode."""
    mode = get_billing_mode()
    return f"Current Billing Mode: {mode.upper()}"

async def generate_sparring(
    topic: str = "",
    context: str = "",
    mode: str = DEFAULT_SPARRING_MODE,  # Default: sparring2 (normal/full)
    session_id: str = "",
    ctx: Optional[Context] = None
) -> str:
    """
    Executes the Sparring Engine v2 with step-by-step execution.
    
    Sparring Levels (use aliases for clarity):
    - sparring1 (flash): Quick 2-step analysis (analyst→drafter), 180s timeout
    - sparring2 (normal): Full session in one call, 180s timeout - DEFAULT
    - sparring3 (pro): Step-by-step with checkpointing, 100s per step
    
    Step-by-step modes (for sparring3/pro):
    - discovery: Create session + define roles (returns session_id)
    - red: Execute Red Cell critique (requires session_id)
    - blue: Execute Blue Cell defense (requires session_id + red critique)
    - white: Execute White Cell synthesis (requires session_id + red + blue)
    
    Args:
        topic: The topic to analyze (required for sparring1/2/3/discovery)
        context: Additional context (optional)
        mode: Sparring level (sparring1, sparring2, sparring3) or step mode (discovery, red, blue, white)
        session_id: Session ID for step modes (required for red/blue/white)
    
    Returns:
        Markdown-formatted response with guided UX hints
    """
    from qwen_mcp.engines.sparring_v2 import SparringEngineV2
    
    client = DashScopeClient()
    engine = SparringEngineV2(client)
    
    # Resolve mode alias (sparring1→flash, sparring2→full, sparring3→pro)
    resolved_mode = resolve_sparring_mode(mode)
    
    # Handle session_id parameter
    if resolved_mode not in ["flash", "full"] and not session_id:
        session_id = None
    
    response = await engine.execute(
        mode=resolved_mode,
        topic=topic or None,
        context=context or None,
        session_id=session_id or None,
        ctx=ctx  # Pass context for progress reporting
    )
    
    return response.to_markdown()


async def heal_registry() -> str:
    client = DashScopeClient()
    return await client.heal_registry()


async def generate_swarm(
    prompt: str,
    task_type: str = "general",
    ctx: Optional[Context] = None
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
    from qwen_mcp.orchestrator import SwarmOrchestrator
    from qwen_mcp.engines.scout import ScoutEngine
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # Scout analysis to determine complexity (fixes truncation issue)
    try:
        scout_result = await scout.analyze_task(
            prompt,
            task_hint=task_type,
            progress_callback=ctx.report_progress if ctx else None
        )
        complexity = scout_result.get("complexity", "high")
        logger.info(f"Scout analysis for swarm: complexity={complexity}, reason={scout_result.get('reason', 'N/A')}")
    except Exception as e:
        logger.warning(f"Scout analysis failed: {e}. TokenScout will handle max_tokens estimation.")
        complexity = None  # DEPRECATED - TokenScout handles estimation
    
    orchestrator = SwarmOrchestrator(completion_handler=client)
    result = await orchestrator.run_swarm(prompt, task_type=task_type)
    return result


async def qwen_init_context(
    workspace_root: str = ".",
    ctx: Context = None
) -> str:
    """
    Initialize project context files using Swarm analysis.
    
    Generates:
    - .context/_PROJECT_CONTEXT.md: Tech stack, structure, conventions
    - .context/_DATA_CONTEXT.md: Data sources, schemas, pipelines
    
    Args:
        workspace_root: Path to workspace root (default: current directory)
        ctx: MCP context for progress reporting
    
    Returns:
        Summary of generated files with paths
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


async def qwen_update_session_context(
    session_summary: str,
    workspace_root: str = ".",
    ctx: Context = None
) -> str:
    """
    Update session supplement with current session insights.
    
    Args:
        session_summary: Summary of work done in this session
        workspace_root: Path to workspace root
        ctx: MCP context for progress reporting
    
    Returns:
        Confirmation of update with session highlights
    """
    from pathlib import Path
    from qwen_mcp.engines.context_builder import ContextBuilderEngine
    
    # Input validation
    if not session_summary or not session_summary.strip():
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
            session_summary,
            workspace_root
        )
        
        # Save file
        saved_path = engine.save_session_context(
            session_content,
            workspace_root
        )
        
        # Extract key highlights from session summary
        highlights = session_summary.split("\n")[:5]  # First 5 lines as highlights
        
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
