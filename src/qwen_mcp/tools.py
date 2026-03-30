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

logger = logging.getLogger(__name__)

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
    use_swarm: bool = True
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
    
    Returns:
        Audit report with findings and recommendations
    """
    from qwen_mcp.engines.scout import ScoutEngine
    from qwen_mcp.orchestrator import SwarmOrchestrator
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # 1. Scout Analysis
    scout_res = await scout.analyze_task(content, context, task_hint="audit")
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

async def generate_code_unified(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    ctx: Optional[Context] = None
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
        ctx=ctx
    )
    
    return response.to_markdown()

async def generate_lp_blueprint(goal: str, context: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    # This remains complex, but keeps the modular structure
    from qwen_mcp.engines.scout import ScoutEngine
    
    client = DashScopeClient()
    scout = ScoutEngine(client)
    
    # 0. Scout for Sizing (Architect blueprints are often large)
    scout_res = await scout.analyze_task(goal, context, task_hint="strategy/architecture")
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
    blueprint = await client.generate_completion(
        messages=arch_msg, 
        task_type="strategist",
        complexity=complexity,
        progress_callback=ctx.report_progress if ctx else None
    )
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
    mode: str = "flash",
    session_id: str = "",
    ctx: Optional[Context] = None
) -> str:
    """
    Executes the Sparring Engine v2 with step-by-step execution.
    
    Modes:
    - flash: Quick analysis + draft (single call, no session)
    - discovery: Create session + define roles (returns session_id)
    - red: Execute Red Cell critique (requires session_id)
    - blue: Execute Blue Cell defense (requires session_id + red critique)
    - white: Execute White Cell synthesis (requires session_id + red + blue)
    - full: Execute entire session (discovery→red→blue→white) in one call with progress reporting
    
    Args:
        topic: The topic to analyze (required for flash/discovery/full)
        context: Additional context (optional)
        mode: One of: flash, discovery, red, blue, white, full
        session_id: Session ID for red/blue/white modes (required for non-flash modes)
    
    Returns:
        Markdown-formatted response with guided UX hints
    """
    from qwen_mcp.engines.sparring_v2 import SparringEngineV2
    
    client = DashScopeClient()
    engine = SparringEngineV2(client)
    
    # Handle session_id parameter
    if mode != "flash" and not session_id:
        session_id = None
    
    response = await engine.execute(
        mode=mode,
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
    
    client = DashScopeClient()
    orchestrator = SwarmOrchestrator(completion_handler=client)
    
    result = await orchestrator.run_swarm(prompt, task_type=task_type)
    return result
