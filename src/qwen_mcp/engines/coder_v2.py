"""
Coder Engine v2 - Unified Code Generation with Mode-Based Routing

This module provides a unified interface for code generation with multiple modes:
- auto: Intelligent routing based on prompt complexity
- standard: Fast generation using qwen3-coder-next
- pro: Heavy-duty generation using qwen3-coder-plus
- expert: Maximum capability for complex refactors/architecture

Similar to SparringEngineV2, this engine handles mode routing and execution.
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from mcp.server.fastmcp import Context

from qwen_mcp.api import DashScopeClient
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.registry import registry
from qwen_mcp.prompts.system import CODER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# =============================================================================
# Local Heuristics for Swarm Detection (Opcja A - NO API CALL)
# =============================================================================

def should_use_swarm(prompt: str, context: Optional[str] = None) -> bool:
    """
    Local heuristics to determine if Swarm is needed - NO API CALL.
    
    This replaces ScoutEngine.analyze_task() for most cases to avoid
    blocking API calls and potential timeouts.
    
    Returns:
        True if Swarm orchestrator should be used, False for single-agent generation
    """
    combined = prompt + " " + (context or "")
    # Check character count (~15K chars ≈ 3-4K tokens)
    if len(combined) > 15000:
        return True
    # Check word count (>500 words suggests complex task)
    if len(combined.split()) > 500:
        return True
    # Check for complexity keywords
    complexity_keywords = [
        "architecture", "refactor", "multiple files", "complex",
        "redesign", "restructure", "comprehensive", "full audit",
        "multi-file", "restructure", "migration", "boilerplate"
    ]
    if any(kw in combined.lower() for kw in complexity_keywords):
        return True
    return False


# =============================================================================
# Mode Configuration
# =============================================================================

# Mode routing: user-facing mode → internal task type
MODE_ROUTING = {
    "auto": "coding",        # Let complexity decide (resolved at runtime)
    "standard": "coding",    # qwen3-coder-next (fast, plan)
    "pro": "coding_pro",     # qwen3-coder-plus (heavy, plan)
    "expert": "coding_expert",  # qwen2.5-coder-32b-instruct (PAYG)
}

# Complexity to mode mapping for auto resolution
COMPLEXITY_TO_MODE = {
    "low": "standard",
    "medium": "standard",
    "high": "pro",
    "critical": "expert",
}

MODE_DESCRIPTIONS = {
    "auto": "Intelligent routing based on prompt complexity (default)",
    "standard": "Fast generation using qwen3-coder-next for simple tasks",
    "pro": "Heavy-duty generation using qwen3-coder-plus for complex features",
    "expert": "Maximum capability using qwen2.5-coder-32b for architecture/refactors",
}

# =============================================================================
# Response Schema
# =============================================================================

@dataclass
class CoderResponse:
    """Structured response for unified coder UX."""
    success: bool
    mode_used: str
    model_used: str
    result: str
    message: str
    error: Optional[str] = None
    routing_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "mode_used": self.mode_used,
            "model_used": self.model_used,
            "result": self.result,
            "message": self.message,
            "error": self.error,
            "routing_reason": self.routing_reason
        }
    
    def to_markdown(self) -> str:
        """Convert to human-readable markdown for MCP output."""
        if not self.success:
            return f"❌ **Error:** {self.error}\n\n{self.message}"
        
        lines = []
        lines.append(f"✅ **Code generation complete!**")
        lines.append("")
        lines.append(f"📋 **Mode:** `{self.mode_used}` - {MODE_DESCRIPTIONS.get(self.mode_used, '')}")
        lines.append(f"🤖 **Model:** `{self.model_used}`")
        
        if self.routing_reason:
            lines.append(f"💡 **Routing:** {self.routing_reason}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(self.result)
        
        return "\n".join(lines)


# =============================================================================
# Coder Engine v2
# =============================================================================

class CoderEngineV2:
    """
    Unified Code Generation Engine with mode-based routing.
    
    Modes:
    - auto: Intelligent routing based on prompt complexity
    - standard: Fast generation using qwen3-coder-next
    - pro: Heavy-duty generation using qwen3-coder-plus
    - expert: Maximum capability for complex refactors/architecture
    """
    
    def __init__(self, client: Optional[DashScopeClient] = None):
        self.client = client or DashScopeClient()
        from qwen_mcp.engines.scout import ScoutEngine
        self.scout_engine = ScoutEngine(self.client)
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    async def execute(
        self,
        prompt: str,
        mode: str = "auto",
        context: str = "",
        ctx: Optional[Context] = None,
        project_id: str = "default"
    ) -> CoderResponse:
        """
        Execute code generation with specified mode.
        
        Args:
            prompt: The code generation request
            mode: One of 'auto', 'standard', 'pro', 'expert'
            context: Additional context (existing code, requirements, etc.)
            ctx: MCP context for progress reporting
            project_id: Project/session ID for telemetry isolation (format: {instance}_{source}_{hash})
            
        Returns:
            CoderResponse with generated code and metadata
        """
        logger.info(f"Executing coder mode={mode}, prompt_length={len(prompt)}")
        start_time = time.time()
        
        try:
            # Validate mode
            if mode not in MODE_ROUTING:
                return CoderResponse(
                    success=False,
                    mode_used=mode,
                    model_used="",
                    result="",
                    message="Invalid mode",
                    error=f"Unknown mode: {mode}. Use: {', '.join(MODE_ROUTING.keys())}"
                )
            
            # BROWNFIELD DETECTION: Always run Scout for brownfield detection (not just complexity)
            # This ensures proper diff-only output for existing code modification
            is_brownfield = False
            brownfield_reason = "Not analyzed"
            
            # BROWNFIELD DETECTION: Always run Scout for brownfield detection (not just complexity)
            # This ensures proper diff-only output for existing code modification
            is_brownfield = False
            brownfield_reason = "Not analyzed"
            
            if mode == "auto":
                # Use local heuristics first (Opcja A)
                if should_use_swarm(prompt, context):
                    # Heuristics suggest Swarm - optionally call Scout with timeout
                    try:
                        # Attempt Scout API call with timeout for additional insight
                        scout_res = await self.scout_engine.analyze_task(
                            prompt, context, task_hint="coding",
                            progress_callback=ctx.report_progress if ctx else None
                        )
                        scout_complexity = scout_res.get("complexity", "high")
                        scout_use_swarm = scout_res.get("use_swarm", True)
                        is_brownfield = scout_res.get("is_brownfield", False)
                        routing_reason = scout_res.get("reason", "Heuristics triggered Swarm")
                        brownfield_reason = "Scout analysis"
                        use_swarm = scout_use_swarm
                    except Exception as e:
                        # Fallback: trust local heuristics on Scout failure
                        logger.warning(f"Scout API failed, using heuristics: {e}")
                        scout_complexity = "high"
                        use_swarm = True
                        routing_reason = f"Local heuristics (Scout failed: {str(e)})"
                        brownfield_reason = f"Scout failed: {str(e)}"
                else:
                    # Heuristics suggest simple task - skip Scout
                    scout_complexity = "low"
                    use_swarm = False
                    routing_reason = "Local heuristics: simple task"
                    # Check for brownfield indicators in context
                    is_brownfield = len(context or "") > 100  # Existing code context = brownfield
                    brownfield_reason = "Context-based heuristic"
            else:
                # Opcja C: For non-auto modes, skip Scout entirely
                # Use local heuristics only
                if should_use_swarm(prompt, context):
                    scout_complexity = "high"
                    use_swarm = True
                    routing_reason = f"Local heuristics (mode={mode})"
                else:
                    scout_complexity = "low"
                    use_swarm = False
                    routing_reason = f"Local heuristics (mode={mode})"
                # Check for brownfield indicators in context
                is_brownfield = len(context or "") > 100  # Existing code context = brownfield
                brownfield_reason = "Context-based heuristic"
            
            logger.info(f"Coder routing: mode={mode}, complexity={scout_complexity}, use_swarm={use_swarm}")

            # Determine task type, model, and resolved mode
            task_type, model_used, resolved_mode = self._resolve_mode(mode, prompt, context, scout_complexity)
            
            await self._report_progress(ctx, 5.0, f"[Coder] Scout: {scout_complexity} complexity. Model: {model_used}")
            
            # Delegate to Swarm if recommended and in auto mode
            if use_swarm:
                from qwen_mcp.orchestrator import SwarmOrchestrator
                await self._report_progress(ctx, 10.0, "[Coder] Task too complex for single agent - Launching Swarm Orchestrator...")
                orchestrator = SwarmOrchestrator(self.client)
                # Include brownfield flag in Swarm prompt for proper diff output
                brownfield_note = " [BROWNFIELD - Output DIFFS ONLY]" if is_brownfield else " [GREENFIELD - Full code OK]"
                swarm_prompt = f"### TASK:\n{prompt}\n\n### CONTEXT:\n{context or 'None'}\n\n### MODE:{brownfield_note}"
                result = await orchestrator.run_swarm(swarm_prompt, task_type=task_type)
                
                elapsed = time.time() - start_time
                return CoderResponse(
                    success=True, mode_used=mode, model_used="Swarm", result=result,
                    message=f"Code generated by Swarm in {elapsed:.1f}s",
                    routing_reason=f"Swarm triggered: {routing_reason}"
                )
            
            # Build messages with brownfield flag in system prompt
            # This ensures model knows whether to output diffs or full code
            system_prompt = CODER_SYSTEM_PROMPT
            if is_brownfield:
                system_prompt += "\n\nBROWNFIELD MODE ACTIVE: You are modifying existing code. Output DIFFS ONLY (SEARCH/REPLACE format with line numbers). Never output full files."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context or 'None'}\n\nPrompt: {prompt}\n\nBROWNFIELD: {is_brownfield} ({brownfield_reason})"}
            ]
            
            # Generate code (Single Agent)
            result = await self.client.generate_completion(
                messages=messages,
                task_type=task_type,
                complexity=scout_complexity,
                tags=["coder", mode],
                progress_callback=ctx.report_progress if ctx else None,
                project_id=project_id
            )
            
            # Validate response
            result = ContentValidator.validate_response(result)
            
            elapsed = time.time() - start_time
            return CoderResponse(
                success=True,
                mode_used=resolved_mode,  # Show actual mode selected (e.g., "pro" not "auto")
                model_used=model_used,
                result=result,
                message=f"Code generated in {elapsed:.1f}s",
                routing_reason=routing_reason
            )
            
        except Exception as e:
            logger.exception(f"Coder execution failed: {e}")
            elapsed = time.time() - start_time
            # For error case, try to use resolved_mode if available, otherwise use original mode
            try:
                error_mode = resolved_mode
            except NameError:
                error_mode = mode
            return CoderResponse(
                success=False,
                mode_used=error_mode,
                model_used="",
                result="",
                message=f"Execution failed after {elapsed:.1f}s",
                error=str(e)
            )
    
    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------
    
    def _resolve_mode(self, mode: str, prompt: str, context: str, scout_complexity: str = "medium") -> Tuple[str, str, str]:
        """
        Resolve mode to actual task_type, model, and resolved_mode.
        
        For 'auto' mode, resolves to explicit user-facing mode (standard/pro/expert)
        based on Scout/heuristics complexity analysis.
        
        Returns:
            Tuple of (task_type, model_id, resolved_mode)
            - resolved_mode is the user-facing mode that was actually used
        """
        # Map internal task types to registry roles
        TASK_TYPE_TO_ROLE = {
            "coding": "coder",
            "coding_pro": "coder",
            "coding_expert": "coder",
        }
        
        if mode == "auto":
            # Resolve auto to explicit user-facing mode based on complexity
            resolved_mode = COMPLEXITY_TO_MODE.get(scout_complexity, "standard")
            # Derive task_type from resolved mode
            task_type = MODE_ROUTING[resolved_mode]
        else:
            # For explicit modes, use predefined routing
            resolved_mode = mode
            task_type = MODE_ROUTING.get(mode, "coding")
        
        # Map task_type to registry role and get model
        role = TASK_TYPE_TO_ROLE.get(task_type, "coder")
        model_used = registry.get_best_model(role)
        
        # Safety fallback - ensure we always have a valid model string
        if not model_used:
            model_used = "qwen3.5-plus"
        
        return task_type, model_used, resolved_mode

    def _estimate_complexity(self, prompt: str, context: str) -> str:
        """
        Estimate task complexity based on prompt analysis.
        
        Returns:
            One of: 'low', 'medium', 'high', 'critical'
        """
        combined = prompt + " " + context
        word_count = len(combined.split())
        
        # Heuristic complexity estimation
        if word_count > 500:
            return "critical"
        elif word_count > 200:
            return "high"
        elif word_count > 50:
            return "medium"
        else:
            return "low"
    
    async def _report_progress(
        self, 
        ctx: Optional[Context], 
        progress: float, 
        message: str
    ) -> None:
        """Safe progress reporting."""
        if ctx:
            try:
                await ctx.report_progress(progress=float(progress), message=message)
            except Exception:
                pass
