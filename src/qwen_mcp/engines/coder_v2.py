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
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from mcp.server.fastmcp import Context

from qwen_mcp.api import DashScopeClient
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.registry import registry
from qwen_mcp.prompts.system import CODER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# =============================================================================
# Mode Configuration
# =============================================================================

MODE_ROUTING = {
    "auto": "coding",        # Let registry decide based on complexity
    "standard": "coder",     # qwen3-coder-next (fast, plan)
    "pro": "coder_pro",      # qwen3-coder-plus (heavy, plan)
    "expert": "specialist",  # qwen2.5-coder-32b-instruct (PAYG)
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
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    async def execute(
        self, 
        prompt: str,
        mode: str = "auto",
        context: str = "",
        ctx: Optional[Context] = None
    ) -> CoderResponse:
        """
        Execute code generation with specified mode.
        
        Args:
            prompt: The code generation request
            mode: One of 'auto', 'standard', 'pro', 'expert'
            context: Additional context (existing code, requirements, etc.)
            ctx: MCP context for progress reporting
            
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
            
            # Determine task type and model
            task_type, model_used, routing_reason = self._resolve_mode(mode, prompt, context)
            
            await self._report_progress(ctx, 0.0, f"[Coder] Mode: {mode} → {model_used}")
            
            # Build messages
            messages = [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context or 'None'}\n\nPrompt: {prompt}"}
            ]
            
            # Generate code
            result = await self.client.generate_completion(
                messages=messages,
                task_type=task_type,
                complexity=self._estimate_complexity(prompt, context),
                tags=["coder", mode],
                progress_callback=ctx.report_progress if ctx else None
            )
            
            # Validate response
            result = ContentValidator.validate_response(result)
            
            elapsed = time.time() - start_time
            return CoderResponse(
                success=True,
                mode_used=mode,
                model_used=model_used,
                result=result,
                message=f"Code generated in {elapsed:.1f}s",
                routing_reason=routing_reason
            )
            
        except Exception as e:
            logger.exception(f"Coder execution failed: {e}")
            elapsed = time.time() - start_time
            return CoderResponse(
                success=False,
                mode_used=mode,
                model_used="",
                result="",
                message=f"Execution failed after {elapsed:.1f}s",
                error=str(e)
            )
    
    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------
    
    def _resolve_mode(self, mode: str, prompt: str, context: str) -> Tuple[str, str, str]:
        """
        Resolve mode to actual task_type and model.
        
        Returns:
            Tuple of (task_type, model_id, routing_reason)
        """
        if mode == "auto":
            # Intelligent auto-routing based on complexity
            estimated_tokens = len(prompt.split()) + len(context.split())
            
            # Auto-upgrade logic from registry
            if estimated_tokens > 15000 or self._estimate_complexity(prompt, context) in ["high", "critical"]:
                task_type = "coding_pro"
                routing_reason = "Auto-upgraded: High complexity detected"
            else:
                task_type = "coding"
                routing_reason = "Standard complexity routing"
        else:
            task_type = MODE_ROUTING[mode]
            routing_reason = f"Explicit mode selection: {mode}"
        
        # Get model from registry
        model_used = registry.get_best_model(task_type)
        
        return task_type, model_used, routing_reason
    
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
