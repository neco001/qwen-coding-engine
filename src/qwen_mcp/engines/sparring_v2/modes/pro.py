"""
Sparring Engine v2 - Pro Mode Executor (sparring3)

Execute sparring session step-by-step with checkpoints:
discovery→red→blue→white (each as separate MCP call).
Uses higher word limits (800 words) and token budgets (4096 tokens) for deep analysis.
"""

import logging
from typing import Optional
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.interfaces import ModeExecutor
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.prompts.sparring import WORD_LIMITS

logger = logging.getLogger(__name__)


class ProExecutor(ModeExecutor):
    """Execute sparring session step-by-step with checkpoints (sparring3)."""
    
    async def execute(
        self,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
    ) -> SparringResponse:
        """
        Execute pro mode sparring - step-by-step with checkpoints.
        
        Args:
            topic: Topic for the sparring
            context: Additional context
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with step results and next_command for continuation
        """
        logger.info(f"Executing pro mode (sparring3) for topic: {topic}")
        session_id = None
        
        try:
            # Import mode executors
            from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
            from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
            from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
            from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
            
            # Create executors sharing the same client and session_store
            discovery_executor = DiscoveryExecutor(self.client, self.session_store)
            red_executor = RedCellExecutor(self.client, self.session_store)
            blue_executor = BlueCellExecutor(self.client, self.session_store)
            white_executor = WhiteCellExecutor(self.client, self.session_store)
            
            # Get word limit for pro mode (800 words per cell)
            pro_word_limit = WORD_LIMITS.get("pro", 800)
            
            # Krok 1: Discovery - Defining roles
            await self._report_progress(ctx, 25.0, "[Pro] 1/4: Discovery - Defining roles...")
            discovery_result = await discovery_executor.execute(
                topic=topic,
                context=context,
                ctx=ctx,
                word_limit=pro_word_limit,
            )
            session_id = discovery_result.session_id
            
            if not session_id:
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed="discovery",
                    next_step="discovery",
                    next_command=f'qwen_sparring(mode="discovery", topic="{topic}", context="{context}")',
                    result=None,
                    message="Pro mode failed at discovery step",
                    error=discovery_result.error or "No session_id returned",
                )
            
            # Krok 2: Red Cell - Auditing risks
            await self._report_progress(ctx, 50.0, "[Pro] 2/4: Red Cell - Auditing risks...")
            red_result = await red_executor.execute(
                session_id=session_id,
                ctx=ctx,
                word_limit=pro_word_limit,
            )
            
            if not red_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="red",
                    next_step="red",
                    next_command=f'qwen_sparring(mode="red", session_id="{session_id}")',
                    result=None,
                    message="Pro mode failed at red cell step",
                    error=red_result.error,
                )
            
            # Krok 3: Blue Cell - Strategic defense
            await self._report_progress(ctx, 75.0, "[Pro] 3/4: Blue Cell - Strategic defense...")
            blue_result = await blue_executor.execute(
                session_id=session_id,
                ctx=ctx,
                word_limit=pro_word_limit,
            )
            
            if not blue_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="blue",
                    next_step="blue",
                    next_command=f'qwen_sparring(mode="blue", session_id="{session_id}")',
                    result=None,
                    message="Pro mode failed at blue cell step",
                    error=blue_result.error,
                )
            
            # Krok 4: White Cell - Final synthesis
            await self._report_progress(ctx, 100.0, "[Pro] 4/4: White Cell - Final synthesis...")
            white_result = await white_executor.execute(
                session_id=session_id,
                ctx=ctx,
                word_limit=pro_word_limit,
            )
            
            if not white_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="white",
                    next_step="white",
                    next_command=f'qwen_sparring(mode="white", session_id="{session_id}")',
                    result=None,
                    message="Pro mode failed at white cell step",
                    error=white_result.error,
                )
            
            # All steps completed successfully
            # Compile full report from all cells
            full_report = self._compile_full_report(
                discovery_result=discovery_result,
                red_result=red_result,
                blue_result=blue_result,
                white_result=white_result,
            )
            
            return SparringResponse(
                success=True,
                session_id=session_id,
                step_completed="pro",
                next_step=None,
                next_command=None,
                result=full_report,
                message="Pro mode (sparring3) completed successfully with full step-by-step analysis",
                error=None,
            )
            
        except Exception as e:
            logger.error(f"Pro mode execution failed: {e}", exc_info=True)
            return SparringResponse(
                success=False,
                session_id=session_id,
                step_completed="pro",
                next_step=None,
                next_command=None,
                result=None,
                message="Pro mode execution failed",
                error=str(e),
            )
    
    def _compile_full_report(
        self,
        discovery_result: SparringResponse,
        red_result: SparringResponse,
        blue_result: SparringResponse,
        white_result: SparringResponse,
    ) -> str:
        """Compile full report from all cell results."""
        report_parts = [
            "# 📊 Pełny Raport Sparring3 - Szczegółowa Analiza Krok-po-Kroku",
            "",
            "**Tryb:** `sparring3 (pro)` - Analiza krok-po-kroku z checkpointami",
            "**Cel:** Głęboka analiza strategiczna z wysokim budżetem tokenów (4096 tokens/cell)",
            "",
            "---",
            "",
        ]
        
        # Helper to extract text from result (dict or string)
        def extract_text(result) -> str:
            if not result:
                return ""
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                # Extract text from known dict keys
                for key in ["critique", "defense", "consensus", "strategy", "roles"]:
                    if key in result:
                        val = result[key]
                        return str(val) if not isinstance(val, dict) else str(val)
                # Fallback: convert entire dict to string
                return str(result)
            return str(result)
        
        # Helper to count words safely
        def count_words(text) -> int:
            if not text:
                return 0
            if isinstance(text, str):
                return len(text.split())
            return 0
        
        # Extract text from each result
        discovery_text = extract_text(discovery_result.result)
        red_text = extract_text(red_result.result)
        blue_text = extract_text(blue_result.result)
        white_text = extract_text(white_result.result)
        
        # Add each cell's result
        if discovery_text:
            report_parts.append("## 🎯 Krok 1: Discovery - Definicja Ról")
            report_parts.append("")
            report_parts.append(discovery_text)
            report_parts.append("")
            report_parts.append("---")
            report_parts.append("")
        
        if red_text:
            report_parts.append("## 🔴 Krok 2: Red Cell - Audyt Ryzyk")
            report_parts.append("")
            report_parts.append(red_text)
            report_parts.append("")
            report_parts.append("---")
            report_parts.append("")
        
        if blue_text:
            report_parts.append("## 🔵 Krok 3: Blue Cell - Obrona Strategiczna")
            report_parts.append("")
            report_parts.append(blue_text)
            report_parts.append("")
            report_parts.append("---")
            report_parts.append("")
        
        if white_text:
            report_parts.append("## ⚪ Krok 4: White Cell - Synteza i Rekomendacje")
            report_parts.append("")
            report_parts.append(white_text)
            report_parts.append("")
            report_parts.append("---")
            report_parts.append("")
        
        # Add summary table
        report_parts.extend([
            "## 📋 Podsumowanie Procesu",
            "",
            "| Krok | Rola | Status | Długość odpowiedzi |",
            "|------|------|--------|-------------------|",
            f"| 1 | Discovery | ✅ | ~{count_words(discovery_text)} słów |",
            f"| 2 | Red Cell | ✅ | ~{count_words(red_text)} słów |",
            f"| 3 | Blue Cell | ✅ | ~{count_words(blue_text)} słów |",
            f"| 4 | White Cell | ✅ | ~{count_words(white_text)} słów |",
            "",
            "**sparring3 (pro)** = Najbardziej szczegółowy tryb analizy z osobnymi wywołaniami MCP dla każdej komórki",
        ])
        
        return "\n".join(report_parts)
