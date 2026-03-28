"""
Sparring Engine v2 - Main Engine Class

This module provides the refactored sparring engine with:
- Unified interface for flash and pro modes
- Step-by-step execution (discovery, red, blue, white)
- Session checkpointing for recovery
- Guided UX with next_step hints
- Reduced timeouts to avoid MCP 300s limit
"""

import logging
import time
from typing import Optional
from mcp.server.fastmcp import Context

from qwen_mcp.api import DashScopeClient
from qwen_mcp.sanitizer import ContentValidator
from qwen_mcp.engines.session_store import SessionStore, SessionCheckpoint
from qwen_mcp.engines.sparring_v2.config import TIMEOUTS, DEFAULT_MODELS
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.helpers import (
    validate_session,
    get_model,
    get_step_result
)
from qwen_mcp.prompts.sparring import (
    get_discovery_prompt,
    FLASH_ANALYST_PROMPT,
    FLASH_DRAFTER_PROMPT,
    RED_CELL_PROMPT,
    BLUE_CELL_PROMPT,
    WHITE_CELL_PROMPT
)
from qwen_mcp.tools import extract_json_from_text
from qwen_mcp.registry import ModelEntitlementRegistry
from qwen_mcp.base import get_billing_mode

logger = logging.getLogger(__name__)


class SparringEngineV2:
    """
    Refactored Sparring Engine with step-by-step execution.
    
    Modes:
    - flash: Fast 2-step analysis (analyst → drafter)
    - discovery: Role discovery only
    - red: Red Cell critique only
    - blue: Blue Cell defense only
    - white: White Cell synthesis only
    """
    
    def __init__(self, client: Optional[DashScopeClient] = None,
                 session_store: Optional[SessionStore] = None):
        self.client = client or DashScopeClient()
        self.session_store = session_store or SessionStore()
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    async def execute(self, mode: str, topic: Optional[str] = None,
                     context: str = "", session_id: Optional[str] = None,
                     ctx: Optional[Context] = None) -> SparringResponse:
        """
        Execute a sparring step.
        
        Args:
            mode: One of 'flash', 'discovery', 'red', 'blue', 'white', 'full'
            topic: Topic for the sparring (required for flash/discovery/full)
            context: Additional context
            session_id: Session ID for step modes
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with result and next_step guidance
        """
        logger.info(f"Executing sparring mode={mode}, session_id={session_id}")
        start_time = time.time()
        
        try:
            if mode == "flash":
                return await self._execute_flash(topic, context, ctx)
            elif mode == "discovery":
                return await self._execute_discovery(topic, context, ctx)
            elif mode == "red":
                return await self._execute_red(session_id, ctx)
            elif mode == "blue":
                return await self._execute_blue(session_id, ctx)
            elif mode == "white":
                return await self._execute_white(session_id, ctx)
            elif mode == "full":
                return await self._execute_full(topic, context, ctx)
            else:
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed=None,
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Invalid mode",
                    error=f"Unknown mode: {mode}. Use: flash, discovery, red, blue, white, full"
                )
        except Exception as e:
            logger.exception(f"Sparring execution failed: {e}")
            elapsed = time.time() - start_time
            if session_id:
                self.session_store.mark_failed(session_id, str(e))
            return SparringResponse(
                success=False,
                session_id=session_id,
                step_completed=mode,
                next_step=None,
                next_command=None,
                result=None,
                message=f"Execution failed after {elapsed:.1f}s",
                error=str(e)
            )
    
    # -------------------------------------------------------------------------
    # Flash Mode
    # -------------------------------------------------------------------------
    
    async def _execute_flash(self, topic: str, context: str,
                            ctx: Optional[Context]) -> SparringResponse:
        """Execute flash mode: Analyst → Drafter (single call, no checkpoint)."""
        # Immediate heartbeat to prevent client timeout during model initialization
        await self._report_progress(ctx, 0.0, "[Flash] Initializing Analysis...")
        
        # Step 1: Analyst
        analyst_messages = [
            {"role": "system", "content": FLASH_ANALYST_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext:\n{context}"},
        ]
        
        analysis = await self.client.generate_completion(
            messages=analyst_messages,
            temperature=0.7,
            task_type="audit",
            timeout=TIMEOUTS["flash_analyst"],
            complexity="medium",
            tags=["sparring", "flash-analyst"],
            # enable_thinking=True is default - restoring full reasoning power
            progress_callback=ctx.report_progress if ctx else None  # CRITICAL: Keep MCP connection alive
        )
        
        await self._report_progress(ctx, 50.0, "[Flash] Turn 2: Drafting Strategy...")
        
        # Step 2: Drafter
        drafter_messages = [
            {"role": "system", "content": FLASH_DRAFTER_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}\n\nAnalysis:\n{analysis}"},
        ]
        
        final_strategy = await self.client.generate_completion(
            messages=drafter_messages,
            temperature=0.1,
            task_type="strategist",
            timeout=TIMEOUTS["flash_drafter"],
            complexity="medium",
            tags=["sparring", "flash-drafter"],
            include_reasoning=False,
            progress_callback=ctx.report_progress if ctx else None  # CRITICAL: Keep MCP connection alive
        )
        
        # Format output
        output = self._format_output(final_strategy, "Flash Analysis")
        
        return SparringResponse(
            success=True,
            session_id=None,
            step_completed="flash",
            next_step=None,
            next_command=None,
            result={"strategy": output},
            message="Flash analysis complete"
        )
    
    # -------------------------------------------------------------------------
    # Discovery Mode
    # -------------------------------------------------------------------------
    
    async def _execute_discovery(self, topic: str, context: str,
                                 ctx: Optional[Context]) -> SparringResponse:
        """Execute discovery mode: Define roles and create session."""
        await self._report_progress(ctx, 0.0, "[Discovery] Assembling Expert Bench...")
        
        # Get billing mode for model selection
        billing_mode = get_billing_mode()
        discovery_prompt = get_discovery_prompt(billing_mode)
        
        discovery_messages = [
            {"role": "system", "content": discovery_prompt},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}"},
        ]
        
        # Use fast model for discovery - it's just JSON extraction
        discovery_raw = await self.client.generate_completion(
            messages=discovery_messages,
            temperature=0.0,
            task_type="scout",  # Use scout (kimi-k2.5) for fast JSON extraction
            timeout=TIMEOUTS["discovery"],
            complexity="low",
            tags=["sparring", "discovery"]
        )
        
        # Parse roles and models
        try:
            parsed = extract_json_from_text(discovery_raw)
            required_keys = ["red_role", "red_profile", "blue_role", "blue_profile",
                           "white_role", "white_profile"]
            if not parsed or not all(k in parsed for k in required_keys):
                raise ValueError("Incomplete role discovery")
            
            # Extract roles
            roles = {k: v for k, v in parsed.items() if not k.endswith("_model")}
            
            # Extract models with validation against billing mode
            raw_models = {
                "red_model": parsed.get("red_model", "glm-5"),
                "blue_model": parsed.get("blue_model", "qwen3.5-plus"),
                "white_model": parsed.get("white_model", "qwen3.5-plus"),
            }
            
            # Validate each model against billing mode
            models = {}
            for role_key, model_id in raw_models.items():
                is_valid, result = ModelEntitlementRegistry.validate_override(model_id, billing_mode)
                if is_valid:
                    models[role_key] = model_id
                else:
                    # Use fallback model
                    task_type = "analyst" if role_key == "red_model" else "strategist"
                    fallback = ModelEntitlementRegistry.get_fallback_model(billing_mode, task_type)
                    models[role_key] = fallback
                    logger.warning(f"[Discovery] {result}. Using fallback: {fallback}")
            
        except Exception:
            # Fallback to default roles and models
            roles = {
                "red_role": "Red Cell (Adversarial Audit)",
                "red_profile": "Cyniczny audytor metod i niuansów persony",
                "blue_role": "Blue Cell (Strategic Defense)",
                "blue_profile": "Adwokat użytkownika i autentyczności tonu",
                "white_role": "White Cell (Final Consensus)",
                "white_profile": "Chief of Staff dbający o logiczną spójność i ROI"
            }
            # Use validated fallbacks
            models = {
                "red_model": ModelEntitlementRegistry.get_fallback_model(billing_mode, "analyst"),
                "blue_model": ModelEntitlementRegistry.get_fallback_model(billing_mode, "strategist"),
                "white_model": ModelEntitlementRegistry.get_fallback_model(billing_mode, "strategist"),
            }
            logger.warning("Using default roles and models due to discovery parse failure")
        
        # Create session
        session = self.session_store.create_session(topic=topic, context=context)
        session.roles = roles
        session.models = models
        self.session_store.save(session)
        
        return SparringResponse(
            success=True,
            session_id=session.session_id,
            step_completed="discovery",
            next_step="red",
            next_command=f"sparring(session_id='{session.session_id}', mode='red')",
            result={"roles": roles, "models": models},
            message="Discovery complete. Session created."
        )
    
    # -------------------------------------------------------------------------
    # Red Cell Mode
    # -------------------------------------------------------------------------
    
    async def _execute_red(self, session_id: Optional[str],
                          ctx: Optional[Context]) -> SparringResponse:
        """Execute Red Cell critique."""
        # Validate session
        loaded_session = self.session_store.load(session_id) if session_id else None
        session, error_response = validate_session(loaded_session, session_id, "red")
        if error_response:
            return error_response
        
        # Checkpoint: Mark step as in-progress
        session.current_step = "red"
        self.session_store.save(session)
        
        await self._report_progress(ctx, 0.0, f"[Red Cell] {session.roles.get('red_role', 'Red')} auditing...")
        
        # Build messages
        red_messages = [
            {"role": "system", "content": f"Jesteś {session.roles.get('red_role', 'Red Cell')}. Profil: {session.roles.get('red_profile', '')}\n\nZADANIE:\n{RED_CELL_PROMPT}"},
            {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}"},
        ]
        
        # Execute API call
        red_model = get_model(session, "red_model")
        logger.debug(f"[Red Cell] red_model: {red_model}, roles type: {type(session.roles)}, roles: {session.roles}")
        red_critique = await self.client.generate_completion(
            messages=red_messages,
            temperature=0.8,
            task_type="audit",
            timeout=TIMEOUTS["red_cell"],
            complexity="high",  # Dynamic max_tokens: 1800
            tags=["sparring", "red-cell"],
            include_reasoning=True,
            model_override=red_model,
            enable_thinking=True
        )
        red_critique = ContentValidator.validate_response(red_critique)
        
        # Update session
        self.session_store.update_step(
            session_id, "red",
            {"critique": red_critique, "raw": red_critique},
            next_step="blue"
        )
        
        return SparringResponse.success(
            session_id=session_id,
            step="red",
            next_step="blue",
            next_command=f"sparring(session_id='{session_id}', mode='blue')",
            result={"critique": red_critique},
            message="Red Cell critique complete"
        )
    
    # -------------------------------------------------------------------------
    # Blue Cell Mode
    # -------------------------------------------------------------------------
    
    async def _execute_blue(self, session_id: Optional[str],
                           ctx: Optional[Context]) -> SparringResponse:
        """Execute Blue Cell defense."""
        # Validate session
        loaded_session = self.session_store.load(session_id) if session_id else None
        session, error_response = validate_session(loaded_session, session_id, "blue")
        if error_response:
            return error_response
        
        # Get red critique prerequisite
        red_critique = get_step_result(session, "red", "critique")
        if not red_critique:
            return SparringResponse.error(
                message="Red Cell critique not found. Run discovery and red first.",
                error="Missing red critique",
                session_id=session_id
            )
        
        # Checkpoint: Mark step as in-progress
        session.current_step = "blue"
        self.session_store.save(session)
        
        await self._report_progress(ctx, 0.0, f"[Blue Cell] {session.roles.get('blue_role', 'Blue')} defending...")
        
        # Build messages
        blue_messages = [
            {"role": "system", "content": f"Jesteś {session.roles.get('blue_role', 'Blue Cell')}. Profil: {session.roles.get('blue_profile', '')}\n\nZADANIE:\n{BLUE_CELL_PROMPT}"},
            {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}\n\nRed Critique:\n{red_critique}"},
        ]
        
        # Execute API call
        blue_model = get_model(session, "blue_model")
        blue_defense = await self.client.generate_completion(
            messages=blue_messages,
            temperature=0.5,
            task_type="audit",
            timeout=TIMEOUTS["blue_cell"],
            complexity="high",  # Dynamic max_tokens: 1800
            tags=["sparring", "blue-cell"],
            include_reasoning=True,
            model_override=blue_model,
            enable_thinking=True
        )
        blue_defense = ContentValidator.validate_response(blue_defense)
        
        # Update session
        self.session_store.update_step(
            session_id, "blue",
            {"defense": blue_defense, "raw": blue_defense},
            next_step="white"
        )
        
        return SparringResponse.success(
            session_id=session_id,
            step="blue",
            next_step="white",
            next_command=f"sparring(session_id='{session_id}', mode='white')",
            result={"defense": blue_defense},
            message="Blue Cell defense complete"
        )
    
    # -------------------------------------------------------------------------
    # White Cell Mode
    # -------------------------------------------------------------------------
    
    async def _execute_white(self, session_id: Optional[str],
                            ctx: Optional[Context],
                            allow_regeneration: bool = True) -> SparringResponse:
        """Execute White Cell synthesis with optional regeneration loop."""
        # Validate session
        loaded_session = self.session_store.load(session_id) if session_id else None
        session, error_response = validate_session(loaded_session, session_id, "white")
        if error_response:
            return error_response
        
        # Get prerequisites
        red_critique = get_step_result(session, "red", "critique")
        blue_defense = get_step_result(session, "blue", "defense")
        
        if not red_critique or not blue_defense:
            return SparringResponse.error(
                message="Red/Blue results not found. Complete previous steps first.",
                error="Missing prerequisites",
                session_id=session_id
            )
        
        # Checkpoint: Mark step as in-progress
        session.current_step = "white"
        self.session_store.save(session)
        
        # Regeneration loop (disabled in full mode to avoid MCP 300s timeout)
        max_loops = 1 if not allow_regeneration else 2
        loop_count = session.loop_count
        white_consensus = ""
        
        # DEBUG: Log loop_count type
        logger.debug(f"[White Cell] loop_count type: {type(loop_count)}, value: {loop_count}")
        logger.debug(f"[White Cell] max_loops type: {type(max_loops)}, value: {max_loops}")
        
        while loop_count < max_loops:
            loop_count += 1
            await self._report_progress(ctx, 0.0, f"[White Cell] {session.roles.get('white_role', 'White')} synthesizing (loop {loop_count})...")
            
            white_messages = [
                {"role": "system", "content": f"Jesteś {session.roles.get('white_role', 'White Cell')}. Profil: {session.roles.get('white_profile', '')}\n\nZADANIE:\n{WHITE_CELL_PROMPT}"},
                {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}\n\nRed Audit:\n{red_critique}\n\nBlue Defense:\n{blue_defense}"},
            ]
            
            # Execute API call
            white_model = get_model(session, "white_model")
            white_consensus = await self.client.generate_completion(
                messages=white_messages,
                temperature=0.1,
                task_type="audit",
                timeout=TIMEOUTS["white_cell"],
                complexity="critical",  # Dynamic max_tokens: 2500
                tags=["sparring", "white-cell", f"loop-{loop_count}"],
                include_reasoning=True,
                model_override=white_model,
                enable_thinking=True
            )
            white_consensus = ContentValidator.validate_response(white_consensus)
            
            # Check for regeneration request
            if "[REGENERATE" not in white_consensus or loop_count >= max_loops:
                if "[REGENERATE" in white_consensus:
                    white_consensus = white_consensus.split("]", 1)[-1].strip()
                break
            
            # If regeneration requested, get new blue defense
            await self._report_progress(ctx, 50.0, "[White Cell] Requesting Blue Cell regeneration...")
            
            regen_reason = white_consensus.split("[REGENERATE:", 1)[1].split("]", 1)[0].strip() if "[REGENERATE:" in white_consensus else "Improvement needed"
            
            blue_messages = [
                {"role": "system", "content": f"Jesteś {session.roles.get('blue_role', 'Blue Cell')}. Profil: {session.roles.get('blue_profile', '')}\n\nZADANIE:\n{BLUE_CELL_PROMPT}\n\nUWAGA: Poprzednia próba odrzucona: {regen_reason}"},
                {"role": "user", "content": f"Topic: {session.topic}\n\nContext: {session.context}\n\nRed Critique:\n{red_critique}"},
            ]
            
            # Execute regeneration API call
            regen_blue_model = get_model(session, "blue_model")
            blue_defense = await self.client.generate_completion(
                messages=blue_messages,
                temperature=0.5,
                task_type="audit",
                timeout=TIMEOUTS["blue_cell"],
                complexity="high",  # Dynamic max_tokens: 1800
                tags=["sparring", "blue-cell", "regen"],
                include_reasoning=True,
                model_override=regen_blue_model,
                enable_thinking=True
            )
            blue_defense = ContentValidator.validate_response(blue_defense)
            
            # Update session with new blue defense
            session.results["blue"] = {"defense": blue_defense, "raw": blue_defense}
            self.session_store.save(session)
        
        # Update session
        session.loop_count = loop_count
        self.session_store.update_step(
            session_id, "white",
            {"consensus": white_consensus, "raw": white_consensus, "loops": loop_count}
        )
        
        # Assemble final report
        report = self._assemble_report(session, red_critique, blue_defense, white_consensus, loop_count)
        
        return SparringResponse.success(
            session_id=session_id,
            step="white",
            next_step=None,
            next_command=None,
            result={"consensus": white_consensus, "report": report, "loops": loop_count},
            message=f"White Cell synthesis complete ({loop_count} loop{'s' if loop_count > 1 else ''})"
        )
    
    # -------------------------------------------------------------------------
    # Full Mode - Execute entire session in one call
    # -------------------------------------------------------------------------
    
    async def _execute_full(self, topic: str, context: str,
                           ctx: Optional[Context]) -> SparringResponse:
        """Execute complete sparring session in one call: discovery→red→blue→white."""
        session_id = None
        
        try:
            # Krok 1: Discovery (25%)
            await self._report_progress(ctx, 25.0, "[Full] 1/4: Discovery - Defining roles...")
            discovery_result = await self._execute_discovery(topic, context, ctx)
            session_id = discovery_result.session_id
            
            if not session_id:
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed="full",
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Full mode failed at discovery step",
                    error=discovery_result.error or "No session_id returned"
                )
            
            # Krok 2: Red Cell (50%)
            await self._report_progress(ctx, 50.0, "[Full] 2/4: Red Cell - Auditing...")
            red_result = await self._execute_red(session_id, ctx)
            
            if not red_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step="red",
                    next_command=f'qwen_sparring(mode="red", session_id="{session_id}")',
                    result=None,
                    message="Full mode failed at red cell step",
                    error=red_result.error
                )
            
            # Krok 3: Blue Cell (75%)
            await self._report_progress(ctx, 75.0, "[Full] 3/4: Blue Cell - Defending...")
            blue_result = await self._execute_blue(session_id, ctx)
            
            if not blue_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step="blue",
                    next_command=f'qwen_sparring(mode="blue", session_id="{session_id}")',
                    result=None,
                    message="Full mode failed at blue cell step",
                    error=blue_result.error
                )
            
            # Krok 4: White Cell (100%) - disable regeneration to avoid MCP 300s timeout
            await self._report_progress(ctx, 100.0, "[Full] 4/4: White Cell - Synthesizing (no regen)...")
            white_result = await self._execute_white(session_id, ctx, allow_regeneration=False)
            
            if not white_result.success:
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step="white",
                    next_command=f'qwen_sparring(mode="white", session_id="{session_id}")',
                    result=None,
                    message="Full mode failed at white cell step",
                    error=white_result.error
                )
            
            # Złóż finalny raport
            final_report = self._assemble_full_report(
                discovery_result, red_result, blue_result, white_result
            )
            
            return SparringResponse(
                success=True,
                session_id=session_id,
                step_completed="full",
                next_step=None,
                next_command=None,
                result={"full_report": final_report},
                message="Full sparring session complete"
            )
            
        except Exception as e:
            # Handle exceptions at each step
            logger.exception(f"Full mode failed: {e}")
            if session_id is None:
                # Failed at discovery
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed="full",
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Full mode failed at discovery step",
                    error=str(e)
                )
            else:
                # Need to determine which step failed based on session state
                session = self.session_store.load(session_id)
                if session:
                    if "red" not in session.steps_completed:
                        return SparringResponse(
                            success=False,
                            session_id=session_id,
                            step_completed="full",
                            next_step="red",
                            next_command=f'qwen_sparring(mode="red", session_id="{session_id}")',
                            result=None,
                            message="Full mode failed at red cell step",
                            error=str(e)
                        )
                    elif "blue" not in session.steps_completed:
                        return SparringResponse(
                            success=False,
                            session_id=session_id,
                            step_completed="full",
                            next_step="blue",
                            next_command=f'qwen_sparring(mode="blue", session_id="{session_id}")',
                            result=None,
                            message="Full mode failed at blue cell step",
                            error=str(e)
                        )
                    else:
                        return SparringResponse(
                            success=False,
                            session_id=session_id,
                            step_completed="full",
                            next_step="white",
                            next_command=f'qwen_sparring(mode="white", session_id="{session_id}")',
                            result=None,
                            message="Full mode failed at white cell step",
                            error=str(e)
                        )
                # Fallback
                return SparringResponse(
                    success=False,
                    session_id=session_id,
                    step_completed="full",
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Full mode failed",
                    error=str(e)
                )
    
    def _assemble_full_report(self, discovery: SparringResponse,
                             red: SparringResponse, blue: SparringResponse,
                             white: SparringResponse) -> str:
        """Assemble full session report from all 4 steps."""
        session_id = discovery.session_id
        
        # Load session to get topic and roles
        session = self.session_store.load(session_id)
        if not session:
            return "# Error: Session not found"
        
        # Extract results from each step
        red_content = red.result.get('critique', '') if red.result else ''
        blue_content = blue.result.get('defense', '') if blue.result else ''
        white_content = white.result.get('consensus', '') if white.result else ''
        
        roles = session.roles if session.roles else {}
        
        report = f"# 🛡️ War Game Report: {session.topic}\n\n"
        report += f"> **Session ID:** `{session_id}`\n\n"
        report += f"> **Selected Roles:** {roles.get('red_role', 'Red')}, {roles.get('blue_role', 'Blue')}, {roles.get('white_role', 'White')}\n\n"
        
        # Format each section - handle empty content gracefully
        if red_content:
            report += f"## 🥊 Turn 2: {roles.get('red_role', 'Red')}\n\n{self._format_output(red_content, roles.get('red_role', 'Red'))}\n\n---\n\n"
        else:
            report += f"## 🥊 Turn 2: {roles.get('red_role', 'Red')}\n\n*No content*\n\n---\n\n"
        
        if blue_content:
            report += f"## 🛡️ Turn 3: {roles.get('blue_role', 'Blue')}\n\n{self._format_output(blue_content, roles.get('blue_role', 'Blue'))}\n\n---\n\n"
        else:
            report += f"## 🛡️ Turn 3: {roles.get('blue_role', 'Blue')}\n\n*No content*\n\n---\n\n"
        
        if white_content:
            report += f"## ⚖️ Turn 4: {roles.get('white_role', 'White')}\n\n{self._format_output(white_content, roles.get('white_role', 'White'))}\n\n"
        else:
            report += f"## ⚖️ Turn 4: {roles.get('white_role', 'White')}\n\n*No content*\n\n"
        
        return report
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    
    def _format_output(self, raw: str, label: str) -> str:
        """Format output with reasoning hidden in details."""
        # Ensure raw is a string (handle None, int, or other types gracefully)
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        
        if "<thought>" in raw:
            parts = raw.split("</thought>")
            thought = parts[0].replace("<thought>", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            return f"<details>\n<summary>🧠 Proces Myślowy ({label})</summary>\n\n{thought}\n</details>\n\n{content}"
        return ContentValidator.validate_response(raw)
    
    def _assemble_report(self, session: SessionCheckpoint, red: str, blue: str,
                        white: str, loops: int) -> str:
        """Assemble final war game report."""
        report = f"# 🛡️ War Game Report: {session.topic}\n\n"
        report += f"> **CONFIDENTIAL: STRATEGIC DRAFT ONLY. NOT FOR EXTERNAL DISTRIBUTION.**\n\n"
        report += f"> **Session ID:** `{session.session_id}`\n\n"
        report += f"> **Selected Roles:** {session.roles.get('red_role', 'Red')}, {session.roles.get('blue_role', 'Blue')}, {session.roles.get('white_role', 'White')}\n\n"
        report += f"## 🥊 Turn 2: {session.roles.get('red_role', 'Red')}\n\n{self._format_output(red, session.roles.get('red_role', 'Red'))}\n\n---\n\n"
        report += f"## 🛡️ Turn 3: {session.roles.get('blue_role', 'Blue')}\n\n{self._format_output(blue, session.roles.get('blue_role', 'Blue'))}\n\n---\n\n"
        report += f"## ⚖️ Turn 4: {session.roles.get('white_role', 'White')}\n\n{self._format_output(white, session.roles.get('white_role', 'White'))}\n\n"
        # Ensure loops is an int (handle JSON deserialization edge cases)
        loops_int = int(loops) if not isinstance(loops, int) else loops
        if loops_int > 1:
            report += f"\n\n*(Note: This report underwent {loops_int} optimization cycles)*"
        return report
    
    async def _report_progress(self, ctx: Optional[Context], 
                               progress: float, message: str) -> None:
        """Safe progress reporting."""
        if ctx:
            try:
                await ctx.report_progress(progress=float(progress), message=message)
            except Exception:
                pass