"""
Sparring Engine v2 - Main Engine Class

Refactored engine with orchestration-only responsibility.
Mode-specific logic is delegated to mode executors.

This module provides:
- Central orchestration for all sparring modes
- Mode executor factory and dispatch
- Session lifecycle management
"""

import logging
import time
from typing import Optional, Dict, Type

from mcp.server.fastmcp import Context

from qwen_mcp.api import DashScopeClient
from qwen_mcp.engines.session_store import SessionStore
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.interfaces import ModeExecutor
from qwen_mcp.engines.sparring_v2.modes.flash import FlashExecutor
from qwen_mcp.engines.sparring_v2.modes.discovery import DiscoveryExecutor
from qwen_mcp.engines.sparring_v2.modes.red_cell import RedCellExecutor
from qwen_mcp.engines.sparring_v2.modes.blue_cell import BlueCellExecutor
from qwen_mcp.engines.sparring_v2.modes.white_cell import WhiteCellExecutor
from qwen_mcp.engines.sparring_v2.modes.full import FullExecutor
from qwen_mcp.engines.sparring_v2.modes.pro import ProExecutor

logger = logging.getLogger(__name__)


class SparringEngineV2:
    """
    Refactored Sparring Engine with orchestration-only responsibility.
    
    Delegates mode-specific execution to dedicated executors:
    - FlashExecutor: Fast 2-step analysis
    - DiscoveryExecutor: Role discovery
    - RedCellExecutor: Adversarial critique
    - BlueCellExecutor: Strategic defense
    - WhiteCellExecutor: Consensus synthesis
    - FullExecutor: End-to-end session (sparring2)
    - ProExecutor: Step-by-step with checkpoints (sparring3)
    """
    
    # Mode executor mapping
    MODE_EXECUTORS: Dict[str, Type[ModeExecutor]] = {
        "flash": FlashExecutor,
        "discovery": DiscoveryExecutor,
        "red": RedCellExecutor,
        "blue": BlueCellExecutor,
        "white": WhiteCellExecutor,
        "full": FullExecutor,
        "pro": ProExecutor,  # sparring3: step-by-step with higher tokens/timeout
    }
    
    def __init__(self, client: Optional[DashScopeClient] = None,
                 session_store: Optional[SessionStore] = None):
        """
        Initialize the sparring engine.
        
        Args:
            client: DashScopeClient for API calls (default: new instance)
            session_store: SessionStore for checkpoint management (default: new instance)
        """
        self.client = client or DashScopeClient()
        self.session_store = session_store or SessionStore()
    
    async def execute(self, mode: str, topic: Optional[str] = None,
                     context: str = "", session_id: Optional[str] = None,
                     ctx: Optional[Context] = None) -> SparringResponse:
        """
        Execute a sparring mode by delegating to the appropriate executor.
        
        Args:
            mode: One of 'flash', 'discovery', 'red', 'blue', 'white', 'full'
            topic: Topic for the sparring (required for flash/discovery/full)
            context: Additional context
            session_id: Session ID for step modes
            ctx: MCP context for progress reporting
            
        Returns:
            SparringResponse with execution results
        """
        logger.info(f"Executing sparring mode={mode}, session_id={session_id}")
        start_time = time.time()
        
        # Get executor class
        executor_class = self.MODE_EXECUTORS.get(mode)
        if not executor_class:
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
        
        # Create executor and execute
        executor = executor_class(self.client, self.session_store)
        
        try:
            # Dispatch to executor with appropriate arguments
            if mode == "flash":
                return await executor.execute(topic=topic, context=context, ctx=ctx)
            elif mode == "discovery":
                return await executor.execute(topic=topic, context=context, ctx=ctx)
            elif mode == "red":
                return await executor.execute(session_id=session_id, ctx=ctx)
            elif mode == "blue":
                return await executor.execute(session_id=session_id, ctx=ctx)
            elif mode == "white":
                return await executor.execute(session_id=session_id, ctx=ctx)
            elif mode == "full":
                return await executor.execute(topic=topic, context=context, ctx=ctx)
            elif mode == "pro":
                # sparring3: Start step-by-step with discovery
                return await executor.execute(topic=topic, context=context, ctx=ctx)
            else:
                return SparringResponse(
                    success=False,
                    session_id=None,
                    step_completed=None,
                    next_step=None,
                    next_command=None,
                    result=None,
                    message="Invalid mode",
                    error=f"Unknown mode: {mode}"
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
