"""
Sparring Engine v2 - Interfaces

This module defines abstract base classes for mode executors and formatters,
enforcing a consistent interface across all sparring modes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from mcp.server.fastmcp import Context

from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.session_store import SessionCheckpoint


class ModeExecutor(ABC):
    """
    Abstract base class for all sparring mode executors.
    
    Each mode (flash, discovery, red, blue, white, full) must implement
    this interface to ensure consistent execution and response handling.
    """
    
    def __init__(self, client, session_store):
        """
        Initialize the mode executor.
        
        Args:
            client: DashScopeClient instance for API calls
            session_store: SessionStore instance for checkpoint management
        """
        self.client = client
        self.session_store = session_store
    
    async def _report_progress(self, ctx: Optional[Context], progress: float, message: str) -> None:
        """
        Safe progress reporting with dual broadcast (MCP + WebSocket for Roo Code HUD).
        
        Args:
            ctx: MCP context for progress reporting
            progress: Progress percentage (0-100)
            message: Status message
        """
        if ctx:
            try:
                # MCP progress notification (protocol compliance)
                await ctx.report_progress(progress=float(progress), message=message)
            except Exception:
                pass
        
        # WebSocket broadcast for Roo Code HUD visibility
        try:
            from qwen_mcp.specter.telemetry import get_broadcaster
            await get_broadcaster().broadcast_state({
                "operation": message,
                "progress": float(progress),
                "is_live": True
            }, project_id="sparring")
        except Exception:
            pass
    
    @abstractmethod
    async def execute(self, ctx: Optional[Context] = None, **kwargs) -> SparringResponse:
        """
        Execute the sparring mode.
        
        Args:
            ctx: MCP context for progress reporting
            **kwargs: Mode-specific arguments (topic, context, session_id, etc.)
            
        Returns:
            SparringResponse with execution results
        """
        pass


class ReportFormatter(ABC):
    """
    Abstract base class for report formatters.
    
    Responsible for assembling and formatting sparring session reports.
    """
    
    @abstractmethod
    def format(self, session: SessionCheckpoint, results: Dict[str, Any]) -> str:
        """
        Format a sparring session report.
        
        Args:
            session: SessionCheckpoint with session data
            results: Dictionary of step results
            
        Returns:
            Formatted report string
        """
        pass
