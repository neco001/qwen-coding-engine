"""
Sparring Engine v2 - Helper Methods

This module provides helper methods for session validation, model selection,
and step result extraction used by the SparringEngineV2.
"""

from typing import Optional, Tuple

from qwen_mcp.engines.session_store import SessionCheckpoint
from qwen_mcp.engines.sparring_v2.models import SparringResponse
from qwen_mcp.engines.sparring_v2.config import DEFAULT_MODELS


def validate_session(session_checkpoint: Optional[SessionCheckpoint], 
                     session_id: Optional[str],
                     step_name: str) -> Tuple[Optional[SessionCheckpoint], Optional[SparringResponse]]:
    """
    Validate session exists and return it, or return error response.
    
    Args:
        session_checkpoint: The loaded session checkpoint (or None)
        session_id: The session ID that was requested
        step_name: Name of the step being executed (for error message)
    
    Returns:
        Tuple of (session, error_response) - one will be None
    """
    if not session_id:
        return None, SparringResponse.error(
            message=f"session_id required for {step_name} mode",
            error="Missing session_id"
        )
    if not session_checkpoint:
        return None, SparringResponse.error(
            message=f"Session not found: {session_id}",
            error="Session not found",
            session_id=session_id
        )
    return session_checkpoint, None


def get_model(session: Optional[SessionCheckpoint], key: str) -> str:
    """
    Get model from session or return default.
    
    Args:
        session: The session checkpoint (may be None)
        key: The model key to look up (e.g., "red_model")
    
    Returns:
        Model ID string
    """
    if session and session.models:
        return session.models.get(key, DEFAULT_MODELS.get(key, "qwen3.5-plus"))
    return DEFAULT_MODELS.get(key, "qwen3.5-plus")


def get_step_result(session: SessionCheckpoint, step: str, 
                    primary_field: str) -> Optional[str]:
    """
    Safely extract result from previous step.
    
    Args:
        session: The session checkpoint
        step: The step name (e.g., "red", "blue")
        primary_field: The primary field to extract (e.g., "critique", "defense")
    
    Returns:
        The extracted result string or None
    """
    result = session.results.get(step, {})
    return result.get(primary_field, result.get("raw", "")) or None