"""Session Isolation Manager for AI-Driven Testing System.

Enforces context separation between Coder, Test, and Validator sessions.
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from qwen_mcp.engines.session_store import SessionStore, SessionCheckpoint
from .prompts import ROLE_CONFIG, CODER_PROMPT, TEST_PROMPT, VALIDATOR_PROMPT


class IsolationViolationError(Exception):
    """Raised when session isolation is breached."""
    pass


@dataclass
class SessionConfig:
    """Configuration for a session role."""
    role: str
    prompt: str
    context_window: int
    temperature: float
    forbidden_contexts: List[str] = field(default_factory=list)


class SessionIsolationManager:
    """Manages isolated sessions for Coder, Test, and Validator roles.
    
    Features:
    - Distinct system prompts per role
    - Context window isolation
    - Forbidden context enforcement
    - Session validation
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize Session Isolation Manager.
        
        Args:
            storage_dir: Directory for session storage (uses SessionStore)
        """
        self.session_store = SessionStore(storage_dir=storage_dir)
        self._active_sessions: Dict[str, SessionConfig] = {}
    
    def get_system_prompt(self, role: str) -> str:
        """Get system prompt for a role.
        
        Args:
            role: Role name (coder, test, validator)
            
        Returns:
            System prompt string
            
        Raises:
            IsolationViolationError: If role is unknown
        """
        if role not in ROLE_CONFIG:
            raise IsolationViolationError(f"Unknown role: {role}")
        
        return ROLE_CONFIG[role]["prompt"]
    
    def get_context_window(self, role: str) -> int:
        """Get context window size for a role.
        
        Args:
            role: Role name (coder, test, validator)
            
        Returns:
            Context window size in tokens
        """
        if role not in ROLE_CONFIG:
            return 8192  # Default
        return ROLE_CONFIG[role]["context_window"]
    
    def get_temperature(self, role: str) -> float:
        """Get temperature for a role.
        
        Args:
            role: Role name (coder, test, validator)
            
        Returns:
            Temperature value (0.0-1.0)
        """
        if role not in ROLE_CONFIG:
            return 0.7  # Default
        return ROLE_CONFIG[role]["temperature"]
    
    def create_session(self, role: str, topic: str, context: str = "") -> SessionCheckpoint:
        """Create a new isolated session.
        
        Args:
            role: Role name (coder, test, validator)
            topic: Session topic
            context: Optional context string
            
        Returns:
            SessionCheckpoint for the new session
            
        Raises:
            IsolationViolationError: If role is unknown or context violation detected
        """
        if role not in ROLE_CONFIG:
            raise IsolationViolationError(f"Unknown role: {role}")
        
        # Validate no forbidden contexts are present
        config = ROLE_CONFIG[role]
        for forbidden in config["forbidden_contexts"]:
            if forbidden in context.lower():
                raise IsolationViolationError(
                    f"Context violation: {role} session cannot access { forbidden} context"
                )
        
        # Create session with role-specific prompt
        full_context = f"{config['prompt']}\n\n---\n\n{context}"
        
        checkpoint = self.session_store.create_session(
            topic=topic,
            context=full_context
        )
        
        # Store session config
        self._active_sessions[checkpoint.session_id] = SessionConfig(
            role=role,
            prompt=config["prompt"],
            context_window=config["context_window"],
            temperature=config["temperature"],
            forbidden_contexts=config["forbidden_contexts"],
        )
        
        return checkpoint
    
    def validate_isolation(self, session_id: str) -> bool:
        """Validate that a session maintains isolation.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if isolation is maintained
            
        Raises:
            IsolationViolationError: If isolation is breached
        """
        if session_id not in self._active_sessions:
            raise IsolationViolationError(f"Unknown session: {session_id}")
        
        config = self._active_sessions[session_id]
        
        # Load session from store
        checkpoint = self.session_store.load(session_id)
        if not checkpoint:
            raise IsolationViolationError(f"Session not found: {session_id}")
        
        # Extract user-provided context (after the separator)
        # System prompt is before "\n\n---\n\n", user context is after
        separator = "\n\n---\n\n"
        if separator in checkpoint.context:
            user_context = checkpoint.context.split(separator, 1)[1]
        else:
            user_context = checkpoint.context
        
        # Check only user context for forbidden content
        for forbidden in config.forbidden_contexts:
            if forbidden in user_context.lower():
                raise IsolationViolationError(
                    f"Isolation breach: {config.role} session contains forbidden {forbidden} context"
                )
        
        return True
    
    def get_session_config(self, session_id: str) -> SessionConfig:
        """Get configuration for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionConfig for the session
            
        Raises:
            IsolationViolationError: If session not found
        """
        if session_id not in self._active_sessions:
            raise IsolationViolationError(f"Unknown session: {session_id}")
        return self._active_sessions[session_id]
    
    async def execute_isolated(
        self,
        session_id: str,
        prompt: str,
        model_client: Any,
    ) -> str:
        """Execute a prompt in an isolated session.
        
        Args:
            session_id: Session ID
            prompt: User prompt
            model_client: DashScope client or similar
            
        Returns:
            Model response string
            
        Raises:
            IsolationViolationError: If isolation validation fails
        """
        # Validate isolation before execution
        self.validate_isolation(session_id)
        
        config = self.get_session_config(session_id)
        
        # Get system prompt
        system_prompt = self.get_system_prompt(config.role)
        
        # Execute with role-specific parameters
        # (This would integrate with the actual model client)
        # For now, return a placeholder
        return f"[{config.role}] Response to: {prompt}"
    
    def list_active_sessions(self) -> List[Dict[str, str]]:
        """List all active sessions.
        
        Returns:
            List of session info dictionaries
        """
        return [
            {
                "session_id": sid,
                "role": config.role,
                "forbidden_contexts": ",".join(config.forbidden_contexts),
            }
            for sid, config in self._active_sessions.items()
        ]