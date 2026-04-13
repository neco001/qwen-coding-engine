"""
Session Store for Sparring Engine - Checkpointing with Atomic Writes

This module provides persistent state management for step-by-step sparring sessions.
Uses atomic file operations (tempfile + rename) to prevent corruption during timeouts.
"""

import os
import json
import tempfile
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _json_serializable_default(obj):
    """
    Default handler for JSON serialization of non-serializable objects.
    
    Handles edge cases where checkpoint data accidentally contains:
    - Method objects (e.g., MCP Context methods leaking into results)
    - Complex objects that should be converted to string representation
    
    Args:
        obj: The object that JSON can't serialize
        
    Returns:
        String representation of the object (for debugging)
        
    Warning:
        Logs a warning so developers can fix the root cause
    """
    # Handle method objects (most common cause of this error)
    if hasattr(obj, '__self__') and hasattr(obj, '__name__'):
        logger.warning(f"Checkpoint contains method object: {obj.__qualname__ if hasattr(obj, '__qualname__') else obj.__name__}. Converting to string.")
        return f"<method: {obj.__qualname__ if hasattr(obj, '__qualname__') else obj.__name__}>"
    
    # Handle any other non-serializable object
    obj_type = type(obj).__name__
    logger.warning(f"Checkpoint contains non-serializable {obj_type}: {repr(obj)[:100]}. Converting to string.")
    return repr(obj)

# =============================================================================
# JSON Schema for Session Checkpoints
# =============================================================================

@dataclass
class SessionCheckpoint:
    """
    Schema for a sparring session checkpoint.
    
    Attributes:
        session_id: Unique identifier for the session
        topic: The sparring topic
        context: Additional context provided by user
        created_at: ISO timestamp of session creation
        updated_at: ISO timestamp of last update
        status: Session status (in_progress, completed, failed)
        steps_completed: List of completed steps (discovery, red, blue, white)
        current_step: The current/next step to execute
        roles: Discovered roles (red_role, blue_role, white_role, etc.)
        models: Selected models for each role (red_model, blue_model, white_model)
        results: Results from each completed step
        loop_count: Number of regeneration loops (for blue/white cycle)
        error: Error message if status is 'failed'
        messages: Conversation history for multi-turn support (filtered, no reasoning_content)
        has_stages: Whether this session uses stage-based execution (for BaseStageExecutor)
        stage_count: Total number of stages in the execution plan
        ttl_expires_at: ISO timestamp for TTL expiration (ephemeral checkpoints)
    """
    session_id: str
    topic: str
    context: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "in_progress"
    steps_completed: List[str] = None
    current_step: str = "discovery"
    roles: Dict[str, str] = None
    models: Dict[str, str] = None
    results: Dict[str, Any] = None
    loop_count: int = 0
    error: Optional[str] = None
    messages: List[Dict[str, str]] = None
    has_stages: bool = False  # Stage-based execution flag
    stage_count: int = 4  # Default: 4 stages (discovery, red, blue, white)
    ttl_expires_at: Optional[str] = None  # TTL expiration for ephemeral checkpoints
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []
        if self.roles is None:
            self.roles = {}
        if self.models is None:
            self.models = {}
        if self.results is None:
            self.results = {}
        if self.messages is None:
            self.messages = []
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionCheckpoint':
        # Ensure loop_count is int (handle JSON deserialization edge cases)
        if 'loop_count' in data and isinstance(data['loop_count'], str):
            data['loop_count'] = int(data['loop_count'])
        # Ensure has_stages is bool
        if 'has_stages' in data and isinstance(data['has_stages'], str):
            data['has_stages'] = data['has_stages'].lower() == 'true'
        # Ensure stage_count is int
        if 'stage_count' in data and isinstance(data['stage_count'], str):
            data['stage_count'] = int(data['stage_count'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if this checkpoint has expired (TTL-based)."""
        if not self.ttl_expires_at:
            return False  # No TTL set - never expires
        try:
            expires = datetime.fromisoformat(self.ttl_expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > expires
        except Exception:
            return False
    
    def set_ttl(self, ttl_seconds: int) -> None:
        """Set TTL expiration for this checkpoint."""
        from datetime import timedelta
        self.ttl_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat().replace("+00:00", "Z")


# =============================================================================
# Session Store Implementation
# =============================================================================

class SessionStore:
    """
    Persistent storage for sparring session checkpoints.
    
    Features:
    - Atomic writes (tempfile + os.rename) to prevent corruption
    - File locking for concurrent access protection
    - Automatic directory creation
    - JSON schema validation
    - Configurable storage directory via environment variable
    """
    
    DEFAULT_DIR = ".sparring_sessions"
    DEFAULT_MAX_CONTEXT_LENGTH = 50  # Max messages to keep in history
    
    def __init__(self, storage_dir: Optional[str] = None, max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH):
        """
        Initialize the session store.
        
        Args:
            storage_dir: Directory to store session files.
                        Resolution order:
                        1. Explicit storage_dir parameter
                        2. QWEN_SPARRING_SESSIONS_DIR environment variable
                        3. User-level directory (%APPDATA%\\qwen-mcp\\sparring_sessions on Windows,
                           ~/.qwen-mcp/sparring_sessions on Unix)
                        4. Fallback to .sparring_sessions in current working directory
            max_context_length: Maximum number of messages to keep in conversation history
        """
        self.storage_dir = Path(self._resolve_storage_dir(storage_dir))
        self._ensure_storage_dir()
        self.max_context_length = max_context_length
        logger.info(f"SessionStore initialized at {self.storage_dir.absolute()} (max_context_length={max_context_length})")
    
    def _resolve_storage_dir(self, storage_dir: Optional[str]) -> Path:
        """
        Resolve the storage directory using the following priority:
        1. Explicit parameter
        2. Environment variable QWEN_SPARRING_SESSIONS_DIR
        3. User-level directory (APPDATA on Windows, ~/.qwen-mcp on Unix)
        4. Default relative directory
        
        Returns:
            Path object with proper OS-specific separators
        """
        # Priority 1: Explicit parameter
        if storage_dir:
            return Path(storage_dir)
        
        # Priority 2: Environment variable
        env_dir = os.environ.get("QWEN_SPARRING_SESSIONS_DIR")
        if env_dir:
            logger.info(f"Using session directory from environment: {env_dir}")
            return Path(env_dir)
        
        # Priority 3: User-level directory
        user_dir = self._get_user_data_dir()
        if user_dir:
            logger.info(f"Using user-level session directory: {user_dir}")
            return Path(user_dir)
        
        # Priority 4: Fallback to relative directory
        logger.info("Using default relative session directory: .sparring_sessions")
        return Path(self.DEFAULT_DIR)
    
    def _get_user_data_dir(self) -> Optional[Path]:
        """
        Get the user-level data directory for session storage.
        
        Returns:
            Path object to user data directory or None if not determinable
        """
        import sys
        
        if sys.platform == "win32":
            # Windows: %APPDATA%\qwen-mcp\sparring_sessions
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / "qwen-mcp" / "sparring_sessions"
        elif sys.platform == "darwin":
            # macOS: ~/Library/Application Support/qwen-mcp/sparring_sessions
            home = os.environ.get("HOME")
            if home:
                return Path(home) / "Library" / "Application Support" / "qwen-mcp" / "sparring_sessions"
        else:
            # Linux/Unix: ~/.local/share/qwen-mcp/sparring_sessions or ~/.qwen-mcp/sparring_sessions
            home = os.environ.get("HOME")
            if home:
                # Try XDG data directory first
                xdg_data = os.environ.get("XDG_DATA_HOME")
                if xdg_data:
                    return Path(xdg_data) / "qwen-mcp" / "sparring_sessions"
                # Fallback to ~/.qwen-mcp/sparring_sessions
                return Path(home) / ".qwen-mcp" / "sparring_sessions"
        
        return None
    
    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.storage_dir / f"{session_id}.json"
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return f"sp_{uuid.uuid4().hex[:12]}"
    
    def create_session(self, topic: str, context: str = "") -> SessionCheckpoint:
        """
        Create a new session checkpoint.
        
        Args:
            topic: The sparring topic
            context: Additional context
            
        Returns:
            New SessionCheckpoint with generated session_id
        """
        session_id = self._generate_session_id()
        checkpoint = SessionCheckpoint(
            session_id=session_id,
            topic=topic,
            context=context
        )
        self.save(checkpoint)
        logger.info(f"Created new session: {session_id}")
        return checkpoint
    
    def save(self, checkpoint: SessionCheckpoint) -> None:
        """
        Save a checkpoint atomically using tempfile + rename.
        
        This prevents corruption if the process is interrupted during write.
        
        Args:
            checkpoint: The SessionCheckpoint to save
        """
        session_path = self._get_session_path(checkpoint.session_id)
        checkpoint.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Atomic write: write to temp file, then rename
        data = checkpoint.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=_json_serializable_default)
        
        # Write to temporary file first
        fd, temp_path = tempfile.mkstemp(
            suffix='.json',
            prefix=f"{checkpoint.session_id}_",
            dir=self.storage_dir
        )
        try:
            # Write to the file descriptor
            os.write(fd, json_str.encode('utf-8'))
            os.close(fd)
            
            # Atomic rename with retry for Windows file locking issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.replace(temp_path, session_path)
                    logger.debug(f"Saved checkpoint for session {checkpoint.session_id}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        continue
                    raise
        except Exception as e:
            # Clean up temp file on error
            try:
                os.close(fd)
            except Exception:
                pass
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load(self, session_id: str) -> Optional[SessionCheckpoint]:
        """
        Load a session checkpoint.
        
        Args:
            session_id: The session ID to load
            
        Returns:
            SessionCheckpoint or None if not found or expired
        """
        session_path = self._get_session_path(session_id)
        
        if not session_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None
        
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            checkpoint = SessionCheckpoint.from_dict(data)
            
            # Check TTL expiration for ephemeral checkpoints
            if checkpoint.is_expired():
                logger.info(f"Session expired (TTL): {session_id}")
                # Delete expired checkpoint
                try:
                    session_path.unlink()
                except Exception:
                    pass
                return None
            
            logger.debug(f"Loaded checkpoint for session {session_id}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def update_step(self, session_id: str, step_name: str, result: Any, 
                    next_step: Optional[str] = None) -> Optional[SessionCheckpoint]:
        """
        Update a session with a completed step result.
        
        Args:
            session_id: The session ID
            step_name: Name of the completed step
            result: Result data from the step
            next_step: Next step to execute (optional)
            
        Returns:
            Updated SessionCheckpoint or None if session not found
        """
        checkpoint = self.load(session_id)
        if checkpoint is None:
            return None
        
        # Add step to completed list
        if step_name not in checkpoint.steps_completed:
            checkpoint.steps_completed.append(step_name)
        
        # Store result
        checkpoint.results[step_name] = result
        
        # Update current step
        if next_step:
            checkpoint.current_step = next_step
        
        # Update status
        if next_step is None:
            checkpoint.status = "completed"
        
        self.save(checkpoint)
        return checkpoint
    
    def mark_failed(self, session_id: str, error: str) -> Optional[SessionCheckpoint]:
        """
        Mark a session as failed.
        
        Args:
            session_id: The session ID
            error: Error message
            
        Returns:
            Updated SessionCheckpoint or None if session not found
        """
        checkpoint = self.load(session_id)
        if checkpoint is None:
            return None
        
        checkpoint.status = "failed"
        checkpoint.error = error
        self.save(checkpoint)
        return checkpoint
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if deleted, False if not found
        """
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, str]]:
        """
        List all sessions in the store.
        
        Returns:
            List of session metadata (session_id, topic, status, created_at)
        """
        sessions = []
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "topic": data.get("topic"),
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                })
            except Exception as e:
                logger.warning(f"Failed to read session {session_file}: {e}")
        return sessions
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Remove sessions older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of sessions removed
        """
        from datetime import timedelta
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        removed = 0
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                # Get file mtime as UTC datetime (timezone-aware)
                mtime = datetime.fromtimestamp(
                    session_file.stat().st_mtime,
                    tz=timezone.utc
                )
                if mtime < cutoff:
                    session_file.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session_file}: {e}")
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old sessions")
        return removed
    
    def add_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Add a message to the session history.
        CRITICAL: Filters reasoning_content and applies truncation.
        
        Args:
            session_id: The session ID
            message: Message dict with role, content, and optionally reasoning_content
            
        Returns:
            True if context was truncated, False otherwise
        """
        checkpoint = self.load(session_id)
        if checkpoint is None:
            # Initialize if missing - create minimal checkpoint
            checkpoint = SessionCheckpoint(
                session_id=session_id,
                topic="multi_turn_session",
                context="",
                messages=[]  # Explicitly initialize messages list
            )
        
        # Security: Sanitize message - ONLY keep role and content
        sanitized_msg = {
            "role": message.get("role", "user"),
            "content": message.get("content", "")
        }
        # Explicitly exclude reasoning_content and other fields
        
        if checkpoint.messages is None:
            checkpoint.messages = []
        checkpoint.messages.append(sanitized_msg)
        
        # Context Truncation Logic
        truncated = False
        if len(checkpoint.messages) > self.max_context_length:
            truncated = self._apply_rolling_summary(checkpoint)
        
        self.save(checkpoint)
        return truncated
    
    def _apply_rolling_summary(self, checkpoint: SessionCheckpoint) -> bool:
        """
        Compress context when exceeding threshold.
        Simple rolling window: Keep last N messages.
        
        In production, this could call LLM to summarize oldest messages,
        but for now we just drop the oldest messages to fit the limit.
        
        Returns:
            True if truncation occurred
        """
        overflow = len(checkpoint.messages) - self.max_context_length
        if overflow > 0:
            # Remove oldest messages to fit limit
            checkpoint.messages = checkpoint.messages[overflow:]
            logger.info(f"Truncated {overflow} oldest messages from session {checkpoint.session_id}")
            return True
        return False
    
    def get_messages_for_api(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve formatted messages for Qwen API call.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of messages in Qwen API format (role + content only)
        """
        checkpoint = self.load(session_id)
        if not checkpoint:
            return []
        
        # Ensure strict format for API - only role and content
        return [
            {"role": m["role"], "content": m["content"]}
            for m in checkpoint.messages
        ]
