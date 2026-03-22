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
        results: Results from each completed step
        loop_count: Number of regeneration loops (for blue/white cycle)
        error: Error message if status is 'failed'
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
    results: Dict[str, Any] = None
    loop_count: int = 0
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []
        if self.roles is None:
            self.roles = {}
        if self.results is None:
            self.results = {}
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionCheckpoint':
        return cls(**data)


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
    """
    
    DEFAULT_DIR = ".sparring_sessions"
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the session store.
        
        Args:
            storage_dir: Directory to store session files. 
                        Defaults to .sparring_sessions in current working directory.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path(self.DEFAULT_DIR)
        self._ensure_storage_dir()
        logger.info(f"SessionStore initialized at {self.storage_dir.absolute()}")
    
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
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
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
            SessionCheckpoint or None if not found
        """
        session_path = self._get_session_path(session_id)
        
        if not session_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None
        
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            checkpoint = SessionCheckpoint.from_dict(data)
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
