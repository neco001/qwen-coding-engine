import os
import hashlib
import uuid
import re


def generate_instance_id() -> str:
    """
    Generate a unique instance ID for this VSCode window/session.
    Returns an 8-character hex string derived from UUID4.
    """
    return uuid.uuid4().hex[:8]


def get_session_id(instance_id: str, client_source: str, cwd: str) -> str:
    """
    Generate unique session ID combining instance, client source, and workspace.
    
    Format: {instanceId}_{clientSource}_{workspaceHash}
    
    Args:
        instance_id: Unique per VSCode window (from generate_instance_id())
        client_source: 'gemini' or 'roocode'
        cwd: Current working directory (workspace folder)
    
    Returns:
        Session ID string like "a1b2c3d4_gemini_7e8f9a0b"
    """
    workspace_hash = _compute_workspace_hash(cwd)
    return f"{instance_id}_{client_source}_{workspace_hash}"


def _compute_workspace_hash(cwd: str) -> str:
    """Compute 8-character hash of normalized workspace path."""
    try:
        normalized_cwd = os.path.realpath(cwd).lower()
    except Exception:
        normalized_cwd = os.path.normpath(cwd).lower()
    return hashlib.sha256(normalized_cwd.encode()).hexdigest()[:8]


# Global instance ID - generated once per process
_cached_instance_id: str = None


def get_or_create_instance_id() -> str:
    """
    Get or create a unique instance ID for this MCP server process.
    Uses QWEN_INSTANCE_ID env var if set, otherwise generates from PID.
    """
    global _cached_instance_id
    if _cached_instance_id:
        return _cached_instance_id
    
    # Check environment first
    instance_id = os.getenv("QWEN_INSTANCE_ID")
    if instance_id:
        _cached_instance_id = instance_id
        return instance_id
    
    # Generate from PID (unique per process)
    pid = os.getpid()
    _cached_instance_id = f"{pid:08x}"[-8:]  # 8-char hex from PID
    return _cached_instance_id


def _sanitize_client_source(client_source: str) -> str:
    """
    Sanitize client_source to prevent injection and ensure valid format.
    
    Only allows alphanumeric, underscore, and hyphen characters.
    Truncates to 32 characters max.
    
    Args:
        client_source: Raw client source string
    
    Returns:
        Sanitized client source safe for use in project_id
    """
    # Remove dangerous characters, keep only safe ones
    safe_source = re.sub(r'[^a-zA-Z0-9_-]', '', client_source)
    # Truncate to reasonable length
    return safe_source[:32] if safe_source else "default"


def get_current_project_id(client_source: str = "default") -> str:
    """
    Identifies the unique project ID based on the environment or CWD.
    
    Priority:
    1. QWEN_PROJECT_NAME environment variable (full override)
    2. Instance ID + client_source + workspace hash (auto-generated)
    
    Args:
        client_source: The client source identifier (e.g., 'gemini', 'roocode', 'default')
                     Will be sanitized to prevent injection attacks.
    
    The display name ("Sesja 1", "Sesja 2") is handled by SessionMapper in telemetry.py.
    """
    # Full override
    project_name = os.getenv("QWEN_PROJECT_NAME")
    if project_name:
        return project_name
    
    # Sanitize client_source for security
    safe_client_source = _sanitize_client_source(client_source)
    
    # Auto-generated: {instanceId}_{clientSource}_{workspaceHash}
    instance_id = get_or_create_instance_id()
    cwd = os.getcwd()
    workspace_hash = _compute_workspace_hash(cwd)
    return f"{instance_id}_{safe_client_source}_{workspace_hash}"
