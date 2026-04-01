"""
IO Utilities for Swarm Context Injection - Secure file reading with path validation.

This module provides:
- Path validation against project root (prevents traversal attacks)
- Safe file reading with size/line limits
- Context resolution for Swarm Orchestrator

Security Features:
- Blocks path traversal (../)
- Blocks absolute paths outside project root
- Blocks symlinks pointing outside root
- Enforces file size limits (100KB)
- Enforces line limits (500 lines)
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Maximum file size to read (100KB)
MAX_FILE_SIZE_BYTES = 100 * 1024

# Maximum lines per file
MAX_FILE_LINES = 500


def validate_path(path: str, project_root: Path) -> Optional[Path]:
    """
    Validate that path is within project root and safe to read.
    
    Security checks:
    - No path traversal (../)
    - No absolute paths outside project root
    - Resolved path must be under project_root
    - Symlinks must resolve within project_root
    
    Args:
        path: Relative path to validate
        project_root: Project root directory
    
    Returns:
        Validated Path object or None if invalid
    """
    try:
        # Reject obvious traversal attempts
        if ".." in path:
            logger.warning(f"Path traversal rejected: {path}")
            return None
        
        # Reject absolute paths (Windows and Unix)
        if os.path.isabs(path):
            logger.warning(f"Absolute path rejected: {path}")
            return None
        
        # Resolve the full path
        root_resolved = project_root.resolve()
        full_path = (project_root / path).resolve()
        
        # Ensure resolved path is under project root
        try:
            full_path.relative_to(root_resolved)
        except ValueError:
            logger.warning(f"Path outside project root rejected: {path}")
            return None
        
        # Check for symlinks pointing outside root
        if full_path.is_symlink():
            real_path = full_path.resolve()
            try:
                real_path.relative_to(root_resolved)
            except ValueError:
                logger.warning(f"Symlink outside root rejected: {path}")
                return None
        
        return full_path
        
    except Exception as e:
        logger.error(f"Path validation error for {path}: {e}")
        return None


async def read_file_safe(path: Path) -> Optional[str]:
    """
    Safely read file content with size and line limits.
    
    Args:
        path: Validated Path object
    
    Returns:
        File content (truncated if needed) or None if unreadable
    """
    try:
        # Check file exists
        if not path.exists():
            return None
        
        # Check it's a file (not directory)
        if not path.is_file():
            return None
        
        # Check file size
        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            logger.warning(f"File too large: {path} ({size} bytes, max {MAX_FILE_SIZE_BYTES})")
            return f"[FILE TOO LARGE: {path.name}, {size} bytes - skipped]"
        
        # Read content asynchronously
        content = await asyncio.to_thread(
            lambda: path.read_text(encoding="utf-8", errors="replace")
        )
        
        # Truncate lines if needed
        lines = content.splitlines()
        if len(lines) > MAX_FILE_LINES:
            content = "\n".join(lines[:MAX_FILE_LINES])
            content += f"\n... [truncated, {len(lines)} total lines]"
        
        return content
        
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None


async def resolve_context_keys(
    context_keys: list,
    project_root: str = "."
) -> Dict[str, str]:
    """
    Resolve context_keys (file paths) to their contents.
    
    This is the main entry point for Swarm context injection.
    
    Args:
        context_keys: List of file paths relative to project root
        project_root: Project root directory (default: current directory)
    
    Returns:
        Dict mapping file path to content (or error message)
    """
    if not context_keys:
        return {}
    
    root = Path(project_root).resolve()
    resolved = {}
    
    for key in context_keys:
        # Validate path
        validated_path = validate_path(key, root)
        
        if validated_path is None:
            resolved[key] = f"[INVALID PATH: {key}]"
            continue
        
        if not validated_path.exists():
            resolved[key] = f"[FILE NOT FOUND: {key}]"
            continue
        
        if not validated_path.is_file():
            resolved[key] = f"[NOT A FILE: {key}]"
            continue
        
        # Read file content
        content = await read_file_safe(validated_path)
        resolved[key] = content or f"[READ ERROR: {key}]"
    
    return resolved