# Sparring Session Storage Fix

## Problem

The `SessionStore` class in `src/qwen_mcp/engines/session_store.py` uses a relative path `.sparring_sessions` which resolves based on the **current working directory** when the MCP server process starts.

### Issue Scenario

1. MCP server starts from `<repo-root>`
2. User opens different workspace in VS Code: `<user-workspace>`
3. User runs `qwen_sparring` from that workspace
4. Sessions are saved to `<user-workspace>/.sparring_sessions` (workspace directory)
5. But MCP server looks for sessions in its own directory

This causes "Session not found" errors when trying to continue sessions across different working directories.

## Solution

Modify `SessionStore` to use a consistent, configurable directory resolution:

### Resolution Priority

1. **Explicit parameter** - `storage_dir` passed to constructor
2. **Environment variable** - `QWEN_SPARRING_SESSIONS_DIR`
3. **User-level directory** - Platform-specific user data directory:
   - Windows: `%APPDATA%\qwen-mcp\sparring_sessions`
   - macOS: `~/Library/Application Support/qwen-mcp/sparring_sessions`
   - Linux: `~/.local/share/qwen-mcp/sparring_sessions` or `~/.qwen-mcp/sparring_sessions`
4. **Fallback** - `.sparring_sessions` in current working directory

## Implementation Changes

### File: `src/qwen_mcp/engines/session_store.py`

1. Add `_resolve_storage_dir()` method to handle directory resolution logic
2. Add `_get_user_data_dir()` method for platform-specific user data directories
3. Update `__init__` to use the new resolution logic
4. Add logging for which directory is being used

### File: `.env.example`

Add new configuration option:

```bash
# 8. Sparring Session Storage
# Directory to store sparring session checkpoints.
# If not set, uses user-level directory (APPDATA on Windows, ~/.qwen-mcp on Unix)
# QWEN_SPARRING_SESSIONS_DIR=%APPDATA%\qwen-mcp\sparring_sessions
```

## Benefits

1. **Consistent location** - Sessions stored in predictable location regardless of workspace
2. **Environment override** - Can be configured via environment variable
3. **Cross-workspace access** - Sessions accessible from any workspace
4. **User-level isolation** - Sessions tied to user, not project

## Migration

Existing sessions in `.sparring_sessions` folders can be manually moved to the new location if needed.
