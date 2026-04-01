# _DATA_CONTEXT.md

## Data Sources
- **DuckDB**: `~/.qwen-mcp/token_usage.duckdb` - Token billing tracking
- **File-based**: `sparring_sessions/sp_*.json` - Session persistence
- **External APIs**: DashScope (Qwen models), HuggingFace (metadata)

## Schema Overview
- `token_usage` table: session_id, model_id, prompt_tokens, completion_tokens, billing_mode
- Sparring session JSON: session_id, topic, red_cell, blue_cell, white_cell, history

## Data Pipelines
- Token tracking: Request → TelemetryBroadcaster → DuckDB
- Sparring: discovery → red/blue/white → SessionStore

## Session Management
- Session ID: `{instanceId}_{clientSource}_{workspaceHash}`
- Storage: `%APPDATA%\qwen-mcp\sparring_sessions\` (Windows)
