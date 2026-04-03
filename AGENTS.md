# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
The Lachman Protocol: Qwen Engineering Engine - MCP server that orchestrates specialized Qwen models for architectural planning, code generation, and auditing.

## Build & Test Commands
```bash
# Install dependencies (uses uv, not pip)
uv pip install -e .

# Run tests (pytest-asyncio with function loop scope)
pytest tests/ -v

# Run single test
pytest tests/test_api_v2.py::test_generate_completion_reasoning_fallback -v

# Build UI (React/Vite in qwen-hud-ui/)
cd qwen-hud-ui && npm run build

# Package VSCode extension
cd vscode-extension && npx @vscode/vsce package --allow-missing-repository
```

## Critical Architecture
- **Telemetry Server**: WebSocket on port 8878 (`src/qwen_mcp/specter/telemetry.py`)
- **Model Registry**: `src/qwen_mcp/registry.py` - dynamic model selection by billing mode
- **Engines**: `src/qwen_mcp/engines/` - CoderV2, SparringV2, SessionStore
- **Billing Modes**: `coding_plan` (strict), `payg`, `hybrid` - affects model availability

## Non-Obvious Conventions
- **TDD-First Mandatory**: RED (test) → GREEN (code) → REFACTOR (audit) - see [`docs/TDD.md`](docs/TDD.md:1)
- **Model Routing**: Auto-upgrades to `coder_pro` when prompt >15K tokens or complexity="high"
- **Session Storage**: Sparring sessions in platform-specific dirs (`%APPDATA%\qwen-mcp\sparring_sessions` on Windows)
- **UTF-8 Force**: Windows stdout/stderr reconfigured in [`server.py`](src/qwen_mcp/server.py:31)
- **Swarm Auto-Detection**: `qwen_audit` automatically uses parallel analysis for multi-file content

## Testing Specifics
- Tests use `pytest-asyncio` with `loop_scope="function"`
- Test path addition: `sys.path.append(os.path.join(os.getcwd(), "src"))`
- Mock `qwen_mcp.base.AsyncOpenAI` for API tests

## Environment Variables
- `DASHSCOPE_API_KEY`: Required for cloud models
- `BILLING_MODE`: Default `coding_plan`
- `LP_MAX_RETRIES`: Circuit breaker (default: 3)
- `QWEN_SPARRING_SESSIONS_DIR`: Optional custom session storage

## AI Assistant Rules
- Call `qwen_init_request()` as FIRST tool for every new task (resets telemetry counters)
- Use `/QW_architect`, `/QW_coder`, `/QW_audit` slash commands for specialized workflows
- See [`docs/LP_SYSTEM_PROMPT.md`](docs/LP_SYSTEM_PROMPT.md) for system instructions

## Sparring Engine Fix (2026-03-28)
- **Timeout Fix**: Reduced per-step timeouts in [`config.py`](src/qwen_mcp/engines/sparring_v2/config.py:11) to fit MCP 300s limit
- **Regeneration Disabled**: White Cell loop disabled in `full` mode to prevent timeout
- **Step Timeouts**: discovery=45s, red=60s, blue=60s, white=60s (single loop only in full mode)
