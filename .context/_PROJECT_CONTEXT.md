# _PROJECT_CONTEXT.md

## Project Overview
- **Name**: qwen-coding-engine (The Lachman Protocol)
- **Purpose**: MCP Server orchestrating specialized Qwen models for architectural planning, code generation, and auditing
- **Primary Language**: Python 3.10+
- **Framework**: FastMCP (MCP Protocol) + FastAPI (Telemetry)
- **Architectural Patterns**: Engine-based orchestration (CoderV2, SparringV2, Scout, Swarm)

## Tech Stack Summary
- **Runtime**: Python 3.10+, Uvicorn (ASGI server)
- **Core Frameworks**: FastMCP, FastAPI, OpenAI SDK (DashScope), Pydantic v2, DuckDB, WebSockets
- **External Services**: Alibaba DashScope (Qwen models), HuggingFace API

## Directory Structure
- `src/qwen_mcp/` - Main source (server.py, tools.py, api.py, registry.py)
- `src/qwen_mcp/engines/` - Engine modules (coder_v2, scout, sparring_v2)
- `tests/` - pytest-asyncio test suites
- `specter-lens-ui/` - React/Vite HUD frontend

## Development Workflow
- Package Manager: uv (preferred)
- Commands: `pytest tests/ -v`, `uv pip install -e .`

## Key Conventions
- TDD-First Mandatory (RED → GREEN → REFACTOR)
- Model Registry with billing mode routing
- Session-based telemetry isolation
