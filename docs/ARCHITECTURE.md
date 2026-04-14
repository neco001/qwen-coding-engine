# Qwen Engineering Engine Architecture

This document describes the system architecture of the Qwen Engineering Engine (Lachman Protocol).

## Overview

The Qwen Engineering Engine is an MCP (Model Context Protocol) server that orchestrates specialized Qwen models for architectural planning, code generation, and auditing.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Qwen Engineering Engine                             │
│                              (MCP Server)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        MCP Tool Layer                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │  │
│  │  │ qwen_architect│  │  qwen_coder  │  │  qwen_audit  │  │ qwen_... │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │        │        │                             │
│                              ▼        ▼        ▼                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    MCP Layer 2: Enforcement                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Pre-flight Checks  │  TDD Enforcement  │  Auto-Add Tasks      │  │  │
│  │  └────────────────────┴────────────────────┴───────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │        │        │                             │
│                              ▼        ▼        ▼                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Engine Layer                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │  │
│  │  │ CoderEngineV2│  │SparringEngine│  │  SessionStore│  │  ...      │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │        │        │                             │
│                              ▼        ▼        ▼                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Model Registry                                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │  │
│  │  │ qwen3-coder  │  │ qwen3.5-plus │  │ qwen3-turbo  │  │  ...      │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## MCP Layer 2: Enforcement

The **Enforcement Layer** validates tool calls at the server level before execution. This prevents misconfigurations and enforces best practices.

### Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MCP Layer 2: Enforcement                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────────────────┐   │
│  │  Pre-flight     │    │   TDD           │    │  Auto-Add Tasks       │   │
│  │  Checks         │    │   Enforcement   │    │  (Backlog Sync)       │   │
│  │                 │    │                 │    │                       │   │
│  │ • require_plan  │    │ • require_test  │    │ • auto_add_tasks      │   │
│  │ • workspace_root│    │ • RED phase     │    │ • Decision Log Sync   │   │
│  │ • project_id    │    │   validation    │    │                       │   │
│  └─────────────────┘    └─────────────────┘    └───────────────────────┘   │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Enforcement Flow                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  1. Tool called with enforcement parameters                                │
│  2. Server validates parameters and pre-flight conditions                  │
│  3. If require_test=True: Check for RED phase (test first)                 │
│  4. If require_plan=True: Check for plan pre-flight                        │
│  5. If auto_add_tasks=True: Add tasks to BACKLOG.md                        │
│  6. Execute underlying engine (CoderEngineV2 / SparringV2)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Signatures

#### qwen_coder
```python
async def qwen_coder(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    ctx: Context = None,
    require_plan: bool = False,  # NEW: Enforce plan pre-flight
    require_test: bool = False   # NEW: Enforce TDD test requirement
) -> str:
```

#### qwen_architect
```python
async def qwen_architect(
    goal: str,
    context: Optional[str] = None,
    ctx: Context = None,
    auto_add_tasks: bool = False,  # NEW: Auto-add tasks to backlog
    workspace_root: Optional[str] = None  # NEW: Workspace root path
) -> str:
```

## Engine Layer

### CoderEngineV2
- **Purpose**: Unified code generation with mode routing
- **Modes**: `auto`, `standard`, `pro`, `expert`
- **Features**:
  - Complexity estimation
  - Mode auto-upgrade for complex prompts
  - Progress reporting via MCP Context

### SparringEngineV2
- **Purpose**: Intellectual debate and strategic analysis
- **Modes**: `sparring1`, `sparring2`, `sparring3`, `discovery`, `red`, `blue`, `white`
- **Features**:
  - Multi-cell analysis (Red/Blue/White)
  - Step-by-step checkpointing
  - Timeout protection

### SessionStore
- **Purpose**: Persistent session storage
- **Features**:
  - Platform-specific storage paths
  - Session isolation
  - State persistence

## Model Registry

The Model Registry dynamically selects models based on billing mode and task requirements.

### Billing Modes
- **coding_plan**: Use Coding Plan API (flat monthly fee)
- **payg**: Use PAYG API (pay-per-token)
- **hybrid**: Intelligent routing between both

### Model Roles
- **strategist**: Architectural planning and audit
- **coder**: Code generation
- **scout**: File tree mapping and context discovery

## Telemetry Server

WebSocket server on port 8878 for real-time telemetry updates.

### Features
- Token usage tracking
- Session state broadcasting
- Active model monitoring

## File Structure

```
qwen-coding-local/
├── src/
│   └── qwen_mcp/
│       ├── server.py          # MCP tool wrappers
│       ├── tools.py           # Core generation functions
│       ├── registry.py        # Model registry
│       ├── engines/           # Engine implementations
│       │   ├── coder_v2.py
│       │   ├── sparring_v2/
│       │   └── session_store.py
│       └── specter/           # Telemetry and enforcement
│           ├── enforcement.py
│           └── telemetry.py
├── tests/
│   ├── test_tools_enforcement.py  # Enforcement layer tests
│   └── test_qwen_init_request.py  # Telemetry tests
├── docs/
│   ├── TDD.md                 # TDD protocol documentation
│   ├── ARCHITECTURE.md        # This file
│   └── LP_SYSTEM_PROMPT.md    # System instructions
└── qwen-hud-ui/               # Web UI for HUD
```

## Testing

Run the enforcement test suite:
```bash
pytest tests/test_tools_enforcement.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

## Development

### Install Dependencies
```bash
uv pip install -e .
```

### Run Server
```bash
python -m qwen_mcp.server
```

### Build UI
```bash
cd qwen-hud-ui && npm run build
```

## License

MIT License - See LICENSE file for details.
