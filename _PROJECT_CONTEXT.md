# Qwen Coding Engine Context Index

**The Lachman Protocol: MCP Server orchestrating specialized Qwen models for architectural planning, code generation, and adversarial auditing.**

---

## Project Overview

### Problem Solved
Current AI assistants suffer from:
1. **Context Amnesia** - Forget requirements mid-session
2. **Placeholder Trap** - Generate `// ... implementation here` instead of complete code
3. **Hallucination Cascades** - One error triggers chain of patches breaking architecture

### Core Value
**"Stop building by trial and error. Start shipping by design."**

The Lachman Protocol (LP) orchestrates specialized Qwen models through a **Self-Healing Loop**:
- **Discovery** → Hire virtual "Expert Squad" tailored to goal
- **Architecting** → Roles debate and produce Detailed Project Blueprint
- **Self-Verification** → Separate "Verifier" audits blueprint (up to 3 retries)

Combined with **TDD-First Workflow** (RED → GREEN → REFACTOR), ensures every line of code serves a verified purpose.

---

## Tech Stack

| Category | Technology | Version | Purpose |
|:---------|:-----------|:--------|:--------|
| **Runtime** | Python | ≥3.10 | Core language |
| **MCP Framework** | `mcp` | ≥1.2.1 | Model Context Protocol server |
| **LLM Client** | `openai` | ≥1.0.0 | DashScope API compatibility |
| **Validation** | `pydantic` | ≥2.0.0 | Schema validation |
| **Retry Logic** | `tenacity` | ≥8.0.0 | Exponential backoff for API calls |
| **Storage** | `duckdb` | ≥1.4.4 | Token billing tracking |
| **Telemetry** | `fastapi` | ≥0.111.0 | WebSocket HUD server |
| **Telemetry** | `uvicorn` | ≥0.30.0 | ASGI server |
| **Telemetry** | `websockets` | ≥12.0 | Real-time broadcasting |
| **Testing** | `pytest-asyncio` | ≥0.23.0 | Async test framework |
| **Config** | `python-dotenv` | ≥1.0.0 | Environment management |
| **Paths** | `platformdirs` | ≥4.0.0 | Cross-platform session storage |

---

## Project Structure

```
qwen-coding-local/
├── src/qwen_mcp/                 # Core MCP server package
│   ├── server.py                 # FastMCP tool definitions (qwen_architect, qwen_coder, etc.)
│   ├── tools.py                  # Tool implementation logic
│   ├── api.py                    # DashScopeClient facade
│   ├── completions.py            # CompletionHandler with retry & streaming
│   ├── registry.py               # ModelEntitlementRegistry (billing mode routing)
│   ├── base.py                   # BaseDashScopeClient, billing mode getters
│   ├── billing.py                # DuckDB token tracking
│   ├── sanitizer.py              # ContentValidator (security redaction)
│   ├── orchestrator.py           # SwarmOrchestrator (parallel decomposition)
│   ├── utils.py                  # Helper utilities
│   │
│   ├── engines/                  # Specialized engines
│   │   ├── coder_v2.py           # Unified code generation (auto/standard/pro/expert)
│   │   ├── session_store.py      # Sparring session checkpointing (atomic writes)
│   │   └── sparring_v2/          # Adversarial analysis engine
│   │       ├── engine.py         # SparringEngineV2 (flash/discovery/red/blue/white/full)
│   │       ├── config.py         # TIMEOUTS, DEFAULT_MODELS
│   │       ├── models.py         # SparringResponse schema
│   │       └── helpers.py        # validate_session, get_model, get_step_result
│   │
│   ├── prompts/                  # System prompts for each role
│   │   ├── lachman.py            # LP_DISCOVERY_PROMPT, LP_ARCHITECT_PROMPT, LP_VERIFIER_PROMPT
│   │   ├── sparring.py           # RED_CELL_PROMPT, BLUE_CELL_PROMPT, WHITE_CELL_PROMPT
│   │   ├── swarm.py              # DECOMPOSE_SYSTEM_PROMPT, SYNTHESIZE_SYSTEM_PROMPT
│   │   └── system.py             # AUDIT_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT
│   │
│   └── specter/                  # Telemetry & HUD system
│       ├── telemetry.py          # TelemetryBroadcaster (WebSocket on port 8878)
│       └── identity.py           # Project ID generation
│
├── tests/                        # pytest test suite
│   ├── test_sparring_v2.py       # 22 tests for sparring engine
│   ├── test_api_v2.py            # API completion tests
│   ├── test_registry_logic.py    # Billing mode routing tests
│   ├── test_swarm_*.py           # Swarm orchestrator tests
│   └── test_telemetry.py         # WebSocket HUD tests
│
├── docs/                         # Documentation
│   ├── TDD.md                    # Test-Driven Development protocol
│   ├── SPARRING_V2.md            # Sparring engine guide
│   ├── LP_SYSTEM_PROMPT.md       # Lachman Protocol system instructions
│   └── workflows/                # Tool-specific guides
│
├── qwen-hud-ui/                  # React/Vite HUD dashboard (optional)
├── vscode-extension/             # VSCode extension for HUD
├── pyproject.toml                # Project metadata & dependencies
├── .env.example                  # Environment template
└── AGENTS.md                     # Agent rules (TDD, model routing, etc.)
```

---

## Getting Started

### Installation
```bash
# Clone and install with uv (NOT pip)
git clone <repo-url>
cd qwen-coding-local
uv pip install -e .
```

### Environment Setup
```bash
# Copy template
cp .env.example .env

# Edit required values
DASHSCOPE_API_KEY=your_dashscope_api_key_here
BILLING_MODE=coding_plan  # or 'payg' or 'hybrid'
```

### Run MCP Server
```bash
# Direct execution
uv run qwen-coding-engine

# Or via MCP config in your IDE
# See docs/INSTALL_MCP.md for Claude/VSCode integration
```

### Run Tests
```bash
# Full suite
uv run pytest tests/ -v

# Single test file
uv run pytest tests/test_sparring_v2.py -v

# With coverage
uv run pytest tests/ -v --cov=src/qwen_mcp
```

### Build UI (Optional)
```bash
cd qwen-hud-ui
npm install
npm run build
```

---

## Environment & Config

### Required Variables
| Variable | Purpose | Default |
|:---------|:--------|:--------|
| `DASHSCOPE_API_KEY` | Alibaba DashScope API key | **Required** |
| `BILLING_MODE` | Model selection policy | `coding_plan` |
| `BAILIAN_CODING_PLAN_API_KEY` | Separate coding plan key | Optional |

### Optional Variables
| Variable | Purpose | Default |
|:---------|:--------|:--------|
| `LP_MAX_RETRIES` | Architect self-healing limit | `3` |
| `DASHSCOPE_TIMEOUT` | HTTP timeout per request | `60.0` |
| `WORKSPACE_ROOT` | Path traversal protection | `.` |
| `SECURITY_REDACTION_ENABLED` | Secret redaction | `true` |
| `QWEN_SPARRING_SESSIONS_DIR` | Custom session storage | Platform-specific |

### Billing Modes
| Mode | Description | Available Models |
|:-----|:------------|:-----------------|
| `coding_plan` | Prepaid Alibaba Coding Plan | `qwen3.5-plus`, `qwen3-coder-*`, `glm-5`, `kimi-k2.5` |
| `payg` | Pay-as-you-go | `qwq-plus`, `qwen2.5-*`, all PAYG models |
| `hybrid` | Plan preferred, PAYG fallback | Both sets |

---

## System Architecture

### Data Flow
```
User Prompt (via MCP)
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastMCP Server (server.py)               │
│  Tools: qwen_architect, qwen_coder, qwen_audit, qwen_sparring │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│              ModelEntitlementRegistry (registry.py)          │
│  Routes request to model based on:                          │
│  - task_type (architect/coder/audit/scout)                  │
│  - complexity_hint (low/medium/high/critical)               │
│  - billing_mode (coding_plan/payg/hybrid)                   │
│  - estimated_tokens (auto-upgrade if >15K)                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│               DashScopeClient (api.py)                       │
│  - CompletionHandler with retry (tenacity)                  │
│  - Streaming for deep-thinking models                       │
│  - max_tokens dynamic by complexity                         │
│  - max_thinking_tokens for reasoning models                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│              Alibaba DashScope API (Cloud)                   │
│  Models: qwen3.5-plus, qwen3-coder-plus, glm-5, qwq-plus    │
└─────────────────────────────────────────────────────────────┘
    ↓
Response → TelemetryBroadcaster → WebSocket HUD (port 8878)
    ↓
DuckDB Billing Tracker (token usage per model)
```

### Specialized Engines

#### 1. CoderEngineV2 (`engines/coder_v2.py`)
Mode-based routing for code generation:
- `auto` → Registry decides based on complexity
- `standard` → `qwen3-coder-next` (fast, inline)
- `pro` → `qwen3-coder-plus` (heavy, large context)
- `expert` → `qwen2.5-coder-32b-instruct` (PAYG, architecture)

#### 2. SparringEngineV2 (`engines/sparring_v2/engine.py`)
Adversarial multi-agent analysis:
- `flash` → Quick 2-step (analyst → drafter)
- `discovery` → Define roles & profiles
- `red` → Red Cell critique (cynical auditor)
- `blue` → Blue Cell defense (strategic advocate)
- `white` → White Cell synthesis (final verdict)
- `full` → Complete session (discovery → red → blue → white)

Timeouts (after fix):
- `discovery`: 20s
- `red_cell`: 45s (deep thinking buffer)
- `blue_cell`: 45s
- `white_cell`: 45s

#### 3. SwarmOrchestrator (`orchestrator.py`)
Parallel task decomposition:
- Decompose complex prompt → atomic sub-tasks
- Execute in parallel (max 5 concurrent)
- Synthesize results into coherent response

---

## Core Business Logic / Rules

### 1. The Lachman Protocol (LP)
**Self-Healing Loop** (max 3 retries):
```
Discovery (Expert Squad Selection)
    ↓
Architecting (Blueprint Generation)
    ↓
Verification (Separate model audits)
    ↓
[If flaws found] → Retry with corrections
    ↓
Final Blueprint (JSON schema)
```

### 2. TDD-First Workflow
**Mandatory sequence** (see `docs/TDD.md`):
```
RED: Write failing test (qwen_coder)
    ↓
GREEN: Implement to pass test (qwen_coder)
    ↓
REFACTOR: Audit & clean (qwen_audit)
```

**Mantra**: "No RED, no GREEN. No GREEN, no commit."

### 3. Dynamic max_tokens
Complexity-based output length scaling:
| Complexity | max_tokens | max_thinking_tokens |
|:-----------|:-----------|:--------------------|
| `low` | 800 | 1024 |
| `medium` | 1200 | 1024 |
| `high` | 1800 | 2048 |
| `critical` | 2500 | 4096 |

### 4. Auto-Model Upgrade
If `estimated_tokens > 15K` OR `complexity="high"`:
- `qwen_coder` → `qwen_coder_pro` automatically

### 5. Deep Thinking Detection
Models with `enable_thinking=True` by default:
- `glm-5`, `glm-4.7`, `qwen3-max`, `qwen3.5-plus`, `qwq`

### 6. Session Checkpointing
Sparring sessions stored in:
- Windows: `%APPDATA%\qwen-mcp\sparring_sessions\`
- macOS: `~/Library/Application Support/qwen-mcp/sparring_sessions/`
- Linux: `~/.local/share/qwen-mcp/sparring_sessions/`

Atomic writes via tempfile + rename to prevent corruption during timeouts.

---

## Database & Data Schema

### DuckDB Billing Tables
```sql
-- Token usage tracking
CREATE TABLE token_usage (
    timestamp TIMESTAMP,
    project_id STRING,
    model STRING,
    prompt_tokens INT,
    completion_tokens INT,
    total_tokens INT,
    billing_mode STRING,
    task_type STRING
);
```

### SessionCheckpoint Schema
```json
{
  "session_id": "sp_<uuid>",
  "topic": "string",
  "context": "string",
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp",
  "status": "in_progress | completed | failed",
  "steps_completed": ["discovery", "red", "blue", "white"],
  "current_step": "string",
  "roles": {
    "red_role": "string",
    "red_profile": "string",
    "blue_role": "string",
    "blue_profile": "string",
    "white_role": "string",
    "white_profile": "string"
  },
  "models": {
    "red_model": "glm-5",
    "blue_model": "qwen3.5-plus",
    "white_model": "qwen3.5-plus"
  },
  "results": {
    "red": { "critique": "string", "raw": "string" },
    "blue": { "defense": "string", "raw": "string" },
    "white": { "consensus": "string", "raw": "string", "loops": 1 }
  },
  "loop_count": 0,
  "error": "string | null"
}
```

---

## Testing & Conventions

### Test Framework
- `pytest-asyncio` with `loop_scope="function"`
- Tests in `tests/` directory
- Mock `qwen_mcp.base.AsyncOpenAI` for API tests

### Test Commands
```bash
# Run all
uv run pytest tests/ -v

# Run single
uv run pytest tests/test_sparring_v2.py::TestFullMode::test_full_mode_executes_all_steps -v
```

### Code Style
- UTF-8 forced on Windows stdout/stderr (`server.py:31`)
- No placeholders (`// ... implementation here` forbidden)
- Complete blocks only
- Surgical edits via `apply_diff`

### Git Workflow
- TDD mandatory before commits
- `qwen_audit` for PR reviews
- No GREEN = No commit

---

## Documentation Map

| Document | Purpose |
|:---------|:--------|
| `README.md` | Project overview, tool arsenal, billing modes |
| `AGENTS.md` | Agent rules (TDD, model routing, UTF-8, Swarm) |
| `docs/TDD.md` | Test-Driven Development protocol |
| `docs/SPARRING_V2.md` | Sparring engine modes & session management |
| `docs/LP_SYSTEM_PROMPT.md` | Lachman Protocol system instructions |
| `docs/INSTALL_MCP.md` | MCP server integration guide |
| `docs/workflows/QW_*.md` | Tool-specific workflow guides |

---

## Common Workflows & Troubleshooting

### Add New Tool
1. Define tool in `server.py` with `@mcp.tool()`
2. Implement logic in `tools.py`
3. Add system prompt in `prompts/` if needed
4. Register model role in `registry.py`
5. Write tests in `tests/`
6. Document in `docs/workflows/`

### Fix Timeout Issues (Sparring)
**Symptom**: Responses truncated mid-generation
**Cause**: Timeout too short for deep-thinking models
**Fix**: Increase `TIMEOUTS` in `config.py`:
```python
TIMEOUTS = {
    "red_cell": 45.0,  # Was 15.0
    "blue_cell": 45.0,
    "white_cell": 45.0,
}
```
Also increase `complexity` to `"critical"` for 2500 max_tokens.

### Switch Billing Mode
```python
# Via tool
qwen_set_billing_mode(mode="hybrid")

# Via .env
BILLING_MODE=payg
```

### Heal Registry (Model Not Found)
```python
qwen_heal_registry()  # Auto-repair model role mappings
```

### Debug Telemetry
```bash
# Check WebSocket server
curl http://localhost:8878/health

# View logs
tail -f ~/.qwen-mcp/telemetry.log
```

### Known Issues
1. **Windows Python alias**: Use `uv run` not `pyv` or `python`
2. **MCP 300s limit**: Full sparring mode must complete in <300s
3. **Session corruption**: Atomic writes prevent this, but check `error` field
4. **Model not found**: Run `qwen_heal_registry()` or check `BILLING_MODE`

---

## Quick Reference: Tool Arsenal

| Tool | Role | Default Model (coding_plan) | Default Model (payg) |
|:-----|:-----|:----------------------------|:---------------------|
| `qwen_architect` | Strategist | `qwen3.5-plus` | `qwen3.5-plus` |
| `qwen_coder` | Coder | `qwen3-coder-next` | `qwen3-coder-plus` |
| `qwen_coder_pro` | Specialist | `qwen3-coder-plus` | `qwen2.5-72b-instruct` |
| `qwen_audit` | Analyst | `glm-5` | `qwq-plus` |
| `qwen_sparring` | Tactician | `glm-5` / `qwen3.5-plus` | `qwq-plus` / `qwen3.5-plus` |
| `qwen_swarm` | Orchestrator | Registry-routed | Registry-routed |
| `qwen_read_file` | Scout | `kimi-k2.5` | `qwen-turbo` |
| `qwen_usage_report` | Billing | N/A | N/A |
| `qwen_init_request` | Telemetry | N/A | N/A |

---

*Last Updated: 2026-03-28*
*Generated by: Code Mode (Roo) via projectcontext command*