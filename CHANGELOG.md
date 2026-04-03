# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-03

### 🎉 Initial Public Release

**The Lachman Protocol: Qwen Engineering Engine** - MCP server for architectural planning and code generation using specialized Qwen models.

### ✨ Core Features

- **The Lachman Protocol (LP)**: Self-healing architectural planning loop (Discovery → Architecting → Self-Verification)
- **TDD-First Workflow**: RED (test) → GREEN (code) → REFACTOR (audit)
- **Dynamic Model Registry**: Automatic model selection based on billing mode (`coding_plan`, `payg`, `hybrid`)
- **Scout-Powered Context Discovery**: File/project analysis using kimi-k2.5 before planning/coding/auditing
- **Swarm Orchestrator**: Parallel task decomposition and execution (max 5 concurrent tasks)
- **Sparring Engine**: Adversarial multi-agent debate (sparring1/2/3 modes)
- **DuckDB Billing Tracking**: Local token/cost reports via `qwen_usage_report`
- **SPECTER Telemetry**: WebSocket server (port 8878) for real-time HUD streaming

### 🛠️ Available Tools

| Tool                | Role          | Default Model        |
| ------------------- | ------------- | -------------------- |
| `qwen_architect`    | Strategist    | qwen3.5-plus         |
| `qwen_coder`        | Coder         | qwen3-coder-next     |
| `qwen_coder_pro`    | Specialist    | qwen3-coder-plus     |
| `qwen_audit`        | Analyst       | glm-5                |
| `qwen_sparring`     | Debate Master | qwen3.5-plus / glm-5 |
| `qwen_read_file`    | Scout         | kimi-k2.5            |
| `qwen_list_files`   | Explorer      | kimi-k2.5            |
| `qwen_usage_report` | Billing       | N/A (DuckDB)         |

### 📦 Tech Stack

- **Runtime**: Python 3.10+
- **Protocol**: MCP (Model Context Protocol)
- **Models**: Qwen via Alibaba DashScope
- **Analytics**: DuckDB for billing/token tracking
- **HUD**: React/Vite + VSCode Extension (qwen-hud-ui)
- **Telemetry**: WebSocket server on port 8878

### 📚 Documentation

- [`README.md`](README.md) - Project overview and installation
- [`docs/LP_SYSTEM_PROMPT.md`](docs/LP_SYSTEM_PROMPT.md) - System instructions for AI assistants
- [`docs/TDD.md`](docs/TDD.md) - TDD-First workflow guide
- [`docs/REPAIR_PROTOCOL.md`](docs/REPAIR_PROTOCOL.md) - Debugging and fixing regressions
- [`docs/workflows/`](docs/workflows/) - Slash command workflows
- [`AGENTS.md`](AGENTS.md) - Agent-specific guidance
- [`BUILD_GUIDE.md`](BUILD_GUIDE.md) - Compilation and packaging

### ⚠️ Known Issues

- **HUD Streaming** (`qwen-hud-ui`): WebSocket connection in VSCode extension is currently broken. The MCP server works fully without the UI component. Use `qwen_usage_report()` for billing data.
- **Looking for contributors**: If you can fix WebSocket streaming in VSCode extensions, please open a PR!

### 🔧 Configuration

**Environment Variables:**

- `DASHSCOPE_API_KEY` - Required (Alibaba DashScope API key)
- `BILLING_MODE` - Optional, default: `coding_plan` (`coding_plan`, `payg`, `hybrid`)
- `LP_MAX_RETRIES` - Optional, default: `3` (circuit breaker for self-healing loop)

**Billing Modes:**

- `coding_plan` - Strict mode using only prepaid Alibaba Coding Plan models
- `payg` - Pay-as-you-go via DashScope API
- `hybrid` - Coding plan preferred, PAYG fallback for complex tasks

---

## [0.1.0] - 2026-03-28

### 🐛 Development Version (Pre-Release)

- Initial development version
- Internal testing and iteration
- **Not suitable for production use**
