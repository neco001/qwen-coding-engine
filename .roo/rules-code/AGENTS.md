# AGENTS.md - Code Mode Rules

## Non-Obvious Coding Rules

- **TDD-First Mandatory**: Never call `qwen_coder` without a failing test first - see [`docs/TDD.md`](../../docs/TDD.md:1)
- **Model Auto-Upgrade**: Coding tasks with >15K tokens or complexity="high" automatically route to `coder_pro` ([`registry.py`](../../src/qwen_mcp/registry.py:507))
- **No Placeholders**: All code generation must be complete - `// ... rest of code` is forbidden by system prompt
- **UTF-8 on Windows**: stdout/stderr forced to UTF-8 in [`server.py`](../../src/qwen_mcp/server.py:31) - don't add redundant encoding fixes
- **Swarm for Multi-File**: `qwen_audit` auto-detects multi-file content and uses parallel Swarm analysis ([`tools.py`](../../src/qwen_mcp/tools.py:56))
- **Billing Mode Matters**: In `coding_plan` mode, only these models work: `qwen3.5-plus`, `qwen3-coder-next/plus`, `glm-5`, `kimi-k2.5` ([`registry.py`](../../src/qwen_mcp/registry.py:30))
- **Dynamic max_tokens**: Output length auto-scales by complexity: low=800, medium=1200, high=1800, critical=2500 ([`completions.py`](../../src/qwen_mcp/completions.py:61))
