# The TDD Shackle: Quality-First Implementation

This guide defines the mandatory **Test-Driven Development (TDD)** workflow when using the Qwen Engineering Engine. Following this protocol prevents "Hallucination Cascades" and ensures that every line of code serves a verified purpose.

## The Core Protocol

Never call `qwen_coder` to implement logic without first having a failing test.

### Phase 1: RED (The Test)
1. **Objective**: Define what "success" looks like.
2. **Action**: Call `qwen_coder` to write an **asymmetrically simple test**.
  -  *Example Prompt*: "Write a pytest benchmark for a function `calculate_roi` that takes (cost, revenue) and returns percentage. It should handle zero cost by raising ValueError."
3. **Verification**: Run the test in your terminal. **It must fail.** If it passes, your test is either redundant or testing the wrong thing.

### 🟢 Phase 2: GREEN (The Code)
1. **Objective**: Pass the test as quickly as possible.
2. **Action**: Call `qwen_coder` with `mode="pro"` or `mode="expert"` for complex logic, providing the failing test as context.
  -  *Example Prompt*: "Implement the `calculate_roi` function to satisfy this test: [paste test code]."
3. **Verification**: Run the test again. It must be **GREEN**.

### Phase 3: REFACTOR (The Audit)
1. **Objective**: Clean up the "dirty" implementation without breaking the test.
2. **Action**: Call `qwen_audit` to review the implementation.
  -  *Focus*: Edge cases, performance, and "Gold Plating" (over-engineering).
3. **Action**: Apply fixes and ensure the test stays **GREEN**.

---

## 🤖 Roles in the TDD Process

| Role | Tool | Responsibility |
| :--- | :--- | :--- |
| **Architect** | `qwen_architect` | Defines the requirements that need testing. |
| **Coder** | `qwen_coder` | Writes both the **Test** (RED) and the **Implementation** (GREEN). |
| **Auditor** | `qwen_audit` | Verifies the logic and identifies hidden bugs (REFACTOR). |

## Why use the Coder for tests?
Writing tests is "cheap" in terms of tokens but "expensive" in terms of logic. By letting Qwen write the test first:
1. You verify that the model understands the requirement.
2. You establish a ground truth before any complex logic is written.
3. You prevent the model from "cheating" by writing code that only looks correct but doesn't work.

**Mantra: No RED, no GREEN. No GREEN, no commit.**

---

## 🛡️ Enforcement Layer (Layer 2)

The Qwen Engineering Engine includes a **MCP Layer 2 Enforcement** system that validates tool calls at the server level before execution. This prevents misconfigurations and enforces best practices.

### Core Principles

1. **Pre-flight Checks**: Tools validate inputs before processing
2. **TDD Enforcement**: `qwen_coder` can enforce test-first development
3. **Architectural Guardrails**: `qwen_architect` can auto-add tasks to backlog

### Tool Parameters

| Tool | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `qwen_coder` | `require_plan` | `bool` | `False` | Enforce plan pre-flight check before code generation |
| `qwen_coder` | `require_test` | `bool` | `False` | Enforce TDD test requirement (RED phase first) |
| `qwen_architect` | `auto_add_tasks` | `bool` | `False` | Auto-add generated tasks to BACKLOG.md |
| `qwen_architect` | `workspace_root` | `str \| None` | `None` | Workspace root path for task generation |

### Usage Examples

#### Enforce TDD with qwen_coder
```python
# This will enforce test-first development
await qwen_coder(
    prompt="Create a user authentication module",
    require_test=True  # Must have failing test first
)
```

#### Auto-Add Tasks with qwen_architect
```python
# This will automatically add tasks to BACKLOG.md
await qwen_architect(
    goal="Implement JWT authentication",
    auto_add_tasks=True  # Auto-add to backlog
)
```

### Enforcement Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Layer 2 Enforcement                   │
├─────────────────────────────────────────────────────────────┤
│ 1. qwen_coder / qwen_architect called with parameters       │
│ 2. Server validates parameters and pre-flight conditions    │
│ 3. If require_test=True: Check for RED phase (test first)   │
│ 4. If require_plan=True: Check for plan pre-flight          │
│ 5. If auto_add_tasks=True: Add tasks to BACKLOG.md          │
│ 6. Execute underlying engine (CoderEngineV2 / SparringV2)   │
└─────────────────────────────────────────────────────────────┘
```

### Testing Enforcement Layer

Run the enforcement test suite:
```bash
pytest tests/test_tools_enforcement.py -v
```

Expected output:
```
============================== 6 passed ==============================
```

### Implementation Details

- **Test File**: [`tests/test_tools_enforcement.py`](tests/test_tools_enforcement.py)
- **Server Implementation**: [`src/qwen_mcp/server.py`](src/qwen_mcp/server.py)
- **Engine Integration**: [`src/qwen_mcp/engines/coder_v2.py`](src/qwen_mcp/engines/coder_v2.py)
- **Tools Module**: [`src/qwen_mcp/tools.py`](src/qwen_mcp/tools.py)
