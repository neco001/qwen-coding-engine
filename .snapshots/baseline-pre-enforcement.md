# Baseline Snapshot: pre-enforcement

**Created**: 2026-04-13T21:31:00Z  
**Purpose**: Baseline snapshot before implementing Qwen-Coding enforcement (Layer 2: MCP Tool Validation)

## Description

This snapshot captures the state of the project before adding server-side enforcement to `qwen_architect` and `qwen_coder` MCP tools.

## Changes Intended

1. Create `qwen_init_request()` utility function for telemetry reset
2. Modify `qwen_architect` to add `auto_add_tasks` parameter and auto-add tasks to backlog
3. Modify `qwen_architect` to add `workspace_root` parameter
4. Modify `qwen_coder` to add `require_plan` parameter for pre-flight check
5. Modify `qwen_coder` to add `require_test` parameter for TDD enforcement
6. Update MCP tool signatures in `server.py` for new parameters
7. Create `tests/test_tools_enforcement.py` with enforcement test cases
8. Update `docs/TDD.md` with enforcement layer documentation
9. Update `docs/ARCHITECTURE.md` with enforcement diagram

## Files to Modify

- `src/qwen_mcp/tools.py`
- `src/qwen_mcp/server.py`
- `tests/test_tools_enforcement.py` (new)
- `docs/TDD.md`
- `docs/ARCHITECTURE.md`

## Risk Level

Medium - these are backward-compatible changes with default parameters maintaining existing behavior.

## Rollback Plan

If tests fail or regressions are detected:
1. Revert `src/qwen_mcp/tools.py` to this baseline state
2. Revert `src/qwen_mcp/server.py` to this baseline state
3. Delete `tests/test_tools_enforcement.py` (new file)
4. Revert documentation changes

## Enforcement Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Roo Code Modes (Existing)                  │
│ - architect.md: blokuje narzędzia kodujące          │
│ - code.md: blokuje narzędzia planujące               │
│ - quick-fix.md: ogranicza zakres zmian               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: MCP Tool Validation (NEW - This Change)     │
│ - qwen_architect: auto_add_tasks=True               │
│ - qwen_coder: require_plan=True, require_test=True    │
│ - All tools: init_request na początku                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Anti-Degradation (Existing)                │
│ - qwen_diff_audit_tool: wykrywa regresje PO fakcie   │
│ - qwen_create_baseline_tool: snapshot przed zmianami    │
└─────────────────────────────────────────────────────────┘
```
