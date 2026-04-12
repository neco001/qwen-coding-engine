# Release Notes - v1.2.0

**Date:** 2026-04-12

## New Features

### 1. Batch Task Creation (`qwen_add_tasks`)

**Problem:** The `qwen_add_task` tool could only add one task at a time. When architect generated 150+ tasks, the agent had to call the tool 150 times, causing MCP timeout and workflow inefficiency.

**Solution:** New `qwen_add_tasks` MCP tool for batch task creation with chunk-based processing.

**Implementation:**
- New MCP tool `qwen_add_tasks` in [`server.py`](src/qwen_mcp/server.py:431-504)
- New function `add_tasks_to_backlog_batch()` in [`tools.py`](src/qwen_mcp/tools.py:785-844)
- New method `add_tasks()` in [`decision_log_sync.py`](src/qwen_mcp/engines/decision_log_sync.py:364-504)

**Key Features:**
- **Chunk-based processing**: Default 20 tasks per chunk to avoid MCP timeout
- **New project support**: Creates BACKLOG.md and decision_log.parquet if missing
- **Preserves existing records**: Safely appends to existing data
- **Full parameter support**: All optional fields (complexity, tags, risk_score, session_id)

**API Signature:**
```python
@mcp.tool()
async def qwen_add_tasks(
    tasks: List[Dict],           # Required: List of task dictionaries
    workspace_root: str = ".",   # Optional: Project root path
    session_id: str = "sos_manual",
    decision_type: str = "manual_task",
    chunk_size: int = 20         # Optional: Tasks per batch
) -> str:
```

**Example Usage:**
```python
tasks = [
    {"task_name": "Fix login bug", "advice": "Add null check"},
    {"task_name": "Update docs", "advice": "Add API examples"},
    {"task_name": "Refactor auth", "advice": "Use JWT", "complexity": "high"}
]
result = await qwen_add_tasks(tasks=tasks, workspace_root=".")
```

**Files Changed:**
- `src/qwen_mcp/server.py`: Added `qwen_add_tasks` MCP tool (lines 431-504)
- `src/qwen_mcp/tools.py`: Added `add_tasks_to_backlog_batch()` (lines 785-844)
- `src/qwen_mcp/engines/decision_log_sync.py`: Added `add_tasks()` method (lines 364-504)

**Tests:**
- `tests/test_batch_tasks.py`: 8 comprehensive tests covering:
  - Empty list handling
  - Single task (backward compatibility)
  - Multiple tasks under chunk limit
  - Multiple chunks (large batches)
  - Optional fields support
  - Missing required fields validation
  - New project (no existing files)
  - Preserves existing records

### 2. Snapshot Naming Convention & Auto-Selection

**Problem:** Snapshot names were random/undefined, making it impossible to know which snapshots to compare. User had to manually specify snapshot names every time.

**Solution:** Standardized naming convention `baseline-YYYYMMDD_HHMMSS.json` (UTC timestamp) and auto-selection of two newest snapshots.

**Implementation:**
- Added `_generate_timestamped_name()` in [`snapshot.py`](src/graph/snapshot.py:724-728)
- Added `list_snapshots()` to enumerate and sort by timestamp [`snapshot.py`](src/graph/snapshot.py:730-781)
- Added `get_two_newest_snapshots()` for auto-selection [`snapshot.py`](src/graph/snapshot.py:783-793)
- Modified `save_snapshot()` to auto-generate timestamped name when `name="auto"` [`snapshot.py`](src/graph/snapshot.py:795-820)
- Modified `load_snapshot()` with backward compatible prefix auto-add [`snapshot.py`](src/graph/snapshot.py:822-851)
- Updated MCP tools with `Optional[str] = "auto"` defaults [`server.py`](src/qwen_mcp/server.py:866-906)

**Key Features:**
- **Standardized naming**: `baseline-YYYYMMDD_HHMMSS.json` using UTC timestamp
- **Auto-selection**: `qwen_compare_snapshots_tool()` selects two newest by default
- **Backward compatible**: Explicit names still work, legacy snapshots load correctly
- **Prefix auto-add**: Can load by short timestamp (e.g., "20260412_153719" → "baseline-20260412_153719")

**API Changes:**
```python
# Before (required explicit names)
qwen_create_baseline_tool(name="baseline")  # Random name
qwen_compare_snapshots_tool(snapshot1_name="baseline", snapshot2_name="latest")  # Manual selection

# After (auto defaults)
qwen_create_baseline_tool()  # Creates baseline-20260412_192730.json
qwen_compare_snapshots_tool()  # Auto-selects two newest snapshots
```

**Files Changed:**
- `src/graph/snapshot.py`: Added timestamped naming and auto-selection methods
- `src/qwen_mcp/diff_audit.py`: Updated comparison logic with auto-selection
- `src/qwen_mcp/server.py`: Updated MCP tool defaults

**Tests:**
- `tests/test_snapshot_naming.py`: 15 comprehensive tests covering:
  - Timestamped name format validation
  - UTC timezone usage
  - Auto-generation on save
  - Explicit name preservation (backward compatibility)
  - Snapshot listing and sorting
  - Two newest selection
  - Backward compatible loading

---

## Changes

### Test File Organization

**Change:** Moved all test files from root directory to `tests/` for better project organization.

**Files Moved:**
- `test_broadcast.py` → `tests/test_broadcast.py`
- `test_coder.py` → `tests/test_coder.py`
- `test_engine.py` → `tests/test_engine.py`
- `test_full_flow.py` → `tests/test_full_flow.py`
- `test_max_tokens_zero.py` → `tests/test_max_tokens_zero.py`
- `test_mcp_tool.py` → `tests/test_mcp_tool.py`
- `test_optimization.py` → `tests/test_optimization.py`
- `test_snapshot_debug.py` → `tests/test_snapshot_debug.py`
- `test_sparring_truncation.py` → `tests/test_sparring_truncation.py`
- `test_white_cell.py` → `tests/test_white_cell.py`
- `test_api_response.txt` → `tests/test_api_response.txt`

---

## Documentation Updates

- **README.md**: Added `qwen_add_tasks` to Arsenal tables (PAYG and Coding Plan modes)
- **CHANGELOG.md**: Added v1.2.0 release entry
- **BACKLOG.md**: Marked batch task creation task as completed

---

## Upgrade Notes

No breaking changes. The new `qwen_add_tasks` tool is additive - existing `qwen_add_task` continues to work for single task scenarios.

**Recommended Migration:**
- Use `qwen_add_task` for single tasks (simpler API)
- Use `qwen_add_tasks` for batch operations (architect output, bulk imports)

---

## Commit Reference

Commit: `e71a6cd` (batch task creation feature)