# Architectural Plan: Qwen-Coding Enforcement

## Executive Summary

Implement server-side MCP tool enforcement to prevent agents from bypassing the development protocol. This adds a second layer of protection beyond Roo Code mode restrictions.

## Context Analysis

### Current State
- **qwen_architect** ([`generate_lp_blueprint()`](src/qwen_mcp/tools.py:65)): Creates blueprint but does NOT auto-add tasks
- **qwen_coder** ([`generate_code_unified()`](src/qwen_mcp/tools.py:370)): Generates code without checking for pending tasks or tests
- **DecisionLogSyncEngine**: Has [`add_task()`](src/qwen_mcp/engines/decision_log_sync.py:201) and [`add_tasks()`](src/qwen_mcp/engines/decision_log_sync.py:364) methods available
- **qwen_init_request**: Does NOT exist - needs to be created

### Problem
Agents can bypass protocol by:
1. Using tools directly without following mode restrictions
2. Coding without creating tasks first
3. Coding without writing tests (TDD violation)
4. Creating architectural plans without materializing tasks

---

## Proposed Architecture

### Layer 1: Roo Code Modes (Existing)
- Architect mode blocks coding tools
- Code mode blocks planning tools
- Quick-fix mode limits changes to ≤10 lines

### Layer 2: MCP Tool Validation (NEW)
```
┌─────────────────────────────────────────────────────────┐
│ qwen_architect                                        │
│ ├── auto_add_tasks=True (default)                     │
│ ├── Extract tasks from blueprint JSON                  │
│ └── Call DecisionLogSyncEngine.add_tasks()            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ qwen_coder                                            │
│ ├── require_plan=True (default)                        │
│ │   └── Check for pending tasks in decision_log.parquet│
│ ├── require_test=False (default, for backward compat)   │
│ │   └── Check for test files if True                  │
│ └── init_request() on entry                           │
└─────────────────────────────────────────────────────────┘
```

### Layer 3: Anti-Degradation (Existing)
- Post-change diff audits
- Baseline snapshots

---

## Detailed Implementation Plan

### Task 1: Create qwen_init_request Utility

**File**: `src/qwen_mcp/tools.py` (new function)

**Purpose**: Reset telemetry counters and broadcast init state for HUD visibility.

```python
async def qwen_init_request(ctx: Optional[Context] = None) -> str:
    """
    Initialize new request state for telemetry.
    
    Resets token counters and broadcasts "request_start" to HUD.
    Must be called at the start of every major tool operation.
    
    Args:
        ctx: MCP context for progress reporting
        
    Returns:
        Confirmation message
    """
    from qwen_mcp.specter.telemetry import get_broadcaster
    from qwen_mcp.billing import billing_tracker
    
    # Reset billing tracker for this request
    # (Implementation depends on billing_tracker capabilities)
    
    # Broadcast to HUD
    await get_broadcaster().broadcast_state({
        "operation": "request_start",
        "timestamp": datetime.now().isoformat(),
        "is_live": True
    }, project_id="default")
    
    return "✅ Request initialized"
```

### Task 2: Modify qwen_architect for Auto-Add Tasks

**File**: `src/qwen_mcp/tools.py` → [`generate_lp_blueprint()`](src/qwen_mcp/tools.py:65)

**Changes**:
1. Add parameter `auto_add_tasks: bool = True`
2. Add parameter `workspace_root: Optional[str] = None`
3. After blueprint generation, extract tasks from JSON
4. Call `DecisionLogSyncEngine.add_tasks()` if `auto_add_tasks=True`

**Implementation Details**:
```python
async def generate_lp_blueprint(
    goal: str,
    context: Optional[str] = None,
    auto_add_tasks: bool = True,  # NEW
    workspace_root: Optional[str] = None,  # NEW
    ctx: Optional[Context] = None
) -> str:
    # ... existing blueprint generation logic ...
    
    # NEW: Auto-add tasks if requested
    if auto_add_tasks and blueprint_data and "swarm_tasks" in blueprint_data:
        from pathlib import Path
        from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
        
        # Determine workspace root
        if workspace_root:
            workspace = Path(workspace_root)
        else:
            workspace = Path.cwd()
        
        decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace)
        backlog_path = DEFAULT_SOS_PATHS.get_backlog_path(workspace)
        
        sync_engine = DecisionLogSyncEngine(decision_log_path)
        
        # Extract tasks from blueprint
        swarm_tasks = blueprint_data.get("swarm_tasks", [])
        if isinstance(swarm_tasks, list):
            tasks_to_add = []
            for task in swarm_tasks:
                if isinstance(task, dict):
                    tasks_to_add.append({
                        "task_name": task.get("task", task.get("id", "Unknown")),
                        "advice": task.get("execution_hint", ""),
                        "complexity": "medium",  # Default
                        "tags": ["architect_generated"],
                        "risk_score": 0.0
                    })
            
            # Add in batch
            if tasks_to_add:
                decision_ids = await sync_engine.add_tasks(
                    tasks=tasks_to_add,
                    backlog_path=backlog_path,
                    workspace_root=str(workspace),
                    session_id="architect_blueprint",
                    decision_type="architect_task"
                )
                
                # Add to response
                result += f"\n\n✅ Auto-added {len(decision_ids)} tasks to BACKLOG.md\n"
    
    return result
```

### Task 3: Modify qwen_coder for Plan Requirement

**File**: `src/qwen_mcp/tools.py` → [`generate_code_unified()`](src/qwen_mcp/tools.py:370)

**Changes**:
1. Add parameter `require_plan: bool = True`
2. Add parameter `require_test: bool = False`
3. Add pre-flight check for pending tasks if `require_plan=True`
4. Add pre-flight check for test files if `require_test=True`
5. Call `qwen_init_request()` at start

**Implementation Details**:
```python
async def generate_code_unified(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    require_plan: bool = True,  # NEW
    require_test: bool = False,  # NEW
    workspace_root: Optional[str] = None,
    ctx: Optional[Context] = None,
    project_id: str = "default"
) -> str:
    from pathlib import Path
    from qwen_mcp.engines.decision_log_sync import DecisionLogSyncEngine
    
    # Determine workspace root
    if workspace_root:
        workspace = Path(workspace_root)
    else:
        workspace = Path.cwd()
    
    # NEW: Initialize request
    await qwen_init_request(ctx=ctx)
    
    # NEW: Pre-flight check - require plan
    if require_plan:
        decision_log_path = DEFAULT_SOS_PATHS.get_decision_log_path(workspace)
        if decision_log_path.exists():
            df = pd.read_parquet(decision_log_path)
            pending_tasks = df[df['task_type'] == 'pending']
            
            if pending_tasks.empty:
                error_msg = (
                    "❌ No pending tasks found in decision_log.parquet\n\n"
                    "To proceed with coding:\n"
                    "1. Run qwen_architect first to create tasks\n"
                    "2. Or set require_plan=False for ad-hoc coding"
                )
                return error_msg
        else:
            error_msg = (
                "❌ decision_log.parquet not found\n\n"
                "To proceed with coding:\n"
                "1. Run qwen_architect first to create tasks\n"
                "2. Or set require_plan=False for ad-hoc coding"
            )
            return error_msg
    
    # NEW: Pre-flight check - require test
    if require_test:
        test_patterns = [
            "test_*.py",
            "*_test.py",
            "tests/**/*.py",
            "spec/**/*.py"
        ]
        
        test_files = []
        for pattern in test_patterns:
            test_files.extend(workspace.glob(pattern))
        
        if not test_files:
            error_msg = (
                "❌ No test files found\n\n"
                "TDD Protocol: Write tests FIRST, then implementation\n\n"
                "To proceed:\n"
                "1. Write a test file (test_*.py or *_test.py)\n"
                "2. Or set require_test=False to skip TDD check"
            )
            return error_msg
    
    # ... existing code generation logic ...
```

### Task 4: Update MCP Tool Signatures in server.py

**File**: `src/qwen_mcp/server.py`

**Changes**: Update tool registration to include new parameters.

```python
# In the FastMCP server setup:
@mcp.tool()
async def qwen_architect(
    goal: str,
    context: Optional[str] = None,
    auto_add_tasks: bool = True,
    workspace_root: Optional[str] = None
) -> str:
    """Generate architecture blueprint with optional auto-task addition."""
    return await generate_lp_blueprint(
        goal=goal,
        context=context,
        auto_add_tasks=auto_add_tasks,
        workspace_root=workspace_root
    )

@mcp.tool()
async def qwen_coder(
    prompt: str,
    mode: str = "auto",
    context: Optional[str] = None,
    require_plan: bool = True,
    require_test: bool = False,
    workspace_root: Optional[str] = None
) -> str:
    """Generate code with optional TDD and plan enforcement."""
    return await generate_code_unified(
        prompt=prompt,
        mode=mode,
        context=context,
        require_plan=require_plan,
        require_test=require_test,
        workspace_root=workspace_root
    )
```

### Task 5: Create Tests

**File**: `tests/test_tools_enforcement.py`

**Test Cases**:
1. Test `qwen_init_request()` broadcasts correctly
2. Test `qwen_architect` with `auto_add_tasks=True` adds tasks to backlog
3. Test `qwen_architect` with `auto_add_tasks=False` does NOT add tasks
4. Test `qwen_coder` with `require_plan=True` blocks without pending tasks
5. Test `qwen_coder` with `require_plan=False` allows coding without tasks
6. Test `qwen_coder` with `require_test=True` blocks without test files
7. Test `qwen_coder` with `require_test=False` allows coding without tests

### Task 6: Update Documentation

**Files to Update**:
1. `docs/TDD.md` - Add section about enforcement layers
2. `docs/ARCHITECTURE.md` - Update with enforcement architecture diagram
3. `README.md` - Document new parameters

---

## Risk Mitigation

### Breaking Change Risk
- **Mitigation**: Default parameters maintain backward compatibility
- `auto_add_tasks=True` (enables enforcement)
- `require_plan=True` (enables enforcement)
- `require_test=False` (disabled for backward compatibility)

### Performance Risk
- **Mitigation**: Pre-flight checks are lightweight:
  - Parquet read: <10ms for typical backlogs
  - File glob: <5ms for typical workspaces

### False Positive Risk
- **Mitigation**: Error messages are actionable and suggest solutions

---

## Success Criteria

1. ✅ `qwen_init_request()` exists and broadcasts correctly
2. ✅ `qwen_architect` auto-adds tasks when `auto_add_tasks=True`
3. ✅ `qwen_coder` blocks without pending tasks when `require_plan=True`
4. ✅ `qwen_coder` blocks without tests when `require_test=True`
5. ✅ All enforcement can be disabled via parameters
6. ✅ Tests pass for all enforcement scenarios
7. ✅ Documentation updated

---

## Next Steps

1. Switch to **Code Mode** to implement
2. Create baseline snapshot before changes
3. Implement in order:
   - Task 1: Create `qwen_init_request()`
   - Task 2: Modify `qwen_architect`
   - Task 3: Modify `qwen_coder`
   - Task 4: Update server.py
   - Task 5: Create tests
   - Task 6: Update documentation
