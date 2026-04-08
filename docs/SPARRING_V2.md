# Sparring Engine v2 - Step-by-Step Adversarial Analysis

## Overview

Sparring Engine v2 is a refactored version of the original `qwen_sparring_pro` tool, designed to solve MCP client timeout issues (300s hard limit) by breaking monolithic multi-agent sessions into discrete, checkpointed steps.

### 2026-04-08 Stage-Based Refactoring

**New Architecture:** Stage-based execution with unified `BaseStageExecutor` providing:
- **BudgetManager**: Dynamic timeout allocation with remaining budget tracking
- **CircuitBreaker**: Failure recovery (3 failures threshold, 60s recovery timeout)
- **Stage Checkpointing**: Automatic checkpointing after each stage
- **Recovery Support**: Resume from failed stage without losing progress

**Files Modified:**
- [`src/qwen_mcp/engines/sparring_v2/base_stage_executor.py`](src/qwen_mcp/engines/sparring_v2/base_stage_executor.py) - Core architecture
- [`src/qwen_mcp/engines/sparring_v2/modes/pro.py`](src/qwen_mcp/engines/sparring_v2/modes/pro.py) - Pro mode (sparring3)
- [`src/qwen_mcp/engines/sparring_v2/modes/full.py`](src/qwen_mcp/engines/sparring_v2/modes/full.py) - Full mode (sparring2)
- [`src/qwen_mcp/engines/sparring_v2/modes/flash.py`](src/qwen_mcp/engines/sparring_v2/modes/flash.py) - Flash mode (sparring1)
- [`src/qwen_mcp/engines/session_store.py`](src/qwen_mcp/engines/session_store.py) - TTL expiration support
- [`src/qwen_mcp/engines/sparring_v2/config.py`](src/qwen_mcp/engines/sparring_v2/config.py) - Budget and stage weights config
- [`tests/test_sparring_stage_recovery.py`](tests/test_sparring_stage_recovery.py) - 23 integration tests

## Problem Solved

**Original Issue:** The monolithic `run_pro()` method executed 4-6 sequential API calls with 300s timeouts each, potentially taking 26+ minutes total - far exceeding the MCP client's 300s timeout limit.

**Solution:** Step-by-step execution with JSON checkpointing, allowing each step to complete within 60-90s while maintaining session state between calls.

## Architecture

### Key Components

1. **[`SessionStore`](src/qwen_mcp/engines/session_store.py)** - Persistent state management
   - Atomic writes using `tempfile.mkstemp()` + `os.replace()`
   - Session lifecycle management (create, load, update_step, mark_failed, delete)
   - Cleanup for old sessions (24-hour TTL)

2. **[`SparringEngineV2`](src/qwen_mcp/engines/sparring_v2.py)** - Unified execution engine
   - Single `execute()` method with mode parameter
   - Reduced timeouts (60-90s per step vs original 300s)
   - Guided UX via `SparringResponse` with `next_step` hints

3. **[`qwen_sparring`](src/qwen_mcp/server.py)** - MCP tool interface
   - Single tool with mode parameter (replaces separate flash/pro tools)
   - Markdown-formatted responses with guided UX

## Modes

### sparring1 / flash (Quick Analysis)

| Mode | Timeout | Max Tokens | Session Required | Description |
|------|---------|------------|------------------|-------------|
| `flash` | 90s + 90s = 180s | 1024 + 1024 = 2048 | No | Quick analysis + draft (single call) |

**Use Case:** Szybkie decyzje, proste pytania. Czas generowania: ~10-20s.

### sparring2 / full (Standard Session)

| Mode | Timeout | Max Tokens | Session Required | Description |
|------|---------|------------|------------------|-------------|
| `discovery` | 100s | 512 | No (creates) | Create session + define roles |
| `red` | 100s | 1024 | Yes | Execute Red Cell critique |
| `blue` | 100s | 1024 | Yes + red | Execute Blue Cell defense |
| `white` | 100s | 1024 | Yes + red + blue | Execute White Cell synthesis |

**Total:** 3584 tokens (~2600 words), czas generowania: ~65-80s. Timeout: 180s z marginesem 100s.

**Use Case:** Standardowe analizy (80% przypadków).

### sparring3 / pro (Deep Analysis)

| Mode | Timeout | Max Tokens | Session Required | Description |
|------|---------|------------|------------------|-------------|
| `discovery` | 100s | 512 | No (creates) | Create session + define roles |
| `red` | 100s | 4096 | Yes | Execute Red Cell critique |
| `blue` | 100s | 4096 | Yes + red | Execute Blue Cell defense |
| `white` | 100s | 4096 | Yes + red + blue | Execute White Cell synthesis |

**Total:** 12800 tokens (~9500 words), czas generowania: ~245s. Timeout: 100s/krok = 400s całkowity.

**Use Case:** Złożone strategie, audyt.

## Usage

### Flash Mode (Quick Analysis)

```python
qwen_sparring(mode="flash", topic="Should we migrate to microservices?")
```

**Returns:** Complete analysis + draft in a single call. No session management needed.

### Step-by-Step Mode (Full Session)

#### Step 1: Discovery

```python
qwen_sparring(mode="discovery", topic="Migration strategy to microservices", context="Current monolith, team of 8")
```

**Returns:**
- Session ID: `sp_abc123`
- Defined roles (Red, Blue, White)
- Next step hint: `red`

#### Step 2: Red Cell Critique

```python
qwen_sparring(mode="red", session_id="sp_abc123")
```

**Returns:**
- Red Cell critique content
- Next step hint: `blue`

#### Step 3: Blue Cell Defense

```python
qwen_sparring(mode="blue", session_id="sp_abc123")
```

**Returns:**
- Blue Cell defense content
- Next step hint: `white`

#### Step 4: White Cell Synthesis

```python
qwen_sparring(mode="white", session_id="sp_abc123")
```

**Returns:**
- Final synthesis report
- Session status: `completed`

## Guided UX

Each response includes:

```markdown
✅ **Discovery completed!**

📋 **Session ID:** `sp_abc123`

🎭 **Wybrane role:**
   • Red: "Security Expert"
   • Blue: "System Architect"
   • White: "Technical Mediator"

---
➡️ **Next Step:** Red Cell Critique
💡 **Command:** `qwen_sparring(mode="red", session_id="sp_abc123")`
```

## Session Checkpointing

### JSON Schema

```json
{
  "session_id": "sp_abc123",
  "topic": "Migration strategy",
  "context": "Current monolith, team of 8",
  "created_at": "2026-03-22T14:00:00Z",
  "updated_at": "2026-03-22T14:05:00Z",
  "status": "in_progress",
  "steps_completed": ["discovery", "red"],
  "current_step": "blue",
  "roles": {
    "red_role": "Security Expert",
    "red_profile": "20 years in cybersecurity...",
    "blue_role": "System Architect",
    "blue_profile": "Designed distributed systems...",
    "white_role": "Technical Mediator",
    "white_profile": "15 years in conflict resolution..."
  },
  "results": {
    "discovery": {...},
    "red": {...}
  },
  "loop_count": 0,
  "error": null
}
```

### Storage Location

Sessions are stored in:
- **Windows:** `%APPDATA%/qwen-mcp/sessions/`
- **Linux/macOS:** `~/.config/qwen-mcp/sessions/`

### Atomic Write Pattern

```python
def _save_session(self, session: SessionCheckpoint) -> None:
    """Save session atomically using tempfile + rename."""
    data = session.to_dict()
    fd, temp_path = tempfile.mkstemp(
        suffix='.json',
        prefix=f'{session.session_id}_'
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self._get_session_path(session.session_id))
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
```

## Timeout and Token Configuration

### Timeouts

```python
TIMEOUTS = {
    "flash_analyst": 90.0,      # sparring1: 180s total for 2 steps
    "flash_drafter": 90.0,      # sparring1: 180s total for 2 steps
    "discovery": 100.0,         # sparring3: step-by-step, 100s per step
    "red_cell": 100.0,          # sparring3: step-by-step, 100s per step
    "blue_cell": 100.0,         # sparring3: step-by-step, 100s per step
    "white_cell": 100.0,        # sparring3: step-by-step, 100s per step
}
```

### Max Tokens (Output Length Control)

```python
MAX_TOKENS_CONFIG = {
    "flash": {
        "analyst": 1024,        # sparring1: quick analysis
        "drafter": 1024,        # sparring1: strategy draft
    },
    "full": {
        "discovery": 512,       # sparring2: role definition (JSON)
        "red": 1024,            # sparring2: critique
        "blue": 1024,           # sparring2: defense
        "white": 1024,          # sparring2: synthesis
    },
    "pro": {
        "discovery": 512,       # sparring3: role definition (JSON)
        "red": 4096,            # sparring3: deep critique
        "blue": 4096,           # sparring3: detailed defense
        "white": 4096,          # sparring3: comprehensive synthesis
    },
}
```

**Total Token Budget:**
- **sparring1 (flash):** 2048 tokens (~1500 words, ~2-3 min gen time)
- **sparring2 (full):** 3584 tokens (~2600 words, ~65-80s gen time)
- **sparring3 (pro):** 12800 tokens (~9500 words, ~245s gen time)

**Rationale:**
- Timeouts are set to ensure completion well under the 300s MCP client limit
- `max_tokens` provides predictable output length and generation time
- Token limits prevent timeout risks by controlling response length

## Migration Guide

### From Old API

**Before:**
```python
# Two separate tools
qwen_sparring_flash(topic="...")
qwen_sparring_pro(topic="...")  # Risk of timeout!
```

**After:**
```python
# Single unified tool
qwen_sparring(mode="flash", topic="...")
qwen_sparring(mode="discovery", topic="...")  # Start session
qwen_sparring(mode="red", session_id="...")   # Continue
qwen_sparring(mode="blue", session_id="...")  # Continue
qwen_sparring(mode="white", session_id="...") # Complete
```

### Breaking Changes

1. **Removed `full` mode** - Replaced by step-by-step execution (flash exists for simple topics)
2. **Session ID required** - Red/Blue/White modes now require explicit `session_id` parameter
3. **No automatic loops** - `[REGENERATE]` responses must be handled manually by user

## Testing

### Run Tests

```bash
uv run pytest tests/test_session_store.py tests/test_sparring_v2.py tests/test_sparring_stage_recovery.py -v
```

### Test Coverage

- **SessionStore:** 31 tests (creation, atomic save/load, step updates, status transitions, cleanup)
- **SparringEngineV2:** 18 tests (flash, discovery, red/blue/white cells, full flow, guided UX)
- **Stage Recovery:** 23 tests (BudgetManager, CircuitBreaker, StageResult, StageContext, TTL, recovery)

**Total:** 72 passing tests

### Stage Recovery Test Suite

The new `test_sparring_stage_recovery.py` validates the stage-based architecture:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestBudgetManager` | 5 | Dynamic timeout allocation, remaining budget tracking, pro/flash weights |
| `TestCircuitBreaker` | 5 | State transitions, failure threshold, recovery timeout, HALF_OPEN state |
| `TestStageDataclasses` | 3 | StageResult success/failure, StageContext creation |
| `TestSessionStoreTTL` | 5 | TTL expiration, checkpoint persistence, ephemeral TTL constant |
| `TestStageRecovery` | 3 | Recovery from failed stage, checkpoint creation, budget tracking |
| `TestConfigModule` | 3 | Stage weights validation, budget config coverage, circuit breaker values |

**Key Test Scenarios:**
- Circuit breaker opens after 3 consecutive failures
- BudgetManager correctly allocates time based on stage weights
- SessionCheckpoint TTL expiration for ephemeral checkpoints (flash mode)
- Recovery from failed stage continues execution
- Checkpoints created after each successful stage

## Files

| File | Purpose |
|------|---------|
| [`src/qwen_mcp/engines/session_store.py`](src/qwen_mcp/engines/session_store.py) | Session checkpointing with atomic writes, TTL support |
| [`src/qwen_mcp/engines/sparring_v2/engine.py`](src/qwen_mcp/engines/sparring_v2/engine.py) | Unified sparring engine |
| [`src/qwen_mcp/engines/sparring_v2/base_stage_executor.py`](src/qwen_mcp/engines/sparring_v2/base_stage_executor.py) | Stage-based execution architecture |
| [`src/qwen_mcp/engines/sparring_v2/config.py`](src/qwen_mcp/engines/sparring_v2/config.py) | Budget, stage weights, circuit breaker config |
| [`src/qwen_mcp/engines/sparring_v2/modes/pro.py`](src/qwen_mcp/engines/sparring_v2/modes/pro.py) | Pro mode executor (sparring3) |
| [`src/qwen_mcp/engines/sparring_v2/modes/full.py`](src/qwen_mcp/engines/sparring_v2/modes/full.py) | Full mode executor (sparring2) |
| [`src/qwen_mcp/engines/sparring_v2/modes/flash.py`](src/qwen_mcp/engines/sparring_v2/modes/flash.py) | Flash mode executor (sparring1) |
| [`src/qwen_mcp/tools.py`](src/qwen_mcp/tools.py:137) | MCP tool wrapper |
| [`src/qwen_mcp/server.py`](src/qwen_mcp/server.py:112) | MCP tool registration |
| [`tests/test_session_store.py`](tests/test_session_store.py) | SessionStore tests |
| [`tests/test_sparring_v2.py`](tests/test_sparring_v2.py) | Engine tests |
| [`tests/test_sparring_stage_recovery.py`](tests/test_sparring_stage_recovery.py) | Stage recovery tests |

## Legacy Files (Archived)

- `_archive/legacy_sparring/sparring.py` - Original monolithic engine
- `_archive/legacy_sparring/test_sparring_loop.py` - Old loop tests
- `_archive/legacy_sparring/test_sparring_discovery.py` - Old discovery tests

## Design Decisions

### Why Single Tool with Mode Parameter?

**User Request:** "czy nie powinno być łatwiejsze niż tworzenie 3 czy 4 osobnych tooli?"

**Decision:** Single tool `qwen_sparring(mode="...")` instead of separate tools (`qwen_sparring_red`, `qwen_sparring_blue`, etc.) because:
- Reduces tool clutter (already 16 tools in qwen-coding)
- Clearer semantic grouping
- Easier to maintain and extend

### Why JSON Checkpointing Instead of Database?

**Core 80% Principle:** JSON files with atomic writes provide sufficient reliability without database complexity.

**Benefits:**
- No external dependencies
- Human-readable debug format
- Simple cleanup (file deletion)
- Cross-platform compatibility

### Why Remove `full` Mode?

**User Quote:** "skoro mamy 'step' - to po jasną cholere trzymać nam ten 'full'..."

**Decision:** Removed `full` mode because:
- Still risks timeout (monolithic execution)
- `flash` mode exists for simple topics
- Step-by-step provides better UX with guided progression

### Why Guided Mode?

**User Request:** "czy tool będzie wiedział i podpowiadał?"

**Implementation:** Each `SparringResponse` includes:
- `next_step`: Suggested next mode
- `next_command`: Copy-paste ready command

## Future Enhancements

1. **Session Resume** - Allow resuming failed sessions from any step ✅ (Phase 1-7 implemented)
2. **Session History** - List all sessions with filtering
3. **Export Reports** - Generate PDF/Markdown reports from completed sessions
4. **Parallel Red/Blue** - Execute Red and Blue cells in parallel (requires model support)

## Stage-Based Architecture Details

### BudgetManager

Dynamic timeout allocation based on stage weights:

```python
from qwen_mcp.engines.sparring_v2.base_stage_executor import BudgetManager

# Pro mode: 225s total budget
weights = {"discovery": 0.15, "red": 0.28, "blue": 0.28, "white": 0.29}
budget = BudgetManager(total_budget_seconds=225, stage_weights=weights)

# Get stage-specific budget
discovery_budget = budget.get_stage_budget("discovery")  # 33 seconds
red_budget = budget.get_stage_budget("red")              # 63 seconds

# Track usage
budget.record_usage("discovery", seconds_used=30.5)

# Check remaining budget
remaining = budget.get_remaining_budget()  # ~194 seconds
```

### CircuitBreaker

Failure recovery with automatic state management:

```python
from qwen_mcp.engines.sparring_v2.base_stage_executor import CircuitBreaker

cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

# Record failures
cb.record_failure()
cb.record_failure()
cb.record_failure()

# Circuit is now OPEN
print(cb.state)  # "OPEN"
print(cb.can_execute())  # False

# After 60 seconds recovery timeout
time.sleep(60)
print(cb.can_execute())  # True (transitions to HALF_OPEN)

# On success, reset to CLOSED
cb.record_success()
print(cb.state)  # "CLOSED"
```

### Stage Weights Configuration

Default weights in [`config.py`](src/qwen_mcp/engines/sparring_v2/config.py):

```python
STAGE_WEIGHTS = {
    "pro": {
        "discovery": 0.15,  # 33.75s of 225s
        "red": 0.28,        # 63s
        "blue": 0.28,       # 63s
        "white": 0.29,      # 65.25s
    },
    "full": {
        "discovery": 0.15,
        "red": 0.28,
        "blue": 0.28,
        "white": 0.29,  # includes regeneration budget
    },
    "flash": {
        "analyst": 0.45,   # 27s of 60s
        "drafter": 0.55,   # 33s
    },
}

BUDGET_CONFIG = {
    "pro": 225,      # 225 seconds for 4-stage execution
    "full": 225,     # 225 seconds (includes regeneration loop)
    "flash": 60,     # 60 seconds for fast 2-step analysis
}

CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 3,   # Open after 3 failures
    "recovery_timeout": 60,   # 60 seconds recovery
}

EPHEMERAL_TTL = 300  # 5 minutes TTL for flash mode checkpoints
```

### BaseStageExecutor Interface

All mode executors inherit from `BaseStageExecutor`:

```python
from qwen_mcp.engines.sparring_v2.base_stage_executor import (
    BaseStageExecutor, StageContext, StageResult
)

class ProExecutor(BaseStageExecutor):
    STAGES = ["discovery", "red", "blue", "white"]
    
    def get_stages(self) -> List[str]:
        return self.STAGES
    
    def get_stage_weights(self) -> Dict[str, float]:
        return STAGE_WEIGHTS["pro"]
    
    async def execute_stage(self, stage_name: str, context: StageContext) -> StageResult:
        # Implement stage-specific logic
        ...
    
    async def execute(self, ctx: Optional[Context] = None, **kwargs) -> SparringResponse:
        # Use execute_with_recovery() for automatic checkpointing
        context = StageContext(...)
        results = await self.execute_with_recovery(context)
        # Format and return response
```

### Recovery Workflow

1. **Stage Execution**: Each stage executes with its allocated budget
2. **Checkpoint**: After successful stage, checkpoint saved to disk
3. **Failure Handling**: On failure, circuit breaker records failure
4. **Recovery**: If circuit is OPEN, skip stage and continue
5. **Resume**: Failed stages can be retried in subsequent calls

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Discovery  │ ──► │     Red     │ ──► │    Blue     │
│  (33.75s)   │     │   (63s)     │     │   (63s)     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  [Checkpoint]       [Checkpoint]       [Checkpoint]
                          │
                    (if failure)
                          ▼
                   ┌─────────────┐
                   │CircuitBreaker│
                   │  OPEN 60s    │
                   └─────────────┘
                          │
                    (after timeout)
                          ▼
                   ┌─────────────┐
                   │   Resume    │
                   │  from Blue  │
                   └─────────────┘
```
