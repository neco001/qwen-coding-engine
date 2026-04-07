# Sparring Engine v2 - Step-by-Step Adversarial Analysis

## Overview

Sparring Engine v2 is a refactored version of the original `qwen_sparring_pro` tool, designed to solve MCP client timeout issues (300s hard limit) by breaking monolithic multi-agent sessions into discrete, checkpointed steps.

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
uv run pytest tests/test_session_store.py tests/test_sparring_v2.py -v
```

### Test Coverage

- **SessionStore:** 31 tests (creation, atomic save/load, step updates, status transitions, cleanup)
- **SparringEngineV2:** 18 tests (flash, discovery, red/blue/white cells, full flow, guided UX)

**Total:** 49 passing tests

## Files

| File | Purpose |
|------|---------|
| [`src/qwen_mcp/engines/session_store.py`](src/qwen_mcp/engines/session_store.py) | Session checkpointing with atomic writes |
| [`src/qwen_mcp/engines/sparring_v2.py`](src/qwen_mcp/engines/sparring_v2.py) | Unified sparring engine |
| [`src/qwen_mcp/tools.py`](src/qwen_mcp/tools.py:137) | MCP tool wrapper |
| [`src/qwen_mcp/server.py`](src/qwen_mcp/server.py:112) | MCP tool registration |
| [`tests/test_session_store.py`](tests/test_session_store.py) | SessionStore tests |
| [`tests/test_sparring_v2.py`](tests/test_sparring_v2.py) | Engine tests |

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

1. **Session Resume** - Allow resuming failed sessions from any step
2. **Session History** - List all sessions with filtering
3. **Export Reports** - Generate PDF/Markdown reports from completed sessions
4. **Parallel Red/Blue** - Execute Red and Blue cells in parallel (requires model support)
