# CHANGELOG

doc: backlog and changelog update

## SOS Sync - 2026-04-12 21:20:19

## [2026-04-12 21:16:51] f4b59636-2c52-4488-aa32-be6efd91245f

**Task**: Snapshot naming convention and auto-selection

**Advice**: ## Problem
Current snapshot naming is inconsistent (e.g., 'baseline', 'batch-task-feature', 'final_perf_test_devnull', 'baseline_20260412'). Users cannot easily identify which snapshots to compare, and manual selection is error-prone.

## Proposed Solution

### 1. Standardized Naming Convention

**Format:** `baseline-YYYYMMDD_HHMMSS.json`

- Example: `baseline-20260412_153719.json`
- Timestamp in UTC for consistency
- Generated automatically when creating baseline

### 2. Auto-Selection Logic for `qwen_compare_snapshots_tool`

**Default behavior (no parameters):**

- Automatically select two newest snapshots by timestamp
- Compare them and return results

**Explicit selection (parameters provided):**

- Allow user to specify snapshot1_name and snapshot2_name
- Validate both snapshots exist before comparison

### 3. Implementation Changes

#### A. `src/graph/snapshot.py` - FunctionalSnapshotGenerator

- Modify `save_snapshot()` to auto-generate timestamped name if name='auto' or None
- Add `list_snapshots()` method to return sorted list of available snapshots
- Add `get_two_newest_snapshots()` helper method

#### B. `src/qwen_mcp/diff_audit.py` - QwenDiffAuditTool

- Update `compare_snapshots()` to support auto-selection
- Add logic to detect 'auto' parameter and select two newest

#### C. `src/qwen_mcp/server.py` - MCP Tool

- Update `qwen_compare_snapshots_tool()` signature:
  ```python
  async def qwen_compare_snapshots_tool(
      snapshot1_name: Optional[str] = "auto",
      snapshot2_name: Optional[str] = "auto",
      workspace_root: str = "."
  ) -> str:
  ```
- When both are "auto", automatically select two newest snapshots
- When one or both specified, use explicit names

#### D. `src/qwen_mcp/anti_degradation_config.py`

- Add config option for snapshot naming pattern
- Add config for auto-compare default behavior

### 4. Migration/Cleanup

- Add utility to rename existing snapshots to new format (optional)
- Document new naming convention in README.md

### 5. Testing

- Test auto-selection with various snapshot counts (0, 1, 2, 10+)
- Test explicit selection still works
- Test edge cases (missing snapshots, corrupted files)

## Files to Modify

1. `src/graph/snapshot.py` - Core snapshot management
2. `src/qwen_mcp/diff_audit.py` - Comparison logic
3. `src/qwen_mcp/server.py` - MCP tool interface
4. `src/qwen_mcp/anti_degradation_config.py` - Configuration
5. `README.md` - Documentation update

## Risk Assessment

- **Risk:** Low - changes are additive, existing explicit selection still works
- **Backward Compatibility:** Maintained - default 'auto' behavior is new, explicit names still work
- **Testing:** Unit tests for auto-selection logic required

---

## 2026-04-12 21:20 - 2bd2e4d8-da2f-4bc2-a3e7-cc77df24bcf3

**Task**: Implement snapshot naming convention changes in src/graph/snapshot.py:

1. Add a static method `_generate_timestamped_name()` that returns format `baseline-YYYYMMDD_HHMMSS` using UTC datetime

2. Add

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-11 23:57:42

## [2026-04-11 21:09:54] 70f149bd-6d08-4256-8522-c98d13307073

**Task**: Pytanie dla architecta: LangGraph w projekcie?

**Advice**: Analiza zasadności LangGraph.

---

## [2026-04-11 21:09:54] 3c84b2d4-d0df-40ff-b91e-6001a1674340

**Task**: Fallback dla braku pliku decision_log.parquet

**Advice**: Implementacja fallbacku dla .parquet w qwen_add_task i qwen_sync_state.

---

## [2026-04-11 21:46:13] e46c7037-73da-410e-ad13-d3517631c208

**Task**: Wdrożenie stabilnego resolwowania ścieżek (Plan Architekta)

**Advice**: Wdrożyć poprawki ścieżek wg projektu architekta: inteligentne wykrywanie roota projektu (.git/pyproject.toml) oraz defensywne tworzenie katalogów w mechanizmie lockowania.

---

## [2026-04-11 22:19:21] e0e88da0-0780-4f10-afdd-3e58be47c9cc

**Task**: Krok 1: Izolacja Strumieni Loggingu - Force all logging to stderr in server.py

**Advice**: ## Problem
MCP protocol requires stdout to be used exclusively for JSON-RPC communication. Any logging on stdout breaks the protocol.

## Implementation

In `src/qwen_mcp/server.py`:

1. Add explicit logging configuration before FastMCP initialization (line ~45)
2. Force all handlers to use `stream=sys.stderr`
3. Verify no `print()` statements in production code

## Code Change

```python
# After imports, before logger = logging.getLogger(__name__)
import logging
import sys

# Force logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,  # NEVER stdout - MCP uses stdout for JSON-RPC
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Verification

Run server and check that no output appears on stdout before JSON-RPC messages.

---

## [2026-04-11 22:19:35] d71c5bb6-5ce1-4e22-b1ce-c0875f80fc3f

**Task**: Krok 2: Ograniczenie Współbieżności I/O - Tune chunk_size and sleep in snapshot.py

**Advice**: ## Problem
`asyncio.gather()` on 107 files can starve the event loop, preventing MCP heartbeat handling. The current implementation already has chunking (chunk_size=20) and `asyncio.sleep(0.01)` - this is good but may need tuning.

## Current State (snapshot.py:235-240)

```python
chunk_size = 20
for i in range(0, len(tasks), chunk_size):
    chunk = tasks[i:i + chunk_size]
    chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
    results.extend(chunk_results)
    await asyncio.sleep(0.01)  # Force yield to let MCP server respond to pings
```

## Potential Improvements

1. Reduce chunk_size from 20 to 10 for more frequent yields
2. Increase sleep from 0.01 to 0.05 for better heartbeat response
3. Add explicit semaphore for controlled concurrency

## Verification

Test with MCP client and monitor event loop latency during execution.

---

## [2026-04-11 22:19:49] 279b0792-d84c-46d0-9302-d3a71a09200b

**Task**: Krok 3: Optymalizacja Payloadu - Verify response returns only metadata, not full snapshot

**Advice**: ## Problem
Returning full snapshot content in JSON-RPC response may exceed stdio buffer limits (64KB on Linux/Unix). The current implementation returns the full path string, but the snapshot itself contains 107 files' data.

## Current State (diff_audit.py:205-209)

```python
async def create_baseline_snapshot(self, name: str = "baseline") -> str:
    """Create a new baseline snapshot."""
    snapshot = await self.snapshot_generator.capture_snapshot(self.repo_path)
    path = self.snapshot_generator.save_snapshot(snapshot, self.repo_path, name)
    return str(path)  # Already returns path only - GOOD!
```

## Analysis

The current implementation already returns only the path string, not the full snapshot content. This is correct! The snapshot is saved to disk and only the path is returned via JSON-RPC.

## Potential Issue

The snapshot file itself may be large, but that's saved to disk, not transmitted via stdio. The current design is already optimized.

## Verification

Check if any other tools return large payloads. Monitor actual JSON-RPC response size.

---

## [2026-04-11 22:20:03] a85af225-c2c0-4341-a088-b7eebb839d1a

**Task**: Krok 4: Weryfikacja Krzyżowa Klienta - Test with alternative MCP client to isolate Roo Code issues

**Advice**: ## Problem
Need to isolate whether the timeout issue is specific to Roo Code client or a general MCP server problem.

## Implementation

1. Test with alternative MCP client (Claude Desktop, mcp-inspector, or raw stdio test)
2. If alternative client works, issue is in Roo Code client implementation
3. If alternative client also fails, issue is in server code

## Test Procedure

```bash
# Option 1: Use mcp-inspector
npx @modelcontextprotocol/inspector python -m qwen_mcp.server

# Option 2: Raw stdio test (already done in scratch/test_mcp_stdio.py)
# This showed SUCCESS in 0.80s

# Option 3: Claude Desktop (if available)
# Configure MCP server in Claude Desktop settings
```

## Current Evidence

The scratch/test_mcp_stdio.py test already passed (0.80s), suggesting the server works correctly when called directly via stdio simulation.

## Verification

If alternative client works, document Roo Code-specific timeout behavior and recommend client-side configuration changes.

---

## [2026-04-11 22:20:19] aba4f3aa-260f-49a1-9709-834ed08bc060

**Task**: Krok 5: Instrumentacja - Replace file-based trace with proper stderr logging in snapshot.py

**Advice**: ## Problem
Need visibility into the execution process for future debugging. Currently there's a debug trace function in snapshot.py that writes to a file, but we need proper stderr logging.

## Current State (snapshot.py:186-188)

```python
def trace(msg):
    with open("debug_trace.log", "a") as f:
        f.write(f"{datetime.now().time()}: {msg}\n")
```

## Implementation

1. Replace file-based trace with proper stderr logging
2. Add timing logs at key points:
   - Start of capture_snapshot
   - After file gathering
   - After asyncio.gather
   - End of operation
3. Log snapshot size before returning

## Code Change

```python
import logging
import sys

logger = logging.getLogger(__name__)

# Replace trace() with:
def trace(msg):
    logger.info(f"[SNAPSHOT] {msg}")
```

## Verification

Run tool and check stderr logs show timing information without polluting stdout.

---

## 2026-04-11 23:57 - 7a654cc9-f300-4065-b666-ece2540e2647

**Task**: Implement the `add_tasks` method (batch version) in `src/qwen_mcp/engines/decision_log_sync.py`.

The method should be added after the existing `add_task` method (around line 362). It should:

1. Acc

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-11 18:59:07

## [2026-04-11 18:56:44] 096f6e58-d2a0-458f-bde5-49c007cf5091

**Task**: Fix snapshot storage location to use .anti_degradation/snapshots from config

**Advice**: FunctionalSnapshotGenerator.save_snapshot() and load_snapshot() use hardcoded '.snapshots' path (lines 680, 700) instead of reading storage_dir from AntiDegradationConfig (line 57). This causes snapshots to be saved in wrong location. Fix by: (1) Add storage_dir parameter to FunctionalSnapshotGenerator.**init**(), (2) Update save_snapshot() and load_snapshot() to use config-based path, (3) Update QwenDiffAuditTool and PreCommitHook to load config and pass storage_dir, (4) Migrate existing snapshots from .snapshots/ to .anti_degradation/snapshots/

---

## 2026-04-11 18:59 - 73735cea-d502-4f06-848f-c2ae60d055e4

**Task**: Implement fix for snapshot storage location in src/graph/snapshot.py:

1. Update FunctionalSnapshotGenerator.**init**() to accept optional storage_dir parameter
2. Import get_config from qwen_mcp.anti

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-10 22:18:11

## [2026-04-10 22:11:58] b5e595af-8949-4393-acc6-d85277225af8

**Task**: T8: Optimize FunctionalSnapshotGenerator.capture_snapshot() to use git diff for file selection

**Advice**: Add \_get_changed_files() method to FunctionalSnapshotGenerator that uses git diff --name-only to get only changed Python files. Modify capture_snapshot() signature to accept commit_range and changed_files parameters. Fallback to rglob if no files changed or git fails.

---

## [2026-04-10 22:12:06] fcd7e99c-4160-47c8-bd66-84cb3a10e869

**Task**: T9: Add parallel processing with asyncio.gather for file snapshots

**Advice**: Add \_capture_file_snapshot_async() helper method to FunctionalSnapshotGenerator. Refactor capture_snapshot() to use asyncio.gather() for parallel file processing instead of sequential loop. This will provide 4x speedup for large file sets.

---

## [2026-04-10 22:12:15] 475ef705-4a2c-4ca7-b205-c4095d233cce

**Task**: T10: Optimize \_generate_content_hashes() to only hash changed files

**Advice**: Modify \_generate_content_hashes() signature to accept changed_files parameter. When changed_files is provided, only hash those files instead of scanning all files with glob patterns. This reduces content hash generation from ~30s to ~3s for typical changes.

---

## [2026-04-10 22:12:23] 00d85a74-48b1-4ea7-9f35-3861ec1dd24d

**Task**: T11: Update qwen_diff_audit to pass changed files to capture_snapshot

**Advice**: Update QwenDiffAuditTool.audit_diff() to extract changed files from GitDiffResult and pass them to snapshot_generator.capture_snapshot(). Also update qwen_diff_audit() wrapper function to support the changed_files parameter flow.

---

## [2026-04-10 22:12:31] 87e79cef-c7f1-4dc6-b0a9-a280a1fe6a93

**Task**: T12: Test optimized snapshot capture performance

**Advice**: Test the optimized FunctionalSnapshotGenerator with git diff integration and parallel processing. Verify that snapshot capture completes in <10s for typical changes (1-5 files) and <60s for larger changes (10-20 files). Compare before/after performance metrics.

---

## 2026-04-10 22:18 - 8ad48368-ed9d-4fad-b8ed-9fbfebebc1a0

**Original Task**: T8: Optimize FunctionalSnapshotGenerator.capture_snapshot() to use git diff for file selection  
**Decision ID**: `b5e595af-8949-4393-acc6-d85277225af8` → `8ad48368-ed9d-4fad-b8ed-9fbfebebc1a0`

**Status**: ✅ Completed

---

## 2026-04-10 14:39 - a5145df9-86f6-4e11-b44c-ba1c7e5af956

**Task**: Create production blocking components for the Anti-Degradation System.

**Task 1**: Create `.github/workflows/anti_degradation_production.yml`

- Same as shadow workflow but WITHOUT continue-on-error
- **Status**: ✅ Completed

  ***

## 2026-04-10 14:36 - 14040cc4-59de-477a-a216-147137f6c958

**Task**: Create a GitHub Actions workflow for the Anti-Degradation System CI integration.

**Task**: Create `.github/workflows/anti_degradation.yml`

**Requirements**:

1. Trigger on: pull_request (opened, sync

**Status**: ✅ Completed

---

## 2026-04-10 14:03 - 82ad4d98-6825-4848-9515-da228e0f9a36

**Task**: Create a configuration loader module for the Anti-Degradation System.

**Task**: Create `src/qwen_mcp/anti_degradation_config.py`

**Requirements**:

1. Use dataclasses for typed configuration
2. Load

**Status**: ✅ Completed

---

## 2026-04-10 13:59 - 90109f25-ed29-423f-89b8-b182188ea5a9

**Task**: Create a shadow mode configuration file for the Anti-Degradation System.

**Task**: Create `.anti_degradation/config.yaml`

**Requirements**:

1. Define shadow_mode section with:
   - enabled: boolean

**Status**: ✅ Completed

---

## 2026-04-10 13:23 - 766d480d-625b-466d-9509-32941795e8d8

**Task**: Create a pre-commit hook script for the Anti-Degradation System.

**Task**: Create `scripts/pre_commit_hook.py`

**Requirements**:

1. Must complete within 3 seconds latency requirement
2. Use the GitD

**Status**: ✅ Completed

---

## 2026-04-10 13:04 - d7db8d3a-6c35-43e3-ad07-a58775655cd6

**Task**: Implement Anti-Degradation System tasks T1-T7 from the backlog.

Context from sparring session sp_603b13d824b6:

- Option A+C approach validated: qwen_diff_audit MCP tool + pre-commit hook with Functi

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-10 12:10:06

## [2026-04-10 11:38:36] 9ab88b43-1995-4bf2-ab47-5c56192002ad

**Task**: T1: Content Hashing w Snapshotach

**Advice**: Dodaj content_hash do function/class entries w \_capture_file_snapshot() w src/graph/snapshot.py. Użyj SHA256 hash treści funkcji/klasy (body). Zmodyfikuj linie 75-81 dla functions i 93-99 dla classes. Wymagane: import hashlib, metoda \_get_content_hash(node, source). Test: test_snapshot_hash_deterministic() musi przejść.

---

## [2026-04-10 11:38:46] d1d59443-1959-4695-841d-75bd907d7dfd

**Task**: T2: Git Diff Parser

**Advice**: Stwórz nowy moduł src/graph/git_diff_parser.py z klasą GitDiffParser. Metoda get_diff_between_refs(before_ref, after_ref, project_dir) zwraca dict z changed_files, added_lines, removed_lines, diff_content. Użyj subprocess do wywołania "git diff". Obsłuż brak git gracefully. Test: test_git_diff_parser() musi przejść.

---

## [2026-04-10 11:38:56] d95d55d3-b124-48e9-a89a-13ea050c66f3

**Task**: T3: qwen_diff_audit MCP Tool

**Advice**: Dodaj nowy MCP tool qwen_diff_audit w src/qwen_mcp/tools.py. Tool przyjmuje before_ref (default HEAD~1), after_ref (default HEAD), project_dir. Używa GitDiffParser i FunctionalSnapshotGenerator. Zwraca: changed_files, risk_score (0.0-1.0), removed_functions, removed_classes, signature_changes, regression_alerts. Dodaj do **all** list. Rejestracja w MCP server. Test: test_qwen_diff_audit() musi przejść.

---

## [2026-04-10 11:39:06] 73421104-421d-4549-95a3-c372c8cb3b8f

**Task**: T4: Pre-Commit Hook Script

**Advice**: Stwórz scripts/pre_commit_degradation.py - lokalny hook do wykrywania degradacji. Używa FunctionalSnapshotGenerator do porównania z ostatnim snapshotem w .qwen/last_snapshot.json. SHADOW_MODE=True domyślnie (tylko ostrzeżenia, nie blokuje commit). Musi działać w <3 sekundy. Zapisuje current snapshot po każdym commicie. Test: test_pre_commit_hook() musi przejść.

---

## [2026-04-10 11:39:19] 343b1a3f-520f-4c8c-a786-1d30ff8da66e

**Task**: T5: Shadow Mode Configuration

**Advice**: Stwórz .qwen/config.yaml z konfiguracją anti-degradation: shadow_mode: true (domyślnie ostrzeżenia bez blokady), block_threshold: 0.7 (próg risk_score do blokady w trybie produkcyjnym), false_positive_log: .qwen/false_positives.log, snapshot_storage: .qwen/snapshots/. Konfiguracja musi być czytana przez T4 i T6.

---

## [2026-04-10 11:39:29] 8e4f115b-88f6-4ff7-9986-7a032580d3a9

**Task**: T6: CI Workflow Integration

**Advice**: Stwórz .github/workflows/degradation_check.yml - GitHub Actions workflow. Uruchamia się na push i pull_request. Krok "Setup Python" i "Install dependencies" (pip install -e .). Główny krok: uruchom python scripts/pre_commit_degradation.py. SHADOW_MODE kontrolowany przez env var (domyślnie true). Upload false-positive metrics jako artifact. Wymaga fetch-depth: 2 do porównania commitów.

---

## [2026-04-10 11:39:37] 6d3a5a83-321f-4ed8-aef0-4cd249c46f73

**Task**: T7: Production Blocking Activation

**Advice**: Aktywuj blokadę produkcyjną w CI po okresie Shadow Mode. Zmien SHADOW_MODE=false w .github/workflows/degradation_check.yml i .qwen/config.yaml. Wymagania: false-positive rate < 5% przez minimum 1 tydzień. Dokumentuj metryki w CHANGELOG.md. Dodaj dokumentację do docs/ANTI_DEGRADATION.md z instrukcją konfiguracji i interpretacji alertów.

---

## [2026-04-10 12:07:31] 00bee066-f4a4-4530-b731-c91b8813b4fe

**Task**: MCP Task Management Tools: qwen_list_tasks, qwen_get_task, qwen_update_task

**Advice**: Implementacja 3 nowych MCP toolow w src/qwen_mcp/tools.py: (1) qwen_list_tasks - listuje taski z BACKLOG.md z opcjonalnym filtrem tagow, zwraca JSON z taskami, (2) qwen_get_task - pobiera pojedynczy task po decision_id z decision_log.parquet, zwraca szczegoly tasku, (3) qwen_update_task - aktualizuje status tasku (pending/completed) w BACKLOG.md i decision_log.parquet. Uzyj dekoratora @mcp.tool() jak w innych toolach. Dodaj do **all**. Wymagany import pandas dla parquet. Struktura: **all** linie 16-33, add_task_to_backlog linie 600+.

---

## 2026-04-10 12:10 - b4a7a837-ec00-4212-985c-bc3c4039f0ef

**Task**: Implement 3 new MCP tools in src/qwen_mcp/tools.py:

1. **qwen_list_tasks** - List all tasks from BACKLOG.md with optional tag filter
   - Parse BACKLOG.md format: "- [ ] TaskName - decision_id" or "-

**Status**: ✅ Completed

---

## 2026-04-10 10:55 - e8408898-0a9c-4c90-ba97-e1ce4ecdcc9a

**Task**: napisz funkcję Pythona: n = a^x + b^y

**Status**: ✅ Completed

---

## 2026-04-09 20:15 - 10fba463-b4a1-484e-a086-3a98bc030666

**Task**: Extract and present the complete sparring analysis results from the session. The user wants to see the full findings from Red Cell, Blue Cell, and White Cell in a readable format. Present it as a comp

**Status**: ✅ Completed

---

## Release v1.1.1 - 2026-04-09

**Sparring Session File Location Fix**

- Fixed agent not knowing where to find sparring session results
- Added session file path display to `SparringResponse.to_markdown()`
- Agent now sees full path: `{storage_dir}/{session_id}.json`
- Fixed file extension mismatch (.json vs .md confusion)
- Works with all storage directory resolution tiers (env, user-level, fallback)
- Added 5 unit tests in `tests/test_sparring_session_path.py`

**max_tokens=0 Truncation Fix**

- Fixed Python falsy evaluation causing sparring3 (pro) responses to be truncated mid-sentence
- Root cause: `if max_tokens:` in completions.py treated 0 as falsy, ignoring unlimited token setting
- Fix: Changed to `if max_tokens is not None:` to properly handle max_tokens=0
- Configuration: `MAX_TOKENS_CONFIG` set to 0 for all sparring modes (flash/full/pro)
- Applied to: qwen_architect, qwen_coder, qwen_audit, qwen_sparring, qwen_update_session_context
- Added unit tests in `tests/test_max_tokens_zero.py` for sparring3 verification

**Files Changed:**

- `src/qwen_mcp/engines/sparring_v2/models.py`: Added `storage_dir` parameter to `to_markdown()`
- `src/qwen_mcp/tools.py`: Pass session store directory to response formatter
- `src/qwen_mcp/completions.py`: Fixed max_tokens zero check (`is not None` instead of falsy check)
- `src/qwen_mcp/engines/sparring_v2/config.py`: Set `MAX_TOKENS_CONFIG` to 0 for unlimited tokens
- `src/qwen_mcp/api.py`: Set `max_tokens=0` for meta-analysis endpoint
- `src/qwen_mcp/engines/context_builder.py`: Set `max_tokens=0` for session supplement generation

---

## SOS Sync - 2026-04-09 18:34:44

## [2026-04-09 18:24:49] 87722483-5a9b-4ebe-b331-c23fd41bec81

**Task**: Synchronizacja BACKLOG.md - dodać wszystkie pending z decision_log.parquet

**Advice**: Run DecisionLogSyncEngine to sync all 11 pending tasks from decision_log.parquet to BACKLOG.md, including the max_tokens=0 sparring tasks (ca0d090b, 96730cc3, 3638bc26, ba042a98, e28078c9)

---

## SOS Sync - 2026-04-09 18:22:52

## [2026-04-09 18:22:43] 1b7f8745-36e8-44fc-b773-132fed40036a

**Advice**: Centralize SOS Sync paths using SOSPathsConfig Pydantic class instead of hardcoded strings in tools.py (3 functions) and decision_log_sync.py. Fix bug where qwen_sync_state wrote to PLAN/ instead of .PLAN/.

---

## SOS Sync Architecture Redesign - 2026-04-08

**Task**: Redesign SOS Sync architecture - BACKLOG.md vs decision_log.parquet granularity - 240730f0-5907-4d86-b717-f997a29706dd

**Completed**:

- Refactored [`src/qwen_mcp/engines/decision_log_sync.py`](src/qwen_mcp/engines/decision_log_sync.py):
  - Added `archive_completed: bool = True` parameter to [`complete_task()`](src/qwen_mcp/engines/decision_log_sync.py:484) - removes tasks from BACKLOG.md instead of just marking [x]
  - Created [`_remove_from_backlog()`](src/qwen_mcp/engines/decision_log_sync.py:583) - surgically removes completed tasks from BACKLOG.md
  - Enhanced [`_append_changelog()`](src/qwen_mcp/engines/decision_log_sync.py:688) with full details from parquet:
    - `matched_task_description` - original task description
    - `files_changed` - list of modified files
    - `tokens_used` - token consumption
  - Added query interface for decision_log.parquet:
    - [`query_decisions()`](src/qwen_mcp/engines/decision_log_sync.py:95) - filter by status, task_type, date range, tags
    - [`get_recent_completions()`](src/qwen_mcp/engines/decision_log_sync.py:154) - recent completions helper
    - [`get_pending_tasks()`](src/qwen_mcp/engines/decision_log_sync.py:173) - pending tasks helper
- Created [`tests/test_sos_sync_redesign.py`](tests/test_sos_sync_redesign.py) with 18 comprehensive tests:
  - `TestRemoveFromBacklog` (5 tests): Task removal from BACKLOG.md
  - `TestAppendChangelogEnhanced` (3 tests): Enhanced changelog entries
  - `TestQueryDecisions` (5 tests): Parquet query interface
  - `TestGetRecentCompletions` (2 tests): Recent completions helper
  - `TestGetPendingTasks` (1 test): Pending tasks helper
  - `TestCompleteTaskWithArchive` (2 tests): Integration tests with archive_completed flag
- All 18 tests passing (100% coverage of SOS Sync redesign features)

**Architecture Decision**:

- **BACKLOG.md** = Pending tasks ONLY (clean, actionable list)
- **CHANGELOG.md** = Completed history with full details from parquet
- **decision_log.parquet** = Source of truth (23 fields, queryable)

---

## Phase 8 - 2026-04-08 10:13

**Task**: Documentation and migration guide

**Completed**:

- Updated [`docs/SPARRING_V2.md`](docs/SPARRING_V2.md) with comprehensive stage-based architecture documentation:
  - Added "2026-04-08 Stage-Based Refactoring" section with architecture overview
  - Documented all modified files (base_stage_executor.py, modes/, config.py, session_store.py)
  - Updated test coverage section with Stage Recovery test suite (23 tests)
  - Added detailed test coverage table for all 6 test classes
  - Added "Stage-Based Architecture Details" section with:
    - BudgetManager usage examples
    - CircuitBreaker state machine documentation
    - Stage weights configuration reference
    - BaseStageExecutor interface guide
    - Recovery workflow diagram
  - Updated files table with new architecture files
- Migration guide for existing users (backward-compatible API preserved)

---

## Phase 7 - 2026-04-08 10:11

**Task**: Integration testing - recovery from failed stage

**Completed**:

- Created `tests/test_sparring_stage_recovery.py` with 23 comprehensive tests:
  - `TestBudgetManager` (5 tests): Dynamic timeout allocation, remaining budget tracking, pro/flash mode weights
  - `TestCircuitBreaker` (5 tests): State transitions, failure threshold, recovery timeout, HALF_OPEN state
  - `TestStageDataclasses` (3 tests): StageResult success/failure, StageContext creation
  - `TestSessionStoreTTL` (5 tests): TTL expiration, checkpoint persistence, ephemeral TTL constant
  - `TestStageRecovery` (3 tests): Recovery from failed stage, checkpoint creation, budget tracking across stages
  - `TestConfigModule` (3 tests): Stage weights sum validation, budget config coverage, circuit breaker values
- All 23 tests passing (100% coverage of stage-based execution features)
- Test coverage includes: BudgetManager, CircuitBreaker, StageResult, StageContext, SessionCheckpoint TTL, BaseStageExecutor recovery

---

## Phase 6 - 2026-04-08 10:07

**Task**: Update config.py with BudgetManager defaults and STAGE_WEIGHTS

**Completed**:

- Updated `src/qwen_mcp/engines/sparring_v2/config.py` with:
  - `STAGE_WEIGHTS` dictionary - default weights for pro (discovery=0.15, red=0.28, blue=0.28, white=0.29), full (same as pro), and flash (analyst=0.45, drafter=0.55)
  - `BUDGET_CONFIG` - total timeout budgets (pro=225s, full=225s, flash=60s)
  - `CIRCUIT_BREAKER_CONFIG` - failure threshold (3) and recovery timeout (60s)
  - `EPHEMERAL_TTL` - 300 seconds TTL for flash mode checkpoints
- Centralized configuration for stage-based executors

---

## Phase 5 - 2026-04-08 10:05

**Task**: Update SessionStore for stage metadata and TTL support

**Completed**:

- Updated `SessionCheckpoint` dataclass in `src/qwen_mcp/engines/session_store.py`:
  - Added `has_stages` field (bool) - stage-based execution flag
  - Added `stage_count` field (int) - total number of stages
  - Added `ttl_expires_at` field (Optional[str]) - TTL expiration for ephemeral checkpoints
  - Added `is_expired()` method - checks if checkpoint has expired
  - Added `set_ttl(ttl_seconds)` method - sets TTL expiration
- Updated `load()` method to check TTL expiration and auto-delete expired checkpoints
- Fixed duplicate `__post_init__` method

---

## Phase 4 - 2026-04-08 10:03

**Task**: Add ephemeral TTL checkpointing to FlashExecutor

**Completed**:

- Refactored `src/qwen_mcp/engines/sparring_v2/modes/flash.py` to inherit from `BaseStageExecutor`
- Added ephemeral TTL checkpointing (300s TTL) for fast mode support
- Stage weights: analyst=0.45, drafter=0.55
- Total budget: 60s for flash mode
- Uses ephemeral session IDs (not persisted)

---

## Phase 3 - 2026-04-08 10:02

**Task**: Split FullExecutor from monolithic to stage-based

**Completed**:

- Refactored `src/qwen_mcp/engines/sparring_v2/modes/full.py` to inherit from `BaseStageExecutor`
- Split monolithic `execute()` method into 4 isolated stage executions via `execute_stage()`
- Re-enabled White Cell regeneration loop (`allow_regeneration=True`, `max_loops=2`)
- Added recovery support for resuming from failed stage
- Timeout budget: 225s with dynamic allocation (discovery=15%, red=28%, blue=28%, white=29%)

---

## Phase 2 - 2026-04-08 10:00

**Task**: Refactor ProExecutor to inherit BaseStageExecutor

**Completed**:

- Refactored `src/qwen_mcp/engines/sparring_v2/modes/pro.py` to inherit from `BaseStageExecutor`
- Implemented `get_stages()`, `execute_stage()`, `get_stage_weights()` methods
- Stage weights: discovery=0.15, red=0.28, blue=0.28, white=0.29
- Preserved backward-compatible API with existing `execute()` method
- Automatic checkpointing after each stage via `execute_with_recovery()`

---

## Phase 1 - 2026-04-08 09:59

**Task**: Create BaseStageExecutor with BudgetManager and CircuitBreaker

**Completed**:

- Created `src/qwen_mcp/engines/sparring_v2/base_stage_executor.py` with:
  - `BudgetManager`: Dynamic timeout allocation with remaining budget tracking
  - `CircuitBreaker`: Failure threshold (3 failures) with recovery timeout (60s)
  - `StageResult` and `StageContext` dataclasses
  - `BaseStageExecutor`: Abstract class with `execute_with_recovery()` method
- Core architecture for unified stage-based sparring execution

---

## Phase 9 - 2026-04-08 09:49

**Task**: Migrate decision_log.parquet to .PLAN/ directory

**Completed**:

- Created `.PLAN/` directory (hidden convention for project metadata)
- Copied all files from `PLAN/` to `.PLAN/` (BACKLOG.md, CHANGELOG.md, SOS_BLUEPRINT.md, sparring_engine.md)
- Moved `decision_log.parquet` from `src/` to `.PLAN/`
- Updated `DEFAULT_DECISION_LOG_PATH` in decision_log_sync.py to `.PLAN/decision_log.parquet`
- Updated backlog_path references from `PLAN/` to `.PLAN/`
- Added `.PLAN/decision_log.parquet` to .gitignore

---

## SOS Sync - 2026-04-08 09:39:41

## [2026-04-08 09:39:29] f033f227-bcfa-4ed9-b156-18cb421b9f98

**Task**: Phase 1: Create BaseStageExecutor with BudgetManager and CircuitBreaker

**Advice**: Create new file src/qwen_mcp/engines/sparring_v2/base_stage_executor.py with: BudgetManager (dynamic timeout allocation), CircuitBreaker (3 failures → 60s recovery), StageResult/StageContext dataclasses, BaseStageExecutor abstract class with execute_with_recovery() method. This is the core architecture extraction for unified stage-based sparring.

---

## [2026-04-08 09:39:29] 9257dd4b-774b-4daa-9eb4-bc9ea1652a84

**Task**: Phase 2: Refactor ProExecutor to inherit BaseStageExecutor

**Advice**: Update src/qwen_mcp/engines/sparring_v2/modes/pro.py to inherit from BaseStageExecutor instead of ModeExecutor. Implement get_stages(), execute_stage(), get_stage_weights() methods. Extract existing monolithic logic into stage-based execute_stage() for discovery→red→blue→white. Preserve backward-compatible API.

---

## [2026-04-08 09:39:30] d35d4531-e4ff-461e-bf16-42a916bf92b1

**Task**: Phase 3: Split FullExecutor from monolithic to stage-based

**Advice**: Refactor src/qwen_mcp/engines/sparring_v2/modes/full.py to inherit BaseStageExecutor. Split monolithic execute() method into 4 isolated stage executions. Re-enable White Cell loop (allow_regeneration=True, max_loops=2). Add recovery support for resuming from failed stage. Timeout budget: 225s with dynamic allocation.

---

## [2026-04-08 09:39:30] d99cff5a-97db-4ed4-8e8b-0d90da01e98c

**Task**: Phase 4: Add ephemeral TTL checkpointing to FlashExecutor

**Advice**: Update src/qwen_mcp/engines/sparring_v2/modes/flash.py to inherit BaseStageExecutor. Add ephemeral checkpointing with 300s TTL (5 minutes) for fast 2-step analyst→drafter mode. Override save_checkpoint() for TTL-based expiration. Lower overhead than full persistence.

---

## [2026-04-08 09:39:30] 56b22d05-baef-49b7-a37f-219ba4d154e6

**Task**: Phase 5: Update SessionStore for stage metadata and TTL support

**Advice**: Extend src/qwen_mcp/engines/session_store.py with: stage metadata fields (has_stages, stage_count), TTL expiration check for ephemeral checkpoints, get_stage_context() helper method. Ensure atomic writes preserve stage data integrity.

---

## [2026-04-08 09:39:30] ae14e2e8-7b50-410a-aa5c-bbce18bbc132

**Task**: Phase 6: Update config.py with BudgetManager defaults and STAGE_WEIGHTS

**Advice**: Add to src/qwen_mcp/engines/sparring_v2/config.py: STAGE_WEIGHTS dict (full: discovery=0.2, red=0.27, blue=0.27, white=0.26; pro: equal weights; flash: analyst=0.4, drafter=0.6), circuit_breaker_threshold=3, circuit_breaker_recovery=60, ephemeral_ttl=300.

---

## [2026-04-08 09:39:31] 9e848f9b-7507-4b6c-8233-b0a864e638c5

**Task**: Phase 7: Integration testing - recovery from failed stage

**Advice**: Create test suite for stage recovery: simulate timeout at red/blue/white stage, verify session can resume from failed stage, confirm checkpoint data integrity, test circuit breaker opens after 3 consecutive failures, verify BudgetManager correctly tracks remaining budget.

---

## [2026-04-08 09:39:31] 2fd5e7d6-df75-49b5-b7f5-4da0b02acefd

**Task**: Phase 8: Documentation and migration guide

**Advice**: Update docs/SPARRING_V2.md with: new stage-based architecture diagram, BudgetManager usage examples, CircuitBreaker configuration, recovery workflow documentation, migration guide for existing users, API compatibility notes.

---

## SOS Sync - 2026-04-08 01:15:19

## [2026-04-08 00:52:53] fecae7b9-5549-4a56-851c-28c5e800391e

**Task**: Integracja raportów użycia tokenów z dynamicznym statusem sesji w Backlogu

**Advice**: Implement token usage analytics integration with session status in BACKLOG.md - connect usage_analytics.duckdb with decision_log.parquet to show token consumption per epic/task

---

## [2026-04-08 00:52:54] a8b8d19c-cedd-4a24-bf6b-35e8780f77df

**Task**: Dashboard ROI - wizualizacja ile energii (tokenów) spaliły poszczególne kłody (Epiki)

**Advice**: Create ROI dashboard visualization showing token consumption per epic/task - use matplotlib or plotly for charts, integrate with decision_log.parquet for data source

---

## [2026-04-08 00:53:38] 50fb5241-6441-47a7-b5bd-73d5b1ce44ad

**Task**: Phase 6: Async Enrichment Pipeline - Non-blocking MCP integration

**Advice**: Implement ADREnrichmentPipeline with queue-based processing, LRU caching, and graceful degradation. Create async pipeline that enriches ADRs with code links, metrics, and cross-references without blocking MCP operations. Use asyncio.Queue for task management, functools.lru_cache for caching, and implement fallback mechanisms for network failures.

---

## [2026-04-08 01:15:19] 26c0d506-9aff-4f13-8e2d-90fc1335354f

**Task**: Implement multi-turn conversation support for qwen_sparring following TDD protocol (RED → GREEN → REFACTOR).

## Requirements:

1. **SessionStore Enhancement** (src/qwen_mcp/engines/session_store.py):

**Status**: ✅ Completed

---

## [2026-04-08 00:58:09] Phase 6: Async Enrichment Pipeline

**Task**: Implement ADREnrichmentPipeline with queue-based processing, LRU caching, and graceful degradation

**Changes**:

- Created [`LRUCache`](src/qwen_mcp/engines/adr_enrichment.py:16) - LRU cache with OrderedDict (max_size=100)
- Created [`ADREnrichmentPipeline`](src/qwen_mcp/engines/adr_enrichment.py:66) - async pipeline for non-blocking enrichment
- Implemented [`enqueue_decision()`](src/qwen_mcp/engines/adr_enrichment.py:88) - queue decisions for async processing
- Implemented [`process_queue_once()`](src/qwen_mcp/engines/adr_enrichment.py:102) - process single item from queue
- Implemented [`process_queue()`](src/qwen_mcp/engines/adr_enrichment.py:126) - background worker for continuous processing
- Implemented [`start_background_worker()`](src/qwen_mcp/engines/adr_enrichment.py:188) - start async worker task
- Implemented [`stop_background_worker()`](src/qwen_mcp/engines/adr_enrichment.py:202) - graceful shutdown
- Implemented graceful degradation when MCP client unavailable
- Created integration tests: [`tests/test_adr_enrichment_pipeline.py`](tests/test_adr_enrichment_pipeline.py:1) (15/15 tests passing)

**Technical Details**:
| Aspect | Implementation |
|--------|----------------|
| Cache | LRUCache with OrderedDict, max 100 items |
| Queue | asyncio.Queue for non-blocking operations |
| Background Worker | asyncio.Task with cancel support |
| Graceful Degradation | Works without MCP client (local enrichment) |
| Test Coverage | 15 tests across 4 test classes |

**Status**: ✅ Completed

---

## [2026-04-08 00:33:00] DecisionLog Auto-Logging Implementation

**Task**: Implement comprehensive DecisionLog auto-logging system with task_type discrimination

**Changes**:

- Added `task_type` field to [`DECISION_LOG_SCHEMA`](src/decision_log/decision_schema.py:12) (23 fields total)
- Renamed `SOSSyncEngine` → `DecisionLogSyncEngine` in [`decision_log_sync.py`](src/qwen_mcp/engines/decision_log_sync.py:16)
- Moved parquet location from `src/decision_log.parquet` to `.decision_log/decision_log.parquet`
- Implemented [`log_completed_task()`](src/qwen_mcp/engines/decision_log_sync.py:257) - logs completed work with task_type="completed"
- Implemented [`log_decision()`](src/qwen_mcp/engines/decision_log_sync.py:342) - logs architectural decisions with task_type="decision"
- Implemented [`complete_task()`](src/qwen_mcp/engines/decision_log_sync.py:484) - main entry point for qwen_coder auto-logging
- Implemented [`_find_matching_task()`](src/qwen_mcp/engines/decision_log_sync.py:532) - fuzzy keyword matching (30% threshold)
- Updated [`generate_code_unified()`](src/qwen_mcp/tools.py:365) to auto-invoke complete_task() after successful code generation
- Created integration tests: [`tests/test_decision_log_auto_logging.py`](tests/test_decision_log_auto_logging.py:1) (12/12 tests passing)
- Migration script: [`scripts/backfill_task_type.py`](scripts/backfill_task_type.py:1) (2 records backfilled)

**Technical Details**:
| Aspect | Implementation |
|--------|----------------|
| Schema | 23 fields (added task_type discriminator) |
| Task Types | "pending", "completed", "decision" |
| Fuzzy Matching | 30% keyword overlap threshold |
| Test Coverage | 12 tests across 5 test classes |
| Combined Suite | 18/18 tests passing |

**Status**: ✅ Completed

---

## [2026-04-08 00:27:06] a7e2f572-ad90-4823-88f9-dadc624f02b9

**Task**: Write a comprehensive integration test file `tests/test_decision_log_auto_logging.py` that tests the new DecisionLog auto-logging functionality.

The tests should cover:

1. `test_task_type_field_exist

**Status**: ✅ Completed

---

---

## [1.0.1] - 2026-04-07

### 📦 Production Release

**Documentation & README Updates:**

- Updated README.md with comprehensive tool tables including Context and SOS categories
- Added Context Tools section (qwen_init_context_tool, qwen_update_session_context_tool)
- Added SOS Sync documentation (qwen_add_task, qwen_sync_state)
- Added project structure diagram
- Updated Sparring Engine documentation with session storage details and guided UX
- Corrected Coder model reference to qwen3-coder-next

**SOS Sync Engine:**

- Automated BACKLOG.md and CHANGELOG.md synchronization with decision_log.parquet
- Atomic writes with file-based locking
- Auto-backlog integration from qwen_audit findings

**Repository Maintenance:**

- Added PLAN/ directory to .git/info/exclude
- Removed PLAN/ from Git cache (git rm --cached)

---

## 2026-04-06 - sparring3 (pro mode) Fix

**Task**: Naprawić tryb sparring3 (pro) - brak executora w MODE_EXECUTORS

**Changes**:

- Created [`ProExecutor`](src/qwen_mcp/engines/sparring_v2/modes/pro.py) - new executor for true step-by-step sparring3 execution
- Updated [`MODE_EXECUTORS`](src/qwen_mcp/engines/sparring_v2/engine.py:47) to use `ProExecutor` instead of `FullExecutor` for "pro" mode
- Updated [`qwen_sparring`](src/qwen_mcp/server.py:227) documentation to explicitly list all modes (sparring1, sparring2, sparring3)

**Technical Details**:
| Aspect | Before (FullExecutor) | After (ProExecutor) |
|--------|----------------------|---------------------|
| Execution | 4 cells in one call | 4 cells separately |
| Word limit | 100-200 words/cell | 800 words/cell |
| Token budget | 512-1024 tokens/cell | 512-4096 tokens/cell |
| Checkpointing | None | Between each cell |

**Result**: sparring3 now runs each cell (discovery, red, blue, white) as separate MCP calls with higher token budgets for deep analysis.

---

## SOS Sync - 2026-04-06 21:39:44

## [2026-04-04 22:57:48] 4c392b8c-a22c-4a37-9022-3a2e623d7f5c

**Task**: Naprawić backlog_ref w decision log

**Advice**: Automatically trigger qwen_sync_state after every significant file change to ensure project documentation is up to date.

---

## [2026-04-04 23:39:10] b91b301f-a319-43e2-a985-a12c58ae4406

**Task**: Naprawić backlog_ref w decision log

**Advice**: Restrict .lock file access to the current user only to prevent local privilege escalation during sync.

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
