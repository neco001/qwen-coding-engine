# Changelog

## [Full changelog](./.PLAN/CHANGELOG.md)

## 1.1.1 Release

**Date:** 2026-04-09

### Fixes

- **Sparring Session File Location**: Fixed agent not knowing where to find session results
  - Added session file path to `SparringResponse.to_markdown()` output
  - Agent now sees full path: `{storage_dir}/{session_id}.json`
  - Fixed file extension mismatch (.json vs .md)
  - Works with all storage directory resolution tiers (env, user-level, fallback)
  - Added 5 unit tests in `tests/test_sparring_session_path.py`

### Changes

- `src/qwen_mcp/engines/sparring_v2/models.py`: Added `storage_dir` parameter to `to_markdown()`
- `src/qwen_mcp/tools.py`: Pass session store directory to response formatter

---

## 1.1.0 Release

CHANGELOG

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
