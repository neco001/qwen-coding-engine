# CHANGELOG


## SOS Sync - 2026-04-20 20:46:58

- [x] FIX: Naprawa izolacji workspace w qwen_architect i tools.py - f6c0c826-5894-4c6c-8550-fbafd6c07d13

- [x] CLEANUP: Usunięcie obcych zadań z backlogu serwera - c3b302ef-f487-4909-8c09-078dfb2778b4


- [x] wiszące zadanie: Poprawię linię 1235, a potem zrefaktoryzuję \_apply_advices_to_files. Dodam metodę \_move_task_in_content(self, content, decision_id, backlog_ref=None), aby obsłużyć logikę przenoszenia zadania. Użyję qwen_coder tylko do refaktoryzacji \_apply_advices_to_files, zachowując jej obecną sygnaturę.


- [x] Integrate DecisionLogOrchestrator into decision_log_sync.py - e68b693c-17b7-4bfb-9c00-527008289d8f


- [x] Create models/task.py - e385bf52-9da7-4734-8758-053db85fac2a

- [x] Create io_layer/path_resolver.py - 343c4ba6-91f6-4b76-bee5-4e2a7cccef50

- [x] Create io_layer/file_handler.py - 2adfc111-10b3-47f9-8d9e-14ce66cea3db

- [x] Create markdown_layer/parser.py - 750be343-7fc7-448f-8389-12808db90775

- [x] Create markdown_layer/formatter.py - 6a6d30a5-4191-42cf-9f46-41d711064bc6

- [x] Create markdown_layer/sections.py - a230f8df-d222-4ab1-9ff8-07b657613d60

- [x] Create orchestrator.py - 5a1536af-5105-4307-9eb2-613b3c48de32

- [x] Implement archival logic - 32740fe0-369e-4306-bed7-199804e0234e

- [x] Update path resolution for CHANGELOG.md - 8bcd4fa6-5773-4085-ab51-3b6c60f0df3f

- [x] Update original decision_log_sync.py - d1d6e677-06aa-42e0-8a2f-ffc6331ea07c

- [x] Add unit tests for each layer - 27e280d4-839b-4b4e-a6c8-4361d8d55c86

- [x] Remove deprecated code paths - d73a6408-6bca-44f0-a636-6aae2326218e


- [x] Implement task archival in DecisionLogSyncEngine flows - bf19af91-cf26-4b9f-ba4f-9f7529c886ee

- [x] Standardize CHANGELOG.md path to root directory - 4d5ba433-f70a-454a-b817-07ac123630f2


- [x] Fix qwen_architect return type mismatch (dict vs str) - 9ac91137-99ec-416c-8a0c-c73cc781e4fe

- [x] Audit all tools for return type consistency - c26935c6-c6ec-44f9-b26e-f563d5ec83c6


- [x] Fix atomic_write leak in ContextBuilderEngine - ebe9d45a-596b-4dcb-abf5-c264d5d4247e

- [x] Cleanup existing .tmp files in .context - 6b11b08b-54a0-42ff-bd35-722ec92c710c

- [x] Verify context cleanup success - 17f72781-63a4-48c9-b897-77dfacae1959


- [x] Align qwen_architect brownfield mode with greenfield mode - remove direct code generation - 92d2e556-69f6-4a5e-855e-57ca2951f844
  - **SOLVED**: Modified LP_BROWNFIELD_PROMPT to remove direct diff generation instructions and add swarm task assignment
  - **SOLVED**: Modified generate_lp_blueprint to parse brownfield architect response into swarm_tasks structure
  - **VERIFIED**: All 6 enforcement tests pass, 0 regression alerts in baseline comparison


- [x] improvement of session_context format in accordance with [proposed template] (C:\Repos_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.PLAN\qwen-coding-project-context-template.md) - a572933b-639f-458b-bc5d-a27ae8129ed7


- [x] Fix Unbound fetcher TypeError and investigate MCP progress blockage - 78ddb567-3d7c-41c2-bf8f-f80caf099e60
  - **CLOSED**: "fetcher" nie istnieje w codebase - to był tytuł taska, nie błąd. Znaleziono i naprawiono brakujący `import time` w `billing.py` (używany w retry loop `_init_db`).


- [x] Create MODE_PROFILES configuration in config.py with flash/full/pro mode definitions - b2e39215-af1a-45e7-a4e1-6db370a5f4d3

- [x] Create get_mode_profile() helper function in config.py - 5e91a317-5018-4c6f-a7ed-20f2d73a6b6f

- [x] Create DynamicBudgetManager class extending BudgetManager - 5a71ef24-8c74-48e0-ab96-dfb0c4efd7a0

- [x] Create UnifiedSparringExecutor class in modes/unified.py - e210f485-ecd8-4501-9dde-4c9d31405c17

- [x] Add force_mode parameter to qwen_sparring tool signature - 6d6edfcd-8080-4056-8ea0-4ce7a01e6620

- [x] Implement mode routing logic with force_mode override - b28f3807-f156-4231-be58-99262f98c7e9

- [x] Create backward compatibility wrappers for Flash/Full/Pro executors - e6ca869d-1599-4363-8104-01b5ef1bb764

- [x] Integrate UnifiedSparringExecutor in engine.py - 919cc14f-23cd-422a-b08a-e06d9cfccc8c

- [x] Implement and test budget borrowing logic in DynamicBudgetManager - 428be552-5a5f-47f5-a5a8-4021a77d5d0e

- [x] Implement and test timeout extension for complex tasks - c833eeb2-e7da-44ad-9fd2-08c9b3a930df

- [x] Create end-to-end integration test for consolidated sparring - 61503428-3671-4da1-8dde-7c87d04be96e


- [x] Rename utils module to qwen_utils to avoid namespace conflicts - bd0cb9b5-6e17-42e8-84b8-5ea51c5a3502


- [x] dodaj task: architect podaje kod. czy powinien? - analyzed, detailed task created: 92d2e556-69f6-4a5e-855e-57ca2951f844

- [x] czy narzędzie `qwen_init_request` jest potrzebne - POTWIERDZONE: jest wywoływane automatycznie przez `_auto_init_request` wewnątrz narzędzi serwera. [2026-04-14]

- [x] Snapshot naming convention and auto-selection - f4b59636-2c52-4488-aa32-be6efd91245f


- [x] Pytanie dla architecta: LangGraph w projekcie? - 70f149bd-6d08-4256-8522-c98d13307073
  - **Answer**: LangGraph not needed - current architecture with specialized engines (CoderV2, SparringV2, Swarm) is sufficient for MCP server use case


- [x] tool `qwen_add_task` dodaje tylko jedno zadanie na raz - **SOLVED v1.2.0**
  - Implemented `qwen_add_tasks` batch tool in server.py (lines 431-504)
  - Added `add_tasks()` method in decision_log_sync.py (lines 364-504)
  - Added `add_tasks_to_backlog_batch()` in tools.py (lines 785-844)
  - Chunk-based processing (default: 20 tasks per chunk) to avoid MCP timeout
  - Full test coverage in tests/test_batch_tasks.py (8 tests passing)

---


- [x] **nie mamy narzędzia do tworzenia pliku decision_log.parquet** - Fallback implemented
  - `qwen_add_task`: Creates parquet if missing (lines 291-295), creates BACKLOG.md if missing (lines 315-317)
  - `_acquire_lock()`: Creates parent directories with `mkdir(parents=True, exist_ok=True)` (lines 50-51)
  - Both tools now handle missing files gracefully in new projects

- [x] **Krok 1: Izolacja Strumieni Loggingu** - stdin=subprocess.DEVNULL added to all subprocess calls (e0e88da0-0780-4f10-afdd-3e58be47c9cc)
  - **Root Cause Found**: Git subprocesses were stealing bytes from MCP's JSON-RPC stdin stream
  - Fixed in: `snapshot.py`, `git_diff_parser.py`, `adr_linker.py`


- [x] **Krok 2: Ograniczenie Współbieżności I/O** - chunk_size=20 and asyncio.sleep(0.01) already implemented (d71c5bb6-5ce1-4e22-b1ce-c0875f80fc3f)
  - Already had chunking and yield mechanism in place


- [x] **Krok 3: Optymalizacja Payloadu** - Already returns only path string, not full snapshot (279b0792-d84c-46d0-9302-d3a71a09200b)
  - Verified: `diff_audit.py:209` returns `str(path)` only


- [x] **Krok 4: Weryfikacja Krzyżowa Klienta** - Tested with alternative MCP client, confirmed fix works (a85af225-c2c0-4341-a088-b7eebb839d1a)
  - Verified: MCP timeout issue resolved


- [x] **Krok 5: Instrumentacja** - Added stderr logging to snapshot.py (aba4f3aa-260f-49a1-9709-834ed08bc060)
  - Added `logging` module import and `logger` instance
  - Added debug/info/warning/error logs at key points
  - Stderr is safe for MCP stdio transport (stdin isolation already in place)


- [x] **Wdrożenie stabilnego resolwowania ścieżek** - e46c7037-73da-410e-ad13-d3517631c208
  - Implemented in `src/qwen_mcp/config/sos_paths.py:resolve_workspace_root()`
  - Searches upward for project markers (.git, pyproject.toml)
  - Returns resolved path or raises error if no markers found


- [x] **Fallback dla braku pliku decision_log.parquet** - 3c84b2d4-d0df-40ff-b91e-6001a1674340
  - Implemented in `src/qwen_mcp/engines/decision_log_sync.py:add_task()` lines 291-295
  - Creates new file with empty records list if parquet doesn't exist
  - Directory creation handled in `_acquire_lock()` with `mkdir(parents=True, exist_ok=True)`

---


- [x] Fix snapshot storage location to use .anti_degradation/snapshots from config - 096f6e58-d2a0-458f-bde5-49c007cf5091


- [x] T12: Test optimized snapshot capture performance - 87e79cef-c7f1-4dc6-b0a9-a280a1fe6a93


- [x] T11: Update qwen_diff_audit to pass changed files to capture_snapshot - 00d85a74-48b1-4ea7-9f35-3861ec1dd24d


- [x] T10: Optimize \_generate_content_hashes() to only hash changed files - 475ef705-4a2c-4ca7-b205-c4095d233cce


- [x] T9: Add parallel processing with asyncio.gather for file snapshots - fcd7e99c-4160-47c8-bd66-84cb3a10e869


- [x] MCP Task Management Tools: qwen_list_tasks, qwen_get_task, qwen_update_task - 00bee066-f4a4-4530-b731-c91b8813b4fe


- [x] T7: Production Blocking Activation - 6d3a5a83-321f-4ed8-aef0-4cd249c46f73


- [x] T6: CI Workflow Integration - 8e4f115b-88f6-4ff7-9986-7a032580d3a9


- [x] T5: Shadow Mode Configuration - 343b1a3f-520f-4c8c-a786-1d30ff8da66e


- [x] T4: Pre-Commit Hook Script - 73421104-421d-4549-95a3-c372c8cb3b8f


- [x] T3: qwen_diff_audit MCP Tool - d95d55d3-b124-48e9-a89a-13ea050c66f3


- [x] T2: Git Diff Parser - d1d59443-1959-4695-841d-75bd907d7dfd


- [x] T1: Content Hashing w Snapshotach - 9ab88b43-1995-4bf2-ab47-5c56192002ad


- [x] Synchronizacja BACKLOG.md - dodać wszystkie pending z decision_log.parquet - 87722483-5a9b-4ebe-b331-c23fd41bec81


- [x] Fix syntax error in tasks_to_add.append dictionary - 9eca7c9b-95f6-49de-a56b-0223e8d99d60

- [x] Fix variable scope bug for swarm_tasks - 7d32100a-dd2f-499f-8e69-43fc11ccdf81

- [x] Add exception handling for auto_add_tasks block - 88cc565f-2374-4999-95d8-e8b7215b4814

- [x] Fix workspace URI validation - ed1f6e61-a343-4506-8692-9163c7cc5b16

- [x] Fix naive datetime in DecisionLogSyncEngine.add_tasks - d4f15717-1887-4f1b-a5d5-21af27a3fb79

### Qwen-Coding Enforcement (Layer 2: MCP Tool Validation) - COMPLETED


- [x] Create qwen_init_request() utility function for telemetry reset - 13

- [x] Modify qwen_architect to add auto_add_tasks parameter and auto-add tasks to backlog - 14

- [x] Modify qwen_architect to add workspace_root parameter - 15

- [x] Modify qwen_coder to add require_plan parameter for pre-flight check - 16

- [x] Modify qwen_coder to add require_test parameter for TDD enforcement - 17

- [x] Update MCP tool signatures in server.py for new parameters - 18

- [x] Create tests/test_tools_enforcement.py with enforcement test cases - 19

- [x] Update docs/TDD.md with enforcement layer documentation - 20

- [x] Update docs/ARCHITECTURE.md with enforcement diagram - 21


- [x] Fix session_id generation timing in UnifiedSparringExecutor.execute() - c58b514c-f752-448e-b2cb-4e8999e1a6c9

- [x] Update DiscoveryExecutor to use existing session_id if provided - e1ec0266-e776-4940-ba8b-de004d33f543

- [x] Fix FullExecutor import chain in engine.py - ed006198-b4d9-4506-bbc0-262864a873cd

- [x] Test session_id propagation in sparring2 full mode - eac11924-ecad-4b8c-8908-c7f5ccd86ec2

- [x] Evaluate and clean up duplicate FullExecutor implementations - 06de5934-2701-4333-afd8-7b6c96fef1a7


- [x] Modify FullExecutor.execute() to accept session_id parameter - f6bc41eb-b733-4c05-828b-0701030854bb

- [x] Implement session detection logic in FullExecutor - 997af485-2dc2-4c72-b54a-3ef941b1c4ec

- [x] Update FullExecutor to run only NEXT stage when session exists - 41480973-3ec6-461d-a6b8-791b97631f37

- [x] Update word limits to use WORD*LIMITS[full*\*] for each stage - ddfbb725-9f3d-48fd-ad83-cf3059aaee19

- [x] Update next_command to guide user through sparring2 step-by-step flow - 897d5d5c-f868-4508-96d7-9c2ff2cfabd9

- [x] Test sparring2 in step-by-step mode - 5ddd9804-27a5-4061-8d1c-138af391bc63

- [x] Update documentation to explain sparring2 vs sparring3 differences - aed96848-a57a-4a65-ae39-21b232e91e26


- [x] Refaktor sparring2 do działania krok-po-kroku (jak sparring3) - 0061563a-4208-464f-9819-da68aee8f9a0


- [x] Fix 5: sparring3 now truly step-by-step (one stage per MCP call) - 7b2e48d0-32b0-49d9-b41b-8476a60c5d6e

- [x] Revert: removed word limit enforcement (obcinanie = bad) - f9989739-195b-438a-a22d-217498737739

- [x] Doc fix: clarify FullExecutor vs ProExecutor behavior - 6ae8b900-ec7b-4887-9945-ff45e2939dbb


- [x] Fix 1: Respect explicit sparring mode choice (was auto-overriding) - bc2cc0d3-ce1c-41cb-a91d-f3cbf390b39e

- [x] Fix 2: Hard word limits + enforcement (timeout doesn't control response length) - 92e266b3-181a-456e-a2e7-f2adcb4492bc

- [x] Fix 3: Reduce timeouts to fit within 300s MCP limit - bab822e5-e051-4a6e-aa9d-5a8bc5051d85

- [x] Fix 4: JSON serialization fallback for method objects - beb69f20-f21d-494a-8a0e-19dc1d3b37a8


---
## 2026-04-14 23:05 - b70ed64e-726e-4d16-bacd-b35458859c0e

**Task**: Implement the following changes to DecisionLogSyncEngine and SOSPathsConfig:

1. In `src/qwen_mcp/config/sos_paths.py`:
   - Modify `get_changelog_path` to return `base / self.changelog_filename` (rem

**Status**: ✅ Completed

---


## SOS Sync - 2026-04-14 22:47:25

## [2026-04-14 16:41:58] a572933b-639f-458b-bc5d-a27ae8129ed7

**Task**: improvement of session_context format in accordance with proposed template

**Advice**: Updated SESSION_SUPPLEMENT_SYSTEM_PROMPT with 12 required sections from template, added workspace_root parameter to _format_session_summary_fallback, implemented full markdown structure matching template at .PLAN/qwen-coding-project-context-template.md

---

## [2026-04-14 16:56:29] 92d2e556-69f6-4a5e-855e-57ca2951f844

**Task**: Align qwen_architect brownfield mode with greenfield mode - remove direct code generation

**Advice**: Modify LP_BROWNFIELD_PROMPT in src/qwen_mcp/prompts/lachman.py to remove direct diff generation instructions and add swarm task assignment. Update generate_lp_blueprint in src/qwen_mcp/tools.py to ensure brownfield mode returns swarm_tasks structure similar to greenfield, not raw diffs. This ensures consistency across both modes and delegates code generation to qwen_coder.

---

## [2026-04-14 20:33:13] ebe9d45a-596b-4dcb-abf5-c264d5d4247e

**Task**: Fix atomic_write leak in ContextBuilderEngine

**Advice**: Remove lines 499-502 in context_builder.py. This redundant mkstemp() call is creating leaked .tmp files that are never used or unlinked.

---

## [2026-04-14 20:33:13] 6b11b08b-54a0-42ff-bd35-722ec92c710c

**Task**: Cleanup existing .tmp files in .context

**Advice**: Run a python one-liner or simple command to delete all *.tmp files inside the .context directory to restore order.

---

## [2026-04-14 20:33:13] 17f72781-63a4-48c9-b897-77dfacae1959

**Task**: Verify context cleanup success

**Advice**: Execute qwen_update_session_context and verify using list_dir that no new .tmp files are left behind.

---

## [2026-04-14 20:35:41] 9ac91137-99ec-416c-8a0c-c73cc781e4fe

**Task**: Fix qwen_architect return type mismatch (dict vs str)

**Advice**: Change brownfield return in generate_lp_blueprint to return string if called via MCP, or modify server.py wrapper to format the dict response into a readable markdown string before returning.

---

## [2026-04-14 20:35:41] c26935c6-c6ec-44f9-b26e-f563d5ec83c6

**Task**: Audit all tools for return type consistency

**Advice**: Ensure all MCP tools return str by default or properly handle Union[str, dict] if the client supports it. Currently, FastMCP Pydantic validation is strict.

---

## [2026-04-14 20:41:03] bf19af91-cf26-4b9f-ba4f-9f7529c886ee

**Task**: Implement task archival in DecisionLogSyncEngine flows

**Advice**: Refactor _apply_advices_to_files to use _mark_task_completed or extract a shared _archive_task utility to ensure tasks are moved to the ## Completed section during sync, not just toggled in-place.

---

## [2026-04-14 20:41:03] 4d5ba433-f70a-454a-b817-07ac123630f2

**Task**: Standardize CHANGELOG.md path to root directory

**Advice**: Update DEFAULT_SOS_PATHS or DecisionLogSyncEngine constants to point CHANGELOG.md to project root by default instead of .PLAN/ subdirectory.

---

## 2026-04-14 22:47 - 7a3adb23-85ba-4d5e-81c6-5fdd9cf7437e

**Task**: Fix the `qwen_architect` tool in `src/qwen_mcp/server.py` (lines 192-239) to handle the case where `generate_lp_blueprint` returns a dict instead of a string (Brownfield mode). The fix should be appli

**Status**: ✅ Completed

---


## SOS Sync - 2026-04-14 07:54:27

## [2026-04-14 00:28:34] 9eca7c9b-95f6-49de-a56b-0223e8d99d60

**Task**: Fix syntax error in tasks_to_add.append dictionary

**Advice**: Add missing commas in dictionary construction at src/qwen_mcp/tools.py:220. Format with proper line breaks and commas for key-value pairs.

---

## [2026-04-14 00:28:34] 7d32100a-dd2f-499f-8e69-43fc11ccdf81

**Task**: Fix variable scope bug for swarm_tasks

**Advice**: Initialize swarm_tasks = [] before the conditional block at line 180 in src/qwen_mcp/tools.py to prevent NameError when auto_add_tasks=True but blueprint has no swarm_tasks.

---

## [2026-04-14 00:28:34] 88cc565f-2374-4999-95d8-e8b7215b4814

**Task**: Add exception handling for auto_add_tasks block

**Advice**: Wrap the auto_add_tasks block (lines 198-225) in src/qwen_mcp/tools.py with try-except to log failures gracefully when DecisionLogSyncEngine.add_tasks() fails.

---

## [2026-04-14 00:28:34] ed1f6e61-a343-4506-8692-9163c7cc5b16

**Task**: Fix workspace URI validation

**Advice**: Replace hardcoded workspace_uri.startswith('file:///') with urllib.parse.urlparse for robust URI handling in src/qwen_mcp/tools.py:208.

---

## [2026-04-14 00:28:34] d4f15717-1887-4f1b-a5d5-21af27a3fb79

**Task**: Fix naive datetime in DecisionLogSyncEngine.add_tasks

**Advice**: Replace datetime.now() with datetime.now(timezone.utc) at src/qwen_mcp/engines/decision_log_sync.py:432 for timezone consistency.

---

## 2026-04-14 00:10 - 65f231f4-f453-48bf-9cc0-f030a2c57963

**Task**: Write a pytest test for the generate_lp_blueprint function in src/qwen_mcp/tools.py that verifies the auto_add_tasks parameter works. The test should: 1. Mock DecisionLogSyncEngine.add_task with Async

**Status**: ✅ Completed

---


## SOS Sync - 2026-04-13 23:40:07

## [2026-04-13 22:55:20] c58b514c-f752-448e-b2cb-4e8999e1a6c9

**Task**: Fix session_id generation timing in UnifiedSparringExecutor.execute()

**Advice**: Remove session_id generation from unified.py:299-300. Set session_id=None initially in StageContext. After discovery stage completes, update session_id from stage_result (pattern from full.py:241-242). This ensures all stages use the same session_id created by discovery.

---

## [2026-04-13 22:55:20] e1ec0266-e776-4940-ba8b-de004d33f543

**Task**: Update DiscoveryExecutor to use existing session_id if provided

**Advice**: Update discovery.py to accept and use existing session_id from StageContext if provided (not None). Only generate new session_id if session_id is None. This preserves session continuity when called from unified executor.

---

## [2026-04-13 22:55:20] ed006198-b4d9-4506-bbc0-262864a873cd

**Task**: Fix FullExecutor import chain in engine.py

**Advice**: Change engine.py line 33 to import FullExecutor from full.py instead of backward_compat.py. Alternatively, remove backward_compat.py.FullExecutor and have it delegate to full.py.FullExecutor. This ensures the correct implementation is used.

---

## [2026-04-13 22:55:20] eac11924-ecad-4b8c-8908-c7f5ccd86ec2

**Task**: Test session_id propagation in sparring2 full mode

**Advice**: Run sparring2 test with topic that triggers full mode. Verify in logs that all 4 stages (discovery, red, blue, white) use the SAME session_id. Expected: session_id created by discovery is used by all subsequent stages.

---

## [2026-04-13 22:55:20] 06de5934-2701-4333-afd8-7b6c96fef1a7

**Task**: Evaluate and clean up duplicate FullExecutor implementations

**Advice**: After fixing unified.py, evaluate if full.py is still needed. Consider removing full.py if unified.py handles all cases correctly. Alternatively, keep full.py as reference implementation and remove backward_compat.py wrapper entirely.

---

## 2026-04-13 23:40 - 01389237-8228-421f-874b-2e338c001d77

**Task**: Write a pytest test for the qwen_init_request utility function that verifies: 1. Telemetry counters are reset via get_broadcaster().broadcast_state with operation 'request_start'; 2. The function retu

**Status**: ✅ Completed

---


## SOS Sync - 2026-04-13 20:03:58

## [2026-04-13 19:54:52] 0061563a-4208-464f-9819-da68aee8f9a0

**Task**: Refaktor sparring2 do działania krok-po-kroku (jak sparring3)

**Advice**: FullExecutor obecnie uruchamia wszystkie 4 etapy (discovery→red→blue→white) w JEDNYM wywołaniu MCP, co ryzykuje timeout 300s. Refaktoryzacja polega na:

1. **SparringEngineV2.execute()** (engine.py:130): Dodać session_id do wywołania FullExecutor
2. **FullExecutor.execute()** (full.py:158): 
   - Przyjmować session_id jako parametr
   - Sprawdzać czy sesja istnieje (kontynuacja) czy to nowa sesja
   - Uruchamiać TYLKO następny etap, nie wszystkie naraz
   - Zwracać wynik etapu + next_command
3. **Nowa metoda _execute_single_stage()**: Wykonuje pojedynczy etap z odpowiednim word_limit z WORD_LIMITS["full_*"]
4. **Zachować różnice**: sparring2 ma word_limits 100-200 słów, sparring3 ma 800 słów

Korzyści: eliminacja timeout, lepsza kontrola, możliwość przerwania po dowolnym etapie, spójność z sparring3.

Pliki do zmiany:
- src/qwen_mcp/engines/sparring_v2/engine.py (linia 130)
- src/qwen_mcp/engines/sparring_v2/modes/full.py (cała klasa FullExecutor)
- src/qwen_mcp/prompts/sparring.py (WORD_LIMITS - już istnieją)

---

## [2026-04-13 20:00:12] 3d4bfd01-183d-4444-b737-186ad8d7c064

**Task**: Update SparringEngineV2 to pass session_id to FullExecutor

**Advice**: W pliku [`engine.py:130`](src/qwen_mcp/engines/sparring_v2/engine.py:130) zmienić wywołanie FullExecutor.execute() aby przekazywało session_id. Obecnie: `await executor.execute(topic=topic, context=context, ctx=ctx)`. Zmiana na: `await executor.execute(topic=topic, context=context, ctx=ctx, session_id=session_id)`. To umożliwia FullExecutor wykrycie czy to kontynuacja istniejącej sesji.

---

## [2026-04-13 20:00:12] f6bc41eb-b733-4c05-828b-0701030854bb

**Task**: Modify FullExecutor.execute() to accept session_id parameter

**Advice**: W pliku [`full.py:158`](src/qwen_mcp/engines/sparring_v2/modes/full.py:158) dodać parametr `session_id: Optional[str] = None` do sygnatury metody execute(). To pierwszy krok do refaktoryzacji - pozwala na wykrycie czy user wywołuje sparring2 z session_id (kontynuacja) czy bez (nowa sesja).

---

## [2026-04-13 20:00:12] 997af485-2dc2-4c72-b54a-3ef941b1c4ec

**Task**: Implement session detection logic in FullExecutor

**Advice**: W FullExecutor.execute() dodać logikę sprawdzającą czy session_id istnieje: `if session_id: existing_session = self.session_store.load(session_id)`. Jeśli sesja istnieje - to kontynuacja, jeśli nie - to nowa sesja. Użyć tej logiki do decyzji czy uruchomić discovery (nowa sesja) czy następny etap (kontynuacja).

---

## [2026-04-13 20:00:12] 41480973-3ec6-461d-a6b8-791b97631f37

**Task**: Update FullExecutor to run only NEXT stage when session exists

**Advice**: Zamiast uruchamiać wszystkie etapy naraz (discovery→red→blue→white), FullExecutor ma uruchamiać TYLKO jeden etap: 1) Dla nowej sesji: discovery, 2) Dla istniejącej sesji: next_stage = _get_next_stage(completed_stages). Po wykonaniu etapu zwrócić SparringResponse z next_command do kolejnego wywołania.

---

## [2026-04-13 20:00:12] ddfbb725-9f3d-48fd-ad83-cf3059aaee19

**Task**: Update word limits to use WORD_LIMITS[full_*] for each stage

**Advice**: W metodzie _execute_single_stage() używać word_limits z WORD_LIMITS dictionary: discovery=WORD_LIMITS["full_discovery"] (100), red=WORD_LIMITS["full_red"] (150), blue=WORD_LIMITS["full_blue"] (150), white=WORD_LIMITS["full_white"] (200). To zapewnia zwięzłe odpowiedzi w sparring2 w przeciwieństwie do sparring3 (800 słów).

---

## [2026-04-13 20:00:12] 897d5d5c-f868-4508-96d7-9c2ff2cfabd9

**Task**: Update next_command to guide user through sparring2 step-by-step flow

**Advice**: Każdy etap sparring2 ma zwracać next_command w formacie: `qwen_sparring(mode='full', session_id='{session_id}')`. Discovery zwraca 'next: red', red zwraca 'next: blue', blue zwraca 'next: white', white zwraca None (koniec). User widzi postęp i wie które wywołanie wykonać dalej.

---

## [2026-04-13 20:00:12] 5ddd9804-27a5-4061-8d1c-138af391bc63

**Task**: Test sparring2 in step-by-step mode

**Advice**: Przetestować refaktoryzowany sparring2: 1) Uruchomić qwen_sparring(mode='full', topic='...') - powinno zwrócić discovery + next: red, 2) Uruchomić qwen_sparring(mode='full', session_id='...') - powinno zwrócić red + next: blue, 3) Kontynuować do white, 4) Sprawdzić czy wszystkie etapy zachowują word_limits (100-200 słów), 5) Sprawdzić czy session_store poprawnie zapisuje/odczytuje sesję.

---

## [2026-04-13 20:00:12] aed96848-a57a-4a65-ae39-21b232e91e26

**Task**: Update documentation to explain sparring2 vs sparring3 differences

**Advice**: Zaktualizować dokumentację (docs/SPARRING_V2.md lub README.md) aby wyjaśnić różnice: sparring2 (full) = krok-po-kroku z word_limits 100-200 słów, ~56s timeout na etap; sparring3 (pro) = krok-po-kroku z word_limit 800 słów, 300s timeout na etap. Oba tryby działają krok-po-kroku, ale sparring3 pozwala na głębszą analizę.

---

## 2026-04-13 20:03 - e202c862-5852-4120-b918-d3acc0835fd4

**Original Task**: Update SparringEngineV2 to pass session_id to FullExecutor  
**Decision ID**: `3d4bfd01-183d-4444-b737-186ad8d7c064` → `e202c862-5852-4120-b918-d3acc0835fd4`

**Status**: ✅ Completed

---


## SOS Sync - 2026-04-13 18:35:24

## [2026-04-13 18:01:20] bc2cc0d3-ce1c-41cb-a91d-f3cbf390b39e

**Task**: Fix 1: Respect explicit sparring mode choice (was auto-overriding)

**Advice**: Fixed auto-detection overriding user's explicit sparring mode choice. Changed logic to respect sparring1/sparring2/sparring3 when explicitly provided, only auto-detecting when no mode specified.

---

## [2026-04-13 18:01:20] 92e266b3-181a-456e-a2e7-f2adcb4492bc

**Task**: Fix 2: Hard word limits + enforcement (timeout doesn't control response length)

**Advice**: Updated get_word_limit_instruction to be a HARD constraint with explicit consequences. Added enforce_word_limit method to ContentValidator that truncates responses exceeding limits. Applied enforcement to red_cell, blue_cell, and white_cell executors after API calls.

---

## [2026-04-13 18:01:20] bab822e5-e051-4a6e-aa9d-5a8bc5051d85

**Task**: Fix 3: Reduce timeouts to fit within 300s MCP limit

**Advice**: Reduced sparring timeouts from 400s total to 165s total: discovery=30s, red=45s, blue=45s, white=45s. With hard word limits, responses complete much faster.

---

## [2026-04-13 18:01:20] beb69f20-f21d-494a-8a0e-19dc1d3b37a8

**Task**: Fix 4: JSON serialization fallback for method objects

**Advice**: Added custom _json_serializable_default handlers in session_store.py and _telemetry_json_default in telemetry.py to gracefully handle method objects that accidentally leak into checkpoint data.

---

## [2026-04-13 18:21:13] 7b2e48d0-32b0-49d9-b41b-8476a60c5d6e

**Task**: Fix 5: sparring3 now truly step-by-step (one stage per MCP call)

**Advice**: Rewrote ProExecutor.execute() to run ONE STAGE PER CALL instead of all 4 stages. This matches your design: each MCP call gets its own 300s timeout. User calls sparring3 repeatedly to progress through discovery→red→blue→white.

---

## [2026-04-13 18:21:13] f9989739-195b-438a-a22d-217498737739

**Task**: Revert: removed word limit enforcement (obcinanie = bad)

**Advice**: Removed enforce_word_limit from sanitizer.py and reverted get_word_limit_instruction. Word limit obcinanie was bad - it produced incomplete responses.

---

## [2026-04-13 18:21:13] 6ae8b900-ec7b-4887-9945-ff45e2939dbb

**Task**: Doc fix: clarify FullExecutor vs ProExecutor behavior

**Advice**: Updated FullExecutor and ProExecutor docstrings to clearly state: FullExecutor runs ALL stages in one call (risk timeout!), ProExecutor runs ONE stage per call (safe).

---

## 2026-04-13 18:35 - a2fa04f4-e583-428a-b73d-24c242ce51d2

**Task**: Generate a complex git commit message following the project standards with type, scope, subject, body, and core changes. The commit should cover the changes made to implement the UnifiedSparringExecut

**Status**: ✅ Completed

---


## 2026-04-13 11:28 - 7eefc18f-7a38-4800-8b82-a6cc673d69b4

**Task**: Generate code to convert a SparringResponse object to a SessionCheckpoint object for use with session_store.save(). Use the following classes: SparringResponse (from src/qwen_mcp/engines/sparring_v2/m

**Status**: ✅ Completed

---


## 2026-04-13 10:52 - 9d4d505e-254a-4898-a801-652883d7838d

**Task**: Modify the execute method in src/qwen_mcp/engines/sparring_v2/modes/unified.py (lines 237-280) to return SparringResponse. Use existing code skeleton and follow these exact steps:

Current execute met

**Status**: ✅ Completed

---


## 2026-04-13 10:11 - 82baf86e-10e6-4a3f-85d2-5c2b6ef36fcc

**Task**: Read the file src/qwen_mcp/engines/sparring_v2/base_stage_executor.py and add a new class DynamicBudgetManager that extends BudgetManager. The class should have these features:
1. Constructor with all

**Status**: ✅ Completed

---


## 2026-04-13 10:10 - cc1ace20-5e9d-4aae-a453-24b5c058fdfb

**Task**: Write a simple Python function that adds two numbers: def add(a, b): return a + b

**Status**: ✅ Completed

---


## 2026-04-13 10:08 - d09bc5d9-d6b6-43e8-be9f-69c472daeef9

**Task**: Implement Task 3: Create DynamicBudgetManager class

Read the current file src/qwen_mcp/engines/sparring_v2/base_stage_executor.py and add a DynamicBudgetManager class that extends BudgetManager with 

**Status**: ✅ Completed

---


## SOS Sync - 2026-04-13 10:06:13

## [2026-04-12 23:22:55] bd0cb9b5-6e17-42e8-84b8-5ea51c5a3502

**Task**: Rename utils module to qwen_utils to avoid namespace conflicts

**Advice**: The generic 'utils' module name conflicts with project-specific utils packages when qwen-coding runs from other project directories. Rename src/utils/ to src/qwen_utils/ and update all imports and pyproject.toml configuration.

---

## [2026-04-13 10:04:30] b2e39215-af1a-45e7-a4e1-6db370a5f4d3

**Task**: Create MODE_PROFILES configuration in config.py with flash/full/pro mode definitions

**Advice**: Add MODE_PROFILES dictionary to src/qwen_mcp/engines/sparring_v2/config.py with configurations for flash, full, and pro modes. Each profile should define: stages, total_budget, stage_weights, word_limits, thinking_tokens, timeout_config. Test file: tests/test_sparring_config.py

---

## [2026-04-13 10:04:30] 5e91a317-5018-4c6f-a7ed-20f2d73a6b6f

**Task**: Create get_mode_profile() helper function in config.py

**Advice**: Add get_mode_profile(mode: str) -> ModeProfile function to src/qwen_mcp/engines/sparring_v2/config.py that validates mode and returns profile from MODE_PROFILES. Raise ValueError for invalid modes. Test in tests/test_sparring_config.py

---

## [2026-04-13 10:04:30] 5a71ef24-8c74-48e0-ab96-dfb0c4efd7a0

**Task**: Create DynamicBudgetManager class extending BudgetManager

**Advice**: Create DynamicBudgetManager in src/qwen_mcp/engines/sparring_v2/base_stage_executor.py with support for: time borrowing across stages (allow_borrow: bool), timeout extension for complex tasks (extend_timeout_pct: float), and complexity-based budget adjustment. Test budget borrowing and timeout extension logic.

---

## [2026-04-13 10:04:30] e210f485-ecd8-4501-9dde-4c9d31405c17

**Task**: Create UnifiedSparringExecutor class in modes/unified.py

**Advice**: Create new file src/qwen_mcp/engines/sparring_v2/modes/unified.py with UnifiedSparringExecutor class. This class inherits from BaseStageExecutor and uses MODE_PROFILES for configuration. It should support all three modes (flash/full/pro) through a single implementation with mode-specific settings from profiles.

---

## [2026-04-13 10:04:30] 6d6edfcd-8080-4056-8ea0-4ce7a01e6620

**Task**: Add force_mode parameter to qwen_sparring tool signature

**Advice**: Update src/qwen_mcp/tools.py generate_sparring() function to add optional force_mode: Optional[str] = None parameter. This allows users to bypass auto-routing and force specific mode. Update MCP tool definition in server.py to expose this parameter.

---

## [2026-04-13 10:04:30] b28f3807-f156-4231-be58-99262f98c7e9

**Task**: Implement mode routing logic with force_mode override

**Advice**: Add _resolve_sparring_mode_with_override() helper in src/qwen_mcp/tools.py that checks force_mode first, then uses existing resolve_sparring_mode(). Update generate_sparring() to use this routing. Validate that forced mode is valid before execution.

---

## [2026-04-13 10:04:30] e6ca869d-1599-4363-8104-01b5ef1bb764

**Task**: Create backward compatibility wrappers for Flash/Full/Pro executors

**Advice**: In src/qwen_mcp/engines/sparring_v2/modes/unified.py, create FlashExecutor, FullExecutor, ProExecutor classes that delegate to UnifiedSparringExecutor with appropriate mode='flash'|'full'|'pro'. This maintains backward compatibility while using unified engine.

---

## [2026-04-13 10:04:30] 919cc14f-23cd-422a-b08a-e06d9cfccc8c

**Task**: Integrate UnifiedSparringExecutor in engine.py

**Advice**: Update src/qwen_mcp/engines/sparring_v2/engine.py to import and use UnifiedSparringExecutor instead of separate mode executors. Modify SparringEngineV2.execute() to instantiate UnifiedSparringExecutor with resolved mode parameter.

---

## [2026-04-13 10:04:30] 428be552-5a5f-47f5-a5a8-4021a77d5d0e

**Task**: Implement and test budget borrowing logic in DynamicBudgetManager

**Advice**: Add borrow_time_from_previous_stages() and extend_current_stage_timeout() methods to DynamicBudgetManager. Test that fast stages can lend unused budget to slower stages. Create tests in tests/test_sparring_budget.py.

---

## [2026-04-13 10:04:30] c833eeb2-e7da-44ad-9fd2-08c9b3a930df

**Task**: Implement and test timeout extension for complex tasks

**Advice**: Add extend_for_complex_task() method to DynamicBudgetManager that increases timeout by configured percentage (default 50%) when complexity indicator is detected. Test that timeout extension works without exceeding total budget limits.

---

## [2026-04-13 10:04:30] 61503428-3671-4da1-8dde-7c87d04be96e

**Task**: Create end-to-end integration test for consolidated sparring

**Advice**: Create tests/test_sparring_consolidation.py with e2e tests for all three modes using UnifiedSparringExecutor. Test: flash mode completes quickly, full mode with dynamic budgeting, pro mode with separate budgets, force_mode override bypasses auto-routing. Validate backward compatibility with old executor names.

---

## 2026-04-13 10:06 - 910f296f-7745-4e2d-ae94-0cb164ba119a

**Task**: Implement Task 1: Create MODE_PROFILES configuration in config.py

Read the current file src/qwen_mcp/engines/sparring_v2/config.py and add a MODE_PROFILES dictionary after the existing configuration 

**Status**: ✅ Completed

---


## 2026-04-12 22:38 - 1e0906a8-efb9-4a7d-a781-0b3a4a0c2cf6

**Task**: Add progress tracking to McpExecution

**Status**: ✅ Completed

---


## 2026-04-12 22:38 - 7d67964a-7d4d-4bb0-9261-8308efd5206a

**Task**: Write a function to add two numbers

**Status**: ✅ Completed

---


## 2026-04-12 22:36 - 83627d6c-7ca6-44a4-bd5c-313aaac2738c

**Task**: Modify the `to_markdown()` method in `src/qwen_mcp/engines/sparring_v2/models.py` (lines 65-136) to include full session content when a `session_store` parameter is provided.

Current implementation o

**Status**: ✅ Completed

---


## 2026-04-12 22:33 - ffe3900b-da31-4793-9644-b94f5ce56851

**Task**: Modify SparringResponse.to_markdown() to include full session content in the output.

CURRENT PROBLEM:
- to_markdown() only returns session_id and file path
- Agent must manually read the session file

**Status**: ✅ Completed

---


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
