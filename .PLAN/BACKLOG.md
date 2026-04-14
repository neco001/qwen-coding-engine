# 🚀 BACKLOG

**Cel:** Dynamiczny System Operacyjny Stanu (SOS) - 100% spójności między kodem, danymi a intencją programisty.

---

## Pending

- [x] Align qwen_architect brownfield mode with greenfield mode - remove direct code generation - 92d2e556-69f6-4a5e-855e-57ca2951f844
  - **SOLVED**: Modified LP_BROWNFIELD_PROMPT to remove direct diff generation instructions and add swarm task assignment
  - **SOLVED**: Modified generate_lp_blueprint to parse brownfield architect response into swarm_tasks structure
  - **VERIFIED**: All 6 enforcement tests pass, 0 regression alerts in baseline comparison

- [x] improvement of session_context format in accordance with [proposed template] (C:\Repos_Toolbox\mcp_servers\Qwen-mcp\qwen-coding-local\.PLAN\qwen-coding-project-context-template.md) - a572933b-639f-458b-bc5d-a27ae8129ed7

- [ ] Fix Unbound fetcher TypeError and investigate MCP progress blockage - 78ddb567-3d7c-41c2-bf8f-f80caf099e60

- [ ] Dashboard ROI - wizualizacja ile energii (tokenów) spaliły poszczególne kłody (Epiki) - a8b8d19c-cedd-4a24-bf6b-35e8780f77df

- [ ] Integracja raportów użycia tokenów z dynamicznym statusem sesji w Backlogu - fecae7b9-5549-4a56-851c-28c5e800391e

## copleted

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
- [ ] tworzenie session context skutkuje produkcją plikow tmp w katalogu .\.context. a te nie są wymazywane.
- [ ] czy narzędzie `qwen_init_request` jest potrzebne - miało być wywoływane w każdym z narzędzi.
- [ ] nie działa HUD (wtyczka vsc)
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

## Completed (MCP Timeout Fix - Sparring3 Action Plan)

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

## Completed

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

## Pending (Legacy)

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
