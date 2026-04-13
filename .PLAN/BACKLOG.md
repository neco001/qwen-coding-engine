# 🚀 BACKLOG

**Cel:** Dynamiczny System Operacyjny Stanu (SOS) - 100% spójności między kodem, danymi a intencją programisty.

---

## Pending

- [ ] Fix 5: sparring3 now truly step-by-step (one stage per MCP call) - 7b2e48d0-32b0-49d9-b41b-8476a60c5d6e
- [ ] Revert: removed word limit enforcement (obcinanie = bad) - f9989739-195b-438a-a22d-217498737739
- [ ] Doc fix: clarify FullExecutor vs ProExecutor behavior - 6ae8b900-ec7b-4887-9945-ff45e2939dbb

- [ ] Fix 1: Respect explicit sparring mode choice (was auto-overriding) - bc2cc0d3-ce1c-41cb-a91d-f3cbf390b39e
- [ ] Fix 2: Hard word limits + enforcement (timeout doesn't control response length) - 92e266b3-181a-456e-a2e7-f2adcb4492bc
- [ ] Fix 3: Reduce timeouts to fit within 300s MCP limit - bab822e5-e051-4a6e-aa9d-5a8bc5051d85
- [ ] Fix 4: JSON serialization fallback for method objects - beb69f20-f21d-494a-8a0e-19dc1d3b37a8

- [ ] ostatnio coraz częściej widzę komunikaty że coder zwrócił kod ktory nie jest taki jak powinien byc.
      trzeba to sprawdzic. przyklad:

  ````zapis_bledu
        Roo said
    Now I'll use qwen_coder to implement Task 1 - creating MODE_PROFILES configuration in config.py.


    Roo wants to use a tool on the qwen-coding MCP server
    qwen-coding

    qwen_coder
    Unified code generation tool with mode-based routing. MODES: - auto: Intelligent routing based on prompt complexity (default) - standard: Fast generation using qwen3-coder-next - pro: Heavy-duty generation using qwen3-coder-plus - expert: Maximum capability for complex refactors/architecture EXAMPLES: 1. Simple task: qwen_coder(prompt="Write a function to add two numbers") 2. Complex task: qwen_coder(prompt="...", mode="pro") 3. Expert refactor: qwen_coder(prompt="...", mode="expert") DEPRECATED TOOLS (still available but use unified internally): - qwen_coder (old) → now calls qwen_coder(mode="standard") - qwen_coder_pro (old) → now calls qwen_coder(mode="pro")
    {
      "prompt": "Implement Task 1: Create MODE_PROFILES configuration in config.py\n\nRead the current file src/qwen_mcp/engines/sparring_v2/config.py and add a MODE_PROFILES dictionary after the existing configuration sections (after DEFAULT_MODELS at the end of the file).\n\nMODE_PROFILES should define configurations for flash, full, and pro modes with the following structure:\n\n```python\nfrom dataclasses import dataclass\nfrom typing import Dict, List, Optional\n\n@dataclass\nclass ModeProfile:\n    \"\"\"Configuration profile for a sparring mode.\"\"\"\n    name: str\n    stages: List[str]\n    total_budget: int  # seconds\n    stage_weights: Dict[str, float]\n    word_limits: Dict[str, int]\n    thinking_tokens: Dict[str, int]\n    timeout_config: Dict[str, float]\n    allow_borrow: bool = False  # Allow time borrowing across stages\n    extend_timeout_pct: float = 0.5  # 50% timeout extension for complex tasks\n\nMODE_PROFILES = {\n    \"flash\": ModeProfile(\n        name=\"flash\",\n        stages=[\"analyst\", \"drafter\"],\n        total_budget=60,  # 60 seconds for fast 2-step analysis\n        stage_weights={\"analyst\": 0.45, \"drafter\": 0.55},\n        word_limits={\"analyst\": 200, \"drafter\": 300},\n        thinking_tokens={\"analyst\": 1024, \"drafter\": 1024},\n        timeout_config={\"analyst\": 30.0, \"drafter\": 30.0},\n        allow_borrow=False,\n        extend_timeout_pct=0.3,  # 30% extension for flash\n    ),\n    \"full\": ModeProfile(\n        name=\"full\",\n        stages=[\"discovery\", \"red\", \"blue\", \"white\"],\n        total_budget=225,  # 225 seconds shared budget\n        stage_weights={\"discovery\": 0.15, \"red\": 0.28, \"blue\": 0.28, \"white\": 0.29},\n        word_limits={\"discovery\": 150, \"red\": 300, \"blue\": 300, \"white\": 600},\n        thinking_tokens={\"discovery\": 1024, \"red\": 1024, \"blue\": 1024, \"white\": 2048},\n        timeout_config={\"discovery\": 33.0, \"red\": 63.0, \"blue\": 63.0, \"white\": 65.0},\n        allow_borrow=True,  # Allow borrowing from previous stages\n        extend_timeout_pct=0.5,  # 50% extension for complex tasks\n    ),\n    \"pro\": ModeProfile(\n        name=\"pro\",\n        stages=[\"discovery\", \"red\", \"blue\", \"white\"],\n        total_budget=900,  # 900 seconds total (225s per stage)\n        stage_weights={\"discovery\": 0.15, \"red\": 0.28, \"blue\": 0.28, \"white\": 0.29},\n        word_limits={\"discovery\": 150, \"red\": 600, \"blue\": 600, \"white\": 800},\n        thinking_tokens={\"discovery\": 2048, \"red\": 2048, \"blue\": 2048, \"white\": 4096},\n        timeout_config={\"discovery\": 120.0, \"red\": 180.0, \"blue\": 180.0, \"white\": 360.0},\n        allow_borrow=True,\n        extend_timeout_pct=0.5,  # 50% extension for complex tasks\n    ),\n}\n```\n\nAdd this to the end of config.py file. The MODE_PROFILES should be added after the DEFAULT_MODELS dictionary.\n\nReturn the complete updated file content."
    }



    API Request
    $0.0000

    Roo said
    The qwen_coder output shows a different file structure than expected. Let me read the current config.py to see what's there now and then apply the correct diff.

  ````

  pomysł: no wlasnie - czy ty mu podajesz odpowiedni kontekst? a moze coder powinien miec sam w sobie zaprogramowane multi-stage, i gdy nie ma tresci plikow to sam wywola read_file?

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

- [ ] dodaj task: architect podaje kod. czy powinien?
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

- [ ] Fix Unbound fetcher TypeError and investigate MCP progress blockage - 78ddb567-3d7c-41c2-bf8f-f80caf099e60

- [ ] Dashboard ROI - wizualizacja ile energii (tokenów) spaliły poszczególne kłody (Epiki) - a8b8d19c-cedd-4a24-bf6b-35e8780f77df

- [ ] Integracja raportów użycia tokenów z dynamicznym statusem sesji w Backlogu - fecae7b9-5549-4a56-851c-28c5e800391e

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
