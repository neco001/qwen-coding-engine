# 🚀 BACKLOG

**Cel:** Dynamiczny System Operacyjny Stanu (SOS) - 100% spójności między kodem, danymi a intencją programisty.

---

## Pending

- [ ] Krok 5: Instrumentacja - Replace file-based trace with proper stderr logging in snapshot.py - aba4f3aa-260f-49a1-9709-834ed08bc060

- [ ] Krok 4: Weryfikacja Krzyżowa Klienta - Test with alternative MCP client to isolate Roo Code issues - a85af225-c2c0-4341-a088-b7eebb839d1a

- [ ] Krok 3: Optymalizacja Payloadu - Verify response returns only metadata, not full snapshot - 279b0792-d84c-46d0-9302-d3a71a09200b

- [ ] Krok 2: Ograniczenie Współbieżności I/O - Tune chunk_size and sleep in snapshot.py - d71c5bb6-5ce1-4e22-b1ce-c0875f80fc3f

- [ ] Krok 1: Izolacja Strumieni Loggingu - Force all logging to stderr in server.py - e0e88da0-0780-4f10-afdd-3e58be47c9cc

- [ ] Wdrożenie stabilnego resolwowania ścieżek (Plan Architekta) - e46c7037-73da-410e-ad13-d3517631c208

- [ ] Fallback dla braku pliku decision_log.parquet - 3c84b2d4-d0df-40ff-b91e-6001a1674340

- [ ] Pytanie dla architecta: LangGraph w projekcie? - 70f149bd-6d08-4256-8522-c98d13307073

- [ ] pytanie dla architecta - czy nie bylo by zasadnym użyc w tym projekcie np langraph? albo czegoś podobnego? a może istnieją inne narzędzia ktore by ułatwiły pracę tutaj?

- [ ] nie mamy narzędzia do tworzenia pliku decision_log.parquet - gdy robię - `qwen_add_task` w nowym projekcie - mowi, że nie ma parquete. jak robie `qwen_sync_state` tez mówi, że nie m może. moim zdaniem powinnismy w obydwu narzędziach zrobic jakis fallback, że jeżeli nie ma ani `.PLAN\BACKLOG.MD` ani `.decision_log\decision_log.parquet` to tworzy plik `decision_log.parquet` i dodaje do niego wpis o stworzeniu pliku.

- [ ] Fix Unbound fetcher TypeError and investigate MCP progress blockage - 78ddb567-3d7c-41c2-bf8f-f80caf099e60

- [ ] Dashboard ROI - wizualizacja ile energii (tokenów) spaliły poszczególne kłody (Epiki) - a8b8d19c-cedd-4a24-bf6b-35e8780f77df

- [ ] Integracja raportów użycia tokenów z dynamicznym statusem sesji w Backlogu - fecae7b9-5549-4a56-851c-28c5e800391e

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
