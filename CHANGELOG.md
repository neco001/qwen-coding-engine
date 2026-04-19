# CHANGELOG

## 2026-04-19 12:44 - 9f5eba54-7c39-4595-857b-117068ad0a1e

**Task**: Refactor src/job_hunter.py and scrapers/LinkedIn_Scraper/fetch_linkedin.py to integrate LinkedIn scraping into the main hunt flow. 

Ensure:
1. JobHunter.run_hunt calls BOTH Pracuj and LinkedIn.
2. Li

**Status**: ✅ Completed

---

## 2026-04-19 08:20 - 0858ba1d-2a1b-4db6-8c50-b2a9cf0dec68

**Task**: Write a pytest test for ScanStatusResponse pydantic model in server.py.
The model should have:
- status: str (READY, BUSY, ERROR)
- message: str
- progress: int (0-100)
- feed: List[str]

The test sho

**Status**: ✅ Completed

---

## 2026-04-19 07:58 - 908b74b6-aeeb-4aef-93b7-ea2021e4bb6e

**Task**: Refactor the calculate_cv_match method in src/job_hunter.py to include keyword tracking.
Requirements:
1. Initialize matched_keywords list.
2. When a keyword is found in the text, add it to the matche

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-19 07:57:58

## [2026-04-18 22:05:31] c68b4abe-4946-42d5-9a5a-900251f444c5

**Task**: [BACKEND] Explainable Scoring Engine

**Advice**: Update calculate_cv_match in src/job_hunter.py to return a dict with total_score, matched_keywords, and breakdown. Ensure server.py mirrors this structure in its response.

---

## [2026-04-18 22:05:31] 271ffb92-2d21-43d2-8802-12cb00973644

**Task**: [BACKEND] Data Integrity Lock

**Advice**: Implement acquire_lock/release_lock in JobHunter class using a .lock file. Add /api/hunt/lock-status endpoint to server.py. Use finally blocks to ensure lock cleanup.

---

## [2026-04-18 22:05:31] 7c9c4570-0044-4d39-8301-7daeb912a41b

**Task**: [BACKEND] Manual Override API

**Advice**: Create /api/job/update-status in server.py. Update status mapping logic to use 'Review' status for scores 60-84. Ensure database persistence.

---

## [2026-04-18 22:05:31] d7b07f66-aa2f-4c9e-aba8-1f876bd533bb

**Task**: [FRONT] Resilient Polling Hub

**Advice**: Update main.ts: use document.visibilityState to pause polling. Implement poll interval doubling on failure and reset on success.

---

## [2026-04-18 22:05:31] 3ee59c7c-6427-4331-9d48-2bc7b013eb90

**Task**: [FRONT] Intelligence Highlighting

**Advice**: Implement highlightKeywords function in main.ts. Bind it to the currentJob.description in index.html using x-html. Ensure sanitization.

---

## [2026-04-18 22:05:31] e0e6e76e-988c-4556-ab0c-45a077ab0c77

**Task**: [UI] Tactical Feed & Controls

**Advice**: Update index.html: Add top-level banner for lock status, Search input in Kanban header, and Pencil icons on cards for manual override.

---

## 2026-04-19 07:57 - d7b10f58-2f82-4d18-8552-700dda44e363

**Task**: Write a pytest for the updated calculate_cv_match function in src/job_hunter.py. 
The test should verify that:
1. The return value is a dictionary.
2. It contains 'score' (int), 'matched_keywords' (li

**Status**: ✅ Completed

---

## 2026-04-18 21:49 - 84bdb767-c4b6-495b-888e-08ffda7de0e6

**Task**: Write a pytest test for the NEW FastAPI endpoints in `SpecterHub/server.py`. \n\nThe test should use `fastapi.testclient.TestClient` and verify:\n1. `POST /api/hunt/trigger` initiates a background tas

**Status**: ✅ Completed

---

## 2026-04-18 21:45 - 779aa345-6b35-4cbf-ba5e-f2d9bdd89456

**Task**: Write a pytest test for the REFACTORED job_hunter.py. \n\nThe test should verify that:\n1. We can instantiate a `JobHunter` class (or similar library-style interface) without immediately triggering CL

**Status**: ✅ Completed

---

## 2026-04-17 01:23 - ed43b0a3-1a7f-4c65-8148-fa2da46f0c15

**Task**: Fix the 'broken' icons in SpecterOS v2 (SpecterHub/v2/).
1. Replace Unicode characters in 'src/main.ts' with a consistent icon naming system or component-based SVG injection.
2. Update 'index.html' (o

**Status**: ✅ Completed

---

## 2026-04-17 01:11 - 13601eba-adce-47db-b38b-bd32a4b082e5

**Task**: Generate the initial infrastructure for SpecterOS v2 in 'SpecterHub/v2/'.
1. Create folder structure: src/core, src/ui, src/db, src/api, src/styles.
2. index.html: Define a modern 3-pane layout using 

**Status**: ✅ Completed

---

## 2026-04-17 00:41 - 1e9997d4-df45-46a1-be52-dd56b8ee5a6d

**Task**: Based on the Lachman Blueprint, implement the database schema alignment.
1. Create a new migration file 'migrations/004_align_offers_schema.sql' containing the ALTER TABLE ADD COLUMN IF NOT EXISTS sta

**Status**: ✅ Completed

---

## 2026-04-17 00:10 - 8c70199e-b725-4506-bd67-cc617418b585

**Task**: Write an integration test 'tests/test_migration_integration.py' that:
1. Tests 'Fresh Bootstrap':
   - Creates a temporary directory.
   - Sets 'db_manager.DATA_DIR' to this directory.
   - Calls 'db_

**Status**: ✅ Completed

---

## 2026-04-17 00:06 - b926b6f7-5a5d-4db2-9969-ca2d182572e4

**Task**: Write a pytest test file 'tests/test_migration_runner.py' that:
1. Imports 'get_conn' from 'src.db_manager'.
2. Mocks a database connection.
3. Tests if 'src.db_manager' has a function/method '_run_mi

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-17 00:01:03

## [2026-04-16 21:26:53] 3e1f6f01-5d20-411e-984b-7332b5d0aad8

**Task**: Build column dictionary from input file (cleaned of \n)

**Advice**: Read the first row of Product_list_2026.4.9.csv and build a mapping between the original column names (containing \n) and cleaned versions. This dictionary will be essential for Tier 1 vs Tier 2 coordination. Save the result to config/original_column_dictionary.json.

---

## 2026-04-17 00:01 - 76651843-30bb-471c-b18a-9b1d6706c1f6

**Task**: Write a pytest test file 'tests/test_migration_baseline.py' that verifies:
1. A directory 'migrations' exists in the project root.
2. A file 'migrations/000_base_schema.sql' exists.
3. The content of 

**Status**: ✅ Completed

---

## 2026-04-16 00:34 - 52ea8361-c4fb-47ce-94c6-521f6ef9fe99

**Task**: Implement Multi-Engine PDF Data Extraction in src/core/extractor.py.

Requirements:
1. Engine 1: PyMuPDF (fitz) - RAW text extraction + regex patterns.
2. Engine 2: pdfplumber - Structural extraction 

**Status**: ✅ Completed

---

## 2026-04-15 23:18 - b7afe695-b892-447f-8911-3526b94609ab

**Task**: Implement 'src/integrations/cloud_providers.py' with the following:

1. Abstract Base Class 'CloudDeliveryProvider':
   - method 'upload_file(local_path, remote_path)': abstract
   - method 'delete_fi

**Status**: ✅ Completed

---

## 2026-04-15 23:16 - 68caf116-c79a-4d5d-be42-fd96a76c125f

**Task**: Write a pytest test file for the new Cloud Delivery System. 
Location: tests/test_cloud_providers.py.

Requirements:
1. Test 'CloudDeliveryProvider' (Abstract Base Class): Ensure it cannot be instanti

**Status**: ✅ Completed

---

## 2026-04-15 23:11 - 2c659a25-5b3c-4f9a-803b-7d469d0ea743

**Task**: Refactor src/db/config_manager.py to the new 'ConfigManager' class using AIOSQLITE, but PRESERVE EXACTLY the backward compatibility logic.

CRITICAL REQUIREMENTS:
1. PRESERVE the `_get_fernet(self, db

**Status**: ✅ Completed

---

## 2026-04-15 23:10 - ff454a74-a8a8-47fd-9918-9dde5c720683

**Task**: Rewrite the previous ConfigManager implementation to use ASYNCIO and AIOSQLITE. 
The project is built on async DB operations.

Key requirements:
1. Use 'aiosqlite' instead of 'sqlite3'.
2. Methods sav

**Status**: ✅ Completed

---

## 2026-04-15 23:09 - 8b5341c0-68dd-4628-91e5-35120e5c9d87

**Task**: Refactor src/db/config_manager.py according to the architectural plan:

1. Rename SMTPConfigManager to ConfigManager.
2. Make it generic: instead of smtp-only fields, use a generic key mapping.
3. Sup

**Status**: ✅ Completed

---

## 2026-04-15 23:06 - efac2e18-2a29-453a-8ac8-558c306b6c62

**Task**: Write a pytest test file for the new 'ConfigManager' (refactored from SMTPConfigManager). 
Current code is in src/db/config_manager.py.

Requirements for the test:
1. Test backward compatibility: Save

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-15 23:04:55

## [2026-04-15 20:52:29] 1837d3f4-8421-48d2-bc44-36519a63e34c

**Task**: [INFRA] Universal ConfigManager with Secret Encryption

**Advice**: Refactor SMTPConfigManager into a generic ConfigManager handle Cloud/SMS/SMTP settings. Ensure all secrets (tokens, api keys) are encrypted via Fernet. Maintain compatibility with the existing config table.

---

## [2026-04-15 20:52:29] c3198d18-ecbf-4430-b41f-74f0ac0b8678

**Task**: [INFRA] Multi-Cloud Provider System (Strategy Pattern)

**Advice**: Implement the CloudDeliveryProvider interface and concrete classes for Dropbox (SDK) and Google Drive (API v3). Use ABC for the interface. Support account_type logic (Basic/Paid) within providers.

---

## [2026-04-15 20:52:29] a1126c50-a573-4b74-a5a8-4ac62404f3a0

**Task**: [SECURITY] Cloud-Safe Password Policy Enforcement

**Advice**: Refactor generate_password() to accept a delivery_mode. If CLOUD, strictly allow only PESEL-based formats. Disable Option D (Word+PESEL) for cloud delivery to enhance security.

---

## [2026-04-15 20:52:29] d74f73d9-e5e7-4ac9-9d1b-993f4fea4d24

**Task**: [UI] Centralized Settings Hub (Tabbed Interface)

**Advice**: Create SettingsView using CTkTabview. Migrate existing EmailSettingsView and SMSView into the new tabs. Add 'Cloud' tab with dynamic fields for Dropbox/GDrive. Implement provider selection logic.

---

## [2026-04-15 20:52:29] 647b4129-f6cf-4f78-ba62-f45dd8bae2b6

**Task**: [LOGIC] Automated 24h Cloud Clean-up Worker

**Advice**: Implement SmartCleanUpWorker as an async task in App. It must trigger on startup and post-upload. Worker logic: fetch file list from cloud, check 24h TTL, delete expired files. Handle errors gracefully.

---

## [2026-04-15 20:52:29] e3877910-3070-42b4-9a06-e1de8c292726

**Task**: [INT] Wire Multi-Cloud Delivery to Main UI Flow

**Advice**: Integrate the new CloudDeliveryProvider into the main EncryptionView workflow. Add a toggle to choose between Email and Cloud delivery. Connect success signals to the cleanup worker.

---

## 2026-04-15 21:22 - b3b51a76-e7e5-4b9c-9a0b-afb01c90a081

**Task**: Write a pytest for Task 55a5ec7c (Extractor part):
1. Update extract_patient_data to support 'name'.
2. Mock fitz (pymupdf) to return a string containing 'Imię: Jan' and 'Nazwisko: Kowalski' and a PES

**Status**: ✅ Completed

---

## 2026-04-15 21:21 - e3edddf6-1e10-459a-8030-351d0e648462

**Task**: Write a pytest for the UI integration logic of Gender Detection (Task 55a5ec7c):
1. Setup: Mode='intelligent'. Detector returns (Gender.MALE, 1.0). Verify logic suggests 'Pan'.
2. Setup: Mode='intelli

**Status**: ✅ Completed

---

## 2026-04-15 21:19 - 4155663c-b982-42ba-8a20-118187a59ed1

**Task**: Write a pytest for Task 03340e4a:
1. Use SMTPConfigManager to save 'gender_mode': 'traditional'.
2. Load config and verify 'gender_mode' is 'traditional'.
3. Verify default 'gender_mode' is 'intellige

**Status**: ✅ Completed

---

## 2026-04-15 21:18 - 5b51d290-8db2-4864-9ac0-27c098df0420

**Task**: Implement src/core/gender_detector.py based on 02_pan_pani_spec.md:
1. Enum Gender (MALE='M', FEMALE='F', NEUTRAL='N').
2. Class GenderDetector:
   - Load 'src/resources/polish_names.json' in __init__

**Status**: ✅ Completed

---

## 2026-04-15 21:17 - 1dc41dc8-9f53-43f4-8370-9e397bb71337

**Task**: Write a pytest for Task 01fe4cd3 (GenderDetector):
1. Test init: Loads 'src/resources/polish_names.json'.
2. Test detect: 'Jan' -> Gender.MALE (Direct match).
3. Test detect: 'Anna' -> Gender.FEMALE (

**Status**: ✅ Completed

---

## 2026-04-15 21:15 - e8f2c89c-10b3-465c-944d-b7ff648e4c85

**Task**: Create a JSON dataset of popular Polish first names in 'name': 'gender' format (M for male, F for female).
Include at least 50 common names (Jan, Anna, Maria, Andrzej, Piotr, Krzysztof, Katarzyna, etc

**Status**: ✅ Completed

---

## 2026-04-15 21:15 - f1a9f01c-00cd-4f53-bec1-2a4415c15599

**Task**: Write a pytest for Task c1633259:
1. Check if 'src/resources/polish_names.json' exists.
2. Verify it is a valid JSON.
3. Verify it contains at least 'Jan': 'M' and 'Anna': 'F'.
Save it to tests/test_g

**Status**: ✅ Completed

---

## 2026-04-15 17:07 - 2ad724c1-935a-4d10-81fb-4708d00f204a

**Task**: Refactor src/integrations/email_sender.py based on audit findings.
Changes:
1. REMOVE self.config = config to avoid keeping credentials in memory.
2. Add path traversal protection for attachments (ens

**Status**: ✅ Completed

---

## 2026-04-15 17:04 - 2c6d1495-a964-471a-ab26-8e4d2c3d8cc0

**Task**: Implement to pass the tests in tests/test_email.py.
Create src/integrations/email_sender.py with the following requirements:
1. Class: EmailService(config: dict)
2. Method: send_result(recipient, subj

**Status**: ✅ Completed

---

## 2026-04-15 16:56 - 91d6bc67-1923-4e0e-86e6-3762383cb4a7

**Task**: Napisz test pytest (test_processor_lockdown.py), który sprawdzi funkcję export_clean_view w core/processor.py.
Test musi zweryfikować:
1. Czy procesor rzuca FileNotFoundError lub RuntimeError, gdy bra

**Status**: ✅ Completed

---

## 2026-04-15 16:54 - 00f0f62c-a261-4f33-85e3-5d6937fd40e9

**Task**: Zaimplementuj skrypt scripts/generate_immutable_schema.py, który:
1. Wczytuje config/column_registry.yaml.
2. Iteruje po wszystkich wpisach.
3. Dla każdego kanaonical_name bierze report_name i czyści 

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-15 16:53:30

## [2026-04-15 13:40:37] d89152a3-ec2e-4138-876d-a5406ddb9da0

**Task**: Implement Core Email Engine (EmailService)

**Advice**: Create 'src/integrations/email_sender.py' with EmailService class. Use standard 'smtplib' and 'email' modules. Implement sending logic with attachment support and basic HTML template placeholder replacement. Ensure it runs in background via AsyncBridge.

---

## [2026-04-15 13:40:37] 0a1ad1a1-e70b-4b57-bd24-762d4724a42f

**Task**: Implement Encrypted SMTP Config Manager

**Advice**: Implement 'EmailConfigManager' to read/write encrypted SMTP settings (host, port, user, password_enc) from the SQLite 'config' table. Use the existing 'cryptography' library for password encryption. Connect it to the Database Manager.

---

## [2026-04-15 13:40:37] f34109d3-22a6-4602-aea5-f69fcffc1d6b

**Task**: Create UI for Email Settings & Connectivity Test

**Advice**: Create 'EmailSettingsView' in 'src/gui/app.py' (or as a separate component). Include fields for SMTP config, TLS/SSL toggle, a 'Save' button, and a 'Test Connection' button that triggers a sample email via AsyncBridge.

---

## [2026-04-15 13:40:37] b5d6d55a-023c-4c5d-83ca-e8805a376711

**Task**: Integrate Email Sending into Encryption Workflow

**Advice**: Integrate email sending into the main encryption workflow. After successful PDF encryption, prompt the user or automatically prepare an email draft with the encrypted PDF attached. Validate MIME type as PDF before sending.

---

## [2026-04-15 14:28:02] ac3bda58-531c-44a8-ac5d-514368e5a18c

**Task**: [OVERSEAS] Refaktoryzacja rejestru kolumn (Pure English)

**Advice**: Zaktualizuj wszystkie pola 'report_name' na czysty angielski. Przykład: '1 Untaxed price 1（CNY)' -> 'Untaxed Price CNY', '69码 EAN code' -> 'EAN code'. Usuń chińskie znaki z wariantów.

---

## [2026-04-15 14:28:02] fe6507b7-4d1c-427a-ab70-37721b5509b9

**Task**: [OVERSEAS] Implementacja logiki De-Sinization w procesorze

**Advice**: Wprowadź filtr Drop Chinese Columns (wyrzucanie '中文'). Dodaj logikę czyszczenia '英文' i spacji z nazw w ColumnResolver. Dodaj eksport słownika zmian do JSON.

---

## [2026-04-15 14:28:02] 24af190c-c634-4e9b-88c1-8d36318d0512

**Task**: [OVERSEAS] Przebudowa skryptu Power Query dla nowych nazw

**Advice**: Stwórz nową wersję skryptu M, która oczekuje czystych nazw angielskich. Usuń zbędne mapowania 'RenameColumns' dla pól, które będą już poprawne w Parquet.

---

## 2026-04-15 16:53 - bd0c6c6b-fba4-4c84-8ce1-92e48f009048

**Task**: Napisz test pytest (test_generate_schema.py), który zweryfikuje logikę czyszczenia nazw w nowym skrypcie generate_immutable_schema.py.
Test musi sprawdzić:
1. Usunięcie chińskich znaków (regex).
2. Us

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-15 15:17:53

## [2026-04-15 15:17:08] 59499826-e4dd-49d3-83ae-4a7926515749

**Task**: [P0] PDF Data Extraction & Preview Integration

**Advice**: Implement automatic data extraction (PESEL, Surname) and PDF preview thumbnail using PyMuPDF (fitz) to fulfill P0 requirements from SPEC.md. Move from manual entry to 'Drop & Extract' workflow.

---

## 2026-04-15 15:17 - 57508475-19ca-4a70-aeac-ac198e5c389f

**Task**: Implement the following core modules for automation:

1. 'src/core/extractor.py':
   - 'extract_patient_data(pdf_path: Path) -> dict':
     - Use 'pymupdf' (fitz) to extract text from the first page.

**Status**: ✅ Completed

---

## 2026-04-15 15:09 - b8a89499-a1fb-40de-bfb1-a0acd6cd9c0e

**Task**: Complete 'EncryptionView' in 'src/gui/app.py':

1. Components:
   - 'input_file_path': Entry + 'Browse' button (using ctk.filedialog).
   - 'patient_surname': Entry.
   - 'patient_pesel': Entry.
   - 

**Status**: ✅ Completed

---

## 2026-04-15 15:07 - b7ea17dd-0c78-4f9f-a6ea-8152aa2d0a73

**Task**: Implement 'src/gui/app.py' using 'customtkinter':

1. 'App(ctk.CTk)' class:
   - Window size: 1100x580.
   - Title: 'SecureExamPDF v1.0 - Desktop Edition'.
   - 'sidebar_frame': A left sidebar with bu

**Status**: ✅ Completed

---

## 2026-04-15 15:06 - 80f34965-9499-4a25-836c-181255565b5c

**Task**: Implement 'src/gui/bridge.py':

1. 'AsyncBridge' class:
   - '__init__': 
     - Create a new 'asyncio' event loop.
     - Start a 'threading.Thread' (daemon=True) that calls 'loop.run_forever()'.
   

**Status**: ✅ Completed

---

## 2026-04-15 15:05 - 19fe0011-8d7e-4e71-b16e-47799c917672

**Task**: Write unit tests in 'tests/test_gui_bridge.py' for a thread-safe 'AsyncBridge':

1. 'src/gui/bridge.py':
   - 'AsyncBridge' class:
     - Starts a background 'threading.Thread' with 'asyncio.AbstractE

**Status**: ✅ Completed

---

## 2026-04-15 15:02 - 5d405fdd-6f25-49a0-ab29-95085c524870

**Task**: Implement 'src/utils/logger.py':

1. 'GDPRFilter(logging.Filter)':
   - Override 'filter(record: logging.LogRecord) -> bool'.
   - Use regex to scrub 'msg' in record:
     - '\b\d{11}\b' -> '[REDACTED

**Status**: ✅ Completed

---

## 2026-04-15 15:02 - 4db5a3c2-de72-48c4-a70f-ae2b6b907939

**Task**: Write unit tests in 'tests/test_logging.py' for:

1. 'src/utils/logger.py':
   - 'setup_logger(log_file: Path)': Configure standard logging to file and console.
   - 'GDPRFilter(logging.Filter)': A fi

**Status**: ✅ Completed

---

## 2026-04-15 14:58 - ab37b75f-6e1f-4501-86ab-1a4365962272

**Task**: Implement the Following modules:

1. 'src/db/manager.py':
   - 'async init_db(db_path: Path)': Create tables if not exist. 
     - 'audit_logs': id (int primary key), operation (text), status (text), 

**Status**: ✅ Completed

---

## 2026-04-15 14:57 - aef9c6ca-0881-4c85-b9f6-17429e4d7db2

**Task**: Write unit tests in 'tests/test_db_and_license.py' for:

1. 'src/db/manager.py':
   - 'async init_db(db_path: Path)': Initialize tables: 'audit_logs', 'sms_stats', 'config'.
   - 'async log_operation(

**Status**: ✅ Completed

---

## 2026-04-15 14:50 - 05db53c9-e6b1-49bc-90a5-b636393124c7

**Task**: Implement the Following modules in src/core/:

1. 'src/core/password_handler.py':
   - 'validate_pesel(pesel: str) -> bool': Use weights [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]. Sum results, take mod 10. Check

**Status**: ✅ Completed

---

## 2026-04-15 14:49 - 3c31fb9b-7379-4d09-9795-3faa753cfec2

**Task**: Write a comprehensive pytest file 'tests/test_crypto.py' for the following requirements:
1. Module 'src/core/password_handler.py':
    - Function 'validate_pesel(pesel: str) -> bool' (Modulo 10 check)

**Status**: ✅ Completed

---

## 2026-04-15 14:45 - 556a0dc0-8ed7-4235-bf4f-b3ddd09b9b51

**Task**: Create a pytest file 'tests/test_scaffolding.py' that verifies the existence of the following project structure in current directory:
- pyproject.toml
- src/main.py
- src/core/ (directory)
- src/db/ (

**Status**: ✅ Completed

---

## 2026-04-15 00:44 - ae37f7e7-f5c4-4bc3-9cc4-da31e97565b4

**Task**: Refactor `_apply_advices_to_files` in `src/qwen_mcp/engines/decision_log_sync.py` to delegate to `DecisionLogOrchestrator`.

The current `_apply_advices_to_files(self, backlog_path: Path, changelog_pa

**Status**: ✅ Completed

---

## 2026-04-15 00:30 - a0f6d200-2bff-406b-aade-2b3e7b312bc5

**Task**: Refactor `SectionManager` in `src/qwen_mcp/engines/markdown_layer/sections.py` to:
1. Pre-compile regex patterns as class-level constants
2. Fix EOF header detection by using `r'(## Completed\n?)'` in

**Status**: ✅ Completed

---

## 2026-04-15 00:27 - 2a7e7e83-efa0-47cc-aa72-20f200bfb7c0

**Task**: I need to refactor `_mark_task_completed` in `DecisionLogSyncEngine` to delegate to `DecisionLogOrchestrator`.

**Current signature of `_mark_task_completed` (line 938-987):**
```python
def _mark_task

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-15 00:26:35

## [2026-04-15 00:26:01] e68b693c-17b7-4bfb-9c00-527008289d8f

**Task**: Integrate DecisionLogOrchestrator into decision_log_sync.py

**Advice**: Replace naive string.replace in _apply_advices_to_files and _mark_task_completed with calls to DecisionLogOrchestrator.archive_task(decision_id). This fixes the archival bug where completed tasks stay in Pending instead of moving to Completed. The Orchestrator uses SectionManager which correctly relocates the task line.

---

## 2026-04-15 00:26 - b8da47e1-a389-4535-af6c-b08d0ee6eb40

**Task**: Write a pytest test (RED phase) for the integration of `DecisionLogOrchestrator` with `decision_log_sync.py`.

Context:
- `DecisionLogSyncEngine` (in `qwen_mcp.engines.decision_log_sync`) has a method

**Status**: ✅ Completed

---

## 2026-04-15 00:21 - bc6aa5bb-a6aa-4603-9d82-ebc0c5533f2e

**Task**: Implement the `DecisionLogOrchestrator` class in `qwen_mcp.engines.orchestrator` to pass my tests.

```python
import pytest
from unittest.mock import MagicMock, patch

from qwen_mcp.engines.orchestrat

**Status**: ✅ Completed

---

## 2026-04-15 00:20 - 51d2f87f-988c-48bd-abfa-464236c73e42

**Task**: Write a pytest test for the `DecisionLogOrchestrator` class in `qwen_mcp.engines.orchestrator`. 
The required functionality for Orchestrator:
- Takes `path_resolver: PathResolver` as dependency.
- Use

**Status**: ✅ Completed

---

## 2026-04-15 00:19 - 42b62f82-7018-42cc-b769-a76cf2db7424

**Task**: Write a pytest test for the `DecisionLogOrchestrator` class in `qwen_mcp.engines.orchestrator`. 
The required functionality for Orchestrator:
- Takes `PathResolver` as dependency.
- Has `add_tasks(tas

**Status**: ✅ Completed

---

## 2026-04-15 00:11 - 89412b58-c99e-46dd-8298-51ab9f47080a

**Task**: There is white space and new line mismatch in `SectionManager.archive_task()`.

```
E           AssertionError: assert '# Tasks\n\n#...sk C [id:789]' == '# Tasks\n\n#...sk A [id:123]'
E             
E

**Status**: ✅ Completed

---

## 2026-04-15 00:10 - d63d39a3-547d-4537-bae1-34602e477881

**Task**: Implement the `SectionManager` class in `qwen_mcp.engines.markdown_layer.sections` to pass the following test:

```python
import pytest
from qwen_mcp.engines.markdown_layer.sections import SectionMana

**Status**: ✅ Completed

---

## 2026-04-15 00:09 - 13232cb8-82da-4a36-85ba-edb46a0e60ab

**Task**: Write a pytest test for the `SectionManager` class in `qwen_mcp.engines.markdown_layer.sections`. The `SectionManager` is responsible for handling markdown sections. It should have the following metho

**Status**: ✅ Completed

---

## 2026-04-15 00:08 - 0792cb8c-eeb2-4f7b-9832-c7267e8fa337

**Task**: Implement the `MarkdownParser` class in `qwen_mcp.engines.markdown_layer.parser` to pass the following test:

```python
import pytest
from qwen_mcp.engines.markdown_layer.parser import MarkdownParser

**Status**: ✅ Completed

---

## 2026-04-15 00:07 - 8f56f595-1eaa-4793-a03b-82531e608b30

**Task**: Write a pytest test for the `MarkdownParser` class in `qwen_mcp.engines.markdown_layer.parser`. The parser should have methods like `extract_section(content: str, header: str) -> str`, `extract_tasks(

**Status**: ✅ Completed

---

## 2026-04-14 23:55 - af97027b-85e7-49b5-9152-7d01da50e572

**Task**: Implement to pass this test:

```python
import pytest
from pathlib import Path
from qwen_mcp.engines.io_layer.file_handler import FileHandler

class TestFileHandler:
    """Test suite for the abstrac

**Status**: ✅ Completed

---

## 2026-04-14 23:53 - 0a3a650a-056b-4b22-b4e0-b814018bf7bd

**Task**: Write a pytest test for an abstract FileHandler class in `qwen_mcp.engines.io_layer.file_handler`. The FileHandler should have static/class methods for atomic file write operations. It requires tests 

**Status**: ✅ Completed

---

## SOS Sync - 2026-04-14 23:34:15

## [2026-04-14 21:31:57] e385bf52-9da7-4734-8758-053db85fac2a

**Task**: Create models/task.py

**Advice**: Create models/task.py with Task dataclass for state logic

---

## [2026-04-14 21:31:57] 343c4ba6-91f6-4b76-bee5-4e2a7cccef50

**Task**: Create io_layer/path_resolver.py

**Advice**: Create io_layer/path_resolver.py with path configuration

---

## [2026-04-14 21:31:57] 2adfc111-10b3-47f9-8d9e-14ce66cea3db

**Task**: Create io_layer/file_handler.py

**Advice**: Create io_layer/file_handler.py with abstracted file operations

---

## [2026-04-14 21:31:57] 750be343-7fc7-448f-8389-12808db90775

**Task**: Create markdown_layer/parser.py

**Advice**: Create markdown_layer/parser.py for section/task extraction

---

## [2026-04-14 21:31:57] 6a6d30a5-4191-42cf-9f46-41d711064bc6

**Task**: Create markdown_layer/formatter.py

**Advice**: Create markdown_layer/formatter.py for output generation

---

## [2026-04-14 21:31:57] a230f8df-d222-4ab1-9ff8-07b657613d60

**Task**: Create markdown_layer/sections.py

**Advice**: Create markdown_layer/sections.py for section management

---

## [2026-04-14 21:31:57] 5a1536af-5105-4307-9eb2-613b3c48de32

**Task**: Create orchestrator.py

**Advice**: Create orchestrator.py coordinating all layers

---

## [2026-04-14 21:31:57] 32740fe0-369e-4306-bed7-199804e0234e

**Task**: Implement archival logic

**Advice**: Implement archival logic in sections.py + orchestrator

---

## [2026-04-14 21:31:57] 8bcd4fa6-5773-4085-ab51-3b6c60f0df3f

**Task**: Update path resolution for CHANGELOG.md

**Advice**: Update path resolution for CHANGELOG.md location

---

## [2026-04-14 21:31:57] d1d6e677-06aa-42e0-8a2f-ffc6331ea07c

**Task**: Update original decision_log_sync.py

**Advice**: Update original decision_log_sync.py to use new modules

---

## [2026-04-14 21:31:57] 27e280d4-839b-4b4e-a6c8-4361d8d55c86

**Task**: Add unit tests for each layer

**Advice**: Add unit tests for each layer

---

## [2026-04-14 21:31:57] d73a6408-6bca-44f0-a636-6aae2326218e

**Task**: Remove deprecated code paths

**Advice**: Remove deprecated code paths

---

## 2026-04-14 23:34 - b7864bfb-890c-4b87-8617-4098e86b78af

**Task**: Write a pytest test for a `Task` dataclass in `qwen_mcp.engines.models.task`. The `Task` dataclass should have fields like `decision_id`, `description`, `state` (e.g., 'pending', 'completed'). It shou

**Status**: ✅ Completed

---

# Changelog

## [Full changelog](./.PLAN/CHANGELOG.md)

## 1.2.0 Release

**Date:** 2026-04-12

### New Features

- **Batch Task Creation (`qwen_add_tasks`)**: Add multiple tasks to BACKLOG.md and decision_log.parquet in a single call
  - New MCP tool `qwen_add_tasks` in [`server.py`](src/qwen_mcp/server.py:431-504)
  - New function `add_tasks_to_backlog_batch()` in [`tools.py`](src/qwen_mcp/tools.py:785-844)
  - New method `add_tasks()` in [`decision_log_sync.py`](src/qwen_mcp/engines/decision_log_sync.py:364-504)
  - Chunk-based processing (default: 20 tasks per chunk) to avoid MCP timeout
  - Full test coverage in [`tests/test_batch_tasks.py`](tests/test_batch_tasks.py) (8 tests passing)
  - Handles new projects (creates files if missing) and preserves existing records

### Changes

- Test files moved from root directory to `tests/` for better organization
- BACKLOG.md updated with completed batch task creation task

---

## 1.1.1 Release

**Date:** 2026-04-09

### Fixes

- **Sparring Session File Location**: Fixed agent not knowing where to find session results
  - Added session file path to `SparringResponse.to_markdown()` output
  - Agent now sees full path: `{storage_dir}/{session_id}.json`
  - Fixed file extension mismatch (.json vs .md)
  - Works with all storage directory resolution tiers (env, user-level, fallback)
  - Added 5 unit tests in `tests/test_sparring_session_path.py`

- **max_tokens=0 Truncation Fix**: Fixed Python falsy evaluation causing sparring3 responses to be truncated
  - Root cause: `if max_tokens:` treated 0 as falsy, ignoring unlimited token setting
  - Fix: Changed to `if max_tokens is not None:` in [`completions.py`](src/qwen_mcp/completions.py:69)
  - Impact: All sparring modes (flash/full/pro) now properly support unlimited output tokens
  - Configuration: `MAX_TOKENS_CONFIG` set to 0 for all sparring modes
  - Applied to: qwen_architect, qwen_coder, qwen_audit, qwen_sparring, qwen_update_session_context

### Changes

- `src/qwen_mcp/engines/sparring_v2/models.py`: Added `storage_dir` parameter to `to_markdown()`
- `src/qwen_mcp/tools.py`: Pass session store directory to response formatter
- `src/qwen_mcp/completions.py`: Fixed max_tokens zero check (`is not None` instead of falsy check)
- `src/qwen_mcp/engines/sparring_v2/config.py`: Set `MAX_TOKENS_CONFIG` to 0 for unlimited tokens
- `src/qwen_mcp/api.py`: Set `max_tokens=0` for meta-analysis endpoint

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
