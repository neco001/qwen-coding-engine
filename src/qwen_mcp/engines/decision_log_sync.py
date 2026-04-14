import os
import sys
from pathlib import Path

# Add parent directory to path for imports when running from external projects
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from qwen_mcp.config.sos_paths import DEFAULT_SOS_PATHS
import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
from qwen_mcp.engines.orchestrator import DecisionLogOrchestrator
from qwen_mcp.engines.io_layer.path_resolver import PathResolver

logger = logging.getLogger(__name__)

class DecisionLogSyncEngine:
    """
    Decision Log Sync Engine - Synchronizes decision_log.parquet with BACKLOG.md and CHANGELOG.md.
    
    Formerly known as SOSSyncEngine. Renamed for clarity.
    """
    
    DEFAULT_DECISION_LOG_PATH = DEFAULT_SOS_PATHS.get_decision_log_path()
    
    def __init__(self, decision_log_path: Optional[Path] = None):
        self.decision_log_path = Path(decision_log_path) if decision_log_path else self.DEFAULT_DECISION_LOG_PATH
        self.lock_path = self.decision_log_path.with_suffix(".lock")
        self._last_mtime = None

    def _acquire_lock(self, timeout: int = 2):
        """Acquire a simple file-based lock with timeout."""
        import time
        start_time = time.time()
        while os.path.exists(str(self.lock_path)):
            if time.time() - start_time > timeout:
                logger.warning(f"Lock timeout after {timeout}s on {self.lock_path}. Possible orphaned lock.")
                raise TimeoutError(f"Locked out of {self.lock_path} for {timeout}s.")
            time.sleep(0.1)
        
        try:
            # Defensive: ensure parent directory exists before touch
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path.touch()
        except Exception as e:
            logger.error(f"Failed to create lock file at {self.lock_path}: {e}")
            raise
        return True

    def has_changes(self) -> bool:
        """Check if the log file has been modified since last check."""
        try:
            if not self.decision_log_path.exists():
                return False
            current_mtime = os.path.getmtime(str(self.decision_log_path))
            if self._last_mtime is None:
                self._last_mtime = current_mtime
                return False  # First check, just establish baseline
            
            changed = current_mtime > self._last_mtime
            if changed:
                self._last_mtime = current_mtime
            return changed
        except Exception:
            return False

    async def poll_and_sync(self, backlog_path: Path, changelog_path: Optional[Path] = None):
        """High-level polling trigger for background automation."""
        if self.has_changes():
            # If changes detected, try to apply all pending advice
            await self.apply_all_advices(backlog_path, changelog_path)

    def _release_lock(self):
        """Release the simple file-based lock safely."""
        try:
            if self.lock_path.exists():
                os.remove(str(self.lock_path))
        except Exception as e:
            logger.debug(f"Lock release warning (non-critical): {e}")

    async def scan_advices(self) -> List[Dict]:
        """Scans for records using a dedicated lock file."""
        if not self.decision_log_path.exists():
            return []

        def read_data():
            lock = self._acquire_lock()
            try:
                # pq.read_table opens and closes the file internally
                table = pq.read_table(str(self.decision_log_path))
                return table.to_pylist()
            finally:
                self._release_lock()

        return await asyncio.to_thread(lambda: [r for r in read_data() if r.get('agentic_advice') and not r.get('patch_applied', False)])
    
    async def query_decisions(
        self,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Query decision_log.parquet with filters.
        
        Args:
            status: Filter by adr_status (e.g., "proposed", "accepted", "implemented")
            task_type: Filter by task_type (e.g., "pending", "completed", "decision")
            date_from: Filter records after this date
            date_to: Filter records before this date
            tags: Filter by tags (must contain at least one of these tags)
            limit: Maximum number of records to return
            
        Returns:
            List of matching decision records
        """
        if not self.decision_log_path.exists():
            return []
        
        def read_and_filter():
            lock = self._acquire_lock()
            try:
                table = pq.read_table(str(self.decision_log_path))
                records = table.to_pylist()
                
                filtered = []
                for record in records:
                    # Filter by status
                    if status and record.get('adr_status') != status:
                        continue
                    
                    # Filter by task_type
                    if task_type and record.get('task_type') != task_type:
                        continue
                    
                    # Filter by date range
                    record_ts = record.get('timestamp')
                    if record_ts:
                        if date_from and record_ts < date_from:
                            continue
                        if date_to and record_ts > date_to:
                            continue
                    
                    # Filter by tags
                    if tags:
                        record_tags = record.get('tags', [])
                        if not any(tag in record_tags for tag in tags):
                            continue
                    
                    filtered.append(record)
                    
                    if len(filtered) >= limit:
                        break
                
                return filtered
            finally:
                self._release_lock()
        
        return await asyncio.to_thread(read_and_filter)
    
    async def get_recent_completions(self, days: int = 7, limit: int = 50) -> List[Dict]:
        """
        Get recently completed tasks for CHANGELOG generation.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of records to return
            
        Returns:
            List of recent completion records
        """
        from datetime import timedelta
        date_from = datetime.now() - timedelta(days=days)
        
        return await self.query_decisions(
            task_type="completed",
            date_from=date_from,
            limit=limit
        )
    
    async def get_pending_tasks(self) -> List[Dict]:
        """
        Get all pending tasks from decision_log.parquet.
        
        Returns:
            List of pending task records
        """
        return await self.query_decisions(task_type="pending")

    async def add_task(
        self,
        task_name: str,
        advice: str,
        backlog_path: Path,
        workspace_root: Optional[str] = None,  # Reserved for future use
        session_id: str = "decision_log_manual",
        decision_type: str = "manual_task",
        complexity: str = "medium",
        tokens_used: int = 0,
        tags: Optional[List[str]] = None,
        risk_score: float = 0.0,
        user_approval: bool = True,
        adr_status: Optional[str] = None,
        adr_context: Optional[str] = None,
        adr_consequences: Optional[str] = None,
        adr_alternatives: Optional[str] = None,
        linked_code_nodes: Optional[List[str]] = None,
        depends_on_adr: Optional[List[str]] = None
    ) -> str:
        """
        Add a new task from natural language to BACKLOG.md and decision_log.parquet.
        
        This is the "Files → Parquet" direction of Decision Log sync.
        
        Args:
            task_name: Human-readable task name (will be used as backlog_ref)
            advice: The agentic advice/recommendation
            backlog_path: Path to BACKLOG.md
            workspace_root: Reserved for future use (default: None)
            session_id: Session identifier (default: "decision_log_manual")
            decision_type: Type of decision (default: "manual_task")
            complexity: Task complexity (default: "medium")
            tokens_used: Token count (default: 0)
            tags: Optional tags list
            risk_score: Risk assessment (default: 0.0)
            user_approval: User approval flag (default: True)
            adr_status: ADR lifecycle status (optional)
            adr_context: ADR context (optional)
            adr_consequences: ADR consequences (optional)
            adr_alternatives: ADR alternatives (optional)
            linked_code_nodes: Linked code file paths (optional)
            depends_on_adr: Dependent ADR IDs (optional)
            
        Returns:
            decision_id: The UUID of the created decision record
        """
        # Input validation
        if not advice or not advice.strip():
            raise ValueError("advice parameter cannot be empty")
        
        if not task_name or not task_name.strip():
            raise ValueError("task_name parameter cannot be empty")
        
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create decision record with ADR fields
        record = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "decision_type": decision_type,
            "task_type": "pending",  # Default for add_task
            "complexity": complexity,
            "tokens_used": tokens_used,
            "content": f"Task: {task_name}",
            "tags": tags or [],
            "backlog_ref": task_name,
            "context_version": "1.0",
            "patch_applied": False,
            "agentic_advice": advice,
            "risk_score": risk_score,
            "validator_triggers": [],
            "user_approval": user_approval,
            "rationale": f"Manually added task: {task_name}",
            # ADR Extension Fields
            "adr_status": adr_status,
            "adr_context": adr_context,
            "adr_consequences": adr_consequences,
            "adr_alternatives": adr_alternatives,
            "linked_code_nodes": linked_code_nodes or [],
            "depends_on_adr": depends_on_adr or []
        }
        
        # Write to parquet
        def write_parquet():
            lock = self._acquire_lock()
            try:
                # Read existing or create empty
                if self.decision_log_path.exists():
                    table = pq.read_table(str(self.decision_log_path))
                    records = table.to_pylist()
                else:
                    records = []
                
                # Add new record
                records.append(record)
                
                # Write back
                from decision_log.decision_schema import DECISION_LOG_SCHEMA
                fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(self.decision_log_path.parent))
                os.close(fd)
                updated_table = pa.Table.from_pylist(records, schema=DECISION_LOG_SCHEMA)
                pq.write_table(updated_table, temp_path)
                
                os.replace(temp_path, str(self.decision_log_path))
            finally:
                self._release_lock()
        
        await asyncio.to_thread(write_parquet)
        
        # Add task to BACKLOG.md with checkbox - find "## Pending" or "## Faza urgent" section
        backlog_path = Path(backlog_path)
        if not backlog_path.exists():
            backlog_path.parent.mkdir(parents=True, exist_ok=True)
            backlog_content = "# BACKLOG\n\n## Pending\n\n"
        else:
            with open(backlog_path, 'r', encoding='utf-8') as f:
                backlog_content = f.read()
        
        # Add task with checkbox
        # Sanitize task_name to prevent markdown injection and path traversal
        safe_task_name = task_name.replace('\n', ' ').replace('\r', ' ')[:200]
        task_line = f"- [ ] {safe_task_name} - {decision_id}\n"
        
        # Try to insert after a pending section header
        pending_patterns = [
            r'(## Faza urgent\n)',
            r'(## Pending\n)',
            r'(## 🔧 Pending\n)',
        ]
        
        inserted = False
        for pattern in pending_patterns:
            match = re.search(pattern, backlog_content)
            if match:
                insert_pos = match.end()
                backlog_content = backlog_content[:insert_pos] + "\n" + task_line + backlog_content[insert_pos:]
                inserted = True
                break
        
        # If no pending section found, append to end
        if not inserted:
            if "## Pending" not in backlog_content and "## Faza urgent" not in backlog_content:
                backlog_content += "\n## Pending\n\n" + task_line
            else:
                backlog_content += "\n" + task_line
        
        # Atomic write to prevent corruption on crash
        fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(backlog_path.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(backlog_content)
            os.replace(tmp_path, str(backlog_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        
        logger.info(f"Added task '{task_name}' with decision_id {decision_id} to BACKLOG.md and parquet")
        return decision_id

    async def add_tasks(
        self,
        tasks: List[Dict[str, Any]],
        backlog_path: Path,
        workspace_root: Optional[str] = None,
        session_id: str = "decision_log_manual",
        decision_type: str = "manual_task",
        chunk_size: int = 20,
        user_approval: bool = True,
        adr_status: Optional[str] = None,
        adr_context: Optional[str] = None,
        adr_consequences: Optional[str] = None,
        adr_alternatives: Optional[str] = None,
        linked_code_nodes: Optional[List[str]] = None,
        depends_on_adr: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple tasks in batch with chunking to avoid MCP timeout.
        
        Args:
            tasks: List of task dictionaries with keys:
                - task_name (required): Name of the task
                - advice (required): Agentic advice for the task
                - complexity (optional): Task complexity, default "medium"
                - tags (optional): List of tags
                - risk_score (optional): Risk score, default 0.0
            backlog_path: Path to BACKLOG.md
            workspace_root: Optional workspace root path
            session_id: Session identifier
            decision_type: Type of decision record
            chunk_size: Number of tasks to process per chunk
            user_approval: Whether user has approved the tasks
            adr_status: ADR status if applicable
            adr_context: ADR context if applicable
            adr_consequences: ADR consequences if applicable
            adr_alternatives: ADR alternatives if applicable
            linked_code_nodes: List of linked code nodes
            depends_on_adr: List of dependent ADR IDs
            
        Returns:
            List of decision_ids for all added tasks
        """
        if not tasks:
            return []
        
        all_decision_ids: List[str] = []
        all_records: List[Dict[str, Any]] = []
        
        # Process tasks in chunks to avoid MCP timeout
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            chunk_records = []
            chunk_decision_ids = []
            
            for task in chunk:
                task_name = task.get("task_name")
                advice = task.get("advice")
                
                if not task_name or not advice:
                    logger.warning(f"Skipping task missing required fields: {task}")
                    continue
                
                complexity = task.get("complexity", "medium")
                tags = task.get("tags")
                risk_score = task.get("risk_score", 0.0)
                
                # Generate decision ID and timestamp
                decision_id = str(uuid.uuid4())
                timestamp = datetime.now(timezone.utc)
                
                # Create decision record with same schema as add_task
                record = {
                    "decision_id": decision_id,
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "decision_type": decision_type,
                    "task_type": "pending",
                    "complexity": complexity,
                    "tokens_used": 0,
                    "content": f"Task: {task_name}",
                    "tags": tags or [],
                    "backlog_ref": task_name,
                    "context_version": "1.0",
                    "patch_applied": False,
                    "agentic_advice": advice,
                    "risk_score": float(risk_score),
                    "validator_triggers": [],
                    "user_approval": user_approval,
                    "rationale": f"Batch added task: {task_name}",
                    "adr_status": adr_status,
                    "adr_context": adr_context,
                    "adr_consequences": adr_consequences,
                    "adr_alternatives": adr_alternatives,
                    "linked_code_nodes": linked_code_nodes or [],
                    "depends_on_adr": depends_on_adr or []
                }
                
                chunk_records.append(record)
                chunk_decision_ids.append(decision_id)
            
            # Write chunk atomically with lock
            if chunk_records:
                def write_chunk():
                    lock = self._acquire_lock(timeout=5)
                    try:
                        # Read existing or create empty
                        if self.decision_log_path.exists():
                            table = pq.read_table(str(self.decision_log_path))
                            records = table.to_pylist()
                        else:
                            records = []
                        
                        # Append chunk records
                        records.extend(chunk_records)
                        
                        # Atomic write using tempfile + os.replace
                        from decision_log.decision_schema import DECISION_LOG_SCHEMA
                        fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(self.decision_log_path.parent))
                        os.close(fd)
                        updated_table = pa.Table.from_pylist(records, schema=DECISION_LOG_SCHEMA)
                        pq.write_table(updated_table, temp_path)
                        os.replace(temp_path, str(self.decision_log_path))
                    finally:
                        self._release_lock()
                
                await asyncio.to_thread(write_chunk)
                
                all_decision_ids.extend(chunk_decision_ids)
                all_records.extend(chunk_records)
                
                logger.info(f"Added {len(chunk_records)} tasks in chunk ({i // chunk_size + 1})")
                
                # Small delay between chunks to avoid timeout
                if i + chunk_size < len(tasks):
                    await asyncio.sleep(0.1)
        
        # Update BACKLOG.md once at the end with all tasks
        if all_records:
            await self._add_tasks_to_backlog(all_records, backlog_path)
        
        return all_decision_ids

    async def _add_tasks_to_backlog(
        self,
        records: List[Dict[str, Any]],
        backlog_path: Path
    ) -> None:
        """
        Update BACKLOG.md with multiple tasks at once.
        
        Args:
            records: List of decision records to add to backlog
            backlog_path: Path to BACKLOG.md file
        """
        if not records:
            return
        
        try:
            # Read existing backlog content or create new
            if not backlog_path.exists():
                backlog_path.parent.mkdir(parents=True, exist_ok=True)
                backlog_content = "# BACKLOG\n\n## Pending\n\n"
            else:
                with open(backlog_path, "r", encoding="utf-8") as f:
                    backlog_content = f.read()
            
            # Find "## Pending" section
            pending_match = re.search(r'^## Pending\s*$', backlog_content, re.MULTILINE)
            if not pending_match:
                # Create Pending section
                backlog_content += "\n## Pending\n\n"
                insert_pos = len(backlog_content)
            else:
                insert_pos = pending_match.end()
            
            # Generate task entries
            tasks_content = ""
            for record in records:
                task_name = record.get("backlog_ref", "Unknown Task")
                decision_id = record.get("decision_id", "unknown")
                
                # Sanitize task_name
                safe_name = task_name.replace('\n', ' ').replace('\r', ' ')[:200]
                tasks_content += f"- [ ] {safe_name} - {decision_id}\n"
            
            # Insert tasks after ## Pending
            new_content = backlog_content[:insert_pos] + "\n" + tasks_content + backlog_content[insert_pos:]
            
            # Atomic write
            fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(backlog_path.parent))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                os.replace(tmp_path, str(backlog_path))
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            
            logger.info(f"Updated BACKLOG.md with {len(records)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to update BACKLOG.md: {e}")
            raise

    async def log_completed_task(
        self,
        task_description: str,
        session_id: str,
        tokens_used: int = 0,
        files_changed: Optional[List[str]] = None,
        complexity: str = "medium",
        tags: Optional[List[str]] = None,
        adr_status: Optional[str] = "implemented"
    ) -> str:
        """
        Log a completed task to decision_log.parquet.
        
        This is called automatically after qwen_coder finishes successfully.
        Creates a record with task_type="completed".
        
        Args:
            task_description: Description of the completed task
            session_id: Session/project identifier
            tokens_used: Number of tokens consumed
            files_changed: List of file paths that were modified
            complexity: Task complexity level
            tags: Optional tags for categorization
            adr_status: ADR status (default: "implemented")
            
        Returns:
            decision_id: The UUID of the created decision record
        """
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create decision record for completed task
        record = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "decision_type": "coder_completion",
            "task_type": "completed",  # Mark as completed work
            "complexity": complexity,
            "tokens_used": tokens_used,
            "content": f"Completed: {task_description}",
            "tags": tags or ["coder", "completed"],
            "backlog_ref": task_description[:100],  # Truncate for ref
            "context_version": "1.0",
            "patch_applied": True,
            "agentic_advice": None,  # No advice - this is a completion record
            "risk_score": 0.0,
            "validator_triggers": [],
            "user_approval": True,
            "rationale": f"Automatically logged after qwen_coder completion: {task_description}",
            # ADR Extension Fields
            "adr_status": adr_status,
            "adr_context": None,
            "adr_consequences": None,
            "adr_alternatives": None,
            "linked_code_nodes": files_changed or [],
            "depends_on_adr": []
        }
        
        # Write to parquet
        def write_parquet():
            lock = self._acquire_lock()
            try:
                if self.decision_log_path.exists():
                    table = pq.read_table(str(self.decision_log_path))
                    records = table.to_pylist()
                else:
                    records = []
                
                records.append(record)
                
                from decision_log.decision_schema import DECISION_LOG_SCHEMA
                fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(self.decision_log_path.parent))
                os.close(fd)
                updated_table = pa.Table.from_pylist(records, schema=DECISION_LOG_SCHEMA)
                pq.write_table(updated_table, temp_path)
                
                os.replace(temp_path, str(self.decision_log_path))
            finally:
                self._release_lock()
        
        await asyncio.to_thread(write_parquet)
        logger.info(f"Logged completed task '{task_description[:50]}...' with decision_id {decision_id}")
        return decision_id

    async def log_decision(
        self,
        decision_description: str,
        session_id: str,
        blueprint_ref: Optional[str] = None,
        adr_link: Optional[str] = None,
        tokens_used: int = 0,
        complexity: str = "high",
        tags: Optional[List[str]] = None,
        adr_status: str = "proposed"
    ) -> str:
        """
        Log an architectural decision to decision_log.parquet.
        
        This is called automatically after qwen_architect generates a blueprint.
        Creates a record with task_type="decision".
        
        Args:
            decision_description: Description of the architectural decision
            session_id: Session/project identifier
            blueprint_ref: Reference to generated blueprint
            adr_link: Path to ADR file if created
            tokens_used: Number of tokens consumed
            complexity: Task complexity level
            tags: Optional tags for categorization
            adr_status: ADR status (default: "proposed")
            
        Returns:
            decision_id: The UUID of the created decision record
        """
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create decision record for architectural decision
        record = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "decision_type": "architect_decision",
            "task_type": "decision",  # Mark as architectural decision
            "complexity": complexity,
            "tokens_used": tokens_used,
            "content": f"Architecture Decision: {decision_description}",
            "tags": tags or ["architect", "blueprint"],
            "backlog_ref": decision_description[:100],
            "context_version": "1.0",
            "patch_applied": False,
            "agentic_advice": None,
            "risk_score": 0.0,
            "validator_triggers": [],
            "user_approval": True,
            "rationale": f"Automatically logged after qwen_architect: {decision_description}",
            # ADR Extension Fields
            "adr_status": adr_status,
            "adr_context": blueprint_ref,
            "adr_consequences": None,
            "adr_alternatives": None,
            "linked_code_nodes": [adr_link] if adr_link else [],
            "depends_on_adr": []
        }
        
        # Write to parquet
        def write_parquet():
            lock = self._acquire_lock()
            try:
                if self.decision_log_path.exists():
                    table = pq.read_table(str(self.decision_log_path))
                    records = table.to_pylist()
                else:
                    records = []
                
                records.append(record)
                
                from decision_log.decision_schema import DECISION_LOG_SCHEMA
                fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(self.decision_log_path.parent))
                os.close(fd)
                updated_table = pa.Table.from_pylist(records, schema=DECISION_LOG_SCHEMA)
                pq.write_table(updated_table, temp_path)
                
                os.replace(temp_path, str(self.decision_log_path))
            finally:
                self._release_lock()
        
        await asyncio.to_thread(write_parquet)
        
        # Also add to BACKLOG.md as a decision record (not pending task)
        if blueprint_ref or adr_link:
            await self._log_decision_to_backlog(
                decision_description, decision_id, blueprint_ref, adr_link
            )
        
        logger.info(f"Logged architectural decision '{decision_description[:50]}...' with decision_id {decision_id}")
        return decision_id

    async def _log_decision_to_backlog(
        self,
        decision_description: str,
        decision_id: str,
        blueprint_ref: Optional[str],
        adr_link: Optional[str]
    ) -> None:
        """Helper to log architectural decisions to BACKLOG.md under a Decisions section."""
        backlog_path = Path(".PLAN/BACKLOG.md")
        
        if not backlog_path.exists():
            backlog_path.parent.mkdir(parents=True, exist_ok=True)
            backlog_content = "# BACKLOG\n\n## Decisions\n\n"
        else:
            with open(backlog_path, 'r', encoding='utf-8') as f:
                backlog_content = f.read()
        
        # Add decision entry
        refs = []
        if blueprint_ref:
            refs.append(f"blueprint: {blueprint_ref}")
        if adr_link:
            refs.append(f"adr: {adr_link}")
        ref_str = f" ({', '.join(refs)})" if refs else ""
        
        decision_line = f"- [x] DECISION: {decision_description[:200]}{ref_str} - {decision_id}\n"
        
        # Try to insert after "## Decisions" section
        decision_pattern = r'(## Decisions\n)'
        match = re.search(decision_pattern, backlog_content)
        if match:
            insert_pos = match.end()
            backlog_content = backlog_content[:insert_pos] + "\n" + decision_line + backlog_content[insert_pos:]
        else:
            # No Decisions section - create one
            backlog_content += "\n## Decisions\n\n" + decision_line
        
        # Atomic write
        fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(backlog_path.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(backlog_content)
            os.replace(tmp_path, str(backlog_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    async def complete_task(
        self,
        task_description: str,
        backlog_path: Path,
        changelog_path: Path,
        session_id: str,
        tokens_used: int = 0,
        files_changed: Optional[List[str]] = None,
        archive_completed: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Complete a task: remove from BACKLOG.md, log to parquet, append to CHANGELOG.md.
        
        This is the main entry point for qwen_coder auto-logging.
        
        Args:
            task_description: Description of the completed task (from prompt)
            backlog_path: Path to BACKLOG.md
            changelog_path: Path to CHANGELOG.md
            session_id: Session/project identifier
            tokens_used: Number of tokens consumed
            files_changed: List of file paths that were modified
            archive_completed: If True, removes task from BACKLOG.md (default).
                              If False, only marks as [x] without removing.
            
        Returns:
            Tuple of (success, matched_decision_id or None)
        """
        # 1. Find matching task in BACKLOG.md using fuzzy match
        matched_id, matched_task = self._find_matching_task(backlog_path, task_description)
        
        # 2. Remove from BACKLOG.md if match found (archive_completed=True)
        if matched_id:
            if archive_completed:
                self._remove_from_backlog(backlog_path, matched_id)
                logger.info(f"Removed task {matched_id} from BACKLOG.md (archived)")
            else:
                self._mark_task_completed(backlog_path, matched_id)
                logger.info(f"Marked task {matched_id} as completed in BACKLOG.md")
        
        # 3. Log to parquet
        decision_id = await self.log_completed_task(
            task_description=task_description,
            session_id=session_id,
            tokens_used=tokens_used,
            files_changed=files_changed,
            tags=["coder", "auto-logged"]
        )
        
        # 4. Append to CHANGELOG.md with full details from parquet
        self._append_changelog(
            changelog_path,
            decision_id,
            task_description,
            matched_id,
            matched_task_description=matched_task,
            files_changed=files_changed,
            tokens_used=tokens_used
        )
        logger.info(f"Appended completion to CHANGELOG.md for {decision_id}")
        
        return (True, matched_id)

    def _find_matching_task(self, backlog_path: Path, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find a matching task in BACKLOG.md using fuzzy text matching.
        
        Uses simple keyword overlap heuristic (no external dependencies).
        
        Args:
            backlog_path: Path to BACKLOG.md
            prompt: The qwen_coder prompt to match
            
        Returns:
            Tuple of (decision_id or None, task_description or None)
        """
        if not backlog_path.exists():
            return (None, None)
        
        with open(backlog_path, 'r', encoding='utf-8') as f:
            backlog_content = f.read()
        
        # Find all pending tasks: - [ ] TASK - ID
        pending_pattern = r'- \[ \]\s*(.+?)\s*-\s*([a-f0-9-]+)'
        pending_tasks = re.findall(pending_pattern, backlog_content)
        
        if not pending_tasks:
            return (None, None)
        
        # Normalize prompt for matching
        prompt_keywords = set(prompt.lower().split())
        
        best_match = None
        best_score = 0
        
        for task_desc, decision_id in pending_tasks:
            # Calculate keyword overlap
            task_keywords = set(task_desc.lower().split())
            overlap = len(prompt_keywords & task_keywords)
            score = overlap / max(len(prompt_keywords), 1)
            
            if score > best_score and score >= 0.3:  # 30% overlap threshold
                best_score = score
                best_match = (decision_id, task_desc)
        
        return best_match if best_match else (None, None)

    def _remove_from_backlog(self, backlog_path: Path, decision_id: str) -> None:
        """Remove a completed task from BACKLOG.md entirely (not just mark [x])."""
        if not backlog_path.exists():
            return
        
        with open(backlog_path, 'r', encoding='utf-8') as f:
            backlog_content = f.read()
        
        # Find the task line with decision_id
        pattern = rf'^\s*-\s*\[[ x]\]\s*.*?-\s*{re.escape(decision_id)}.*?(?=\n|$)'
        matches = list(re.finditer(pattern, backlog_content, re.MULTILINE))
        
        if matches:
            # Remove all matched tasks
            for match in reversed(matches):  # Reverse to preserve indices
                # Include trailing newline if present
                end_pos = match.end()
                if end_pos < len(backlog_content) and backlog_content[end_pos] == '\n':
                    end_pos += 1
                backlog_content = backlog_content[:match.start()] + backlog_content[end_pos:]
            
            # Clean up extra blank lines
            backlog_content = re.sub(r'\n{3,}', '\n\n', backlog_content)
            
            # Atomic write
            fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(backlog_path.parent))
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(backlog_content)
                os.replace(tmp_path, str(backlog_path))
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
    
    def _mark_task_completed(self, decision_id: str) -> bool:
        """
        Mark a task as completed by delegating to DecisionLogOrchestrator.

        Args:
            decision_id: The UUID of the decision to mark as completed

        Returns:
            bool: True if successful, False otherwise
        """
        if hasattr(self, 'decision_log_path') and self.decision_log_path:
            workspace_root = self.decision_log_path.parent.parent
        else:
            workspace_root = Path.cwd()

        resolver = PathResolver(workspace_root=workspace_root)
        orchestrator = DecisionLogOrchestrator(path_resolver=resolver)

        return orchestrator.archive_task(decision_id)

    def _append_changelog(
        self,
        changelog_path: Path,
        decision_id: str,
        task_description: str,
        matched_task_id: Optional[str] = None,
        matched_task_description: Optional[str] = None,
        files_changed: Optional[List[str]] = None,
        tokens_used: int = 0
    ) -> None:
        """
        Append a completion entry to CHANGELOG.md via Orchestrator.
        
        Args:
            changelog_path: Path to CHANGELOG.md (backward compatible, not used for IO)
            decision_id: The UUID of the completed decision
            task_description: Description of the completed task
            matched_task_id: Original decision_id from BACKLOG.md (if matched)
            matched_task_description: Original task description from BACKLOG.md
            files_changed: List of files that were modified
            tokens_used: Number of tokens consumed
        """
        # Build changelog entry with full details
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        entry = f"## {timestamp} - {decision_id}\n\n"
        
        # Task reference (link to original backlog item if exists)
        if matched_task_id and matched_task_description:
            entry += f"**Original Task**: {matched_task_description}  \n"
            entry += f"**Decision ID**: `{matched_task_id}` → `{decision_id}`\n\n"
        else:
            entry += f"**Task**: {task_description}\n\n"
        
        # Files modified
        if files_changed:
            entry += "**Files Modified**:\n"
            for file_path in files_changed:
                entry += f"- `{file_path}`\n"
            entry += "\n"
        
        # Token usage
        if tokens_used > 0:
            entry += f"**Tokens Used**: {tokens_used:,}\n\n"
        
        entry += f"**Status**: ✅ Completed\n\n"
        entry += "---\n\n"
        
        # Delegate IO to Orchestrator
        from qwen_mcp.engines.io_layer.path_resolver import PathResolver
        from qwen_mcp.engines.orchestrator import DecisionLogOrchestrator
        
        if hasattr(self, 'decision_log_path') and self.decision_log_path:
            workspace_root = self.decision_log_path.parent.parent
        else:
            workspace_root = Path.cwd()
        
        resolver = PathResolver(workspace_root=workspace_root)
        orchestrator = DecisionLogOrchestrator(path_resolver=resolver)
        orchestrator.append_to_changelog(entry)

    async def apply_advice(self, decision_id: str, backlog_path: Path, changelog_path: Optional[Path] = None):
        """Applies a single advice by ID."""
        return await self._apply_internal([decision_id], backlog_path, changelog_path)

    async def apply_all_advices(self, backlog_path: Path, changelog_path: Optional[Path] = None):
        """Applies all pending advices in one go."""
        pending = await self.scan_advices()
        ids = [p['decision_id'] for p in pending]
        if not ids:
            return False
        return await self._apply_internal(ids, backlog_path, changelog_path)

    def _apply_advices_to_files(self, backlog_path: Path, changelog_path: Path, advices: List[Dict]) -> Tuple[bool, bool]:
        """
        Apply advices to BACKLOG.md and CHANGELOG.md.
        
        This method:
        1. Archives corresponding tasks in BACKLOG.md
        2. Moves completed items to CHANGELOG.md via Orchestrator
        
        Args:
            backlog_path: Path to BACKLOG.md (backward compatible)
            changelog_path: Path to CHANGELOG.md (backward compatible)
            advices: List of advice records with decision_id, agentic_advice, backlog_ref
            
        Returns:
            Tuple of (backlog_updated, changelog_updated)
        """
        backlog_updated = False
        changelog_updated = False
        
        if not advices:
            return False, False
            
        if hasattr(self, 'decision_log_path') and self.decision_log_path:
            workspace_root = self.decision_log_path.parent.parent
        else:
            workspace_root = Path.cwd()

        # Import locally to avoid circular imports during init if paths differ
        from qwen_mcp.engines.io_layer.path_resolver import PathResolver
        from qwen_mcp.engines.orchestrator import DecisionLogOrchestrator
        
        resolver = PathResolver(workspace_root=workspace_root)
        orchestrator = DecisionLogOrchestrator(path_resolver=resolver)
        
        changelog_entries = []
        for advice in advices:
            decision_id = advice.get('decision_id', '')
            agentic_advice = advice.get('agentic_advice', '')
            backlog_ref = advice.get('backlog_ref', '')
            
            # 1. Archive task via orchestrator
            if decision_id:
                if orchestrator.archive_task(decision_id):
                    backlog_updated = True
                    logger.info(f"Marked task with decision_id '{decision_id}' as completed via Orchestrator")
            elif backlog_ref:
                # Orchestrator uses decision_id, fallback to backlog_ref if decision_id is missing
                if orchestrator.archive_task(backlog_ref):
                    backlog_updated = True
                    logger.info(f"Marked task '{backlog_ref}' as completed via Orchestrator")
            
            # 2. Build changelog entry format identical to old system
            timestamp = advice.get('timestamp', '')
            if timestamp:
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = str(timestamp)
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            entry = f"## [{timestamp}] {decision_id}\n\n"
            if backlog_ref:
                entry += f"**Task**: {backlog_ref}\n\n"
            entry += f"**Advice**: {agentic_advice}\n\n"
            entry += "---\n\n"
            changelog_entries.append(entry)
            
        if changelog_entries:
            # Sync cluster mega-entry directly delegates to append_to_changelog
            mega_entry = f"## SOS Sync - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            mega_entry += "".join(changelog_entries)
            
            # SectionManager.append_to_changelog automatically inserts below the '# CHANGELOG' header
            orchestrator.append_to_changelog(mega_entry)
            changelog_updated = True
            logger.info(f"Appended {len(changelog_entries)} entries to CHANGELOG.md via Orchestrator")
            
        return backlog_updated, changelog_updated

    async def _apply_internal(self, decision_ids: List[str], backlog_path: Path, changelog_path: Optional[Path] = None):
        """Internal shared logic for applying one or more advices."""
        def process_atomic():
            lock = self._acquire_lock()
            try:
                # 1. Read
                table = pq.read_table(str(self.decision_log_path))
                records = table.to_pylist()
                
                # 2. Update and collect content
                id_set = set(decision_ids)
                applied_records = []
                for r in records:
                    if r.get('decision_id') in id_set:
                        r['patch_applied'] = True
                        applied_records.append(r)
                
                if not applied_records:
                    return None, []
                
                # 3. Write Updated Parquet
                from decision_log.decision_schema import DECISION_LOG_SCHEMA
                updated_table = pa.Table.from_pylist(records, schema=DECISION_LOG_SCHEMA)
                fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(self.decision_log_path.parent))
                os.close(fd)
                pq.write_table(updated_table, temp_path)
                
                os.replace(temp_path, str(self.decision_log_path))
                
                return applied_records, applied_records
            finally:
                self._release_lock()

        result, applied_records = await asyncio.to_thread(process_atomic)
        
        if not result:
            return False
        
        # Update BACKLOG.md and CHANGELOG.md
        backlog_path = Path(backlog_path)
        backlog_path.parent.mkdir(parents=True, exist_ok=True)
        
        changelog_path = Path(changelog_path) if changelog_path else None
        
        # Use the new file management method
        if applied_records:
            backlog_updated, changelog_updated = self._apply_advices_to_files(
                backlog_path=backlog_path,
                changelog_path=changelog_path or (backlog_path.parent / "CHANGELOG.md"),
                advices=applied_records
            )
            
            logger.info(f"Decision Log Sync complete: BACKLOG updated={backlog_updated}, CHANGELOG updated={changelog_updated}")
            return True
        
        return False
