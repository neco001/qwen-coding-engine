import os
import sys
import asyncio
import logging
import re
import uuid
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

logger = logging.getLogger(__name__)

class SOSSyncEngine:
    def __init__(self, decision_log_path: Path):
        self.decision_log_path = Path(decision_log_path)
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
            self.lock_path.touch()
        except Exception as e:
            logger.error(f"Failed to create lock file: {e}")
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

    async def add_task(
        self,
        task_name: str,
        advice: str,
        backlog_path: Path,
        workspace_root: Optional[str] = None,  # Reserved for future use
        session_id: str = "sos_manual",
        decision_type: str = "manual_task",
        complexity: str = "medium",
        tokens_used: int = 0,
        tags: Optional[List[str]] = None,
        risk_score: float = 0.0,
        user_approval: bool = True
    ) -> str:
        """
        Add a new task from natural language to BACKLOG.md and decision_log.parquet.
        
        This is the "Files → Parquet" direction of SOS sync.
        
        Args:
            task_name: Human-readable task name (will be used as backlog_ref)
            advice: The agentic advice/recommendation
            backlog_path: Path to BACKLOG.md
            workspace_root: Reserved for future use (default: None)
            session_id: Session identifier (default: "sos_manual")
            decision_type: Type of decision (default: "manual_task")
            complexity: Task complexity (default: "medium")
            tokens_used: Token count (default: 0)
            tags: Optional tags list
            risk_score: Risk assessment (default: 0.0)
            user_approval: User approval flag (default: True)
            
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
        
        # Create decision record
        record = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "decision_type": decision_type,
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
            "rationale": f"Manually added task: {task_name}"
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
        
        # Add task to BACKLOG.md with checkbox
        backlog_path = Path(backlog_path)
        if not backlog_path.exists():
            backlog_path.parent.mkdir(parents=True, exist_ok=True)
            backlog_content = "# BACKLOG\n\n## Pending\n\n"
        else:
            with open(backlog_path, 'r', encoding='utf-8') as f:
                backlog_content = f.read()
        
        # Add task with checkbox - find "## Pending" or "## Faza urgent" section
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
        1. Marks corresponding tasks in BACKLOG.md as completed [x]
        2. Moves completed items to CHANGELOG.md
        
        Args:
            backlog_path: Path to BACKLOG.md
            changelog_path: Path to CHANGELOG.md
            advices: List of advice records with decision_id, agentic_advice, backlog_ref
            
        Returns:
            Tuple of (backlog_updated, changelog_updated)
        """
        backlog_updated = False
        changelog_updated = False
        
        # Read BACKLOG.md
        if not backlog_path.exists():
            logger.warning(f"BACKLOG.md not found at {backlog_path}")
            return False, False
            
        with open(backlog_path, 'r', encoding='utf-8') as f:
            backlog_content = f.read()
        
        # Read CHANGELOG.md (create if doesn't exist)
        if changelog_path.exists():
            with open(changelog_path, 'r', encoding='utf-8') as f:
                changelog_content = f.read()
        else:
            changelog_content = "# CHANGELOG\n\n"
            changelog_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process each advice
        changelog_entries = []
        for advice in advices:
            decision_id = advice.get('decision_id', '')
            agentic_advice = advice.get('agentic_advice', '')
            backlog_ref = advice.get('backlog_ref', '')
            
            # Mark task as completed in BACKLOG.md
            # Priority: match by backlog_ref first, fall back to decision_id
            matched = False
            if backlog_ref:
                pattern = rf'(- \[ \]\s*.*?{re.escape(backlog_ref)}.*?(?=\n|$))'
                match = re.search(pattern, backlog_content)
                if match:
                    old_line = match.group(1)
                    new_line = old_line.replace('- [ ]', '- [x]', 1)
                    backlog_content = backlog_content.replace(old_line, new_line)
                    backlog_updated = True
                    matched = True
                    logger.info(f"Marked task '{backlog_ref}' as completed in BACKLOG.md")
            
            # Only try decision_id match if backlog_ref didn't match
            if not matched and decision_id:
                pattern = rf'(- \[ \]\s*.*?{re.escape(decision_id)}.*?(?=\n|$))'
                match = re.search(pattern, backlog_content)
                if match:
                    old_line = match.group(1)
                    new_line = old_line.replace('- [ ]', '- [x]', 1)
                    backlog_content = backlog_content.replace(old_line, new_line)
                    backlog_updated = True
                    logger.info(f"Marked task with decision_id '{decision_id}' as completed in BACKLOG.md")
            
            # Prepare changelog entry
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
        
        # Write updated BACKLOG.md
        if backlog_updated:
            with open(backlog_path, 'w', encoding='utf-8') as f:
                f.write(backlog_content)
            logger.info(f"Updated BACKLOG.md with {len(advices)} completed tasks")
        
        # Write updated CHANGELOG.md (prepend new entries)
        if changelog_entries:
            new_changelog = "# CHANGELOG\n\n"
            new_changelog += f"## SOS Sync - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            new_changelog += ''.join(changelog_entries)
            
            # Append existing content, preserving all entries
            if changelog_content:
                # Skip header if present, preserve everything else
                if changelog_content.startswith("# CHANGELOG\n\n"):
                    existing_content = changelog_content[len("# CHANGELOG\n\n"):]
                else:
                    existing_content = changelog_content
                new_changelog += existing_content
            
            with open(changelog_path, 'w', encoding='utf-8') as f:
                f.write(new_changelog)
            changelog_updated = True
            logger.info(f"Updated CHANGELOG.md with {len(changelog_entries)} entries")
        
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
            
            logger.info(f"SOS Sync complete: BACKLOG updated={backlog_updated}, CHANGELOG updated={changelog_updated}")
            return True
        
        return False
