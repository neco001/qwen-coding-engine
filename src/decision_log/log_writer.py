import os
import sys
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .decision_schema import DecisionSchema, DECISION_LOG_SCHEMA

class DecisionLogWriter:
    def __init__(self, log_path: Path, backup_dir: Optional[Path] = None, plan_dir: Optional[Path] = None):
        """Initialize Decision Log Writer.
        
        Args:
            log_path: Path to decision_log.parquet
            backup_dir: Path to backup directory (APPDATA)
            plan_dir: Path to project .PLAN/ directory
        """
        self.log_path = Path(log_path)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.plan_dir = Path(plan_dir) if plan_dir else None
        
        # Ensure directories exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        if self.plan_dir:
            self.plan_dir.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            self._initialize_empty_log()
    
    def _initialize_empty_log(self) -> None:
        """Create empty log file matching schema."""
        empty_table = DecisionSchema.create_empty_table()
        # Ensure path string for pq.write_table
        pq.write_table(empty_table, str(self.log_path))
    
    def _acquire_lock(self, file_path: Path) -> None:
        """Acquire OS-level lock on the file."""
        if not file_path.exists():
            file_path.touch()
        
        if sys.platform == "win32":
            import msvcrt
            f = open(file_path, "r+b")
            # Lock the first byte
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            self._lock_file_handle = f
        else:
            import fcntl
            f = open(file_path, "r+b")
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            self._lock_file_handle = f

    def _release_lock(self, file_path: Path) -> None:
        """Release OS-level lock on the file."""
        if hasattr(self, "_lock_file_handle") and self._lock_file_handle:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file_handle.fileno(), fcntl.LOCK_UN)
            self._lock_file_handle.close()
            self._lock_file_handle = None

    async def write_decision(self, decision: Dict[str, Any]) -> str:
        """Write a decision record to the log atomically."""
        if not DecisionSchema.validate_record(decision):
            raise ValueError("Invalid decision record")
        
        # Ensure timestamp is datetime object and fill missing SOS fields
        if "timestamp" not in decision or not isinstance(decision["timestamp"], datetime):
            decision["timestamp"] = datetime.now()

        # Schema defaults/handling
        defaults = {
            "decision_id": f"dec-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "session_id": "default",
            "decision_type": "unknown",
            "complexity": "medium",
            "tokens_used": 0,
            "content": "",
            "tags": [],
            "backlog_ref": "",
            "context_version": "1.0.0",
            "patch_applied": False,
            "agentic_advice": "",
            "risk_score": 0.0,
            "validator_triggers": [],
            "user_approval": False,
            "rationale": ""
        }
        for key, default_val in defaults.items():
            if key not in decision or decision[key] is None:
                decision[key] = default_val

        lock_acquired = False
        lock_file = self.log_path.with_suffix(".lock")
        
        try:
            # Atomic lock acquisition (thread-safe for msvcrt/fcntl)
            await asyncio.to_thread(self._acquire_lock, lock_file)
            lock_acquired = True
            
            # Read existing data
            if self.log_path.exists():
                existing_table = pq.read_table(str(self.log_path))
            else:
                existing_table = DecisionSchema.create_empty_table()

            # Prepare new record matching schema exactly using name order from DECISION_LOG_SCHEMA
            record_data = {k: [decision.get(k)] for k in DECISION_LOG_SCHEMA.names}
            new_table = pa.table(record_data, schema=DECISION_LOG_SCHEMA)

            # Append and write atomically
            combined_table = pa.concat_tables([existing_table, new_table])
            
            # Write to temp file then rename
            temp_fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(self.log_path.parent))
            os.close(temp_fd)
            pq.write_table(combined_table, temp_path)
            
            if self.log_path.exists():
                self.log_path.unlink()
            os.rename(temp_path, str(self.log_path))

            # Optional Backup
            if self.backup_dir:
                await self._backup_to_appdata(combined_table.to_pandas())
                
            return decision.get("decision_id", "")
        finally:
            if lock_acquired:
                await asyncio.to_thread(self._release_lock, lock_file)

    async def _backup_to_appdata(self, df: pd.DataFrame) -> None:
        """Backup log to APPDATA."""
        if not self.backup_dir: return
        backup_path = self.backup_dir / "decision_log_backup.parquet"
        df.to_parquet(str(backup_path))
