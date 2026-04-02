"""Decision Log Writer with atomic writes and file locking."""

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
    """Writer for Decision Log with atomic writes and file locking.
    
    Features:
    - Atomic writes using tempfile + rename pattern
    - File locking (msvcrt on Windows, fcntl on Unix)
    - Backup to APPDATA directory
    - Read/query operations
    """
    
    def __init__(
        self,
        log_path: Path,
        backup_dir: Optional[Path] = None,
    ):
        """Initialize Decision Log Writer.
        
        Args:
            log_path: Path to decision_log.parquet
            backup_dir: Path to backup directory (APPDATA)
        """
        self.log_path = Path(log_path)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        
        # Ensure directories exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty parquet if not exists
        if not self.log_path.exists():
            self._initialize_empty_log()
    
    def _initialize_empty_log(self) -> None:
        """Create empty parquet file with schema."""
        empty_table = DecisionSchema.create_empty_table()
        pq.write_table(empty_table, str(self.log_path))
    
    def _acquire_lock(self, file_path: Path) -> None:
        """Acquire file lock for concurrent access.
        
        Uses platform-specific locking:
        - Windows: msvcrt.locking
        - Unix: fcntl.flock
        
        Creates lock file if it doesn't exist.
        """
        # Create lock file if it doesn't exist
        if not file_path.exists():
            file_path.touch()
        
        if sys.platform == "win32":
            import msvcrt
            f = open(file_path, "r+b")
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            # Store file handle for later release
            self._lock_file_handle = f
        else:
            import fcntl
            f = open(file_path, "r+b")
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            self._lock_file_handle = f
    
    def _release_lock(self, file_path: Path) -> None:
        """Release file lock."""
        if hasattr(self, "_lock_file_handle") and self._lock_file_handle:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                self._lock_file_handle.close()
            else:
                import fcntl
                fcntl.flock(self._lock_file_handle.fileno(), fcntl.LOCK_UN)
                self._lock_file_handle.close()
            self._lock_file_handle = None
    
    async def write_decision(self, decision: Dict[str, Any]) -> str:
        """Write a decision record to the log atomically.
        
        Args:
            decision: Dictionary containing decision record
            
        Returns:
            Path to the log file
        """
        # Validate record
        if not DecisionSchema.validate_record(decision):
            raise ValueError("Invalid decision record")
        
        # Convert timestamp to datetime if string
        if isinstance(decision["timestamp"], str):
            decision["timestamp"] = pd.to_datetime(decision["timestamp"])
        
        # Create new record as DataFrame
        new_df = pd.DataFrame([decision])
        
        # Acquire lock for atomic write
        lock_file = self.log_path.with_suffix(".lock")
        
        try:
            # Async lock acquisition
            await asyncio.to_thread(self._acquire_lock, lock_file)
            
            # Read existing data
            if self.log_path.exists():
                existing_df = pd.read_parquet(str(self.log_path))
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Atomic write: write to temp file, then rename
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".parquet",
                dir=str(self.log_path.parent)
            )
            os.close(temp_fd)
            
            # Write to temp file
            combined_df.to_parquet(temp_path, index=False)
            
            # Atomic rename (remove destination first if exists)
            if self.log_path.exists():
                self.log_path.unlink()
            os.rename(temp_path, str(self.log_path))
            
            # Backup to APPDATA
            if self.backup_dir:
                await self._backup_to_appdata(combined_df)
            
        finally:
            # Release lock
            await asyncio.to_thread(self._release_lock, lock_file)
            # Clean up lock file (ignore errors if still locked)
            try:
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass  # Lock file will be cleaned up later
        
        return str(self.log_path)
    
    async def _backup_to_appdata(self, df: pd.DataFrame) -> None:
        """Backup decision log to APPDATA directory.
        
        Args:
            df: DataFrame to backup
        """
        if not self.backup_dir:
            return
        
        backup_filename = f"decision_log_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        backup_path = self.backup_dir / backup_filename
        
        await asyncio.to_thread(df.to_parquet, str(backup_path), index=False)
    
    async def read_decisions(self, limit: int = 5) -> pd.DataFrame:
        """Read recent decisions from the log.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with decision records
        """
        if not self.log_path.exists():
            return pd.DataFrame()
        
        df = await asyncio.to_thread(pd.read_parquet, str(self.log_path))
        
        # Return most recent records
        if len(df) > limit:
            df = df.tail(limit)
        
        return df
    
    async def query_by_session_id(self, session_id: str) -> pd.DataFrame:
        """Query decisions by session ID.
        
        Args:
            session_id: Session ID to query
            
        Returns:
            DataFrame with matching records
        """
        if not self.log_path.exists():
            return pd.DataFrame()
        
        df = await asyncio.to_thread(pd.read_parquet, str(self.log_path))
        
        # Filter by session_id
        matching = df[df["session_id"] == session_id]
        
        return matching