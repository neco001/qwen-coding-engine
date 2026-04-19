import os
import json
import atexit
import tempfile
import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from platformdirs import user_cache_dir
import logging
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# Platform-specific file locking
import msvcrt

logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path(user_cache_dir("qwen-coding", "Qwen"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "usage_analytics.duckdb"
DB_LOCK_FILE = CACHE_DIR / ".duckdb.lock"
RETENTION_FILE = CACHE_DIR / "retention.json"
SESSION_RETENTION_DAYS = 7
SPARRING_RETENTION_DAYS = 30
VACUUM_INTERVAL_DAYS = 7

# PyArrow schema for session parquet
SESSION_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us")),
    ("project_name", pa.string()),
    ("model_name", pa.string()),
    ("prompt_tokens", pa.int32()),
    ("completion_tokens", pa.int32()),
    ("total_tokens", pa.int32()),
    ("image_count", pa.int32()),
])


class SessionBuffer:
    """
    Per-process session buffer that writes to session_{pid}.parquet.
    
    This avoids DuckDB file locks during active sessions by using
    append-only Parquet files (one per process).
    """
    
    def __init__(self, pid: int = None):
        self.pid = pid or os.getpid()
        self.parquet_path = CACHE_DIR / f"session_{self.pid}.parquet"
        self.buffer: list = []
        self._lock = threading.Lock()
        self._flushed = False
        
        # Load existing session data if present
        self._load_existing()
        
        # Register atexit handler for flush on process exit
        atexit.register(self.flush)
        logger.debug(f"SessionBuffer initialized for PID {self.pid}: {self.parquet_path}")
    
    def _load_existing(self):
        """Load existing session parquet data into buffer."""
        if self.parquet_path.exists():
            try:
                table = pq.read_table(self.parquet_path)
                self.buffer = table.to_pylist()
                logger.debug(f"Loaded {len(self.buffer)} existing records from session parquet")
            except Exception as e:
                logger.warning(f"Failed to load existing session parquet: {e}")
                self.buffer = []
    
    def append(self, record: dict):
        """
        Append a usage record to the session buffer.
        Flushes to parquet every 10 records to minimize I/O.
        """
        with self._lock:
            self.buffer.append(record)
            if len(self.buffer) >= 10:
                self._write_parquet()
    
    def _write_parquet(self):
        """Write buffer to parquet file (atomic append)."""
        if not self.buffer:
            return
        
        try:
            # Create new table from buffer
            new_table = pa.Table.from_pylist(self.buffer, schema=SESSION_SCHEMA)
            
            # Combine with existing data if present
            if self.parquet_path.exists():
                try:
                    existing = pq.read_table(self.parquet_path)
                    combined = pa.concat_tables([existing, new_table])
                except Exception:
                    combined = new_table
            else:
                combined = new_table
            
            # Atomic write: temp file + rename
            fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=str(CACHE_DIR))
            os.close(fd)
            pq.write_table(combined, temp_path)
            
            # Rename temp to final (atomic on most filesystems)
            Path(temp_path).replace(self.parquet_path)
            
            logger.debug(f"Wrote {len(self.buffer)} records to session parquet")
            self.buffer = []
            
        except Exception as e:
            logger.error(f"Failed to write session parquet: {e}")
    
    def flush(self):
        """
        Flush session buffer to DuckDB and cleanup.
        Called at process exit via atexit handler.
        
        Only deletes parquet file if merge succeeded.
        If merge fails, parquet is preserved for next process to retry.
        """
        if self._flushed:
            return
        
        with self._lock:
            self._flushed = True
            
            # Write remaining buffer to parquet
            self._write_parquet()
            
            # Merge to DuckDB
            if self.parquet_path.exists():
                success = merge_to_duckdb(self.parquet_path)
                
                if success:
                    # Delete session parquet after successful merge
                    self.parquet_path.unlink(missing_ok=True)
                    logger.info(f"Flushed session {self.pid} to DuckDB")
                else:
                    # Merge failed - keep parquet for retry
                    logger.warning(f"Merge failed for session {self.pid} - parquet preserved: {self.parquet_path}")


class BillingTracker:
    """
    Hybrid billing tracker using per-process Parquet buffers + DuckDB analytics.
    
    Architecture:
    - During session: Write to session_{pid}.parquet (no locks)
    - At exit: Merge parquet to DuckDB (batch write)
    - For queries: Read from DuckDB (analytics)
    """
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.db_path = DB_PATH
        self.session_buffer: SessionBuffer = None
        self._initialized = False
        self._init_lock = threading.Lock()
        
        # Run retention check on startup (async, non-blocking)
        self._check_retention_startup()
    
    def _ensure_initialized(self):
        """Lazy initialization of session buffer."""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    # Retry merging orphaned session files from previous failed merges
                    self._retry_orphaned_sessions()
                    
                    self._init_db()
                    self.session_buffer = SessionBuffer()
                    self._initialized = True
                    logger.info("BillingTracker initialized with session buffer")
    
    def _retry_orphaned_sessions(self):
        """
        Retry merging orphaned session parquet files from previous failed merges.
        Called once at startup.
        """
        orphaned_files = list(CACHE_DIR.glob("session_*.parquet"))
        if not orphaned_files:
            return
        
        logger.info(f"Found {len(orphaned_files)} orphaned session files to merge")
        
        for parquet_file in orphaned_files:
            try:
                # Skip our own PID (shouldn't happen, but safety check)
                pid_str = parquet_file.stem.replace("session_", "")
                if int(pid_str) == os.getpid():
                    continue
                
                success = merge_to_duckdb(parquet_file)
                if success:
                    parquet_file.unlink()
                    logger.info(f"Merged orphaned session: {parquet_file.name}")
                else:
                    logger.warning(f"Failed to merge orphaned session: {parquet_file.name} (preserved)")
            except Exception as e:
                logger.error(f"Error processing orphaned session {parquet_file.name}: {e}")
    
    def _init_db(self):
        """Initialize DuckDB with retry logic for handling file locks."""
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with duckdb.connect(str(self.db_path)) as con:
                    con.execute("""
                        CREATE TABLE IF NOT EXISTS usage_stats (
                            timestamp TIMESTAMP,
                            project_name VARCHAR,
                            model_name VARCHAR,
                            prompt_tokens INTEGER,
                            completion_tokens INTEGER,
                            total_tokens INTEGER,
                            image_count INTEGER DEFAULT 0
                        )
                    """)
                    # Migration: Add image_count to legacy tables if missing
                    try:
                        con.execute("ALTER TABLE usage_stats ADD COLUMN image_count INTEGER DEFAULT 0")
                    except:
                        pass
                return  # Success
            except duckdb.IOException as e:
                error_str = str(e).lower()
                if "locked" in error_str or "used by another process" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (attempt + 1)
                        logger.warning(f"BillingTracker: DB locked (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.warning(f"BillingTracker: DB still locked after {max_retries} attempts.")
                        return
                else:
                    logger.error(f"BillingTracker: DuckDB IO error: {e}")
                    return
            except Exception as e:
                logger.error(f"BillingTracker: Failed to initialize DB: {e}")
                return
    
    def _check_retention_startup(self):
        """Check if retention is due on startup (runs async, non-blocking)."""
        try:
            today = datetime.now().date()
            
            if RETENTION_FILE.exists():
                data = json.loads(RETENTION_FILE.read_text())
                last_run_str = data.get("last_retention")
                if last_run_str:
                    last_run = datetime.strptime(last_run_str, "%Y-%m-%d").date()
                    if (today - last_run).days < SESSION_RETENTION_DAYS:
                        logger.debug(f"Retention not due (last: {last_run})")
                        return
            else:
                data = {}
            
            # Run retention in background thread (non-blocking)
            thread = threading.Thread(target=self._run_retention_cleanup, daemon=True)
            thread.start()
            logger.info("Retention cleanup started in background")
            
            # Update retention file
            data["last_retention"] = str(today)
            data["next_retention"] = str(today + timedelta(days=SESSION_RETENTION_DAYS))
            RETENTION_FILE.write_text(json.dumps(data, indent=2))
            
        except Exception as e:
            logger.error(f"Failed to check retention: {e}")
    
    def _run_retention_cleanup(self):
        """Run retention cleanup (delete old session files and sparring sessions)."""
        try:
            cutoff = datetime.now() - timedelta(days=SESSION_RETENTION_DAYS)
            deleted_count = 0
            
            # Clean session parquet files older than cutoff
            for parquet_file in CACHE_DIR.glob("session_*.parquet"):
                try:
                    mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)
                    if mtime < cutoff:
                        parquet_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old session: {parquet_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {parquet_file.name}: {e}")
            
            # Clean sparring sessions older than cutoff
            sparring_dir = Path.home() / "AppData" / "Roaming" / "qwen-mcp" / "sparring_sessions"
            if sparring_dir.exists():
                for session_file in sparring_dir.glob("session_*.json"):
                    try:
                        # Extract date from filename: session_YYYY-MM-DD_*.json
                        parts = session_file.stem.split("_")
                        if len(parts) >= 2:
                            date_str = parts[1]
                            file_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if file_date < cutoff:
                                session_file.unlink()
                                deleted_count += 1
                                logger.info(f"Deleted old sparring session: {session_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete sparring session {session_file.name}: {e}")
            
            # VACUUM DuckDB if needed
            self._vacuum_duckdb_if_needed()
            
            logger.info(f"Retention cleanup complete: deleted {deleted_count} files")
            
        except Exception as e:
            logger.error(f"Retention cleanup failed: {e}")
    
    def _vacuum_duckdb_if_needed(self):
        """VACUUM DuckDB if >7 days since last vacuum."""
        try:
            vacuum_marker = CACHE_DIR / ".last_vacuum"
            
            if vacuum_marker.exists():
                last_vacuum = datetime.fromtimestamp(vacuum_marker.stat().st_mtime)
                if (datetime.now() - last_vacuum).days < VACUUM_INTERVAL_DAYS:
                    return
            
            # Run VACUUM
            with duckdb.connect(str(self.db_path)) as con:
                con.execute("VACUUM")
            
            # Update marker
            vacuum_marker.touch()
            logger.info("DuckDB VACUUM complete")
            
        except Exception as e:
            logger.warning(f"Failed to VACUUM DuckDB: {e}")
    
    def log_usage(self, project_name: str, model_name: str, prompt_tokens: int, completion_tokens: int, image_count: int = 0):
        """Log usage to session buffer (parquet)."""
        try:
            self._ensure_initialized()
            
            record = {
                "timestamp": datetime.now(),
                "project_name": project_name,
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "image_count": image_count,
            }
            
            self.session_buffer.append(record)
            
        except Exception as e:
            logger.error(f"BillingTracker: Failed to log usage: {e}")
    
    def get_daily_project_report(self):
        """Get daily project report from DuckDB."""
        try:
            self._ensure_initialized()
            
            with duckdb.connect(str(self.db_path)) as con:
                result = con.execute("""
                    SELECT
                        CAST(timestamp AS DATE) as usage_date,
                        project_name,
                        model_name,
                        SUM(prompt_tokens) as total_prompt,
                        SUM(completion_tokens) as total_completion,
                        SUM(total_tokens) as total_all
                    FROM usage_stats
                    GROUP BY usage_date, project_name, model_name
                    ORDER BY usage_date DESC, project_name, model_name
                """).fetchall()
                
                return [{
                    "date": str(row[0]),
                    "project_name": row[1],
                    "model_name": row[2],
                    "prompt_tokens": row[3],
                    "completion_tokens": row[4],
                    "total_tokens": row[5]
                } for row in result]
                
        except Exception as e:
            logger.error(f"BillingTracker: Failed to generate report: {e}")
            return []
    
    def get_summary(self) -> dict:
        """Get overall usage summary from DuckDB."""
        try:
            self._ensure_initialized()
            
            with duckdb.connect(str(self.db_path)) as con:
                # Total stats
                total_result = con.execute("""
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(prompt_tokens) as total_prompt,
                        SUM(completion_tokens) as total_completion
                    FROM usage_stats
                """).fetchone()
                
                # By model breakdown
                model_result = con.execute("""
                    SELECT
                        model_name,
                        COUNT(*) as requests,
                        SUM(prompt_tokens) as prompt,
                        SUM(completion_tokens) as completion
                    FROM usage_stats
                    GROUP BY model_name
                    ORDER BY requests DESC
                """).fetchall()
                
                by_model = {}
                for row in model_result:
                    by_model[row[0]] = {
                        "requests": row[1],
                        "prompt": row[2],
                        "completion": row[3]
                    }
                
                return {
                    "total_requests": total_result[0] if total_result else 0,
                    "total_prompt_tokens": total_result[1] if total_result else 0,
                    "total_completion_tokens": total_result[2] if total_result else 0,
                    "by_model": by_model
                }
                
        except Exception as e:
            logger.error(f"BillingTracker: Failed to get summary: {e}")
            return {
                "total_requests": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "by_model": {}
            }


def merge_to_duckdb(parquet_path: Path, max_retries: int = 10) -> bool:
    """
    Merge session parquet file into DuckDB with deduplication.
    
    Uses INSERT OR IGNORE to avoid duplicates based on unique constraint
    (timestamp, project_name, model_name, total_tokens).
    
    Args:
        parquet_path: Path to session parquet file
        max_retries: Maximum retry attempts for file lock conflicts
    
    Returns:
        True if merge succeeded, False if failed (data preserved in parquet)
    """
    if not parquet_path.exists():
        return True  # Nothing to merge
    
    # Read parquet data once (outside retry loop)
    try:
        table = pq.read_table(parquet_path)
        data = table.to_pylist()
    except Exception as e:
        logger.error(f"Failed to read parquet {parquet_path}: {e}")
        return False
    
    if not data:
        return True  # Empty file, nothing to merge
    
    # Acquire file lock for serialized access (Windows: msvcrt, Unix: fcntl)
    lock_fd = None
    try:
        lock_fd = open(DB_LOCK_FILE, 'w')
        # Windows: msvcrt.locking, Unix: fcntl.flock
        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)
        logger.debug(f"Acquired DuckDB lock for merge: {parquet_path.name}")
    except Exception as e:
        logger.warning(f"Failed to acquire lock: {e}")
        # Continue without lock (fallback to retry logic)
    
    try:
        # Retry loop for file lock conflicts during merge
        base_delay = 0.1  # seconds
        
        for attempt in range(max_retries):
            try:
                with duckdb.connect(str(DB_PATH)) as con:
                    # Create temp table for deduplication
                    con.execute("""
                        CREATE TEMP TABLE IF NOT EXISTS temp_session_data (
                            timestamp TIMESTAMP,
                            project_name VARCHAR,
                            model_name VARCHAR,
                            prompt_tokens INTEGER,
                            completion_tokens INTEGER,
                            total_tokens INTEGER,
                            image_count INTEGER
                        )
                    """)
                    
                    # Insert parquet data into temp table
                    for record in data:
                        con.execute("""
                            INSERT INTO temp_session_data
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            record["timestamp"],
                            record["project_name"],
                            record["model_name"],
                            record["prompt_tokens"],
                            record["completion_tokens"],
                            record["total_tokens"],
                            record["image_count"],
                        ))
                    
                    # Insert into main table with deduplication
                    # Using a composite key check to avoid duplicates
                    con.execute("""
                        INSERT INTO usage_stats
                        SELECT * FROM temp_session_data
                        WHERE NOT EXISTS (
                            SELECT 1 FROM usage_stats u
                            WHERE u.timestamp = temp_session_data.timestamp
                              AND u.project_name = temp_session_data.project_name
                              AND u.model_name = temp_session_data.model_name
                              AND u.total_tokens = temp_session_data.total_tokens
                        )
                    """)
                    
                    # Drop temp table
                    con.execute("DROP TABLE temp_session_data")
                    
                logger.debug(f"Merged {len(data)} records from {parquet_path.name} to DuckDB")
                return True  # Success
                
            except duckdb.IOException as e:
                error_str = str(e).lower()
                if "locked" in error_str or "used by another process" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (attempt + 1)
                        logger.warning(f"merge_to_duckdb: DB locked (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"merge_to_duckdb: DB still locked after {max_retries} attempts. Parquet preserved for retry.")
                        return False  # Data preserved in parquet
                else:
                    logger.error(f"merge_to_duckdb: DuckDB IO error: {e}")
                    return False  # Data preserved in parquet
            except Exception as e:
                logger.error(f"merge_to_duckdb: Failed to merge: {e}")
                return False  # Data preserved in parquet
        
        return False  # Exhausted retries
    
    finally:
        # Release file lock
        if lock_fd:
            try:
                lock_fd.seek(0)
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                lock_fd.close()
                logger.debug("Released DuckDB lock")
            except Exception:
                pass


# Singleton instance (lazy initialization)
_billing_tracker: BillingTracker = None


def get_billing_tracker() -> BillingTracker:
    """Get or create BillingTracker singleton."""
    global _billing_tracker
    if _billing_tracker is None:
        _billing_tracker = BillingTracker()
    return _billing_tracker


# Backward compatibility alias
billing_tracker = get_billing_tracker()
