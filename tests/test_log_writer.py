# tests/test_log_writer.py
import os
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

# Import the module under test (will fail if not implemented yet)
from src.logging.log_writer import DecisionLogWriter


class TestDecisionLogWriter:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def appdata_dir(self, temp_dir):
        """Mock APPDATA directory."""
        appdata = Path(temp_dir) / "appdata"
        appdata.mkdir()
        with patch.dict(os.environ, {"APPDATA": str(appdata)}):
            yield appdata

    @pytest.fixture
    def log_writer(self, temp_dir, appdata_dir):
        """Create a DecisionLogWriter instance with test paths."""
        log_path = Path(temp_dir) / "decision_log.parquet"
        backup_dir = appdata_dir / "qwen-mcp" / "decision_log_backup"
        return DecisionLogWriter(log_path, backup_dir)

    @pytest.mark.asyncio
    async def test_atomic_write_uses_tempfile_and_rename(self, log_writer, temp_dir):
        """Verify atomic write uses tempfile + rename pattern."""
        decision = {
            "timestamp": "2024-01-01T00:00:00Z",
            "session_id": "sess_123",
            "session_type": "edit",
            "change_hash": "abc123",
            "files_modified": ["file.py"],
            "lines_changed": 5,
            "dependency_graph_hash": "def456",
            "verdict": "approve",
            "risk_score": 0.1,
            "validator_triggers": ["lint"],
            "user_approval": True,
            "rationale": "Safe change"
        }

        # Patch os.rename to verify usage (don't call original - just record the call)
        temp_files_created = []

        def mock_rename(src, dst):
            # Record the rename operation
            temp_files_created.append(src)
            # Don't call original - just simulate success

        with patch("os.rename", side_effect=mock_rename):

            # Write decision
            await log_writer.write_decision(decision)

            # Verify a temp file was used
            assert len(temp_files_created) == 1, "Atomic write should use rename pattern"
            assert Path(temp_files_created[0]).parent == Path(temp_dir)

    @pytest.mark.asyncio
    async def test_file_locking_prevents_race_conditions(self, log_writer, temp_dir):
        """Verify file locking prevents concurrent writes."""
        decision1 = {
            "timestamp": "2024-01-01T00:00:00Z",
            "session_id": "sess_1",
            "session_type": "edit",
            "change_hash": "hash1",
            "files_modified": ["file1.py"],
            "lines_changed": 1,
            "dependency_graph_hash": "hash1",
            "verdict": "approve",
            "risk_score": 0.1,
            "validator_triggers": [],
            "user_approval": True,
            "rationale": "Test"
        }

        decision2 = {
            "timestamp": "2024-01-01T00:00:01Z",
            "session_id": "sess_2",
            "session_type": "edit",
            "change_hash": "hash2",
            "files_modified": ["file2.py"],
            "lines_changed": 2,
            "dependency_graph_hash": "hash2",
            "verdict": "approve",
            "risk_score": 0.2,
            "validator_triggers": [],
            "user_approval": True,
            "rationale": "Test"
        }

        # Try concurrent writes
        async def write_decision(decision):
            await log_writer.write_decision(decision)

        # Run both concurrently
        tasks = [write_decision(decision1), write_decision(decision2)]
        await asyncio.gather(*tasks)

        # Verify only one file exists and contains both decisions
        log_path = log_writer.log_path
        assert log_path.exists(), "Log file should exist after concurrent writes"

        # Read back and verify both decisions are present
        df = pd.read_parquet(log_path)
        assert len(df) == 2, "Both decisions should be written"
        session_ids = set(df["session_id"].tolist())
        assert session_ids == {"sess_1", "sess_2"}, "Both session IDs should be present"

    @pytest.mark.asyncio
    async def test_backup_to_appdata_succeeds(self, log_writer, appdata_dir):
        """Verify backup to APPDATA directory succeeds."""
        decision = {
            "timestamp": "2024-01-01T00:00:00Z",
            "session_id": "sess_backup",
            "session_type": "edit",
            "change_hash": "hash_backup",
            "files_modified": ["backup.py"],
            "lines_changed": 3,
            "dependency_graph_hash": "hash_backup",
            "verdict": "approve",
            "risk_score": 0.1,
            "validator_triggers": [],
            "user_approval": True,
            "rationale": "Test backup"
        }

        # Write decision
        await log_writer.write_decision(decision)

        # Check backup directory exists and contains backup file
        backup_dir = log_writer.backup_dir
        assert backup_dir.exists(), "Backup directory should exist"
        backup_files = list(backup_dir.glob("*.parquet"))
        assert len(backup_files) > 0, "Backup file should be created"

    @pytest.mark.asyncio
    async def test_read_decisions_returns_correct_records(self, log_writer, temp_dir):
        """Verify read_decisions returns correct records."""
        decisions = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "sess_read1",
                "session_type": "edit",
                "change_hash": "hash_read1",
                "files_modified": ["read1.py"],
                "lines_changed": 1,
                "dependency_graph_hash": "hash_read1",
                "verdict": "approve",
                "risk_score": 0.1,
                "validator_triggers": [],
                "user_approval": True,
                "rationale": "Read test 1"
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "session_id": "sess_read2",
                "session_type": "edit",
                "change_hash": "hash_read2",
                "files_modified": ["read2.py"],
                "lines_changed": 2,
                "dependency_graph_hash": "hash_read2",
                "verdict": "approve",
                "risk_score": 0.2,
                "validator_triggers": [],
                "user_approval": True,
                "rationale": "Read test 2"
            }
        ]

        # Write decisions
        for decision in decisions:
            await log_writer.write_decision(decision)

        # Read decisions
        df = await log_writer.read_decisions()
        assert isinstance(df, pd.DataFrame), "read_decisions should return a DataFrame"
        assert len(df) == 2, "Should return 2 records"
        assert set(df["session_id"]) == {"sess_read1", "sess_read2"}, "Session IDs should match"

    @pytest.mark.asyncio
    async def test_query_by_session_id_works(self, log_writer, temp_dir):
        """Verify query by session_id works correctly."""
        decisions = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "sess_query1",
                "session_type": "edit",
                "change_hash": "hash_query1",
                "files_modified": ["query1.py"],
                "lines_changed": 1,
                "dependency_graph_hash": "hash_query1",
                "verdict": "approve",
                "risk_score": 0.1,
                "validator_triggers": [],
                "user_approval": True,
                "rationale": "Query test 1"
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "session_id": "sess_query2",
                "session_type": "edit",
                "change_hash": "hash_query2",
                "files_modified": ["query2.py"],
                "lines_changed": 2,
                "dependency_graph_hash": "hash_query2",
                "verdict": "approve",
                "risk_score": 0.2,
                "validator_triggers": [],
                "user_approval": True,
                "rationale": "Query test 2"
            }
        ]

        # Write decisions
        for decision in decisions:
            await log_writer.write_decision(decision)

        # Query by session_id
        df = await log_writer.query_by_session_id("sess_query1")
        assert isinstance(df, pd.DataFrame), "query_by_session_id should return a DataFrame"
        assert len(df) == 1, "Should return 1 record"
        assert df.iloc[0]["session_id"] == "sess_query1", "Session ID should match query"