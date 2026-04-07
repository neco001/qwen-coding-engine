import pytest
import pandas as pd
import os
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

@pytest.fixture
def temp_parquet_dir():
    """Create a temporary directory for parquet files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def decision_log_parquet(temp_parquet_dir):
    """Create a decision_log.parquet with one agentic_advice entry matching our real schema"""
    from decision_log.decision_schema import DecisionSchema
    parquet_path = Path(temp_parquet_dir) / "decision_log.parquet"
    
    # Create sample data matching DECISION_LOG_SCHEMA
    record = {
        "decision_id": "test-sync-001",
        "timestamp": datetime.now(),
        "session_id": "test-session",
        "decision_type": "audit",
        "complexity": "low",
        "tokens_used": 500,
        "content": "Audit found an issue",
        "tags": ["test"],
        "backlog_ref": "",
        "context_version": "1.0.0",
        "patch_applied": False,
        "agentic_advice": "### SOS ADVICE\nAdd Task: Implement security headers\nPriority: High",
        "risk_score": 0.2,
        "validator_triggers": [],
        "user_approval": True,
        "rationale": "Test advice"
    }
    
    # Use our schema to create the table
    import pyarrow as pa
    import pyarrow.parquet as pq
    from decision_log.decision_schema import DECISION_LOG_SCHEMA
    
    table = pa.table({k: [v] for k, v in record.items()}, schema=DECISION_LOG_SCHEMA)
    pq.write_table(table, str(parquet_path))
    return parquet_path

@pytest.mark.asyncio
async def test_sos_sync_detects_agentic_advice(decision_log_parquet, tmp_path):
    """
    RED TEST: sos_sync engine should detect agentic_advice entries and propose implementation to BACKLOG.md
    This test will fail because the engine doesn't exist yet (ModuleNotFoundError)
    """
    # Import the engine (should fail with ModuleNotFoundError)
    try:
        from qwen_mcp.engines.sos_sync import SOSSyncEngine
    except ImportError:
        pytest.fail("RED TEST SUCCESS: Engine module not found as expected")
    
    engine = SOSSyncEngine(decision_log_path=decision_log_parquet)
    
    # Create a temporary BACKLOG.md file
    backlog_path = tmp_path / "PLAN" / "BACKLOG.md"
    backlog_path.parent.mkdir(parents=True, exist_ok=True)
    backlog_path.write_text("# BACKLOG\n\n## Planned Features\n\n")
    
    # Run the engine scan
    advices = await engine.scan_advices()
    
    # Verify the engine detected the agentic_advice entry
    assert len(advices) > 0, "Engine should return detected advices"
    assert "security headers" in advices[0]["agentic_advice"].lower()
    
    # Apply the advice
    await engine.apply_advice(advices[0]["decision_id"], backlog_path=backlog_path)
    
    # Verify BACKLOG.md was updated
    backlog_content = backlog_path.read_text()
    assert "security headers" in backlog_content.lower()
