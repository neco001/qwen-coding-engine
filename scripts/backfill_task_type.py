#!/usr/bin/env python
"""
Backfill task_type field for existing decision_log.parquet records.

This script migrates existing records by adding task_type="decision" 
(since they are historical decision records from the migration).

Usage:
    pyv scripts/backfill_task_type.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pyarrow as pa
import pyarrow.parquet as pq
from decision_log.decision_schema import DECISION_LOG_SCHEMA


def backfill_task_type():
    """Backfill task_type field for existing records."""
    parquet_path = Path(".decision_log/decision_log.parquet")
    
    if not parquet_path.exists():
        print(f"❌ No parquet file found at {parquet_path}")
        return False
    
    print(f"=== Backfilling task_type field ===")
    print(f"File: {parquet_path}")
    
    # Read existing data
    table = pq.read_table(str(parquet_path))
    records = table.to_pylist()
    
    print(f"Existing records: {len(records)}")
    
    # Backfill task_type for records missing it
    modified = False
    for i, record in enumerate(records):
        if "task_type" not in record or record["task_type"] is None:
            # Default to "decision" for historical records
            record["task_type"] = "decision"
            modified = True
            print(f"  Record {i}: Added task_type='decision'")
    
    if not modified:
        print("✅ All records already have task_type field")
        return True
    
    # Rewrite parquet with new schema
    try:
        new_table = pa.Table.from_pylist(records, schema=DECISION_LOG_SCHEMA)
        pq.write_table(new_table, str(parquet_path))
        print(f"✅ Successfully backfilled {len(records)} records")
        return True
    except Exception as e:
        print(f"❌ Failed to write parquet: {e}")
        return False


if __name__ == "__main__":
    success = backfill_task_type()
    sys.exit(0 if success else 1)
