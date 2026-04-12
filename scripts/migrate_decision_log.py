#!/usr/bin/env python3
"""
Migration Script: decision_log.parquet Schema Migration

Migrates from old schema (16 fields) to new schema (22 fields with ADR extension).
Also moves the file from src/ to .decision_log/ directory.

Usage: .\.venv\Scripts\python.exe scripts/migrate_decision_log.py
"""

import sys
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decision_log.decision_schema import DECISION_LOG_SCHEMA


def migrate_schema():
    """Migrate decision_log.parquet to new schema with ADR fields."""
    
    old_path = Path("src/decision_log.parquet")
    new_dir = Path(".decision_log")
    new_path = new_dir / "decision_log.parquet"
    
    print("="*60)
    print("DECISION LOG SCHEMA MIGRATION")
    print("="*60)
    print(f"\nSource: {old_path.absolute()}")
    print(f"Target: {new_path.absolute()}")
    
    # Check if old file exists
    if not old_path.exists():
        print(f"\n❌ Source file not found: {old_path}")
        print("No migration needed - creating fresh parquet file.")
        
        # Create new directory
        new_dir.mkdir(exist_ok=True)
        
        # Create empty table with new schema
        from decision_log.decision_schema import DecisionSchema
        table = DecisionSchema.create_empty_table()
        pq.write_table(table, str(new_path))
        
        print(f"\n✅ Created new empty parquet at {new_path}")
        print(f"   Schema: {len(DECISION_LOG_SCHEMA)} fields")
        return True
    
    # Load existing data
    print(f"\n📖 Reading existing parquet...")
    old_table = pq.read_table(old_path)
    print(f"   Records: {old_table.num_rows}")
    print(f"   Old schema: {len(old_table.schema)} fields")
    
    # Show old columns
    print(f"\n📋 Old columns ({len(old_table.schema)}):")
    for col in old_table.column_names:
        print(f"   - {col}")
    
    # Create new directory
    new_dir.mkdir(exist_ok=True)
    
    # Build new table with extended schema
    print(f"\n📋 New schema ({len(DECISION_LOG_SCHEMA)} fields):")
    
    # Get old data as list of dicts
    old_data = old_table.to_pylist()
    
    # New ADR fields to add (all nullable)
    adr_fields = [
        "adr_status",
        "adr_context", 
        "adr_consequences",
        "adr_alternatives",
        "linked_code_nodes",
        "depends_on_adr"
    ]
    
    # Add ADR fields with None values
    for record in old_data:
        for field in adr_fields:
            if field not in record:
                record[field] = None
    
    # Create new table from extended records
    new_table = pa.Table.from_pylist(old_data, schema=DECISION_LOG_SCHEMA)
    
    print(f"\n✨ ADR fields added (6):")
    for field in adr_fields:
        print(f"   + {field} (nullable)")
    
    # Write new parquet
    pq.write_table(new_table, str(new_path))
    print(f"\n💾 Written to {new_path}")
    print(f"   Records: {new_table.num_rows}")
    print(f"   Schema: {len(DECISION_LOG_SCHEMA)} fields")
    
    # Verify
    verify_table = pq.read_table(new_path)
    assert len(verify_table.schema) == len(DECISION_LOG_SCHEMA)
    print(f"\n✅ Migration VERIFIED")
    
    print(f"\n📁 Summary:")
    print(f"   Old file: {old_path} (kept as backup)")
    print(f"   New file: {new_path} (ready to use)")
    print(f"   Records migrated: {old_table.num_rows}")
    print(f"   New fields added: {len(adr_fields)}")
    
    return True


if __name__ == "__main__":
    try:
        migrate_schema()
        print("\n" + "="*60)
        print("MIGRATION COMPLETE")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Migration FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
