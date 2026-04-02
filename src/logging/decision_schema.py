"""PyArrow schema definition for Decision Log."""

import pyarrow as pa
from datetime import datetime
from typing import Dict, Any, List


# PyArrow schema for .qwen/decision_log.parquet
DECISION_LOG_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("ms")),
    pa.field("session_id", pa.string()),
    pa.field("session_type", pa.string()),
    pa.field("change_hash", pa.string()),
    pa.field("files_modified", pa.list_(pa.string())),
    pa.field("lines_changed", pa.int64()),
    pa.field("dependency_graph_hash", pa.string()),
    pa.field("verdict", pa.string()),
    pa.field("risk_score", pa.float32()),
    pa.field("validator_triggers", pa.list_(pa.string())),
    pa.field("user_approval", pa.bool_()),
    pa.field("rationale", pa.string()),
])


class DecisionSchema:
    """Helper class for working with Decision Log schema."""
    
    @staticmethod
    def get_schema() -> pa.Schema:
        """Return the PyArrow schema for Decision Log."""
        return DECISION_LOG_SCHEMA
    
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        """Validate a record against the schema.
        
        Args:
            record: Dictionary containing decision record fields
            
        Returns:
            True if record is valid, False otherwise
        """
        required_fields = [
            "timestamp", "session_id", "session_type", "change_hash",
            "files_modified", "lines_changed", "dependency_graph_hash",
            "verdict", "risk_score", "validator_triggers", 
            "user_approval", "rationale"
        ]
        
        for field in required_fields:
            if field not in record:
                return False
        
        # Type validation
        if not isinstance(record["timestamp"], (str, datetime)):
            return False
        if not isinstance(record["session_id"], str):
            return False
        if not isinstance(record["session_type"], str):
            return False
        if not isinstance(record["files_modified"], list):
            return False
        if not isinstance(record["lines_changed"], int):
            return False
        if not isinstance(record["risk_score"], (int, float)):
            return False
        if not isinstance(record["validator_triggers"], list):
            return False
        if not isinstance(record["user_approval"], bool):
            return False
        
        return True
    
    @staticmethod
    def create_empty_table() -> pa.Table:
        """Create an empty table with the Decision Log schema."""
        return pa.table({
            "timestamp": pa.array([], type=pa.timestamp("ms")),
            "session_id": pa.array([], type=pa.string()),
            "session_type": pa.array([], type=pa.string()),
            "change_hash": pa.array([], type=pa.string()),
            "files_modified": pa.array([], type=pa.list_(pa.string())),
            "lines_changed": pa.array([], type=pa.int64()),
            "dependency_graph_hash": pa.array([], type=pa.string()),
            "verdict": pa.array([], type=pa.string()),
            "risk_score": pa.array([], type=pa.float32()),
            "validator_triggers": pa.array([], type=pa.list_(pa.string())),
            "user_approval": pa.array([], type=pa.bool_()),
            "rationale": pa.array([], type=pa.string()),
        })