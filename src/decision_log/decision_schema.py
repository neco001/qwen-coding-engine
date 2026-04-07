"""PyArrow schema definition for Decision Log."""
import pyarrow as pa
from datetime import datetime
from typing import Dict, Any, List

# Complete SOS-ready Schema
DECISION_LOG_SCHEMA = pa.schema([
    pa.field("decision_id", pa.string()),
    pa.field("timestamp", pa.timestamp("ns")),
    pa.field("session_id", pa.string()),
    pa.field("decision_type", pa.string()),
    pa.field("complexity", pa.string()),
    pa.field("tokens_used", pa.int64()),
    pa.field("content", pa.string()),
    pa.field("tags", pa.list_(pa.string())),
    # SOS Extensions
    pa.field("backlog_ref", pa.string()),
    pa.field("context_version", pa.string()),
    pa.field("patch_applied", pa.bool_()),
    pa.field("agentic_advice", pa.string()),
    pa.field("risk_score", pa.float32()),
    pa.field("validator_triggers", pa.list_(pa.string())),
    pa.field("user_approval", pa.bool_()),
    pa.field("rationale", pa.string()),
])

class DecisionSchema:
    @staticmethod
    def get_schema() -> pa.Schema:
        """Returns the PyArrow schema for the decision log."""
        return DECISION_LOG_SCHEMA
    
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        """Validates if a record dictionary contains all mandatory fields."""
        required_fields = ["decision_id", "timestamp", "session_id", "decision_type", "complexity", "tokens_used", "content"]
        for field in required_fields:
            if field not in record:
                return False
        return True

    @staticmethod
    def create_empty_table() -> pa.Table:
        """Creates an empty PyArrow table matching the schema."""
        return pa.table({
            k: pa.array([], type=v.type) 
            for k, v in zip(DECISION_LOG_SCHEMA.names, DECISION_LOG_SCHEMA.types)
        })
