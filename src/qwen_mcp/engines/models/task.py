"""
Task dataclass with state logic for DecisionLogSyncEngine.

Represents a single task entry from BACKLOG.md / decision_log.parquet.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    """
    Represents a task in the decision log / backlog system.

    Attributes:
        decision_id: UUID string identifying this task in decision_log.parquet
        description: Human-readable description of the task
        state: Current state – 'pending' | 'completed' | 'in_progress'
        backlog_ref: Optional short label used for matching in BACKLOG.md
        agentic_advice: Full advice text from the decision log record
    """
    decision_id: str
    description: str
    state: str = "pending"
    backlog_ref: Optional[str] = None
    agentic_advice: Optional[str] = None

    def is_complete(self) -> bool:
        """Return True if task state is 'completed'."""
        return self.state == "completed"

    def mark_complete(self) -> None:
        """Transition state to 'completed'."""
        self.state = "completed"

    def to_markdown_line(self) -> str:
        """Render task as a BACKLOG.md checkbox line."""
        checkbox = "[x]" if self.is_complete() else "[ ]"
        label = self.backlog_ref or self.description
        return f"- {checkbox} {label} - {self.decision_id}"

    @classmethod
    def from_parquet_record(cls, record: dict) -> "Task":
        """Build a Task from a raw decision_log.parquet row dict."""
        return cls(
            decision_id=record.get("decision_id", ""),
            description=record.get("agentic_advice", record.get("task_name", "")),
            state="completed" if record.get("patch_applied") else "pending",
            backlog_ref=record.get("backlog_ref"),
            agentic_advice=record.get("agentic_advice"),
        )
