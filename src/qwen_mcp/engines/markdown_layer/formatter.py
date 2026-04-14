"""MarkdownFormatter - task line and changelog entry formatting utilities."""
from typing import Optional, List


class MarkdownFormatter:
    """Stateless formatter for markdown task lines and changelog entries."""

    @staticmethod
    def format_task_line(task_name: str, decision_id: Optional[str] = None, completed: bool = False) -> str:
        """Format a single task line for BACKLOG.md.
        
        Args:
            task_name: Human-readable task name
            decision_id: UUID identifier (appended after ' - ')
            completed: Whether to mark as [x] or [ ]
            
        Returns:
            Formatted markdown task line
        """
        checkbox = "[x]" if completed else "[ ]"
        name = task_name.strip()
        if decision_id:
            return f"- {checkbox} {name} - {decision_id}"
        return f"- {checkbox} {name}"

    @staticmethod
    def format_changelog_entry(
        decision_id: str,
        advice: str,
        timestamp: str,
        task_name: Optional[str] = None
    ) -> str:
        """Format a single changelog entry block.
        
        Args:
            decision_id: UUID of the decision
            advice: Agentic advice text
            timestamp: Formatted timestamp string
            task_name: Optional task name for context
            
        Returns:
            Formatted changelog entry string
        """
        entry = f"### [{timestamp}] {decision_id}\n\n"
        if task_name:
            entry += f"**Task**: {task_name}\n\n"
        entry += f"**Advice**: {advice}\n\n"
        entry += "---\n"
        return entry

    @staticmethod
    def format_sync_cluster(entries: List[str], sync_timestamp: str) -> str:
        """Format a cluster of changelog entries under a single SOS Sync header.
        
        Args:
            entries: List of formatted changelog entry strings
            sync_timestamp: Timestamp for the sync cluster header
            
        Returns:
            Combined sync cluster string
        """
        header = f"## SOS Sync - {sync_timestamp}\n\n"
        return header + "\n".join(entries)
