"""Tests for MarkdownFormatter - task line formatting and changelog entry building."""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from qwen_mcp.engines.markdown_layer.formatter import MarkdownFormatter


class TestFormatTaskLine:
    """Test task line formatting for BACKLOG.md entries."""

    def test_format_pending_task_line(self):
        line = MarkdownFormatter.format_task_line("Fix the bug", "abc-123", completed=False)
        assert line == "- [ ] Fix the bug - abc-123"

    def test_format_completed_task_line(self):
        line = MarkdownFormatter.format_task_line("Fix the bug", "abc-123", completed=True)
        assert line == "- [x] Fix the bug - abc-123"

    def test_format_task_line_without_id(self):
        line = MarkdownFormatter.format_task_line("Fix the bug", decision_id=None, completed=False)
        assert line == "- [ ] Fix the bug"

    def test_format_task_line_strips_whitespace(self):
        line = MarkdownFormatter.format_task_line("  Fix the bug  ", "abc-123", completed=False)
        assert line == "- [ ] Fix the bug - abc-123"


class TestFormatChangelogEntry:
    """Test changelog entry formatting."""

    def test_format_single_changelog_entry(self):
        entry = MarkdownFormatter.format_changelog_entry(
            decision_id="abc-123",
            advice="Refactored the module",
            timestamp="2026-04-15 00:30:00",
            task_name="Fix the bug"
        )
        assert "### [2026-04-15 00:30:00] abc-123" in entry
        assert "**Task**: Fix the bug" in entry
        assert "**Advice**: Refactored the module" in entry
        assert "---" in entry

    def test_format_changelog_entry_without_task_name(self):
        entry = MarkdownFormatter.format_changelog_entry(
            decision_id="abc-123",
            advice="Done",
            timestamp="2026-04-15"
        )
        assert "**Task**" not in entry
        assert "**Advice**: Done" in entry

    def test_format_sync_cluster_entry(self):
        entries = [
            MarkdownFormatter.format_changelog_entry("id1", "advice1", "2026-04-15"),
            MarkdownFormatter.format_changelog_entry("id2", "advice2", "2026-04-15"),
        ]
        cluster = MarkdownFormatter.format_sync_cluster(entries, "2026-04-15 01:00:00")
        assert cluster.startswith("## SOS Sync - 2026-04-15 01:00:00")
        assert "advice1" in cluster
        assert "advice2" in cluster
