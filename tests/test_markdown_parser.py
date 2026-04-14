import pytest
from qwen_mcp.engines.markdown_layer.parser import MarkdownParser


@pytest.fixture
def parser() -> MarkdownParser:
    """Provide a fresh instance of MarkdownParser for each test."""
    return MarkdownParser()


class TestExtractSection:
    """Tests for the extract_section method."""

    def test_extract_existing_header(self, parser: MarkdownParser) -> None:
        content = """
# Introduction
This is the intro.

# Features
- Feature 1
- Feature 2

# Conclusion
End of doc.
"""
        result = parser.extract_section(content, "Features")
        assert "- Feature 1" in result
        assert "- Feature 2" in result
        assert "Introduction" not in result
        assert "Conclusion" not in result

    def test_extract_non_existing_header(self, parser: MarkdownParser) -> None:
        content = "# Header 1\nContent 1"
        result = parser.extract_section(content, "Missing Header")
        assert result == ""

    def test_extract_empty_content(self, parser: MarkdownParser) -> None:
        result = parser.extract_section("", "Any Header")
        assert result == ""

    def test_extract_header_case_sensitivity(self, parser: MarkdownParser) -> None:
        content = "# Configuration\nSettings here."
        # Assuming implementation is case-sensitive based on standard markdown parsing
        result = parser.extract_section(content, "configuration")
        assert result == ""


class TestExtractTasks:
    """Tests for the extract_tasks method."""

    def test_extract_mixed_tasks(self, parser: MarkdownParser) -> None:
        content = """
# Todo List
- [ ] Implement login
- [x] Setup database
- [ ] Write tests
"""
        result = parser.extract_tasks(content)
        assert len(result) == 3
        assert "Implement login" in result[0]
        assert "Setup database" in result[1]
        assert "Write tests" in result[2]

    def test_extract_only_pending_tasks(self, parser: MarkdownParser) -> None:
        content = """
- [ ] Task A
- [x] Task B
"""
        result = parser.extract_tasks(content)
        assert len(result) == 2

    def test_extract_no_tasks(self, parser: MarkdownParser) -> None:
        content = """
# Notes
Just some regular text.
No tasks here.
"""
        result = parser.extract_tasks(content)
        assert result == []

    def test_extract_empty_content(self, parser: MarkdownParser) -> None:
        result = parser.extract_tasks("")
        assert result == []

    def test_extract_tasks_with_nested_content(self, parser: MarkdownParser) -> None:
        content = """
- [ ] Parent task
  - Sub item
- [x] Another task
"""
        result = parser.extract_tasks(content)
        assert len(result) == 2
