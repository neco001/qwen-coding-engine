import pytest
from qwen_mcp.engines.markdown_layer.sections import SectionManager


class TestSectionManager:
    @pytest.fixture
    def manager(self):
        return SectionManager()

    def test_archive_task_moves_task_to_completed(self, manager):
        content = """# Tasks

## Pending
- [ ] Task A [id:123]
- [ ] Task B [id:456]

## Completed
- [x] Task C [id:789]
"""
        task_id = "123"
        expected = """# Tasks

## Pending
- [ ] Task B [id:456]

## Completed
- [x] Task A [id:123]
- [x] Task C [id:789]
"""
        result = manager.archive_task(content, task_id)
        assert result.strip() == expected.strip()

    def test_archive_task_marks_task_as_done(self, manager):
        content = """## Pending
- [ ] Task A [id:123]

## Completed
"""
        task_id = "123"
        result = manager.archive_task(content, task_id)
        
        assert "- [x] Task A [id:123]" in result
        assert "- [ ] Task A [id:123]" not in result

    def test_archive_task_raises_if_id_not_found(self, manager):
        content = """## Pending
- [ ] Task A [id:123]

## Completed
"""
        task_id = "999"
        
        with pytest.raises(ValueError):
            manager.archive_task(content, task_id)

    def test_append_to_changelog_prepends_entry(self, manager):
        content = """# CHANGELOG\n\n- 2023-01-01: Initial release\n"""
        entry = "- 2023-01-02: New feature"
        expected = """# CHANGELOG\n\n- 2023-01-02: New feature\n\n- 2023-01-01: Initial release\n"""
        result = manager.append_to_changelog(content, entry)
        assert result.strip() == expected.strip()

    def test_append_to_changelog_creates_section_if_missing(self, manager):
        content = """# README
Some content.
"""
        entry = "- 2023-01-02: New feature"
        result = manager.append_to_changelog(content, entry)
        assert "# CHANGELOG" in result
        assert entry in result

    def test_append_to_changelog_handles_empty_content(self, manager):
        content = ""
        entry = "- 2023-01-02: New feature"
        result = manager.append_to_changelog(content, entry)
        assert "# CHANGELOG" in result
        assert entry in result
