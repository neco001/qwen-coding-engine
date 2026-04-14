import pytest
from pathlib import Path
from qwen_mcp.engines.io_layer.file_handler import FileHandler


class TestFileHandler:
    """Test suite for the abstract FileHandler class."""

    def test_write_atomic_creates_file_with_content(self, tmp_path: Path) -> None:
        """Verify write_atomic creates a new file with the specified content."""
        target_file = tmp_path / "test_output.txt"
        content = "Hello, Atomic World!"

        FileHandler.write_atomic(target_file, content)

        assert target_file.exists()
        assert target_file.read_text() == content

    def test_write_atomic_replaces_existing_content(self, tmp_path: Path) -> None:
        """Verify write_atomic overwrites existing file content atomically."""
        target_file = tmp_path / "test_output.txt"
        initial_content = "Old Content"
        new_content = "New Content"

        target_file.write_text(initial_content)
        FileHandler.write_atomic(target_file, new_content)

        assert target_file.read_text() == new_content
        assert initial_content not in target_file.read_text()

    def test_write_atomic_cleans_up_temp_files(self, tmp_path: Path) -> None:
        """Verify no temporary files are left behind after atomic write."""
        target_file = tmp_path / "test_output.txt"
        content = "Temporary Content"

        FileHandler.write_atomic(target_file, content)

        temp_files = list(tmp_path.glob("*.*"))
        assert len(temp_files) == 1
        assert temp_files[0] == target_file

    def test_read_text_returns_content(self, tmp_path: Path) -> None:
        """Verify read_text returns the file content correctly."""
        target_file = tmp_path / "test_input.txt"
        expected_content = "Read Me"
        target_file.write_text(expected_content)

        actual_content = FileHandler.read_text(target_file)

        assert actual_content == expected_content

    def test_read_text_handles_missing_file(self, tmp_path: Path) -> None:
        """Verify read_text handles missing files gracefully without raising."""
        missing_file = tmp_path / "nonexistent.txt"

        actual_content = FileHandler.read_text(missing_file)

        assert actual_content is None
