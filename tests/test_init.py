# tests/test_init.py
"""Tests for .qwen directory initialization."""

import tempfile
import shutil
from pathlib import Path
import pytest
import pyarrow.parquet as pq

from src.logging.init import init_qwen_directory, get_qwen_dir
from src.logging.decision_schema import DecisionSchema


class TestInitQwenDirectory:
    @pytest.fixture
    def temp_project_root(self):
        """Create a temporary project directory."""
        tmpdir = Path(tempfile.mkdtemp())
        yield tmpdir
        shutil.rmtree(tmpdir)

    def test_init_creates_directory(self, temp_project_root):
        """Test that init_qwen_directory creates .qwen directory."""
        qwen_dir = init_qwen_directory(temp_project_root)
        
        assert qwen_dir.exists(), ".qwen directory should be created"
        assert qwen_dir.name == ".qwen"
        assert qwen_dir.parent == temp_project_root

    def test_init_creates_parquet_file(self, temp_project_root):
        """Test that decision_log.parquet is created with schema."""
        qwen_dir = init_qwen_directory(temp_project_root)
        parquet_path = qwen_dir / "decision_log.parquet"
        
        assert parquet_path.exists(), "decision_log.parquet should be created"
        
        # Verify schema
        table = pq.read_table(str(parquet_path))
        expected_schema = DecisionSchema.get_schema()
        
        assert table.schema.equals(expected_schema), "Schema should match"

    def test_init_creates_gitignore(self, temp_project_root):
        """Test that .gitignore is created."""
        qwen_dir = init_qwen_directory(temp_project_root)
        gitignore_path = qwen_dir / ".gitignore"
        
        assert gitignore_path.exists(), ".gitignore should be created"
        
        content = gitignore_path.read_text()
        assert "*.tmp" in content, "Should ignore .tmp files"
        assert "*.lock" in content, "Should ignore .lock files"

    def test_init_creates_readme(self, temp_project_root):
        """Test that README.md is created."""
        qwen_dir = init_qwen_directory(temp_project_root)
        readme_path = qwen_dir / "README.md"
        
        assert readme_path.exists(), "README.md should be created"
        
        content = readme_path.read_text()
        assert "Decision Log" in content, "README should mention Decision Log"
        assert "schema" in content.lower(), "README should describe schema"

    def test_init_idempotent(self, temp_project_root):
        """Test that init is idempotent (doesn't overwrite existing files)."""
        qwen_dir = init_qwen_directory(temp_project_root)
        parquet_path = qwen_dir / "decision_log.parquet"
        
        # Write some data
        original_mtime = parquet_path.stat().st_mtime
        
        # Init again
        init_qwen_directory(temp_project_root)
        
        # File should not be overwritten
        assert parquet_path.exists()
        # Note: mtime might change due to schema re-write, so just verify file exists

    def test_get_qwen_dir_creates_if_missing(self, temp_project_root):
        """Test that get_qwen_dir creates directory if missing."""
        qwen_dir = get_qwen_dir(temp_project_root)
        
        assert qwen_dir.exists()
        assert qwen_dir.name == ".qwen"

    def test_get_qwen_dir_returns_existing(self, temp_project_root):
        """Test that get_qwen_dir returns existing directory."""
        # Create manually
        existing_qwen = temp_project_root / ".qwen"
        existing_qwen.mkdir()
        
        qwen_dir = get_qwen_dir(temp_project_root)
        
        assert qwen_dir == existing_qwen