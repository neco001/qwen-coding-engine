"""
Tests for Swarm Context Injection - TDD RED Phase

These tests verify:
1. Path validation security (traversal attacks blocked)
2. Context resolution (file contents retrieved)
3. Backward compatibility (tasks without context_keys work unchanged)
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.append(os.path.join(os.getcwd(), "src"))


class TestPathValidation:
    """Security tests for path validation."""
    
    def test_reject_traversal_parent(self):
        """Reject paths with ../"""
        from qwen_mcp.io_utils import validate_path
        
        root = Path("/safe/project")
        result = validate_path("../etc/passwd", root)
        assert result is None, "Path traversal should be rejected"
    
    def test_reject_traversal_double_parent(self):
        """Reject paths with ../../"""
        from qwen_mcp.io_utils import validate_path
        
        root = Path("/safe/project")
        result = validate_path("../../etc/passwd", root)
        assert result is None, "Double parent traversal should be rejected"
    
    def test_reject_absolute_outside_root(self):
        """Reject absolute paths outside project root."""
        from qwen_mcp.io_utils import validate_path
        
        root = Path("/safe/project")
        result = validate_path("/etc/passwd", root)
        assert result is None, "Absolute path outside root should be rejected"
    
    def test_accept_valid_relative(self):
        """Accept valid relative paths."""
        from qwen_mcp.io_utils import validate_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_file = root / "test.py"
            test_file.write_text("content")
            
            result = validate_path("test.py", root)
            assert result is not None, "Valid relative path should be accepted"
            assert result == test_file.resolve()
    
    def test_accept_nested_relative(self):
        """Accept nested relative paths like src/module/file.py."""
        from qwen_mcp.io_utils import validate_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested_dir = root / "src" / "module"
            nested_dir.mkdir(parents=True)
            test_file = nested_dir / "file.py"
            test_file.write_text("content")
            
            result = validate_path("src/module/file.py", root)
            assert result is not None, "Nested path should be accepted"
            assert result == test_file.resolve()


class TestContextResolution:
    """Tests for context_keys resolution."""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_resolve_existing_file(self):
        """Resolve existing file returns content."""
        from qwen_mcp.io_utils import resolve_context_keys
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.py"
            test_file.write_text("FEATURES_BATCH_THRESHOLD = 30")
            
            result = await resolve_context_keys(["config.py"], tmpdir)
            assert "config.py" in result
            assert "FEATURES_BATCH_THRESHOLD" in result["config.py"]
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_resolve_multiple_files(self):
        """Resolve multiple files returns all contents."""
        from qwen_mcp.io_utils import resolve_context_keys
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "config.py"
            file1.write_text("THRESHOLD = 30")
            file2 = Path(tmpdir) / "processor.py"
            file2.write_text("def process(): pass")
            
            result = await resolve_context_keys(["config.py", "processor.py"], tmpdir)
            assert len(result) == 2
            assert "THRESHOLD" in result["config.py"]
            assert "def process" in result["processor.py"]
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_resolve_missing_file(self):
        """Missing file returns error message."""
        from qwen_mcp.io_utils import resolve_context_keys
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await resolve_context_keys(["nonexistent.py"], tmpdir)
            assert "nonexistent.py" in result
            assert "NOT FOUND" in result["nonexistent.py"]
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_resolve_invalid_path(self):
        """Invalid path (traversal) returns error message."""
        from qwen_mcp.io_utils import resolve_context_keys
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await resolve_context_keys(["../etc/passwd"], tmpdir)
            assert "../etc/passwd" in result
            assert "INVALID" in result["../etc/passwd"]
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_empty_context_keys(self):
        """Empty context_keys returns empty dict."""
        from qwen_mcp.io_utils import resolve_context_keys
        
        result = await resolve_context_keys([], "/tmp")
        assert result == {}


class TestFileSizeLimits:
    """Tests for file size and line limits."""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_large_file_truncated(self):
        """Large files are truncated with warning."""
        from qwen_mcp.io_utils import resolve_context_keys, MAX_FILE_SIZE_BYTES
        
        with tempfile.TemporaryDirectory() as tmpdir:
            large_file = Path(tmpdir) / "large.py"
            # Create file larger than limit
            large_content = "x = 1\n" * 10000  # ~70KB+
            large_file.write_text(large_content)
            
            result = await resolve_context_keys(["large.py"], tmpdir)
            # Should either truncate or skip
            assert "large.py" in result
            # Content should be truncated or have warning
            content = result["large.py"]
            # Either truncated or marked as too large
            assert "truncated" in content.lower() or "too large" in content.lower() or len(content) < len(large_content)


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_task_without_context_keys(self):
        """Tasks without context_keys execute unchanged."""
        from qwen_mcp.orchestrator import SubTask, SwarmOrchestrator
        
        mock_handler = MagicMock()
        mock_handler.generate_completion = AsyncMock(return_value="test result")
        
        orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
        
        task = SubTask(id="1", task="Simple task", priority=5)
        result = await orchestrator._execute_single_task(task)
        
        # Should have called with just the task text
        call_args = mock_handler.generate_completion.call_args
        messages = call_args[0][0]
        assert messages[0]["content"] == "Simple task"
        assert result == "test result"
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_task_with_context_keys_injects_content(self):
        """Tasks with context_keys have file content injected."""
        from qwen_mcp.orchestrator import SubTask, SwarmOrchestrator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "target.py"
            test_file.write_text("def existing_function(): return 42")
            
            mock_handler = MagicMock()
            mock_handler.generate_completion = AsyncMock(return_value="implemented")
            
            orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
            
            # Task with context_keys
            task = SubTask(
                id="1",
                task="Add new function to target.py",
                priority=5,
                context_keys=["target.py"]
            )
            
            # Change working directory to tmpdir for path resolution
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                result = await orchestrator._execute_single_task(task)
                
                # Verify content was injected
                call_args = mock_handler.generate_completion.call_args
                messages = call_args[0][0]
                content = messages[0]["content"]
                
                # Should have task description
                assert "Add new function" in content
                # Should have file content injected
                assert "existing_function" in content or "target.py" in content
            finally:
                os.chdir(old_cwd)


class TestIntegration:
    """Integration tests for full Swarm flow."""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_decompose_creates_context_keys(self):
        """Decomposition should populate context_keys for file-targeted tasks."""
        from qwen_mcp.orchestrator import SwarmOrchestrator
        
        mock_handler = MagicMock()
        mock_handler.generate_completion = AsyncMock(return_value="""{
            "intent_validation": true,
            "sub_tasks": [
                {"id": "1", "task": "Modify config.py", "priority": 5, "context_keys": ["config.py"]}
            ]
        }""")
        
        orchestrator = SwarmOrchestrator(completion_handler=mock_handler)
        
        result = await orchestrator.decompose("Implement feature in config.py")
        
        assert result.intent_validation == True
        assert len(result.sub_tasks) == 1
        assert result.sub_tasks[0].context_keys == ["config.py"]