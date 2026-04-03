"""
Phase 1 RED Test for ContextBuilderEngine - TDD Protocol

This test is designed to FAIL initially because ContextBuilderEngine
doesn't exist yet. It will pass after implementation (Phase 2 GREEN).
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# This import will fail in Phase 1 RED - that's expected
from qwen_mcp.engines.context_builder import ContextBuilderEngine


@pytest.fixture
def temp_dir(tmp_path: Path):
    """Temporary directory for context files."""
    return tmp_path


@pytest.mark.asyncio(loop_scope="function")
async def test_context_builder_engine_instantiation(temp_dir: Path):
    """Test that ContextBuilderEngine can be instantiated with required params."""
    # Mock DashScopeClient to avoid real API calls
    mock_client = AsyncMock()
    mock_client.generate_completion = AsyncMock(return_value="mocked context")

    # Instantiate ContextBuilderEngine
    engine = ContextBuilderEngine(
        client=mock_client,
        context_dir=temp_dir,
    )

    # Verify instantiation
    assert engine.client == mock_client
    assert engine.context_dir == temp_dir


@pytest.mark.asyncio(loop_scope="function")
async def test_atomic_write_method(temp_dir: Path):
    """Test that _atomic_write writes files atomically (temp + rename pattern)."""
    mock_client = AsyncMock()
    engine = ContextBuilderEngine(client=mock_client, context_dir=temp_dir)

    # Test atomic write
    test_file = temp_dir / "_TEST_CONTEXT.md"
    test_content = "# Test Context\n\nThis is test content."

    engine._atomic_write(test_file, test_content)

    # Verify file exists and has correct content
    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == test_content


@pytest.mark.asyncio(loop_scope="function")
async def test_generate_project_context_returns_strings(temp_dir: Path):
    """Test that generate_project_context returns two string contexts."""
    mock_client = AsyncMock()
    
    # Mock generate_completion to return context content
    # ScoutEngine calls it first, then project context, then data context
    mock_client.generate_completion = AsyncMock(
        side_effect=[
            '{"complexity": "low", "use_swarm": false, "reason": "test"}',  # Scout response
            "# Project Context\n\nTech stack: Python",  # Project context
            "# Data Context\n\nData sources: None",  # Data context
        ]
    )

    engine = ContextBuilderEngine(client=mock_client, context_dir=temp_dir)

    # Generate contexts
    project_ctx, data_ctx = await engine.generate_project_context(str(temp_dir))

    # Verify both are strings
    assert isinstance(project_ctx, str)
    assert isinstance(data_ctx, str)
    assert len(project_ctx) > 0
    assert len(data_ctx) > 0


@pytest.mark.asyncio(loop_scope="function")
async def test_save_context_files_creates_directory(temp_dir: Path):
    """Test that save_context_files creates .context directory and saves files."""
    mock_client = AsyncMock()
    engine = ContextBuilderEngine(client=mock_client, context_dir=temp_dir)

    # Create context content
    project_content = "# Project Context\n\nTest project."
    data_content = "# Data Context\n\nTest data."

    # Save files
    saved_paths = engine.save_context_files(project_content, data_content)

    # Verify files were saved
    assert "project" in saved_paths
    assert "data" in saved_paths
    assert saved_paths["project"].exists()
    assert saved_paths["data"].exists()
    assert saved_paths["project"].read_text(encoding="utf-8") == project_content
    assert saved_paths["data"].read_text(encoding="utf-8") == data_content