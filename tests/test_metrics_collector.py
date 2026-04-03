"""Tests for Metrics Collector."""

import pytest
from pathlib import Path
from src.validator.metrics_collector import MetricsCollector, ChangeMetrics


@pytest.fixture
def collector():
    """Create a MetricsCollector instance."""
    return MetricsCollector()


@pytest.fixture
def sample_project_dir(tmp_path):
    """Create a sample project directory."""
    # Create main module
    main_file = tmp_path / "main.py"
    main_file.write_text("""
def func_a():
    pass

def func_b():
    pass

class ClassA:
    pass
""")
    
    # Create utility module
    utils_file = tmp_path / "utils.py"
    utils_file.write_text("""
def helper():
    pass
""")
    
    return tmp_path


@pytest.mark.asyncio
async def test_collect_metrics_returns_change_metrics(collector, sample_project_dir):
    """collect_metrics returns a ChangeMetrics object."""
    metrics = await collector.collect_metrics(sample_project_dir)
    
    assert isinstance(metrics, ChangeMetrics)
    assert metrics.total_files >= 0
    assert metrics.total_lines >= 0
    assert metrics.total_functions >= 0
    assert metrics.total_classes >= 0


@pytest.mark.asyncio
async def test_collect_metrics_counts_files_correctly(collector, sample_project_dir):
    """collect_metrics counts Python files correctly."""
    metrics = await collector.collect_metrics(sample_project_dir)
    
    assert metrics.total_files == 2  # main.py and utils.py


@pytest.mark.asyncio
async def test_collect_metrics_counts_functions(collector, sample_project_dir):
    """collect_metrics counts functions correctly."""
    metrics = await collector.collect_metrics(sample_project_dir)
    
    assert metrics.total_functions == 3  # func_a, func_b, helper


@pytest.mark.asyncio
async def test_collect_metrics_counts_classes(collector, sample_project_dir):
    """collect_metrics counts classes correctly."""
    metrics = await collector.collect_metrics(sample_project_dir)
    
    assert metrics.total_classes == 1  # ClassA


@pytest.mark.asyncio
async def test_compare_metrics_detects_file_changes(collector):
    """compare_metrics detects file count changes."""
    before = ChangeMetrics(total_files=5, total_lines=500, total_functions=20, total_classes=5)
    after = ChangeMetrics(total_files=7, total_lines=600, total_functions=25, total_classes=6)
    
    diff = collector.compare_metrics(before, after)
    
    assert "files_added" in diff
    assert diff["files_added"] == 2


@pytest.mark.asyncio
async def test_compare_metrics_detects_line_changes(collector):
    """compare_metrics detects line count changes."""
    before = ChangeMetrics(total_files=5, total_lines=500, total_functions=20, total_classes=5)
    after = ChangeMetrics(total_files=5, total_lines=700, total_functions=20, total_classes=5)
    
    diff = collector.compare_metrics(before, after)
    
    assert "lines_added" in diff
    assert diff["lines_added"] == 200


@pytest.mark.asyncio
async def test_compare_metrics_detects_function_changes(collector):
    """compare_metrics detects function count changes."""
    before = ChangeMetrics(total_files=5, total_lines=500, total_functions=20, total_classes=5)
    after = ChangeMetrics(total_files=5, total_lines=500, total_functions=18, total_classes=5)
    
    diff = collector.compare_metrics(before, after)
    
    assert "functions_removed" in diff
    assert diff["functions_removed"] == 2


@pytest.mark.asyncio
async def test_compare_metrics_detects_class_changes(collector):
    """compare_metrics detects class count changes."""
    before = ChangeMetrics(total_files=5, total_lines=500, total_functions=20, total_classes=5)
    after = ChangeMetrics(total_files=5, total_lines=500, total_functions=20, total_classes=7)
    
    diff = collector.compare_metrics(before, after)
    
    assert "classes_added" in diff
    assert diff["classes_added"] == 2


@pytest.mark.asyncio
async def test_collect_metrics_handles_empty_directory(collector, tmp_path):
    """collect_metrics handles empty directory."""
    metrics = await collector.collect_metrics(tmp_path)
    
    assert metrics.total_files == 0
    assert metrics.total_lines == 0
    assert metrics.total_functions == 0
    assert metrics.total_classes == 0
