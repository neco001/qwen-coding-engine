"""Tests for Dependency Tracker."""
import pytest
import tempfile
from pathlib import Path
from src.graph.dependency_tracker import DependencyTracker


@pytest.fixture
def tracker():
    """Create a DependencyTracker instance."""
    return DependencyTracker()


@pytest.fixture
def sample_project_dir():
    """Create a temporary project directory with multiple files."""
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    
    # Create main.py
    main_py = Path(temp_dir) / "main.py"
    main_py.write_text("""
from utils import helper_function
from services.data_service import DataService
import os

def main():
    service = DataService()
    return helper_function(service)
""")
    
    # Create utils.py
    utils_py = Path(temp_dir) / "utils.py"
    utils_py.write_text("""
import json

def helper_function(service):
    return service.process()
""")
    
    # Create services/data_service.py
    services_dir = Path(temp_dir) / "services"
    services_dir.mkdir()
    data_service_py = services_dir / "data_service.py"
    data_service_py.write_text("""
from utils import helper_function

class DataService:
    def process(self):
        return "processed"
""")
    
    # Create services/__init__.py
    (services_dir / "__init__.py").write_text("")
    
    yield Path(temp_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_analyze_project_returns_dependencies(tracker, sample_project_dir):
    """analyze_project returns dependency graph."""
    result = await tracker.analyze_project(sample_project_dir)
    
    assert "files" in result
    assert "dependencies" in result
    assert isinstance(result["files"], dict)
    assert isinstance(result["dependencies"], list)


@pytest.mark.asyncio
async def test_analyze_project_finds_all_files(tracker, sample_project_dir):
    """analyzes all Python files in project."""
    result = await tracker.analyze_project(sample_project_dir)
    
    # Should find main.py, utils.py, services/data_service.py, services/__init__.py
    assert len(result["files"]) >= 3
    
    file_paths = list(result["files"].keys())
    assert any("main.py" in f for f in file_paths)
    assert any("utils.py" in f for f in file_paths)
    assert any("data_service.py" in f for f in file_paths)


@pytest.mark.asyncio
async def test_analyze_project_tracks_imports(tracker, sample_project_dir):
    """tracks import dependencies between files."""
    result = await tracker.analyze_project(sample_project_dir)
    
    dependencies = result["dependencies"]
    
    # main.py imports from utils
    assert any(
        d["from_file"] and "main.py" in d["from_file"] and 
        "utils" in d["import"]
        for d in dependencies
    )
    
    # main.py imports from services.data_service
    assert any(
        d["from_file"] and "main.py" in d["from_file"] and 
        "services.data_service" in d["import"]
        for d in dependencies
    )


@pytest.mark.asyncio
async def test_get_dependents_finds_all_dependent_files(tracker, sample_project_dir):
    """get_dependents returns files that depend on a given file."""
    await tracker.analyze_project(sample_project_dir)
    
    # Find utils.py path
    utils_path = None
    for file_path in tracker._file_modules.keys():
        if "utils.py" in str(file_path):
            utils_path = file_path
            break
    
    if utils_path:
        dependents = tracker.get_dependents(utils_path)
        # main.py and data_service.py depend on utils.py
        assert len(dependents) >= 1


@pytest.mark.asyncio
async def test_get_dependencies_returns_file_dependencies(tracker, sample_project_dir):
    """get_dependencies returns what a file depends on."""
    await tracker.analyze_project(sample_project_dir)
    
    # Find main.py path
    main_path = None
    for file_path in tracker._file_modules.keys():
        if "main.py" in str(file_path):
            main_path = file_path
            break
    
    if main_path:
        dependencies = tracker.get_dependencies(main_path)
        # main.py depends on utils and services.data_service
        assert len(dependencies) >= 1


@pytest.mark.asyncio
async def test_detect_circular_dependencies(tracker):
    """detect_circular_dependencies finds cycles in dependency graph."""
    # Create a simple circular dependency scenario
    tracker._file_modules = {
        Path("a.py"): {"b"},
        Path("b.py"): {"c"},
        Path("c.py"): {"a"},
    }
    tracker._module_files = {
        "a": Path("a.py"),
        "b": Path("b.py"),
        "c": Path("c.py"),
    }
    
    cycles = tracker.detect_circular_dependencies()
    
    # Should detect at least one cycle
    assert len(cycles) >= 1


@pytest.mark.asyncio
async def test_analyze_project_handles_missing_files(tracker):
    """analyze_project handles files that don't exist."""
    fake_dir = Path("/nonexistent/directory")
    result = await tracker.analyze_project(fake_dir)
    
    assert result == {"files": {}, "dependencies": []}
