"""Tests for Functional Snapshot Generator."""
import pytest
import tempfile
from pathlib import Path
from src.graph.snapshot import FunctionalSnapshotGenerator


@pytest.fixture
def snapshot_gen():
    """Create a FunctionalSnapshotGenerator instance."""
    return FunctionalSnapshotGenerator()


@pytest.fixture
def sample_project_dir():
    """Create a temporary project directory."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    # Create a module with functions
    module_py = Path(temp_dir) / "module.py"
    module_py.write_text("""
def add(a, b):
    '''Add two numbers.'''
    return a + b

def multiply(a, b):
    '''Multiply two numbers.'''
    return a * b

class Calculator:
    '''Simple calculator class.'''
    
    def divide(self, a, b):
        return a / b
""")
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_capture_snapshot_returns_snapshot_dict(snapshot_gen, sample_project_dir):
    """capture_snapshot returns a snapshot dictionary."""
    snapshot = await snapshot_gen.capture_snapshot(sample_project_dir)
    
    assert "timestamp" in snapshot
    assert "files" in snapshot
    assert "functions" in snapshot
    assert "classes" in snapshot
    assert "mappings" in snapshot


@pytest.mark.asyncio
async def test_capture_snapshot_extracts_functions(snapshot_gen, sample_project_dir):
    """capture_snapshot extracts function signatures."""
    snapshot = await snapshot_gen.capture_snapshot(sample_project_dir)
    
    functions = snapshot["functions"]
    func_names = [f["name"] for f in functions]
    
    assert "add" in func_names
    assert "multiply" in func_names


@pytest.mark.asyncio
async def test_capture_snapshot_extracts_classes(snapshot_gen, sample_project_dir):
    """capture_snapshot extracts class information."""
    snapshot = await snapshot_gen.capture_snapshot(sample_project_dir)
    
    classes = snapshot["classes"]
    class_names = [c["name"] for c in classes]
    
    assert "Calculator" in class_names


@pytest.mark.asyncio
async def test_compare_snapshots_detects_added_functions(snapshot_gen):
    """compare_snapshots detects added functions."""
    before = {
        "functions": [{"name": "func_a", "signature": "func_a()"}],
        "classes": [],
        "mappings": {},
    }
    
    after = {
        "functions": [
            {"name": "func_a", "signature": "func_a()"},
            {"name": "func_b", "signature": "func_b()"},
        ],
        "classes": [],
        "mappings": {},
    }
    
    diff = await snapshot_gen.compare_snapshots(before, after)
    
    assert "added_functions" in diff
    assert len(diff["added_functions"]) == 1
    assert diff["added_functions"][0]["name"] == "func_b"


@pytest.mark.asyncio
async def test_compare_snapshots_detects_removed_functions(snapshot_gen):
    """compare_snapshots detects removed functions."""
    before = {
        "functions": [
            {"name": "func_a", "signature": "func_a()"},
            {"name": "func_b", "signature": "func_b()"},
        ],
        "classes": [],
        "mappings": {},
    }
    
    after = {
        "functions": [{"name": "func_a", "signature": "func_a()"}],
        "classes": [],
        "mappings": {},
    }
    
    diff = await snapshot_gen.compare_snapshots(before, after)
    
    assert "removed_functions" in diff
    assert len(diff["removed_functions"]) == 1
    assert diff["removed_functions"][0]["name"] == "func_b"


@pytest.mark.asyncio
async def test_compare_snapshots_detects_modified_signatures(snapshot_gen):
    """compare_snapshots detects modified function signatures."""
    before = {
        "functions": [{"name": "func_a", "signature": "func_a()"}],
        "classes": [],
        "mappings": {},
    }
    
    after = {
        "functions": [{"name": "func_a", "signature": "func_a(x, y)"}],
        "classes": [],
        "mappings": {},
    }
    
    diff = await snapshot_gen.compare_snapshots(before, after)
    
    assert "modified_functions" in diff
    assert len(diff["modified_functions"]) == 1


@pytest.mark.asyncio
async def test_detect_regression_returns_alerts(snapshot_gen):
    """detect_regression returns alerts for regressions."""
    diff = {
        "added_functions": [],
        "removed_functions": [{"name": "important_func"}],
        "modified_functions": [],
        "added_classes": [],
        "removed_classes": [],
    }
    
    alerts = await snapshot_gen.detect_regression(diff)
    
    assert isinstance(alerts, list)
    assert len(alerts) >= 1
    assert any("removed" in alert.get("type", "") for alert in alerts)


@pytest.mark.asyncio
async def test_detect_regression_no_alerts_for_additions_only(snapshot_gen):
    """detect_regression returns no alerts for additions only."""
    diff = {
        "added_functions": [{"name": "new_func"}],
        "removed_functions": [],
        "modified_functions": [],
        "added_classes": [],
        "removed_classes": [],
    }
    
    alerts = await snapshot_gen.detect_regression(diff)
    
    # Additions alone shouldn't trigger regression alerts
    assert len(alerts) == 0
