"""Regression Detection Tests for AI-Driven Testing System.

These tests verify that the regression detection system works correctly:
1. Functional snapshot capture
2. Snapshot comparison and diff detection
3. Regression alert generation
4. Integration with validator triggers
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.graph.snapshot import FunctionalSnapshotGenerator
from src.graph.static_parser import StaticASTParser
from src.validator.trigger_logic import TriggerLogic, TriggerThresholds


@pytest.fixture
def snapshot_generator():
    """Create a FunctionalSnapshotGenerator instance."""
    return FunctionalSnapshotGenerator()


@pytest.fixture
def static_parser():
    """Create a StaticASTParser instance."""
    return StaticASTParser()


@pytest.fixture
def trigger_logic():
    """Create a TriggerLogic instance."""
    return TriggerLogic()


@pytest.fixture
def sample_project_before(tmp_path):
    """Create a sample project representing 'before' state."""
    # Use unique subdirectory to avoid conflicts with other fixtures
    before_dir = tmp_path / "before"
    before_dir.mkdir()
    
    # Main module with functions
    main_file = before_dir / "main.py"
    main_file.write_text("""
def process_data(data):
    '''Process input data and return result.'''
    return data * 2

def validate_input(value):
    '''Validate input value.'''
    return value is not None

class DataHandler:
    '''Handle data operations.'''
    
    def __init__(self):
        self.data = []
    
    def add(self, item):
        self.data.append(item)
    
    def get_all(self):
        return self.data
""")
    
    # Utils module
    utils_file = before_dir / "utils.py"
    utils_file.write_text("""
def format_output(result):
    '''Format result for display.'''
    return str(result)

def calculate_metric(value):
    '''Calculate metric from value.'''
    return value ** 2
""")
    
    return before_dir


@pytest.fixture
def sample_project_after_removed(tmp_path):
    """Create a sample project with removed functionality."""
    # Use unique subdirectory to avoid conflicts with other fixtures
    removed_dir = tmp_path / "after_removed"
    removed_dir.mkdir()
    
    # Main module with removed function
    main_file = removed_dir / "main.py"
    main_file.write_text("""
def process_data(data):
    '''Process input data and return result.'''
    return data * 2

# validate_input was removed

class DataHandler:
    '''Handle data operations.'''
    
    def __init__(self):
        self.data = []
    
    # get_all was removed
    def add(self, item):
        self.data.append(item)
""")
    
    # Utils module with removed function
    utils_file = removed_dir / "utils.py"
    utils_file.write_text("""
def format_output(result):
    '''Format result for display.'''
    return str(result)

# calculate_metric was removed
""")
    
    return removed_dir


@pytest.fixture
def sample_project_after_modified(tmp_path):
    """Create a sample project with modified signatures."""
    # Use unique subdirectory to avoid conflicts with other fixtures
    modified_dir = tmp_path / "after_modified"
    modified_dir.mkdir()
    
    # Main module with modified function
    main_file = modified_dir / "main.py"
    main_file.write_text("""
def process_data(data, options=None):
    '''Process input data and return result.'''
    return data * 2 if options is None else data * options.get('multiplier', 2)

def validate_input(value, strict=False):
    '''Validate input value.'''
    if strict:
        return value is not None and value != ''
    return value is not None

class DataHandler:
    '''Handle data operations.'''
    
    def __init__(self, max_size=100):
        self.data = []
        self.max_size = max_size
    
    def add(self, item):
        if len(self.data) < self.max_size:
            self.data.append(item)
    
    def get_all(self):
        return self.data.copy()
""")
    
    return modified_dir


@pytest.mark.asyncio
async def test_snapshot_capture_before_state(snapshot_generator, sample_project_before):
    """Test capturing snapshot of 'before' state."""
    snapshot = await snapshot_generator.capture_snapshot(sample_project_before)
    
    assert "timestamp" in snapshot
    assert "functions" in snapshot
    assert "classes" in snapshot
    
    # Should have 4 functions: process_data, validate_input, format_output, calculate_metric
    function_names = [f["name"] for f in snapshot["functions"]]
    assert "process_data" in function_names
    assert "validate_input" in function_names
    assert "format_output" in function_names
    assert "calculate_metric" in function_names
    
    # Should have 1 class: DataHandler
    class_names = [c["name"] for c in snapshot["classes"]]
    assert "DataHandler" in class_names


@pytest.mark.asyncio
async def test_snapshot_detect_removed_functions(snapshot_generator, sample_project_before, sample_project_after_removed):
    """Test detecting removed functions between snapshots."""
    before = await snapshot_generator.capture_snapshot(sample_project_before)
    after = await snapshot_generator.capture_snapshot(sample_project_after_removed)
    
    diff = await snapshot_generator.compare_snapshots(before, after)
    
    # Should detect removed functions
    assert "removed_functions" in diff
    removed_names = [f["name"] for f in diff["removed_functions"]]
    assert "validate_input" in removed_names
    assert "calculate_metric" in removed_names
    
    # Note: get_all is a method inside DataHandler class, not a standalone function
    # The snapshot captures classes separately, so we check for removed/modified classes
    # or we can check that the method count changed in the class


@pytest.mark.asyncio
async def test_snapshot_detect_modified_signatures(snapshot_generator, sample_project_before, sample_project_after_modified):
    """Test detecting modified function signatures."""
    before = await snapshot_generator.capture_snapshot(sample_project_before)
    after = await snapshot_generator.capture_snapshot(sample_project_after_modified)
    
    diff = await snapshot_generator.compare_snapshots(before, after)
    
    # Should detect modified functions
    assert "modified_functions" in diff
    modified_names = [f["name"] for f in diff["modified_functions"]]
    assert "process_data" in modified_names
    assert "validate_input" in modified_names


@pytest.mark.asyncio
async def test_detect_regression_alerts(snapshot_generator, sample_project_before, sample_project_after_removed):
    """Test regression detection generates alerts."""
    before = await snapshot_generator.capture_snapshot(sample_project_before)
    after = await snapshot_generator.capture_snapshot(sample_project_after_removed)
    
    diff = await snapshot_generator.compare_snapshots(before, after)
    alerts = await snapshot_generator.detect_regression(diff)
    
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    
    # Should have alerts for removed functions
    alert_types = [a["type"] for a in alerts]
    assert "function_removed" in alert_types


@pytest.mark.asyncio
async def test_no_regression_for_additions_only(snapshot_generator, tmp_path):
    """Test no regression alerts for additions only."""
    # Before: empty project
    before_dir = tmp_path / "before"
    before_dir.mkdir()
    
    # After: project with new functions
    after_dir = tmp_path / "after"
    after_dir.mkdir()
    (after_dir / "new.py").write_text("""
def new_feature():
    pass
""")
    
    before = await snapshot_generator.capture_snapshot(before_dir)
    after = await snapshot_generator.capture_snapshot(after_dir)
    
    diff = await snapshot_generator.compare_snapshots(before, after)
    alerts = await snapshot_generator.detect_regression(diff)
    
    # Should have no regression alerts for additions
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_regression_integration_with_trigger(trigger_logic):
    """Test regression detection integrates with trigger logic."""
    # Simulate structural change detected by regression
    change_info = {
        "lines_changed": 50,
        "files_modified": 2,
        "dependencies_affected": 3,
        "risk_score": 0.4,
        "structural_change_detected": True,  # From regression detection
    }
    
    result = trigger_logic.evaluate_triggers(change_info)
    
    # Should trigger validator due to structural change
    assert result.should_trigger is True
    assert "structural" in result.reason.lower()


@pytest.mark.asyncio
async def test_full_regression_workflow(snapshot_generator, trigger_logic, sample_project_before, sample_project_after_removed):
    """Test full regression detection workflow."""
    # Step 1: Capture before snapshot
    before = await snapshot_generator.capture_snapshot(sample_project_before)
    
    # Step 2: Capture after snapshot
    after = await snapshot_generator.capture_snapshot(sample_project_after_removed)
    
    # Step 3: Compare snapshots
    diff = await snapshot_generator.compare_snapshots(before, after)
    
    # Step 4: Detect regressions
    alerts = await snapshot_generator.detect_regression(diff)
    
    # Step 5: Evaluate if validator should be triggered
    structural_detected = len(alerts) > 0
    change_info = {
        "lines_changed": 100,
        "files_modified": 2,
        "dependencies_affected": 5,
        "risk_score": 0.5,
        "structural_change_detected": structural_detected,
    }
    
    trigger_result = trigger_logic.evaluate_triggers(change_info)
    
    # Verify workflow results
    assert structural_detected is True
    assert len(alerts) > 0
    assert trigger_result.should_trigger is True
