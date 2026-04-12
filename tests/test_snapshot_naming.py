"""Tests for snapshot naming convention and auto-selection functionality."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from src.graph.snapshot import FunctionalSnapshotGenerator


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory with snapshots storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        snapshots_dir = project_dir / ".anti_degradation" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        yield project_dir


@pytest.fixture
def snapshot_generator():
    """Create FunctionalSnapshotGenerator instance."""
    return FunctionalSnapshotGenerator()


class TestTimestampedNaming:
    """Tests for timestamped snapshot naming."""

    def test_generate_timestamped_name_format(self, snapshot_generator):
        """Test that generated name follows baseline-YYYYMMDD_HHMMSS format."""
        name = snapshot_generator._generate_timestamped_name()
        
        # Check format: baseline-YYYYMMDD_HHMMSS (24 chars total)
        assert name.startswith("baseline-")
        assert len(name) == 24
        
        # Extract timestamp part and verify it's parseable
        timestamp_str = name[9:]  # Remove "baseline-" prefix
        parsed = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        assert parsed is not None

    def test_generate_timestamped_name_uses_utc(self, snapshot_generator):
        """Test that timestamp uses UTC timezone."""
        with patch('src.graph.snapshot.datetime') as mock_datetime:
            # Mock UTC time
            mock_utc = datetime(2026, 4, 12, 15, 30, 45, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_utc
            mock_datetime.strftime = mock_utc.strftime
            
            name = snapshot_generator._generate_timestamped_name()
            assert name == "baseline-20260412_153045"

    def test_save_snapshot_auto_generates_name(self, snapshot_generator, temp_project_dir):
        """Test that save_snapshot with 'auto' generates timestamped name."""
        snapshot = {"test": "data", "timestamp": "2026-04-12T15:30:45Z"}
        
        path = snapshot_generator.save_snapshot(snapshot, temp_project_dir, name="auto")
        
        # Verify path has timestamped format
        filename = path.stem
        assert filename.startswith("baseline-")
        assert len(filename) == 24

    def test_save_snapshot_none_generates_name(self, snapshot_generator, temp_project_dir):
        """Test that save_snapshot with None generates timestamped name."""
        snapshot = {"test": "data", "timestamp": "2026-04-12T15:30:45Z"}
        
        path = snapshot_generator.save_snapshot(snapshot, temp_project_dir, name=None)
        
        filename = path.stem
        assert filename.startswith("baseline-")
        assert len(filename) == 24

    def test_save_snapshot_explicit_name_preserved(self, snapshot_generator, temp_project_dir):
        """Test that explicit name is preserved (backward compatibility)."""
        snapshot = {"test": "data"}
        
        path = snapshot_generator.save_snapshot(snapshot, temp_project_dir, name="my-custom-name")
        
        filename = path.stem
        assert filename == "my-custom-name"


class TestListSnapshots:
    """Tests for listing snapshots."""

    def test_list_snapshots_empty_dir(self, snapshot_generator, temp_project_dir):
        """Test listing when no snapshots exist."""
        snapshots = snapshot_generator.list_snapshots(temp_project_dir)
        assert snapshots == []

    def test_list_snapshots_with_timestamped_files(self, snapshot_generator, temp_project_dir):
        """Test listing timestamped snapshot files."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        # Create test snapshots with different timestamps
        for ts in ["20260412_100000", "20260412_110000", "20260412_120000"]:
            path = snapshots_dir / f"baseline-{ts}.json"
            with open(path, 'w') as f:
                json.dump({"timestamp": f"2026-04-12T{ts[:8]}T{ts[9:]}Z"}, f)
        
        snapshots = snapshot_generator.list_snapshots(temp_project_dir)
        
        assert len(snapshots) == 3
        # Should be sorted newest first
        assert snapshots[0]['name'] == "baseline-20260412_120000"
        assert snapshots[1]['name'] == "baseline-20260412_110000"
        assert snapshots[2]['name'] == "baseline-20260412_100000"

    def test_list_snapshots_with_legacy_names(self, snapshot_generator, temp_project_dir):
        """Test listing snapshots with legacy names (no timestamp in filename)."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        # Create legacy snapshot with timestamp in content
        legacy_path = snapshots_dir / "baseline.json"
        with open(legacy_path, 'w') as f:
            json.dump({"timestamp": "2026-04-10T10:00:00Z"}, f)
        
        # Create timestamped snapshot
        ts_path = snapshots_dir / "baseline-20260412_120000.json"
        with open(ts_path, 'w') as f:
            json.dump({"timestamp": "2026-04-12T12:00:00Z"}, f)
        
        snapshots = snapshot_generator.list_snapshots(temp_project_dir)
        
        assert len(snapshots) == 2
        # Timestamped should be first (newer)
        assert snapshots[0]['name'] == "baseline-20260412_120000"
        assert snapshots[1]['name'] == "baseline"


class TestGetTwoNewestSnapshots:
    """Tests for auto-selection of two newest snapshots."""

    def test_get_two_newest_none_when_less_than_two(self, snapshot_generator, temp_project_dir):
        """Test returns None when less than 2 snapshots exist."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        # Create only one snapshot
        path = snapshots_dir / "baseline-20260412_100000.json"
        with open(path, 'w') as f:
            json.dump({"test": "data"}, f)
        
        result = snapshot_generator.get_two_newest_snapshots(temp_project_dir)
        assert result is None

    def test_get_two_newest_returns_tuple(self, snapshot_generator, temp_project_dir):
        """Test returns tuple of two newest names."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        # Create three snapshots
        for ts in ["20260412_100000", "20260412_110000", "20260412_120000"]:
            path = snapshots_dir / f"baseline-{ts}.json"
            with open(path, 'w') as f:
                json.dump({"timestamp": f"2026-04-12T{ts[:8]}T{ts[9:]}Z"}, f)
        
        result = snapshot_generator.get_two_newest_snapshots(temp_project_dir)
        
        assert result is not None
        assert result[0] == "baseline-20260412_120000"  # newest
        assert result[1] == "baseline-20260412_110000"  # second newest

    def test_get_two_newest_empty_dir(self, snapshot_generator, temp_project_dir):
        """Test returns None when no snapshots exist."""
        result = snapshot_generator.get_two_newest_snapshots(temp_project_dir)
        assert result is None


class TestLoadSnapshotBackwardCompatible:
    """Tests for backward compatible snapshot loading."""

    def test_load_snapshot_by_timestamped_name(self, snapshot_generator, temp_project_dir):
        """Test loading snapshot by full timestamped name."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        path = snapshots_dir / "baseline-20260412_100000.json"
        test_data = {"test": "timestamped"}
        with open(path, 'w') as f:
            json.dump(test_data, f)
        
        loaded = snapshot_generator.load_snapshot(temp_project_dir, "baseline-20260412_100000")
        assert loaded == test_data

    def test_load_snapshot_by_short_timestamp(self, snapshot_generator, temp_project_dir):
        """Test loading snapshot by timestamp part only (without baseline- prefix)."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        path = snapshots_dir / "baseline-20260412_100000.json"
        test_data = {"test": "short_timestamp"}
        with open(path, 'w') as f:
            json.dump(test_data, f)
        
        # Load by short timestamp (should auto-add baseline- prefix)
        loaded = snapshot_generator.load_snapshot(temp_project_dir, "20260412_100000")
        assert loaded == test_data

    def test_load_snapshot_legacy_name(self, snapshot_generator, temp_project_dir):
        """Test loading snapshot by legacy name (backward compatibility)."""
        snapshots_dir = temp_project_dir / ".anti_degradation" / "snapshots"
        
        path = snapshots_dir / "baseline.json"
        test_data = {"test": "legacy"}
        with open(path, 'w') as f:
            json.dump(test_data, f)
        
        loaded = snapshot_generator.load_snapshot(temp_project_dir, "baseline")
        assert loaded == test_data

    def test_load_snapshot_not_found(self, snapshot_generator, temp_project_dir):
        """Test returns None when snapshot not found."""
        loaded = snapshot_generator.load_snapshot(temp_project_dir, "nonexistent")
        assert loaded is None