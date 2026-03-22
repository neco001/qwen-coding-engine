"""
Unit Tests for SessionStore - TDD Foundation

Tests verify:
- Session creation with proper schema
- Atomic save/load integrity
- Step updates and state transitions
- Error handling and recovery
- Cleanup operations
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone

from qwen_mcp.engines.session_store import SessionStore, SessionCheckpoint


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test sessions."""
    temp_dir = tempfile.mkdtemp(prefix="sparring_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def store(temp_storage_dir):
    """Create a SessionStore with temporary storage."""
    return SessionStore(storage_dir=temp_storage_dir)


# =============================================================================
# Test: Session Creation
# =============================================================================

class TestSessionCreation:
    """Tests for session creation functionality."""
    
    def test_create_session_generates_id(self, store):
        """Session creation should generate a unique session_id."""
        session = store.create_session(topic="Test Topic")
        assert session.session_id.startswith("sp_")
        assert len(session.session_id) > 3  # sp_ + at least some chars
    
    def test_create_session_sets_timestamps(self, store):
        """Session should have created_at and updated_at timestamps."""
        session = store.create_session(topic="Test Topic")
        assert session.created_at is not None
        assert session.updated_at is not None
        assert "Z" in session.created_at  # ISO format with Z suffix
    
    def test_create_session_sets_initial_status(self, store):
        """New session should have 'in_progress' status."""
        session = store.create_session(topic="Test Topic")
        assert session.status == "in_progress"
    
    def test_create_session_sets_initial_step(self, store):
        """New session should start at 'discovery' step."""
        session = store.create_session(topic="Test Topic")
        assert session.current_step == "discovery"
    
    def test_create_session_with_context(self, store):
        """Session should store context."""
        session = store.create_session(
            topic="Test Topic", 
            context="Additional context here"
        )
        assert session.context == "Additional context here"
    
    def test_create_session_empty_steps_completed(self, store):
        """New session should have empty steps_completed list."""
        session = store.create_session(topic="Test Topic")
        assert session.steps_completed == []
    
    def test_create_session_empty_roles(self, store):
        """New session should have empty roles dict."""
        session = store.create_session(topic="Test Topic")
        assert session.roles == {}
    
    def test_create_session_empty_results(self, store):
        """New session should have empty results dict."""
        session = store.create_session(topic="Test Topic")
        assert session.results == {}


# =============================================================================
# Test: Atomic Save/Load
# =============================================================================

class TestAtomicSaveLoad:
    """Tests for atomic save and load operations."""
    
    def test_save_creates_file(self, store):
        """Save should create a JSON file."""
        session = store.create_session(topic="Test")
        session_path = store._get_session_path(session.session_id)
        assert session_path.exists()
    
    def test_save_uses_atomic_write(self, store):
        """Save should use atomic write (no partial files on interrupt)."""
        session = store.create_session(topic="Test")
        
        # Verify no temp files left behind
        # Temp files have format: {session_id}_random.json
        # Session files have format: {session_id}.json
        # So we check for files with multiple underscores after sp_ prefix
        all_json = list(store.storage_dir.glob("*.json"))
        temp_files = [f for f in all_json if f.stem.count('_') > 1]
        assert len(temp_files) == 0  # No temp files should remain
    
    def test_load_returns_session(self, store):
        """Load should return the saved session."""
        session = store.create_session(topic="Test Topic", context="ctx")
        loaded = store.load(session.session_id)
        
        assert loaded is not None
        assert loaded.session_id == session.session_id
        assert loaded.topic == session.topic
        assert loaded.context == session.context
    
    def test_load_nonexistent_returns_none(self, store):
        """Load should return None for nonexistent session."""
        loaded = store.load("sp_nonexistent")
        assert loaded is None
    
    def test_save_updates_timestamp(self, store):
        """Save should update the updated_at timestamp."""
        session = store.create_session(topic="Test")
        original_updated = session.updated_at
        
        # Wait a tiny bit and save again
        import time
        time.sleep(0.01)
        
        store.save(session)
        loaded = store.load(session.session_id)
        
        assert loaded.updated_at >= original_updated
    
    def test_json_schema_integrity(self, store):
        """Saved JSON should match SessionCheckpoint schema."""
        session = store.create_session(topic="Test")
        session.roles = {
            "red_role": "Red Team",
            "blue_role": "Blue Team",
            "white_role": "White Team"
        }
        session.results["discovery"] = {"roles": session.roles}
        store.save(session)
        
        # Read raw JSON and verify structure
        session_path = store._get_session_path(session.session_id)
        with open(session_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify all required fields exist
        required_fields = [
            "session_id", "topic", "context", "created_at", "updated_at",
            "status", "steps_completed", "current_step", "roles", "results",
            "loop_count", "error"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# =============================================================================
# Test: Step Updates
# =============================================================================

class TestStepUpdates:
    """Tests for step update functionality."""
    
    def test_update_step_adds_to_completed(self, store):
        """update_step should add step to steps_completed."""
        session = store.create_session(topic="Test")
        
        store.update_step(session.session_id, "discovery", {"roles": {}})
        loaded = store.load(session.session_id)
        
        assert "discovery" in loaded.steps_completed
    
    def test_update_step_stores_result(self, store):
        """update_step should store the result."""
        session = store.create_session(topic="Test")
        result_data = {"critique": "Test critique"}
        
        store.update_step(session.session_id, "red", result_data)
        loaded = store.load(session.session_id)
        
        assert loaded.results["red"] == result_data
    
    def test_update_step_changes_current_step(self, store):
        """update_step should update current_step if next_step provided."""
        session = store.create_session(topic="Test")
        
        store.update_step(
            session.session_id, 
            "discovery", 
            {"roles": {}},
            next_step="red"
        )
        loaded = store.load(session.session_id)
        
        assert loaded.current_step == "red"
    
    def test_update_step_nonexistent_session(self, store):
        """update_step should return None for nonexistent session."""
        result = store.update_step("sp_nonexistent", "discovery", {})
        assert result is None
    
    def test_update_step_duplicate_not_added(self, store):
        """update_step should not add duplicate steps."""
        session = store.create_session(topic="Test")
        
        store.update_step(session.session_id, "discovery", {})
        store.update_step(session.session_id, "discovery", {})
        loaded = store.load(session.session_id)
        
        assert loaded.steps_completed.count("discovery") == 1


# =============================================================================
# Test: Status Transitions
# =============================================================================

class TestStatusTransitions:
    """Tests for session status management."""
    
    def test_mark_failed_sets_status(self, store):
        """mark_failed should set status to 'failed'."""
        session = store.create_session(topic="Test")
        
        store.mark_failed(session.session_id, "Test error message")
        loaded = store.load(session.session_id)
        
        assert loaded.status == "failed"
        assert loaded.error == "Test error message"
    
    def test_mark_failed_nonexistent_session(self, store):
        """mark_failed should return None for nonexistent session."""
        result = store.mark_failed("sp_nonexistent", "Error")
        assert result is None
    
    def test_full_session_lifecycle(self, store):
        """Test complete session lifecycle: create → steps → complete."""
        session = store.create_session(topic="Full Test")
        
        # Discovery
        store.update_step(
            session.session_id, 
            "discovery", 
            {"roles": {"red_role": "R", "blue_role": "B", "white_role": "W"}},
            next_step="red"
        )
        
        # Red
        store.update_step(
            session.session_id,
            "red",
            {"critique": "Test critique"},
            next_step="blue"
        )
        
        # Blue
        store.update_step(
            session.session_id,
            "blue",
            {"defense": "Test defense"},
            next_step="white"
        )
        
        # White (final)
        store.update_step(
            session.session_id,
            "white",
            {"consensus": "Test consensus"}
        )
        
        loaded = store.load(session.session_id)
        
        assert loaded.steps_completed == ["discovery", "red", "blue", "white"]
        assert loaded.status == "completed"
        assert loaded.current_step == "white"


# =============================================================================
# Test: Session Management
# =============================================================================

class TestSessionManagement:
    """Tests for session management operations."""
    
    def test_delete_session(self, store):
        """delete should remove the session file."""
        session = store.create_session(topic="Test")
        session_path = store._get_session_path(session.session_id)
        
        assert session_path.exists()
        result = store.delete(session.session_id)
        
        assert result is True
        assert not session_path.exists()
    
    def test_delete_nonexistent_session(self, store):
        """delete should return False for nonexistent session."""
        result = store.delete("sp_nonexistent")
        assert result is False
    
    def test_list_sessions(self, store):
        """list_sessions should return all sessions."""
        store.create_session(topic="Test 1")
        store.create_session(topic="Test 2")
        
        sessions = store.list_sessions()
        
        assert len(sessions) == 2
        topics = [s["topic"] for s in sessions]
        assert "Test 1" in topics
        assert "Test 2" in topics
    
    def test_list_sessions_returns_metadata(self, store):
        """list_sessions should return session metadata."""
        session = store.create_session(topic="Test")
        
        sessions = store.list_sessions()
        
        assert len(sessions) == 1
        meta = sessions[0]
        assert "session_id" in meta
        assert "topic" in meta
        assert "status" in meta
        assert "created_at" in meta
    
    def test_cleanup_old_sessions(self, store):
        """cleanup_old_sessions should remove old sessions."""
        # Create a session
        session = store.create_session(topic="Test")
        session_path = store._get_session_path(session.session_id)
        
        # Manipulate file time to make it "old"
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
        os.utime(session_path, (old_time, old_time))
        
        # Cleanup should remove it
        removed = store.cleanup_old_sessions(max_age_hours=24)
        
        assert removed == 1
        assert not session_path.exists()
    
    def test_cleanup_keeps_recent_sessions(self, store):
        """cleanup_old_sessions should keep recent sessions."""
        session = store.create_session(topic="Recent")
        
        removed = store.cleanup_old_sessions(max_age_hours=24)
        
        assert removed == 0
        assert store.load(session.session_id) is not None


# =============================================================================
# Test: Schema Validation
# =============================================================================

class TestSchemaValidation:
    """Tests for JSON schema validation."""
    
    def test_from_dict_creates_checkpoint(self):
        """SessionCheckpoint.from_dict should create valid checkpoint."""
        data = {
            "session_id": "sp_test123",
            "topic": "Test Topic",
            "context": "Test Context",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "status": "in_progress",
            "steps_completed": [],
            "current_step": "discovery",
            "roles": {},
            "results": {},
            "loop_count": 0,
            "error": None
        }
        
        checkpoint = SessionCheckpoint.from_dict(data)
        
        assert checkpoint.session_id == "sp_test123"
        assert checkpoint.topic == "Test Topic"
    
    def test_to_dict_serializes_checkpoint(self):
        """SessionCheckpoint.to_dict should serialize properly."""
        checkpoint = SessionCheckpoint(
            session_id="sp_test",
            topic="Test"
        )
        
        data = checkpoint.to_dict()
        
        assert isinstance(data, dict)
        assert data["session_id"] == "sp_test"
        assert data["topic"] == "Test"
    
    def test_roundtrip_serialization(self):
        """Checkpoint should survive roundtrip serialization."""
        original = SessionCheckpoint(
            session_id="sp_test",
            topic="Test Topic",
            context="Context"
        )
        original.roles = {"red_role": "Red"}
        original.results = {"discovery": {"roles": {}}}
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = SessionCheckpoint.from_dict(data)
        
        assert restored.session_id == original.session_id
        assert restored.topic == original.topic
        assert restored.roles == original.roles
        assert restored.results == original.results
