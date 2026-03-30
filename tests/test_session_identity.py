import pytest
import hashlib
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions we expect to exist (but don't yet)
from qwen_mcp.specter.identity import get_session_id, generate_instance_id


def compute_workspace_hash(cwd: str) -> str:
    """Helper to compute workspace hash as expected by the implementation."""
    try:
        normalized = os.path.realpath(cwd).lower()
    except Exception:
        normalized = os.path.normpath(cwd).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


class TestSessionIdentity:
    """Tests for session identity generation and formatting."""

    def test_get_session_id_format(self, tmp_path):
        """Verify session ID follows {instanceId}_{clientSource}_{workspaceHash} format."""
        instance_id = "testinst"
        client_source = "gemini"
        cwd = str(tmp_path)

        session_id = get_session_id(instance_id, client_source, cwd)
        expected_hash = compute_workspace_hash(cwd)

        assert session_id == f"{instance_id}_{client_source}_{expected_hash}"

    def test_generate_instance_id_returns_unique_8_char(self):
        """Ensure generate_instance_id() returns a unique 8-character string."""
        ids = [generate_instance_id() for _ in range(100)]

        # Check length
        assert all(len(i) == 8 for i in ids)

        # Check uniqueness
        assert len(set(ids)) == len(ids)

        # Verify it's hex-like (UUID4-ish)
        for id_str in ids:
            assert all(c in "0123456789abcdef" for c in id_str)

    def test_different_instance_ids_produce_different_session_ids(self, tmp_path):
        """Different instance_ids must yield different session IDs, even with same cwd/client_source."""
        client_source = "gemini"
        cwd = str(tmp_path)

        session_id_1 = get_session_id("instA", client_source, cwd)
        session_id_2 = get_session_id("instB", client_source, cwd)

        assert session_id_1 != session_id_2
        expected_hash = compute_workspace_hash(cwd)
        assert session_id_1.endswith(f"_gemini_{expected_hash}")
        assert session_id_2.endswith(f"_gemini_{expected_hash}")

    def test_different_client_sources_produce_different_session_ids(self, tmp_path):
        """Different client_sources must yield different session IDs, even with same instance_id/cwd."""
        instance_id = "testinst"
        cwd = str(tmp_path)

        session_id_gemini = get_session_id(instance_id, "gemini", cwd)
        session_id_roo = get_session_id(instance_id, "roocode", cwd)

        assert session_id_gemini != session_id_roo
        expected_hash = compute_workspace_hash(cwd)
        assert session_id_gemini == f"{instance_id}_gemini_{expected_hash}"
        assert session_id_roo == f"{instance_id}_roocode_{expected_hash}"

    def test_same_params_produce_same_session_id(self, tmp_path):
        """Same parameters should always produce the same session ID."""
        instance_id = "testinst"
        client_source = "gemini"
        cwd = str(tmp_path)

        session_id_1 = get_session_id(instance_id, client_source, cwd)
        session_id_2 = get_session_id(instance_id, client_source, cwd)

        assert session_id_1 == session_id_2