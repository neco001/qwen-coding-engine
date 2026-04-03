"""
Tests for identity.py client_source parameter.

TDD RED Phase: These tests should FAIL because get_current_project_id()
doesn't accept client_source parameter yet.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen_mcp.specter.identity import (
    get_session_id,
    get_current_project_id,
    get_or_create_instance_id,
    _compute_workspace_hash,
    _sanitize_client_source
)


class TestSanitizeClientSource:
    """Tests for client_source sanitization to prevent injection."""
    
    def test_sanitize_valid_source(self):
        """Valid client source should pass through unchanged."""
        assert _sanitize_client_source("gemini") == "gemini"
        assert _sanitize_client_source("roocode") == "roocode"
        assert _sanitize_client_source("default") == "default"
    
    def test_sanitize_source_with_special_chars(self):
        """Special characters should be removed, alphanumeric and hyphen/underscore kept."""
        assert _sanitize_client_source("gemini<script>") == "geminiscript"
        assert _sanitize_client_source("roo-code!@#") == "roo-code"  # hyphen is safe
        assert _sanitize_client_source("test spaces") == "testspaces"
    
    def test_sanitize_long_source(self):
        """Long client source should be truncated to 32 chars."""
        long_source = "a" * 50
        assert len(_sanitize_client_source(long_source)) == 32
    
    def test_sanitize_empty_source(self):
        """Empty or invalid source should return 'default'."""
        assert _sanitize_client_source("") == "default"
        assert _sanitize_client_source("!!!") == "default"


class TestGetSessionIdWithClientSource:
    """Tests for session ID generation with client_source."""
    
    def test_session_id_format_includes_client_source(self):
        """Session ID should include client_source in format: {instance}_{source}_{hash}."""
        instance_id = "abc12345"
        client_source = "gemini"
        cwd = "/test/workspace"
        
        session_id = get_session_id(instance_id, client_source, cwd)
        
        # Should be format: {instanceId}_{clientSource}_{workspaceHash}
        assert session_id.startswith("abc12345_gemini_")
        assert len(session_id) == 8 + 1 + 6 + 1 + 8  # instance + _ + source + _ + hash
    
    def test_session_id_different_client_sources(self):
        """Different client sources should produce different session IDs."""
        instance_id = "abc12345"
        cwd = "/test/workspace"
        
        gemini_id = get_session_id(instance_id, "gemini", cwd)
        roocode_id = get_session_id(instance_id, "roocode", cwd)
        
        # Same instance and workspace, different client source
        assert gemini_id != roocode_id
        assert gemini_id.startswith("abc12345_gemini_")
        assert roocode_id.startswith("abc12345_roocode_")
    
    def test_session_id_same_workspace_hash(self):
        """Same workspace should produce same hash regardless of client source."""
        instance_id = "abc12345"
        cwd = "/test/workspace"
        
        gemini_id = get_session_id(instance_id, "gemini", cwd)
        roocode_id = get_session_id(instance_id, "roocode", cwd)
        
        # Extract hash portion (last 8 chars after last underscore)
        gemini_hash = gemini_id.split("_")[-1]
        roocode_hash = roocode_id.split("_")[-1]
        
        assert gemini_hash == roocode_hash


class TestGetCurrentProjectIdWithClientSource:
    """Tests for get_current_project_id with client_source parameter."""
    
    def test_get_current_project_id_accepts_client_source(self):
        """get_current_project_id should accept client_source parameter."""
        # This test will FAIL because current implementation doesn't have this param
        with patch.dict(os.environ, {"QWEN_INSTANCE_ID": "test1234"}, clear=False):
            with patch('os.getcwd', return_value="/test/workspace"):
                # Should accept client_source parameter
                project_id = get_current_project_id(client_source="gemini")
                
                assert "gemini" in project_id
                assert project_id.startswith("test1234_gemini_")
    
    def test_get_current_project_id_default_client_source(self):
        """get_current_project_id should use 'default' as fallback client_source."""
        with patch.dict(os.environ, {"QWEN_INSTANCE_ID": "test1234"}, clear=False):
            with patch('os.getcwd', return_value="/test/workspace"):
                # Without client_source param, should use 'default'
                project_id = get_current_project_id()
                
                assert "default" in project_id
                assert project_id.startswith("test1234_default_")
    
    def test_get_current_project_id_env_override(self):
        """QWEN_PROJECT_NAME env var should override everything."""
        with patch.dict(os.environ, {
            "QWEN_PROJECT_NAME": "custom_project",
            "QWEN_INSTANCE_ID": "test1234"
        }, clear=False):
            with patch('os.getcwd', return_value="/test/workspace"):
                # Env override should ignore client_source
                project_id = get_current_project_id(client_source="gemini")
                
                assert project_id == "custom_project"


class TestInstanceIdGeneration:
    """Tests for instance ID generation."""
    
    def test_instance_id_from_env(self):
        """Instance ID should be read from QWEN_INSTANCE_ID env var."""
        with patch.dict(os.environ, {"QWEN_INSTANCE_ID": "env_instance"}, clear=False):
            # Reset cached value
            import qwen_mcp.specter.identity as identity_module
            identity_module._cached_instance_id = None
            
            instance_id = get_or_create_instance_id()
            
            assert instance_id == "env_instance"
    
    def test_instance_id_uuid_generation(self):
        """Instance ID should generate UUID if no env var."""
        with patch.dict(os.environ, {}, clear=True):
            # Reset cached value
            import qwen_mcp.specter.identity as identity_module
            identity_module._cached_instance_id = None
            
            instance_id = get_or_create_instance_id()
            
            # Should be 8-char hex
            assert len(instance_id) == 8
            assert all(c in '0123456789abcdef' for c in instance_id)
    
    def test_instance_id_caching(self):
        """Instance ID should be cached after first generation."""
        with patch.dict(os.environ, {}, clear=True):
            import qwen_mcp.specter.identity as identity_module
            identity_module._cached_instance_id = None
            
            id1 = get_or_create_instance_id()
            id2 = get_or_create_instance_id()
            
            assert id1 == id2


class TestWorkspaceHash:
    """Tests for workspace hash computation."""
    
    def test_workspace_hash_consistent(self):
        """Same workspace should produce same hash."""
        cwd = "/test/workspace"
        
        hash1 = _compute_workspace_hash(cwd)
        hash2 = _compute_workspace_hash(cwd)
        
        assert hash1 == hash2
        assert len(hash1) == 8
    
    def test_workspace_hash_different_paths(self):
        """Different workspaces should produce different hashes."""
        hash1 = _compute_workspace_hash("/workspace1")
        hash2 = _compute_workspace_hash("/workspace2")
        
        assert hash1 != hash2
    
    def test_workspace_hash_case_insensitive(self):
        """Workspace hash should be case-insensitive (normalized)."""
        hash1 = _compute_workspace_hash("/Test/Workspace")
        hash2 = _compute_workspace_hash("/test/workspace")
        
        # Normalized to lowercase, should be same
        assert hash1 == hash2


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])