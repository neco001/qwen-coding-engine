"""
Tests for TokenScout - Pre-flight token estimation module.

TDD RED Phase: These tests define expected behavior before implementation.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTokenScout:
    """Test suite for TokenScout token estimation."""
    
    @pytest.fixture
    def scout(self):
        """Create TokenScout instance."""
        from qwen_mcp.engines.token_scout import TokenScout
        return TokenScout()
    
    def test_scout_instantiation(self, scout):
        """TokenScout should instantiate without errors."""
        assert scout is not None
    
    def test_estimate_simple_prompt(self, scout):
        """Simple prompt should return low token estimate."""
        prompt = "Write a hello world function in Python."
        result = scout.estimate_output_tokens(prompt)
        
        assert "estimated_tokens" in result
        assert result["estimated_tokens"] > 0
        assert result["estimated_tokens"] < 2000  # Simple task (base + code indicators)
    
    def test_estimate_complex_prompt(self, scout):
        """Complex multi-file prompt should return high token estimate."""
        prompt = """
        Implement a complete Column Registry System with:
        1. JSON Schema for validation
        2. YAML configuration file with 50+ column mappings
        3. Python ColumnResolver class with resolve(), validate(), rename() methods
        4. Integration hooks in processor.py
        5. Full test suite
        
        Files to modify: ml_processor.py, core/processor.py, config.py, status_tracker.py
        """
        result = scout.estimate_output_tokens(prompt)
        
        assert result["estimated_tokens"] > 5000  # Complex task needs many tokens
    
    def test_estimate_with_context_files(self, scout):
        """Prompt with file context should estimate higher."""
        prompt = "Refactor this module"
        context = "def process(): pass\n" * 100  # 100 lines of context
        
        result_simple = scout.estimate_output_tokens(prompt)
        result_with_context = scout.estimate_output_tokens(prompt, context=context)
        
        assert result_with_context["estimated_tokens"] > result_simple["estimated_tokens"]
    
    def test_should_decompose_returns_false_for_small_task(self, scout):
        """Small tasks should not require decomposition."""
        prompt = "Add a docstring to this function."
        result = scout.estimate_output_tokens(prompt)
        
        should_decompose = scout.should_decompose(result["estimated_tokens"])
        assert should_decompose is False
    
    def test_should_decompose_returns_true_for_large_task(self, scout):
        """Large tasks (>50K tokens) should require decomposition."""
        # Simulate a very large estimated output
        large_estimate = 75000  # tokens
        
        should_decompose = scout.should_decompose(large_estimate)
        assert should_decompose is True
    
    def test_threshold_configurable(self, scout):
        """Decomposition threshold should be configurable."""
        from qwen_mcp.engines.token_scout import TokenScout
        
        # Default threshold
        scout_default = TokenScout()
        assert scout_default.decomposition_threshold == 50000
        
        # Custom threshold
        scout_custom = TokenScout(decomposition_threshold=100000)
        assert scout_custom.decomposition_threshold == 100000
    
    def test_caching_works(self, scout):
        """Identical prompts should use cached estimates."""
        prompt = "Write a function to sort a list."
        
        result1 = scout.estimate_output_tokens(prompt)
        result2 = scout.estimate_output_tokens(prompt)
        
        # Should return same result
        assert result1 == result2
        # Cache should be used (check internal cache - key includes context separator)
        cache_key = f"{prompt}|||"
        assert cache_key in scout._cache


class TestTokenScoutIntegration:
    """Integration tests for TokenScout with real prompts."""
    
    @pytest.fixture
    def scout(self):
        from qwen_mcp.engines.token_scout import TokenScout
        return TokenScout()
    
    def test_swarm_style_prompt(self, scout):
        """Swarm-style multi-task prompt should estimate high."""
        prompt = """
        ## Tasks to Execute in Parallel
        
        ### Task 1: Create JSON Schema
        Create `schemas/column_registry_schema.json` with validation rules.
        
        ### Task 2: Create YAML Config
        Create `config/column_registry.yaml` with 50+ column mappings.
        
        ### Task 3: Implement Python Class
        Create `utils/column_resolver.py` with ColumnResolver class.
        
        ### Task 4: Add Integration Hooks
        Modify `core/processor.py` to use ColumnResolver.
        
        ### Task 5: Create Tests
        Create `tests/test_column_resolver.py` with full coverage.
        """
        
        result = scout.estimate_output_tokens(prompt)
        
        # Multi-file implementation should estimate high
        assert result["estimated_tokens"] > 10000
        # Should NOT recommend decomposition (12K < 50K threshold)
        # Decomposition is for VERY large tasks (>50K tokens)
        assert not scout.should_decompose(result["estimated_tokens"])
    
    def test_large_context_prompt_needs_decomposition(self, scout):
        """Large prompts should trigger decomposition."""
        from qwen_mcp.engines.token_scout import TokenScout
        
        scout_low = TokenScout(decomposition_threshold=10000)
        
        # Simulate a massive prompt that would exceed threshold
        prompt = """
        Task 1: Implement complete user authentication system with JWT
        Task 2: Create database models for User, Role, Permission, Session
        Task 3: Build REST API endpoints for login, register, logout, refresh
        Task 4: Add password hashing with bcrypt and validation
        Task 5: Implement role-based access control middleware
        Task 6: Create unit tests for all authentication components
        Task 7: Add logging and audit trail for security events
        Task 8: Build frontend login form with validation
        Task 9: Create user profile management page
        Task 10: Add email verification workflow
        Task 11: Implement password reset functionality
        Task 12: Add two-factor authentication support
        
        Files to modify:
        - src/auth/jwt_handler.py
        - src/auth/password_hasher.py
        - src/models/user.py
        - src/models/role.py
        - src/models/permission.py
        - src/api/auth_routes.py
        - src/api/user_routes.py
        - src/middleware/auth_middleware.py
        - src/tests/auth_test.py
        - src/tests/user_test.py
        - frontend/components/LoginForm.tsx
        - frontend/components/ProfilePage.tsx
        """
        
        result = scout_low.estimate_output_tokens(prompt)
        
        # Should have many factors
        assert len(result["factors"]) >= 4
        
        # Should estimate large output (12 tasks + 12 files)
        # 12 × 1500 + 12 × 800 + base + code indicators = ~30K+
        assert result["estimated_tokens"] > 10000
        
        # Should need decomposition with low threshold
        assert result["should_decompose"] is True
    
    def test_recursive_decomposer_large_prompt(self, scout):
        """Test RecursiveDecomposer with large multi-task prompt."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=10000)
        
        prompt = """
        Task 1: Create file auth.py with UserAuth class
        Task 2: Create file models.py with User model
        Task 3: Create file api.py with login endpoint
        Task 4: Create file tests.py with unit tests
        """
        
        plan = decomposer.create_plan(prompt)
        
        # Should decompose into subtasks
        assert len(plan["subtasks"]) >= 1
        
        # Should have total estimated tokens
        assert plan["total_estimated_tokens"] > 0
        
        # Should track decomposition depth
        assert plan["decomposition_depth"] >= 0
        
        # Should have threshold recorded
        assert plan["threshold"] == 10000