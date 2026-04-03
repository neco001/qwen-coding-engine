"""
Tests for RecursiveDecomposer - TDD Phase 1: RED

RecursiveDecomposer should:
1. Check if estimated output tokens > threshold (50K)
2. If yes, decompose task into smaller subtasks
3. Recursively check each subtask
4. Have depth limit of 3 to prevent infinite recursion
5. Return decomposition plan with atomic subtasks
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src"))


class TestRecursiveDecomposerInstantiation:
    """Test basic instantiation and configuration."""
    
    def test_instantiate_with_defaults(self):
        """Should instantiate with default threshold and depth limit."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer()
        assert decomposer.threshold == 50000
        assert decomposer.max_depth == 3
    
    def test_instantiate_with_custom_threshold(self):
        """Should accept custom threshold."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=100000)
        assert decomposer.threshold == 100000
        assert decomposer.max_depth == 3
    
    def test_instantiate_with_custom_depth_limit(self):
        """Should accept custom depth limit."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(max_depth=5)
        assert decomposer.threshold == 50000
        assert decomposer.max_depth == 5


class TestRecursiveDecomposerAnalysis:
    """Test task analysis and decomposition decision."""
    
    def test_simple_task_no_decomposition(self):
        """Simple task below threshold should not decompose."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer()
        simple_prompt = "Write a function to add two numbers"
        
        result = decomposer.analyze(simple_prompt)
        
        assert result["needs_decomposition"] is False
        assert result["depth"] == 0
        assert result["subtasks"] == []
    
    def test_large_task_needs_decomposition(self):
        """Large task above threshold should need decomposition."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=10000)  # Lower threshold for test
        large_prompt = """
        Task 1: Implement a complete user authentication system with JWT tokens
        Task 2: Create database models for User, Role, and Permission
        Task 3: Build REST API endpoints for login, register, logout
        Task 4: Add password hashing and validation
        Task 5: Implement role-based access control
        Task 6: Create unit tests for all components
        Task 7: Add logging and audit trail
        Task 8: Build frontend login form
        Task 9: Create user profile management
        Task 10: Add email verification workflow
        """
        
        result = decomposer.analyze(large_prompt)
        
        assert result["needs_decomposition"] is True
        assert result["depth"] == 0
        assert result["estimated_tokens"] > 10000


class TestRecursiveDecomposerDecomposition:
    """Test actual decomposition logic."""
    
    def test_decompose_returns_subtasks(self):
        """Decomposition should return list of subtasks."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=5000)
        multi_task_prompt = """
        Task 1: Create file auth.py with UserAuth class
        Task 2: Create file models.py with User model
        Task 3: Create file api.py with login endpoint
        """
        
        result = decomposer.decompose(multi_task_prompt)
        
        assert len(result["subtasks"]) >= 1
        assert all("prompt" in st for st in result["subtasks"])
        assert all("estimated_tokens" in st for st in result["subtasks"])
    
    def test_decompose_respects_depth_limit(self):
        """Decomposition should stop at max_depth."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=100, max_depth=2)
        # Very low threshold will force multiple decomposition attempts
        huge_prompt = "Build a complete enterprise ERP system with 50 modules"
        
        result = decomposer.decompose(huge_prompt, depth=0)
        
        # Should not exceed max_depth
        max_found_depth = max(
            st.get("depth", 0) for st in result["subtasks"]
        ) if result["subtasks"] else 0
        assert max_found_depth <= decomposer.max_depth
    
    def test_atomic_task_not_decomposed(self):
        """Atomic tasks (below threshold) should not be further decomposed."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=5000)
        atomic_prompt = "Write a simple hello world function"
        
        result = decomposer.decompose(atomic_prompt)
        
        # Atomic task should return itself as single subtask
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["prompt"] == atomic_prompt
        assert result["subtasks"][0]["atomic"] is True


class TestRecursiveDecomposerSafety:
    """Test safety mechanisms."""
    
    def test_depth_limit_prevents_infinite_recursion(self):
        """Depth limit should prevent infinite recursion."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        # Extremely low threshold would cause infinite recursion without limit
        decomposer = RecursiveDecomposer(threshold=10, max_depth=3)
        
        # This would normally keep decomposing forever
        recursive_prompt = "Build everything"
        
        result = decomposer.decompose(recursive_prompt, depth=0)
        
        # Should complete without hanging
        assert result is not None
        assert "subtasks" in result
        assert "depth_exceeded" in result  # Flag indicating depth limit hit
    
    def test_empty_prompt_handled(self):
        """Empty prompt should be handled gracefully."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer()
        result = decomposer.analyze("")
        
        assert result["needs_decomposition"] is False
        assert result["estimated_tokens"] == 0


class TestRecursiveDecomposerIntegration:
    """Test integration with TokenScout."""
    
    def test_uses_token_scout_for_estimation(self):
        """Should use TokenScout for token estimation."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer()
        prompt = "Write a Python class with 10 methods"
        
        result = decomposer.analyze(prompt)
        
        # Should have estimation from TokenScout
        assert "estimated_tokens" in result
        assert result["estimated_tokens"] > 0
        assert "factors" in result  # TokenScout provides factors
    
    def test_decomposition_plan_structure(self):
        """Decomposition plan should have proper structure."""
        from qwen_mcp.engines.recursive_decomposer import RecursiveDecomposer
        
        decomposer = RecursiveDecomposer(threshold=5000)
        prompt = """
        Task 1: Implement authentication
        Task 2: Create database models
        Task 3: Build API endpoints
        """
        
        plan = decomposer.create_plan(prompt)
        
        assert "original_prompt" in plan
        assert "total_estimated_tokens" in plan
        assert "subtasks" in plan
        assert "decomposition_depth" in plan
        assert plan["original_prompt"] == prompt