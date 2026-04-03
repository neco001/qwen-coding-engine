"""Tests for AI Enhancement Layer."""
import pytest
from pathlib import Path
from src.graph.ai_enhancer import AIEnhancer


@pytest.fixture
def enhancer():
    """Create an AIEnhancer instance."""
    return AIEnhancer()


@pytest.mark.asyncio
async def test_enhance_dependencies_adds_ai_suggestions(enhancer):
    """enhance_dependencies adds AI suggestions to dependency analysis."""
    static_analysis = {
        "files": {
            "main.py": {"imports": ["utils", "services"]},
            "utils.py": {"imports": []},
        },
        "dependencies": [
            {"from_file": "main.py", "to_file": "utils.py", "import": "utils"}
        ],
    }
    
    result = await enhancer.enhance_dependencies(static_analysis)
    
    assert "suggestions" in result
    assert isinstance(result["suggestions"], list)


@pytest.mark.asyncio
async def test_enhance_dependencies_detects_tight_coupling(enhancer):
    """detects tight coupling patterns."""
    static_analysis = {
        "files": {
            "a.py": {"imports": ["b", "c", "d", "e"]},
            "b.py": {"imports": []},
            "c.py": {"imports": []},
            "d.py": {"imports": []},
            "e.py": {"imports": []},
        },
        "dependencies": [
            {"from_file": "a.py", "to_file": "b.py", "import": "b"},
            {"from_file": "a.py", "to_file": "c.py", "import": "c"},
            {"from_file": "a.py", "to_file": "d.py", "import": "d"},
            {"from_file": "a.py", "to_file": "e.py", "import": "e"},
        ],
    }
    
    result = await enhancer.enhance_dependencies(static_analysis)
    
    # Should detect tight coupling (many dependencies from one file)
    suggestions = result.get("suggestions", [])
    assert any("coupling" in str(s).lower() for s in suggestions)


@pytest.mark.asyncio
async def test_enhance_dependencies_detects_missing_dependencies(enhancer):
    """detects potential missing dependencies."""
    static_analysis = {
        "files": {
            "main.py": {"imports": ["os"]},
            "utils.py": {"imports": ["main"]},  # Circular/suspicious
        },
        "dependencies": [
            {"from_file": "utils.py", "to_file": "main.py", "import": "main"}
        ],
    }
    
    result = await enhancer.enhance_dependencies(static_analysis)
    
    # Should have suggestions about the dependency pattern
    assert "suggestions" in result


@pytest.mark.asyncio
async def test_get_risk_score_returns_value(enhancer):
    """get_risk_score returns a risk score for changes."""
    change_info = {
        "modified_files": ["main.py", "utils.py"],
        "lines_changed": 100,
        "dependencies_affected": 5,
    }
    
    score = await enhancer.get_risk_score(change_info)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_get_risk_score_high_for_large_changes(enhancer):
    """get_risk_score returns higher score for large changes."""
    small_change = {
        "modified_files": ["one.py"],
        "lines_changed": 10,
        "dependencies_affected": 1,
    }
    
    large_change = {
        "modified_files": ["a.py", "b.py", "c.py", "d.py", "e.py"],
        "lines_changed": 500,
        "dependencies_affected": 20,
    }
    
    small_score = await enhancer.get_risk_score(small_change)
    large_score = await enhancer.get_risk_score(large_change)
    
    assert large_score > small_score
