"""
TokenScout - Pre-flight token estimation for task decomposition.

This module estimates output token count BEFORE execution to:
1. Prevent truncation by setting appropriate max_tokens
2. Trigger decomposition for tasks exceeding threshold
3. Remove arbitrary hardcoded token limits

Key insight: max_tokens is a HARD CUTOFF, not a budget hint.
The model doesn't "plan" output size - it generates until stopped.
"""

import logging
import re
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger("qwen_mcp.token_scout")

# Default threshold for decomposition (50K tokens)
DEFAULT_DECOMPOSITION_THRESHOLD = 50000

# Safety ceiling - maximum tokens we'll ever request (buffer below 65536 API limit)
SAFETY_MAX_TOKENS = 60000


class TokenScout:
    """
    Estimates output token count for a given prompt.
    
    Uses heuristic-based estimation (no external API calls for speed).
    Estimation factors:
    - Prompt length and complexity
    - Number of distinct tasks
    - File references (each file ~ adds tokens)
    - Code vs prose ratio
    """
    
    def __init__(self, decomposition_threshold: int = DEFAULT_DECOMPOSITION_THRESHOLD):
        """
        Initialize TokenScout.
        
        Args:
            decomposition_threshold: Token count above which decomposition is recommended
        """
        self.decomposition_threshold = decomposition_threshold
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def estimate_output_tokens(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """
        Estimate the output token count for a given prompt.
        
        Args:
            prompt: The user prompt/task description
            context: Additional context (file contents, etc.)
        
        Returns:
            Dict with:
            - estimated_tokens: Estimated output token count
            - confidence: "low" | "medium" | "high"
            - factors: List of factors considered
            - should_decompose: bool
        """
        # Check cache
        cache_key = f"{prompt}|||{context[:500]}"  # Truncate context for cache key
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        factors = []
        base_estimate = 500  # Minimum output for any task
        
        # Factor 1: Prompt length (input tokens ~ output tokens for many tasks)
        prompt_tokens = self._count_tokens_heuristic(prompt)
        length_factor = prompt_tokens * 2  # Output often 2x input for code generation
        factors.append(f"prompt_length: {prompt_tokens} tokens")
        
        # Factor 2: Number of distinct tasks
        task_count = self._count_tasks(prompt)
        if task_count > 1:
            task_factor = task_count * 1500  # Each task ~ 1500 tokens
            factors.append(f"task_count: {task_count} tasks × 1500")
        else:
            task_factor = 0
        
        # Factor 3: File references
        file_refs = self._count_file_references(prompt)
        if file_refs > 0:
            file_factor = file_refs * 800  # Each file ~ 800 tokens of output
            factors.append(f"file_refs: {file_refs} files × 800")
        else:
            file_factor = 0
        
        # Factor 4: Code indicators
        code_indicators = self._count_code_indicators(prompt)
        code_factor = code_indicators * 500
        factors.append(f"code_indicators: {code_indicators} × 500")
        
        # Factor 5: Context size
        if context:
            context_tokens = self._count_tokens_heuristic(context)
            context_factor = context_tokens * 0.5  # Context adds ~50% to output
            factors.append(f"context_size: {context_tokens} tokens")
        else:
            context_factor = 0
        
        # Combine factors
        estimated_tokens = base_estimate + length_factor + task_factor + file_factor + code_factor + context_factor
        
        # Determine confidence
        confidence = "medium"
        if task_count > 3 or file_refs > 3:
            confidence = "low"  # Complex tasks are harder to estimate
        elif task_count == 1 and file_refs == 0:
            confidence = "high"  # Simple tasks are predictable
        
        result = {
            "estimated_tokens": int(estimated_tokens),
            "confidence": confidence,
            "factors": factors,
            "should_decompose": self.should_decompose(estimated_tokens)
        }
        
        # Cache result
        self._cache[cache_key] = result
        
        logger.debug(f"TokenScout estimate: {estimated_tokens} tokens (confidence: {confidence})")
        return result
    
    def should_decompose(self, estimated_tokens: int) -> bool:
        """
        Determine if a task should be decomposed based on estimated token count.
        
        Args:
            estimated_tokens: Estimated output token count
        
        Returns:
            True if task should be decomposed, False otherwise
        """
        return estimated_tokens > self.decomposition_threshold
    
    def get_max_tokens(self, estimated_tokens: int) -> int:
        """
        Get appropriate max_tokens value for a task.
        
        Args:
            estimated_tokens: Estimated output token count
        
        Returns:
            max_tokens value (estimated + 50% buffer, capped at safety limit)
        """
        # Add 50% buffer for estimation error
        buffered = int(estimated_tokens * 1.5)
        
        # Cap at safety ceiling
        return min(buffered, SAFETY_MAX_TOKENS)
    
    def _count_tokens_heuristic(self, text: str) -> int:
        """
        Estimate token count using heuristic (no tiktoken dependency).
        
        Rule of thumb: ~4 characters per token for English/code.
        """
        if not text:
            return 0
        return len(text) // 4
    
    def _count_tasks(self, prompt: str) -> int:
        """
        Count distinct tasks in a prompt.
        
        Looks for patterns like:
        - "Task 1:", "Task 2:", etc.
        - Numbered lists
        - "### Task" headers
        """
        # Pattern 1: "Task N:" or "### Task N"
        task_pattern = r'(?:^|\n)\s*(?:###\s*)?[Tt]ask\s*\d+[:\.]'
        task_matches = len(re.findall(task_pattern, prompt))
        
        if task_matches > 0:
            return task_matches
        
        # Pattern 2: Numbered list items (1., 2., etc.)
        list_pattern = r'(?:^|\n)\s*\d+\.\s+[A-Z]'
        list_matches = len(re.findall(list_pattern, prompt))
        
        if list_matches > 0:
            return list_matches
        
        # Pattern 3: Bullet points with action verbs
        bullet_pattern = r'(?:^|\n)\s*[-*]\s+(?:Create|Implement|Add|Modify|Write|Build|Update)'
        bullet_matches = len(re.findall(bullet_pattern, prompt, re.IGNORECASE))
        
        return bullet_matches if bullet_matches > 0 else 1
    
    def _count_file_references(self, prompt: str) -> int:
        """
        Count file references in a prompt.
        
        Looks for patterns like:
        - `filename.py`
        - "file.py"
        - path/to/file.ext
        """
        # Pattern: filename with extension
        file_pattern = r'[\w/\\-]+\.[\w]+'
        matches = re.findall(file_pattern, prompt)
        
        # Filter to common code files
        code_extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.md', '.txt', '.toml'}
        code_files = [m for m in matches if any(m.endswith(ext) for ext in code_extensions)]
        
        return len(set(code_files))  # Unique files only
    
    def _count_code_indicators(self, prompt: str) -> int:
        """
        Count code-related indicators in a prompt.
        
        Looks for:
        - Code blocks (```)
        - Function/class definitions
        - Import statements
        - Code keywords (def, class, function, etc.)
        """
        count = 0
        
        # Code blocks
        count += prompt.count('```')
        
        # Code keywords
        keywords = ['def ', 'class ', 'function ', 'import ', 'from ', 'return ', 'async ']
        for kw in keywords:
            count += prompt.lower().count(kw)
        
        return count


# Convenience function for direct use
def estimate_tokens(prompt: str, context: str = "") -> int:
    """
    Quick estimation of output tokens for a prompt.
    
    Args:
        prompt: The user prompt
        context: Additional context
    
    Returns:
        Estimated token count
    """
    scout = TokenScout()
    result = scout.estimate_output_tokens(prompt, context)
    return result["estimated_tokens"]