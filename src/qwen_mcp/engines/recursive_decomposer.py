"""
RecursiveDecomposer - Token-aware task decomposition with recursion safety.

This module implements the user's vision:
- Output up to 199,999 tokens is acceptable
- If estimated output > threshold (50K), decompose into atomic subtasks
- Recursively check each subtask until all fit within budget
- Depth limit of 3 prevents infinite recursion

Key insight: max_tokens is a HARD CUTOFF, not a budget hint.
We estimate BEFORE execution to prevent truncation.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from .token_scout import TokenScout, DEFAULT_DECOMPOSITION_THRESHOLD

logger = logging.getLogger("qwen_mcp.recursive_decomposer")

# Default maximum recursion depth
DEFAULT_MAX_DEPTH = 3


class RecursiveDecomposer:
    """
    Decomposes tasks recursively until each subtask fits within token budget.
    
    Workflow:
    1. Use TokenScout to estimate output tokens
    2. If estimated > threshold, decompose into subtasks
    3. Recursively analyze each subtask
    4. Stop at max_depth to prevent infinite recursion
    5. Return decomposition plan with atomic subtasks
    """
    
    def __init__(
        self,
        threshold: int = DEFAULT_DECOMPOSITION_THRESHOLD,
        max_depth: int = DEFAULT_MAX_DEPTH
    ):
        """
        Initialize RecursiveDecomposer.
        
        Args:
            threshold: Token count above which decomposition is triggered
            max_depth: Maximum recursion depth for decomposition
        """
        self.threshold = threshold
        self.max_depth = max_depth
        self.scout = TokenScout(decomposition_threshold=threshold)
    
    def analyze(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze a prompt to determine if decomposition is needed.
        
        Args:
            prompt: The user prompt/task description
            context: Additional context (file contents, etc.)
        
        Returns:
            Dict with:
            - needs_decomposition: bool
            - estimated_tokens: int
            - depth: Current depth (0 for initial analysis)
            - subtasks: List of subtasks (empty if no decomposition needed)
            - factors: Estimation factors from TokenScout
        """
        if not prompt:
            return {
                "needs_decomposition": False,
                "estimated_tokens": 0,
                "depth": 0,
                "subtasks": [],
                "factors": []
            }
        
        # Use TokenScout for estimation
        estimation = self.scout.estimate_output_tokens(prompt, context)
        estimated_tokens = estimation["estimated_tokens"]
        
        needs_decomposition = self.scout.should_decompose(estimated_tokens)
        
        return {
            "needs_decomposition": needs_decomposition,
            "estimated_tokens": estimated_tokens,
            "depth": 0,
            "subtasks": [],
            "factors": estimation.get("factors", []),
            "confidence": estimation.get("confidence", "medium")
        }
    
    def decompose(self, prompt: str, context: str = "", depth: int = 0) -> Dict[str, Any]:
        """
        Decompose a prompt into subtasks if needed.
        
        Args:
            prompt: The user prompt/task description
            context: Additional context
            depth: Current recursion depth
        
        Returns:
            Dict with:
            - subtasks: List of decomposed subtasks
            - depth_exceeded: bool (True if max_depth was hit)
            - original_prompt: The original prompt
            - total_estimated_tokens: Sum of all subtask estimates
        """
        # Analyze the prompt
        analysis = self.analyze(prompt, context)
        estimated_tokens = analysis["estimated_tokens"]
        
        # Check if decomposition is needed
        if not analysis["needs_decomposition"]:
            # Task is atomic - return as single subtask
            return {
                "subtasks": [{
                    "prompt": prompt,
                    "estimated_tokens": estimated_tokens,
                    "atomic": True,
                    "depth": depth
                }],
                "depth_exceeded": False,
                "original_prompt": prompt,
                "total_estimated_tokens": estimated_tokens
            }
        
        # Check depth limit
        if depth >= self.max_depth:
            # Cannot decompose further - return as atomic with warning
            logger.warning(f"Depth limit ({self.max_depth}) reached for prompt: {prompt[:100]}...")
            return {
                "subtasks": [{
                    "prompt": prompt,
                    "estimated_tokens": estimated_tokens,
                    "atomic": True,
                    "depth": depth,
                    "depth_limit_hit": True
                }],
                "depth_exceeded": True,
                "original_prompt": prompt,
                "total_estimated_tokens": estimated_tokens
            }
        
        # Decompose the prompt into subtasks
        raw_subtasks = self._extract_subtasks(prompt)
        
        # Recursively analyze each subtask
        decomposed_subtasks = []
        total_tokens = 0
        
        for subtask_prompt in raw_subtasks:
            # Recursively decompose each subtask
            subtask_result = self.decompose(subtask_prompt, "", depth + 1)
            
            # Flatten subtasks from recursive decomposition
            for st in subtask_result["subtasks"]:
                decomposed_subtasks.append(st)
                total_tokens += st["estimated_tokens"]
        
        return {
            "subtasks": decomposed_subtasks,
            "depth_exceeded": any(st.get("depth_limit_hit", False) for st in decomposed_subtasks),
            "original_prompt": prompt,
            "total_estimated_tokens": total_tokens,
            "decomposition_depth": depth + 1
        }
    
    def create_plan(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """
        Create a complete decomposition plan for a prompt.
        
        Args:
            prompt: The user prompt/task description
            context: Additional context
        
        Returns:
            Dict with:
            - original_prompt: The original prompt
            - total_estimated_tokens: Sum of all subtask estimates
            - subtasks: List of atomic subtasks
            - decomposition_depth: Maximum depth reached
            - threshold: The decomposition threshold used
        """
        result = self.decompose(prompt, context, depth=0)
        
        # Calculate max depth reached
        max_depth = max(
            st.get("depth", 0) for st in result["subtasks"]
        ) if result["subtasks"] else 0
        
        return {
            "original_prompt": prompt,
            "total_estimated_tokens": result["total_estimated_tokens"],
            "subtasks": result["subtasks"],
            "decomposition_depth": max_depth,
            "threshold": self.threshold,
            "depth_exceeded": result["depth_exceeded"]
        }
    
    def _extract_subtasks(self, prompt: str) -> List[str]:
        """
        Extract subtasks from a multi-task prompt.
        
        Strategies:
        1. Look for explicit "Task N:" markers
        2. Look for numbered lists
        3. Look for bullet points with action verbs
        4. Split by logical sections (### headers)
        
        Args:
            prompt: The prompt to decompose
        
        Returns:
            List of subtask prompts
        """
        subtasks = []
        
        # Strategy 1: Explicit "Task N:" markers
        task_pattern = r'(?:^|\n)\s*(?:###\s*)?[Tt]ask\s*\d+[:\.]\s*(.+?)(?=(?:Task\s*\d+[:\.])|$)'
        task_matches = re.findall(task_pattern, prompt, re.DOTALL)
        
        if task_matches:
            return [m.strip() for m in task_matches if m.strip()]
        
        # Strategy 2: Numbered list items
        list_pattern = r'(?:^|\n)\s*\d+\.\s+([A-Z][^\n]+(?:\n(?!\d\.)[^\n]*)*)'
        list_matches = re.findall(list_pattern, prompt)
        
        if list_matches and len(list_matches) >= 2:
            return [m.strip() for m in list_matches if m.strip()]
        
        # Strategy 3: Bullet points with action verbs
        bullet_pattern = r'(?:^|\n)\s*[-*]\s+(Create|Implement|Add|Modify|Write|Build|Update|Fix|Refactor)[^\n]+'
        bullet_matches = re.findall(bullet_pattern, prompt, re.IGNORECASE)
        
        if bullet_matches and len(bullet_matches) >= 2:
            return [m.strip() for m in bullet_matches if m.strip()]
        
        # Strategy 4: Split by ### headers
        header_pattern = r'###\s+([^\n]+)'
        headers = re.findall(header_pattern, prompt)
        
        if headers and len(headers) >= 2:
            # Split by headers
            sections = re.split(r'###\s+[^\n]+', prompt)
            # Combine headers with their content
            result = []
            for i, header in enumerate(headers):
                if i + 1 < len(sections):
                    content = sections[i + 1].strip()
                    if content:
                        result.append(f"{header}: {content}")
            return result if result else [prompt]
        
        # Strategy 5: Split by sentences for very long prompts
        if len(prompt) > 1000:
            # Split by periods followed by space and capital letter
            sentences = re.split(r'\.\s+(?=[A-Z])', prompt)
            if len(sentences) >= 2:
                return [s.strip() + "." for s in sentences if s.strip()]
        
        # Fallback: Return original prompt as single task
        return [prompt]


# Convenience function
def decompose_prompt(prompt: str, threshold: int = DEFAULT_DECOMPOSITION_THRESHOLD) -> Dict[str, Any]:
    """
    Quick decomposition of a prompt.
    
    Args:
        prompt: The user prompt
        threshold: Decomposition threshold
    
    Returns:
        Decomposition plan
    """
    decomposer = RecursiveDecomposer(threshold=threshold)
    return decomposer.create_plan(prompt)