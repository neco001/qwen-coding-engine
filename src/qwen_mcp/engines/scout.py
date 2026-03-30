"""
Scout Engine - Centralized Task Complexity & Routing Analysis
"""

import logging
import re
import json
from typing import Optional, Dict, Any, Tuple
from qwen_mcp.api import DashScopeClient
from qwen_mcp.registry import registry

logger = logging.getLogger("qwen_mcp.scout")

class ScoutEngine:
    def __init__(self, client: Optional[DashScopeClient] = None):
        self.client = client or DashScopeClient()

    async def analyze_task(self, prompt: str, context: str = "", task_hint: str = "general", progress_callback=None) -> Dict[str, Any]:
        """
        Intelligently analyzes a task to determine complexity and recommended swarm usage.
        """
        # Truncate context for scout for speed
        scout_prompt = f"""Analyze this {task_hint} request and categorize it:
TASK: {prompt[:2000]}
CONTEXT: {context[:1000]}

Categorize by size:
- low: single function, snippet, shell command, simple answer
- medium: full script, single file refactor, standard audit
- high: complex feature, multi-file changes (>2-3 files), thorough audit
- critical: architecture redesign, entire server boilerplate, massive codebase audit

Rules:
1. Recommend use_swarm=true IF task requires more than 300 lines of code OR touches more than 3 distinct files/modules OR if it's a multi-file audit.
2. Output ONLY JSON:
{{
  "complexity": "low|medium|high|critical",
  "score": 1-10,
  "use_swarm": true|false,
  "reason": "short explanation"
}}
"""
        try:
            if progress_callback:
                await progress_callback(progress=2.0, message=f"[Scout] Analyzing {task_hint} complexity...")
                
            # Use strategist model (highest reasoning)
            messages = [{"role": "user", "content": scout_prompt}]
            raw = await self.client.generate_completion(
                messages, 
                task_type="strategist", 
                complexity="low",
                tags=["scout"],
                progress_callback=progress_callback
            )
            
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                logger.info(f"Scout Result: {data['complexity']} (score: {data['score']}), use_swarm={data['use_swarm']}")
                return data
                
            return {
                "complexity": self._heuristic_complexity(prompt, context),
                "use_swarm": False,
                "reason": "Scout failed to follow JSON format"
            }
        except Exception as e:
            logger.warning(f"Scout failed: {e}")
            return {
                "complexity": self._heuristic_complexity(prompt, context),
                "use_swarm": False,
                "reason": f"Scout error: {str(e)}"
            }

    def _heuristic_complexity(self, prompt: str, context: str) -> str:
        """Fallback word-count based heuristic."""
        combined = prompt + " " + context
        word_count = len(combined.split())
        if word_count > 500: return "critical"
        if word_count > 200: return "high"
        if word_count > 50: return "medium"
        return "low"
