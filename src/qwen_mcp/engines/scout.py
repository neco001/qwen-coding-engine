"""
Scout Engine - Centralized Task Complexity & Routing Analysis
"""

import os
import re
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from qwen_mcp.api import DashScopeClient
from qwen_mcp.registry import registry

logger = logging.getLogger("qwen_mcp.scout")

class ScoutEngine:
    def __init__(self, client: Optional[DashScopeClient] = None):
        self.client = client or DashScopeClient()

    async def analyze_task(self, prompt: str, context: str = "", task_hint: str = "general", progress_callback=None) -> Dict[str, Any]:
        """
        Intelligently analyzes a task to determine complexity and recommended swarm usage.
        Also detects if task is brownfield (modifying existing code) vs greenfield (new code).
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

Detect BROWNFIELD vs GREENFIELD:
- BROWNFIELD (is_brownfield=true): Task involves modifying existing code, fixing bugs, refactoring, adding to existing files
- GREENFIELD (is_brownfield=false): Task is creating new files from scratch, new project, new feature with no existing code

Rules:
1. Recommend use_swarm=true IF task requires more than 300 lines of code OR touches more than 3 distinct files/modules OR if it's a multi-file audit.
2. Output ONLY JSON:
{{
  "complexity": "low|medium|high|critical",
  "score": 1-10,
  "use_swarm": true|false,
  "is_brownfield": true|false,
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
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, dict):
                        logger.info(f"Scout Result: {data.get('complexity', 'unknown')} (score: {data.get('score', 0)}), use_swarm={data.get('use_swarm', False)}, is_brownfield={data.get('is_brownfield', False)}")
                        return data
                except Exception as je:
                    logger.warning(f"Failed to parse Scout JSON: {je}")
            
            # Fallback for parsing failure
            logger.warning(f"Scout failed to find/parse JSON in response: {raw[:100]}...")
            return {
                "complexity": self._heuristic_complexity(prompt, context),
                "use_swarm": False,
                "is_brownfield": False,
                "reason": "Scout failed to follow JSON format or returned invalid JSON"
            }
        except Exception as e:
            logger.warning(f"Scout failed: {e}")
            return {
                "complexity": self._heuristic_complexity(prompt, context),
                "use_swarm": False,
                "is_brownfield": False,
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

    async def deep_discovery(self, workspace_root: str) -> str:
        """
        Autonomous context discovery. Scans for .context files, config files, and project structure.
        """
        root = Path(workspace_root)
        discovery_log = ["## 🔍 Scout Deep Discovery Report\n"]
        
        # 1. Check for .context directory
        context_dir = root / ".context"
        if context_dir.exists() and context_dir.is_dir():
            discovery_log.append("### 📁 Context Files Found:")
            for cf in context_dir.glob("*.md"):
                try:
                    content = cf.read_text(encoding='utf-8')
                    # Snippet of content
                    discovery_log.append(f"#### {cf.name}\n{content[:1000]}...\n")
                except Exception as e:
                    discovery_log.append(f"#### {cf.name} (Error reading: {e})")
        else:
            discovery_log.append("> No .context/ directory found. Sourcing from root structure.\n")

        # 2. Check for key config files
        config_files = ["pyproject.toml", "package.json", "requirements.txt", "setup.py", "docker-compose.yml", ".env.example", "AGENTS.md"]
        found_configs = []
        for cfg in config_files:
            cfg_path = root / cfg
            if cfg_path.exists():
                found_configs.append(cfg)
                try:
                    content = cfg_path.read_text(encoding='utf-8')
                    discovery_log.append(f"### ⚙️ Config: {cfg}\n```\n{content[:500]}\n```\n")
                except:
                    pass
        
        if not found_configs:
            discovery_log.append("> No standard config files detected in root.\n")

        # 3. Quick Tree Discovery (Top-level dirs)
        try:
            items = os.listdir(workspace_root)
            dirs = [d for d in items if os.path.isdir(os.path.join(workspace_root, d)) and not d.startswith('.')]
            files = [f for f in items if os.path.isfile(os.path.join(workspace_root, f)) and not f.startswith('.')]
            discovery_log.append(f"### 🌳 Project Structure:\n- **Directories**: {', '.join(dirs[:15])}{'...' if len(dirs) > 15 else ''}")
            discovery_log.append(f"- **Root Files**: {', '.join(files[:15])}{'...' if len(files) > 15 else ''}")
        except Exception as e:
            discovery_log.append(f"### 🌳 Project Structure (Discovery Error: {e})")

        return "\n".join(discovery_log)
