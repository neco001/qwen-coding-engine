import os
import json
import re
import asyncio
import logging
import urllib.request
from pathlib import Path
from datetime import datetime
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Dynamic registry for ROI-optimized model selection (JSON Cached)."""

    ROLE_PROMPTS = {
        "strategist": "Advanced reasoning, strategic planning, and adversary simulation. Focused on ROI.",
        "coding": "High-fidelity code generation and refactoring following production standards. Default model: qwen3-coder-plus.",
        "scout": "Exploration and identification of codebase patterns and structures.",
        "analyst": "Deep dive into logs, performance metrics, and debugging complex runtime issues.",
        "artist": "Creative image generation and visual design refinement."
    }

    COGNITIVE_LEVEL_MAP = {
        "text-generation": [2, 3],
        "conversational": [3, 4],
        "math": [4],
        "reasoning": [4],
        "coding": [3, 4],
        "code": [3, 4],
    }

    def __init__(self):
        self.cache_dir = Path(user_cache_dir("qwen-coding", "Qwen"))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                os.chmod(self.cache_dir, 0o700)
        except Exception as e:
            logger.error(f"Failed to setup cache directory: {e}")

        self.cache_file = self.cache_dir / "models_cache.json"
        self._lock = asyncio.Lock()

        self.models = {
            "strategist": "qwen3.5-plus",
            "coder": "qwen3-coder-plus",
            "specialist": "qwen2.5-coder-32b-instruct",
            "analyst": "qwq-plus",
            "scout": "qwen-turbo",
            "artist": "qwen-image-edit",
        }
        self.metadata = {}  # Store HF metadata: {model_id: {tags, params, levels}}
        self.last_updated = datetime.min
        self.load_cache()

    async def sync_with_hf(self) -> str:
        """Fetches latest Qwen model metadata from HF and updates levels/tags with scoring priority."""
        url = "https://huggingface.co/api/models?search=qwen&sort=downloads&direction=-1&limit=100"
        try:
            loop = asyncio.get_event_loop()

            def fetch():
                with urllib.request.urlopen(url, timeout=10) as response:
                    return json.loads(response.read().decode())

            models = await loop.run_in_executor(None, fetch)
            
            # Phase 1: Filter and Score
            processed_models = self._filter_and_score_models(models)
            
            # Phase 2: Metadata Extraction & Level Mapping
            updates = 0
            for m in processed_models:
                mid = m.get("id")
                if not mid:
                    continue

                pipeline = m.get("pipeline_tag", "text-generation")
                tags = m.get("tags", [])
                downloads = m.get("downloads", 0)
                priority_score = m.get("priority_score", 0)

                # Determine levels (heuristic)
                levels = [2]  # default
                m_lower = mid.lower()

                if any(x in m_lower for x in ["qwq", "max", "plus", "72b", "110b", "math"]):
                    levels.extend([3, 4])
                if "coder" in m_lower or "next" in m_lower:
                    levels.extend([3, 4])
                if any(x in m_lower for x in ["turbo", "flash", "7b", "1.5b", "0.5b"]):
                    levels.extend([1, 2])
                if "3.5" in m_lower:
                    levels.append(4)

                self.metadata[mid] = {
                    "full_id": mid,
                    "pipeline": pipeline or "text-generation",
                    "tags": tags,
                    "downloads": downloads,
                    "levels": list(set(levels)),
                    "priority_score": priority_score,
                }
                updates += 1

            await self.save_cache()
            return f"Successfully synced {updates} models from HF (ROI-Filtered)."
        except Exception as e:
            logger.error(f"HF Sync failed: {e}")
            return f"HF Sync failed: {str(e)}"

    def _filter_and_score_models(self, raw_metadata: list) -> list:
        """Filters noisy pipelines and calculates priority based on name length and year."""
        filtered = []
        garbage = ["audio", "tts", "video", "omni", "asr", "vc", "realtime", "mt"]
        
        for model in raw_metadata:
            mid = model.get("id", "").lower()
            name = model.get("name", mid).lower()
            
            # Filter noise
            if any(kw in mid for kw in garbage):
                continue
            
            # Calculate priority: (Shorter name = Higher Priority) + (Later year = Higher Priority)
            year = self._extract_year_from_name(mid)
            # Use a large multiplier for length to ensure shorter name ALWAYS wins over year tie-breakers.
            priority_score = ((100 - len(mid)) * 10000) + year
            
            model["priority_score"] = priority_score
            filtered.append(model)
            
        return filtered

    def _extract_year_from_name(self, name: str) -> int:
        """Extracts 4-digit years (2024-2027) from name as priority tie-breaker."""
        match = re.search(r"(202[4-7])", name)
        return int(match.group(1)) if match else 0


    def load_cache(self, path: Path = None):
        """Loads repository state from JSON cache. Handles schema migrations."""
        target_file = path or self.cache_file
        if target_file.exists():
            try:
                data = json.loads(target_file.read_text())
                schema_v = data.get("schema_version", 1)
                
                if schema_v != 3:
                    logger.warning(
                        f"ModelRegistry: Cache schema mismatch (v{schema_v} vs v3). Starting fresh."
                    )
                    return

                self.models.update(data.get("models", {}))
                self.metadata.update(data.get("metadata", {}))
                updated_at = data.get("updated_at")
                if updated_at:
                    self.last_updated = datetime.fromisoformat(updated_at)
                logger.info("ModelRegistry: Cache loaded successfully.")
            except Exception as e:
                logger.error(
                    f"ModelRegistry: Cache corrupted. Backing up and starting fresh: {e}"
                )
                try:
                    target_file.rename(target_file.with_suffix(".json.bak"))
                except Exception:
                    pass

    def load_cache_from_path(self, path: str):
        """API wrapper for testing."""
        self.load_cache(Path(path))

    async def save_cache(self, path: Path = None):
        """Atomic write of the model registry to cache file (schema v3)."""
        async with self._lock:
            try:
                target_file = path or self.cache_file
                self.last_updated = datetime.now()
                data = {
                    "schema_version": 3,
                    "updated_at": self.last_updated.isoformat(),
                    "models": self.models,
                    "metadata": self.metadata,
                }

                temp_file = target_file.with_suffix(".json.tmp")
                temp_file.write_text(json.dumps(data, indent=2))
                os.replace(temp_file, target_file)
            except Exception as e:
                logger.error(f"ModelRegistry: Failed to save cache: {e}")

    def save_cache_to_path(self, path: str):
        """API wrapper for testing (blocking)."""
        target_file = Path(path)
        self.last_updated = datetime.now()
        data = {
            "schema_version": 3,
            "updated_at": self.last_updated.isoformat(),
            "models": self.models,
            "metadata": self.metadata,
        }
        temp_file = target_file.with_suffix(".json.tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        os.replace(temp_file, target_file)

    def route_request(
        self, task_type: str, complexity_hint: str = "auto", context_tags: list = []
    ) -> str:
        """Dynamic route to best model fitting the criteria with heuristic scoring."""
        # Debug print to catch recursion in logs
        # print(f"DEBUG: Routing request for {task_type} (complexity: {complexity_hint})")

        complexity_map = {"low": 1, "auto": 2, "high": 3, "critical": 4}
        # 0. Priority: Manual overrides/pinned roles
        mapping = {
            "discovery": "scout",
            "scout": "scout",
            "coding": "coder",
            "coder": "coder",
            "refactoring": "specialist",
            "specialist": "specialist",
            "audit": "analyst",
            "analyst": "analyst",
            "planning": "strategist",
            "strategist": "strategist",
        }
        role = mapping.get(task_type)
        if role and role in self.models:
            return self.models[role]

        # 1. Ensure metadata is not empty
        if not self.metadata:
            role = mapping.get(task_type, "strategist")
            return self.models.get(role, "qwen-plus")

        # 2. Map hints to target level
        hint_map = {"low": 1, "medium": 2, "high": 3, "critical": 4, "scout": 0}
        target_level = (
            hint_map.get(complexity_hint.lower(), 2) if complexity_hint != "auto" else 2
        )

        # Override for specific tasks
        if "coding" in task_type or "coder" in task_type:
            target_level = max(target_level, 3)
        if "audit" in task_type or "analyst" in task_type:
            target_level = 4

        # 3. Filter and Score Candidates
        scored_candidates = []
        for mid, meta in self.metadata.items():
            levels = meta.get("levels", [])
            if target_level in levels:
                # Check for task compatibility
                pipeline = meta.get("pipeline", "")
                tags = meta.get("tags", [])
                if task_type in pipeline or any(task_type in t for t in tags):
                    score = 0
                    mid_lower = mid.lower()

                    # Heuristic A: Instruct preference
                    if "instruct" in mid_lower or "chat" in mid_lower:
                        score += 1000

                    # Heuristic B: SOTA preference (from metadata priority)
                    score += meta.get("priority_score", 0)

                    scored_candidates.append((mid, score))

        # 4. Handle context-specific isolation (no_system_leak)
        if "no_system_leak" in context_tags:
            # Filter candidates for those with 'instruct' in name specifically
            instruct_candidates = [c for c in scored_candidates if "instruct" in c[0]]
            if instruct_candidates:
                scored_candidates = instruct_candidates

        # 5. Return results
        if not scored_candidates:
            # Last ditch attempt: check by task_type alone regardless of level
            for mid, meta in self.metadata.items():
                if task_type in meta["pipeline"]:
                    scored_candidates.append((mid, 1))

            if not scored_candidates:
                return self.get_best_model(task_type)

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def get_best_model(self, task_type: str) -> str:
        """Fallback of last resort using hardcoded roles or local models."""
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")

        mapping = {
            "discovery": "scout",
            "scout": "scout",
            "coding": "coder",
            "coder": "coder",
            "refactoring": "specialist",
            "specialist": "specialist",
            "audit": "analyst",
            "analyst": "analyst",
            "planning": "strategist",
            "strategist": "strategist",
        }
        role = mapping.get(task_type, "strategist")
        return self.models.get(role, "qwen-plus")

    @property
    def STRATEGIST(self):
        return self.models.get("strategist", "qwen-max")

    @property
    def SCOUT(self):
        return self.models.get("scout", "qwen-turbo")

    @property
    def CODER_SPECIALIST(self):
        return self.models.get("coder", "qwen3.5-plus")

    @property
    def LOGIC_SPECIALIST(self):
        return self.models.get("specialist", "qwen2.5-coder-32b-instruct")

    @property
    def ANALYST(self):
        return self.models.get("analyst", "qwq-plus")


registry = ModelRegistry()
