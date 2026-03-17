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
        "analyst": "Deep dive into logs, performance metrics, and debugging complex runtime issues."
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
            "coder_pro": "qwen2.5-72b-instruct",
            "specialist": "qwen2.5-coder-32b-instruct",
            "analyst": "qwq-plus",
            "scout": "qwen-turbo",
        }
        self.metadata = {}  # Store HF metadata: {model_id: {tags, params, levels}}
        self.last_updated = datetime.min
        self.load_cache()

    async def sync_with_hf(self) -> str:
        """Fetches latest Qwen model metadata from HF and updates levels/tags."""
        url = "https://huggingface.co/api/models?search=qwen&sort=downloads&direction=-1&limit=100"
        try:
            # Use run_in_executor for blocking urlopen
            loop = asyncio.get_event_loop()

            def fetch():
                with urllib.request.urlopen(url, timeout=10) as response:
                    return json.loads(response.read().decode())

            models = await loop.run_in_executor(None, fetch)

            updates = 0
            for m in models:
                mid = m.get("id")
                if not mid:
                    continue

                # Basic metadata extraction
                pipeline = m.get("pipeline_tag", "text-generation")
                tags = m.get("tags", [])
                downloads = m.get("downloads", 0)

                # Determine levels (heuristic)
                levels = [2]  # default
                m_lower = mid.lower()

                # Tier 4: Heavyweights
                if any(
                    x in m_lower for x in ["qwq", "max", "plus", "72b", "110b", "math"]
                ):
                    levels.extend([3, 4])

                # Tier 3: Specialists
                if "coder" in m_lower or "next" in m_lower:
                    levels.extend([3, 4])

                # Tier 1-2: Efficiency
                if any(x in m_lower for x in ["turbo", "flash", "7b", "1.5b", "0.5b"]):
                    levels.extend([1, 2])

                # SOTA override
                if "3.5" in m_lower:
                    levels.append(4)

                self.metadata[mid] = {
                    "full_id": mid,
                    "pipeline": pipeline or "text-generation",
                    "tags": tags,
                    "downloads": downloads,
                    "levels": list(set(levels)),
                }
                updates += 1

            await self.save_cache()
            return f"Successfully synced {updates} models from HF."
        except Exception as e:
            logger.error(f"HF Sync failed: {e}")
            return f"HF Sync failed: {str(e)}"

    def score_model(self, model_id: str, criteria: dict) -> int:
        score: int = 0
        mid = model_id.lower()

        garbage = ["vl", "audio", "image", "tts", "asr", "vc", "omni", "mt", "realtime"]
        if any(g in mid for g in garbage) and not any(
            m in mid for m in criteria.get("must_have", [])
        ):
            return -500

        for word in criteria.get("must_have", []):
            if word in mid:
                score = score + 100

        for word in criteria.get("avoid", []):
            if word in mid:
                score = score - 50

        numbers = re.findall(r"(\d+(?:\.\d+)?)", mid)
        for num_str in numbers:
            try:
                num = float(num_str)
                if 1.0 <= num <= 10.0:
                    score = score + int(num * 10)
                elif num > 2000:
                    score = score + 1
            except ValueError:
                continue

        if mid == criteria.get("fallback"):
            score = score + 20

        last_part = mid.split("-")[-1]
        if "-" in mid and any(char.isdigit() for char in last_part):
            score = score - 5

        return score

    def load_cache(self):
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                if data.get("schema_version", 1) != 2:
                    logger.warning(
                        "Cache schema version mismatch or old version. Regenerating..."
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
                    self.cache_file.rename(self.cache_file.with_suffix(".json.bak"))
                except Exception:
                    pass

    async def save_cache(self):
        async with self._lock:
            try:
                self.last_updated = datetime.now()
                data = {
                    "schema_version": 2,
                    "updated_at": self.last_updated.isoformat(),
                    "models": self.models,
                    "metadata": self.metadata,
                }

                temp_file = self.cache_file.with_suffix(".json.tmp")
                temp_file.write_text(json.dumps(data, indent=2))
                os.replace(temp_file, self.cache_file)
            except Exception as e:
                logger.error(f"ModelRegistry: Failed to save cache: {e}")

    def get_plan_model_for_role(self, task_type: str) -> str:
        """Mappings specifically tailored for the Alibaba Coding Plan."""
        plan_models = {
            "strategist": "qwen3.5-plus",
            "coder": "qwen3-coder-next",       # Fast, inline, lightweight
            "coder_pro": "qwen3-coder-plus",   # Heavy lifter, huge context, system-wide refactors
            "specialist": "qwen3-coder-plus",  # High capability logic/tooling operations
            "analyst": "glm-5",
            "scout": "kimi-k2.5",
        }
        mapping = {
            "discovery": "scout",
            "scout": "scout",
            "coding": "coder",
            "coder": "coder",
            "coding_pro": "coder_pro",
            "coder_pro": "coder_pro",
            "refactoring": "specialist",
            "specialist": "specialist",
            "audit": "analyst",
            "analyst": "analyst",
            "planning": "strategist",
            "strategist": "strategist",
        }
        role = mapping.get(task_type, "strategist")
        return plan_models.get(role, "qwen3.5-plus")

    def route_request(
        self, task_type: str, complexity_hint: str = "auto", context_tags: list = [],
        billing_mode: str = "hybrid", estimated_tokens: int = 0
    ) -> str:
        """Dynamic route to best model fitting the criteria with heuristic scoring."""
        
        # 1. Map task to logical role
        mapping = {
            "discovery": "scout",
            "scout": "scout",
            "coding": "coder",
            "coder": "coder",
            "coding_pro": "coder_pro",
            "coder_pro": "coder_pro",
            "refactoring": "specialist",
            "specialist": "specialist",
            "audit": "analyst",
            "analyst": "analyst",
            "planning": "strategist",
            "strategist": "strategist",
        }
        role = mapping.get(task_type)

        # 2. Intelligent Auto-Upgrade: 
        # If it's a standard coding task but looks heavy (long prompt), 
        # force upgrade to coder_pro unless explicitly forbidden.
        if role == "coder" and (estimated_tokens > 15000 or complexity_hint in ["high", "critical"]):
            # Verification: In some billing modes, coder_pro might be restricted.
            # For 'coding_plan' and 'hybrid' (which uses plan for coding), it's allowed.
            if billing_mode in ["coding_plan", "hybrid"]:
                role = "coder_pro"
                # We map task_type to the next logical level for the plan mapping
                task_type = "coding_pro"
                logger.info(f"🚀 Intelligent Routing: Upgrading coding task to CODER_PRO (tokens: {estimated_tokens})")
            else:
                logger.warning(f"⚠️ Intelligent Routing: Auto-upgrade to CODER_PRO suppressed due to billing_mode: {billing_mode}")
        
        # 3. Mode-specific routing
        if billing_mode == "coding_plan":
            return self.get_plan_model_for_role(task_type)
        
        elif billing_mode == "hybrid":
            # Cost optimization: route high-volume tasks to the prepaid Coding Plan.
            if role in ["coder", "coder_pro", "specialist", "strategist", "scout", "discovery"]:
                return self.get_plan_model_for_role(task_type)
            # Use role-based mapping for others (e.g. analyst -> qwq-plus via PAYG)

        # 4. Fallback to Manual overrides/pinned roles if exists
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
            # Explicit type check for levels to avoid lint error
            if isinstance(levels, list) and target_level in levels:
                # Check for task compatibility
                pipeline = meta.get("pipeline", "")
                tags = meta.get("tags", [])
                if (isinstance(task_type, str) and task_type in pipeline) or any(task_type in t for t in tags):
                    score = 0
                    mid_lower = mid.lower()

                    # Heuristic A: Instruct preference
                    if "instruct" in mid_lower or "chat" in mid_lower:
                        score += 1000

                    # Heuristic B: SOTA preference
                    if "3.5" in mid_lower:
                        score += 500
                    elif "2.5" in mid_lower:
                        score += 100

                    # Heuristic C: Parameter tie-breaker (prefer larger for higher tiers)
                    score += int(meta.get("params", 0) / 1e9)

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
