import os
import json
import re
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Dynamic registry for ROI-optimized model selection (JSON Cached)."""

    ROLE_PROMPTS = {
        "strategist": "Expert in high-level planning and JSON architecture. (Priority: qwen3.5-plus, qwen-plus)",
        "coder": "Production-grade code generation, long outputs. (Priority: qwen3-coder-plus, qwen-coder-plus)",
        "specialist": "Complex logic, algorithms, and refactoring specialist. (Priority: qwen3-coder-next, qwen2.5-coder-32b)",
        "analyst": "Deep reasoning, SRE audit, finding hidden bugs. (Priority: qwq-plus, qwq-32b)",
        "scout": "Fast, cheap, discovery, and summarization. (Priority: qwen-turbo, qwen-flash)",
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
            "strategist": "qwen-plus",
            "coder": "qwen2.5-coder-32b-instruct",
            "specialist": "qwen2.5-coder-32b-instruct",
            "analyst": "qwen-plus",
            "scout": "qwen-turbo",
        }
        self.last_updated = datetime.min
        self.load_cache()

    def score_model(self, model_id: str, criteria: dict) -> int:
        score = 0
        mid = model_id.lower()

        garbage = ["vl", "audio", "image", "tts", "asr", "vc", "omni", "mt", "realtime"]
        if any(g in mid for g in garbage) and not any(
            m in mid for m in criteria["must_have"]
        ):
            return -500

        for word in criteria["must_have"]:
            if word in mid:
                score += 100

        for word in criteria["avoid"]:
            if word in mid:
                score -= 50

        numbers = re.findall(r"(\d+(?:\.\d+)?)", mid)
        for num_str in numbers:
            num = float(num_str)
            if 1.0 <= num <= 10.0:
                score += int(num * 10)
            elif num > 2000:
                score += 1

        if mid == criteria["fallback"]:
            score += 20

        if "-" in mid and any(char.isdigit() for char in mid.split("-")[-1]):
            score -= 5

        return score

    def load_cache(self):
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                if data.get("schema_version", 1) > 1:
                    logger.warning("Cache schema version mismatch. Regenerating...")
                    return

                self.models.update(data.get("models", {}))
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
                except:
                    pass

    async def save_cache(self):
        async with self._lock:
            try:
                self.last_updated = datetime.now()
                data = {
                    "schema_version": 1,
                    "updated_at": self.last_updated.isoformat(),
                    "models": self.models,
                }

                temp_file = self.cache_file.with_suffix(".json.tmp")
                temp_file.write_text(json.dumps(data, indent=2))
                os.replace(temp_file, self.cache_file)
            except Exception as e:
                logger.error(f"ModelRegistry: Failed to save cache: {e}")

    def get_best_model(self, task_type: str) -> str:
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
        return self.get_best_model("strategist")

    @property
    def SCOUT(self):
        return self.get_best_model("scout")

    @property
    def CODER_SPECIALIST(self):
        return self.get_best_model("coder")

    @property
    def LOGIC_SPECIALIST(self):
        return self.get_best_model("specialist")

    @property
    def ANALYST(self):
        return self.get_best_model("analyst")


registry = ModelRegistry()
