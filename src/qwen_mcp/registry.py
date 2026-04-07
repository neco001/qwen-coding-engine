import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class ModelRegistry:
    CODING_PLAN_MODELS = {
        "qwen3.5-plus": {"priority": 10, "brand": "Qwen", "description": "Flagship 3.5 Plus (Best ROI)"},
        "qwen3-coder-plus": {"priority": 9, "brand": "Qwen", "description": "SOTA Coder (Complex Refactors)"},
        "qwen3-max-2026-01-23": {"priority": 9, "brand": "Qwen", "description": "Newest 3.0 Max (Reasoning)"},
        "glm-5": {"priority": 8, "brand": "Zhipu", "description": "GLM-5 Specialist (Analytical)"},
        "kimi-k2.5": {"priority": 8, "brand": "Moonshot", "description": "Kimi K2.5 (Context/Files)"},
    }

    PAYG_ONLY_MODELS = {
        "qwen-v2.5-coder-32b-instruct": {"priority": 5, "brand": "Qwen", "description": "Standard Coder (PAYG)"},
        "qwen-max": {"priority": 5, "brand": "Qwen", "description": "Standard Max (PAYG)"},
    }

    ROLE_PROMPTS = {
        "strategist": "Architectural decision making and LP planning",
        "coder": "Code generation and TDD Shackle implementation",
        "analyst": "Data analysis and logic auditing",
        "scout": "Complexity analysis and task routing",
        "audit": "Runtime bug hunting and security auditing",
    }

    def __init__(self, cache_file: str = "model_registry.json"):
        self.cache_file = cache_file
        self.models: Dict[str, str] = {
            "strategist": "qwen3.5-plus",
            "coder": "qwen3-coder-plus",
            "analyst": "glm-5",
            "scout": "qwen3.5-plus",
            "audit": "glm-5",
        }
        self.billing_mode = os.getenv("BILLING_MODE", "coding_plan")
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    cached = json.load(f)
                    if isinstance(cached, dict):
                        self.models.update(cached)
            except Exception as e:
                logger.warning(f"Failed to load registry cache: {e}")

    async def save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.models, f)
        except Exception as e:
            logger.error(f"Failed to save registry cache: {e}")

    def route_request(
        self,
        task_type: str,
        complexity_hint: str = "medium",
        context_tags: Optional[List[str]] = None,
        billing_mode: Optional[str] = None,
        estimated_tokens: int = 0
    ) -> str:
        """
        Routes request to the best available model based on role and complexity.
        This matches the signature expected by CompletionHandler.
        
        Args:
            task_type: Type of task (coder, strategist, audit, analyst, scout)
            complexity_hint: Complexity level (low, medium, high, critical)
            context_tags: Optional tags for additional context
            billing_mode: Optional billing mode override
            estimated_tokens: Estimated input tokens for budget-aware routing
        """
        # Mapping task types to internal roles
        role_map = {
            "coder": "coder",
            "strategist": "strategist",
            "audit": "audit",
            "analyst": "analyst",
            "scout": "scout"
        }
        role = role_map.get(task_type, "strategist")
        
        # Use complexity_hint instead of complexity
        complexity = complexity_hint
        
        # SOTA Logic: Upgrade to pro for high complexity
        if complexity in ["high", "critical"] and role == "coder":
            return "qwen3-coder-plus"
            
        return self.models.get(role, "qwen3.5-plus")

    def get_best_model(self, role: str) -> str:
        """Returns the assigned model for a role, with safety fallback."""
        return self.models.get(role, "qwen3.5-plus")

    @classmethod
    def get_available_models(cls, billing_mode: str) -> Dict[str, Any]:
        if billing_mode == "coding_plan":
            return cls.CODING_PLAN_MODELS.copy()
        elif billing_mode == "hybrid":
            return {**cls.CODING_PLAN_MODELS, **cls.PAYG_ONLY_MODELS}
        return cls.PAYG_ONLY_MODELS.copy()

    @classmethod
    def get_fallback_model(cls, billing_mode: str, task_type: str = "strategist") -> str:
        available = cls.get_available_models(billing_mode)
        if not available:
            return "qwen3.5-plus"
        
        # Hardcoded task list for fallback logic
        task_preferences = {
            "strategist": ["qwen3.5-plus", "qwen3-max-2026-01-23", "glm-5"],
            "coder": ["qwen3-coder-plus", "qwen3.5-plus"],
        }
        
        prefs = task_preferences.get(task_type, task_preferences["strategist"])
        for m_id in prefs:
            if m_id in available:
                return m_id
        
        # Priority sort fallback
        sorted_models = sorted(
            available.items(), 
            key=lambda x: (x[1].get("priority", 0) if isinstance(x[1], dict) else 0),
            reverse=True
        )
        return sorted_models[0][0] if sorted_models else "qwen3.5-plus"

    async def sync_with_hf(self) -> str:
        return "HF sync completed (STUB)"

# Global Singleton
registry = ModelRegistry()
