import os
import asyncio
import logging
import textwrap
from typing import List, Dict, Any, Optional

# Base modules resulting from modularization
from .registry import ModelRegistry, registry as global_registry
from .completions import CompletionHandler

logger = logging.getLogger(__name__)

class DashScopeClient(CompletionHandler):
    """
    Unified client facade for DashScope interactions.
    Delegates logic to CompletionHandler while maintaining
    registry healing and common configuration from BaseDashScopeClient.
    """

    def __init__(
        self, registry: Optional[ModelRegistry] = None, sanitizer_cls=None
    ):
        # Initializing through multiple inheritance
        # sanitizer_cls logic is handled in BaseDashScopeClient.__init__
        if sanitizer_cls:
            super().__init__(registry=registry, sanitizer_cls=sanitizer_cls)
        else:
            super().__init__(registry=registry)

    async def probe_model(self, model_id: str, retries: int = 2) -> bool:
        """Pings the model with a minimal payload to ensure it is healthy. Retries with backoff."""
        for attempt in range(retries + 1):
            try:
                # Increasing timeout slightly for subsequent attempts
                timeout = 2.0 + (attempt * 1.5)
                logger.info(
                    f"Probing candidate model: {model_id} (Attempt {attempt + 1}/{retries + 1})..."
                )
                # Select the right client for this specific model probe
                client_to_use = self.get_client_for_model(model_id)
                await client_to_use.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    timeout=timeout,
                )
                logger.info(f"Model probe successful: {model_id}")
                return True
            except Exception as e:
                if attempt == retries:
                    logger.warning(
                        f"Model probe failed for {model_id} after {retries + 1} attempts: {e}"
                    )
                    return False
                wait_time = 0.5 * (2**attempt)
                await asyncio.sleep(wait_time)
        return False

    async def list_models(self) -> List[str]:
        """Fetch list of available models from DashScope."""
        try:
            response = await self.client.models.list()
            return [m.id for m in response.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def refresh_registry(self) -> str:
        """Alias for heal_registry to provide seamless transition to LLM-driven selection."""
        return await self.heal_registry()

    async def heal_registry(self) -> str:
        """Advanced Self-Healing: Uses meta-analysis via qwen-turbo to map roles by ROI."""
        from .base import get_billing_mode
        if get_billing_mode() == "coding_plan":
            return "Self-healing disabled in Strict Coding Plan mode. Roles are hardcoded to plan models."
            
        logger.info("Initiating Self-Healing Meta-Analysis via Qwen3.5-Plus...")
        all_models = await self.list_models()
        if not all_models:
            return "Self-healing failed: Could not fetch model list from API."

        # Filter out obvious multimedia noise to keep prompt clean
        noise = ["audio", "tts", "asr", "vc", "image", "omni", "mt", "realtime"]
        clean_list = [m for m in all_models if not any(x in m.lower() for x in noise)]

        roles_desc = "\n".join(
            [f"- {r}: {d}" for r, d in self.registry.ROLE_PROMPTS.items()]
        )

        prompt = textwrap.dedent(f"""\
            You are the Qwen Project Administrator. Your goal is ROI optimization.
            Assign the BEST model ID from the list below to each required role.
            PRIORITIZE:
            1. SOTA (Generation Jump): ALWAYS prefer 'qwen3.5' models over 'qwen2.5' or generic 'qwen-plus/turbo' if available. 3.5 is the current gold standard.
            2. Stability: Prefer pure aliases (e.g. 'qwen3.5-plus') over specific snapshot dates (e.g. 'qwen3.5-plus-2025...') unless no pure alias exists.
            3. Capability: Match the role description to the model specialty (e.g. 'coder' role -> 'qwen3-coder-plus').
            
            Roles to assign:
            {roles_desc}
            
            AVAILABLE MODELS:
            {clean_list}
            
            Return ONLY a JSON object: {{"strategist": "id", "coder": "id", "specialist": "id", "analyst": "id", "scout": "id"}}""")

        try:
            # We use the known stable 'qwen3.5-plus' for the meta-analysis itself
            client_to_use = self.get_client_for_model("qwen3.5-plus")
            response = await client_to_use.chat.completions.create(
                model="qwen3.5-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                timeout=20.0,
            )

            from .tools import extract_json_from_text
            new_models = extract_json_from_text(response.choices[0].message.content)

            if new_models and isinstance(new_models, dict):
                updates = []
                for role, m_id in new_models.items():
                    if m_id in all_models:
                        if m_id != self.registry.models.get(role):
                            if await self.probe_model(m_id):
                                self.registry.models[role] = m_id
                                updates.append(f"{role.capitalize()} -> {m_id}")
                    else:
                        logger.warning(f"Self-healing: Model {m_id} not in API list. Skipping.")

                await self.registry.save_cache()
                if updates:
                    return f"Registry healed via Qwen-Turbo Meta-Analysis! Updates: {', '.join(updates)}"
                return "Registry validated. All LLM-selected models are healthy and optimal."

            return "Self-healing failed: LLM response was not valid JSON."

        except Exception as e:
            logger.error(f"Self-healing error: {e}")
            return f"Self-healing failed: {str(e)}"
