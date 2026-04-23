"""OpenRouter provider profile."""

from typing import Any, Dict, Optional

from providers.base import ProviderProfile
from providers import register_provider


class OpenRouterProfile(ProviderProfile):
    """OpenRouter — provider preferences, reasoning config, HTTP headers."""

    def build_extra_body(self, *, session_id: str = None, **context) -> Dict[str, Any]:
        body = {}
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs
        return body

    def get_reasoning_config(self, *, model: str = "", reasoning_config: dict = None,
                             **context) -> Optional[Dict[str, Any]]:
        if not reasoning_config:
            return None
        effort = reasoning_config.get("effort")
        enabled = reasoning_config.get("enabled", True)
        if effort == "none" or enabled is False:
            return None
        # OpenRouter: {"reasoning": {"effort": "high"}} in extra_body
        return {"reasoning": {"effort": effort or "high"}}


openrouter = OpenRouterProfile(
    name="openrouter",
    aliases=("or",),
    env_vars=("OPENROUTER_API_KEY",),
    base_url="https://openrouter.ai/api/v1",
)

register_provider(openrouter)
