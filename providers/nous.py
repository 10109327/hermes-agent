"""Nous Portal provider profile."""

from typing import Any, Dict, Optional

from providers.base import ProviderProfile
from providers import register_provider


class NousProfile(ProviderProfile):
    """Nous Portal — product attribution tags, reasoning config."""

    def build_extra_body(self, *, session_id: str = None, **context) -> Dict[str, Any]:
        return {"tags": ["product=hermes-agent"]}

    def get_reasoning_config(self, *, model: str = "", reasoning_config: dict = None,
                             **context) -> Optional[Dict[str, Any]]:
        if not reasoning_config:
            return None
        effort = reasoning_config.get("effort")
        enabled = reasoning_config.get("enabled", True)
        if effort == "none" or enabled is False:
            return None
        return {"reasoning": {"effort": effort or "high"}}


nous = NousProfile(
    name="nous",
    aliases=("nous-portal", "nousresearch"),
    env_vars=("NOUS_API_KEY",),
    base_url="https://inference-api.nousresearch.com/v1",
    auth_type="oauth",
)

register_provider(nous)
