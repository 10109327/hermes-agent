"""Kimi / Moonshot provider profile.

Kimi has dual endpoints:
  - sk-kimi-* keys → api.kimi.com/coding (Anthropic Messages API)
  - legacy keys → api.moonshot.ai/v1 (OpenAI chat completions)

The /coding endpoint requires User-Agent: "claude-code/0.1.0" and uses
api_mode="anthropic_messages". The /v1 endpoint uses api_mode="chat_completions".

This profile covers the chat_completions path. The /coding endpoint is handled
by the Anthropic transport + adapter (Kimi-specific quirks in anthropic_adapter).
"""

from typing import Any, Dict, List, Optional

from providers.base import ProviderProfile, OMIT_TEMPERATURE
from providers import register_provider


class KimiProfile(ProviderProfile):
    """Kimi/Moonshot — temperature omitted, thinking/reasoning_effort config."""

    def get_reasoning_config(self, *, model: str = "", reasoning_config: dict = None,
                             **context) -> Optional[Dict[str, Any]]:
        """Kimi uses thinking.type + reasoning_effort in extra_body."""
        if not reasoning_config:
            return None

        effort = reasoning_config.get("effort")
        enabled = reasoning_config.get("enabled", True)

        result = {}
        if effort == "none" or enabled is False:
            result["thinking"] = {"type": "disabled"}
        else:
            result["thinking"] = {"type": "enabled"}
            # Kimi accepts low/medium/high (default: medium)
            if effort and effort in ("low", "medium", "high"):
                result["reasoning_effort"] = effort
            else:
                result["reasoning_effort"] = "medium"
        return result


kimi = KimiProfile(
    name="kimi-coding",
    aliases=("kimi", "moonshot", "kimi-coding-cn"),
    env_vars=("KIMI_API_KEY", "MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
    fixed_temperature=OMIT_TEMPERATURE,
    default_max_tokens=32000,
    default_headers={"User-Agent": "hermes-agent/1.0"},
)

register_provider(kimi)
