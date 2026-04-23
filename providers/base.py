"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, client quirks, request-time quirks. The transport reads this
instead of receiving 20+ boolean flags.

Provider profiles are DECLARATIVE — they describe the provider's behavior.
They do NOT own client construction, credential rotation, or streaming.
Those stay on AIAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Sentinel for "omit temperature entirely" (Kimi: server manages it)
OMIT_TEMPERATURE = object()


@dataclass
class ProviderProfile:
    """Base provider profile — subclass or instantiate with overrides."""

    # ── Identity ─────────────────────────────────────────────
    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()

    # ── Auth ─────────────────────────────────────────────────
    env_vars: tuple = ()
    base_url: str = ""
    auth_type: str = "api_key"  # api_key | oauth | copilot | aws

    # ── Client-level quirks (set once at client construction) ─
    default_headers: Dict[str, str] = field(default_factory=dict)

    # ── Request-level quirks ─────────────────────────────────
    # Temperature: None = use caller's default, OMIT_TEMPERATURE = don't send
    fixed_temperature: Any = None
    default_max_tokens: Optional[int] = None
    developer_role: bool = False  # swap system → developer role

    # ── Hooks (override in subclass for complex providers) ───

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Provider-specific message preprocessing.

        Called AFTER codex field sanitization, BEFORE developer role swap.
        Default: pass-through.
        """
        return messages

    def build_extra_body(self, *, session_id: str = None, **context) -> Dict[str, Any]:
        """Provider-specific extra_body fields.

        Merged into the API kwargs. Default: empty dict.
        """
        return {}

    def build_extra_headers(self, *, session_id: str = None, **context) -> Dict[str, str]:
        """Provider-specific per-request HTTP headers.

        Default: empty dict.
        """
        return {}

    def get_reasoning_config(self, *, model: str = "", reasoning_config: dict = None,
                             **context) -> Optional[Dict[str, Any]]:
        """Provider-specific reasoning/thinking configuration.

        Returns a dict to merge into extra_body, or None to skip.
        Default: None (no reasoning config).
        """
        return None
