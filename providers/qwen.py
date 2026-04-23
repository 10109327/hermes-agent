"""Qwen Portal provider profile."""

from typing import Any, Dict, List

from providers.base import ProviderProfile
from providers import register_provider


class QwenProfile(ProviderProfile):
    """Qwen Portal — message normalization, high-res VL, session metadata."""

    def build_extra_body(self, *, session_id: str = None, **context) -> Dict[str, Any]:
        import uuid
        body: Dict[str, Any] = {"vl_high_resolution_images": True}
        if session_id:
            body["metadata"] = {
                "sessionId": session_id,
                "promptId": str(uuid.uuid4()),
            }
        return body


qwen = QwenProfile(
    name="qwen-portal",
    aliases=("qwen", "qwen-oauth"),
    env_vars=("QWEN_API_KEY",),
    base_url="https://portal.qwen.ai/api/v1",
    auth_type="oauth",
    default_max_tokens=65536,
)

register_provider(qwen)
