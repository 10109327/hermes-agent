"""Tests for the provider module registry and profiles."""

import pytest
from providers import get_provider_profile, _REGISTRY, _discovered
from providers.base import ProviderProfile, OMIT_TEMPERATURE


class TestRegistry:
    """Provider registry discovery and lookup."""

    def test_discovery_populates_registry(self):
        p = get_provider_profile("nvidia")
        assert p is not None
        assert p.name == "nvidia"

    def test_alias_lookup(self):
        assert get_provider_profile("kimi").name == "kimi-coding"
        assert get_provider_profile("moonshot").name == "kimi-coding"
        assert get_provider_profile("or").name == "openrouter"
        assert get_provider_profile("nous-portal").name == "nous"
        assert get_provider_profile("qwen").name == "qwen-portal"

    def test_unknown_provider_returns_none(self):
        assert get_provider_profile("nonexistent-provider") is None

    def test_all_providers_have_name(self):
        get_provider_profile("nvidia")  # trigger discovery
        for name, profile in _REGISTRY.items():
            assert profile.name == name


class TestNvidiaProfile:
    def test_max_tokens(self):
        p = get_provider_profile("nvidia")
        assert p.default_max_tokens == 16384

    def test_no_special_temperature(self):
        p = get_provider_profile("nvidia")
        assert p.fixed_temperature is None

    def test_base_url(self):
        p = get_provider_profile("nvidia")
        assert "nvidia.com" in p.base_url


class TestKimiProfile:
    def test_temperature_omit(self):
        p = get_provider_profile("kimi")
        assert p.fixed_temperature is OMIT_TEMPERATURE

    def test_max_tokens(self):
        p = get_provider_profile("kimi")
        assert p.default_max_tokens == 32000

    def test_reasoning_enabled(self):
        p = get_provider_profile("kimi")
        result = p.get_reasoning_config(reasoning_config={"effort": "high", "enabled": True})
        assert result is not None
        assert result["thinking"]["type"] == "enabled"
        assert result["reasoning_effort"] == "high"

    def test_reasoning_disabled(self):
        p = get_provider_profile("kimi")
        result = p.get_reasoning_config(reasoning_config={"enabled": False})
        assert result["thinking"]["type"] == "disabled"

    def test_reasoning_none(self):
        p = get_provider_profile("kimi")
        assert p.get_reasoning_config(reasoning_config=None) is None

    def test_reasoning_effort_default(self):
        p = get_provider_profile("kimi")
        result = p.get_reasoning_config(reasoning_config={"enabled": True})
        assert result["reasoning_effort"] == "medium"


class TestOpenRouterProfile:
    def test_extra_body_with_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(provider_preferences={"allow": ["anthropic"]})
        assert body["provider"] == {"allow": ["anthropic"]}

    def test_extra_body_no_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body()
        assert body == {}

    def test_reasoning_high(self):
        p = get_provider_profile("openrouter")
        result = p.get_reasoning_config(reasoning_config={"effort": "high"})
        assert result == {"reasoning": {"effort": "high"}}

    def test_reasoning_disabled(self):
        p = get_provider_profile("openrouter")
        result = p.get_reasoning_config(reasoning_config={"enabled": False})
        assert result is None


class TestNousProfile:
    def test_tags(self):
        p = get_provider_profile("nous")
        body = p.build_extra_body()
        assert body["tags"] == ["product=hermes-agent"]

    def test_reasoning(self):
        p = get_provider_profile("nous")
        result = p.get_reasoning_config(reasoning_config={"effort": "medium"})
        assert result == {"reasoning": {"effort": "medium"}}


class TestQwenProfile:
    def test_max_tokens(self):
        p = get_provider_profile("qwen-portal")
        assert p.default_max_tokens == 65536

    def test_extra_body_has_vl(self):
        p = get_provider_profile("qwen-portal")
        body = p.build_extra_body(session_id="test-123")
        assert body["vl_high_resolution_images"] is True
        assert "metadata" in body
        assert body["metadata"]["sessionId"] == "test-123"

    def test_extra_body_no_session(self):
        p = get_provider_profile("qwen-portal")
        body = p.build_extra_body()
        assert body["vl_high_resolution_images"] is True
        assert "metadata" not in body


class TestBaseProfile:
    def test_prepare_messages_passthrough(self):
        p = ProviderProfile(name="test")
        msgs = [{"role": "user", "content": "hi"}]
        assert p.prepare_messages(msgs) is msgs

    def test_build_extra_body_empty(self):
        p = ProviderProfile(name="test")
        assert p.build_extra_body() == {}

    def test_get_reasoning_config_none(self):
        p = ProviderProfile(name="test")
        assert p.get_reasoning_config() is None
