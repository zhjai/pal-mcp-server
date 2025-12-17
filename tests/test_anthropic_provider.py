"""Tests for Anthropic provider base URL customization."""

import os
from unittest.mock import patch

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.shared import ProviderType
from utils.env import reload_env


class TestAnthropicProvider:
    """Test cases for Anthropic provider."""

    def test_provider_initialization(self):
        """Test Anthropic provider initialization with default URL."""
        provider = AnthropicModelProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://api.anthropic.com/v1"
        assert provider.FRIENDLY_NAME == "Anthropic"

    def test_custom_base_url_from_env(self):
        """Test that custom base URL is used when ANTHROPIC_BASE_URL is set."""
        custom_url = "https://custom-anthropic-api.example.com/v1"

        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": custom_url}):
            reload_env()
            provider = AnthropicModelProvider(api_key="test-key")
            assert provider.base_url == custom_url

    def test_custom_base_url_various_endpoints(self):
        """Test various custom endpoint formats."""
        test_cases = [
            "https://anthropic-proxy.example.com/v1",
            "http://localhost:8080/anthropic",
            "https://gateway.example.com/anthropic",
        ]

        for custom_url in test_cases:
            with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": custom_url}):
                reload_env()
                provider = AnthropicModelProvider(api_key="test-key")
                assert provider.base_url == custom_url

    def test_explicit_base_url_override(self):
        """Test that explicit base_url in kwargs overrides environment variable."""
        env_url = "https://env-api.example.com/v1"
        explicit_url = "https://explicit-api.example.com/v1"

        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": env_url}):
            reload_env()
            provider = AnthropicModelProvider(api_key="test-key", base_url=explicit_url)
            assert provider.base_url == explicit_url

    def test_provider_type(self):
        """Test that provider type is correctly set."""
        provider = AnthropicModelProvider(api_key="test-key")
        assert provider.get_provider_type() == ProviderType.ANTHROPIC

    def test_model_validation(self):
        """Test model validation for Anthropic models."""
        provider = AnthropicModelProvider(api_key="test-key")

        # Known Claude models should be valid
        assert provider.validate_model_name("claude-3-opus") is True
        assert provider.validate_model_name("claude-3-sonnet") is True
        assert provider.validate_model_name("claude-3-haiku") is True

        # Any model starting with "claude-" should be valid
        assert provider.validate_model_name("claude-4-new-model") is True

        # Non-Claude models should be invalid
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False

    def test_get_capabilities(self):
        """Test capability generation for Anthropic models."""
        provider = AnthropicModelProvider(api_key="test-key")

        # Test with known model
        caps = provider.get_capabilities("claude-3-opus")
        assert caps.provider == ProviderType.ANTHROPIC
        assert caps.model_name == "claude-3-opus"
        assert caps.context_window == 200_000
        assert caps.supports_images is True
        assert caps.supports_streaming is True

        # Test with unknown Claude model - should get generic capabilities
        caps = provider.get_capabilities("claude-4-future")
        assert caps.provider == ProviderType.ANTHROPIC
        assert caps.model_name == "claude-4-future"
        assert caps.context_window == 200_000  # Conservative default

        # Test with invalid model - should raise error
        with pytest.raises(ValueError, match="Unsupported model 'gpt-4' for provider anthropic"):
            provider.get_capabilities("gpt-4")

    def test_list_models(self):
        """Test listing available Anthropic models."""
        provider = AnthropicModelProvider(api_key="test-key")

        # Should return known models
        models = provider.list_models(respect_restrictions=False)
        assert "claude-3-opus" in models
        assert "claude-3-sonnet" in models
        assert "claude-3-haiku" in models

    def test_get_preferred_model_extended_reasoning(self):
        """Test preferred model selection for extended reasoning."""
        from tools.models import ToolModelCategory

        provider = AnthropicModelProvider(api_key="test-key")
        allowed = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

        preferred = provider.get_preferred_model(ToolModelCategory.EXTENDED_REASONING, allowed)
        assert preferred == "claude-3-opus"  # Opus is preferred for reasoning

    def test_get_preferred_model_fast_response(self):
        """Test preferred model selection for fast responses."""
        from tools.models import ToolModelCategory

        provider = AnthropicModelProvider(api_key="test-key")
        allowed = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

        preferred = provider.get_preferred_model(ToolModelCategory.FAST_RESPONSE, allowed)
        assert preferred == "claude-3-haiku"  # Haiku is preferred for speed

    def test_get_preferred_model_balanced(self):
        """Test preferred model selection for balanced performance."""
        from tools.models import ToolModelCategory

        provider = AnthropicModelProvider(api_key="test-key")
        allowed = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

        preferred = provider.get_preferred_model(ToolModelCategory.BALANCED, allowed)
        assert preferred == "claude-3-sonnet"  # Sonnet is preferred for balance
