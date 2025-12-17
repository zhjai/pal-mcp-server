"""Tests for OpenAI provider base URL customization."""

import os
from unittest.mock import patch

from providers.openai import OpenAIModelProvider
from utils.env import reload_env


class TestOpenAIBaseURL:
    """Test cases for OpenAI base URL customization."""

    def test_default_base_url(self):
        """Test that default base URL is used when OPENAI_BASE_URL is not set."""
        # Ensure OPENAI_BASE_URL is not set
        with patch.dict(os.environ, {}, clear=False):
            # Remove OPENAI_BASE_URL if it exists
            os.environ.pop("OPENAI_BASE_URL", None)
            reload_env()

            provider = OpenAIModelProvider(api_key="test-key")
            assert provider.base_url == "https://api.openai.com/v1"

    def test_custom_base_url_from_env(self):
        """Test that custom base URL is used when OPENAI_BASE_URL is set."""
        custom_url = "https://custom-openai-api.example.com/v1"

        with patch.dict(os.environ, {"OPENAI_BASE_URL": custom_url}):
            reload_env()
            provider = OpenAIModelProvider(api_key="test-key")
            assert provider.base_url == custom_url

    def test_custom_base_url_various_endpoints(self):
        """Test various custom endpoint formats."""
        test_cases = [
            "https://api.openai.azure.com/v1",
            "http://localhost:8080/v1",
            "https://proxy.example.com/openai",
        ]

        for custom_url in test_cases:
            with patch.dict(os.environ, {"OPENAI_BASE_URL": custom_url}):
                reload_env()
                provider = OpenAIModelProvider(api_key="test-key")
                assert provider.base_url == custom_url

    def test_explicit_base_url_override(self):
        """Test that explicit base_url in kwargs overrides environment variable."""
        env_url = "https://env-api.example.com/v1"
        explicit_url = "https://explicit-api.example.com/v1"

        with patch.dict(os.environ, {"OPENAI_BASE_URL": env_url}):
            reload_env()
            # When base_url is explicitly provided in kwargs, it should be used
            provider = OpenAIModelProvider(api_key="test-key", base_url=explicit_url)
            assert provider.base_url == explicit_url
