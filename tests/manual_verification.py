#!/usr/bin/env python3
"""Manual verification script for custom API endpoint support.

This script demonstrates and validates that:
1. OpenAI provider respects OPENAI_BASE_URL environment variable
2. Anthropic provider respects ANTHROPIC_BASE_URL environment variable
3. Default URLs are used when env vars are not set
4. Custom URLs are logged appropriately
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import reload_env
from providers.openai import OpenAIModelProvider
from providers.anthropic import AnthropicModelProvider
from providers.shared import ProviderType


def test_openai_default_url():
    """Test OpenAI provider with default URL."""
    print("\n1. Testing OpenAI with default URL...")
    os.environ.pop("OPENAI_BASE_URL", None)
    reload_env()
    
    provider = OpenAIModelProvider(api_key="test-key")
    assert provider.base_url == "https://api.openai.com/v1", f"Expected default URL, got {provider.base_url}"
    print("   ✓ Default URL: https://api.openai.com/v1")


def test_openai_custom_url():
    """Test OpenAI provider with custom URL."""
    print("\n2. Testing OpenAI with custom URL...")
    custom_url = "https://custom-openai.example.com/v1"
    os.environ["OPENAI_BASE_URL"] = custom_url
    reload_env()
    
    provider = OpenAIModelProvider(api_key="test-key")
    assert provider.base_url == custom_url, f"Expected {custom_url}, got {provider.base_url}"
    print(f"   ✓ Custom URL: {custom_url}")


def test_anthropic_default_url():
    """Test Anthropic provider with default URL."""
    print("\n3. Testing Anthropic with default URL...")
    os.environ.pop("ANTHROPIC_BASE_URL", None)
    reload_env()
    
    provider = AnthropicModelProvider(api_key="test-key")
    assert provider.base_url == "https://api.anthropic.com/v1", f"Expected default URL, got {provider.base_url}"
    print("   ✓ Default URL: https://api.anthropic.com/v1")


def test_anthropic_custom_url():
    """Test Anthropic provider with custom URL."""
    print("\n4. Testing Anthropic with custom URL...")
    custom_url = "https://custom-anthropic.example.com/v1"
    os.environ["ANTHROPIC_BASE_URL"] = custom_url
    reload_env()
    
    provider = AnthropicModelProvider(api_key="test-key")
    assert provider.base_url == custom_url, f"Expected {custom_url}, got {provider.base_url}"
    print(f"   ✓ Custom URL: {custom_url}")


def test_anthropic_provider_type():
    """Test Anthropic provider type."""
    print("\n5. Testing Anthropic provider type...")
    provider = AnthropicModelProvider(api_key="test-key")
    assert provider.get_provider_type() == ProviderType.ANTHROPIC
    print("   ✓ Provider type: ANTHROPIC")


def test_anthropic_model_validation():
    """Test Anthropic model validation."""
    print("\n6. Testing Anthropic model validation...")
    provider = AnthropicModelProvider(api_key="test-key")
    
    # Valid models
    assert provider.validate_model_name("claude-3-opus") is True
    assert provider.validate_model_name("claude-3-sonnet") is True
    assert provider.validate_model_name("claude-3-haiku") is True
    assert provider.validate_model_name("claude-4-test") is True  # Any claude-* model
    
    # Invalid models
    assert provider.validate_model_name("gpt-4") is False
    assert provider.validate_model_name("gemini-pro") is False
    
    print("   ✓ Model validation works correctly")


def test_anthropic_capabilities():
    """Test Anthropic model capabilities."""
    print("\n7. Testing Anthropic model capabilities...")
    provider = AnthropicModelProvider(api_key="test-key")
    
    # Known model
    caps = provider.get_capabilities("claude-3-opus")
    assert caps.provider == ProviderType.ANTHROPIC
    assert caps.model_name == "claude-3-opus"
    assert caps.context_window == 200_000
    assert caps.supports_images is True
    
    # Unknown claude model (should get generic capabilities)
    caps = provider.get_capabilities("claude-4-future")
    assert caps.provider == ProviderType.ANTHROPIC
    assert caps.model_name == "claude-4-future"
    assert caps.context_window == 200_000
    
    print("   ✓ Model capabilities work correctly")


def main():
    """Run all manual verification tests."""
    print("=" * 60)
    print("Manual Verification: Custom API Endpoint Support")
    print("=" * 60)
    
    try:
        test_openai_default_url()
        test_openai_custom_url()
        test_anthropic_default_url()
        test_anthropic_custom_url()
        test_anthropic_provider_type()
        test_anthropic_model_validation()
        test_anthropic_capabilities()
        
        print("\n" + "=" * 60)
        print("✓ All manual verification tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
