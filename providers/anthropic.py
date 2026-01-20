"""Anthropic model provider implementation.

This provider supports Anthropic-compatible APIs, allowing users to:
1. Use the official Anthropic API with custom base URL
2. Use third-party Anthropic-compatible endpoints (e.g., proxies, gateways)

Note: This implementation uses the OpenAI-compatible interface since many
Anthropic-compatible services support this format. For full native Anthropic
API support, the anthropic SDK would need to be added to requirements.txt.
"""

import logging
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class AnthropicModelProvider(OpenAICompatibleProvider):
    """Provider for Anthropic-compatible APIs using OpenAI-compatible format.

    This provider enables:
    - Custom base URL via ANTHROPIC_BASE_URL environment variable
    - Support for Anthropic-compatible proxy services
    - Flexibility for enterprise deployments with custom endpoints

    Default endpoint: https://api.anthropic.com/v1 (would require native SDK)
    For OpenAI-compatible Anthropic proxies, set ANTHROPIC_BASE_URL accordingly.
    """

    FRIENDLY_NAME = "Anthropic"

    # Track whether we've already logged the native SDK warning
    _sdk_warning_logged = False

    # Basic model capabilities for common Anthropic models
    # These are conservative defaults; actual capabilities may vary
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {
        # Claude 4 Series (Latest - May 2025)
        "claude-opus-4": ModelCapabilities(
            model_name="claude-opus-4",
            friendly_name="Claude Opus 4",
            provider=ProviderType.ANTHROPIC,
            context_window=200_000,
            max_output_tokens=32_000,
            supports_images=True,
            supports_streaming=True,
            supports_extended_thinking=True,
            aliases=["opus", "opus-4"],
            description="Most intelligent Claude model - Complex reasoning, research, analysis",
        ),
        "claude-sonnet-4": ModelCapabilities(
            model_name="claude-sonnet-4",
            friendly_name="Claude Sonnet 4",
            provider=ProviderType.ANTHROPIC,
            context_window=200_000,
            max_output_tokens=16_000,
            supports_images=True,
            supports_streaming=True,
            supports_extended_thinking=True,
            aliases=["sonnet", "sonnet-4"],
            description="Balanced performance and speed - Coding, writing, general tasks",
        ),
        # Claude 3.7 Series
        "claude-3.7-sonnet": ModelCapabilities(
            model_name="claude-3.7-sonnet",
            friendly_name="Claude 3.7 Sonnet",
            provider=ProviderType.ANTHROPIC,
            context_window=200_000,
            max_output_tokens=8_192,
            supports_images=True,
            supports_streaming=True,
            supports_extended_thinking=True,
            aliases=["sonnet-3.7"],
            description="Hybrid reasoning model with extended thinking capabilities",
        ),
        # Claude 3.5 Series (October 2024)
        "claude-3.5-sonnet": ModelCapabilities(
            model_name="claude-3.5-sonnet",
            friendly_name="Claude 3.5 Sonnet",
            provider=ProviderType.ANTHROPIC,
            context_window=200_000,
            max_output_tokens=8_192,
            supports_images=True,
            supports_streaming=True,
            supports_extended_thinking=False,
            aliases=["sonnet-3.5"],
            description="Fast and capable - General purpose tasks",
        ),
        "claude-3.5-haiku": ModelCapabilities(
            model_name="claude-3.5-haiku",
            friendly_name="Claude 3.5 Haiku",
            provider=ProviderType.ANTHROPIC,
            context_window=200_000,
            max_output_tokens=8_192,
            supports_images=True,
            supports_streaming=True,
            supports_extended_thinking=False,
            aliases=["haiku", "haiku-3.5"],
            description="Fastest Claude model - Quick responses, simple tasks",
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Anthropic provider with API key and optional base URL.

        Args:
            api_key: Anthropic API key from ANTHROPIC_API_KEY
            **kwargs: Additional arguments, including optional base_url override
        """
        # Allow override via ANTHROPIC_BASE_URL for custom endpoints
        default_base_url = get_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        kwargs.setdefault("base_url", default_base_url)

        # Log if using custom endpoint
        if default_base_url != "https://api.anthropic.com/v1":
            logger.info(f"Using custom Anthropic endpoint: {default_base_url}")
        elif not AnthropicModelProvider._sdk_warning_logged:
            # Only warn once about native SDK requirement
            logger.debug(
                "Anthropic provider initialized with default endpoint. "
                "Note: Full native Anthropic API support requires the 'anthropic' SDK. "
                "For OpenAI-compatible Anthropic proxies, set ANTHROPIC_BASE_URL."
            )
            AnthropicModelProvider._sdk_warning_logged = True

        super().__init__(api_key, **kwargs)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ANTHROPIC

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if a model name is supported by this provider.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if the model is supported, False otherwise
        """
        # Accept known Anthropic model patterns
        model_lower = model_name.lower()
        return model_lower.startswith("claude-") or model_name in self.MODEL_CAPABILITIES

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelCapabilities for the requested model

        Raises:
            ValueError: If the model is not supported
        """
        if not self.validate_model_name(model_name):
            raise ValueError(f"Unsupported model '{model_name}' for provider anthropic")

        # Return known capabilities if available
        if model_name in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[model_name]

        # Generate generic capabilities for unknown Claude models
        logger.debug(f"Using generic capabilities for Anthropic model: {model_name}")
        return ModelCapabilities(
            model_name=model_name,
            friendly_name=f"Anthropic ({model_name})",
            provider=ProviderType.ANTHROPIC,
            context_window=200_000,  # Conservative default
            max_output_tokens=4_096,  # Conservative default
            supports_images=True,
            supports_streaming=True,
            supports_extended_thinking=False,
        )

    def list_models(self, respect_restrictions: bool = True) -> list[str]:
        """List available Anthropic models.

        Args:
            respect_restrictions: Whether to filter by ANTHROPIC_ALLOWED_MODELS

        Returns:
            List of available model names
        """
        models = list(self.MODEL_CAPABILITIES.keys())

        if respect_restrictions:
            from utils.model_restrictions import get_restriction_service

            restriction_service = get_restriction_service()
            models = [m for m in models if restriction_service.is_allowed(ProviderType.ANTHROPIC, m)]

        return models

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get Anthropic's preferred model for a given category.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        # Helper to find first available from preference list
        def find_first(preferences: list[str]) -> Optional[str]:
            for model in preferences:
                if model in allowed_models:
                    return model
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer Opus 4 for complex reasoning (supports extended thinking)
            preferred = find_first(["claude-opus-4", "claude-sonnet-4", "claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3.5-haiku"])
            return preferred if preferred else allowed_models[0]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer Haiku for fast responses
            preferred = find_first(["claude-3.5-haiku", "claude-3.5-sonnet", "claude-sonnet-4", "claude-opus-4"])
            return preferred if preferred else allowed_models[0]

        else:  # BALANCED or default
            # Prefer Sonnet 4 for balanced performance
            preferred = find_first(["claude-sonnet-4", "claude-3.7-sonnet", "claude-3.5-sonnet", "claude-opus-4", "claude-3.5-haiku"])
            return preferred if preferred else allowed_models[0]
