# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""OpenAI backend."""

import os
from typing import Any, Dict, List, cast

from openai import OpenAI

try:
    from llm_base import BaseLLM, image_data_url, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, image_data_url, logger, resolve_config_value
    except Exception:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, image_data_url, logger, resolve_config_value


class LLMOpenAI(BaseLLM):
    """OpenAI chat-completions backend.

    Configuration keys:
        LLM_VERSION: Model name.
        API_KEY_NAME: Environment variable holding the API key.
        BASE_URL: Optional custom base URL (also accepts ``BASE_URL_ENV``).
        ORGANIZATION / PROJECT: Optional OpenAI account scoping.
        IMAGE_DETAIL: Detail level sent with images ("auto", "low", "high").
        LLM_CONFIG: Request parameters, forwarded as-is. Use the names the model expects,
            e.g. ``max_completion_tokens`` for the gpt-5 family and ``max_tokens`` before it.
    """

    PROVIDER = "openai"
    SUPPORTS_IMAGES = True

    def _setup(self) -> None:
        """Read the OpenAI connection settings from the configuration."""
        self.api_key_name = self.config.get("API_KEY_NAME") or "OPENAI_API_KEY"
        self.api_key = self.config.get("API_KEY")
        self.base_url = resolve_config_value(self.config, "BASE_URL", None, allow_bare_env=True)
        self.organization = resolve_config_value(self.config, "ORGANIZATION", None)
        self.project = resolve_config_value(self.config, "PROJECT", None)
        self.image_detail = self.config.get("IMAGE_DETAIL", "auto")

        if "ENDPOINT" in self.config or "API_VERSION" in self.config:
            logger.warning("Azure-style fields detected in OpenAI config. Use LLMAzureOpenAI for Azure endpoints.")

        logger.info("Model: %s", self.model)
        logger.info("Base URL: %s", self.base_url)
        logger.info("Request parameters: %s", self.request_params())

    def _create_client(self) -> OpenAI:
        """Create the OpenAI client.

        Returns:
            OpenAI: Configured SDK client.

        Raises:
            ValueError: If the API key is missing.
        """
        api_key = self.api_key or os.environ.get(self.api_key_name)
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Set {} or provide API_KEY in the config.".format(self.api_key_name)
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.organization:
            client_kwargs["organization"] = self.organization
        if self.project:
            client_kwargs["project"] = self.project

        return OpenAI(**client_kwargs)

    def _send(self, client: OpenAI, messages: List[Dict[str, Any]]) -> Any:
        """Send a chat completion request."""
        return client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            **self.request_params(),
        )

    def _extract_text(self, response: Any) -> str:
        """Extract the assistant message content."""
        return response.choices[0].message.content or ""

    def image_part(self, image: Any) -> Dict[str, Any]:
        """Encode an image as an OpenAI ``image_url`` content part."""
        return {
            "type": "image_url",
            "image_url": {"url": image_data_url(image), "detail": self.image_detail},
        }


LLM = LLMOpenAI
