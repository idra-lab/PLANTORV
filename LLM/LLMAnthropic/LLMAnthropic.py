# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Anthropic backend."""

import os
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic, AnthropicFoundry

try:
    from llm_base import BaseLLM, encode_image, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, encode_image, logger, resolve_config_value
    except Exception:
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, encode_image, logger, resolve_config_value


class LLMAnthropic(BaseLLM):
    """Anthropic Messages API backend.

    Configuration keys:
        LLM_VERSION: Model name.
        API_KEY_NAME: Environment variable holding the API key.
        BASE_URL: Optional base URL. When set, the Foundry client is used.
        LLM_CONFIG: Request parameters, forwarded as-is. ``max_tokens`` is required by the API,
            so it is always sent.
    """

    PROVIDER = "anthropic"
    SUPPORTS_IMAGES = True

    DEFAULT_PARAMS = {"max_tokens": 4096}
    PARAM_ALIASES = {"max_completion_tokens": "max_tokens", "stop": "stop_sequences"}

    def _setup(self) -> None:
        """Read the Anthropic connection settings from the configuration."""
        self.api_key_name = (
            self.config.get("API_KEY_NAME") or self.config.get("API_KEY_ENV") or "ANTHROPIC_API_KEY"
        )
        self.api_key = self.config.get("API_KEY")
        self.base_url = resolve_config_value(self.config, "BASE_URL", None, allow_bare_env=True)

        logger.info("Model: %s", self.model)
        logger.info("Base URL: %s", self.base_url)
        logger.info("Request parameters: %s", self.request_params())

    def _create_client(self) -> Any:
        """Create the Anthropic client.

        Returns
        -------
            Any: ``AnthropicFoundry`` when a base URL is configured, ``Anthropic`` otherwise.

        Raises
        ------
            ValueError: If the API key is missing.
        """
        api_key = self.api_key or os.environ.get(self.api_key_name)
        if not api_key:
            raise ValueError(
                "Missing Anthropic API key. Set {} or provide API_KEY in the config.".format(
                    self.api_key_name
                )
            )

        if self.base_url:
            logger.info("Initializing Anthropic client with base URL: %s", self.base_url)
            return AnthropicFoundry(api_key=api_key, base_url=self.base_url)

        return Anthropic(api_key=api_key)

    @staticmethod
    def _split_system(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Split system messages out of the message list, as the API expects them apart."""
        system_chunks: List[str] = []
        conversation: List[Dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            content = message.get("content", "")

            if role == "system":
                if isinstance(content, str) and content.strip():
                    system_chunks.append(content)
                continue

            conversation.append(
                {"role": role if role in ("user", "assistant") else "user", "content": content}
            )

        if not conversation:
            conversation.append({"role": "user", "content": ""})

        system_prompt = "\n\n".join(system_chunks)
        return conversation, (system_prompt or None)

    def _send(self, client: Any, messages: List[Dict[str, Any]]) -> Any:
        """Send a messages request."""
        conversation, system_prompt = self._split_system(messages)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            **self.request_params(),
        }
        if system_prompt is not None:
            request_kwargs["system"] = system_prompt

        return client.messages.create(**request_kwargs)

    def _extract_text(self, response: Any) -> str:
        """Concatenate the text blocks of the response."""
        blocks = getattr(response, "content", None) or []
        chunks = [
            block.text
            for block in blocks
            if getattr(block, "type", None) == "text" and getattr(block, "text", None) is not None
        ]
        return "".join(chunks).strip()

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage, which Anthropic names input/output tokens."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        return {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }

    def image_part(self, image: Any) -> Dict[str, Any]:
        """Encode an image as an Anthropic base64 image block."""
        mime_type, encoded = encode_image(image)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": encoded},
        }


LLM = LLMAnthropic
