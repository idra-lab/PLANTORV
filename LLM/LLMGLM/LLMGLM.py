# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""GLM backend (OpenAI-compatible API)."""

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


class LLMGLM(BaseLLM):
    """GLM chat-completions backend, served through the OpenAI-compatible API.

    Configuration keys:
        LLM_VERSION: Model name.
        API_KEY_NAME: Environment variable holding the API key. ``ZHIPUAI_API_KEY`` is used as a
            fallback when the named variable is unset.
        BASE_URL: API base URL. Defaults to the BigModel endpoint.
        LLM_CONFIG: Request parameters, forwarded as-is.
    """

    PROVIDER = "glm"
    SUPPORTS_IMAGES = True

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    def _setup(self) -> None:
        """Read the GLM connection settings from the configuration."""
        self.api_key_name = self.config.get("API_KEY_NAME") or "GLM_API_KEY"
        self.api_key = self.config.get("API_KEY")
        self.base_url = resolve_config_value(
            self.config, "BASE_URL", self.DEFAULT_BASE_URL, allow_bare_env=True
        )

        logger.info("Model: %s", self.model)
        logger.info("Base URL: %s", self.base_url)
        logger.info("Request parameters: %s", self.request_params())

    def _create_client(self) -> OpenAI:
        """Create the GLM client.

        Returns
        -------
            OpenAI: SDK client pointed at the GLM endpoint.

        Raises
        ------
            ValueError: If the API key is missing.
        """
        api_key = (
            self.api_key or os.environ.get(self.api_key_name) or os.environ.get("ZHIPUAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Missing GLM API key. Set {} (or ZHIPUAI_API_KEY) or provide API_KEY in the "
                "config.".format(self.api_key_name)
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

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
        """Encode an image as an OpenAI-style ``image_url`` content part."""
        return {"type": "image_url", "image_url": {"url": image_data_url(image)}}


LLM = LLMGLM
