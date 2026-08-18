"""Azure OpenAI backend."""

import os
from typing import Any, Dict, List, cast

from openai import AzureOpenAI

try:
    from llm_base import BaseLLM, image_data_url, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, image_data_url, logger, resolve_config_value
    except Exception:
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, image_data_url, logger, resolve_config_value


class LLMAzureOpenAI(BaseLLM):
    """Azure OpenAI chat-completions backend.

    Configuration keys:
        LLM_VERSION: Model name.
        DEPLOYMENT: Azure deployment name. Defaults to ``LLM_VERSION``.
        API_KEY_NAME: Environment variable holding the API key.
        ENDPOINT / ENDPOINT_ENV: Endpoint URL, or the environment variable holding it.
        API_VERSION: Azure API version.
        IMAGE_DETAIL: Detail level sent with images ("auto", "low", "high").
        LLM_CONFIG: Request parameters, forwarded as-is. Use the names the deployment expects,
            e.g. ``max_completion_tokens`` for the gpt-5 family and ``max_tokens`` before it.
    """

    PROVIDER = "azure_openai"
    SUPPORTS_IMAGES = True

    def _setup(self) -> None:
        """Read the Azure connection settings from the configuration."""
        self.deployment = self.config.get("DEPLOYMENT") or self.model
        self.api_key_name = self.config.get("API_KEY_NAME") or "AZURE_OPENAI_API_KEY"
        self.api_key = self.config.get("API_KEY")
        self.endpoint = resolve_config_value(self.config, "ENDPOINT", allow_bare_env=True)
        self.api_version = self.config.get("API_VERSION")
        self.image_detail = self.config.get("IMAGE_DETAIL", "auto")

        logger.info("Model: %s (deployment: %s)", self.model, self.deployment)
        logger.info("Endpoint: %s", self.endpoint)
        logger.info("API version: %s", self.api_version)
        logger.info("Request parameters: %s", self.request_params())

    def _create_client(self) -> AzureOpenAI:
        """Create the Azure OpenAI client.

        Returns
        -------
        AzureOpenAI
            Configured SDK client.

        Raises
        ------
        ValueError
            If the API key or the endpoint is missing.
        """
        api_key = self.api_key or os.environ.get(self.api_key_name)
        if not api_key:
            raise ValueError(
                "Missing Azure OpenAI API key. Set {} in LLM/.env or in the shell "
                "environment, or provide API_KEY in the config.".format(self.api_key_name)
            )
        if not self.endpoint:
            raise ValueError(
                "Missing Azure OpenAI endpoint. Set ENDPOINT or ENDPOINT_ENV in the config."
            )

        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def _send(self, client: AzureOpenAI, messages: List[Dict[str, Any]]) -> Any:
        """Send a chat completion request.

        Parameters
        ----------
        client : AzureOpenAI
            The SDK client returned by :meth:`connect`.
        messages : List[Dict[str, Any]]
            Messages in the shared chat format.

        Returns
        -------
        Any
            The raw chat-completion response.
        """
        return client.chat.completions.create(
            model=self.deployment,
            messages=cast(Any, messages),
            **self.request_params(),
        )

    def _extract_text(self, response: Any) -> str:
        """Extract the assistant message content.

        Parameters
        ----------
        response : Any
            Raw chat-completion response.

        Returns
        -------
        str
            The assistant answer, or the empty string when the message has no content.
        """
        return response.choices[0].message.content or ""

    def image_part(self, image: Any) -> Dict[str, Any]:
        """Encode an image as an OpenAI ``image_url`` content part.

        Parameters
        ----------
        image : Any
            A ``PIL.Image.Image``, or the path of an image file.

        Returns
        -------
        Dict[str, Any]
            An ``image_url`` content part holding a ``data:`` URL and the configured detail level.
        """
        return {
            "type": "image_url",
            "image_url": {"url": image_data_url(image), "detail": self.image_detail},
        }


LLM = LLMAzureOpenAI
