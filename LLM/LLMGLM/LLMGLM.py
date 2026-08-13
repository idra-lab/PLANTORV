# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""GLM backend for the shared BaseLLM interface."""

import os
import yaml
from typing import Any, Dict, Iterable, List

from openai import OpenAI

try:
    from llm_base import BaseLLM, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, logger, resolve_config_value
    except Exception:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger, resolve_config_value


class LLMGLM(BaseLLM):
    """GLM chat-completions implementation (OpenAI-compatible API)."""

    def __init__(
        self,
        llm_config_file: str = os.path.join(os.path.dirname(__file__), "../conf/glm4_7.yaml"),
        examples_yaml_file: Iterable[str] = ("",),
    ) -> None:
        """Initialize the GLM backend from a YAML config.

        Args:
            llm_config_file (str): Path to YAML with API settings.
            examples_yaml_file (Iterable[str]): Optional few-shot examples.

        Raises:
            FileNotFoundError: If the YAML config file is missing.
            ValueError: If required config values are missing.
        """
        self._client = None

        logger.info("LLM configuration file: %s", llm_config_file)
        if not llm_config_file.endswith(".yaml") or not os.path.isfile(llm_config_file):
            raise FileNotFoundError(
                "The selected file {} does not exist or is not a yaml file".format(llm_config_file)
            )

        with open(llm_config_file) as file:
            llm_connection_config = yaml.load(file, Loader=yaml.FullLoader)

        self.engine = (
            llm_connection_config.get("LLM_VERSION")
            or llm_connection_config.get("MODEL")
            or llm_connection_config.get("MODEL_NAME")
        )
        if not self.engine:
            raise ValueError("Missing model name in config. Expected LLM_VERSION (or MODEL/MODEL_NAME).")

        self.API_KEY_NAME = (
            llm_connection_config.get("API_KEY_NAME")
            or llm_connection_config.get("API_KEY_ENV")
            or "GLM_API_KEY"
        )
        self.API_KEY = llm_connection_config.get("API_KEY", None)
        self.BASE_URL = resolve_config_value(
            llm_connection_config,
            "BASE_URL",
            "https://open.bigmodel.cn/api/paas/v4/",
            allow_bare_env=True,
        )

        config_from_yaml = llm_connection_config.get("LLM_CONFIG", {})
        if not isinstance(config_from_yaml, dict):
            raise ValueError("LLM_CONFIG must be a dict when provided.")

        logger.info("LLM_VERSION: %s", self.engine)
        logger.info("API_KEY_NAME: %s", self.API_KEY_NAME)
        logger.info("BASE_URL: %s", self.BASE_URL)
        logger.info("LLM_CONFIG: %s", config_from_yaml)

        super().__init__(examples_yaml_file=examples_yaml_file, llm_config=config_from_yaml)

    def _connect(self, messages: List[Dict[str, Any]]) -> Any:
        """Send a chat completion request and return the raw response."""
        api_key = self.API_KEY or os.environ.get(self.API_KEY_NAME) or os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GLM API key. Set {} (or ZHIPUAI_API_KEY) or provide API_KEY in config.".format(
                    self.API_KEY_NAME
                )
            )

        client_kwargs = {"api_key": api_key}
        if self.BASE_URL not in [None, "", "None"]:
            client_kwargs["base_url"] = self.BASE_URL

        if self._client is None:
            self._client = OpenAI(**client_kwargs)
        client = self._client

        request_kwargs: Dict[str, Any] = {
            "model": self.engine,
            "messages": messages,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None and self.top_p > 0:
            request_kwargs["top_p"] = self.top_p
        if self.stop not in [None, "", []]:
            request_kwargs["stop"] = self.stop

        response = client.chat.completions.create(**request_kwargs)
        return response

    def _extract_content(self, response: Any) -> str:
        """Extract the assistant message content."""
        return response.choices[0].message.content

    def _extract_completion_tokens(self, response: Any) -> int:
        """Extract completion token count if available."""
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.completion_tokens
        return 0

    def _extract_prompt_tokens(self, response: Any) -> int:
        """Extract prompt token count if available."""
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.prompt_tokens
        return 0


LLM = LLMGLM
