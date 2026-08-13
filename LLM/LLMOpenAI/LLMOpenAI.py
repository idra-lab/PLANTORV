# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""OpenAI API backend for the shared BaseLLM interface."""

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


class LLMOpenAI(BaseLLM):
    """OpenAI chat-completions implementation."""

    def __init__(
        self,
        llm_config_file: str = os.path.join(os.path.dirname(__file__), "../conf/gpt4o.yaml"),
        examples_yaml_file: Iterable[str] = ("",),
        gpt5_mini: bool = False,
    ) -> None:
        """Initialize the OpenAI backend from a YAML config.

        Args:
            llm_config_file (str): Path to YAML with API settings.
            examples_yaml_file (Iterable[str]): Optional few-shot examples.
            gpt5_mini (bool): Use restricted parameter set when True.

        Raises:
            FileNotFoundError: If the YAML config file is missing.
            ValueError: If required config values are missing.
        """
        self.gpt5_mini = gpt5_mini
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
            or "OPENAI_API_KEY"
        )
        self.API_KEY = llm_connection_config.get("API_KEY", None)

        self.BASE_URL = resolve_config_value(llm_connection_config, "BASE_URL", None, allow_bare_env=True)
        self.ORGANIZATION = resolve_config_value(llm_connection_config, "ORGANIZATION", None)
        self.PROJECT = resolve_config_value(llm_connection_config, "PROJECT", None)
        llm_base_config = llm_connection_config.get("LLM_CONFIG", {})
        if not isinstance(llm_base_config, dict):
            raise ValueError("LLM_CONFIG must be a dict when provided.")

        if "ENDPOINT" in llm_connection_config or "API_VERSION" in llm_connection_config:
            logger.warning("Azure-style fields detected in OpenAI config. Use LLMAzureOpenAI for Azure endpoints.")

        logger.info("LLM_VERSION: %s", self.engine)
        logger.info("API_KEY_NAME: %s", self.API_KEY_NAME)
        logger.info("BASE_URL: %s", self.BASE_URL)
        logger.info("ORGANIZATION: %s", self.ORGANIZATION)
        logger.info("PROJECT: %s", self.PROJECT)
        logger.info("LLM_CONFIG: %s", llm_base_config)

        super().__init__(examples_yaml_file=examples_yaml_file, llm_config=llm_base_config)

    def _connect(self, messages: List[Dict[str, Any]]) -> Any:
        """Send a chat completion request and return the raw response.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            Any: OpenAI SDK response object.

        Raises:
            ValueError: If the API key is missing.
        """
        api_key = self.API_KEY or os.environ.get(self.API_KEY_NAME)
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set {} or provide API_KEY in config.".format(self.API_KEY_NAME))

        client_kwargs = {"api_key": api_key}

        if self.BASE_URL not in [None, "", "None"]:
            client_kwargs["base_url"] = self.BASE_URL

        if self.ORGANIZATION not in [None, "", "None"]:
            client_kwargs["organization"] = self.ORGANIZATION
        if self.PROJECT not in [None, "", "None"]:
            client_kwargs["project"] = self.PROJECT

        if self._client is None:
            self._client = OpenAI(**client_kwargs)
        client = self._client

        if not self.gpt5_mini:
            response = client.chat.completions.create(
                model=self.engine,
                **self._build_completion_kwargs(messages)
            )
        else:
            prepared_messages = self._prepare_messages(messages)
            response = client.chat.completions.create(
                model=self.engine,
                messages=prepared_messages,
                seed=self.seed,
                stop=self.stop,
            )

        return response

    def _extract_content(self, response: Any) -> str:
        """Extract the assistant message content.

        Args:
            response (Any): OpenAI SDK response object.

        Returns:
            str: Assistant response text.
        """
        return response.choices[0].message.content

    def _extract_completion_tokens(self, response: Any) -> int:
        """Extract completion token count if available.

        Args:
            response (Any): OpenAI SDK response object.

        Returns:
            int: Completion token count.
        """
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.completion_tokens
        return 0

    def _extract_prompt_tokens(self, response: Any) -> int:
        """Extract prompt token count if available.

        Args:
            response (Any): OpenAI SDK response object.

        Returns:
            int: Prompt token count.
        """
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.prompt_tokens
        return 0


LLM = LLMOpenAI
