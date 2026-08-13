# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Azure OpenAI backend"""

import yaml
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import AzureOpenAI

try:
    from llm_base import BaseLLM, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, logger, resolve_config_value
    except Exception:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger, resolve_config_value


class LLM(BaseLLM):
    """Azure OpenAI chat-completions implementation."""

    def __init__(
        self,
        llm_config_file: str = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "conf",
            "azure_gpt40-32k.yaml",
        ),
        examples_yaml_file: Iterable[str] = ("",),
        gpt5_mini: Optional[bool] = None,
    ) -> None:
        """Initialize the Azure OpenAI backend from a YAML config.

        Args:
            llm_config_file (str): Path to YAML with Azure settings.
            examples_yaml_file (Iterable[str]): Optional few-shot examples.
            gpt5_mini (Optional[bool]): Optional override for reduced parameter mode.

        Raises:
            FileNotFoundError: If the YAML config file is missing.
            KeyError: If required config values are missing.
        """
        llm_base_config = {}

        logger.info("LLM configuration file: %s", llm_config_file)
        if llm_config_file.endswith(".yaml") and os.path.isfile(llm_config_file):
            with open(llm_config_file) as file:
                llm_connection_config = yaml.load(file, Loader=yaml.FullLoader)

                self.engine       = llm_connection_config["LLM_VERSION"]
                self.API_KEY_NAME = llm_connection_config["API_KEY_NAME"]
                self.ENDPOINT     = resolve_config_value(llm_connection_config, "ENDPOINT", allow_bare_env=True)
                self.API_VERSION  = llm_connection_config["API_VERSION"]
                auto_gpt5_mode = ("gpt" in self.engine.lower() and "5" in self.engine.lower())
                self.gpt5_mini = auto_gpt5_mode if gpt5_mini is None else gpt5_mini
                llm_base_config = llm_connection_config.get("LLM_CONFIG", {})
                if not isinstance(llm_base_config, dict):
                    raise ValueError("LLM_CONFIG must be a dict when provided.")

                logger.info("LLM_VERSION: %s", self.engine)
                logger.info("API_KEY_NAME: %s", self.API_KEY_NAME)
                logger.info("ENDPOINT: %s", self.ENDPOINT)
                logger.info("API_VERSION: %s", self.API_VERSION)
                logger.info("LLM_CONFIG: %s", llm_base_config)

        else:
            raise FileNotFoundError("The selected file {} does not exist or is not a yaml file".format(llm_config_file))

        super().__init__(examples_yaml_file=examples_yaml_file, llm_config=llm_base_config)

    def _connect(self, messages: List[Dict[str, Any]]) -> Any:
        """Send a chat completion request and return the raw response.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            Any: Azure OpenAI SDK response object.

        Raises:
            KeyError: If the API key environment variable is missing.
        """
        client = AzureOpenAI(
            api_key        = os.environ[self.API_KEY_NAME],
            azure_endpoint = self.ENDPOINT,
            api_version    = self.API_VERSION,
        )

        if not client:
            raise ConnectionError("Failed to connect to Azure OpenAI. Check your API key and endpoint configuration.")

        # Added support for gpt-5-mini which does not support all parameters
        if not self.gpt5_mini:
            response = client.chat.completions.create(
                model = self.engine,
                **self._build_completion_kwargs(messages)
            )
        else:
            prepared_messages = self._prepare_messages(messages)
            response = client.chat.completions.create(
                model             = self.engine,
                messages          = prepared_messages,
                seed              = self.seed,
                stop              = self.stop,
            )
        
        return response

    def _extract_content(self, response: Any) -> str:
        """Extract the assistant message content.

        Args:
            response (Any): Azure OpenAI SDK response object.

        Returns:
            str: Assistant response text.
        """
        return response.choices[0].message.content

    def _extract_completion_tokens(self, response: Any) -> int:
        """Extract completion token count if available.

        Args:
            response (Any): Azure OpenAI SDK response object.

        Returns:
            int: Completion token count.
        """
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.completion_tokens
        return 0

    def _extract_prompt_tokens(self, response: Any) -> int:
        """Extract prompt token count if available.

        Args:
            response (Any): Azure OpenAI SDK response object.

        Returns:
            int: Prompt token count.
        """
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.prompt_tokens
        return 0


LLMAzureOpenAI = LLM
