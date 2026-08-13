# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Anthropic API backend for the shared BaseLLM interface."""

import os
import yaml
from typing import Any, Dict, Iterable, List, Optional, Tuple

from anthropic import Anthropic, AnthropicFoundry

try:
    from llm_base import BaseLLM, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, logger, resolve_config_value
    except Exception:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger, resolve_config_value


class LLMAnthropic(BaseLLM):
    """Anthropic Messages API implementation."""

    def __init__(
        self,
        llm_config_file: str,
        examples_yaml_file: Iterable[str] = ("",),
    ) -> None:
        """Initialize the Anthropic backend from a YAML config.

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
            or "ANTHROPIC_API_KEY"
        )
        self.API_KEY = llm_connection_config.get("API_KEY", None)
        self.BASE_URL = resolve_config_value(llm_connection_config, "BASE_URL", None, allow_bare_env=True)

        config_from_yaml = llm_connection_config.get("LLM_CONFIG", {})
        if not isinstance(config_from_yaml, dict):
            raise ValueError("LLM_CONFIG must be a dict when provided.")

        logger.info("LLM_VERSION: %s", self.engine)
        logger.info("API_KEY_NAME: %s", self.API_KEY_NAME)
        logger.info("BASE_URL: %s", self.BASE_URL)
        logger.info("LLM_CONFIG: %s", config_from_yaml)

        super().__init__(examples_yaml_file=examples_yaml_file, llm_config=config_from_yaml)

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Normalize shared message format to Anthropic-compatible payload."""
        system_chunks: List[str] = []
        normalized: List[Dict[str, str]] = []

        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).strip().lower()
                content = msg.get("content", "")
            else:
                role = "system"
                content = str(msg)

            if isinstance(content, list):
                content = " ".join(str(chunk) for chunk in content)
            else:
                content = str(content)

            if role == "system":
                if content.strip():
                    system_chunks.append(content)
                continue

            if role not in ("user", "assistant"):
                role = "user"

            normalized.append({"role": role, "content": content})

        if not normalized:
            normalized.append({"role": "user", "content": ""})

        system_prompt = "\n\n".join(chunk for chunk in system_chunks if chunk.strip())
        return normalized, (system_prompt if system_prompt else None)

    def _connect(self, messages: List[Dict[str, Any]]) -> Any:
        """Send a messages request and return the raw response.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            Any: Anthropic SDK response object.

        Raises:
            ValueError: If the API key is missing.
        """
        api_key = self.API_KEY or os.environ.get(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                "Missing Anthropic API key. Set {} or provide API_KEY in config.".format(self.API_KEY_NAME)
            )

        client_kwargs = {"api_key": api_key}
        if self.BASE_URL not in [None, "", "None"]:
            client_kwargs["base_url"] = self.BASE_URL

        if self._client is None:
            if self.BASE_URL not in [None, "", "None"]:
                logger.info("Initializing Anthropic client with base URL: %s", self.BASE_URL)
                self._client = AnthropicFoundry(**client_kwargs)
            else:
                logger.info("Initializing Anthropic client without base URL")
                self._client = Anthropic(**client_kwargs)
        client = self._client

        normalized_messages, system_prompt = self._normalize_messages(messages)

        request_kwargs: Dict[str, Any] = {
            "model": self.engine,
            "messages": normalized_messages,
            "max_tokens": self.max_tokens if self.max_tokens is not None else BaseLLM.llm_default_config["max_tokens"],
        }

        if system_prompt is not None:
            request_kwargs["system"] = system_prompt
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None and self.top_p > 0:
            request_kwargs["top_p"] = self.top_p
        if self.stop not in [None, "", []]:
            if isinstance(self.stop, str):
                request_kwargs["stop_sequences"] = [self.stop]
            else:
                request_kwargs["stop_sequences"] = [str(seq) for seq in self.stop if str(seq).strip()]

        response = client.messages.create(**request_kwargs)
        return response

    def _extract_content(self, response: Any) -> str:
        """Extract concatenated assistant text content."""
        if not hasattr(response, "content") or response.content is None:
            return ""

        chunks: List[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
            if block_type == "text" and text is not None:
                chunks.append(text)

        return "".join(chunks).strip()

    def _extract_completion_tokens(self, response: Any) -> int:
        """Extract completion token count if available."""
        if hasattr(response, "usage") and response.usage is not None:
            output_tokens = getattr(response.usage, "output_tokens", None)
            if isinstance(output_tokens, int):
                return output_tokens
        return 0

    def _extract_prompt_tokens(self, response: Any) -> int:
        """Extract prompt token count if available."""
        if hasattr(response, "usage") and response.usage is not None:
            input_tokens = getattr(response.usage, "input_tokens", None)
            if isinstance(input_tokens, int):
                return input_tokens
        return 0


LLM = LLMAnthropic
