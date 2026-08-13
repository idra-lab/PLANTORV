# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Google Gemini backend for the shared BaseLLM interface."""

import os
import yaml
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types

try:
    from llm_base import BaseLLM, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, logger, resolve_config_value
    except Exception:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger, resolve_config_value


class LLMGemini(BaseLLM):
    """Gemini implementation via the google-genai SDK."""

    def __init__(
        self,
        llm_config_file: str,
        examples_yaml_file: Iterable[str] = ("",),
    ) -> None:
        """Initialize the Gemini backend from a YAML config.

        Args:
            llm_config_file (str): Path to YAML with API settings.
            examples_yaml_file (Iterable[str]): Optional few-shot examples.

        Raises:
            FileNotFoundError: If the YAML config file is missing.
            ValueError: If required config values are missing.
        """
        self._configured_key: Optional[str] = None
        self._configured_base_url: Optional[str] = None
        self._client: Optional[Any] = None

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
            or "GEMINI_API_KEY"
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
    def _normalize_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Any], Optional[str]]:
        """Normalize shared message format to Gemini-compatible payload."""
        system_chunks: List[str] = []
        normalized: List[Any] = []

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

            gemini_role = "model" if role == "assistant" else "user"
            normalized.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        if not normalized:
            normalized.append(types.Content(role="user", parts=[types.Part.from_text(text="")]))

        system_prompt = "\n\n".join(chunk for chunk in system_chunks if chunk.strip())
        return normalized, (system_prompt if system_prompt else None)

    def _connect(self, messages: List[Dict[str, Any]]) -> Any:
        """Send a Gemini request and return the raw response.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            Any: Gemini SDK response object.

        Raises:
            ValueError: If the API key is missing.
        """
        api_key = self.API_KEY or os.environ.get(self.API_KEY_NAME)
        if not api_key:
            raise ValueError("Missing Gemini API key. Set {} or provide API_KEY in config.".format(self.API_KEY_NAME))

        configured_base_url = None if self.BASE_URL in [None, "", "None"] else self.BASE_URL
        if self._configured_key != api_key or self._configured_base_url != configured_base_url or self._client is None:
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            try:
                if configured_base_url is not None:
                    # google-genai supports custom base URL via http_options.
                    client_kwargs["http_options"] = {"base_url": configured_base_url}
                self._client = genai.Client(**client_kwargs)
            except Exception:
                # Fallback for SDK variants where custom endpoint options differ.
                if configured_base_url is not None:
                    logger.warning("Unable to apply BASE_URL='%s' with google-genai; using default endpoint.", configured_base_url)
                self._client = genai.Client(api_key=api_key)
            self._configured_key = api_key
            self._configured_base_url = configured_base_url

        normalized_messages, system_prompt = self._normalize_messages(messages)

        generation_config_kwargs: Dict[str, Any] = {}
        if self.temperature is not None:
            generation_config_kwargs["temperature"] = self.temperature
        if self.top_p is not None and self.top_p > 0:
            generation_config_kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = self.max_tokens
        if self.stop not in [None, "", []]:
            if isinstance(self.stop, str):
                generation_config_kwargs["stop_sequences"] = [self.stop]
            else:
                generation_config_kwargs["stop_sequences"] = [str(seq) for seq in self.stop if str(seq).strip()]
        if system_prompt is not None:
            generation_config_kwargs["system_instruction"] = system_prompt

        response = self._client.models.generate_content(
            model=self.engine,
            contents=normalized_messages,
            config=types.GenerateContentConfig(**generation_config_kwargs)
            if generation_config_kwargs
            else None,
        )
        return response

    def _extract_content(self, response: Any) -> str:
        """Extract the assistant message text from Gemini response."""
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates is None:
            return ""

        chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)
        return "".join(chunks).strip()

    def _extract_completion_tokens(self, response: Any) -> int:
        """Extract completion token count if available."""
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return 0

        candidates_token_count = getattr(usage_metadata, "candidates_token_count", None)
        if isinstance(candidates_token_count, int):
            return candidates_token_count

        output_tokens = getattr(usage_metadata, "output_tokens", None)
        if isinstance(output_tokens, int):
            return output_tokens

        output_token_count = getattr(usage_metadata, "output_token_count", None)
        if isinstance(output_token_count, int):
            return output_token_count

        return 0

    def _extract_prompt_tokens(self, response: Any) -> int:
        """Extract prompt token count if available."""
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return 0

        prompt_token_count = getattr(usage_metadata, "prompt_token_count", None)
        if isinstance(prompt_token_count, int):
            return prompt_token_count

        input_tokens = getattr(usage_metadata, "input_tokens", None)
        if isinstance(input_tokens, int):
            return input_tokens

        return 0


LLM = LLMGemini
