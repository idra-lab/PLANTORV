"""Google Gemini backend."""

import base64
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google import genai
from google.genai import types

try:
    from llm_base import BaseLLM, ImageInput, encode_image, logger, resolve_config_value
except Exception:
    try:
        from ..llm_base import BaseLLM, ImageInput, encode_image, logger, resolve_config_value
    except Exception:
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, ImageInput, encode_image, logger, resolve_config_value


class LLMGemini(BaseLLM):
    """Gemini backend built on the google-genai SDK.

    Configuration keys:
        LLM_VERSION: Model name.
        API_KEY_NAME: Environment variable holding the API key.
        BASE_URL: Optional custom endpoint.
        LLM_CONFIG: Request parameters, forwarded to ``GenerateContentConfig``. Generic names are
            translated, so ``max_tokens`` reaches the API as ``max_output_tokens``.
    """

    PROVIDER = "gemini"
    SUPPORTS_IMAGES = True

    PARAM_ALIASES = {
        "max_tokens": "max_output_tokens",
        "max_completion_tokens": "max_output_tokens",
        "stop": "stop_sequences",
    }

    def _setup(self) -> None:
        """Read the Gemini connection settings from the configuration."""
        self.api_key_name = (
            self.config.get("API_KEY_NAME") or self.config.get("API_KEY_ENV") or "GEMINI_API_KEY"
        )
        self.api_key = self.config.get("API_KEY")
        self.base_url = resolve_config_value(self.config, "BASE_URL", None, allow_bare_env=True)

        logger.info("Model: %s", self.model)
        logger.info("Base URL: %s", self.base_url)
        logger.info("Request parameters: %s", self.request_params())

    def _create_client(self) -> Any:
        """Create the Gemini client.

        Returns
        -------
        Any
            Configured ``genai.Client``. The default endpoint is used when ``BASE_URL`` cannot be
            applied by the installed SDK version.

        Raises
        ------
        ValueError
            If the API key is missing.
        """
        api_key = self.api_key or os.environ.get(self.api_key_name)
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set {} or provide API_KEY in the config.".format(
                    self.api_key_name
                )
            )

        if not self.base_url:
            return genai.Client(api_key=api_key)

        try:
            # google-genai takes a custom endpoint through http_options.
            return genai.Client(api_key=api_key, http_options={"base_url": self.base_url})
        except Exception:
            logger.warning(
                "Unable to apply BASE_URL='%s' with google-genai; using the default endpoint.",
                self.base_url,
            )
            return genai.Client(api_key=api_key)

    def build_content(
        # ImageInput is quoted on purpose: the try/except import block above declares it more than
        # once, so type checkers only resolve it as a type alias through a forward reference.
        self,
        message: str,
        images: Optional[Sequence["ImageInput"]] = None,
    ) -> List[types.Part]:
        """Build message content as Gemini parts.

        Parameters
        ----------
        message : str
            The text of the message.
        images : Optional[Sequence[ImageInput]], optional
            Images to attach.

        Returns
        -------
        List[types.Part]
            The text part followed by one inline data part per image.
        """
        parts: List[types.Part] = [types.Part.from_text(text=message)]
        for image in images or []:
            parts.append(self.image_part(image))
        return parts

    def image_part(self, image: Any) -> types.Part:
        """Encode an image as a Gemini inline data part.

        Parameters
        ----------
        image : Any
            A ``PIL.Image.Image``, or the path of an image file.

        Returns
        -------
        types.Part
            A part holding the raw image bytes and their MIME type.
        """
        mime_type, encoded = encode_image(image)
        return types.Part.from_bytes(data=base64.b64decode(encoded), mime_type=mime_type)

    def _to_contents(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[types.Content], Optional[str]]:
        """Convert shared messages into Gemini contents plus the system instruction.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Messages in the shared chat format.

        Returns
        -------
        Tuple[List[types.Content], Optional[str]]
            The conversation contents, and the joined system instruction (``None`` when no
            system message is present).
        """
        system_chunks: List[str] = []
        contents: List[types.Content] = []

        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            content = message.get("content", "")

            if role == "system":
                if isinstance(content, str) and content.strip():
                    system_chunks.append(content)
                continue

            parts = (
                content if isinstance(content, list) else [types.Part.from_text(text=str(content))]
            )
            contents.append(
                types.Content(role="model" if role == "assistant" else "user", parts=parts)
            )

        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="")]))

        system_prompt = "\n\n".join(system_chunks)
        return contents, (system_prompt or None)

    def _send(self, client: Any, messages: List[Dict[str, Any]]) -> Any:
        """Send a generate-content request.

        Parameters
        ----------
        client : Any
            The ``genai.Client`` returned by :meth:`connect`.
        messages : List[Dict[str, Any]]
            Messages in the shared chat format.

        Returns
        -------
        Any
            The raw generate-content response.
        """
        contents, system_prompt = self._to_contents(messages)

        config_kwargs: Dict[str, Any] = dict(self.request_params())
        if system_prompt is not None:
            config_kwargs["system_instruction"] = system_prompt

        return client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )

    def _extract_text(self, response: Any) -> str:
        """Extract the answer text from a Gemini response.

        Parameters
        ----------
        response : Any
            Raw generate-content response.

        Returns
        -------
        str
            The response ``text`` when present, otherwise the text parts of every candidate
            joined together.
        """
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        chunks: List[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []) if content else []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)

        return "".join(chunks).strip()

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage from the response metadata.

        Parameters
        ----------
        response : Any
            Raw generate-content response.

        Returns
        -------
        Dict[str, int]
            Keys ``prompt_tokens`` and ``completion_tokens``; zeros when the response carries no
            usage metadata.
        """
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        return {
            "prompt_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(metadata, "candidates_token_count", 0) or 0),
        }


LLM = LLMGemini
