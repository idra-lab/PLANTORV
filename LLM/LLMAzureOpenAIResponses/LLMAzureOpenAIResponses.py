"""Azure OpenAI backend speaking the Responses API."""

import os
from typing import Any, Dict, List, Optional, Sequence, cast

from openai import AzureOpenAI

try:
    from llm_base import ImageInput, image_data_url, logger
except Exception:
    try:
        from ..llm_base import ImageInput, image_data_url, logger
    except Exception:
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import ImageInput, image_data_url, logger

try:
    from LLMAzureOpenAI.LLMAzureOpenAI import LLMAzureOpenAI
except Exception:
    try:
        from ..LLMAzureOpenAI.LLMAzureOpenAI import LLMAzureOpenAI
    except Exception:
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from LLMAzureOpenAI.LLMAzureOpenAI import LLMAzureOpenAI


class LLMAzureOpenAIResponses(LLMAzureOpenAI):
    """Azure OpenAI backend for deployments served through the Responses API.

    The reasoning-heavy deployments (the ``-pro`` tier, and the ``o*-pro`` models before it) do not
    expose ``/chat/completions`` at all: every request to it comes back as
    ``400 The requested operation is unsupported``. They are only reachable through
    ``/responses``, which differs from chat completions in more than the URL:

    * ``max_completion_tokens`` is named ``max_output_tokens``, and ``seed`` does not exist.
    * Content parts are ``input_text`` / ``input_image``, and ``input_image`` carries the URL
      directly rather than nesting it under an ``image_url`` object.
    * The answer is one flat ``output_text``, and usage is reported as ``input_tokens`` /
      ``output_tokens``.

    Connection settings are read by :class:`~LLMAzureOpenAI.LLMAzureOpenAI.LLMAzureOpenAI` and are
    unchanged, except that ``API_VERSION`` must be one that serves the Responses API
    (``2025-04-01-preview`` or later). ``LLM_CONFIG`` is still forwarded as-is, so it has to use the
    Responses names.
    """

    PROVIDER = "azure_openai_responses"
    SUPPORTS_IMAGES = True

    def _send(self, client: AzureOpenAI, messages: List[Dict[str, Any]]) -> Any:
        """Send a request to the Responses API.

        Parameters
        ----------
        client : AzureOpenAI
            The SDK client returned by :meth:`connect`.
        messages : List[Dict[str, Any]]
            Messages in the shared chat format, with Responses-flavoured content parts.

        Returns
        -------
        Any
            The raw response object.
        """
        return client.responses.create(
            model=self.deployment,
            input=cast(Any, messages),
            **self.request_params(),
        )

    def _extract_text(self, response: Any) -> str:
        """Extract the answer from a Responses API result.

        Parameters
        ----------
        response : Any
            Raw response object.

        Returns
        -------
        str
            The assistant answer, or the empty string when the response carries no text. A
            response that stopped on the token cap before emitting any text is reported, since it
            is otherwise indistinguishable from an empty answer.
        """
        text = getattr(response, "output_text", None)
        if text:
            return text

        # output_text is a convenience field; fall back to walking the output items for SDKs or
        # responses that do not provide it.
        chunks = []
        for item in getattr(response, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                chunk = getattr(part, "text", None)
                if chunk:
                    chunks.append(chunk)

        if not chunks and getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            logger.warning(
                "The model returned no text: the response is incomplete (%s). Reasoning models "
                "spend the budget on reasoning tokens first, so max_output_tokens has to leave "
                "room for the answer.",
                reason or "no reason given",
            )

        return "".join(chunks)

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage, which the Responses API names differently.

        Parameters
        ----------
        response : Any
            Raw response object.

        Returns
        -------
        Dict[str, int]
            Keys ``prompt_tokens`` and ``completion_tokens``, mapped from ``input_tokens`` and
            ``output_tokens``; zeros when the response reports no usage.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        return {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }

    def build_content(
        self,
        message: str,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Any:
        """Build the content of a single message, using Responses content parts.

        Parameters
        ----------
        message : str
            The text of the message.
        images : Optional[Sequence[ImageInput]], optional
            Images to attach.

        Returns
        -------
        Any
            The plain string without images, and a list of ``input_text`` / ``input_image`` parts
            with them. The base class builds a ``text`` part instead, which the Responses API
            rejects.
        """
        if not images:
            return message

        parts: List[Any] = [{"type": "input_text", "text": message}]
        parts.extend(self.image_part(image) for image in images)
        return parts

    def image_part(self, image: ImageInput) -> Dict[str, Any]:
        """Encode an image as a Responses ``input_image`` content part.

        Parameters
        ----------
        image : ImageInput
            A ``PIL.Image.Image``, or the path of an image file.

        Returns
        -------
        Dict[str, Any]
            An ``input_image`` part holding a ``data:`` URL and the configured detail level. Unlike
            the chat-completions part, the URL is a plain string field.
        """
        return {
            "type": "input_image",
            "image_url": image_data_url(image),
            "detail": self.image_detail,
        }


LLM = LLMAzureOpenAIResponses
