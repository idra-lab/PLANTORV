# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Hugging Face LLM wrapper with optional quantization and config loading."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from llm_base import BaseLLM, logger, normalize_messages
except Exception:
    try:
        from ..llm_base import BaseLLM, logger, normalize_messages
    except Exception:
        import os
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger, normalize_messages

NOT_SET = object()


class LLMHuggingFace(BaseLLM):
    """Text-only Hugging Face backend, running the model in-process.

    The weights are loaded on the first :meth:`query` (through :meth:`connect`), so building the
    instance from a configuration file stays cheap.

    Configuration keys:
        MODEL_NAME: Hugging Face model identifier.
        DEVICE: Device to run on. Defaults to CUDA when available.
        QUANTIZE: 8, 4, or 0 (no quantization). Defaults to 8.
        DEVICE_MAP / MULTI_GPU: Model sharding across GPUs.
        CACHE_DIR: Where weights are downloaded.
        DISABLE_SYSTEM: Rewrite system messages for templates without a system role.
        ENABLE_THINKING: Passed to chat templates that support it (Qwen3 family).
        LLM_CONFIG: Generation parameters (``max_tokens``, ``temperature``, ``top_p``, ``stop``,
            ``use_cache``), consumed locally rather than sent to a server.
    """

    PROVIDER = "huggingface"
    SUPPORTS_IMAGES = False

    DEFAULT_PARAMS = {"max_tokens": 4096, "temperature": 0.0}
    # Every parameter is consumed locally by generate(); nothing is forwarded to a server.
    NON_REQUEST_PARAMS = (
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "use_cache",
        "seed",
        "enable_thinking",
        "frequency_penalty",
        "presence_penalty",
    )

    def __init__(
        self,
        model: str,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[Union[str, Path]] = None,
        examples: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        quantize: Optional[int] = None,
        cache_dir: Optional[str] = None,
        device_map: Optional[Any] = None,
    ) -> None:
        """Create a Hugging Face backend.

        Args:
            model (str): Hugging Face model identifier.
            params (Optional[Dict[str, Any]]): Generation parameters (``LLM_CONFIG``).
            config (Optional[Dict[str, Any]]): Full configuration dictionary.
            config_file (Optional[Union[str, Path]]): Path the configuration came from.
            examples (Optional[Union[str, Path]]): Folder with few-shot examples.
            device (Optional[str]): Device override.
            quantize (Optional[int]): Quantization override (8, 4, or 0).
            cache_dir (Optional[str]): Cache directory override.
            device_map (Optional[Any]): ``device_map`` override for model sharding.
        """
        self._device_override = device
        self._quantize_override = quantize
        self._cache_dir_override = cache_dir
        self._device_map_override = device_map

        self.tokenizer: Any = None
        self.model_input_device: Optional[str] = None
        self._warned_missing_chat_template = False
        self._warned_missing_enable_thinking = False

        super().__init__(
            model=model,
            params=params,
            config=config,
            config_file=config_file,
            examples=examples,
        )

    def _setup(self) -> None:
        """Resolve device, quantization and sharding settings from the configuration."""
        config = self.config
        self.model_name = self.model

        config_quantize = config.get("QUANTIZE")
        if config_quantize is None and "QUANTIZE_8BIT" in config:
            config_quantize = 8 if config.get("QUANTIZE_8BIT") else 0

        # The thinking-mode flag is accepted both at the top level and inside LLM_CONFIG.
        resolved_enable_thinking = config.get("ENABLE_THINKING")
        if resolved_enable_thinking is None:
            resolved_enable_thinking = config.get("enable_thinking")
        if resolved_enable_thinking is None:
            resolved_enable_thinking = self.param("enable_thinking")
        logger.debug(f"Resolved configuration ENABLE_THINKING: {resolved_enable_thinking}")

        resolved_device = (
            self._device_override if self._device_override is not None else config.get("DEVICE")
        )
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        resolved_quantize = (
            self._quantize_override
            if self._quantize_override is not None
            else (config_quantize if config_quantize is not None else 8)
        )
        if resolved_quantize not in (0, 4, 8):
            raise ValueError(
                "Invalid QUANTIZE value: {} (expected 0, 4, or 8)".format(resolved_quantize)
            )

        resolved_device_map = (
            self._device_map_override
            if self._device_map_override is not None
            else config.get("DEVICE_MAP")
        )
        if isinstance(resolved_device_map, str) and not resolved_device_map.strip():
            resolved_device_map = None
        if resolved_device_map is not None and not isinstance(resolved_device_map, (str, dict)):
            raise ValueError(
                "Invalid DEVICE_MAP value: {} (expected string, dict, or null).".format(
                    resolved_device_map
                )
            )

        resolved_multi_gpu = _coerce_bool(config.get("MULTI_GPU"), default=False)
        if resolved_multi_gpu and resolved_device_map is None and _is_cuda_device(resolved_device):
            if torch.cuda.device_count() > 1:
                resolved_device_map = "auto"
                logger.debug(f"Using {torch.cuda.device_count()} GPUs")
            else:
                logger.warning("MULTI_GPU is enabled but only one CUDA device is visible.")
        if resolved_device_map is not None and not _is_cuda_device(resolved_device):
            logger.warning(
                "Ignoring DEVICE_MAP=%s because DEVICE is '%s' (expected a CUDA device).",
                resolved_device_map,
                resolved_device,
            )
            resolved_device_map = None

        self.device = resolved_device
        self.quantize = resolved_quantize
        self.device_map = resolved_device_map
        self.multi_gpu = resolved_multi_gpu
        self.enable_thinking = _coerce_bool(resolved_enable_thinking, default=NOT_SET)

        logger.debug(f"Final resolved enable_thinking value: {self.enable_thinking}")

        resolved_cache_dir = (
            self._cache_dir_override
            if self._cache_dir_override is not None
            else config.get("CACHE_DIR")
        )
        default_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.cache_dir = resolved_cache_dir if resolved_cache_dir is not None else default_cache_dir

    def _create_client(self) -> Any:
        """Load the tokenizer and the model.

        Returns
        -------
            Any: The loaded ``transformers`` model.

        Raises
        ------
            torch.OutOfMemoryError: If the model does not fit, and no fallback applies.
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        quantized = False
        if _is_cuda_device(self.device) and self.quantize in (4, 8):
            if BitsAndBytesConfig is None:
                logger.warning("bitsandbytes is not available; loading model without quantization.")
            else:
                if self.quantize == 8:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    # Allow CPU offload for 8-bit when GPU memory is insufficient.
                    if hasattr(bnb_config, "llm_int8_enable_fp32_cpu_offload"):
                        bnb_config.llm_int8_enable_fp32_cpu_offload = True
                    model_kwargs["quantization_config"] = bnb_config
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                model_kwargs["device_map"] = (
                    self.device_map if self.device_map is not None else "auto"
                )
                model_kwargs["dtype"] = torch.float16
                quantized = True
        if not quantized and self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            model_kwargs["low_cpu_mem_usage"] = True

        try:
            logger.debug("Loading model '%s' with kwargs: %s", self.model_name, model_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, **model_kwargs
            )
        except Exception as error:
            is_oom = (
                isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
            )
            can_fallback = (
                _is_cuda_device(self.device)
                and self.quantize == 8
                and BitsAndBytesConfig is not None
            )
            if is_oom and can_fallback:
                logger.warning("8-bit load failed with OOM; retrying with 4-bit quantization.")
                torch.cuda.empty_cache()
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = (
                    self.device_map if self.device_map is not None else "auto"
                )
                model_kwargs["dtype"] = torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    **model_kwargs,
                )
                quantized = True
            else:
                raise

        if not quantized and "device_map" not in model_kwargs:
            logger.debug("Model loaded without quantization; moving to device '%s'.", self.device)
            model = model.to(self.device)

        self.model_input_device = _resolve_model_input_device(model, fallback=self.device)
        return model

    def format_prompt(self, examples: List[Dict[str, str]], query: str) -> str:
        """Format a simple Q/A few-shot prompt.

        Args:
            examples (List[Dict[str, str]]): Few-shot examples with "question"/"answer".
            query (str): User question to append.

        Returns
        -------
            str: Rendered prompt string.
        """
        prompt = ""
        for example in examples:
            prompt += "Q: {}\nA: {}\n\n".format(example["question"], example["answer"])
        prompt += "Q: {}\nA:".format(query)
        return prompt

    def generate_response(
        self, examples: List[Dict[str, str]], query: str, max_length: int = 512
    ) -> str:
        """Generate a response from Q/A few-shot examples.

        Args:
            examples (List[Dict[str, str]]): Few-shot examples with "question"/"answer".
            query (str): User question to answer.
            max_length (int): Maximum number of generated tokens.

        Returns
        -------
            str: Assistant response.
        """
        prompt = self.format_prompt(examples, query)
        response, _ = self._generate_from_prompt(prompt, max_new_tokens=max_length)
        if "A:" in response:
            return response.split("A:")[-1].strip()
        return response.strip()

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert structured messages to a model prompt.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns
        -------
            str: Prompt string for generation.
        """
        normalized = normalize_messages(messages, disable_system=self.disable_system)

        if hasattr(self.tokenizer, "apply_chat_template"):
            # Some tokenizers (Qwen3 family) accept enable_thinking in their chat template.
            extra_kwargs = {}
            if self.enable_thinking is not NOT_SET:
                extra_kwargs["enable_thinking"] = self.enable_thinking
                logger.debug(
                    f"Passing enable_thinking={self.enable_thinking} to tokenizer template."
                )
            try:
                return self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
            except ValueError as error:
                message = str(error).lower()
                missing_template = (
                    "chat_template is not set" in message
                    or "no template argument was passed" in message
                )
                if not missing_template:
                    raise
                if not self._warned_missing_chat_template:
                    logger.warning(
                        "Tokenizer chat template is not set; using generic role-based prompt formatting."
                    )
                    self._warned_missing_chat_template = True

        lines = []
        for msg in normalized:
            role = msg["role"].strip().upper()
            lines.append("{}: {}".format(role, msg["content"]))
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _truncate_at_stop_sequences(self, text: str) -> str:
        """Trim model output at configured stop sequences.

        Args:
            text (str): Raw generated text.

        Returns
        -------
            str: Trimmed text.
        """
        stop = self.param("stop")
        stop_sequences = []
        if isinstance(stop, (list, tuple)):
            stop_sequences.extend([s for s in stop if s])
        if not stop_sequences:
            stop_sequences = ["USER:", "SYSTEM:"]

        # Always add these for reasoning models
        # stop_sequences += ["```\n\n", "\nQ:", "\nUSER:"]

        earliest = None
        for seq in stop_sequences:
            idx = text.find(seq)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx

        if earliest is None:
            return text.strip()
        return text[:earliest].strip()

    def _generate_from_prompt(self, prompt: str, max_new_tokens: int) -> Tuple[str, int]:
        """Run generation from a raw prompt string.

        Args:
            prompt (str): Prompt text.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns
        -------
            Tuple[str, int]: Generated text and token count.
        """
        model = self.connect()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_input_device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        use_cache = self.param("use_cache")
        if use_cache is not None:
            generation_kwargs["use_cache"] = use_cache

        temperature = self.param("temperature")
        top_p = self.param("top_p")
        if temperature is not None and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if top_p is not None and top_p > 0:
                generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        response = self._truncate_at_stop_sequences(response)
        return response, int(generated_tokens.shape[-1])

    def _send(self, client: Any, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a completion for the given messages.

        Args:
            client (Any): The loaded model, from :meth:`connect`.
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns
        -------
            Dict[str, Any]: Response payload with text and token counts.
        """
        prompt = self._messages_to_prompt(messages)
        try:
            prompt_tokens = int(len(self.tokenizer.encode(prompt, add_special_tokens=False)))
        except Exception:
            prompt_tokens = 0

        response, completion_tokens = self._generate_from_prompt(
            prompt, max_new_tokens=self.param("max_tokens", 4096)
        )
        return {
            "content": response,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count prompt tokens locally using the model tokenizer."""
        self.connect()
        prompt = self._messages_to_prompt(messages)
        return int(len(self.tokenizer.encode(prompt, add_special_tokens=False)))

    def _extract_text(self, response: Dict[str, Any]) -> str:
        """Extract the text content from the response dict.

        Args:
            response (Dict[str, Any]): Backend response payload.

        Returns
        -------
            str: Assistant response text.
        """
        text = response["content"]
        # Strip <think>...</think> blocks produced by reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # If model stopped mid-think (no closing tag), keep only what's before
        if "<think>" in text:
            text = text.split("<think>", 1)[0].strip()
        return text

    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token counts from the response dict."""
        return {
            "prompt_tokens": int(response.get("prompt_tokens", 0)),
            "completion_tokens": int(response.get("completion_tokens", 0)),
        }

    def close(self) -> None:
        """Release model/tokenizer references and free CUDA cache."""
        import gc

        logger.debug("Closing LLMHuggingFace instance and freeing resources.")
        if torch.cuda.is_available():
            logger.debug(
                f"Memory used before cleanup: {torch.cuda.memory_allocated() / (1024**2):.2f} MB"
            )

        try:
            self._client = None
        except Exception:
            pass

        try:
            self.tokenizer = None
        except Exception:
            pass

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        logger.debug("LLMHuggingFace instance closed and resources freed.")
        if torch.cuda.is_available():
            logger.debug(
                f"Memory used after cleanup: {torch.cuda.memory_allocated() / (1024**2):.2f} MB"
            )


def _is_cuda_device(device: Optional[str]) -> bool:
    """Return True when a device string points to CUDA."""
    return str(device).strip().lower().startswith("cuda")


def _coerce_bool(value: Any, default: Union[bool, object] = False) -> Union[bool, object]:
    """Parse loose boolean config values (e.g., true/false, yes/no, 1/0)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("Invalid boolean value: '{}'.".format(value))
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError("Invalid boolean value type: {}.".format(type(value).__name__))


def _resolve_model_input_device(model: Any, fallback: str) -> str:
    """Infer the device where prompt tensors should be placed before generation."""
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for mapped_device in device_map.values():
            if isinstance(mapped_device, torch.device):
                if mapped_device.type == "cuda":
                    return str(mapped_device)
                continue
            if isinstance(mapped_device, int):
                return "cuda:{}".format(mapped_device)
            if isinstance(mapped_device, str) and mapped_device.strip().lower().startswith("cuda"):
                return mapped_device
        if any(
            str(mapped_device).strip().lower() == "cpu" for mapped_device in device_map.values()
        ):
            return "cpu"
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return fallback


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        config_file (str): Path to YAML config file.

    Returns
    -------
        Dict[str, Any]: Parsed configuration.

    Raises
    ------
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file) or {}


def _list_hf_configs(conf_dir: str) -> List[str]:
    """Return sorted Hugging Face config paths from a directory.

    Args:
        conf_dir (str): Config directory path.

    Returns
    -------
        List[str]: Sorted list of config paths.
    """
    if not os.path.isdir(conf_dir):
        return []
    configs = []
    for name in os.listdir(conf_dir):
        if name.startswith("hf_") and name.endswith(".yaml"):
            configs.append(os.path.join(conf_dir, name))
    return sorted(configs)


def _select_config(conf_dir: str) -> str:
    """Prompt the user to select a Hugging Face config file.

    Args:
        conf_dir (str): Config directory path.

    Returns
    -------
        str: Selected config file path.

    Raises
    ------
        FileNotFoundError: If no matching config files are found.
    """
    configs = _list_hf_configs(conf_dir)
    if not configs:
        raise FileNotFoundError("No HuggingFace config files found in {}".format(conf_dir))

    display = []
    for path in configs:
        data = _load_yaml_config(path)
        model_name = data.get("MODEL_NAME") or data.get("LLM_VERSION") or "Unknown"
        quantize = data.get("QUANTIZE")
        if quantize is None and "QUANTIZE_8BIT" in data:
            quantize = 8 if data.get("QUANTIZE_8BIT") else 0
        display.append((path, model_name, quantize))

    logger.info("Available HuggingFace configs:")
    for idx, (_, model_name, quantize) in enumerate(display, start=1):
        if quantize == 8:
            quantize_label = "8-bit"
        elif quantize == 4:
            quantize_label = "4-bit"
        else:
            quantize_label = "full"
        logger.info("%d) %s (%s)", idx, model_name, quantize_label)

    while True:
        choice = input("Select model [1-{}]: ".format(len(display))).strip()
        if not choice:
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(display):
                return display[index - 1][0]
        logger.info("Invalid selection. Please enter a number between 1 and %d.", len(display))


LLM = LLMHuggingFace


if __name__ == "__main__":
    conf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))
    config_path = _select_config(conf_dir)
    chatbot = LLMHuggingFace.from_config(config_path)
    ok, response = chatbot.query("Hi, who are you?")
    logger.info("Response: %s", response)
