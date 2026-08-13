# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Hugging Face LLM wrapper with optional quantization and config loading."""

import os
import torch
import yaml
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from llm_base import BaseLLM, logger
except Exception:
    try:
        from ..llm_base import BaseLLM, logger
    except Exception:
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger

NOT_SET = object()

class LLMHuggingFace(BaseLLM):
    """Text-only Hugging Face LLM backend."""

    def __init__(
        self,
        llm_config_file: str,
        device: Optional[str] = None,
        quantize: Optional[int] = None,
        cache_dir: Optional[str] = None,
        examples_yaml_file: Optional[Iterable[str]] = None,
        device_map: Optional[Any] = None,
    ) -> None:
        """Create a Hugging Face chatbot instance.

        Args:
            llm_config_file (str): Path to YAML config.
            device (Optional[str]): Optional device override.
            quantize (Optional[int]): Optional quantization override (8, 4, or 0).
            cache_dir (Optional[str]): Optional cache dir override.
            examples_yaml_file (Optional[Iterable[str]]): Optional example files for few-shot.
            device_map (Optional[Any]): Optional device_map override for model sharding.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required keys are missing or invalid.
        """
        logger.debug(f"Initializing LLMHuggingFace with config: {llm_config_file}")
        config = _load_yaml_config(llm_config_file)
        model_name = config.get("MODEL_NAME") or config.get("LLM_VERSION")
        if model_name is None:
            raise ValueError("Missing MODEL_NAME in config: {}".format(llm_config_file))

        config_device = config.get("DEVICE")
        config_quantize = config.get("QUANTIZE")
        if config_quantize is None and "QUANTIZE_8BIT" in config:
            config_quantize = 8 if config.get("QUANTIZE_8BIT") else 0
        config_device_map = config.get("DEVICE_MAP")
        config_multi_gpu = config.get("MULTI_GPU")
        config_cache_dir = config.get("CACHE_DIR")
        llm_config = config.get("LLM_CONFIG")
        if llm_config is None:
            llm_config = {}
        if not isinstance(llm_config, dict):
            raise ValueError("LLM_CONFIG must be a dict when provided.")
        llm_config = dict(llm_config)

        # Accept system-role rewrite flag from both top-level and LLM_CONFIG scopes.
        llm_disable_system = llm_config.pop("disable_system", None)
        if llm_disable_system is None:
            llm_disable_system = llm_config.pop("DISABLE_SYSTEM", None)

        resolved_disable_system = config.get("DISABLE_SYSTEM")
        if resolved_disable_system is None:
            resolved_disable_system = config.get("disable_system")
        if resolved_disable_system is None:
            resolved_disable_system = llm_disable_system

        # Accept thinking-mode flag from both legacy top-level and LLM_CONFIG scopes.
        llm_enable_thinking = llm_config.pop("enable_thinking", None)
        if llm_enable_thinking is None:
            llm_enable_thinking = llm_config.pop("ENABLE_THINKING", None)

        resolved_enable_thinking = config.get("ENABLE_THINKING")
        if resolved_enable_thinking is None:
            resolved_enable_thinking = config.get("enable_thinking")
        if resolved_enable_thinking is None:
            resolved_enable_thinking = llm_enable_thinking

        logger.debug(f"Resolved configuration ENABLE_THINKING: {resolved_enable_thinking}")

        resolved_device = device if device is not None else config_device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_quantize = (
            quantize if quantize is not None else (config_quantize if config_quantize is not None else 8)
        )
        if resolved_quantize not in (0, 4, 8):
            raise ValueError("Invalid QUANTIZE value: {} (expected 0, 4, or 8)".format(resolved_quantize))
        resolved_device_map = device_map if device_map is not None else config_device_map
        if isinstance(resolved_device_map, str) and not resolved_device_map.strip():
            resolved_device_map = None
        if resolved_device_map is not None and not isinstance(resolved_device_map, (str, dict)):
            raise ValueError(
                "Invalid DEVICE_MAP value: {} (expected string, dict, or null).".format(resolved_device_map)
            )
        resolved_multi_gpu = _coerce_bool(config_multi_gpu, default=False)
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
        resolved_cache_dir = cache_dir if cache_dir is not None else config_cache_dir
        resolved_examples = examples_yaml_file if examples_yaml_file is not None else [""]

        self.device = resolved_device
        self.model_name = model_name
        self.quantize = resolved_quantize
        self.device_map = resolved_device_map
        self.multi_gpu = resolved_multi_gpu
        self.enable_thinking = _coerce_bool(resolved_enable_thinking, default=NOT_SET)
        self.disable_system = _coerce_bool(resolved_disable_system, default=False)

        logger.debug(f"Final resolved enable_thinking value: {self.enable_thinking}")

        default_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.cache_dir = resolved_cache_dir if resolved_cache_dir is not None else default_cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self._warned_missing_chat_template = False
        self._warned_missing_enable_thinking = False

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
                model_kwargs["device_map"] = self.device_map if self.device_map is not None else "auto"
                model_kwargs["dtype"] = torch.float16
                quantized = True
        if not quantized and self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            model_kwargs["low_cpu_mem_usage"] = True

        try:
            logger.debug("Loading model '%s' with kwargs: %s", model_name, model_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir, **model_kwargs)
        except Exception as error:
            is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
            can_fallback = _is_cuda_device(self.device) and self.quantize == 8 and BitsAndBytesConfig is not None
            if is_oom and can_fallback:
                logger.warning("8-bit load failed with OOM; retrying with 4-bit quantization.")
                torch.cuda.empty_cache()
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = self.device_map if self.device_map is not None else "auto"
                model_kwargs["dtype"] = torch.float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    **model_kwargs,
                )
                quantized = True
            else:
                raise
        if not quantized and "device_map" not in model_kwargs:
            logger.debug("Model loaded without quantization; moving to device '%s'.", self.device)
            self.model = self.model.to(self.device)
        self.model_input_device = _resolve_model_input_device(self.model, fallback=self.device)
        super().__init__(examples_yaml_file=resolved_examples, llm_config=llm_config)

    @classmethod
    def from_config(
        cls,
        llm_config_file: str,
        device: Optional[str] = None,
        quantize: Optional[int] = None,
        cache_dir: Optional[str] = None,
        examples_yaml_file: Optional[Iterable[str]] = None,
        device_map: Optional[Any] = None,
    ) -> "LLMHuggingFace":
        """Create an instance from a YAML config file.

        Args:
            llm_config_file (str): Path to YAML config.
            device (Optional[str]): Optional device override.
            quantize (Optional[int]): Optional quantization override (8, 4, or 0).
            cache_dir (Optional[str]): Optional cache dir override.
            examples_yaml_file (Optional[Iterable[str]]): Optional examples override.
            device_map (Optional[Any]): Optional device_map override for model sharding.

        Returns:
            LLMHuggingFace: Configured chatbot instance.
        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required keys are missing from the config.
        """
        return cls(
            llm_config_file=llm_config_file,
            device=device,
            quantize=quantize,
            device_map=device_map,
            cache_dir=cache_dir,
            examples_yaml_file=examples_yaml_file,
        )

    def format_prompt(self, examples: List[Dict[str, str]], query: str) -> str:
        """Format a simple Q/A few-shot prompt.

        Args:
            examples (List[Dict[str, str]]): Few-shot examples with "question"/"answer".
            query (str): User question to append.

        Returns:
            str: Rendered prompt string.
        """
        prompt = ""
        for example in examples:
            prompt += "Q: {}\nA: {}\n\n".format(example["question"], example["answer"])
        prompt += "Q: {}\nA:".format(query)
        return prompt

    def generate_response(self, examples: List[Dict[str, str]], query: str, max_length: int = 512) -> str:
        """Generate a response from Q/A few-shot examples.

        Args:
            examples (List[Dict[str, str]]): Few-shot examples with "question"/"answer".
            query (str): User question to answer.
            max_length (int): Maximum number of generated tokens.

        Returns:
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

        Returns:
            str: Prompt string for generation.
        """
        normalized = self._prepare_messages(messages)

        if hasattr(self.tokenizer, "apply_chat_template"):
            # Check if this tokenizer supports enable_thinking (Qwen3 family)
            import inspect
            template_sig = inspect.signature(self.tokenizer.apply_chat_template)
            extra_kwargs = {}
            if self.enable_thinking is not NOT_SET:
                extra_kwargs["enable_thinking"] = self.enable_thinking
                logger.debug(f"Passing enable_thinking={self.enable_thinking} to tokenizer template.")
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

        Returns:
            str: Trimmed text.
        """
        stop_sequences = []
        if isinstance(self.stop, (list, tuple)):
            stop_sequences.extend([s for s in self.stop if s])
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

        Returns:
            Tuple[str, int]: Generated text and token count.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_input_device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.use_cache is not None:
            generation_kwargs["use_cache"] = self.use_cache

        if self.temperature is not None and self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
            if self.top_p is not None and self.top_p > 0:
                generation_kwargs["top_p"] = self.top_p
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        response = self._truncate_at_stop_sequences(response)
        return response, int(generated_tokens.shape[-1])

    def _connect(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a completion for the given messages.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            Dict[str, Any]: Response payload with text and token count.
        """
        prompt = self._messages_to_prompt(messages)
        prompt_tokens = 0
        try:
            prompt_tokens = int(len(self.tokenizer.encode(prompt, add_special_tokens=False)))
        except Exception:
            prompt_tokens = 0
        response, completion_tokens = self._generate_from_prompt(prompt, max_new_tokens=self.max_tokens)
        return {
            "content": response,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate prompt tokens locally using the model tokenizer."""
        prompt = self._messages_to_prompt(messages)
        return int(len(self.tokenizer.encode(prompt, add_special_tokens=False)))

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract the text content from the response dict.

        Args:
            response (Dict[str, Any]): Backend response payload.

        Returns:
            str: Assistant response text.
        """
        text = response["content"]
        # Strip <think>...</think> blocks produced by reasoning models
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # If model stopped mid-think (no closing tag), keep only what's after
        if "<think>" in text:
            parts = text.split("<think>", 1)
            text = parts[0].strip()
        return text

    def _extract_completion_tokens(self, response: Dict[str, Any]) -> int:
        """Extract completion token count from the response dict.

        Args:
            response (Dict[str, Any]): Backend response payload.

        Returns:
            int: Completion token count.
        """
        return response["completion_tokens"]

    def _extract_prompt_tokens(self, response: Dict[str, Any]) -> int:
        """Extract prompt token count from the response dict.

        Args:
            response (Dict[str, Any]): Backend response payload.

        Returns:
            int: Prompt token count.
        """
        return response.get("prompt_tokens", 0)
    
    def close(self) -> None:
        """Release model/tokenizer references and free CUDA cache."""
        import gc

        logger.debug("Closing LLMHuggingFace instance and freeing resources.")
        logger.debug(f"Amount of memory used before cleanup: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        try:
            if hasattr(self, "model"):
                del self.model
        except Exception:
            pass

        try:
            if hasattr(self, "tokenizer"):
                del self.tokenizer
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
        logger.debug(f"Amount of memory used after cleanup: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")



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
        if any(str(mapped_device).strip().lower() == "cpu" for mapped_device in device_map.values()):
            return "cpu"
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return fallback


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        config_file (str): Path to YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file) or {}


def _list_hf_configs(conf_dir: str) -> List[str]:
    """Return sorted Hugging Face config paths from a directory.

    Args:
        conf_dir (str): Config directory path.

    Returns:
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

    Returns:
        str: Selected config file path.

    Raises:
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




