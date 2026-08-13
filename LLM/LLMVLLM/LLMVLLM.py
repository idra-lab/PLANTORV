# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""vLLM offline backend with process-local shared engine reuse."""

import atexit
import gc
import hashlib
import inspect
import json
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

import yaml

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams
except Exception as error:
    VLLMEngine = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = error
else:
    _VLLM_IMPORT_ERROR = None

try:
    from llm_base import BaseLLM, logger
except Exception:
    try:
        from ..llm_base import BaseLLM, logger
    except Exception:
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from llm_base import BaseLLM, logger

NOT_SET  = object()

class LLMVLLM(BaseLLM):
    """Offline vLLM backend implementing the shared BaseLLM interface.

    Model weights and KV runtime are kept in a process-level cache, so creating
    multiple instances with the same engine config does not reload the model.
    """

    _ENGINE_CACHE_LOCK = threading.Lock()
    _ENGINE_CACHE: Dict[str, Dict[str, Any]] = {}
    _ATEXIT_REGISTERED = False

    def __init__(
        self,
        llm_config_file: str,
        examples_yaml_file: Iterable[str] = ("",),
        download_dir: Optional[str] = None,
        download_workers: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        dtype: Optional[str] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        swap_space: Optional[float] = None,
        trust_remote_code: Optional[bool] = None,
        enable_prefix_caching: Optional[bool] = None,
        disable_log_stats: Optional[bool] = None,
        enforce_eager: Optional[bool] = None,
    ) -> None:
        """Initialize an in-process vLLM backend from YAML config."""
        init_started_at = time.perf_counter()
        if VLLMEngine is None or SamplingParams is None:
            raise ImportError(
                "vLLM is not available. Install it with `pip install vllm`."
            ) from _VLLM_IMPORT_ERROR

        logger.info("LLM configuration file: %s", llm_config_file)
        if not llm_config_file.endswith((".yaml", ".yml")) or not os.path.isfile(llm_config_file):
            raise FileNotFoundError(
                "The selected file {} does not exist or is not a yaml file".format(llm_config_file)
            )

        llm_connection_config = _load_yaml_config(llm_config_file)
        logger.debug(
            "Loaded vLLM config with keys: %s",
            sorted(llm_connection_config.keys()),
        )

        self.engine = (
            llm_connection_config.get("MODEL_NAME")
            or llm_connection_config.get("LLM_VERSION")
            or llm_connection_config.get("MODEL")
        )
        if not self.engine:
            raise ValueError("Missing model name in config. Expected MODEL_NAME (or LLM_VERSION/MODEL).")

        llm_base_config = llm_connection_config.get("LLM_CONFIG", {})
        if not isinstance(llm_base_config, dict):
            raise ValueError("LLM_CONFIG must be a dict when provided.")
        llm_base_config = dict(llm_base_config)

        llm_disable_system = llm_base_config.pop("disable_system", None)
        if llm_disable_system is None:
            llm_disable_system = llm_base_config.pop("DISABLE_SYSTEM", None)

        resolved_disable_system = llm_connection_config.get("DISABLE_SYSTEM")
        if resolved_disable_system is None:
            resolved_disable_system = llm_connection_config.get("disable_system")
        if resolved_disable_system is None:
            resolved_disable_system = llm_disable_system

        resolved_enable_thinking = llm_base_config.pop("enable_thinking", NOT_SET)
        if resolved_enable_thinking is NOT_SET:
            resolved_enable_thinking = llm_base_config.pop("ENABLE_THINKING", NOT_SET)
        if resolved_enable_thinking is NOT_SET:
            self.enable_thinking = NOT_SET
        else:
            self.enable_thinking = _coerce_bool(resolved_enable_thinking)
        self.disable_system = _coerce_bool(resolved_disable_system, default=False)

        config_download_dir = (
            llm_connection_config.get("DOWNLOAD_DIR")
            or llm_connection_config.get("MODEL_DOWNLOAD_DIR")
            or llm_connection_config.get("CACHE_DIR")
        )
        config_download_workers = (
            llm_connection_config.get("DOWNLOAD_WORKERS")
            if llm_connection_config.get("DOWNLOAD_WORKERS") is not None
            else llm_connection_config.get("MODEL_DOWNLOAD_WORKERS")
        )
        if config_download_workers is None:
            config_download_workers = llm_connection_config.get("HF_DOWNLOAD_WORKERS")

        resolved_download_dir = (
            download_dir if download_dir is not None else config_download_dir
        )
        resolved_download_dir = _resolve_optional_path(
            resolved_download_dir,
            base_dir=os.path.dirname(os.path.abspath(llm_config_file)),
        )
        if resolved_download_dir is not None:
            os.makedirs(resolved_download_dir, exist_ok=True)

        resolved_download_workers = _coerce_optional_int(
            download_workers if download_workers is not None else config_download_workers,
            field_name="DOWNLOAD_WORKERS",
            minimum=1,
        )

        logger.debug(
            "Resolved download settings: download_dir=%s download_workers=%s model_is_local=%s",
            resolved_download_dir,
            resolved_download_workers,
            _looks_like_local_path(self.engine),
        )

        if resolved_download_workers is not None and not _looks_like_local_path(self.engine):
            download_started_at = time.perf_counter()
            self.engine = _download_model_snapshot(
                model_name=self.engine,
                download_dir=resolved_download_dir,
                download_workers=resolved_download_workers,
            )
            logger.debug(
                "Model prefetch completed in %.2f s; resolved model path=%s",
                _elapsed_s(download_started_at),
                self.engine,
            )

        self.download_dir = resolved_download_dir
        self.download_workers = resolved_download_workers

        resolved_tensor_parallel_size = _coerce_positive_int(
            tensor_parallel_size if tensor_parallel_size is not None else llm_connection_config.get("TENSOR_PARALLEL_SIZE"),
            default=1,
            field_name="TENSOR_PARALLEL_SIZE",
        )
        resolved_dtype = dtype if dtype is not None else llm_connection_config.get("DTYPE", "auto")
        resolved_max_model_len = _coerce_optional_int(
            max_model_len if max_model_len is not None else llm_connection_config.get("MAX_MODEL_LEN"),
            field_name="MAX_MODEL_LEN",
            minimum=1,
        )
        resolved_max_num_seqs = _coerce_optional_int(
            max_num_seqs if max_num_seqs is not None else llm_connection_config.get("MAX_NUM_SEQS"),
            field_name="MAX_NUM_SEQS",
            minimum=1,
        )
        resolved_max_num_batched_tokens = _coerce_optional_int(
            max_num_batched_tokens
            if max_num_batched_tokens is not None
            else llm_connection_config.get("MAX_NUM_BATCHED_TOKENS"),
            field_name="MAX_NUM_BATCHED_TOKENS",
            minimum=1,
        )
        resolved_gpu_memory_utilization = _coerce_optional_float(
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else llm_connection_config.get("GPU_MEMORY_UTILIZATION"),
            field_name="GPU_MEMORY_UTILIZATION",
            minimum=0.0,
            maximum=1.0,
        )
        resolved_swap_space = _coerce_optional_float(
            swap_space if swap_space is not None else llm_connection_config.get("SWAP_SPACE"),
            field_name="SWAP_SPACE",
            minimum=0.0,
        )
        resolved_trust_remote_code = _coerce_bool(
            trust_remote_code if trust_remote_code is not None else llm_connection_config.get("TRUST_REMOTE_CODE"),
            default=False,
        )
        resolved_enable_prefix_caching = _coerce_bool(
            enable_prefix_caching
            if enable_prefix_caching is not None
            else llm_connection_config.get("ENABLE_PREFIX_CACHING"),
            default=True,
        )
        resolved_disable_log_stats = _coerce_bool(
            disable_log_stats
            if disable_log_stats is not None
            else llm_connection_config.get("DISABLE_LOG_STATS"),
            default=True,
        )
        resolved_enforce_eager = _coerce_bool(
            enforce_eager if enforce_eager is not None else llm_connection_config.get("ENFORCE_EAGER"),
            default=False,
        )

        self.enable_prefix_caching = resolved_enable_prefix_caching

        self._engine_kwargs: Dict[str, Any] = {
            "model": self.engine,
            "tensor_parallel_size": resolved_tensor_parallel_size,
            "trust_remote_code": resolved_trust_remote_code,
            "enable_prefix_caching": resolved_enable_prefix_caching,
            "disable_log_stats": resolved_disable_log_stats,
            "enforce_eager": resolved_enforce_eager,
        }
        if (
            resolved_download_dir is not None
            and _vllm_supports_init_kwarg("download_dir")
        ):
            self._engine_kwargs["download_dir"] = resolved_download_dir
        if resolved_dtype not in [None, "", "None"]:
            self._engine_kwargs["dtype"] = resolved_dtype
        if resolved_max_model_len is not None:
            self._engine_kwargs["max_model_len"] = resolved_max_model_len
        if resolved_max_num_seqs is not None:
            self._engine_kwargs["max_num_seqs"] = resolved_max_num_seqs
        if resolved_max_num_batched_tokens is not None:
            self._engine_kwargs["max_num_batched_tokens"] = resolved_max_num_batched_tokens
        if resolved_gpu_memory_utilization is not None:
            self._engine_kwargs["gpu_memory_utilization"] = resolved_gpu_memory_utilization
        if resolved_swap_space is not None:
            self._engine_kwargs["swap_space"] = resolved_swap_space

        logger.debug("Resolved vLLM engine kwargs: %s", self._engine_kwargs)

        self._engine_key = _engine_key(self._engine_kwargs)
        logger.debug("Computed shared engine cache key: %s", self._engine_key)

        with LLMVLLM._ENGINE_CACHE_LOCK:
            cache_entry = LLMVLLM._ENGINE_CACHE.get(self._engine_key)
            if cache_entry is None:
                logger.debug("vLLM engine cache miss for key=%s", self._engine_key)
                logger.info(
                    "Loading vLLM engine for model '%s' (tensor_parallel_size=%s, prefix_caching=%s, download_dir=%s)",
                    self.engine,
                    self._engine_kwargs.get("tensor_parallel_size"),
                    self._engine_kwargs.get("enable_prefix_caching"),
                    self._engine_kwargs.get("download_dir"),
                )
                engine_load_started_at = time.perf_counter()
                llm = VLLMEngine(**self._engine_kwargs)
                cache_entry = {
                    "llm": llm,
                    "tokenizer": llm.get_tokenizer(),
                    "refs": 0,
                }
                LLMVLLM._ENGINE_CACHE[self._engine_key] = cache_entry
                logger.debug(
                    "Loaded new vLLM engine in %.2f s",
                    _elapsed_s(engine_load_started_at),
                )
            else:
                logger.debug(
                    "vLLM engine cache hit for key=%s current_refs=%s",
                    self._engine_key,
                    cache_entry.get("refs", 0),
                )
            cache_entry["refs"] = int(cache_entry.get("refs", 0)) + 1
            self._llm = cache_entry["llm"]
            self.tokenizer = cache_entry["tokenizer"]
            logger.debug(
                "Attached to shared engine key=%s refs_now=%s",
                self._engine_key,
                cache_entry.get("refs", 0),
            )

            if not LLMVLLM._ATEXIT_REGISTERED:
                atexit.register(LLMVLLM._shutdown_all_engines)
                LLMVLLM._ATEXIT_REGISTERED = True

        self._warned_missing_chat_template = False
        super().__init__(examples_yaml_file=examples_yaml_file, llm_config=llm_base_config)
        logger.debug(
            "LLMVLLM initialization completed in %.2f s (engine=%s)",
            _elapsed_s(init_started_at),
            self.engine,
        )

    @classmethod
    def _shutdown_all_engines(cls) -> None:
        """Release all shared vLLM engines on process shutdown."""
        with cls._ENGINE_CACHE_LOCK:
            entries = list(cls._ENGINE_CACHE.values())
            cls._ENGINE_CACHE.clear()

        for entry in entries:
            try:
                if "llm" in entry:
                    del entry["llm"]
            except Exception:
                pass
            try:
                if "tokenizer" in entry:
                    del entry["tokenizer"]
            except Exception:
                pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @classmethod
    def from_config(
        cls,
        llm_config_file: str,
        examples_yaml_file: Iterable[str] = ("",),
        download_dir: Optional[str] = None,
        download_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> "LLMVLLM":
        """Construct a backend instance from a YAML config path."""
        return cls(
            llm_config_file=llm_config_file,
            examples_yaml_file=examples_yaml_file,
            download_dir=download_dir,
            download_workers=download_workers,
            **kwargs,
        )

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert structured chat messages into a single prompt string."""
        started_at = time.perf_counter()
        normalized = self._prepare_messages(messages)

        logger.debug(
            "Formatting prompt from %s messages; roles=%s",
            len(normalized),
            [msg.get("role", "") for msg in normalized],
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            extra_kwargs = {}
            if self.enable_thinking is not NOT_SET:
                logger.debug(f"Passing enable_thinking={self.enable_thinking} to tokenizer template.")
                extra_kwargs["enable_thinking"] = self.enable_thinking

            try:
                prompt = self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
                logger.debug(
                    "Prompt formatted with tokenizer template in %.2f ms (chars=%s hash=%s)",
                    _elapsed_ms(started_at),
                    len(prompt),
                    _text_hash(prompt),
                )
                return prompt
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
            lines.append("{}: {}".format(msg["role"].upper(), msg["content"]))
        lines.append("ASSISTANT:")
        prompt = "\n".join(lines)
        logger.debug(
            "Prompt formatted with fallback template in %.2f ms (chars=%s hash=%s)",
            _elapsed_ms(started_at),
            len(prompt),
            _text_hash(prompt),
        )
        return prompt

    def _build_sampling_params(self) -> Any:
        """Build vLLM sampling params from shared generation settings."""
        started_at = time.perf_counter()
        max_tokens = self.max_tokens if self.max_tokens is not None else BaseLLM.llm_default_config["max_tokens"]
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = int(BaseLLM.llm_default_config["max_tokens"])

        temperature = 0.0
        if self.temperature is not None:
            try:
                temperature = float(self.temperature)
            except Exception:
                temperature = 0.0
        if temperature < 0:
            temperature = 0.0

        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.top_p is not None:
            try:
                top_p = float(self.top_p)
                if top_p > 0:
                    sampling_kwargs["top_p"] = top_p
            except Exception:
                pass

        if self.frequency_penalty is not None:
            try:
                sampling_kwargs["frequency_penalty"] = float(self.frequency_penalty)
            except Exception:
                pass

        if self.presence_penalty is not None:
            try:
                sampling_kwargs["presence_penalty"] = float(self.presence_penalty)
            except Exception:
                pass

        if self.seed is not None:
            try:
                sampling_kwargs["seed"] = int(self.seed)
            except Exception:
                pass

        stop_sequences = _normalize_stop_sequences(self.stop)
        if stop_sequences:
            sampling_kwargs["stop"] = stop_sequences

        logger.debug("Sampling kwargs before construction: %s", sampling_kwargs)

        try:
            params = SamplingParams(**sampling_kwargs)
            logger.debug("SamplingParams built in %.2f ms", _elapsed_ms(started_at))
            return params
        except TypeError:
            # Compatibility fallback for older vLLM versions that may not expose
            # all sampling fields (for example `seed`).
            sampling_kwargs.pop("seed", None)
            logger.debug(
                "SamplingParams seed unsupported by current vLLM; retrying without seed"
            )
            params = SamplingParams(**sampling_kwargs)
            logger.debug("SamplingParams fallback built in %.2f ms", _elapsed_ms(started_at))
            return params

    def _connect(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate one completion using the in-process vLLM engine."""
        connect_started_at = time.perf_counter()
        prompt = self._messages_to_prompt(messages)
        logger.debug(
            "Prepared generation prompt (chars=%s hash=%s)",
            len(prompt),
            _text_hash(prompt),
        )

        sampling_started_at = time.perf_counter()
        sampling_params = self._build_sampling_params()
        logger.debug("Sampling parameter build took %.2f ms", _elapsed_ms(sampling_started_at))

        generate_started_at = time.perf_counter()
        outputs = self._llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )
        logger.debug("vLLM generate call finished in %.2f s", _elapsed_s(generate_started_at))
        if not outputs:
            raise RuntimeError("vLLM returned no output.")

        request_output = outputs[0]
        if len(getattr(request_output, "outputs", [])) <= 0:
            raise RuntimeError("vLLM returned an empty completion.")

        generated = request_output.outputs[0]
        text = str(getattr(generated, "text", "")).strip()
        text = _truncate_at_stop_sequences(text, self.stop)

        completion_tokens = 0
        token_ids = getattr(generated, "token_ids", None)
        if token_ids is not None:
            try:
                completion_tokens = int(len(token_ids))
            except Exception:
                completion_tokens = 0

        prompt_tokens = 0
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            try:
                prompt_tokens = int(len(prompt_token_ids))
            except Exception:
                prompt_tokens = 0
        if prompt_tokens <= 0:
            prompt_tokens = self._count_prompt_tokens(prompt)

        cached_prompt_tokens = _extract_cached_prompt_tokens(request_output)

        logger.debug(
            "Generation completed in %.2f s (prompt_tokens=%s completion_tokens=%s cached_prompt_tokens=%s preview=%s)",
            _elapsed_s(connect_started_at),
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
            _preview_text(text),
        )

        return {
            "content": text,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
        }

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract assistant content from a normalized vLLM response payload."""
        text = str(response.get("content", ""))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text:
            text = text.split("<think>", 1)[0].strip()
        return text

    def _extract_completion_tokens(self, response: Dict[str, Any]) -> int:
        """Extract completion tokens from a normalized vLLM response payload."""
        try:
            return int(response.get("completion_tokens", 0))
        except Exception:
            return 0

    def _extract_prompt_tokens(self, response: Dict[str, Any]) -> int:
        """Extract prompt tokens from a normalized vLLM response payload."""
        try:
            return int(response.get("prompt_tokens", 0))
        except Exception:
            return 0

    def _estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate prompt tokens using the backend tokenizer for dry runs."""
        prompt = self._messages_to_prompt(messages)
        return self._count_prompt_tokens(prompt)

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Tokenize prompt text with the backend tokenizer."""
        started_at = time.perf_counter()
        try:
            count = int(len(self.tokenizer.encode(prompt, add_special_tokens=False)))
            logger.debug(
                "Prompt tokenized with add_special_tokens=False in %.2f ms (tokens=%s)",
                _elapsed_ms(started_at),
                count,
            )
            return count
        except Exception:
            pass

        try:
            # Some tokenizers do not expose add_special_tokens.
            count = int(len(self.tokenizer.encode(prompt)))
            logger.debug(
                "Prompt tokenized with fallback encode in %.2f ms (tokens=%s)",
                _elapsed_ms(started_at),
                count,
            )
            return count
        except Exception:
            logger.debug("Prompt tokenization failed after %.2f ms", _elapsed_ms(started_at))
            return 0

    def close(self) -> None:
        """Release this instance while keeping the shared engine warm.

        The shared engine is intentionally retained until process exit so multiple
        phases in a single script execution reuse loaded weights and prefix cache.
        """
        with LLMVLLM._ENGINE_CACHE_LOCK:
            cache_entry = LLMVLLM._ENGINE_CACHE.get(getattr(self, "_engine_key", ""))
            if cache_entry is None:
                return
            refs = int(cache_entry.get("refs", 0))
            cache_entry["refs"] = max(0, refs - 1)


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as file:
        return yaml.safe_load(file) or {}


def _resolve_optional_path(path_value: Any, base_dir: str) -> Optional[str]:
    if path_value in [None, "", "None"]:
        return None
    expanded = os.path.expanduser(str(path_value))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _looks_like_local_path(model_ref: str) -> bool:
    model_ref = str(model_ref).strip()
    if model_ref.startswith(("./", "../", "~/")):
        return True
    if os.path.isabs(model_ref):
        return True
    return os.path.exists(os.path.expanduser(model_ref))


def _download_model_snapshot(
    model_name: str,
    download_dir: Optional[str],
    download_workers: int,
) -> str:
    if snapshot_download is None:
        raise RuntimeError(
            "DOWNLOAD_WORKERS was set, but huggingface_hub is unavailable. "
            "Install it with `pip install huggingface_hub` or unset DOWNLOAD_WORKERS."
        )

    download_kwargs: Dict[str, Any] = {
        "repo_id": model_name,
        "max_workers": int(download_workers),
    }
    if download_dir not in [None, "", "None"]:
        download_kwargs["cache_dir"] = download_dir

    logger.info(
        "Prefetching model '%s' with max_workers=%s into cache_dir=%s",
        model_name,
        download_workers,
        download_dir,
    )
    started_at = time.perf_counter()
    snapshot_path = snapshot_download(**download_kwargs)
    logger.info("Model snapshot ready at: %s", snapshot_path)
    logger.debug("snapshot_download completed in %.2f s", _elapsed_s(started_at))
    return snapshot_path


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _elapsed_s(started_at: float) -> float:
    return time.perf_counter() - started_at


def _text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def _preview_text(text: str, max_chars: int = 120) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "..."


def _vllm_supports_init_kwarg(kwarg_name: str) -> bool:
    if VLLMEngine is None:
        return False
    try:
        parameters = inspect.signature(VLLMEngine.__init__).parameters.values()
    except Exception:
        return True

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    return kwarg_name in inspect.signature(VLLMEngine.__init__).parameters


def _engine_key(engine_kwargs: Dict[str, Any]) -> str:
    material = json.dumps(engine_kwargs, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _coerce_bool(value: Any, default: bool = False) -> bool:
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
        raise ValueError("Invalid boolean value: '{}'".format(value))
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError("Invalid boolean value type: {}".format(type(value).__name__))


def _coerce_positive_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("{} must be > 0, got {}".format(field_name, value))
    return parsed


def _coerce_optional_int(
    value: Any,
    field_name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    if value in [None, "", "None"]:
        return None
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError("{} must be >= {}, got {}".format(field_name, minimum, value))
    if maximum is not None and parsed > maximum:
        raise ValueError("{} must be <= {}, got {}".format(field_name, maximum, value))
    return parsed


def _coerce_optional_float(
    value: Any,
    field_name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value in [None, "", "None"]:
        return None
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ValueError("{} must be >= {}, got {}".format(field_name, minimum, value))
    if maximum is not None and parsed > maximum:
        raise ValueError("{} must be <= {}, got {}".format(field_name, maximum, value))
    return parsed


def _normalize_stop_sequences(stop: Any) -> Optional[List[str]]:
    if stop in [None, "", []]:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        result = []
        for value in stop:
            if value in [None, ""]:
                continue
            result.append(str(value))
        return result if result else None
    return [str(stop)]


def _truncate_at_stop_sequences(text: str, stop: Any) -> str:
    stop_sequences = _normalize_stop_sequences(stop)
    if not stop_sequences:
        return text.strip()

    earliest = None
    for sequence in stop_sequences:
        idx = text.find(sequence)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text.strip()
    return text[:earliest].strip()


def _extract_cached_prompt_tokens(request_output: Any) -> int:
    for attr in ("cached_tokens", "num_cached_tokens", "prefix_cache_hit_tokens"):
        value = getattr(request_output, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    metrics = getattr(request_output, "metrics", None)
    if metrics is not None:
        for attr in ("cached_tokens", "num_cached_tokens", "prefix_cache_hit_tokens"):
            value = getattr(metrics, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass

    return 0


LLM = LLMVLLM
