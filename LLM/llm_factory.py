# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Factory and config helpers for creating LLM backends."""

import importlib
import inspect
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Type, Union

import yaml

if TYPE_CHECKING:
    from .llm_base import BaseLLM


_LLM_PROVIDER_ALIASES = {
    "azure": "azure_openai",
    "azure_openai": "azure_openai",
    "azure-openai": "azure_openai",
    "azureopenai": "azure_openai",
    "openai_azure": "azure_openai",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "google_genai": "gemini",
    "glm": "glm",
    "zhipu": "glm",
    "zhipuai": "glm",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "local": "huggingface",
    "vllm": "vllm",
    "v_llm": "vllm",
}

_LLM_PROVIDER_IMPORTS = {
    "openai": [
        ("KMS.LLM.LLMOpenAI.LLMOpenAI", "LLMOpenAI"),
        ("LLMOpenAI.LLMOpenAI", "LLMOpenAI"),
    ],
    "azure_openai": [
        ("KMS.LLM.LLMAzureOpenAI.LLMAzureOpenAI", "LLMAzureOpenAI"),
        ("LLMAzureOpenAI.LLMAzureOpenAI", "LLMAzureOpenAI"),
    ],
    "anthropic": [
        ("KMS.LLM.LLMAnthropic.LLMAnthropic", "LLMAnthropic"),
        ("LLMAnthropic.LLMAnthropic", "LLMAnthropic"),
    ],
    "gemini": [
        ("KMS.LLM.LLMGemini.LLMGemini", "LLMGemini"),
        ("LLMGemini.LLMGemini", "LLMGemini"),
    ],
    "glm": [
        ("KMS.LLM.LLMGLM.LLMGLM", "LLMGLM"),
        ("LLMGLM.LLMGLM", "LLMGLM"),
    ],
    "huggingface": [
        ("KMS.LLM.LLMHuggingFace.LLMHuggingFace", "LLMHuggingFace"),
        ("LLMHuggingFace.LLMHuggingFace", "LLMHuggingFace"),
    ],
    "vllm": [
        ("KMS.LLM.LLMVLLM.LLMVLLM", "LLMVLLM"),
        ("LLMVLLM.LLMVLLM", "LLMVLLM"),
    ],
}

_LLM_PROVIDER_CLASS_NAMES = {
    "openai": "LLMOpenAI",
    "azure_openai": "LLMAzureOpenAI",
    "anthropic": "LLMAnthropic",
    "gemini": "LLMGemini",
    "glm": "LLMGLM",
    "huggingface": "LLMHuggingFace",
    "vllm": "LLMVLLM",
}


def _default_llm_config_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "conf")


def _normalize_provider_name(provider: str) -> str:
    normalized = str(provider).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _LLM_PROVIDER_ALIASES:
        return _LLM_PROVIDER_ALIASES[normalized]
    raise ValueError("Unsupported LLM provider '{}'".format(provider))


def load_llm_config_file(llm_config_file: str) -> Dict[str, Any]:
    """Load a YAML LLM config and return its dictionary representation."""
    if not isinstance(llm_config_file, str) or not llm_config_file.strip():
        raise ValueError("llm_config_file must be a non-empty string.")

    if not llm_config_file.endswith((".yaml", ".yml")):
        raise ValueError("Config file must be a YAML file: {}".format(llm_config_file))

    if not os.path.isfile(llm_config_file):
        raise FileNotFoundError("LLM config file not found: {}".format(llm_config_file))

    with open(llm_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if not isinstance(config, dict):
        raise ValueError("LLM config must define a mapping: {}".format(llm_config_file))

    return config


def infer_llm_provider_from_config(llm_config: Dict[str, Any]) -> str:
    """Infer provider slug from a loaded LLM config dictionary."""
    if not isinstance(llm_config, dict):
        raise ValueError("llm_config must be a dictionary.")

    explicit_provider = llm_config.get("PROVIDER") or llm_config.get("provider")
    if explicit_provider not in [None, ""]:
        return _normalize_provider_name(str(explicit_provider))

    if "ENDPOINT" in llm_config or "API_VERSION" in llm_config:
        return "azure_openai"

    model_name = (
        llm_config.get("LLM_VERSION")
        or llm_config.get("MODEL_NAME")
        or llm_config.get("MODEL")
        or ""
    )
    model_name_l = str(model_name).strip().lower()

    api_key_name = (
        llm_config.get("API_KEY_NAME")
        or llm_config.get("API_KEY_ENV")
        or ""
    )
    api_key_name_l = str(api_key_name).strip().lower()

    if "anthropic" in api_key_name_l or model_name_l.startswith("claude"):
        return "anthropic"

    if "gemini" in api_key_name_l or model_name_l.startswith("gemini"):
        return "gemini"

    if "glm" in api_key_name_l or "zhipu" in api_key_name_l or model_name_l.startswith("glm"):
        return "glm"

    if any(
        key in llm_config
        for key in (
            "ENABLE_PREFIX_CACHING",
            "TENSOR_PARALLEL_SIZE",
            "MAX_NUM_BATCHED_TOKENS",
            "MAX_NUM_SEQS",
            "GPU_MEMORY_UTILIZATION",
        )
    ):
        if "API_KEY_NAME" not in llm_config and "API_KEY_ENV" not in llm_config:
            return "vllm"

    if any(key in llm_config for key in ("MODEL_NAME", "DEVICE", "QUANTIZE", "CACHE_DIR")):
        if "API_KEY_NAME" not in llm_config and "API_KEY_ENV" not in llm_config:
            return "huggingface"
        if str(llm_config.get("PROVIDER", "")).strip().lower() in {"hf", "huggingface"}:
            return "huggingface"

    if "openai" in api_key_name_l:
        return "openai"

    if model_name_l.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"

    raise ValueError(
        "Unable to infer provider from config. Add PROVIDER to the YAML file."
    )


def infer_llm_provider_from_config_file(llm_config_file: str) -> str:
    """Infer provider slug from a YAML config file path."""
    config = load_llm_config_file(llm_config_file)
    return infer_llm_provider_from_config(config)


def resolve_llm_class_from_provider(provider: str) -> Type["BaseLLM"]:
    """Resolve backend class for a provider slug."""
    normalized_provider = _normalize_provider_name(provider)
    candidates = _LLM_PROVIDER_IMPORTS.get(normalized_provider, [])
    if not candidates:
        raise ValueError("No class resolver registered for provider '{}'".format(provider))

    import_errors = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            llm_class = getattr(module, class_name)
            if inspect.isclass(llm_class):
                return llm_class
        except Exception as error:
            import_errors.append("{}: {}".format(module_name, error))
            continue

    raise ImportError(
        "Unable to import class for provider '{}'. Tried: {}".format(
            normalized_provider, " | ".join(import_errors)
        )
    )


def resolve_llm_class_from_config_file(llm_config_file: str) -> Type["BaseLLM"]:
    """Resolve backend class from a YAML config file path."""
    provider = infer_llm_provider_from_config_file(llm_config_file)
    return resolve_llm_class_from_provider(provider)


def infer_llm_class_name_from_provider(provider: str) -> str:
    """Return the expected backend class name for a provider slug."""
    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider not in _LLM_PROVIDER_CLASS_NAMES:
        raise ValueError("No class name registered for provider '{}'".format(provider))
    return _LLM_PROVIDER_CLASS_NAMES[normalized_provider]


def infer_llm_class_name_from_config_file(llm_config_file: str) -> str:
    """Return the expected backend class name for a YAML config file path."""
    provider = infer_llm_provider_from_config_file(llm_config_file)
    return infer_llm_class_name_from_provider(provider)


def _build_llm_init_kwargs(
    llm_class: Type["BaseLLM"],
    llm_config_file: str,
    examples_yaml_file: Optional[Iterable[str]],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    signature = inspect.signature(llm_class.__init__)
    parameters = signature.parameters
    supports_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )

    init_kwargs: Dict[str, Any] = {}
    if "llm_config_file" in parameters:
        init_kwargs["llm_config_file"] = llm_config_file
    elif "llm_connection_config_file" in parameters:
        init_kwargs["llm_connection_config_file"] = llm_config_file
    else:
        raise TypeError(
            "{}.__init__ does not accept a config path argument.".format(llm_class.__name__)
        )

    if examples_yaml_file is not None and "examples_yaml_file" in parameters:
        init_kwargs["examples_yaml_file"] = examples_yaml_file

    unknown_kwargs = []
    for key, value in kwargs.items():
        if key in parameters or supports_var_kwargs:
            init_kwargs[key] = value
        else:
            unknown_kwargs.append(key)

    if unknown_kwargs:
        raise TypeError(
            "Unsupported initialization argument(s) for {}: {}".format(
                llm_class.__name__, ", ".join(sorted(unknown_kwargs))
            )
        )

    return init_kwargs


def create_llm_from_config(
    llm_config_file: str,
    examples_yaml_file: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> "BaseLLM":
    """Instantiate the correct LLM backend based on a YAML config file."""
    llm_class = resolve_llm_class_from_config_file(llm_config_file)
    init_kwargs = _build_llm_init_kwargs(
        llm_class=llm_class,
        llm_config_file=llm_config_file,
        examples_yaml_file=examples_yaml_file,
        kwargs=kwargs,
    )
    return llm_class(**init_kwargs)


def list_llm_config_files(config_dir: Optional[str] = None) -> List[str]:
    """List YAML config files from the configured LLM config directory."""
    target_dir = config_dir if config_dir is not None else _default_llm_config_dir()
    if not os.path.isdir(target_dir):
        raise FileNotFoundError("LLM config directory not found: {}".format(target_dir))

    config_files = []
    for file_name in sorted(os.listdir(target_dir)):
        if not file_name.endswith((".yaml", ".yml")):
            continue
        full_path = os.path.join(target_dir, file_name)
        if os.path.isfile(full_path):
            config_files.append(full_path)
    return config_files


def _llm_config_summary(config_file: str) -> Dict[str, str]:
    summary = {
        "path": config_file,
        "name": os.path.basename(config_file),
        "provider": "unknown",
        "class_name": "unknown",
        "model": "unknown",
    }
    try:
        config = load_llm_config_file(config_file)
        summary["provider"] = infer_llm_provider_from_config(config)
        summary["class_name"] = infer_llm_class_name_from_provider(summary["provider"])
        model_name = (
            config.get("LLM_VERSION")
            or config.get("MODEL_NAME")
            or config.get("MODEL")
            or "unknown"
        )
        summary["model"] = str(model_name)
    except Exception:
        pass
    return summary


def _resolve_config_selection(
    config_files: List[str],
    selection: Optional[Union[int, str]],
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> str:
    if not config_files:
        raise FileNotFoundError("No LLM config files were found.")

    if selection is not None:
        if isinstance(selection, int):
            index = selection - 1
            if index < 0 or index >= len(config_files):
                raise IndexError("Selection {} is out of range.".format(selection))
            return config_files[index]

        selection_str = str(selection).strip()
        if selection_str.isdigit():
            return _resolve_config_selection(
                config_files=config_files,
                selection=int(selection_str),
                input_fn=input_fn,
                print_fn=print_fn,
            )

        if os.path.isfile(selection_str):
            return selection_str

        by_name = {os.path.basename(path): path for path in config_files}
        if selection_str in by_name:
            return by_name[selection_str]

        raise ValueError("Unknown selection '{}'".format(selection))

    summaries = [_llm_config_summary(path) for path in config_files]
    print_fn("Available LLM configs:")
    for idx, summary in enumerate(summaries, start=1):
        print_fn(
            "{}. {} | provider={} | class={} | model={}".format(
                idx,
                summary["name"],
                summary["provider"],
                summary["class_name"],
                summary["model"],
            )
        )

    while True:
        choice = input_fn("Select an LLM config by number (or q to quit): ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            raise ValueError("LLM selection cancelled by user.")
        if not choice.isdigit():
            print_fn("Please enter a valid number.")
            continue
        index = int(choice) - 1
        if index < 0 or index >= len(config_files):
            print_fn("Selection out of range.")
            continue
        return config_files[index]


def select_llm_from_config_dir(
    config_dir: Optional[str] = None,
    examples_yaml_file: Optional[Iterable[str]] = None,
    selection: Optional[Union[int, str]] = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    **kwargs: Any,
) -> "BaseLLM":
    """Select a config from a directory and return an initialized LLM instance.

    Args:
        config_dir (Optional[str]): Directory containing YAML configs. Defaults to KMS/LLM/conf.
        examples_yaml_file (Optional[Iterable[str]]): Optional few-shot examples.
        selection (Optional[Union[int, str]]): Optional non-interactive selector.
            - int: 1-based index in sorted config list
            - str: absolute/relative config path, config file name, or numeric string
            - None: interactive prompt
        input_fn (Callable[[str], str]): Input function used for interactive mode.
        print_fn (Callable[[str], None]): Print function used for interactive mode.
        **kwargs (Any): Extra kwargs forwarded to the selected backend constructor.

    Returns:
        BaseLLM: Initialized backend instance.
    """
    config_files = list_llm_config_files(config_dir=config_dir)
    selected_config = _resolve_config_selection(
        config_files=config_files,
        selection=selection,
        input_fn=input_fn,
        print_fn=print_fn,
    )
    return create_llm_from_config(
        llm_config_file=selected_config,
        examples_yaml_file=examples_yaml_file,
        **kwargs,
    )


__all__ = [
    "create_llm_from_config",
    "infer_llm_class_name_from_config_file",
    "infer_llm_class_name_from_provider",
    "infer_llm_provider_from_config",
    "infer_llm_provider_from_config_file",
    "list_llm_config_files",
    "load_llm_config_file",
    "resolve_llm_class_from_config_file",
    "resolve_llm_class_from_provider",
    "select_llm_from_config_dir",
]
