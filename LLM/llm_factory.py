"""Pick the right backend for a configuration file.

The configuration decides which class is used: an explicit ``PROVIDER`` key when present, and a
handful of heuristics on the remaining keys otherwise. :func:`create_llm` then hands the file to
that class' :meth:`~llm_base.BaseLLM.from_config`.
"""

import importlib
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

try:
    from llm_base import load_config_file, logger
except Exception:
    try:
        from .llm_base import load_config_file, logger
    except Exception:
        import sys

        sys.path.append(os.path.dirname(__file__))
        from llm_base import load_config_file, logger

if TYPE_CHECKING:
    from llm_base import BaseLLM


PROVIDER_ALIASES = {
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

# provider -> (sub-package, module and class name)
PROVIDER_MODULES = {
    "openai": ("LLMOpenAI", "LLMOpenAI"),
    "azure_openai": ("LLMAzureOpenAI", "LLMAzureOpenAI"),
    "anthropic": ("LLMAnthropic", "LLMAnthropic"),
    "gemini": ("LLMGemini", "LLMGemini"),
    "glm": ("LLMGLM", "LLMGLM"),
    "huggingface": ("LLMHuggingFace", "LLMHuggingFace"),
    "vllm": ("LLMVLLM", "LLMVLLM"),
}


def default_config_dir() -> str:
    """Return the directory holding the bundled configuration files.

    Returns
    -------
    str
        Path of the ``conf`` folder sitting next to this module.
    """
    return os.path.join(os.path.dirname(__file__), "conf")


def normalize_provider(provider: str) -> str:
    """Normalize a provider name to its canonical slug.

    Parameters
    ----------
    provider : str
        Provider name, in any of its accepted spellings.

    Returns
    -------
    str
        Canonical provider slug.

    Raises
    ------
    ValueError
        If the provider is unknown.
    """
    normalized = str(provider).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[normalized]
    raise ValueError("Unsupported LLM provider '{}'".format(provider))


def infer_provider(config: Dict[str, Any]) -> str:
    """Infer the provider from a loaded configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed configuration.

    Returns
    -------
    str
        Canonical provider slug.

    Raises
    ------
    ValueError
        If the provider cannot be determined.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")

    explicit = config.get("PROVIDER") or config.get("provider")
    if explicit not in [None, ""]:
        return normalize_provider(str(explicit))

    if "ENDPOINT" in config or "ENDPOINT_ENV" in config or "API_VERSION" in config:
        return "azure_openai"

    model = (
        str(config.get("LLM_VERSION") or config.get("MODEL_NAME") or config.get("MODEL") or "")
        .strip()
        .lower()
    )
    api_key_name = (
        str(config.get("API_KEY_NAME") or config.get("API_KEY_ENV") or "").strip().lower()
    )
    has_api_key = "API_KEY_NAME" in config or "API_KEY_ENV" in config

    if "anthropic" in api_key_name or model.startswith("claude"):
        return "anthropic"
    if "gemini" in api_key_name or model.startswith("gemini"):
        return "gemini"
    if "glm" in api_key_name or "zhipu" in api_key_name or model.startswith("glm"):
        return "glm"

    vllm_keys = (
        "ENABLE_PREFIX_CACHING",
        "TENSOR_PARALLEL_SIZE",
        "MAX_NUM_BATCHED_TOKENS",
        "MAX_NUM_SEQS",
        "GPU_MEMORY_UTILIZATION",
    )
    if any(key in config for key in vllm_keys) and not has_api_key:
        return "vllm"

    if (
        any(key in config for key in ("MODEL_NAME", "DEVICE", "QUANTIZE", "CACHE_DIR"))
        and not has_api_key
    ):
        return "huggingface"

    if "openai" in api_key_name or model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"

    raise ValueError("Unable to infer provider from config. Add PROVIDER to the YAML file.")


def infer_provider_from_file(config_file: Union[str, Path]) -> str:
    """Infer the provider from a configuration file path.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML configuration file.

    Returns
    -------
    str
        Canonical provider slug.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a YAML file.
    ValueError
        If the provider cannot be determined.
    """
    return infer_provider(load_config_file(config_file))


def resolve_class(provider: str) -> Type["BaseLLM"]:
    """Import and return the backend class for a provider.

    Parameters
    ----------
    provider : str
        Provider name or slug.

    Returns
    -------
    Type[BaseLLM]
        The backend class.

    Raises
    ------
    ValueError
        If the provider is unknown.
    ImportError
        If the backend module cannot be imported.
    """
    slug = normalize_provider(provider)
    if slug not in PROVIDER_MODULES:
        raise ValueError("No backend registered for provider '{}'".format(provider))

    package, name = PROVIDER_MODULES[slug]

    # The package is imported as "LLM.<backend>" here, as "<backend>" when this folder is itself
    # on sys.path, and as "KMS.LLM.<backend>" in the project it is shared with.
    candidates = []
    if __package__:
        candidates.append("{}.{}.{}".format(__package__, package, name))
    candidates.append("{}.{}".format(package, name))
    candidates.append("KMS.LLM.{}.{}".format(package, name))

    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            errors.append("{}: {}".format(module_name, error))
            continue

        backend = getattr(module, name, None) or getattr(module, "LLM", None)
        if inspect.isclass(backend):
            return backend
        errors.append("{}: no backend class found".format(module_name))

    raise ImportError(
        "Unable to import the backend for provider '{}'. Tried: {}".format(slug, " | ".join(errors))
    )


def resolve_class_from_file(config_file: Union[str, Path]) -> Type["BaseLLM"]:
    """Return the backend class a configuration file asks for.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML configuration file.

    Returns
    -------
    Type[BaseLLM]
        The backend class.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a YAML file.
    ValueError
        If the provider cannot be determined.
    ImportError
        If the backend module cannot be imported.
    """
    return resolve_class(infer_provider_from_file(config_file))


def create_llm(
    config_file: Union[str, Path],
    examples: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> "BaseLLM":
    """Build the backend described by a configuration file.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML configuration file.
    examples : Optional[Union[str, Path]], optional
        Folder with few-shot examples.
    **overrides : Any
        Forwarded to the backend's ``from_config``.

    Returns
    -------
    BaseLLM
        A configured backend instance.
    """
    backend = resolve_class_from_file(config_file)
    logger.debug("Selected backend %s for %s", backend.__name__, config_file)
    return backend.from_config(config_file, examples=examples, **overrides)


def list_config_files(config_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """List the YAML configuration files of a directory.

    Parameters
    ----------
    config_dir : Optional[Union[str, Path]], optional
        Directory to scan. Defaults to ``LLM/conf``.

    Returns
    -------
    List[str]
        Sorted configuration file paths.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    target_dir = Path(config_dir) if config_dir is not None else Path(default_config_dir())
    if not target_dir.is_dir():
        raise FileNotFoundError("LLM config directory not found: {}".format(target_dir))

    return sorted(str(path) for path in target_dir.iterdir() if path.suffix in (".yaml", ".yml"))


def select_llm(
    config_dir: Optional[Union[str, Path]] = None,
    selection: Optional[Union[int, str]] = None,
    examples: Optional[Union[str, Path]] = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    **overrides: Any,
) -> "BaseLLM":
    """Pick a configuration file from a directory and build its backend.

    Parameters
    ----------
    config_dir : Optional[Union[str, Path]], optional
        Directory to pick from. Defaults to ``LLM/conf``.
    selection : Optional[Union[int, str]], optional
        A 1-based index, a file name or path, or ``None`` to ask interactively.
    examples : Optional[Union[str, Path]], optional
        Folder with few-shot examples.
    input_fn : Callable[[str], str], optional
        Input function used in interactive mode.
    print_fn : Callable[[str], None], optional
        Print function used in interactive mode.
    **overrides : Any
        Forwarded to the backend's ``from_config``.

    Returns
    -------
    BaseLLM
        A configured backend instance.

    Raises
    ------
    FileNotFoundError
        If no configuration file matches.
    ValueError
        If an interactive selection is cancelled.
    """
    config_files = list_config_files(config_dir)
    if not config_files:
        raise FileNotFoundError(
            "No configuration files found in {}".format(config_dir or default_config_dir())
        )

    if isinstance(selection, str) and not selection.isdigit():
        candidate = Path(selection)
        if not candidate.is_file():
            matches = [path for path in config_files if Path(path).name == selection]
            if not matches:
                raise FileNotFoundError("Configuration file not found: {}".format(selection))
            candidate = Path(matches[0])
        return create_llm(candidate, examples=examples, **overrides)

    if selection is None:
        for index, path in enumerate(config_files, start=1):
            print_fn("{}) {}".format(index, Path(path).name))
        while True:
            choice = input_fn("Select a configuration [1-{}]: ".format(len(config_files))).strip()
            if choice.lower() in {"q", "quit", "exit"}:
                raise ValueError("LLM selection cancelled by user.")
            if choice.isdigit() and 1 <= int(choice) <= len(config_files):
                selection = int(choice)
                break
            print_fn("Please enter a number between 1 and {}.".format(len(config_files)))

    index = int(selection)
    if not 1 <= index <= len(config_files):
        raise FileNotFoundError("Selection out of range: {}".format(index))

    return create_llm(config_files[index - 1], examples=examples, **overrides)


__all__ = [
    "PROVIDER_ALIASES",
    "PROVIDER_MODULES",
    "create_llm",
    "default_config_dir",
    "infer_provider",
    "infer_provider_from_file",
    "list_config_files",
    "normalize_provider",
    "resolve_class",
    "resolve_class_from_file",
    "select_llm",
]
