# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

"""Base classes and utilities for LLM connectors."""

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

try:
    from utility.logger import logger
except Exception:
    import sys

    def _find_repo_root(start_path):
        path = os.path.abspath(start_path)
        for _ in range(8):
            candidate = os.path.join(path, "utility", "logger.py")
            if os.path.isfile(candidate):
                return path
            new_path = os.path.dirname(path)
            if new_path == path:
                break
            path = new_path
        return None

    _root = _find_repo_root(os.path.dirname(__file__))
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)

    # Avoid shadowing by local KMS/LLM/utility.py
    if "utility" in sys.modules:
        module_path = getattr(sys.modules["utility"], "__file__", "") or ""
        if module_path.endswith(os.path.join("KMS", "LLM", "utility.py")):
            del sys.modules["utility"]

    from utility.logger import logger


def default_env_path() -> str:
    """Return the default LLM environment file path."""
    return os.path.join(os.path.dirname(__file__), ".env")


def load_llm_env(env_path: Optional[str] = None) -> str:
    """Load LLM environment variables and return the path used."""
    dotenv_path = env_path if env_path is not None else default_env_path()
    load_dotenv(dotenv_path=dotenv_path)
    return dotenv_path


def _is_empty_config_value(value: Any) -> bool:
    return value in [None, "", "None"]


def _resolve_env_reference(env_name: Any, config_key: str) -> Optional[str]:
    if _is_empty_config_value(env_name):
        return None

    env_var = str(env_name).strip()
    value = os.environ.get(env_var)
    if _is_empty_config_value(value):
        raise ValueError(
            "Missing environment variable {} referenced by {}. "
            "Set it in KMS/LLM/.env or in the shell environment.".format(env_var, config_key)
        )
    return value


def _looks_like_env_name(value: str) -> bool:
    return re.match(r"^[A-Z_][A-Z0-9_]*$", value.strip()) is not None


def resolve_config_value(
    config: Dict[str, Any],
    key: str,
    default: Any = None,
    env_key: Optional[str] = None,
    allow_bare_env: bool = False,
) -> Any:
    """Resolve a config value, allowing KEY_ENV, ${ENV_VAR}, or bare env names."""
    load_llm_env()

    reference_key = env_key or "{}_ENV".format(key)
    if reference_key in config and not _is_empty_config_value(config.get(reference_key)):
        return _resolve_env_reference(config.get(reference_key), reference_key)

    value = config.get(key, default)
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return _resolve_env_reference(value[2:-1], key)
    if isinstance(value, str) and allow_bare_env and _looks_like_env_name(value):
        return _resolve_env_reference(value, key)

    return value


class BaseLLM(ABC):
    """Abstract base class for LLM backends.

    Handles environment loading, optional few-shot examples, and shared
    generation configuration.
    """

    llm_default_config = {
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None,
        "use_cache": True,
        "seed": 42,
    }

    def __init__(
        self,
        examples_yaml_file: Optional[Iterable[str]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        env_path: Optional[str] = None,
    ) -> None:
        """Initialize the LLM base with optional examples and config.

        Args:
            examples_yaml_file (Optional[Iterable[str]]): List of YAML files with few-shot examples.
            llm_config (Optional[Dict[str, Any]]): Optional dict of generation parameters.
            env_path (Optional[str]): Optional .env path to load.
        """
        dotenv_path = env_path if env_path is not None else default_env_path()
        logger.info("Loading environment variables from: %s", dotenv_path)
        load_llm_env(env_path=dotenv_path)

        self.messages = []
        self.system_msg = ""
        self.disable_system = False

        if examples_yaml_file is None:
            examples_yaml_file = [""]

        self._load_examples(examples_yaml_file)
        self._config(llm_config)

    def _load_examples(self, examples_yaml_file: Iterable[str]) -> None:
        """Load conversation examples from YAML files.

        Few-shot examples are optional: missing files are skipped with a
        warning so that a backend can still be used without an examples
        folder.

        Args:
            examples_yaml_file (Iterable[str]): YAML file paths with examples.
        """
        examples_yaml_file = list(examples_yaml_file)

        if len(examples_yaml_file) <= 0:
            logger.info("No examples provided")
            return

        for file_name in examples_yaml_file:
            if file_name in [None, ""]:
                continue

            if not os.path.isfile(file_name):
                logger.warning(
                    "Examples file %s does not exist, skipping it. "
                    "The model will run without these few-shot examples.",
                    file_name,
                )
                continue

            self._include_yaml(file_name, self.messages)

    def _include_yaml(self, file_name: str, messages: List[Dict[str, Any]]) -> None:
        """Append YAML example entries into a messages list.

        Args:
            file_name (str): Path to a YAML examples file.
            messages (List[Dict[str, Any]]): Messages list to append into.

        Raises:
            KeyError: If required keys are missing from the YAML.
            yaml.YAMLError: If the YAML file cannot be parsed.
        """
        if not os.path.isfile(file_name):
            logger.warning("Examples file %s does not exist, skipping it.", file_name)
            return

        logger.info("Adding examples from file: %s", file_name)

        with open(file_name) as file:
            yaml_file = yaml.load(file, Loader=yaml.FullLoader)["entries"]

        if self.system_msg == "" and "system_msg" in yaml_file.keys():
            self.system_msg = yaml_file["system_msg"]
            messages.append(self.system_msg)

        for key in yaml_file.keys():
            if key == "convo":
                for msg_id in yaml_file["convo"]:
                    convo = yaml_file["convo"][msg_id]
                    if ["A", "Q"] == sorted(convo.keys()):
                        messages.append(convo["Q"])
                        messages.append(convo["A"])

            if key == "files":
                for child_file in yaml_file["files"]:
                    if os.path.isabs(child_file):
                        if os.path.exists(child_file):
                            self._include_yaml(child_file, messages)
                        continue

                    abs_file = os.path.join(
                        os.path.dirname(file_name), *child_file.split("/")
                    )
                    self._include_yaml(abs_file, messages)

    def _config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Merge user config with defaults and expose as attributes.

        Args:
            config (Optional[Dict[str, Any]]): Generation config overrides.

        Raises:
            ValueError: If an unknown config key is provided.
        """
        merged_config = dict(BaseLLM.llm_default_config)

        if config is not None:
            for key in config.keys():
                if key not in BaseLLM.llm_default_config:
                    raise ValueError("Unknown key in config: {}".format(key))
            merged_config.update(config)

        self.max_tokens = merged_config["max_tokens"]
        self.temperature = merged_config["temperature"]
        self.top_p = merged_config["top_p"]
        self.frequency_penalty = merged_config["frequency_penalty"]
        self.presence_penalty = merged_config["presence_penalty"]
        self.stop = merged_config["stop"]
        self.use_cache = merged_config["use_cache"]
        self.seed = merged_config["seed"]

    def _save_sent_query(self) -> None:
        """Persist the outgoing prompt to an output file for debugging."""
        if not os.path.exists("output"):
            os.makedirs("output")

        file_name = "sent_query_{}.txt".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(os.path.join("output", file_name), "w") as file:
            for msg in self.messages:
                file.write("{}: {}\n".format(msg["role"], msg["content"]))

    def _build_completion_kwargs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build provider-agnostic generation kwargs.

        Args:
            messages (List[Dict[str, Any]]): Message list to send to the backend.

        Returns:
            Dict[str, Any]: Keyword arguments for the backend call.
        """
        prepared_messages = self._prepare_messages(messages)
        kwargs = {
            "messages": prepared_messages,
            "seed": self.seed,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_tokens
        if self.top_p is not None and self.top_p > 0:
            kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.stop not in [None, "", []]:
            kwargs["stop"] = self.stop

        return kwargs

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize shared messages and optionally rewrite system messages.

        When ``self.disable_system`` is true, each ``system`` message is rewritten
        into an equivalent ``user`` + ``assistant`` pair. This is useful for
        chat templates that do not support a dedicated system role.
        """
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

            normalized.append({"role": role, "content": str(content)})

        if not getattr(self, "disable_system", False):
            return normalized

        rewritten: List[Dict[str, str]] = []
        for msg in normalized:
            if msg["role"] != "system":
                rewritten.append(msg)
                continue

            system_content = msg.get("content", "").strip()
            if not system_content:
                continue

            rewritten.append(
                {
                    "role": "user",
                    "content": "SYSTEM INSTRUCTIONS:\n{}".format(system_content),
                }
            )
            rewritten.append(
                {
                    "role": "assistant",
                    "content": "Understood. I will follow these instructions.",
                }
            )

        return rewritten

    def _query(
        self,
        prompt: str,
        end_when_error: bool = False,
        max_retry: int = 5,
        dry_run: bool = False,
    ) -> Tuple[bool, str, int, int]:
        """Send a prompt and return normalized query outputs.

        Returns:
            Tuple[bool, str, int, int]:
                (success, output_text, output_tokens, input_tokens)
        """
        conn_success = False
        llm_output = ""
        response = None

        self.messages.append({"role": "user", "content": prompt})
        self._save_sent_query()

        if dry_run:
            input_tokens = 0
            try:
                input_tokens = self._estimate_prompt_tokens(self.messages)
            except Exception:
                input_tokens = 0

            output_tokens = (
                self.max_tokens
                if self.max_tokens is not None
                else BaseLLM.llm_default_config["max_tokens"]
            )
            try:
                output_tokens = int(output_tokens)
            except Exception:
                output_tokens = int(BaseLLM.llm_default_config["max_tokens"])

            self.messages = self.messages[:-1]
            logger.info(
                "Dry run token estimate - input: %s, output (configured max_tokens): %s",
                input_tokens,
                output_tokens,
            )
            return True, "", output_tokens, input_tokens

        n_retry = 0
        while not conn_success and n_retry < max_retry:
            n_retry += 1
            try:
                logger.info("Sending query to the LLM ...")
                response = self._connect(self.messages)
                llm_output = self._extract_content(response)
                logger.debug("LLM response: %s", llm_output)
                conn_success = True
            except Exception as error:
                logger.error("LLM error: %s", error)
                if end_when_error:
                    break

        self.messages = self.messages[:-1]

        input_tokens = 0
        output_tokens = 0
        if response is not None:
            try:
                input_tokens = self._extract_prompt_tokens(response)
            except Exception:
                input_tokens = 0
            try:
                output_tokens = self._extract_completion_tokens(response)
            except Exception:
                output_tokens = 0

        return conn_success, llm_output, output_tokens, input_tokens

    def query(
        self,
        prompt: str,
        end_when_error: bool = False,
        max_retry: int = 5,
    ) -> Tuple[bool, str]:
        """Send a prompt and return model text output."""
        conn_success, llm_output, _, _ = self._query(
            prompt=prompt,
            end_when_error=end_when_error,
            max_retry=max_retry,
            dry_run=False,
        )
        return conn_success, llm_output

    def query_with_stats(
        self,
        prompt: str,
        end_when_error: bool = False,
        max_retry: int = 5,
    ) -> Tuple[bool, str, int]:
        """Send a prompt and return model text plus completion token count."""
        conn_success, llm_output, output_tokens, _ = self._query(
            prompt=prompt,
            end_when_error=end_when_error,
            max_retry=max_retry,
            dry_run=False,
        )
        return conn_success, llm_output, output_tokens

    def query_dry_run(self, prompt: str) -> Tuple[bool, int]:
        """Estimate prompt tokens without contacting the backend."""
        conn_success, _, _, input_tokens = self._query(
            prompt=prompt,
            end_when_error=False,
            max_retry=0,
            dry_run=True,
        )
        return conn_success, input_tokens

    @abstractmethod
    def _connect(self, messages: List[Dict[str, Any]]) -> Any:
        """Send messages to the backend and return a provider response.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            Any: Provider-specific response object.
        """
        pass

    @abstractmethod
    def _extract_content(self, response: Any) -> str:
        """Extract the text content from a provider response.

        Args:
            response (Any): Provider-specific response object.

        Returns:
            str: Assistant response text.
        """
        pass

    @abstractmethod
    def _extract_completion_tokens(self, response: Any) -> int:
        """Extract completion token count from a provider response.

        Args:
            response (Any): Provider-specific response object.

        Returns:
            int: Completion token count.
        """
        pass

    def _extract_prompt_tokens(self, response: Any) -> int:
        """Extract prompt token count from a provider response.

        Args:
            response (Any): Provider-specific response object.

        Returns:
            int: Prompt token count. Defaults to 0 when unavailable.
        """
        return 0

    def _estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate prompt token count without calling a provider.

        This is a provider-agnostic heuristic fallback intended for dry-run
        cost estimation when exact tokenizer parity is unavailable.

        Args:
            messages (List[Dict[str, Any]]): Chat-style message list.

        Returns:
            int: Estimated prompt token count.
        """
        normalized_chunks: List[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")
            else:
                role = "system"
                content = str(msg)

            if isinstance(content, list):
                content = " ".join(str(chunk) for chunk in content)
            else:
                content = str(content)

            normalized_chunks.append("{}: {}".format(role, content))

        if not normalized_chunks:
            return 0

        text = "\n".join(normalized_chunks)
        # Rough rule-of-thumb for many BPE tokenizers.
        return max(1, int(len(text) / 4))
    
    def close(self) -> None:
        """Release backend resources. Default is no-op."""
        return None

try:
    from .llm_factory import create_llm_from_config
except Exception:
    from llm_factory import create_llm_from_config
