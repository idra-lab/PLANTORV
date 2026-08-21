"""Generic LLM interface.

A single :class:`BaseLLM` describes what every model backend must be able to do, and each
provider specializes it. Three methods form the public surface and are meant to be overloaded:

* :meth:`BaseLLM.from_config` -- build an instance from a YAML configuration file.
* :meth:`BaseLLM.query`       -- connect to the model and send a message (optionally with images).
* :meth:`BaseLLM.prepare`     -- load few-shot examples from a folder (not implemented yet).

Generation parameters are *not* hard-coded here. Whatever sits under ``LLM_CONFIG`` in the YAML
file is forwarded to the provider request as-is, so a model that wants ``max_completion_tokens``
and one that wants ``max_tokens`` are both expressed by their own configuration file rather than
by a flag in the code. As an example, consider a configuration file like this

.. code-block:: yaml
    LLM_VERSION : "gpt-5.2-chat"
    API_KEY_NAME : "AZURE_OPENAI_API_KEY"
    ENDPOINT_ENV : "AZURE_OPENAI_ENDPOINT"
    API_VERSION : "2024-12-01-preview"
    SYSTEM_FINGERPRINT : "None"

    LLM_CONFIG:
    max_completion_tokens: 16384
    seed: 42
"""

import base64
import os
import re
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from dotenv import load_dotenv

try:
    from utility.logger import logger
except Exception:
    import sys

    def _find_repo_root(start_path: str) -> Optional[str]:
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

    # Avoid shadowing by a local utility.py sitting next to a backend.
    if "utility" in sys.modules:
        module_path = getattr(sys.modules["utility"], "__file__", "") or ""
        if module_path.endswith(os.path.join("LLM", "utility.py")):
            del sys.modules["utility"]

    from utility.logger import logger


# Type of an image accepted by query(): a PIL image, or a path to one.
ImageInput = Union["Any", str, Path]


## ENVIRONMENT #########################################################################################################


# Environment file the LLM layer reads, as set by :func:`configure_env`. ``None`` means
# "not configured", so the project root's ``.env`` is used.
_ENV_PATH: Optional[str] = None

# Set by :func:`configure_env` when an entry point asks for no file to be read at all.
_ENV_DISABLED: bool = False


def configure_env(env_path: Optional[Union[str, Path]] = None, load: bool = True) -> None:
    """Choose the environment file the LLM layer reads, for the rest of the process.

    Lets an entry point propagate its own choice: ``samgpt.py --env-file other.env`` calls
    this so the backends read ``other.env`` too, rather than silently falling back to the
    project root's ``.env`` and picking up credentials the caller meant to replace.
    ``--no-env-file`` calls it with ``load=False``, which honours "use the shell environment
    and nothing else" everywhere rather than only in the entry point.

    Parameters
    ----------
    env_path : str or Path, optional
        File to read. ``None`` restores the default, the project root's ``.env``.
    load : bool, optional
        When False, no environment file is read at all and ``env_path`` is ignored.
    """
    global _ENV_PATH, _ENV_DISABLED

    _ENV_DISABLED = not load
    _ENV_PATH = None if env_path is None else str(env_path)

    if not _ENV_DISABLED:
        load_llm_env()


def default_env_path() -> str:
    """Return the environment file path the LLM layer reads.

    Whatever :func:`configure_env` was given, or else the project root's ``.env``: ``LLM/``
    sits one directory below the root, so this resolves to the same file ``samgpt.py``
    loads. One file rather than two means credentials cannot end up duplicated across them,
    or worse, disagreeing about which endpoint a key belongs to.

    Returns
    -------
    str
        Path of the configured file, or of the ``.env`` in the directory containing this
        module's package.
    """
    if _ENV_PATH is not None:
        return _ENV_PATH
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(package_dir), ".env")


def load_llm_env(env_path: Optional[str] = None) -> Optional[str]:
    """Load LLM environment variables and return the path used.

    Parameters
    ----------
    env_path : Optional[str], optional
        Environment file to read. Defaults to :func:`default_env_path`.

    Returns
    -------
    Optional[str]
        Path of the environment file that was loaded, or None when
        :func:`configure_env` disabled reading one and no explicit path was given.
    """
    if env_path is None and _ENV_DISABLED:
        return None

    dotenv_path = env_path if env_path is not None else default_env_path()
    load_dotenv(dotenv_path=dotenv_path)
    return dotenv_path


def _is_empty_config_value(value: Any) -> bool:
    """Return whether a configuration value counts as unset.

    Parameters
    ----------
    value : Any
        Value read from a configuration file or from the environment.

    Returns
    -------
    bool
        ``True`` for ``None``, the empty string, and the literal string ``"None"``.
    """
    return value in [None, "", "None"]


def _resolve_env_reference(env_name: Any, config_key: str) -> Optional[str]:
    """Read the environment variable a configuration entry points to.

    Parameters
    ----------
    env_name : Any
        Name of the environment variable holding the value.
    config_key : str
        Configuration key that declared the reference, quoted in the error message.

    Returns
    -------
    Optional[str]
        Value of the environment variable, or ``None`` when no variable is named.

    Raises
    ------
    ValueError
        If the named environment variable is unset or empty.
    """
    if _is_empty_config_value(env_name):
        return None

    env_var = str(env_name).strip()
    value = os.environ.get(env_var)
    if _is_empty_config_value(value):
        raise ValueError(
            "Missing environment variable {} referenced by {}. "
            "Set it in {} or in the shell environment.".format(
                env_var, config_key, default_env_path()
            )
        )
    return value


def _looks_like_env_name(value: str) -> bool:
    """Return whether a string is shaped like an environment variable name.

    Parameters
    ----------
    value : str
        Candidate string, typically a raw configuration value.

    Returns
    -------
    bool
        ``True`` when the string holds upper-case letters, digits and underscores only.
    """
    return re.match(r"^[A-Z_][A-Z0-9_]*$", value.strip()) is not None


def resolve_config_value(
    config: Dict[str, Any],
    key: str,
    default: Any = None,
    env_key: Optional[str] = None,
    allow_bare_env: bool = False,
) -> Any:
    """Resolve a config value, allowing KEY_ENV, ${ENV_VAR}, or bare env names.

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed configuration.
    key : str
        Configuration key to resolve.
    default : Any, optional
        Value returned when the key is absent.
    env_key : Optional[str], optional
        Companion key naming an environment variable. Defaults to ``"<key>_ENV"``.
    allow_bare_env : bool, optional
        Read a plain upper-case value as an environment variable name.

    Returns
    -------
    Any
        The configured value, read from the environment when the configuration points to it.

    Raises
    ------
    ValueError
        If a referenced environment variable is unset or empty.
    """
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


## CONFIGURATION #######################################################################################################


def load_config_file(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a YAML file.
    ValueError
        If the file does not contain a mapping.
    """
    config_path = Path(config_file)

    if config_path.suffix not in (".yaml", ".yml") or not config_path.is_file():
        raise FileNotFoundError(
            "The selected file {} does not exist or is not a yaml file".format(config_path)
        )

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if not isinstance(config, dict):
        raise ValueError("LLM config must define a mapping: {}".format(config_path))

    return config


## MESSAGES ############################################################################################################


def normalize_messages(
    messages: List[Dict[str, Any]],
    disable_system: bool = False,
) -> List[Dict[str, str]]:
    """Flatten messages into plain-text roles.

    Used by local backends, which render the conversation into a single prompt and therefore
    cannot carry structured content.

    Parameters
    ----------
    messages : List[Dict[str, Any]]
        Messages in the shared chat format.
    disable_system : bool, optional
        Rewrite every ``system`` message as a ``user``/``assistant`` pair, for chat templates
        without a system role.

    Returns
    -------
    List[Dict[str, str]]
        Messages whose content is always a string.
    """
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "user")).strip().lower()
            content = message.get("content", "")
        else:
            role = "system"
            content = message

        if isinstance(content, list):
            # Keep the text parts only: a local text model cannot consume the others.
            texts = [
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            ]
            content = " ".join(text for text in texts if text)

        normalized.append({"role": role, "content": str(content)})

    if not disable_system:
        return normalized

    rewritten: List[Dict[str, str]] = []
    for message in normalized:
        if message["role"] != "system":
            rewritten.append(message)
            continue

        system_content = message["content"].strip()
        if not system_content:
            continue

        rewritten.append(
            {"role": "user", "content": "SYSTEM INSTRUCTIONS:\n{}".format(system_content)}
        )
        rewritten.append(
            {"role": "assistant", "content": "Understood. I will follow these instructions."}
        )

    return rewritten


## IMAGES ##############################################################################################################


def encode_image(image: ImageInput, image_format: str = "PNG") -> Tuple[str, str]:
    """Encode an image as base64.

    Parameters
    ----------
    image : ImageInput
        A ``PIL.Image.Image``, or the path of an image file.
    image_format : str, optional
        Format used when re-encoding an in-memory image.

    Returns
    -------
    Tuple[str, str]
        The MIME type and the base64-encoded payload.

    Raises
    ------
    TypeError
        If the image is neither a PIL image nor a readable path.
    FileNotFoundError
        If a path is given but does not exist.
    """
    # Imported lazily: text-only deployments do not need Pillow.
    from PIL import Image

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Images must be PIL.Image.Image instances or paths, got {}".format(type(image).__name__)
        )

    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")

    with BytesIO() as buffer:
        image.save(buffer, format=image_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return "image/{}".format(image_format.lower()), encoded


def image_data_url(image: ImageInput, image_format: str = "PNG") -> str:
    """Return an image encoded as a ``data:`` URL.

    Parameters
    ----------
    image : ImageInput
        A ``PIL.Image.Image``, or the path of an image file.
    image_format : str, optional
        Format used when re-encoding an in-memory image.

    Returns
    -------
    str
        The image as a ``data:<mime>;base64,<payload>`` URL.
    """
    mime_type, encoded = encode_image(image, image_format=image_format)
    return "data:{};base64,{}".format(mime_type, encoded)


## BASE CLASS ##########################################################################################################


class BaseLLM(ABC):
    """Common interface shared by every LLM backend.

    Subclasses describe a provider by setting the class attributes below and by implementing the
    connection hooks. The three public methods (:meth:`from_config`, :meth:`query`,
    :meth:`prepare`) already work for any backend that implements those hooks, and can be
    overloaded when a provider needs something different.

    Attributes
    ----------
    PROVIDER : str
        Slug used by the factory to select this backend.
    DEFAULT_PARAMS : Dict[str, Any]
        Request parameters applied when the config omits them.
    PARAM_ALIASES : Dict[str, str]
        Renames applied to ``LLM_CONFIG`` keys, so that a config written with a generic name
        still reaches the provider under the name it expects.
    NON_REQUEST_PARAMS : Tuple[str, ...]
        ``LLM_CONFIG`` keys consumed by the backend itself and never forwarded to the provider
        request.
    SUPPORTS_IMAGES : bool
        Whether :meth:`query` accepts images.
    """

    PROVIDER: str = ""
    DEFAULT_PARAMS: Dict[str, Any] = {}
    PARAM_ALIASES: Dict[str, str] = {}
    NON_REQUEST_PARAMS: Tuple[str, ...] = ()
    SUPPORTS_IMAGES: bool = False

    def __init__(
        self,
        model: str,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[Union[str, Path]] = None,
        examples: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize a backend.

        Parameters
        ----------
        model : str
            Model/deployment identifier sent to the provider.
        params : Optional[Dict[str, Any]], optional
            Request parameters, normally the ``LLM_CONFIG`` block of the configuration file.
            Forwarded to the provider as-is.
        config : Optional[Dict[str, Any]], optional
            The full configuration dictionary, kept so that backends can read their own
            provider-specific keys.
        config_file : Optional[Union[str, Path]], optional
            Path the configuration was read from.
        examples : Optional[Union[str, Path]], optional
            Folder with few-shot examples, passed to :meth:`prepare`.

        Raises
        ------
        ValueError
            If no model name is given.
        """
        if not model:
            raise ValueError("Missing model name. Expected LLM_VERSION (or MODEL/MODEL_NAME).")

        self.model = str(model)
        self.config = dict(config or {})
        self.config_file = str(config_file) if config_file is not None else None
        self.params = self._merge_params(params)

        self.system_msg: str = str(self.config.get("SYSTEM_MSG") or "")
        self.disable_system: bool = bool(self.config.get("DISABLE_SYSTEM", False))

        # Few-shot examples and/or conversation history, in the shared chat format.
        self.messages: List[Dict[str, Any]] = []
        self.last_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        self._client: Any = None

        load_llm_env()
        self._setup()

        if examples not in (None, ""):
            self.prepare(examples)

    ## Configuration ###################################################################################################

    @classmethod
    def from_config(
        cls,
        config_file: Union[str, Path],
        examples: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "BaseLLM":
        """Instantiate the backend from a YAML configuration file.

        The configuration file carries everything the model needs: its name, how to reach it, and
        the request parameters under ``LLM_CONFIG``. Those parameters are forwarded verbatim, so a
        model expecting ``max_completion_tokens`` and one expecting ``max_tokens`` differ only by
        their configuration file.

        Parameters
        ----------
        config_file : Union[str, Path]
            Path to the YAML configuration file.
        examples : Optional[Union[str, Path]], optional
            Folder with few-shot examples.
        **overrides : Any
            Values overriding the configuration, e.g. ``model="..."`` or
            ``params={"temperature": 0.2}`` (merged on top of ``LLM_CONFIG``).

        Returns
        -------
        BaseLLM
            A configured backend instance.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist or is not YAML.
        ValueError
            If the configuration does not name a model.
        """
        config = load_config_file(config_file)
        logger.info("LLM configuration file: %s", config_file)

        params = dict(config.get("LLM_CONFIG") or {})
        if not isinstance(config.get("LLM_CONFIG") or {}, dict):
            raise ValueError("LLM_CONFIG must be a dict when provided.")
        params.update(overrides.pop("params", None) or {})

        model = overrides.pop("model", None) or cls.model_from_config(config)
        if not model:
            raise ValueError(
                "Missing model name in config {}. Expected LLM_VERSION (or MODEL/MODEL_NAME).".format(
                    config_file
                )
            )

        return cls(
            model=model,
            params=params,
            config=config,
            config_file=config_file,
            examples=examples,
            **overrides,
        )

    @staticmethod
    def model_from_config(config: Dict[str, Any]) -> Optional[str]:
        """Return the model name declared by a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Parsed configuration.

        Returns
        -------
        Optional[str]
            The first of ``LLM_VERSION``, ``MODEL`` and ``MODEL_NAME`` that is set, or ``None``.
        """
        return config.get("LLM_VERSION") or config.get("MODEL") or config.get("MODEL_NAME")

    def _merge_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge configured parameters with the backend defaults and apply aliases.

        Parameters
        ----------
        params : Optional[Dict[str, Any]]
            Request parameters read from the configuration.

        Returns
        -------
        Dict[str, Any]
            :attr:`DEFAULT_PARAMS` updated with ``params``, with :attr:`PARAM_ALIASES` applied
            and empty values dropped.
        """
        merged: Dict[str, Any] = dict(self.DEFAULT_PARAMS)
        merged.update(params or {})

        for source, target in self.PARAM_ALIASES.items():
            if source in merged and target not in merged:
                merged[target] = merged.pop(source)

        return {key: value for key, value in merged.items() if not _is_empty_config_value(value)}

    def param(self, name: str, default: Any = None) -> Any:
        """Return a configured parameter, or ``default`` when it is not set.

        Parameters
        ----------
        name : str
            Parameter name.
        default : Any, optional
            Value returned when the parameter is missing.

        Returns
        -------
        Any
            The configured value, or ``default``.
        """
        return self.params.get(name, default)

    def request_params(self) -> Dict[str, Any]:
        """Return the parameters to forward to the provider request.

        Returns
        -------
        Dict[str, Any]
            The configured parameters without the :attr:`NON_REQUEST_PARAMS` keys.
        """
        return {
            key: value for key, value in self.params.items() if key not in self.NON_REQUEST_PARAMS
        }

    def _setup(self) -> None:
        """Read provider-specific keys from ``self.config``. Hook for subclasses."""
        return None

    ## Connection ######################################################################################################

    def connect(self) -> Any:
        """Return the live connection to the model, creating it on first use.

        For API backends this builds the SDK client; for local backends it loads the model.
        The result is cached, so repeated :meth:`query` calls reuse the same connection.

        Returns
        -------
        Any
            The provider client or the loaded local model handle.
        """
        if self._client is None:
            logger.info(
                "Connecting to %s model '%s'", self.PROVIDER or type(self).__name__, self.model
            )
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the provider client (or load the local model).

        Returns
        -------
        Any
            The provider client or the loaded local model handle.
        """

    @abstractmethod
    def _send(self, client: Any, messages: List[Dict[str, Any]]) -> Any:
        """Send a prepared message list and return the raw provider response.

        Parameters
        ----------
        client : Any
            The connection returned by :meth:`connect`.
        messages : List[Dict[str, Any]]
            Messages in the shared chat format.

        Returns
        -------
        Any
            The raw provider response.
        """

    @abstractmethod
    def _extract_text(self, response: Any) -> str:
        """Extract the assistant text from a raw provider response.

        Parameters
        ----------
        response : Any
            Raw provider response.

        Returns
        -------
        str
            The assistant answer.
        """

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage from a raw provider response.

        Parameters
        ----------
        response : Any
            Raw provider response.

        Returns
        -------
        Dict[str, int]
            Keys ``prompt_tokens`` and ``completion_tokens``; zeros when the provider does not
            report usage.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    def close(self) -> None:
        """Release the connection. Backends holding real resources should override this."""
        self._client = None

    ## Query ###########################################################################################################

    def query(
        self,
        message: str,
        images: Optional[Sequence[ImageInput]] = None,
        role: str = "user",
        max_retry: int = 3,
        end_when_error: bool = False,
        keep_history: bool = False,
    ) -> Tuple[bool, str]:
        """Connect to the model and send a query.

        Parameters
        ----------
        message : str
            The message sent to the model.
        images : Optional[Sequence[ImageInput]], optional
            Images to send along with the message, as ``PIL.Image.Image`` instances (paths are
            also accepted). Only backends with :attr:`SUPPORTS_IMAGES` set accept them.
        role : str, optional
            Role of the message. Defaults to ``"user"``.
        max_retry : int, optional
            How many times to attempt the request before giving up.
        end_when_error : bool, optional
            Stop at the first failure instead of retrying.
        keep_history : bool, optional
            Append the exchange to :attr:`messages`, so the next query continues the same
            conversation.

        Returns
        -------
        Tuple[bool, str]
            Whether the request succeeded, and the model's answer.

        Raises
        ------
        NotImplementedError
            If images are given to a text-only backend.
        """
        if images and not self.SUPPORTS_IMAGES:
            raise NotImplementedError("{} does not support images.".format(type(self).__name__))

        client = self.connect()
        messages = self.build_messages(message, images=images, role=role)

        attempts = max(1, int(max_retry))
        for attempt in range(1, attempts + 1):
            try:
                logger.info("Sending query to %s (attempt %d/%d)", self.model, attempt, attempts)
                response = self._send(client, messages)
                answer = self._extract_text(response)
                self.last_usage = self._extract_usage(response)
                logger.debug("LLM response: %s", answer)

                if keep_history:
                    self.messages = messages + [{"role": "assistant", "content": answer}]

                return True, answer
            except Exception as error:
                logger.error("LLM error: %s", error)
                if end_when_error:
                    break

        return False, ""

    def build_messages(
        self,
        message: str,
        images: Optional[Sequence[ImageInput]] = None,
        role: str = "user",
    ) -> List[Dict[str, Any]]:
        """Build the full message list for a query, including examples and system message.

        Parameters
        ----------
        message : str
            The message sent to the model.
        images : Optional[Sequence[ImageInput]], optional
            Images to attach to the message.
        role : str, optional
            Role of the message.

        Returns
        -------
        List[Dict[str, Any]]
            Messages in the shared chat format.
        """
        messages: List[Dict[str, Any]] = []

        if self.system_msg and not self.disable_system:
            messages.append({"role": "system", "content": self.system_msg})

        messages.extend(self.messages)
        messages.append({"role": role, "content": self.build_content(message, images)})

        return messages

    def build_content(
        self,
        message: str,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Any:
        """Build the content of a single message.

        Without images the content is the plain string. With images it becomes the list of parts
        expected by the provider, built through :meth:`image_part`.

        Parameters
        ----------
        message : str
            The text of the message.
        images : Optional[Sequence[ImageInput]], optional
            Images to attach.

        Returns
        -------
        Any
            Provider-ready message content.
        """
        if not images:
            return message

        parts: List[Any] = [{"type": "text", "text": message}]
        parts.extend(self.image_part(image) for image in images)
        return parts

    def image_part(self, image: ImageInput) -> Any:
        """Return a single image encoded the way the provider expects it.

        Parameters
        ----------
        image : ImageInput
            The image to encode.

        Returns
        -------
        Any
            A provider-specific content part.

        Raises
        ------
        NotImplementedError
            If the backend does not support images.
        """
        raise NotImplementedError("{} does not support images.".format(type(self).__name__))

    ## Examples ########################################################################################################

    def prepare(self, examples_dir: Union[str, Path]) -> None:
        """Load few-shot examples from a folder.

        The folder is expected to contain a ``main.yaml`` file. That file may declare a ``files``
        field listing further YAML files to include, and each of those may declare a ``files``
        field of its own, so examples can be split across a tree of files. Paths are resolved
        relative to the file declaring them. Loaded examples end up in :attr:`messages` and are
        sent ahead of every query.

        Parameters
        ----------
        examples_dir : Union[str, Path]
            Folder containing ``main.yaml``.

        Raises
        ------
        NotImplementedError
            Always, for the moment.
        """
        raise NotImplementedError(
            "prepare() is not implemented yet: few-shot examples from '{}' will not be loaded. "
            "It is meant to read <folder>/main.yaml and recursively include the files listed "
            "under its 'files' field.".format(examples_dir)
        )
