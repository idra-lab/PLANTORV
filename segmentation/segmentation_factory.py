"""Build the segmentation backend a YAML configuration file asks for.

The counterpart of :mod:`LLM.llm_factory` for the segmentation side. A file in
``segmentation/conf`` carries everything that decides what a run produces -- the
backend, its checkpoint, the arguments forwarded to it and the thresholds that
decide which masks are kept -- so swapping model means pointing
``segmentation.py --segmenter-config`` at another file.

Recognised keys::

    MODEL               Backend: "sam", "fastsam" or "sam3". Inferred from
                        CHECKPOINT when absent.
    CHECKPOINT          Path of the weights.
    SEGMENTATION_CONFIG Arguments forwarded to the backend as-is (points_stride for
                        SAM, conf/iou/imgsz for FastSAM and SAM 3).
    FILTERS             Overrides of the backend's DEFAULT_FILTERS.

SAM 3 reads three more, since a VLM names the concepts it segments::

    LLM_CONFIG_FILE     The LLM YAML configuration of that VLM.
    EXAMPLES_FILE       Phrasing examples steering the concepts.
    MERGE_GAP, TASK, CONCEPTS   Passed to SAM3Model as they are.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from LLM.llm_base import load_config_file
from LLM.llm_factory import create_llm
from segmentation.segmentation import SegmentationModel
from utility.utility import logger

# Accepted spellings of MODEL, mapped to the canonical name.
MODEL_ALIASES = {
    "sam": "sam",
    "sam1": "sam",
    "sam2": "sam",
    "sam21": "sam",
    "sam_2": "sam",
    "mobile_sam": "sam",
    "mobilesam": "sam",
    "fastsam": "fastsam",
    "fast_sam": "fastsam",
    "sam3": "sam3",
    "sam_3": "sam3",
}

# Checkpoint names that pick a backend on their own, checked as suffixes so that a
# path works as well as a bare file name. SAM is the fallback, which is what
# SAMModel's own checkpoint check assumes.
CHECKPOINT_MARKERS = (("fastsam", "fastsam"), ("sam3.pt", "sam3"))

# Keys that are neither forwarded parameters nor filters, per backend.
SAM3_CONSTRUCTOR_KEYS = ("MERGE_GAP", "TASK", "CONCEPTS", "EXAMPLES_FILE")


def normalize_model(model: str) -> str:
    """Normalize a backend name to its canonical spelling.

    Parameters
    ----------
    model : str
        Value of ``MODEL``, in any of its accepted spellings.

    Returns
    -------
    str
        ``"sam"``, ``"fastsam"`` or ``"sam3"``.

    Raises
    ------
    ValueError
        If the backend is unknown.
    """
    normalized = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]

    raise ValueError(
        f"Unsupported segmentation backend {model!r}. "
        f"Expected one of: {', '.join(sorted(set(MODEL_ALIASES.values())))}."
    )


def infer_model(config: Dict[str, Any]) -> str:
    """Infer the backend from a loaded configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed configuration.

    Returns
    -------
    str
        Canonical backend name.

    Raises
    ------
    ValueError
        If the configuration names neither a backend nor a checkpoint.
    """
    explicit = config.get("MODEL") or config.get("SEGMENTER")
    if explicit not in (None, ""):
        return normalize_model(str(explicit))

    checkpoint = str(config.get("CHECKPOINT") or "").lower()
    if not checkpoint:
        raise ValueError("The configuration names neither MODEL nor CHECKPOINT.")

    for marker, model in CHECKPOINT_MARKERS:
        if marker in Path(checkpoint).name:
            return model

    return "sam"


def resolve_class(model: str) -> Type[SegmentationModel]:
    """Import and return the backend class.

    Imported here rather than at module scope: FastSAM and SAM 3 pull dependencies a
    run of the other backend has no use for.

    Parameters
    ----------
    model : str
        Backend name, in any of its accepted spellings.

    Returns
    -------
    Type[SegmentationModel]
        The backend class.
    """
    name = normalize_model(model)

    if name == "fastsam":
        from segmentation.fastsam_model import FastSAMModel

        return FastSAMModel

    if name == "sam3":
        from segmentation.sam3_model import SAM3Model

        return SAM3Model

    from segmentation.sam_model import SAMModel

    return SAMModel


def create_segmenter(
    config_file: Union[str, Path],
    *,
    save_dir: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    debug_masks: bool = False,
    **overrides: Any,
) -> SegmentationModel:
    """Build the backend described by a configuration file.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML configuration file.
    save_dir : Union[str, Path] or None
        Directory the backend writes its own figures and debug dumps to.
    device : str
        Device the model runs on. A property of the machine rather than of the
        configuration, so it comes from the caller.
    debug_masks : bool
        Whether to dump every mask before filtering.
    **overrides : Any
        Forwarded to the backend constructor, after the configuration.

    Returns
    -------
    SegmentationModel
        A configured backend.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the backend cannot be determined, or a filter is not one it declares.
    """
    config = load_config_file(config_file)
    logger.info(f"Segmentation configuration file: {config_file}")

    model = infer_model(config)
    backend = resolve_class(model)

    checkpoint = config.get("CHECKPOINT")
    if not checkpoint:
        raise ValueError(f"Missing CHECKPOINT in {config_file}.")

    params = dict(config.get("SEGMENTATION_CONFIG") or {})
    if not isinstance(config.get("SEGMENTATION_CONFIG") or {}, dict):
        raise ValueError("SEGMENTATION_CONFIG must be a dict when provided.")

    filters = dict(config.get("FILTERS") or {})
    if not isinstance(config.get("FILTERS") or {}, dict):
        raise ValueError("FILTERS must be a dict when provided.")

    kwargs: Dict[str, Any] = {
        "device": device,
        "save_dir": save_dir,
        "debug_masks": debug_masks,
        "filters": filters,
        **params,
    }

    # Every backend takes its checkpoint first, under a name of its own
    # (``sam_checkpoint``, ``fastsam_checkpoint``, ``sam3_checkpoint``), so it is passed
    # positionally. SAM 3 takes the VLM ahead of it.
    positional: tuple = (checkpoint,)

    if model == "sam3":
        # SAM 3 segments what a VLM names, so its configuration carries the
        # configuration of that VLM rather than a concept list.
        llm_config = config.get("LLM_CONFIG_FILE")
        if not llm_config:
            raise ValueError(f"Missing LLM_CONFIG_FILE in {config_file}, required by SAM 3.")
        positional = (create_llm(llm_config), checkpoint)

        for key in SAM3_CONSTRUCTOR_KEYS:
            if config.get(key) is not None:
                kwargs[key.lower()] = config[key]

    kwargs.update(overrides)

    logger.info(f"Segmentation backend {backend.__name__}, checkpoint {checkpoint}")

    return backend(*positional, **kwargs)


def describe(model: SegmentationModel, config_file: Union[str, Path]) -> Dict[str, Any]:
    """Describe a built backend, for the record written next to a run.

    Reads the model rather than the file, so the defaults its constructor filled in
    are part of the answer.

    Parameters
    ----------
    model : SegmentationModel
        The backend, as returned by :func:`create_segmenter`.
    config_file : Union[str, Path]
        The configuration it was built from.

    Returns
    -------
    Dict[str, Any]
        The effective configuration, in the shape of the file it came from.
    """
    described: Dict[str, Any] = {
        "SOURCE_CONFIG": str(config_file),
        "MODEL": infer_model(load_config_file(config_file)),
        "BACKEND": type(model).__name__,
        "SEGMENTATION_CONFIG": model.effective_params(),
        "FILTERS": dict(model.filters),
    }

    llm = getattr(model, "llm", None)
    if llm is not None:
        described["LLM_CONFIG_FILE"] = llm.config_file
        described["LLM_VERSION"] = llm.model

    return described


def list_config_files(config_dir: Optional[Union[str, Path]] = None) -> list[str]:
    """List the segmentation configuration files of a directory.

    Parameters
    ----------
    config_dir : Union[str, Path] or None
        Directory to scan. Defaults to ``segmentation/conf``.

    Returns
    -------
    list[str]
        Sorted configuration file paths.
    """
    directory = Path(config_dir) if config_dir else Path(__file__).parent / "conf"

    return sorted(str(path) for path in directory.glob("*.yaml"))
