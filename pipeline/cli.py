"""Command-line options and start-up shared by the three stages.

Each stage is its own program, but they agree on where the environment file is,
which device they run on, and how a bad path is reported. The option groups below
are the single definition of those flags: the stage scripts add the groups they
need, and ``samgpt.py`` adds all of them to run the three in a row.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union

import torch

from LLM.llm_base import configure_env
from mapping.depth_anything import DEFAULT_FOCAL_LENGTH_PX
from mapping.rgbd_mapper import DEFAULT_DEPTH_ASSOCIATION, DEPTH_ASSOCIATIONS
from scene_understanding.dam_annotator import DAM_QUERY, DEFAULT_DAM_MODEL_PATH
from scene_understanding.gpt_annotator import DEFAULT_LLM_CONFIG_FILE
from utility.utility import logger

# Values of `--depth-source` that replace the depth dataset with an inferred depth map.
# `sensor` is the remaining value and reads the depth images from `--depth-dir`.
ESTIMATED_DEPTH_SOURCES = ("depth-anything-v2", "monocular")

# The segmentation configuration used when --segmenter-config is not given. It picks
# the backend, its checkpoint, what is forwarded to it and the thresholds that decide
# which masks survive; see segmentation/conf.
DEFAULT_SEGMENTER_CONFIG_FILE = (
    Path(__file__).resolve().parent.parent / "segmentation" / "conf" / "sam21_l.yaml"
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options every stage understands.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser the options are added to.
    """
    parser.add_argument("--env-file", type=str, default=".env", help="Path to the environment file")
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help="Flag to indicate not to load the environment file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')",
    )


def add_segmentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of the segmentation stage.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser the options are added to.
    """
    group = parser.add_argument_group("segmentation")
    group.add_argument(
        "--segmenter-config",
        type=str,
        default=str(DEFAULT_SEGMENTER_CONFIG_FILE),
        help=(
            "Path to the segmentation YAML configuration file (see segmentation/conf). It "
            "names the backend, its checkpoint, the arguments forwarded to it and the "
            "thresholds deciding which masks are kept (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--view-masks",
        action="store_true",
        help=(
            "After segmenting, write an HTML page of the masks that were kept to "
            "<output-dir>/segmentation_outputs/ and open it in a browser"
        ),
    )
    group.add_argument(
        "--debug-masks",
        action="store_true",
        help=(
            "Save every mask SAM produces, before filtering, to "
            "<output-dir>/segmentation_outputs/debug/ and log why each mask was kept or dropped"
        ),
    )


def add_annotation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of the annotation stage.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser the options are added to.
    """
    group = parser.add_argument_group("annotation")
    group.add_argument(
        "--annotator",
        choices=("gpt", "dam"),
        default="gpt",
        help=(
            "Which annotator describes the objects: 'gpt' sends the crops to the remote "
            "model named by --llm-config, 'dam' runs Describe Anything locally over the "
            "masks (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--llm-config",
        type=str,
        default=str(DEFAULT_LLM_CONFIG_FILE),
        help="Path to the LLM YAML configuration file used for annotation (see LLM/conf)",
    )

    # Describe Anything. This is the counterpart of --llm-config: there the model is
    # picked by a YAML file naming a remote endpoint, here it is a local checkpoint,
    # so the parameters that the YAML would carry are flags instead.
    group.add_argument(
        "--dam-model",
        type=str,
        default=str(DEFAULT_DAM_MODEL_PATH),
        help=(
            "Directory holding the DAM checkpoint, or a Hugging Face repository id. "
            "Download the default with `python3 scripts/install_models.py dam_3b` "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--dam-device",
        choices=("cpu", "cuda"),
        help=(
            "Device for the DAM model. Independent of --device; falls back to it "
            "when unset. DAM only generates on CUDA"
        ),
    )
    group.add_argument(
        "--dam-query",
        type=str,
        default=DAM_QUERY,
        help=(
            "Instructions sent with every masked region. Must contain the <image> "
            "token, and should ask for '<object>, <description>' so the answer can "
            "be split into a tag and a description"
        ),
    )
    group.add_argument(
        "--dam-temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for DAM (default: %(default)s)",
    )
    group.add_argument(
        "--dam-top-p",
        type=float,
        default=0.5,
        help="Nucleus sampling cutoff for DAM (default: %(default)s)",
    )
    group.add_argument(
        "--dam-max-new-tokens",
        type=int,
        default=512,
        help="Longest description DAM may generate, in tokens (default: %(default)s)",
    )


def add_depth_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of the depth stage.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser the options are added to.
    """
    group = parser.add_argument_group("depth")
    group.add_argument(
        "--depth-dir",
        type=str,
        default="dataset/depth",
        help="Path to the depth images directory, read by --depth-source sensor",
    )
    # Depth backend. `sensor` reads the images in --depth-dir; the other two infer depth
    # from the RGB frame instead, which makes --depth-dir unused.
    group.add_argument(
        "--depth-source",
        choices=("sensor", *ESTIMATED_DEPTH_SOURCES),
        default="sensor",
        help="Where depth comes from (default: %(default)s)",
    )
    group.add_argument(
        "--depth-association",
        choices=DEPTH_ASSOCIATIONS,
        default=DEFAULT_DEPTH_ASSOCIATION,
        help=(
            "How one depth is taken per object: 'bbox-center' samples the centre of the "
            "bounding box, 'mask-median' takes the median of the valid depths under the "
            "object's segmentation mask (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--depth-model",
        type=str,
        help="Checkpoint ID (defaults depend on --depth-source)",
    )
    group.add_argument(
        "--depth-device",
        choices=("cpu", "cuda"),
        help=(
            "Device for the depth model. Independent of --device; the provider "
            "chooses on its own when unset"
        ),
    )
    group.add_argument(
        "--depth-focal-length-px",
        type=float,
        default=DEFAULT_FOCAL_LENGTH_PX,
        help="Mean RGB focal length for monocular metric scaling (default: %(default)s)",
    )
    group.add_argument(
        "--depth-process-res",
        type=int,
        default=504,
        help="Depth Anything 3 processing resolution (default: %(default)s)",
    )


def require_dir(path: Union[str, Path], what: str) -> None:
    """Exit unless a directory exists.

    Parameters
    ----------
    path : Union[str, Path]
        Path that has to be an existing directory.
    what : str
        How the path is named in the error message.
    """
    directory = Path(path)
    if not directory.exists():
        logger.error(f"{what} does not exist: {directory}")
        sys.exit(1)
    if not directory.is_dir():
        logger.error(f"{what} is not a directory: {directory}")
        sys.exit(1)


def require_file(path: Union[str, Path], what: str) -> None:
    """Exit unless a file exists.

    Parameters
    ----------
    path : Union[str, Path]
        Path that has to be an existing file.
    what : str
        How the path is named in the error message.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.error(f"{what} does not exist: {file_path}")
        sys.exit(1)
    if not file_path.is_file():
        logger.error(f"{what} is not a file: {file_path}")
        sys.exit(1)


def validate_device(device: str) -> None:
    """Exit unless the requested device can be used.

    Parameters
    ----------
    device : str
        Value of ``--device``.
    """
    if device not in ("cuda", "cpu"):
        logger.error(f"Invalid device specified: {device}. Must be 'cuda' or 'cpu'.")
        sys.exit(1)
    if device == "cuda" and not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Please check your PyTorch installation and GPU configuration."
        )
        sys.exit(1)


def setup(args: argparse.Namespace) -> None:
    """Load the environment file and prepare the device.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, holding at least ``env_file``, ``no_env_file`` and ``device``.
    """
    # Also tells the LLM layer which file to read, so `--env-file` reaches the backends
    # instead of them falling back to the project root's `.env`.
    configure_env(args.env_file, load=not args.no_env_file)

    validate_device(args.device)

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
