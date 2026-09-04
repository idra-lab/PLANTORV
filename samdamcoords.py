import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from LLM.llm_base import configure_env
from mapping.depth_anything import (
    DEFAULT_FOCAL_LENGTH_PX,
    DEFAULT_MODEL_ID,
    DEFAULT_V3_MODEL_ID,
    DepthAnythingV2Provider,
    DepthAnythingV3Provider,
)
from mapping.depth_provider import DepthProvider
from mapping.rgbd_mapper import attach_object_depths, main_coords
from scene_understanding.dam_annotator import (
    DAM_QUERY,
    DEFAULT_DAM_MODEL_PATH,
    DAMAnnotator,
    looks_like_repo_id,
)
from segmentation.sam_model import SAMModel
from utility.json_serialization import to_json_compatible as convert
from utility.utility import logger

np.set_printoptions(threshold=sys.maxsize)

# Values of `--depth-source` that replace the depth dataset with an inferred depth map.
# `sensor` is the remaining value and reads the depth images from `--depth-dir`.
ESTIMATED_DEPTH_SOURCES = ("depth-anything-v2", "monocular")


def build_depth_provider(args: argparse.Namespace) -> DepthProvider | None:
    """
    Build the monocular depth backend selected by ``--depth-source``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    DepthProvider or None
        The provider to estimate depth with, or None for ``--depth-source sensor``,
        in which case the depth images in ``--depth-dir`` are used instead.
    """
    if args.depth_source == "depth-anything-v2":
        return DepthAnythingV2Provider(
            model_id=args.depth_model or DEFAULT_MODEL_ID,
            device=args.depth_device,
        )
    if args.depth_source == "monocular":
        return DepthAnythingV3Provider(
            model_id=args.depth_model or DEFAULT_V3_MODEL_ID,
            focal_length_px=args.depth_focal_length_px,
            process_res=args.depth_process_res,
            device=args.depth_device,
        )
    return None


def main(args: argparse.Namespace) -> None:
    """
    Execute main function.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments parsed into a Namespace object.
    """
    # DAM needs no credentials of its own, but the depth backends pull their
    # checkpoints from Hugging Face, so the token in the environment file still
    # has to reach them.
    configure_env(args.env_file, load=not args.no_env_file)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Please check your PyTorch installation and GPU configuration."
        )
        sys.exit(1)

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    images_path = Path(args.images_dir)

    # Instantiate the segmentation model. Unlike samgpt.py this shares the GPU with
    # a 7.1 GB annotator rather than with a remote one, so the default is the small
    # encoder: DAM-3B plus sam_l does not fit on a 12 GB card. Swap it for one of
    # the larger checkpoints below when there is memory to spare.
    sam = SAMModel(
        # "models/sam/sam_b.pt",
        # "models/sam/sam_h.pt",
        # "models/sam/sam_l.pt",
        # "models/sam/sam2.1_l.pt",
        "models/sam/mobile_sam.pt",
        save_dir=Path(output_dir) / "segmentation_outputs",
        device=args.device,
        debug_masks=args.debug_masks,
        points_stride=12,
    )

    # sam = FastSAMModel(
    #     "models/fastsam/FastSAM-s.pt",
    #     save_dir=Path(output_dir) / "segmentation_outputs",
    #     device=args.device,
    #     debug_masks=args.debug_masks,
    # )

    # Instantiate the annotator. Where samgpt.py sends crops to a remote LLM, DAM
    # runs locally and describes a masked region, so it is given `sam.last_masks`
    # further down rather than the crops.
    try:
        dam = DAMAnnotator(
            args.dam_model,
            query=args.dam_query,
            device=args.dam_device or args.device,
            temperature=args.dam_temperature,
            top_p=args.dam_top_p,
            max_new_tokens=args.dam_max_new_tokens,
        )
    except (FileNotFoundError, ImportError) as error:
        logger.error(f"Unable to configure the annotator: {error}")
        sys.exit(1)

    # Depth comes either from the dataset in `--depth-dir` or from a monocular
    # estimator, so only one of the two is prepared.
    depth_provider = build_depth_provider(args)
    if depth_provider is None:
        depth_images = sorted(
            Path(args.depth_dir).glob("*.png"), key=lambda x: int(x.stem.split("_")[-1])
        )
    else:
        depth_images = []
        logger.info(f"Estimating depth with {type(depth_provider).__name__}")

    images = sorted(images_path.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))
    for image_id, image_path in enumerate(images):
        logger.info(f"Processing image {image_id + 1}/{len(images)}: {image_path}")

        # Open image
        image = Image.open(image_path)

        # Segment the image
        logger.debug("Starting segmentation...")
        start_segmentation = time.time()
        masked_rgb, mask_bin = sam.obtain_bg(image, image_id)
        rgb_masks, bboxes = sam.individual_mask(image, mask_bin, masked_rgb, image_id)
        logger.debug(f"Segmentation done in {time.time() - start_segmentation}s")

        # Shows the masks that were actually kept. Only SAM3Model ships a viewer, so this
        # stays an attribute lookup rather than a direct call: the model above is swappable.
        visualize = getattr(sam, "visualize", None)
        if args.view_masks and visualize is not None:
            visualize(image, image_id)

        # Annotate elements. DAM reads `masks` rather than the crops: `last_masks`
        # holds the binary mask of each object `individual_mask` kept, in the same
        # order as the crops it returned.
        logger.debug("Starting DAM annotation...")
        start_dam = time.time()
        image_dict = dam.annotate(image, rgb_masks, bboxes, masks=sam.last_masks)
        logger.debug(f"DAM tagging and description done in {time.time() - start_dam}s")

        # Map coordinates and depth
        logger.debug("Starting coordinate and depth mapping...")
        start_coords = time.time()
        if depth_provider is None:
            image_dict = main_coords(image, str(depth_images[image_id]), image_dict)
        else:
            # The providers work on the BGR array OpenCV produces, not on the PIL image.
            color_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if color_bgr is None:
                raise FileNotFoundError(f"RGB image not found at path: {image_path}")
            depth_result = depth_provider.estimate(color_bgr)
            image_dict = attach_object_depths(image_dict, depth_result.depth_mm)
        logger.debug(f"Coordinates and depth done in {time.time() - start_coords}s")

        with open(f"{output_dir}/output_img{image_id + 1}.json", "w") as k:
            json.dump(image_dict, k, indent=4, default=convert)

    dam.close()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Segment images, describe the objects with Describe Anything, and map depth."
    )
    parser.add_argument(
        "--images-dir", type=str, default="dataset/rgb", help="Path to the images directory"
    )
    parser.add_argument(
        "--depth-dir", type=str, default="dataset/depth", help="Path to the depth images directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Path to the output directory"
    )
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

    parser.add_argument(
        "--view-masks",
        action="store_true",
        help=(
            "After segmenting, write an HTML page of the masks that were kept to "
            "<output-dir>/segmentation_outputs/ and open it in a browser"
        ),
    )

    parser.add_argument(
        "--debug-masks",
        action="store_true",
        help=(
            "Save every mask SAM produces, before filtering, to "
            "<output-dir>/segmentation_outputs/debug/ and log why each mask was kept or dropped"
        ),
    )

    # Describe Anything. This is the counterpart of samgpt.py's --llm-config: there
    # the model is picked by a YAML file naming a remote endpoint, here it is a local
    # checkpoint, so the parameters that the YAML would carry are flags instead.
    parser.add_argument(
        "--dam-model",
        type=str,
        default=str(DEFAULT_DAM_MODEL_PATH),
        help=(
            "Directory holding the DAM checkpoint, or a Hugging Face repository id. "
            "Download the default with `python3 scripts/install_models.py dam_3b` "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dam-device",
        choices=("cpu", "cuda"),
        help=(
            "Device for the DAM model. Independent of --device; falls back to it "
            "when unset. DAM only generates on CUDA"
        ),
    )
    parser.add_argument(
        "--dam-query",
        type=str,
        default=DAM_QUERY,
        help=(
            "Instructions sent with every masked region. Must contain the <image> "
            "token, and should ask for '<object>, <description>' so the answer can "
            "be split into a tag and a description"
        ),
    )
    parser.add_argument(
        "--dam-temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for DAM (default: %(default)s)",
    )
    parser.add_argument(
        "--dam-top-p",
        type=float,
        default=0.5,
        help="Nucleus sampling cutoff for DAM (default: %(default)s)",
    )
    parser.add_argument(
        "--dam-max-new-tokens",
        type=int,
        default=512,
        help="Longest description DAM may generate, in tokens (default: %(default)s)",
    )

    # Depth backend. `sensor` reads the images in --depth-dir; the other two infer depth
    # from the RGB frame instead, which makes --depth-dir unused.
    parser.add_argument(
        "--depth-source",
        choices=("sensor", *ESTIMATED_DEPTH_SOURCES),
        default="sensor",
        help="Where depth comes from (default: %(default)s)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        help="Checkpoint ID (defaults depend on --depth-source)",
    )
    parser.add_argument(
        "--depth-device",
        choices=("cpu", "cuda"),
        help=(
            "Device for the depth model. Independent of --device; the provider "
            "chooses on its own when unset"
        ),
    )
    parser.add_argument(
        "--depth-focal-length-px",
        type=float,
        default=DEFAULT_FOCAL_LENGTH_PX,
        help="Mean RGB focal length for monocular metric scaling (default: %(default)s)",
    )
    parser.add_argument(
        "--depth-process-res",
        type=int,
        default=504,
        help="Depth Anything 3 processing resolution (default: %(default)s)",
    )

    args = parser.parse_args()

    if not Path(args.images_dir).exists():
        logger.error(f"Images directory does not exist: {args.images_dir}")
        sys.exit(1)
    if not Path(args.images_dir).is_dir():
        logger.error(f"Images path is not a directory: {args.images_dir}")
        sys.exit(1)

    # Only checked for the sensor backend: the estimators never read --depth-dir, so
    # requiring it would force a depth dataset to exist for a run that ignores it.
    if args.depth_source == "sensor":
        if not Path(args.depth_dir).exists():
            logger.error(f"Depth images directory does not exist: {args.depth_dir}")
            sys.exit(1)
        if not Path(args.depth_dir).is_dir():
            logger.error(f"Depth images path is not a directory: {args.depth_dir}")
            sys.exit(1)

    # A repository id such as "nvidia/DAM-3B" is resolved by DAM itself, so only a
    # path is checked here.
    if not looks_like_repo_id(args.dam_model) and not Path(args.dam_model).exists():
        logger.error(
            f"DAM checkpoint does not exist: {args.dam_model}. Download it with "
            "`python3 scripts/install_models.py dam_3b`."
        )
        sys.exit(1)

    if "<image>" not in args.dam_query:
        logger.error("The DAM query must contain the <image> token.")
        sys.exit(1)

    if args.device not in ["cuda", "cpu"]:
        logger.error(f"Invalid device specified: {args.device}. Must be 'cuda' or 'cpu'.")
        sys.exit(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Please check your PyTorch installation and GPU configuration."
        )
        sys.exit(1)

    return args


if __name__ == "__main__":
    start_all = time.time()
    main(parse_arguments())
    end_all = time.time()

    logger.info(f"Total time image process: {end_all - start_all}s")
