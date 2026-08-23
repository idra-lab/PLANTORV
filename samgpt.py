import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from LLM.llm_base import configure_env
from LLM.llm_factory import create_llm
from mapping.rgbd_mapper import main_coords
from scene_understanding.gpt_annotator import DEFAULT_LLM_CONFIG_FILE, GPTAnnotator
from segmentation.sam3_model import SAM3Model
from utility.utility import logger

np.set_printoptions(threshold=sys.maxsize)


def np_array_to_list(o: np.ndarray) -> list:
    """
    Convert a NumPy array to a list.

    Parameters
    ----------
    o : np.ndarray
        The NumPy array to be converted.

    Returns
    -------
    list
        The converted list if the input is a NumPy array; otherwise, returns the input object unchanged.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(args: argparse.Namespace) -> None:
    """
    Execute main function.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments parsed into a Namespace object.
    """
    # Also tells the LLM layer which file to read, so `--env-file` reaches the backends
    # instead of them falling back to the project root's `.env`.
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

    depth_path = Path(args.depth_dir)
    images_path = Path(args.images_dir)

    # Instantiate the segmentation model
    # sam = SAMModel(
    #     # "models/sam/sam_b.pt",
    #     # "models/sam/sam_h.pt",
    #     # "models/sam/sam_l.pt",
    #     # "models/sam/sam2.1_l.pt",
    #     "models/sam/mobile_sam.pt",
    #     save_dir=Path(output_dir) / "segmentation_outputs",
    #     device=args.device,
    #     debug_masks=args.debug_masks,
    #     points_stride=48,
    # )

    # sam = FastSAMModel(
    #     "models/fastsam/FastSAM-s.pt",
    #     save_dir=Path(output_dir) / "segmentation_outputs",
    #     device=args.device,
    #     debug_masks=args.debug_masks,
    # )

    llm = create_llm(args.llm_config)
    sam = SAM3Model(
        llm,
        Path(os.path.dirname(__file__)) / "models" / "sam" / "sam3.pt",
        save_dir=Path(output_dir) / "segmentation_outputs",
        device=args.device,
        debug_masks=args.debug_masks,
        examples_file=Path(os.path.dirname(__file__))
        / "LLM"
        / "examples"
        / "SAM3"
        / "concept_prompts.yaml",
        # task="The task considers the structures as wholes and not as individual parts.",
        # concepts=["robotic arm", "long blue object", "red structure", "yellow structure", "tall green tower"],
        # max_refinements=3,  # parked: segment() no longer calls the refinement loop
        imgsz=1036,
        conf=0.2,
        iou=0.1,
    )

    # Instantiate the annotator. The YAML file selects the model, the endpoint, the credentials
    # and the request parameters, so switching model means pointing --llm-config elsewhere.
    try:
        gpt = GPTAnnotator.from_config(args.llm_config)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Unable to configure the annotator: {error}")
        sys.exit(1)

    images = sorted(images_path.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))
    depth_images = sorted(depth_path.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))
    for image_id, image_path in enumerate(images):
        logger.info(f"Processing image {image_id + 1}/{len(images)}: {image_path}")

        # Open image
        image = Image.open(image_path)

        # Segment the image
        logger.debug("Starting segmentation...")
        masked_rgb, mask_bin = sam.obtain_bg(image, image_id)
        rgb_masks, bboxes = sam.individual_mask(image, mask_bin, masked_rgb, image_id)

        # Shows the masks that were actually kept.
        if args.view_masks and sam.visualize is not None:
            sam.visualize(image, image_id)

        # Annotate elements
        logger.debug("Starting GPT annotation...")
        image_dict = gpt.main_gpt(image, rgb_masks, bboxes)

        # Map coordinates and depth
        logger.debug("Starting coordinate and depth mapping...")
        image_dict = main_coords(image, str(depth_images[image_id]), image_dict)

        with open(f"{output_dir}/output_img{image_id + 1}.json", "w") as k:
            json.dump(image_dict, k, indent=4, default=np_array_to_list)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "--images-dir", type=str, default="dataset/rgb", help="Path to the images directory"
    )
    parser.add_argument(
        "--depth-dir", type=str, default="dataset/depth", help="Path to the depth images directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Path to the output directory"
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=str(DEFAULT_LLM_CONFIG_FILE),
        help="Path to the LLM YAML configuration file used for annotation (see LLM/conf)",
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

    return parser.parse_args()


if __name__ == "__main__":
    start_all = time.time()

    main(parse_arguments())
    end_all = time.time()

    logger.info(f"Total time image process: {end_all - start_all}s")
