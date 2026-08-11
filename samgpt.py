import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from mapping.rgbd_mapper import main_coords
from scene_understanding.gpt_annotator import GPTAnnotator
from segmentation.sam_model import SAMModel
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
    if not args.no_env_file:
        load_dotenv(args.env_file)

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
    sam = SAMModel(
        # "models/sam/sam_b.pt",
        # "models/sam/sam_h.pt",
        # "models/sam/sam_l.pt",
        # "models/sam/sam2.1_l.pt",
        "models/sam/mobile_sam.pt",
        save_dir=Path(output_dir) / "segmentation_outputs",
        device=args.device,
        debug_masks=args.debug_masks,
        points_stride=48,
    )

    # Instantiate the GPT annotator
    model_name = "gpt-5.2-chat"
    deployment = "gpt-5.2-chat"
    api_version = "2024-12-01-preview"
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        logger.error("Azure endpoint is not set. Please check your environment variables.")
        sys.exit(1)
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not azure_key:
        logger.error("Azure API key is not set. Please check your environment variables.")
        sys.exit(1)
    gpt = GPTAnnotator(azure_endpoint, model_name, deployment, azure_key, api_version)

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

        # Annotate elements
        logger.debug("Starting GPT annotation...")
        image_dict = gpt.main_gpt(image, rgb_masks, bboxes)

        # Map coordinates and depth
        logger.debug("Starting coordinate and depth mapping...")
        image_dict = main_coords(image, str(depth_images[image_id]), image_dict)

        with open(f"{output_dir}/output_img{image_id + 1}.json", "w") as k:
            json.dump(image_dict, k, indent=4, default=np_array_to_list)
        break


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
