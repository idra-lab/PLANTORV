import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import cv2
import torch
from dotenv import load_dotenv

from mapping.depth_anything import (
    DEFAULT_FOCAL_LENGTH_PX,
    DEFAULT_MODEL_ID,
    DEFAULT_V3_MODEL_ID,
    DepthAnythingV2Provider,
    DepthAnythingV3Provider,
)
from mapping.depth_provider import DepthProvider
from mapping.rgbd_mapper import attach_object_depths, main_coords
from scene_understanding.gpt_annotator import GPTAnnotator
from segmentation.sam_model import SAMModel
from utility.json_serialization import to_json_compatible as convert
from utility.utility import logger

"""Main Function"""


def main(
    images: Sequence[Path],
    depth_path: Sequence[Path] | None = None,
    depth_provider: DepthProvider | None = None,
) -> None:

    load_dotenv()

    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_API_KEY")

    full_dict = {}

    sam = SAMModel("models/sam/sam_vit_h_4b8939.pth")

    endpoint = azure_endpoint
    model_name = "gpt-5.2-chat"
    deployment = "gpt-5.2-chat"

    subscription_key = azure_key
    api_version = "2024-12-01-preview"

    if endpoint is None or subscription_key is None:
        raise RuntimeError("AZURE_ENDPOINT and AZURE_API_KEY must be configured")
    gpt = GPTAnnotator(endpoint, model_name, deployment, subscription_key, api_version)

    for f, image in enumerate(images):
        logger.info(f"Processing image {f + 1}/{len(images)}: {image}")
        image_path = str(image)
        rute = f"ppt_outputs/image{f + 1}"
        os.makedirs(rute, exist_ok=True)

        masked_rgb, mask_bin = sam.obtain_bg(image_path, f)
        rgb_masks, bboxes, masks_path = sam.individual_mask(mask_bin, masked_rgb, image_path, f)

        start_gpt = time.time()
        full_dict[f"Image_{f}"] = gpt.main_gpt(
            image_path, [Path(path) for path in masks_path], bboxes
        )
        end_gpt = time.time()
        logger.info(
            f"GPT tagging and description for image {f + 1} obtained in {end_gpt - start_gpt}s"
        )

        start_coords = time.time()
        if depth_provider is None:
            if depth_path is None:
                raise ValueError("Sensor depth mode requires depth paths")
            full_dict[f"Image_{f}"] = main_coords(
                image_path, str(depth_path[f]), full_dict[f"Image_{f}"]
            )
        else:
            color_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if color_bgr is None:
                raise FileNotFoundError(f"RGB image not found at path: {image}")
            depth_result = depth_provider.estimate(color_bgr)
            full_dict[f"Image_{f}"] = attach_object_depths(
                full_dict[f"Image_{f}"], depth_result.depth_mm
            )
        end_coords = time.time()
        logger.info(
            f"Coordinates and depth for image {f + 1} obtained in {end_coords - start_coords}s"
        )

        # print(f"Image {f+1}: {full_dict[f"Image_{f}"]}")

        with open(f"outputs_json_labeled/output_img{f + 1}.json", "w") as k:
            json.dump(full_dict[f"Image_{f}"], k, indent=4, default=convert)


def parse_args() -> argparse.Namespace:
    """Parse depth-backend options for the main pipeline."""
    parser = argparse.ArgumentParser(description="Run the PLANTORV image pipeline")
    parser.add_argument(
        "--depth-source",
        choices=("sensor", "depth-anything-v2", "monocular"),
        default="sensor",
    )
    parser.add_argument(
        "--depth-model",
        help="Checkpoint ID (defaults depend on --depth-source)",
    )
    parser.add_argument("--depth-device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--depth-focal-length-px",
        type=float,
        default=DEFAULT_FOCAL_LENGTH_PX,
        help="Mean RGB focal length for monocular metric scaling",
    )
    parser.add_argument(
        "--depth-process-res",
        type=int,
        default=504,
        help="Depth Anything 3 processing resolution",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_all = time.time()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    rute = "outputs_json_labeled"
    os.makedirs(rute, exist_ok=True)

    # images = ["dataset/rgb/rgb_dataset_1.png"]
    # depth = ["dataset/depth/depth_dataset_1.png"]

    path_img = Path.cwd() / "dataset/rgb"
    path_depth = Path.cwd() / "dataset/depth"

    images = sorted(path_img.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))
    depth = sorted(path_depth.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))

    provider = None
    if args.depth_source == "depth-anything-v2":
        provider = DepthAnythingV2Provider(
            model_id=args.depth_model or DEFAULT_MODEL_ID,
            device=args.depth_device,
        )
    elif args.depth_source == "monocular":
        provider = DepthAnythingV3Provider(
            model_id=args.depth_model or DEFAULT_V3_MODEL_ID,
            focal_length_px=args.depth_focal_length_px,
            process_res=args.depth_process_res,
            device=args.depth_device,
        )
    main(images, depth if provider is None else None, depth_provider=provider)
    end_all = time.time()

    logger.info(f"Total time image process: {end_all - start_all}s")
