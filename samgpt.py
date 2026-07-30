import numpy as np
import torch
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import json

from segmentation.sam_model import *
from mapping.rgbd_mapper import *
from scene_understanding.gpt_annotator import GPTAnnotator

from utility.utility import logger

def convert(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


"""Main Function"""


def main(images, depth_path):

    load_dotenv()

    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_API_KEY")

    masks_dic = {}
    bboxes_dic = {}
    full_dict = {}

    sam = SAMModel("models/sam/sam_vit_h_4b8939.pth")

    endpoint = azure_endpoint
    model_name = "gpt-5.2-chat"
    deployment = "gpt-5.2-chat"

    subscription_key = azure_key
    api_version = "2024-12-01-preview"

    gpt = GPTAnnotator(endpoint, model_name, deployment, subscription_key, api_version)

    for f, image in enumerate(images):
        logger.info(f"Processing image {f + 1}/{len(images)}: {image}")
        rute = f"ppt_outputs/image{f + 1}"
        os.makedirs(rute, exist_ok=True)

        masked_rgb, mask_bin = sam.obtain_bg(image, f)
        rgb_masks, bboxes, masks_path = sam.individual_mask(mask_bin, masked_rgb, image, f)

        start_gpt = time.time()
        full_dict[f"Image_{f}"] = gpt.main_gpt(image, masks_path, bboxes)
        end_gpt = time.time()
        logger.info(f"GPT tagging and description for image {f + 1} obtained in {end_gpt - start_gpt}s")

        start_coords = time.time()
        full_dict[f"Image_{f}"] = main_coords(image, depth_path[f], full_dict[f"Image_{f}"])
        end_coords = time.time()
        logger.info(f"Coordinates and depth for image {f + 1} obtained in {end_coords - start_coords}s")

        # print(f"Image {f+1}: {full_dict[f"Image_{f}"]}")

        with open(f"outputs_json_labeled/output_img{f + 1}.json", "w") as k:
            json.dump(full_dict[f"Image_{f}"], k, indent=4, default=convert)


if __name__ == "__main__":
    start_all = time.time()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    rute = f"outputs_json_labeled"
    os.makedirs(rute, exist_ok=True)

    # images = ["dataset/rgb/rgb_dataset_1.png"]
    # depth = ["dataset/depth/depth_dataset_1.png"]

    path_img = Path.cwd() / "dataset/rgb"
    path_depth = Path.cwd() / "dataset/depth"

    images = sorted(path_img.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))
    depth = sorted(path_depth.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1]))

    main(images, depth)
    end_all = time.time()

    logger.info(f"Total time image process: {end_all - start_all}s")
