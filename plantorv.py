"""Run the full RGB segmentation, VLM tagging, and depth-coordinate pipeline."""

import os
import time

import dotenv
import torch

from depth.femto_camera import main_coords
from segmentation.sam import SAMModel
from vlm.gpt import GPTModel


def main(images, depth_path):
    """Process RGB/depth image pairs through the Plantor pipeline.

    Parameters
    ----------
    images : list[str]
        Paths to RGB images.
    depth_path : list[str]
        Paths to depth images aligned by index with ``images``.
    """
    dotenv.load_dotenv()

    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_KEY")

    masks_dic = {}
    bboxes_dic = {}
    full_dict = {}

    sam = SAMModel("segmentation/checkpoints/sam/sam_vit_h_4b8939.pth", model_type="vit_h")
    # sam = SAMModel("segmentation/checkpoints/sam/sam_vit_b_01ec64.pth", model_type="vit_b")

    endpoint = azure_endpoint
    model_name = "gpt-5.2-chat"
    deployment = "gpt-5.2-chat"

    subscription_key = azure_key
    api_version = "2024-12-01-preview"

    gpt = GPTModel(endpoint, model_name, deployment, subscription_key, api_version)

    for f, image in enumerate(images):
        # Remove the background
        masked_rgb, mask_bin = sam.remove_bg(image, f, save_dir="output")
        masks, bboxes, masks_path = sam.individual_mask(mask_bin, masked_rgb, image, f, save_dir="output")
        break

        start_gpt = time.time()
        full_dict[f"Image_{f}"] = gpt.main_gpt(image, masks_path, masks, bboxes)
        end_gpt = time.time()
        print(f"GPT tagging and description for image {f + 1} obtained in {end_gpt - start_gpt}s")
        start_coords = time.time()
        full_dict[f"Image_{f}"] = main_coords(image, depth_path[f], full_dict[f"Image_{f}"])
        end_coords = time.time()
        print(f"Coordinates and depth for image {f + 1} obtained in {end_coords - start_coords}s")

        print(f"Image {f + 1}: {full_dict[f'Image_{f}']}")


if __name__ == "__main__":
    start_all = time.time()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    images = ["dataset/rgb/rgb_dataset_2.png"]
    depth_path = ["dataset/depth/depth_dataset_2.png"]

    main(images, depth_path)
    end_all = time.time()

    print(f"Total time image process: {end_all - start_all}s")
