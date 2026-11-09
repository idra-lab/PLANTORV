"""Quick smoke test for the Depth Anything 3 metric-depth model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from mapping.depth_anything import (
    DEFAULT_FOCAL_LENGTH_PX,
    DEFAULT_V3_MODEL_ID,
    DepthAnythingV3Provider,
)

DEFAULT_IMAGE = Path("dataset/rgb/rgb_dataset_1.png")
DEFAULT_OUTPUT = Path("output/monocular_depth_preview.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 on one image and save a depth preview."
    )
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_V3_MODEL_ID)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--focal-length-px", type=float, default=DEFAULT_FOCAL_LENGTH_PX)
    parser.add_argument("--process-res", type=int, default=504)
    return parser.parse_args()


def colorize_depth(depth_mm: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convert valid metric depth values to a robust color preview."""
    valid_depth = depth_mm[valid_mask]
    if valid_depth.size == 0:
        raise RuntimeError("The model did not produce any valid depth values")

    near_mm, far_mm = np.percentile(valid_depth, (2.0, 98.0))
    if near_mm == far_mm:
        normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    else:
        clipped = np.clip(depth_mm, near_mm, far_mm)
        normalized = np.asarray((clipped - near_mm) * (255.0 / (far_mm - near_mm)), dtype=np.uint8)

    preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    preview[~valid_mask] = 0
    return preview


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.image}")

    started_at = time.perf_counter()
    provider = DepthAnythingV3Provider(
        model_id=args.model,
        focal_length_px=args.focal_length_px,
        process_res=args.process_res,
        device=args.device,
    )
    result = provider.estimate(image)
    elapsed_s = time.perf_counter() - started_at

    preview = colorize_depth(result.depth_mm, result.valid_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), preview):
        raise OSError(f"Could not write depth preview: {args.output}")

    valid_depth = result.depth_mm[result.valid_mask]
    valid_percent = 100.0 * valid_depth.size / result.depth_mm.size
    print("PASS: Depth Anything 3 produced a valid metric depth map")
    print(f"  image:       {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"  depth:       {result.depth_mm.shape[1]}x{result.depth_mm.shape[0]}")
    print(f"  valid:       {valid_percent:.1f}%")
    print(f"  range:       {valid_depth.min():.0f}-{valid_depth.max():.0f} mm")
    print(f"  device:      {result.metadata['device']}")
    print(f"  elapsed:     {elapsed_s:.2f} s (including model load)")
    print(f"  preview:     {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
