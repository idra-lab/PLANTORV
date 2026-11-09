"""Turn raw 16-bit depth PNGs into viewable colour previews.

The depth maps in ``dataset/depth`` store millimetres in a uint16 PNG. Their values
span roughly 0-1500 out of the 65535 the format allows, so a normal image viewer
renders them almost entirely black. This script rescales each map over its own
valid range and applies a colormap, optionally next to the matching RGB frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "gray": None,
}

INVALID_COLOR = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth", type=Path, help="Depth PNG, or a directory of them")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/depth_preview"),
        help="Output file (single input) or directory (directory input)",
    )
    parser.add_argument("--colormap", choices=sorted(COLORMAPS), default="turbo")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Percentiles of the valid depths mapped to the ends of the colormap",
    )
    parser.add_argument(
        "--range-mm",
        type=float,
        nargs=2,
        default=None,
        metavar=("NEAR", "FAR"),
        help="Fixed millimetre range instead of per-image percentiles. Use this to "
        "compare frames: percentiles rescale every frame independently, so the same "
        "colour means a different distance in each one.",
    )
    parser.add_argument(
        "--near-is-bright",
        action="store_true",
        help="Invert so close surfaces get the warm end of the colormap",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Directory of RGB frames to show side by side (matched by frame number)",
    )
    parser.add_argument("--show", action="store_true", help="Open a window per image")
    return parser.parse_args()


def colorize(
    depth_mm: np.ndarray,
    *,
    colormap: str,
    percentiles: tuple[float, float],
    range_mm: Optional[tuple[float, float]],
    near_is_bright: bool,
) -> np.ndarray:
    """Rescale one depth map over its valid values and colour it."""
    depth_f = depth_mm.astype(np.float32)
    valid = np.isfinite(depth_f) & (depth_f > 0)
    preview = np.zeros((*depth_f.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return preview

    if range_mm is not None:
        near, far = range_mm
    else:
        near, far = np.percentile(depth_f[valid], percentiles)
    if far <= near:
        far = near + 1.0

    scaled = np.clip((depth_f - near) * (255.0 / (far - near)), 0, 255).astype(np.uint8)
    if near_is_bright:
        scaled = 255 - scaled

    code = COLORMAPS[colormap]
    coloured = (
        cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        if code is None
        else cv2.applyColorMap(scaled, code)
    )
    preview[valid] = coloured[valid]
    preview[~valid] = INVALID_COLOR
    return preview


def find_rgb(depth_path: Path, rgb_dir: Path) -> Optional[Path]:
    """Locate the RGB frame that goes with a depth frame by its trailing number."""
    stem = depth_path.stem
    suffix = stem.rsplit("_", 1)[-1]
    for candidate in sorted(rgb_dir.glob(f"*_{suffix}{depth_path.suffix}")):
        return candidate
    return None


def stack_with_rgb(preview: np.ndarray, rgb_path: Path) -> np.ndarray:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return preview
    if rgb.shape[:2] != preview.shape[:2]:
        rgb = cv2.resize(rgb, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA)
    return np.hstack((rgb, preview))


def render(depth_path: Path, out_path: Path, args: argparse.Namespace) -> str:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]

    preview = colorize(
        depth,
        colormap=args.colormap,
        percentiles=tuple(args.percentiles),
        range_mm=tuple(args.range_mm) if args.range_mm else None,
        near_is_bright=args.near_is_bright,
    )
    if args.rgb_dir is not None:
        rgb_path = find_rgb(depth_path, args.rgb_dir)
        if rgb_path is not None:
            preview = stack_with_rgb(preview, rgb_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), preview):
        raise OSError(f"Could not write preview: {out_path}")

    if args.show:
        cv2.imshow(depth_path.name, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    valid = depth[(depth > 0)]
    coverage = 100.0 * valid.size / depth.size
    span = f"{valid.min():.0f}-{valid.max():.0f} mm" if valid.size else "no valid depth"
    return f"{depth_path.name}: {span}, {coverage:.1f}% valid -> {out_path}"


def main() -> int:
    args = parse_args()
    if args.depth.is_dir():
        depth_paths = sorted(args.depth.glob("*.png"))
        if not depth_paths:
            raise FileNotFoundError(f"No PNG depth images in {args.depth}")
        for depth_path in depth_paths:
            print(render(depth_path, args.output / depth_path.name, args))
    else:
        out_path = args.output
        if out_path.suffix == "":
            out_path = out_path / args.depth.name
        print(render(args.depth, out_path, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
