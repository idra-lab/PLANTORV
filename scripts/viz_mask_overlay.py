"""Draw the segmentation masks of a frame on top of its RGB image.

Each frame directory produced by the segmentation stage holds a ``masks.npz`` with
a boolean ``masks`` array of shape (N, H, W) and a ``meta.json`` naming the source
image and the bounding box of every object. This script paints one colour per mask
over the image with a bit of transparency, so the pixels underneath stay readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Golden-ratio hue steps keep neighbouring masks far apart in colour.
GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frame",
        type=Path,
        help="Frame directory holding masks.npz and meta.json, or a directory of them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mask_overlay"),
        help="Output file (single frame) or directory (directory of frames)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Mask opacity, 0 keeps the image untouched and 1 hides it completely",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="RGB image to draw on, overriding the path recorded in meta.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Directory the image path in meta.json is relative to (default: repo root)",
    )
    parser.add_argument(
        "--masks",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="Only draw these mask indices",
    )
    parser.add_argument("--no-outline", action="store_true", help="Skip the mask contours")
    parser.add_argument("--boxes", action="store_true", help="Also draw the bounding boxes")
    parser.add_argument(
        "--no-labels", action="store_true", help="Skip the index (or tag) drawn on each mask"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Annotation JSON (mask_<i> -> tag) to label the masks with instead of indices",
    )
    parser.add_argument("--show", action="store_true", help="Open a window per frame")
    return parser.parse_args()


def mask_colors(count: int) -> np.ndarray:
    """One saturated BGR colour per mask, spread evenly around the hue circle."""
    hues = np.array([(i * GOLDEN_RATIO_CONJUGATE) % 1.0 for i in range(count)])
    hsv = np.zeros((count, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (hues * 180).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(count, 3)


def find_image(meta: dict, frame_dir: Path, args: argparse.Namespace) -> Path:
    """Resolve the RGB frame recorded in meta.json."""
    if args.image is not None:
        return args.image

    recorded = Path(meta.get("image", ""))
    if not recorded.parts:
        raise FileNotFoundError(f"No image recorded in {frame_dir / 'meta.json'}")
    if recorded.is_absolute():
        return recorded

    roots = [args.image_root] if args.image_root is not None else [Path.cwd(), REPO_ROOT]
    for root in roots:
        candidate = root / recorded
        if candidate.exists():
            return candidate
    tried = ", ".join(str(root / recorded) for root in roots)
    raise FileNotFoundError(f"Could not find the image for {frame_dir.name}, tried: {tried}")


def load_labels(annotations: Optional[Path]) -> dict[int, str]:
    if annotations is None:
        return {}
    data = json.loads(annotations.read_text())
    labels: dict[int, str] = {}
    for key, entry in data.items():
        if not key.startswith("mask_"):
            continue
        tag = entry.get("tag") if isinstance(entry, dict) else None
        if tag:
            labels[int(key.removeprefix("mask_"))] = str(tag).strip()
    return labels


def draw_label(image: np.ndarray, text: str, mask: np.ndarray, color: np.ndarray) -> None:
    """Write a caption at the centroid of the mask, on a filled backing box."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return
    cx, cy = int(xs.mean()), int(ys.mean())
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(np.clip(cx - tw // 2, 2, image.shape[1] - tw - 2))
    y = int(np.clip(cy + th // 2, th + 2, image.shape[0] - baseline - 2))
    box = color.tolist()
    cv2.rectangle(image, (x - 3, y - th - 3), (x + tw + 3, y + baseline), box, cv2.FILLED)
    # Dark text on bright colours, white on dark ones.
    luma = 0.114 * box[0] + 0.587 * box[1] + 0.299 * box[2]
    ink = (0, 0, 0) if luma > 140 else (255, 255, 255)
    cv2.putText(image, text, (x, y), font, scale, ink, thickness, cv2.LINE_AA)


def overlay(frame_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    meta = json.loads((frame_dir / "meta.json").read_text())
    masks = np.load(frame_dir / "masks.npz")["masks"].astype(bool)

    image_path = find_image(meta, frame_dir, args)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if masks.shape[1:] != image.shape[:2]:
        raise ValueError(
            f"{frame_dir.name}: masks are {masks.shape[1:]} but the image is {image.shape[:2]}"
        )

    indices = list(range(len(masks))) if args.masks is None else args.masks
    unknown = [i for i in indices if not 0 <= i < len(masks)]
    if unknown:
        raise IndexError(f"{frame_dir.name} has {len(masks)} masks, asked for {unknown}")

    colors = mask_colors(len(masks))
    labels = load_labels(args.annotations)
    boxes = {obj["index"]: obj.get("bbox") for obj in meta.get("objects", [])}

    # Blend every mask in one pass so overlapping masks stay equally transparent
    # instead of the later ones piling up opaque paint on the earlier ones.
    tint = np.zeros_like(image, dtype=np.float32)
    covered = np.zeros(image.shape[:2], dtype=np.float32)
    for i in indices:
        mask = masks[i]
        tint[mask] += colors[i]
        covered[mask] += 1.0
    blended = image.astype(np.float32)
    painted = covered > 0
    weight = args.alpha
    blended[painted] = (1 - weight) * blended[painted] + weight * (
        tint[painted] / covered[painted, None]
    )
    result = np.clip(blended, 0, 255).astype(np.uint8)

    for i in indices:
        color = colors[i].tolist()
        if not args.no_outline:
            contours, _ = cv2.findContours(
                masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
        if args.boxes and boxes.get(i):
            x, y, w, h = boxes[i]
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        if not args.no_labels:
            draw_label(result, labels.get(i, str(i)), masks[i], colors[i])

    return result, len(indices)


def is_frame_dir(path: Path) -> bool:
    return (path / "masks.npz").is_file() and (path / "meta.json").is_file()


def render(frame_dir: Path, out_path: Path, args: argparse.Namespace) -> str:
    result, drawn = overlay(frame_dir, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), result):
        raise OSError(f"Could not write overlay: {out_path}")
    if args.show:
        cv2.imshow(frame_dir.name, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return f"{frame_dir.name}: {drawn} mask(s) -> {out_path}"


def main() -> int:
    args = parse_args()
    if is_frame_dir(args.frame):
        out_path = args.output
        if out_path.suffix == "":
            out_path = out_path / f"{args.frame.name}.png"
        print(render(args.frame, out_path, args))
        return 0

    frame_dirs = sorted(p for p in args.frame.iterdir() if p.is_dir() and is_frame_dir(p))
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories with masks.npz in {args.frame}")
    for frame_dir in frame_dirs:
        print(render(frame_dir, args.output / f"{frame_dir.name}.png", args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
