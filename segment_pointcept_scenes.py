"""Segment exported Pointcept scenes with one forward pass per frame, and summarise them.

``tools/test.py`` predicts every frame from every point of every voxel under ten
scale/flip variants. On dense RGB-D clouds that is ~1,900 forward passes, ~47 s per
frame. This script uses the single-pass protocol of Pointcept's own validation instead:
keep one point per voxel, run the model once, and give every point the prediction of
its voxel. Expect slightly noisier labels than the full test, in exchange for seconds.

With ``--predictions`` the model is not run at all: the ``<frame>_pred.npy`` files that
``tools/test.py`` already wrote are only turned into images, clouds and a summary.

Outputs, per frame, in ``--output``:

- ``frame_XXXX_pred.npy``        int64 (N,), class id per point, ``coord.npy`` order
- ``frame_XXXX_confidence.npy``  float32 (N,), top softmax score (model runs only)
- ``frame_XXXX_labels.npy``      int32 image in the aligned depth grid, -1 without depth
- ``frame_XXXX_semantic.ply``    cloud coloured by class (unless ``--no-ply``)

plus ``summary.csv`` with the share of points per class for every frame.

Run in the Pointcept conda env from the plantorv root, after export_pointcept_scenes.py:

    PYTHONNOUSERSITE=1 python segment_pointcept_scenes.py dataset
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_POINTCEPT_ROOT = Path("models/Pointcept")
DEFAULT_CONFIG = Path("configs/plantorv/semseg-pt-v3m1-0-rpe-s3dis.py")
DEFAULT_OUTPUT_ROOT = Path("results/pointcept")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-pass Pointcept segmentation of exported plantorv scenes."
    )
    parser.add_argument("split", help="Split folder under the config's data_root, e.g. dataset")
    parser.add_argument(
        "--pointcept-root",
        type=Path,
        default=DEFAULT_POINTCEPT_ROOT,
        help=f"Pointcept checkout; relative config paths resolve here (default: {DEFAULT_POINTCEPT_ROOT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config, relative to --pointcept-root (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--weight", type=Path, help="Checkpoint overriding the config's weight")
    parser.add_argument(
        "--data-root", type=Path, help="Scene root overriding the config's data_root"
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="Only process these frame numbers (default: every scene in the split)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}/<split>)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Skip the model and read <frame>_pred.npy from this directory, e.g. a tools/test.py result/ folder",
    )
    parser.add_argument("--no-ply", action="store_true", help="Do not write coloured .ply clouds")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random choice of each voxel's representative point (default: 0)",
    )
    return parser.parse_args()


def resolve(path: Path | str, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def list_scenes(scene_root: Path, frames: list[int] | None) -> list[Path]:
    scenes = sorted(path for path in scene_root.iterdir() if (path / "coord.npy").is_file())
    if frames is not None:
        wanted = {f"frame_{frame:04d}" for frame in frames}
        missing = sorted(wanted - {scene.name for scene in scenes})
        if missing:
            raise FileNotFoundError(f"Scenes {missing} not found in {scene_root}")
        scenes = [scene for scene in scenes if scene.name in wanted]
    if not scenes:
        raise FileNotFoundError(f"No scenes with coord.npy under {scene_root}")
    return scenes


def feat_keys_from(cfg: Any) -> tuple[str, ...]:
    """Read the model's input features from the Collect step of the test pipeline."""
    for transform in cfg.data.test.test_cfg.post_transform:
        if transform["type"] == "Collect":
            return tuple(transform["feat_keys"])
    raise ValueError("The config's test post_transform has no Collect step")


class SinglePassSegmenter:
    """Pointcept model plus the validation-style single-pass input pipeline."""

    def __init__(self, cfg: Any, weight: Path) -> None:
        import torch
        from pointcept.datasets import collate_fn
        from pointcept.datasets.transform import Compose
        from pointcept.models import build_model

        self._torch = torch
        self._collate = collate_fn
        self.model = build_model(cfg.model).cuda().eval()
        checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
        state = {
            key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()
        }
        self.model.load_state_dict(state, strict=True)

        # The config's val pipeline, minus labels: GridSample in train mode keeps one
        # random point per voxel and records which voxel every original point fell in.
        self.transform = Compose(
            [
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=cfg.data.test.test_cfg.voxelize.grid_size,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "inverse"),
                    feat_keys=feat_keys_from(cfg),
                ),
            ]
        )

    def __call__(self, scene: Path) -> tuple[np.ndarray, np.ndarray]:
        data = {
            name: np.load(scene / f"{name}.npy").astype(np.float32)
            for name in ("coord", "color", "normal")
            if (scene / f"{name}.npy").is_file()
        }
        batch = self._collate([self.transform(data)])
        batch = {
            key: value.cuda(non_blocking=True) if isinstance(value, self._torch.Tensor) else value
            for key, value in batch.items()
        }
        with self._torch.no_grad():
            logits = self.model(batch)["seg_logits"]
            confidence, pred = self._torch.softmax(logits, dim=-1).max(dim=1)
        inverse = batch["inverse"]
        return pred[inverse].cpu().numpy(), confidence[inverse].cpu().numpy()


def label_image(scene: Path, pred: np.ndarray) -> np.ndarray | None:
    """Place per-point labels into the aligned depth grid recorded by the exporter."""
    pixel_path, meta_path = scene / "pixel.npy", scene / "meta.json"
    if not (pixel_path.is_file() and meta_path.is_file()):
        return None
    pixel = np.load(pixel_path)
    height, width = json.loads(meta_path.read_text())["aligned_shape"]
    image = np.full((height, width), -1, dtype=np.int32)
    image[pixel[:, 0], pixel[:, 1]] = pred
    return image


def write_ply(path: Path, coord: np.ndarray, pred: np.ndarray, palette: np.ndarray) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(palette[pred])
    o3d.io.write_point_cloud(str(path), cloud)


def main() -> None:
    args = parse_args()
    pointcept_root = args.pointcept_root.resolve()
    sys.path.insert(0, str(pointcept_root))
    from pointcept.utils.config import Config

    cfg = Config.fromfile(str(resolve(args.config, pointcept_root)))
    names = list(cfg.data.names)
    palette = np.array([colorsys.hsv_to_rgb(i / len(names), 0.65, 0.95) for i in range(len(names))])
    scene_root = resolve(args.data_root or cfg.data.test.data_root, pointcept_root) / args.split
    scenes = list_scenes(scene_root, args.frames)
    output = args.output or DEFAULT_OUTPUT_ROOT / args.split
    output.mkdir(parents=True, exist_ok=True)

    segmenter = None
    if not args.predictions:
        weight = args.weight or resolve(cfg.weight, pointcept_root)
        print(f"Loading {weight}")
        segmenter = SinglePassSegmenter(cfg, weight)
    print(f"{len(scenes)} scenes from {scene_root} -> {output}")

    counts_per_frame = {}
    for scene in scenes:
        start = time.time()
        if segmenter is None:
            pred = np.load(args.predictions / f"{scene.name}_pred.npy")
        else:
            np.random.seed(args.seed)
            pred, confidence = segmenter(scene)
            np.save(output / f"{scene.name}_confidence.npy", confidence.astype(np.float32))
        np.save(output / f"{scene.name}_pred.npy", pred.astype(np.int64))

        image = label_image(scene, pred)
        if image is not None:
            np.save(output / f"{scene.name}_labels.npy", image)
        if not args.no_ply:
            write_ply(
                output / f"{scene.name}_semantic.ply", np.load(scene / "coord.npy"), pred, palette
            )

        counts = np.bincount(pred, minlength=len(names))
        counts_per_frame[scene.name] = counts
        top = ", ".join(
            f"{names[i]} {100 * counts[i] / pred.size:.0f}%" for i in np.argsort(-counts)[:3]
        )
        print(f"  {scene.name}: {pred.size} points, {time.time() - start:.1f} s, {top}")

    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "points", *names])
        for frame, counts in counts_per_frame.items():
            writer.writerow(
                [frame, counts.sum(), *(f"{100 * c / counts.sum():.2f}" for c in counts)]
            )

    total = np.sum(list(counts_per_frame.values()), axis=0)
    print("Share of all points:")
    for i in np.argsort(-total):
        if total[i]:
            print(f"  {names[i]:10s} {100 * total[i] / total.sum():5.1f}%")
    print(f"Wrote {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
