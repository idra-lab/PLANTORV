"""Export RGB-D frames as Pointcept scene folders for semantic-segmentation testing.

Every frame becomes ``<output-root>/<split>/frame_XXXX/`` holding the arrays that
Pointcept's ``DefaultDataset`` loads:

- ``coord.npy``  float32 (N, 3), metres, gravity-aligned Z-up
- ``color.npy``  uint8   (N, 3), RGB in [0, 255]
- ``normal.npy`` float32 (N, 3), unit normals oriented towards the camera

``pixel.npy`` (int32 (N, 2), row and column in the RGB-aligned depth grid) and
``meta.json`` are written beside them. Pointcept ignores both; they let per-point
predictions (``<save_path>/result/frame_XXXX_pred.npy``, in ``coord.npy`` order) be
put back into image layout.

Run in the plantorv ``.venv``, which provides Open3D and the Femto Mega calibration.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from mapping.camera_model import load_camera_to_world_transform
from mapping.rgbd_pointcloud import RGBDPointCloudGenerator

DEFAULT_OUTPUT_ROOT = Path("models/Pointcept/data/plantorv")

# RGBDPointCloudGenerator returns points with Y and Z negated relative to the optical
# frame (x right, y down, z forward); multiplying by this undoes it.
GENERATOR_TO_OPTICAL = np.array([1.0, -1.0, -1.0])

# Optical frame to x forward, y left, z up, for a camera held level. Without a
# recorded camera-to-world transform this is the best available guess at gravity.
OPTICAL_TO_LEVEL_Z_UP = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Femto Mega RGB-D frames as Pointcept scene folders."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory holding rgb/ and depth/ images paired by the trailing frame number",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Pointcept data_root to write into (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--split",
        help="Split folder name under --output-root (default: the dataset_root folder name)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="Only export these frame numbers (default: every paired frame)",
    )
    parser.add_argument(
        "--depth-unit-scale",
        type=float,
        default=1.0,
        help="Raw-depth-units to millimetres multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--depth-trunc",
        type=float,
        default=3.0,
        help="Maximum retained depth in metres (default: 3.0)",
    )
    parser.add_argument(
        "--camera-to-world",
        type=Path,
        help=(
            "camera_to_world_transform.yaml recorded for the pose these frames were "
            "captured from; without it the camera is assumed level"
        ),
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.05,
        help="Neighbourhood radius in metres for normal estimation (default: 0.05)",
    )
    parser.add_argument(
        "--normal-max-nn",
        type=int,
        default=30,
        help="Maximum neighbours for normal estimation (default: 30)",
    )
    return parser.parse_args()


def index_frames(directory: Path) -> dict[int, Path]:
    """Map the trailing integer of every PNG stem in ``directory`` to its path."""
    frames = {}
    for path in sorted(directory.glob("*.png")):
        match = re.search(r"(\d+)$", path.stem)
        if match is not None:
            frames[int(match.group(1))] = path
    return frames


def pair_frames(dataset_root: Path, selected: list[int] | None) -> list[tuple[int, Path, Path]]:
    rgb = index_frames(dataset_root / "rgb")
    depth = index_frames(dataset_root / "depth")
    paired = sorted(set(rgb) & set(depth))
    unpaired = sorted(set(rgb) ^ set(depth))
    if unpaired:
        print(f"Skipping {len(unpaired)} frames without both rgb and depth: {unpaired[:10]}")
    if selected is not None:
        missing = sorted(set(selected) - set(paired))
        if missing:
            raise FileNotFoundError(f"No rgb/depth pair for frames {missing} in {dataset_root}")
        paired = sorted(set(selected))
    if not paired:
        raise FileNotFoundError(f"No rgb/depth pairs found under {dataset_root}")
    return [(frame, rgb[frame], depth[frame]) for frame in paired]


def optical_to_model_frame(camera_to_world: Path | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the rotation, translation and a description of the output frame."""
    if camera_to_world is None:
        return OPTICAL_TO_LEVEL_Z_UP, np.zeros(3), "level camera, x forward, y left, z up"
    transform = load_camera_to_world_transform(camera_to_world)
    if transform is None:
        raise FileNotFoundError(f"Camera-to-world transform not found: {camera_to_world}")
    rotation = Rotation.from_quat(transform.rotation).as_matrix()  # x, y, z, w
    return rotation, transform.translation, f"{transform.target_frame} from {camera_to_world}"


def export_frame(
    generator: RGBDPointCloudGenerator,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    rgbd = generator.generate(color_bgr, depth_raw)
    cloud = rgbd.point_cloud
    if len(cloud.points) == 0:
        raise ValueError("frame produced an empty point cloud")

    # The camera sits at the origin in the generator frame too, so normals can be
    # oriented there before either frame change.
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=args.normal_max_nn)
    )
    cloud.orient_normals_towards_camera_location(np.zeros(3))

    points_optical = np.asarray(cloud.points) * GENERATOR_TO_OPTICAL
    normals_optical = np.asarray(cloud.normals) * GENERATOR_TO_OPTICAL

    # Same validity rule and row-major order as segment_pcd.point_values_to_image.
    aligned = rgbd.aligned_depth_mm
    valid = (aligned > 0.0) & (aligned <= generator.depth_trunc_m * 1000.0)
    rows, cols = np.nonzero(valid)
    if rows.size != len(points_optical):
        raise RuntimeError(
            f"{rows.size} valid aligned-depth pixels but {len(points_optical)} points; "
            "the pixel mapping would be wrong"
        )

    return {
        "coord": (points_optical @ rotation.T + translation).astype(np.float32),
        "color": np.clip(np.rint(np.asarray(cloud.colors) * 255.0), 0, 255).astype(np.uint8),
        "normal": (normals_optical @ rotation.T).astype(np.float32),
        "pixel": np.stack([rows, cols], axis=1).astype(np.int32),
        "aligned_shape": np.asarray(aligned.shape),
    }


def main() -> None:
    args = parse_args()
    frames = pair_frames(args.dataset_root, args.frames)
    rotation, translation, frame_description = optical_to_model_frame(args.camera_to_world)
    split_dir = args.output_root / (args.split or args.dataset_root.resolve().name)
    print(f"Exporting {len(frames)} frames to {split_dir} ({frame_description})")

    generators: dict[tuple, RGBDPointCloudGenerator] = {}
    for frame, rgb_path, depth_path in frames:
        color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_raw is None:
            raise FileNotFoundError(f"Could not read {rgb_path} or {depth_path}")

        sizes = (color_bgr.shape[:2], depth_raw.shape[:2])
        if sizes not in generators:
            generators[sizes] = RGBDPointCloudGenerator.from_frames(
                color_bgr,
                depth_raw,
                depth_unit_scale=args.depth_unit_scale,
                depth_trunc_m=args.depth_trunc,
            )

        arrays = export_frame(generators[sizes], color_bgr, depth_raw, rotation, translation, args)
        scene_dir = split_dir / f"frame_{frame:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for name in ("coord", "color", "normal", "pixel"):
            np.save(scene_dir / f"{name}.npy", arrays[name])
        meta = {
            "frame": frame,
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "aligned_shape": arrays["aligned_shape"].tolist(),
            "depth_unit_scale": args.depth_unit_scale,
            "depth_trunc_m": args.depth_trunc,
            "coordinate_frame": frame_description,
        }
        (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  {scene_dir.name}: {len(arrays['coord'])} points")


if __name__ == "__main__":
    main()
