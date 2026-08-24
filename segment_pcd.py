"""Create and semantically segment a coloured point cloud from one RGB-D pair."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from mapping.rgbd_pointcloud import RGBDPointCloudGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and semantically segment a Femto Mega RGB-D point cloud."
    )

    parser.add_argument("color", type=Path, help="Path to the RGB image")
    parser.add_argument("depth", type=Path, help="Path to the raw depth image")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Open3D-ML model configuration YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pretrained Open3D-ML model checkpoint",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Device used for semantic inference",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional original RGB point cloud (.ply/.pcd)",
    )
    parser.add_argument(
        "--semantic-output",
        type=Path,
        help="Optional semantic-coloured point cloud (.ply/.pcd)",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        help="Optional semantic label image (.npy)",
    )
    parser.add_argument(
        "--confidence-output",
        type=Path,
        help="Optional per-pixel semantic confidence image (.npy)",
    )

    parser.add_argument(
        "--depth-unit-scale",
        type=float,
        default=1.0,
        help="Raw-depth-units to millimetres multiplier",
    )
    parser.add_argument(
        "--depth-trunc",
        type=float,
        default=3.0,
        help="Maximum retained depth in metres",
    )

    parser.add_argument(
        "--no-z-up",
        action="store_true",
        help=(
            "Do not rotate camera-frame points into a gravity-aligned "
            "Z-up frame before inference"
        ),
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Do not open the Open3D viewer",
    )

    return parser.parse_args()


def load_rgbd(
    color_path: Path,
    depth_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if color_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {color_path}")

    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")

    return color_bgr, depth_raw


def camera_to_z_up(points: np.ndarray) -> np.ndarray:
    """
    Convert a conventional optical camera frame

        x -> right
        y -> down
        z -> forward

    to a gravity-oriented frame

        x -> forward
        y -> left
        z -> up

    Point correspondence is unchanged; only coordinates passed to the
    semantic network are rotated.
    """
    transformed = np.empty_like(points)

    transformed[:, 0] = points[:, 2]
    transformed[:, 1] = -points[:, 0]
    transformed[:, 2] = -points[:, 1]

    return transformed


def create_semantic_pipeline(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
):
    """
    Construct an Open3D-ML semantic segmentation pipeline from a config.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {checkpoint_path}"
        )

    cfg = _ml3d.utils.Config.load_from_file(str(config_path))

    Model = _ml3d.utils.get_module(
        "model",
        cfg.model.name,
        "torch",
    )

    Pipeline = _ml3d.utils.get_module(
        "pipeline",
        cfg.pipeline.name,
        "torch",
    )

    model = Model(**cfg.model)

    pipeline = Pipeline(
        model=model,
        device=device,
        **cfg.pipeline,
    )

    pipeline.load_ckpt(ckpt_path=str(checkpoint_path))

    return pipeline, model, cfg


def prepare_semantic_input(
    point_cloud: o3d.geometry.PointCloud,
    *,
    model,
    use_z_up: bool,
) -> dict[str, np.ndarray | None]:
    """
    Convert an Open3D point cloud to Open3D-ML inference input.

    RandLA-Net expects:
        point: N x 3 XYZ
        feat:  N x F optional features
        label: N dummy labels for the inference pipeline

    For a model with 6 input channels:
        3 channels = XYZ
        3 channels = RGB features
    """
    points = np.asarray(point_cloud.points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point array: {points.shape}")

    if len(points) == 0:
        raise ValueError("Point cloud contains no points")

    if use_z_up:
        inference_points = camera_to_z_up(points)
    else:
        inference_points = points.copy()

    in_channels = int(model.cfg.in_channels)

    if in_channels == 3:
        features = None

    elif in_channels == 6:
        colors = np.asarray(point_cloud.colors, dtype=np.float32)

        if colors.shape != points.shape:
            raise ValueError(
                "The selected model expects RGB features, but the point "
                "cloud does not contain one RGB value per point."
            )

        # Open3D geometry stores colours in [0, 1].
        #
        # S3DIS/Semantic3D RandLA-Net configurations expect RGB features
        # in approximately [0, 255] and perform their own normalization.
        features = colors * 255.0

    else:
        raise ValueError(
            f"Model expects {in_channels} input channels. "
            "This script currently supports XYZ (3) or XYZ+RGB (6)."
        )

    dummy_labels = np.zeros(len(points), dtype=np.int32)

    return {
        "point": inference_points.astype(np.float32),
        "feat": features,
        "label": dummy_labels,
    }


def run_semantic_segmentation(
    pipeline,
    point_cloud: o3d.geometry.PointCloud,
    *,
    model,
    use_z_up: bool,
) -> tuple[np.ndarray, np.ndarray]:
    data = prepare_semantic_input(
        point_cloud,
        model=model,
        use_z_up=use_z_up,
    )

    result = pipeline.run_inference(data)

    labels = np.asarray(
        result["predict_labels"],
        dtype=np.int32,
    ).reshape(-1)

    scores = np.asarray(
        result["predict_scores"],
        dtype=np.float32,
    )

    if len(labels) != len(point_cloud.points):
        raise RuntimeError(
            "Semantic inference returned a different number of labels "
            f"({len(labels)}) than input points "
            f"({len(point_cloud.points)})."
        )

    confidence = np.max(scores, axis=-1)

    return labels, confidence


def get_label_names(cfg) -> dict[int, str]:
    """
    Obtain the semantic class names from the dataset associated with
    the selected Open3D-ML configuration.
    """
    try:
        Dataset = _ml3d.utils.get_module(
            "dataset",
            cfg.dataset.name,
        )
        return Dataset.get_label_to_names()
    except Exception:
        return {}


def print_semantic_statistics(
    labels: np.ndarray,
    confidence: np.ndarray,
    label_names: dict[int, str],
) -> None:
    unique_labels, counts = np.unique(labels, return_counts=True)

    print("\nSemantic segmentation:")

    for label, count in zip(unique_labels, counts):
        mask = labels == label
        mean_confidence = float(np.mean(confidence[mask]))

        name = label_names.get(int(label), f"class_{label}")

        print(
            f"  {label:2d}  {name:<20} "
            f"{count:7d} points  "
            f"confidence={mean_confidence:.3f}"
        )


def semantic_palette(num_classes: int) -> np.ndarray:
    """
    Produce deterministic, visually separated RGB colours.
    """
    palette = np.zeros((num_classes, 3), dtype=np.float64)

    golden_ratio = 0.618033988749895

    hue = 0.0

    for i in range(num_classes):
        hue = (hue + golden_ratio) % 1.0

        palette[i] = colorsys.hsv_to_rgb(
            hue,
            0.70,
            0.95,
        )

    return palette


def create_semantic_cloud(
    point_cloud: o3d.geometry.PointCloud,
    labels: np.ndarray,
) -> o3d.geometry.PointCloud:
    semantic_cloud = o3d.geometry.PointCloud(point_cloud)

    num_classes = int(labels.max()) + 1
    palette = semantic_palette(num_classes)

    semantic_cloud.colors = o3d.utility.Vector3dVector(
        palette[labels]
    )

    return semantic_cloud


def point_values_to_image(
    values: np.ndarray,
    aligned_depth_mm: np.ndarray,
    *,
    depth_trunc_m: float,
    fill_value: int | float,
    dtype,
) -> np.ndarray:
    """
    Project values associated with point-cloud points back into the
    aligned depth-image layout.

    Assumes RGBDPointCloudGenerator preserves row-major valid-depth
    ordering when constructing the Open3D point cloud.
    """
    depth = np.asarray(aligned_depth_mm)

    valid_mask = (
        (depth > 0.0)
        & (depth <= depth_trunc_m * 1000.0)
    )

    expected_points = int(np.count_nonzero(valid_mask))

    if expected_points != len(values):
        raise RuntimeError(
            "Could not map semantic predictions back to image pixels. "
            f"Depth mask contains {expected_points} valid pixels, but "
            f"the point cloud contains {len(values)} points. "
            "RGBDPointCloudGenerator may use a different validity mask."
        )

    image = np.full(
        depth.shape,
        fill_value,
        dtype=dtype,
    )

    image[valid_mask] = values

    return image


def write_outputs(
    *,
    point_cloud: o3d.geometry.PointCloud,
    semantic_cloud: o3d.geometry.PointCloud,
    label_image: np.ndarray,
    confidence_image: np.ndarray,
    output: Path | None,
    semantic_output: Path | None,
    labels_output: Path | None,
    confidence_output: Path | None,
) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

        if not o3d.io.write_point_cloud(str(output), point_cloud):
            raise RuntimeError(
                f"Open3D could not write point cloud: {output}"
            )

        print(f"Wrote {output}")

    if semantic_output is not None:
        semantic_output.parent.mkdir(parents=True, exist_ok=True)

        if not o3d.io.write_point_cloud(
            str(semantic_output),
            semantic_cloud,
        ):
            raise RuntimeError(
                "Open3D could not write semantic point cloud: "
                f"{semantic_output}"
            )

        print(f"Wrote {semantic_output}")

    if labels_output is not None:
        labels_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_output, label_image)
        print(f"Wrote {labels_output}")

    if confidence_output is not None:
        confidence_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(confidence_output, confidence_image)
        print(f"Wrote {confidence_output}")


def main() -> None:
    args = parse_args()

    color_bgr, depth_raw = load_rgbd(
        args.color,
        args.depth,
    )

    # ------------------------------------------------------------------
    # RGB-D -> coloured point cloud
    # ------------------------------------------------------------------

    pointcloud_generator = RGBDPointCloudGenerator.from_frames(
        color_bgr,
        depth_raw,
        depth_unit_scale=args.depth_unit_scale,
        depth_trunc_m=args.depth_trunc,
    )

    rgbd_cloud = pointcloud_generator.generate(
        color_bgr,
        depth_raw,
    )

    point_cloud = rgbd_cloud.point_cloud

    print(
        f"Aligned depth: "
        f"{rgbd_cloud.aligned_depth_mm.shape[1]}x"
        f"{rgbd_cloud.aligned_depth_mm.shape[0]}, "
        f"points: {len(point_cloud.points)}"
    )

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    pipeline, model, cfg = create_semantic_pipeline(
        args.config,
        args.checkpoint,
        args.device,
    )

    labels, confidence = run_semantic_segmentation(
        pipeline,
        point_cloud,
        model=model,
        use_z_up=not args.no_z_up,
    )

    label_names = get_label_names(cfg)

    print_semantic_statistics(
        labels,
        confidence,
        label_names,
    )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    semantic_cloud = create_semantic_cloud(
        point_cloud,
        labels,
    )

    # ------------------------------------------------------------------
    # Map per-point semantics back to image coordinates
    # ------------------------------------------------------------------

    label_image = point_values_to_image(
        labels,
        rgbd_cloud.aligned_depth_mm,
        depth_trunc_m=pointcloud_generator.depth_trunc_m,
        fill_value=-1,
        dtype=np.int32,
    )

    confidence_image = point_values_to_image(
        confidence,
        rgbd_cloud.aligned_depth_mm,
        depth_trunc_m=pointcloud_generator.depth_trunc_m,
        fill_value=0.0,
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    write_outputs(
        point_cloud=point_cloud,
        semantic_cloud=semantic_cloud,
        label_image=label_image,
        confidence_image=confidence_image,
        output=args.output,
        semantic_output=args.semantic_output,
        labels_output=args.labels_output,
        confidence_output=args.confidence_output,
    )

    if not args.no_visualize:
        o3d.visualization.draw_geometries(
            [semantic_cloud],
            window_name="Open3D semantic segmentation",
        )


if __name__ == "__main__":
    main()