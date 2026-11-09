"""Create a coloured point cloud from a Femto Mega RGB/depth pair."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from mapping.rgbd_mapper import RGBDMapper


@dataclass(frozen=True)
class GeometricSegmentation:
    """Plane and cluster labels that retain the point-to-RGB-pixel mapping."""

    visualized_cloud: Any
    point_labels: np.ndarray
    label_image: np.ndarray
    plane_models: tuple[tuple[float, float, float, float], ...]


def _label_color(label: int) -> np.ndarray:
    """Return a stable RGB colour for a non-negative cluster label."""
    hue = ((label * 0.618033988749895) % 1.0) * 179.0
    hsv = np.asarray([[[hue, 210.0, 255.0]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float64) / 255.0


def create_aligned_point_cloud(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    *,
    depth_unit_scale: float = 1.0,
    depth_trunc_m: float = 3.0,
) -> tuple[Any, np.ndarray]:
    """Align a Femto Mega depth frame to RGB and construct an Open3D point cloud.

    Parameters
    ----------
    color_bgr
        OpenCV BGR colour image. It may be the camera's full-resolution frame.
    depth_raw
        Single-channel raw depth image.
    depth_unit_scale
        Multiplier that converts one raw depth unit to millimetres.
    depth_trunc_m
        Discard points farther than this distance in metres.

    Returns
    -------
    tuple
        The coloured Open3D point cloud and the depth image aligned to RGB, in mm.

    Notes
    -----
    ``RGBDMapper`` produces an image at the calibrated RGB resolution (usually
    640x360 for a 1920x1080 Femto colour frame). The colour frame is resized to
    that calibration resolution; the raw depth frame must never be resized to
    make the two images match.
    """
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 image")
    if depth_raw.ndim != 2:
        raise ValueError("depth_raw must be a single-channel image")
    if depth_unit_scale <= 0.0:
        raise ValueError("depth_unit_scale must be positive")
    if depth_trunc_m <= 0.0:
        raise ValueError("depth_trunc_m must be positive")

    color_size = (color_bgr.shape[1], color_bgr.shape[0])
    depth_size = (depth_raw.shape[1], depth_raw.shape[0])
    mapper = RGBDMapper.from_hardcoded(color_size=color_size, depth_size=depth_size)
    aligned_depth_mm = mapper.align_depth_to_color(depth_raw, depth_unit_scale=depth_unit_scale)

    intr = mapper.calibration.rgb_intrinsic
    calibrated_size = (intr.width, intr.height)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    if color_size != calibrated_size:
        color_rgb = cv2.resize(color_rgb, calibrated_size, interpolation=cv2.INTER_AREA)

    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_rgb, dtype=np.uint8))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(aligned_depth_mm, dtype=np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1000.0,  # aligned_depth_mm is always expressed in millimetres
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
    )
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return point_cloud, aligned_depth_mm


def segment_planes_and_clusters(
    point_cloud: Any,
    aligned_depth_mm: np.ndarray,
    *,
    depth_trunc_m: float = 3.0,
    max_planes: int = 1,
    plane_distance_m: float = 0.01,
    min_plane_points: int = 1000,
    dbscan_eps_m: float = 0.025,
    dbscan_min_points: int = 30,
) -> GeometricSegmentation:
    """Remove dominant planes and DBSCAN-cluster the remaining points.

    Cluster IDs are non-negative. Noise is ``-1`` and planes are ``-2``, ``-3``,
    etc. The same values are written into ``label_image`` at the corresponding
    calibrated RGB pixels, making it directly comparable with a semantic mask.
    """
    if max_planes < 0:
        raise ValueError("max_planes cannot be negative")
    if plane_distance_m <= 0.0 or dbscan_eps_m <= 0.0:
        raise ValueError("plane_distance_m and dbscan_eps_m must be positive")
    if min_plane_points < 3 or dbscan_min_points < 1:
        raise ValueError("min_plane_points must be >= 3 and dbscan_min_points >= 1")

    point_count = len(point_cloud.points)
    valid_depth = (aligned_depth_mm > 0.0) & (aligned_depth_mm < depth_trunc_m * 1000.0)
    pixel_v, pixel_u = np.nonzero(valid_depth)
    if pixel_u.size != point_count:
        raise RuntimeError(
            "Open3D point order cannot be mapped to RGB pixels: "
            f"{point_count} points but {pixel_u.size} valid aligned-depth pixels"
        )

    point_labels = np.full(point_count, -1, dtype=np.int32)
    remaining_indices = np.arange(point_count, dtype=np.int64)
    plane_models: list[tuple[float, float, float, float]] = []

    for plane_id in range(max_planes):
        if remaining_indices.size < min_plane_points:
            break
        remaining_cloud = point_cloud.select_by_index(remaining_indices.tolist())
        model, local_inliers = remaining_cloud.segment_plane(
            distance_threshold=plane_distance_m, ransac_n=3, num_iterations=1000
        )
        if len(local_inliers) < min_plane_points:
            break
        global_inliers = remaining_indices[np.asarray(local_inliers, dtype=np.int64)]
        point_labels[global_inliers] = -(plane_id + 2)
        plane_models.append(tuple(float(value) for value in model))
        keep = np.ones(remaining_indices.size, dtype=bool)
        keep[np.asarray(local_inliers, dtype=np.int64)] = False
        remaining_indices = remaining_indices[keep]

    if remaining_indices.size:
        object_cloud = point_cloud.select_by_index(remaining_indices.tolist())
        cluster_labels = np.asarray(
            object_cloud.cluster_dbscan(
                eps=dbscan_eps_m, min_points=dbscan_min_points, print_progress=False
            ),
            dtype=np.int32,
        )
        point_labels[remaining_indices] = cluster_labels

    colors = np.full((point_count, 3), 0.15, dtype=np.float64)
    for plane_id in range(len(plane_models)):
        colors[point_labels == -(plane_id + 2)] = (0.55, 0.55, 0.55)
    cluster_ids = np.unique(point_labels[point_labels >= 0])
    for cluster_id in cluster_ids:
        colors[point_labels == cluster_id] = _label_color(int(cluster_id))

    visualized_cloud = point_cloud.select_by_index(list(range(point_count)))
    visualized_cloud.colors = o3d.utility.Vector3dVector(colors)
    label_image = np.full(aligned_depth_mm.shape, -1, dtype=np.int32)
    label_image[pixel_v, pixel_u] = point_labels
    return GeometricSegmentation(
        visualized_cloud=visualized_cloud,
        point_labels=point_labels,
        label_image=label_image,
        plane_models=tuple(plane_models),
    )


def compare_label_images(
    geometric_labels: np.ndarray, semantic_labels: np.ndarray
) -> list[dict[str, float | int]]:
    """Compute pairwise IoU between geometric clusters and semantic segments.

    A full-resolution semantic label image is reduced to the calibrated RGB plane
    with nearest-neighbour interpolation, which preserves its integer class IDs.
    """
    if geometric_labels.ndim != 2 or semantic_labels.ndim != 2:
        raise ValueError("geometric_labels and semantic_labels must be 2D label images")
    if geometric_labels.shape != semantic_labels.shape:
        semantic_labels = cv2.resize(
            semantic_labels,
            (geometric_labels.shape[1], geometric_labels.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    comparisons: list[dict[str, float | int]] = []
    geometric_ids = np.unique(geometric_labels[geometric_labels >= 0])
    semantic_ids = np.unique(semantic_labels[semantic_labels >= 0])
    for cluster_id in geometric_ids:
        cluster_mask = geometric_labels == cluster_id
        for semantic_id in semantic_ids:
            semantic_mask = semantic_labels == semantic_id
            intersection = int(np.count_nonzero(cluster_mask & semantic_mask))
            if intersection == 0:
                continue
            union = int(np.count_nonzero(cluster_mask | semantic_mask))
            comparisons.append(
                {
                    "cluster_id": int(cluster_id),
                    "semantic_id": int(semantic_id),
                    "intersection": intersection,
                    "union": union,
                    "iou": intersection / union,
                }
            )
    return sorted(comparisons, key=lambda item: float(item["iou"]), reverse=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align a Femto Mega depth frame to RGB and create an Open3D point cloud."
    )
    parser.add_argument("color", type=Path, help="Path to the RGB image")
    parser.add_argument("depth", type=Path, help="Path to the raw depth image")
    parser.add_argument("--output", type=Path, help="Optional .ply/.pcd output path")
    parser.add_argument("--cluster-output", type=Path, help="Optional coloured cluster cloud")
    parser.add_argument("--labels-output", type=Path, help="Optional geometric label .npy file")
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
    parser.add_argument("--no-visualize", action="store_true", help="Do not open the Open3D viewer")
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--plane-distance", type=float, default=0.01, help="Metres")
    parser.add_argument("--min-plane-points", type=int, default=1000)
    parser.add_argument("--cluster-eps", type=float, default=0.025, help="DBSCAN radius in metres")
    parser.add_argument("--cluster-min-points", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    color_bgr = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {args.color}")
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {args.depth}")

    point_cloud, aligned_depth_mm = create_aligned_point_cloud(
        color_bgr,
        depth_raw,
        depth_unit_scale=args.depth_unit_scale,
        depth_trunc_m=args.depth_trunc,
    )
    valid_depth = aligned_depth_mm > 0.0
    print(
        f"Aligned depth: {aligned_depth_mm.shape[1]}x{aligned_depth_mm.shape[0]}, "
        f"valid pixels: {np.count_nonzero(valid_depth)}, points: {len(point_cloud.points)}"
    )

    segmentation = segment_planes_and_clusters(
        point_cloud,
        aligned_depth_mm,
        depth_trunc_m=args.depth_trunc,
        max_planes=args.max_planes,
        plane_distance_m=args.plane_distance,
        min_plane_points=args.min_plane_points,
        dbscan_eps_m=args.cluster_eps,
        dbscan_min_points=args.cluster_min_points,
    )
    cluster_ids = np.unique(segmentation.point_labels[segmentation.point_labels >= 0])
    noise_count = int(np.count_nonzero(segmentation.point_labels == -1))
    print(
        f"Planes: {len(segmentation.plane_models)}, clusters: {cluster_ids.size}, "
        f"unclustered points: {noise_count}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(args.output), point_cloud):
            raise RuntimeError(f"Open3D could not write point cloud: {args.output}")
        print(f"Wrote {args.output}")

    if args.cluster_output is not None:
        args.cluster_output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(args.cluster_output), segmentation.visualized_cloud):
            raise RuntimeError(f"Open3D could not write cluster cloud: {args.cluster_output}")
        print(f"Wrote {args.cluster_output}")

    if args.labels_output is not None:
        args.labels_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.labels_output, segmentation.label_image)
        print(f"Wrote {args.labels_output}")

    if not args.no_visualize:
        o3d.visualization.draw_geometries([segmentation.visualized_cloud])


if __name__ == "__main__":
    main()
