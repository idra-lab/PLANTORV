"""Plane removal, geometric clustering, and cluster-label comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class GeometricSegmentation:
    """Plane and cluster labels retaining the point-to-RGB-pixel mapping."""

    visualized_cloud: Any
    point_labels: np.ndarray
    label_image: np.ndarray
    plane_models: tuple[tuple[float, float, float, float], ...]


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
    etc. The same labels are written into ``label_image`` at the corresponding
    calibrated RGB pixels.
    """
    if max_planes < 0:
        raise ValueError("max_planes cannot be negative")
    if plane_distance_m <= 0.0 or dbscan_eps_m <= 0.0:
        raise ValueError("plane_distance_m and dbscan_eps_m must be positive")
    if min_plane_points < 3 or dbscan_min_points < 1:
        raise ValueError("min_plane_points must be >= 3 and dbscan_min_points >= 1")

    point_count = len(point_cloud.points)
    valid_depth = (aligned_depth_mm > 0.0) & (
        aligned_depth_mm < depth_trunc_m * 1000.0
    )
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
            distance_threshold=plane_distance_m,
            ransac_n=3,
            num_iterations=1000,
        )
        if len(local_inliers) < min_plane_points:
            break

        local_inliers_array = np.asarray(local_inliers, dtype=np.int64)
        global_inliers = remaining_indices[local_inliers_array]
        point_labels[global_inliers] = -(plane_id + 2)
        plane_models.append(tuple(float(value) for value in model))

        keep = np.ones(remaining_indices.size, dtype=bool)
        keep[local_inliers_array] = False
        remaining_indices = remaining_indices[keep]

    if remaining_indices.size:
        object_cloud = point_cloud.select_by_index(remaining_indices.tolist())
        cluster_labels = np.asarray(
            object_cloud.cluster_dbscan(
                eps=dbscan_eps_m,
                min_points=dbscan_min_points,
                print_progress=False,
            ),
            dtype=np.int32,
        )
        point_labels[remaining_indices] = cluster_labels

    visualized_cloud = _make_visualized_cloud(
        point_cloud,
        point_labels,
        plane_count=len(plane_models),
    )

    label_image = np.full(aligned_depth_mm.shape, -1, dtype=np.int32)
    label_image[pixel_v, pixel_u] = point_labels

    return GeometricSegmentation(
        visualized_cloud=visualized_cloud,
        point_labels=point_labels,
        label_image=label_image,
        plane_models=tuple(plane_models),
    )


def compare_label_images(
    geometric_labels: np.ndarray,
    semantic_labels: np.ndarray,
) -> list[dict[str, float | int]]:
    """Compute pairwise IoU between geometric clusters and semantic segments."""
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

    return sorted(
        comparisons,
        key=lambda item: float(item["iou"]),
        reverse=True,
    )


def _make_visualized_cloud(
    point_cloud: Any,
    point_labels: np.ndarray,
    *,
    plane_count: int,
) -> Any:
    point_count = len(point_cloud.points)
    colors = np.full((point_count, 3), 0.15, dtype=np.float64)

    for plane_id in range(plane_count):
        colors[point_labels == -(plane_id + 2)] = (0.55, 0.55, 0.55)

    cluster_ids = np.unique(point_labels[point_labels >= 0])
    for cluster_id in cluster_ids:
        colors[point_labels == cluster_id] = _label_color(int(cluster_id))

    visualized_cloud = point_cloud.select_by_index(list(range(point_count)))
    visualized_cloud.colors = o3d.utility.Vector3dVector(colors)
    return visualized_cloud


def _label_color(label: int) -> np.ndarray:
    """Return a stable RGB colour for a non-negative cluster label."""
    hue = ((label * 0.618033988749895) % 1.0) * 179.0
    hsv = np.asarray([[[hue, 210.0, 255.0]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float64) / 255.0
