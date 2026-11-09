"""Create and geometrically segment a coloured point cloud from one RGB-D pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from mapping.geometric_clustering import segment_planes_and_clusters
from mapping.rgbd_pointcloud import RGBDPointCloudGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and cluster a Femto Mega RGB-D point cloud."
    )
    parser.add_argument("color", type=Path, help="Path to the RGB image")
    parser.add_argument("depth", type=Path, help="Path to the raw depth image")
    parser.add_argument("--output", type=Path, help="Optional .ply/.pcd output path")
    parser.add_argument(
        "--cluster-output",
        type=Path,
        help="Optional coloured cluster cloud",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        help="Optional geometric label .npy file",
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
        "--no-visualize",
        action="store_true",
        help="Do not open the Open3D viewer",
    )
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--plane-distance", type=float, default=0.01, help="Metres")
    parser.add_argument("--min-plane-points", type=int, default=1000)
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.025,
        help="DBSCAN radius in metres",
    )
    parser.add_argument("--cluster-min-points", type=int, default=30)
    return parser.parse_args()


def load_rgbd(color_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if color_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {color_path}")
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")

    return color_bgr, depth_raw


def main() -> None:
    args = parse_args()
    color_bgr, depth_raw = load_rgbd(args.color, args.depth)

    pointcloud_generator = RGBDPointCloudGenerator.from_frames(
        color_bgr,
        depth_raw,
        depth_unit_scale=args.depth_unit_scale,
        depth_trunc_m=args.depth_trunc,
    )
    rgbd_cloud = pointcloud_generator.generate(color_bgr, depth_raw)

    valid_depth = rgbd_cloud.aligned_depth_mm > 0.0
    print(
        f"Aligned depth: {rgbd_cloud.aligned_depth_mm.shape[1]}x"
        f"{rgbd_cloud.aligned_depth_mm.shape[0]}, "
        f"valid pixels: {np.count_nonzero(valid_depth)}, "
        f"points: {len(rgbd_cloud.point_cloud.points)}"
    )

    segmentation = segment_planes_and_clusters(
        rgbd_cloud.point_cloud,
        rgbd_cloud.aligned_depth_mm,
        depth_trunc_m=pointcloud_generator.depth_trunc_m,
        max_planes=args.max_planes,
        plane_distance_m=args.plane_distance,
        min_plane_points=args.min_plane_points,
        dbscan_eps_m=args.cluster_eps,
        dbscan_min_points=args.cluster_min_points,
    )

    cluster_ids = np.unique(segmentation.point_labels[segmentation.point_labels >= 0])
    noise_count = int(np.count_nonzero(segmentation.point_labels == -1))
    print(
        f"Planes: {len(segmentation.plane_models)}, "
        f"clusters: {cluster_ids.size}, "
        f"unclustered points: {noise_count}"
    )

    write_outputs(
        point_cloud=rgbd_cloud.point_cloud,
        segmentation_cloud=segmentation.visualized_cloud,
        label_image=segmentation.label_image,
        output=args.output,
        cluster_output=args.cluster_output,
        labels_output=args.labels_output,
    )

    if not args.no_visualize:
        o3d.visualization.draw_geometries([segmentation.visualized_cloud])


def write_outputs(
    *,
    point_cloud: object,
    segmentation_cloud: object,
    label_image: np.ndarray,
    output: Path | None,
    cluster_output: Path | None,
    labels_output: Path | None,
) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(output), point_cloud):
            raise RuntimeError(f"Open3D could not write point cloud: {output}")
        print(f"Wrote {output}")

    if cluster_output is not None:
        cluster_output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(cluster_output), segmentation_cloud):
            raise RuntimeError(
                f"Open3D could not write cluster cloud: {cluster_output}"
            )
        print(f"Wrote {cluster_output}")

    if labels_output is not None:
        labels_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_output, label_image)
        print(f"Wrote {labels_output}")


if __name__ == "__main__":
    main()