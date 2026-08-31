#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utility.utility import logger  # noqa: E402


def load_camera_calibration(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera calibration parameters from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file containing camera calibration parameters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the camera matrix (K) and distortion coefficients (dist).
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    # logger.debug(f"Loaded camera calibration from {path}:")
    return K, dist


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to a 4x4 transformation matrix.

    Parameters
    ----------
    rvec : np.ndarray
        Rotation vector (3x1).
    tvec : np.ndarray
        Translation vector (3x1).

    Returns
    -------
    np.ndarray
        4x4 transformation matrix representing the pose.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def matrix_to_list(T: np.ndarray) -> list[list[float]]:
    """
    Convert a 4x4 transformation matrix to a list of lists for JSON serialization.

    Parameters
    ----------
    T : np.ndarray
        4x4 transformation matrix.

    Returns
    -------
    list[list[float]]
        List of lists representing the transformation matrix.
    """
    return [[float(v) for v in row] for row in T]


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 transformation matrix.

    Parameters
    ----------
    T : np.ndarray
        4x4 transformation matrix to be inverted.

    Returns
    -------
    np.ndarray
        Inverted 4x4 transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def load_marker_config(path: Union[str, Path]) -> dict:
    """
    Load marker configuration from a YAML file.

    Example config::

        world_marker_id: 0
        marker_size_m: 0.04
        objects:
          10:
        name: mug
        T_object_tag:
          [[1,0,0,0],
           [0,1,0,0],
           [0,0,1,0],
           [0,0,0,1]]
          11:
        name: box
        T_object_tag:
          [[1,0,0,0],
           [0,1,0,0],
           [0,0,1,0],
           [0,0,0,1]]
        robot:
          marker_id: 100
          frame_name: end_effector
          T_robot_tag:
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]

    Parameters
    ----------
    path : str or Path
        Path to the YAML file containing marker configuration.

    Returns
    -------
    dict
        Dictionary containing the marker configuration.
    """
    # logger.debug(f"Loaded marker config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def detect_aruco_poses(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_size_m: float,
    dictionary_name: str = "DICT_6X6_250",
) -> tuple[dict, Sequence[np.ndarray], Optional[np.ndarray]]:
    """
    Detect ArUco markers in the given image and estimate their poses.

    Parameters
    ----------
    image : np.ndarray
        Input image in which to detect ArUco markers.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Distortion coefficients.
    marker_size_m : float
        Size of the ArUco marker in meters.
    dictionary_name : str, optional
        Name of the predefined ArUco dictionary to use (default is "DICT_6X6_250").

    Returns
    -------
    tuple[dict, np.ndarray, np.ndarray]
        A tuple containing:
        - detections: A dictionary mapping marker IDs to their detection data, including corners, rotation vector, translation vector, and transformation matrix.
        - corners: Detected marker corners in the image.
        - ids: Detected marker IDs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict_id = getattr(cv2.aruco, dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(gray)

    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    detections = {}
    if ids is None:
        return detections, corners, ids
    ids = ids.flatten()
    for marker_corners, marker_id in zip(corners, ids):
        object_points = np.array(
            [
                [-marker_size_m / 2, marker_size_m / 2, 0],
                [marker_size_m / 2, marker_size_m / 2, 0],
                [marker_size_m / 2, -marker_size_m / 2, 0],
                [-marker_size_m / 2, -marker_size_m / 2, 0],
            ],
            dtype=np.float32,
        )
        image_points = marker_corners.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not ok:
            logger.warning(f" [WARN] solvePnP failed for marker ID {marker_id}")
            continue
        cv2.drawFrameAxes(image, K, dist, rvec, tvec, 0.1)
        T_camera_tag = rvec_tvec_to_matrix(rvec, tvec)
        detections[int(marker_id)] = {
            "corners_px": image_points.tolist(),
            "rvec": rvec.reshape(3).tolist(),
            "tvec": tvec.reshape(3).tolist(),
            "T_camera_tag": T_camera_tag,
        }

        marker_corners_flattened = marker_corners.reshape(-1, 2)
        for i, corner in enumerate(marker_corners_flattened):
            cv2.circle(image, tuple(corner.astype(int)), 5, (0, 255, 0), -1)

        for object_point, image_point in zip(object_points, image_points):
            projected_point, _ = cv2.projectPoints(
                object_point.reshape(1, 1, 3), rvec, tvec, K, dist
            )
            logger.info(
                f"Object Point: {object_point}, Image Point: {image_point}, Projected Point: {projected_point.flatten()}"
            )
            cv2.circle(image, tuple(projected_point.reshape(2).astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(image, tuple(image_point.astype(int)), 5, (255, 0, 0), -1)

    cv2.imshow("detections", image)
    cv2.waitKey()
    return detections, corners, ids


def annotate_pair(
    clean_image_path: Path,
    tag_image_path: Path,
    output_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    config: dict,
    idx: int,
) -> dict:
    """
    Annotate a pair of clean and tag images with detected ArUco markers and their poses.

    Parameters
    ----------
    clean_image_path : Path
        Path to the clean image.
    tag_image_path : Path
        Path to the tag image.
    output_path : Path
        Path to the output directory where the annotation will be saved.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Distortion coefficients.
    config : dict
        Configuration dictionary containing marker information.
    idx : int
        Index of the image pair being processed.

    Returns
    -------
    dict
        A dictionary containing the annotation data for the image pair.
    """
    clean_image = cv2.imread(str(clean_image_path))
    tag_image = cv2.imread(str(tag_image_path))
    if clean_image is None:
        raise RuntimeError(f"Could not read clean image: {clean_image_path}")
    if tag_image is None:
        raise RuntimeError(f"Could not read tag image: {tag_image_path}")
    marker_size_m = float(config["marker_size_m"])
    world_marker_id = int(config["world_marker_id"])
    detections, corners, ids = detect_aruco_poses(tag_image, K, dist, marker_size_m)

    if world_marker_id not in detections:
        raise RuntimeError(f"World marker {world_marker_id} not detected in {tag_image_path}")
    T_camera_world = detections[world_marker_id]["T_camera_tag"]
    logger.debug(f"T_camera_world:\n{T_camera_world}")
    T_world_camera = invert_transform(T_camera_world)
    objects_out = []
    for marker_id_str, obj_cfg in config.get("objects", {}).items():
        marker_id = int(marker_id_str)
        if marker_id not in detections:
            logger.warning(
                f"Object marker {marker_id} not detected, skipping object '{obj_cfg['name']}'"
            )
            continue
        T_camera_tag = detections[marker_id]["T_camera_tag"]
        logger.debug(f"T_camera_tag:\n{T_camera_tag}")
        T_world_tag = T_world_camera @ T_camera_tag
        T_object_tag = np.array(obj_cfg.get("T_object_tag", np.eye(4)), dtype=np.float64)
        T_tag_object = invert_transform(T_object_tag)
        T_world_object = T_world_tag @ T_tag_object
        T_m = T_camera_world @ T_world_object
        logger.info(f"{marker_id} T_m Z = {float(T_m[2, 3] * 1000):.3f} mm")
        corners_px = np.array(detections[marker_id]["corners_px"], dtype=np.float32)
        x_min, y_min = corners_px.min(axis=0)
        x_max, y_max = corners_px.max(axis=0)
        objects_out.append(
            {
                "name": obj_cfg["name"],
                "marker_id": marker_id,
                "bbox_from_tag_px": [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                ],
                "depth": float(T_m[2, 3] * 1000),
                "T_world_object": matrix_to_list(T_world_object),
                "T_world_tag": matrix_to_list(T_world_tag),
                "tag_corners_px": corners_px.tolist(),
            }
        )

    logger.info(f"Detected {len(objects_out)} objects in {tag_image_path.name}")
    robot_out = None
    if "robot" in config:
        robot_marker_id = int(config["robot"]["base_marker_id"])
        if robot_marker_id in detections:
            T_camera_tag = detections[robot_marker_id]["T_camera_tag"]
            T_world_tag = T_world_camera @ T_camera_tag
            T_robot_tag = np.array(config["robot"].get("T_robot_tag", np.eye(4)), dtype=np.float64)
            T_tag_robot = invert_transform(T_robot_tag)
            T_world_robot = T_world_tag @ T_tag_robot
            robot_out = {
                "marker_id": robot_marker_id,
                "frame_name": config["robot"].get("frame_name", "robot"),
                "T_world_robot": matrix_to_list(T_world_robot),
                "T_world_tag": matrix_to_list(T_world_tag),
            }
        else:
            logger.warning(
                f"Robot marker {robot_marker_id} not detected, skipping robot annotation"
            )
    logger.info(str(clean_image_path.name))
    logger.info(str(tag_image_path.name))

    annotation = {
        "clean_image": str(clean_image_path.name),
        "tag_image": str(tag_image_path.name),
        "camera": {"camera_matrix": K.tolist(), "dist_coeffs": dist.reshape(-1).tolist()},
        "world": {"marker_id": world_marker_id, "T_camera_world": matrix_to_list(T_camera_world)},
        "objects": objects_out,
        "robot": robot_out,
    }
    logger.info(f"Saving annotation to {output_path}")
    output_pathjson = Path(output_path) / f"aruco_pos_img{idx + 1}.json"
    output_pathjson.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pathjson, "w") as f:
        json.dump(annotation, f, indent=2)
    debug = tag_image.copy()
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        for marker_id, det in detections.items():
            rvec = np.array(det["rvec"], dtype=np.float64)
            tvec = np.array(det["tvec"], dtype=np.float64)
            cv2.drawFrameAxes(debug, K, dist, rvec, tvec, marker_size_m * 0.75)
    output_pathdebug = Path(output_path) / f"image_{idx + 1}_representation"
    debug_path = output_pathdebug.with_suffix(".debug.png")
    cv2.imwrite(str(debug_path), debug)

    return annotation


def main() -> None:
    """Parse command-line arguments and process image pairs for ArUco marker detection and annotation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--tag_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--camera_yaml", required=True)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--clean_suffix", default=".png")
    parser.add_argument("--tag_suffix", default=".png")
    args = parser.parse_args()
    clean_dir = Path(args.clean_dir)
    tag_dir = Path(args.tag_dir)
    out_dir = Path(args.out_dir)
    K, dist = load_camera_calibration(args.camera_yaml)
    config = load_marker_config(args.config_yaml)
    clean_images = sorted(
        clean_dir.glob(f"*{args.clean_suffix}"), key=lambda x: int(x.stem.split("_")[-1])
    )
    logger.info(
        f"Found {len(clean_images)} clean images in {clean_dir} with suffix '{args.clean_suffix}'"
    )
    for i, clean_path in enumerate(clean_images):
        stem = clean_path.name.replace(args.clean_suffix, "")
        tag_path = tag_dir / f"{stem}{args.tag_suffix}"
        out_path = out_dir / f"{stem}"
        if not tag_path.exists():
            logger.warning(f"Missing tag image for {clean_path.name}")
            continue
        try:
            annotate_pair(clean_path, tag_path, out_path, K, dist, config, i)
            logger.info(f"{clean_path.name} -> {out_path.name}")
        except Exception as e:
            logger.error(f"{clean_path.name}: {e}")


if __name__ == "__main__":
    main()
