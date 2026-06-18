#!/usr/bin/env python3
import cv2
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
def load_camera_calibration(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    # print(f"Loaded camera calibration from {path}:")
    return K, dist
def rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T
def matrix_to_list(T):
    return [[float(v) for v in row] for row in T]
def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
def load_marker_config(path):
    """
    Example config:
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
    """
    # print(f"Loaded marker config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
def detect_aruco_poses(image, K, dist, marker_size_m, dictionary_name="DICT_6X6_250"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict_id = getattr(cv2.aruco, dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(gray)

    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    # cv2.imshow("detections", image)
    # cv2.waitKey()

    detections = {}
    if ids is None:
        return detections, corners, ids
    ids = ids.flatten()
    for marker_corners, marker_id in zip(corners, ids):

        object_points = np.array([
            [-marker_size_m / 2,  marker_size_m / 2, 0],
            [ marker_size_m / 2,  marker_size_m / 2, 0],
            [ marker_size_m / 2, -marker_size_m / 2, 0],
            [-marker_size_m / 2, -marker_size_m / 2, 0],
        ], dtype=np.float32)
        image_points = marker_corners.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not ok:
            print(" [WARN] solvePnP failed for marker ID {marker_id}")
            continue
        cv2.drawFrameAxes(image, K, dist, rvec, tvec, 0.1)
        T_camera_tag = rvec_tvec_to_matrix(rvec, tvec)
        detections[int(marker_id)] = {
            "corners_px": image_points.tolist(),
            "rvec": rvec.reshape(3).tolist(),
            "tvec": tvec.reshape(3).tolist(),
            "T_camera_tag": T_camera_tag,
        }
    
    cv2.imshow("detections", image)
    cv2.waitKey()
    return detections, corners, ids



def annotate_pair(clean_image_path, tag_image_path, output_path, K, dist, config,idx):
    clean_image = cv2.imread(str(clean_image_path))
    tag_image = cv2.imread(str(tag_image_path))
    if clean_image is None:
        raise RuntimeError(f"Could not read clean image: {clean_image_path}")
    if tag_image is None:
        raise RuntimeError(f"Could not read tag image: {tag_image_path}")
    marker_size_m = float(config["marker_size_m"])
    world_marker_id = int(config["world_marker_id"])
    detections, corners, ids = detect_aruco_poses(
        tag_image,
        K,
        dist,
        marker_size_m
    )

    if world_marker_id not in detections:
        raise RuntimeError(f"World marker {world_marker_id} not detected in {tag_image_path}")
    T_camera_world = detections[world_marker_id]["T_camera_tag"]
    T_world_camera = invert_transform(T_camera_world)
    objects_out = []
    for marker_id_str, obj_cfg in config.get("objects", {}).items():
        marker_id = int(marker_id_str)
        if marker_id not in detections:
            print(f"[WARN] Object marker {marker_id} not detected, skipping object '{obj_cfg['name']}'")
            continue
        T_camera_tag = detections[marker_id]["T_camera_tag"]
        T_world_tag = T_world_camera @ T_camera_tag
        T_object_tag = np.array(obj_cfg.get("T_object_tag", np.eye(4)), dtype=np.float64)
        T_tag_object = invert_transform(T_object_tag)
        T_world_object = T_world_tag @ T_tag_object
        T_m=T_camera_world @ T_world_object
        print(f"{marker_id} T_m Z =", T_m[2,3]*1000)
        corners_px = np.array(detections[marker_id]["corners_px"], dtype=np.float32)
        x_min, y_min = corners_px.min(axis=0)
        x_max, y_max = corners_px.max(axis=0)
        objects_out.append({
            "name": obj_cfg["name"],
            "marker_id": marker_id,
            "bbox_from_tag_px": [
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min)
            ],
            "T_world_object": matrix_to_list(T_world_object),
            "T_world_tag": matrix_to_list(T_world_tag),
            "tag_corners_px": corners_px.tolist()
        })

    print(f"Detected {len(objects_out)} objects in {tag_image_path.name}")
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
                "T_world_tag": matrix_to_list(T_world_tag)
            }
        else: 
            print(f"[WARN] Robot marker {robot_marker_id} not detected, skipping robot annotation")
    print(str(clean_image_path.name))
    print(str(tag_image_path.name))

    annotation = {
        "clean_image": str(clean_image_path.name),
        "tag_image": str(tag_image_path.name),
        "camera": {
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.reshape(-1).tolist()
        },
        "world": {
            "marker_id": world_marker_id,
            "T_camera_world": matrix_to_list(T_camera_world)
        },
        "objects": objects_out,
        "robot": robot_out
    }
    print(f"Saving annotation to {output_path}")
    output_pathjson = Path(output_path) / f"aruco_pos_img{idx+1}.json"
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
    output_pathdebug = Path(output_path) / f"image{idx+1}_representation"
    debug_path = output_pathdebug.with_suffix(".debug.png")
    cv2.imwrite(str(debug_path), debug)
    return annotation

def main():
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
    clean_images = sorted(clean_dir.glob(f"*{args.clean_suffix}"), key = lambda x: int(x.stem.split("_")[-1]))
    print(f"Found {len(clean_images)} clean images in {clean_dir} with suffix '{args.clean_suffix}'")
    for i,clean_path in enumerate(clean_images):
        stem = clean_path.name.replace(args.clean_suffix, "")
        tag_path = tag_dir / f"{stem}{args.tag_suffix}"
        out_path = out_dir / f"{stem}"
        if not tag_path.exists():
            print(f"[WARN] Missing tag image for {clean_path.name}")
            continue
        try:
            annotate_pair(clean_path, tag_path, out_path, K, dist, config,i)
            print(f"[OK] {clean_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"[ERROR] {clean_path.name}: {e}")
if __name__ == "__main__":
    main()