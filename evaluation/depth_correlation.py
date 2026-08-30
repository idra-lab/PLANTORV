import numpy as np
import pandas as pd

from .matching import IGNORE_LABELS, normalize_name


def extract_aruco_depth_mm(aruco_data: dict, obj: dict) -> float:
    """
    Compute object depth from: (T_camera_world, T_world_object) and return depth in mm.

    Parameters
    ----------
    aruco_data : dict
        The ArUco data containing the camera pose and object transformations.
    obj : dict
        The object data containing the transformation from world to object.

    Returns
    -------
    float
        The depth of the object in millimeters.
    """
    T_camera_world = np.array(aruco_data["world"]["T_camera_world"])

    T_world_object = np.array(obj["T_world_object"])

    T_camera_object = T_camera_world @ T_world_object

    depth_m = T_camera_object[2, 3]
    depth_mm = abs(depth_m) * 1000.0

    return depth_mm


def compute_depth_correlation(seg_data: dict, aruco_data: dict) -> tuple[pd.DataFrame, int]:
    """
    Compute the correlation between RGB-D depth and ArUco depth for each object in the segmentation data.

    Parameters
    ----------
    seg_data : dict
        The segmentation data containing object information and their corresponding depths.
    aruco_data : dict
        The ArUco data containing the camera pose and object transformations.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple containing a DataFrame with the correlation results and the number of ignored objects due to invalid or missing depth information.
    """
    aruco_lookup = {normalize_name(obj["name"]): obj for obj in aruco_data["objects"]}

    results = []
    ignored = 0
    for mask_id, seg_obj in seg_data.items():
        tag = normalize_name(seg_obj["tag"])

        if tag in IGNORE_LABELS:
            continue

        if tag not in aruco_lookup:
            continue

        matched = aruco_lookup[tag]

        rgbd_depth_mm = seg_obj["coord_center&depth"][2]

        if rgbd_depth_mm is None or np.isnan(rgbd_depth_mm) or rgbd_depth_mm <= 0.0:
            ignored += 1
            continue

        aruco_depth_mm = extract_aruco_depth_mm(aruco_data, matched)

        abs_error_mm = abs(rgbd_depth_mm - aruco_depth_mm)

        error_percent = (abs_error_mm / aruco_depth_mm) * 100.0

        results.append(
            {
                "mask_id": mask_id,
                "object_name": matched["name"],
                "rgbd_depth_mm": rgbd_depth_mm,
                "aruco_depth_mm": aruco_depth_mm,
                "abs_error_mm": abs_error_mm,
                "error_percent": error_percent,
            }
        )

    return pd.DataFrame(results), ignored
