import numpy as np

ALIASES = {
    # The robot is annotated through its base marker, so every phrasing the annotator
    # produces for the arm is matched against that single ground truth object.
    "robotic arm": "robot base",
    "full robotic arm": "robot base",
    "partial robotic arm": "robot base",
}

IGNORE_LABELS = {"Unknown Object", "Unknown object", "unknown object"}

# Objects excluded from the localization statistics. ``bbox_from_tag_px`` is the box around
# the printed tag, which approximates the object centre only when the tag sits on the object.
# The robot base tag does not, so the distance to a segmented arm centre is not comparable to
# the block errors. These objects are still matched and still count towards detection recall.
LOCALIZATION_EXCLUDED = {"robot base"}


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """
    Calculate the center of a bounding box.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        A tuple representing the bounding box in the format (x, y, width, height).

    Returns
    -------
    tuple[float, float]
        A tuple representing the center of the bounding box in the format (center_x, center_y).
    """
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)


def normalize_name(name: str) -> str:
    """
    Normalize the object name by converting it to lowercase, stripping whitespace, and applying any defined aliases.

    Parameters
    ----------
    name : str
        The original object name to be normalized.

    Returns
    -------
    str
        The normalized object name.
    """
    name = name.lower().strip()
    return ALIASES.get(name, name)


def aruco_object_position_m(aruco_data: dict, obj: dict) -> np.ndarray:
    """
    Return the position of an ArUco object in the camera frame.

    Parameters
    ----------
    aruco_data : dict
        The ArUco data of the image, holding ``world.T_camera_world``.
    obj : dict
        One entry of ``aruco_data["objects"]``, holding ``T_world_object``.

    Returns
    -------
    np.ndarray
        ``[x, y, z]`` in metres, in the camera's optical frame (x right, y down, z along the
        optical axis). That is the frame the depth stage writes ``object_point_camera_m`` in.
    """
    T_camera_world = np.asarray(aruco_data["world"]["T_camera_world"], dtype=float)
    T_world_object = np.asarray(obj["T_world_object"], dtype=float)
    return (T_camera_world @ T_world_object)[:3, 3]


def estimated_position_m(seg_obj: dict) -> np.ndarray:
    """
    Return the 3D point the depth stage estimated for an object.

    Parameters
    ----------
    seg_obj : dict
        One object of the per-image result file.

    Returns
    -------
    np.ndarray
        ``[x, y, z]`` in metres, read from ``object_point_camera_m``. The depth stage
        back-projects that point through the pixel its depth association chose: the box
        centre for ``bbox-center``, a pixel of the mask for ``mask-median``. All NaN when
        there is no point, e.g. no valid depth, or a run that had no depth stage.
    """
    point = seg_obj.get("object_point_camera_m")
    if point is None:
        return np.full(3, np.nan)
    try:
        position = np.asarray(point, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return np.full(3, np.nan)
    return position if np.all(np.isfinite(position)) else np.full(3, np.nan)


def match_objects(seg_data: dict, aruco_data: dict) -> list[dict]:
    """
    Match objects from segmentation data to ArUco data based on normalized names, and compute the errors of each match.

    Two errors are computed. The pixel error is the distance between the bounding box centres.
    The metric error is the distance, in millimetres, between the 3D point the depth stage
    estimated for the object and the ArUco object position, both in the camera frame. It is
    NaN when the object has no 3D point.

    Parameters
    ----------
    seg_data : dict
        The segmentation data containing object information and their corresponding bounding boxes.
    aruco_data : dict
        The ArUco data containing the camera pose and object transformations.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the matching results, including pixel and metric errors and other relevant information.
    """
    aruco_objects = aruco_data["objects"]
    aruco_lookup = {normalize_name(obj["name"]): obj for obj in aruco_objects}

    results = []
    ignored_count = 0
    for mask_id, seg_obj in seg_data.items():
        tag = normalize_name(seg_obj["tag"])

        if tag in IGNORE_LABELS:
            ignored_count += 1
            continue

        if tag not in aruco_lookup:
            print(f"[NO MATCH] {seg_obj['tag']}")
            is_matched = False
            continue
        else:
            is_matched = True

        matched = aruco_lookup[tag]

        seg_center = bbox_center(seg_obj["bbox"])

        aruco_center = bbox_center(matched["bbox_from_tag_px"])

        dx = seg_center[0] - aruco_center[0]
        dy = seg_center[1] - aruco_center[1]

        error_px = float(np.sqrt(dx**2 + dy**2))

        estimated = estimated_position_m(seg_obj)
        aruco_position = aruco_object_position_m(aruco_data, matched)
        # NaN propagates, so an object without a 3D point gets NaN errors rather than zero.
        delta_mm = (estimated - aruco_position) * 1000.0

        results.append(
            {
                "mask_id": mask_id,
                "seg_tag": seg_obj["tag"],
                "matched": 1 if is_matched else 0,
                "aruco_name": matched["name"],
                "seg_center_x": seg_center[0],
                "seg_center_y": seg_center[1],
                "aruco_center_x": aruco_center[0],
                "aruco_center_y": aruco_center[1],
                "dx_px": dx,
                "dy_px": dy,
                "abs_dx_px": abs(dx),
                "abs_dy_px": abs(dy),
                "error_px": error_px,
                "depth_association": seg_obj.get("depth_association"),
                "est_x_m": float(estimated[0]),
                "est_y_m": float(estimated[1]),
                "est_z_m": float(estimated[2]),
                "aruco_x_m": float(aruco_position[0]),
                "aruco_y_m": float(aruco_position[1]),
                "aruco_z_m": float(aruco_position[2]),
                "dx_mm": float(delta_mm[0]),
                "dy_mm": float(delta_mm[1]),
                "dz_mm": float(delta_mm[2]),
                "error_mm": float(np.linalg.norm(delta_mm)),
            }
        )

    return results
