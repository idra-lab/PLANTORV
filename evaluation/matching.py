import numpy as np


def bbox_center(bbox):
    return (
        bbox[0] + bbox[2] / 2,
        bbox[1] + bbox[3] / 2
    )

ALIASES = {
    "partial robotic arm": "Unknown Object",
    "robotic arm": ["robot base", "end effector"],
}

IGNORE_LABELS = {
    "Unknown Object", "Unknown object", "unknown object"
}

def normalize_name(name):
    name = (
        name.lower()
        .strip()
    )
    return ALIASES.get(name, name)


def match_objects(seg_data, aruco_data):
    
    aruco_objects = aruco_data["objects"]

    aruco_lookup = {
        normalize_name(obj["name"]): obj
        for obj in aruco_objects
    }

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

        seg_center = bbox_center(
            seg_obj["bbox"]
        )

        aruco_center = bbox_center(
            matched["bbox_from_tag_px"]
        )

        dx = seg_center[0] - aruco_center[0]
        dy = seg_center[1] - aruco_center[1]

        error_px = float(
            np.sqrt(dx**2 + dy**2)
        )

        results.append({

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

            "error_px": error_px

        })

    return results