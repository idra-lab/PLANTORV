import numpy as np
import pandas as pd
import json
from matching import normalize_name, IGNORE_LABELS


def compute_global_metrics(df):

    errors = df["error_px"]
    matched = df["matched"].sum()
    matched_rate = 100 * matched / len(df)
    return {
        "num_measurements": len(df),
        "num_matched": int(matched),
        "matched_rate_percent": float(matched_rate),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(errors.median()),
        "rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
        "std_error_px": float(errors.std()),
        "min_error_px": float(errors.min()),
        "max_error_px": float(errors.max()),
        "success_rate_25px": float(100 * (errors < 25).mean()),
        "success_rate_50px": float(100 * (errors < 50).mean()),
    }


def object_statistics(df):

    return (
        df.groupby("aruco_name")
        .agg(
            mean_error_px=("error_px", "mean"),
            std_error_px=("error_px", "std"),
            max_error_px=("error_px", "max"),
            count=("error_px", "count"),
        )
        .reset_index()
    )


def image_statistics(df):

    return (
        df.groupby("image")
        .agg(
            mean_error_px=("error_px", "mean"),
            std_error_px=("error_px", "std"),
            max_error_px=("error_px", "max"),
            count=("error_px", "count"),
        )
        .reset_index()
    )


def save_summary(summary, path):

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def compute_detection_metrics(seg_data, aruco_data):

    aruco_names = set(normalize_name(obj["name"]) for obj in aruco_data["objects"])

    seg_names = set(normalize_name(obj["tag"]) for obj in seg_data.values())

    seg_names = {x for x in seg_names if x not in IGNORE_LABELS}

    detected_gt = seg_names & aruco_names

    missed_gt = aruco_names - seg_names

    extra_objects = seg_names - aruco_names

    recall = len(detected_gt) / len(aruco_names) if len(aruco_names) > 0 else 0

    return {
        "gt_objects": len(aruco_names),
        "detected_gt": len(detected_gt),
        "missed_gt": len(missed_gt),
        "extra_objects": len(extra_objects),
        "recall": recall,
    }
