import json

import numpy as np
import pandas as pd

from .matching import IGNORE_LABELS, normalize_name


def compute_global_metrics(df: pd.DataFrame) -> dict:
    """
    Compute global metrics from the DataFrame containing matching results.

    The metrics include the number of measurements, number of matched objects, matched rate, mean error, median error, RMSE, standard deviation of error, minimum and maximum errors, and success rates for different pixel thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the matching results, including pixel errors and other relevant information.

    Returns
    -------
    dict
        A dictionary containing the computed global metrics.
    """
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


def object_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each object based on the matching results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the matching results, including pixel errors and other relevant information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed statistics for each object, including mean error, standard deviation of error, maximum error, and count of measurements.
    """
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


def image_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each image based on the matching results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the matching results, including pixel errors and other relevant information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed statistics for each image, including mean error, standard deviation of error, maximum error, and count of measurements.
    """
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


def save_summary(summary: dict, path: str) -> None:
    """
    Save the summary dictionary to a JSON file at the specified path.

    Parameters
    ----------
    summary : dict
        A dictionary containing the summary metrics to be saved.
    path : str
        The file path where the summary JSON file will be saved.
    """
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def compute_detection_metrics(seg_data: dict, aruco_data: dict) -> dict:
    """
    Compute detection metrics based on the segmentation data and ArUco data.

    Parameters
    ----------
    seg_data : dict
        The segmentation data containing object information and their corresponding bounding boxes.
    aruco_data : dict
        The ArUco data containing the camera pose and object transformations.

    Returns
    -------
    dict
        A dictionary containing the computed detection metrics, including the number of ground truth objects, detected ground truth objects, missed ground truth objects, extra objects, and recall.
    """
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
