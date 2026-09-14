import json
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .matching import IGNORE_LABELS, normalize_name


def finite_values(values: Iterable[float]) -> np.ndarray:
    """
    Return the finite values of a sequence as a float array, dropping NaN, infinities and non-numbers.

    Parameters
    ----------
    values : Iterable[float]
        The values to filter.

    Returns
    -------
    np.ndarray
        The finite values, in order.
    """
    array = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def error_statistics(values: Iterable[float], unit: str, prefix: str = "") -> dict:
    """
    Compute the summary statistics of a set of errors.

    Non-finite values are left out. A statistic that is undefined for the number of values left
    is None, so that it is written as ``null`` in JSON rather than as an invalid ``NaN``. The
    standard deviation and the variance are sample statistics (``ddof=1``), as in pandas, and
    need at least two values.

    Parameters
    ----------
    values : Iterable[float]
        The errors.
    unit : str
        Unit suffix of the keys, e.g. ``"px"`` or ``"mm"``.
    prefix : str
        Inserted before ``error`` in the keys, e.g. ``"depth_"``.

    Returns
    -------
    dict
        ``mean_<prefix>error_<unit>``, ``median_<prefix>error_<unit>``, ``rmse_<prefix><unit>``,
        ``std_<prefix>error_<unit>``, ``var_<prefix>error_<unit>``, ``min_<prefix>error_<unit>``
        and ``max_<prefix>error_<unit>``.
    """
    errors = finite_values(values)

    def statistic(function: Callable[[np.ndarray], float], minimum: int = 1) -> float | None:
        return float(function(errors)) if errors.size >= minimum else None

    return {
        f"mean_{prefix}error_{unit}": statistic(np.mean),
        f"median_{prefix}error_{unit}": statistic(np.median),
        f"rmse_{prefix}{unit}": statistic(lambda e: np.sqrt(np.mean(np.square(e)))),
        f"std_{prefix}error_{unit}": statistic(lambda e: np.std(e, ddof=1), minimum=2),
        f"var_{prefix}error_{unit}": statistic(lambda e: np.var(e, ddof=1), minimum=2),
        f"min_{prefix}error_{unit}": statistic(np.min),
        f"max_{prefix}error_{unit}": statistic(np.max),
    }


def compute_global_metrics(df: pd.DataFrame) -> dict:
    """
    Compute global metrics from the DataFrame containing matching results.

    The pixel metrics include the number of measurements, number of matched objects, matched rate, mean error, median error, RMSE, standard deviation and variance of error, minimum and maximum errors, and success rates for different pixel thresholds.

    The metric position metrics are the same statistics of ``error_mm``, the distance between the estimated 3D point and the ArUco position, over the matches that have a 3D point. They also include the mean absolute error along each camera axis, and how many matches each depth association produced.

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

    summary = {
        "num_measurements": len(df),
        "num_matched": int(matched),
        "matched_rate_percent": float(matched_rate),
    }
    summary.update(error_statistics(errors, "px"))
    summary["success_rate_25px"] = float(100 * (errors < 25).mean())
    summary["success_rate_50px"] = float(100 * (errors < 50).mean())

    empty = pd.Series(dtype=float)
    metric_errors = df["error_mm"] if "error_mm" in df else empty
    summary["num_metric_measurements"] = int(finite_values(metric_errors).size)
    summary.update(error_statistics(metric_errors, "mm"))
    for axis in ("x", "y", "z"):
        column = f"d{axis}_mm"
        axis_errors = finite_values(np.abs(df[column])) if column in df else np.array([])
        summary[f"mean_abs_d{axis}_mm"] = float(axis_errors.mean()) if axis_errors.size else None

    # An object whose mask covers no valid depth falls back from mask-median to bbox-center,
    # so the counts say how many of the points really came from each association.
    if "depth_association" in df:
        counts = df["depth_association"].dropna().value_counts()
        summary["depth_association_counts"] = {str(k): int(v) for k, v in counts.items()}

    return summary


def compute_depth_metrics(depth_df: pd.DataFrame, ignored: int) -> dict:
    """
    Compute the depth metrics of a run from its depth correlation rows.

    Parameters
    ----------
    depth_df : pd.DataFrame
        The depth correlation rows of every image, holding ``abs_error_mm`` and ``signed_error_mm``.
    ignored : int
        The number of matched objects whose depth was missing or invalid.

    Returns
    -------
    dict
        The number of measured and ignored objects, the statistics of the absolute depth error
        (see :func:`error_statistics`, with the ``depth_`` prefix), and the mean and variance of
        the signed error, which measure the bias and the spread of the depth estimate.
    """
    empty = pd.Series(dtype=float)
    abs_errors = depth_df["abs_error_mm"] if "abs_error_mm" in depth_df else empty
    signed_errors = finite_values(depth_df["signed_error_mm"] if "signed_error_mm" in depth_df else empty)

    metrics = {
        "num_depth_measurements": int(finite_values(abs_errors).size),
        "num_depth_ignored": int(ignored),
    }
    metrics.update(error_statistics(abs_errors, "mm", prefix="depth_"))
    metrics["mean_signed_depth_error_mm"] = (
        float(signed_errors.mean()) if signed_errors.size else None
    )
    metrics["var_signed_depth_error_mm"] = (
        float(np.var(signed_errors, ddof=1)) if signed_errors.size > 1 else None
    )
    return metrics


def _group_statistics(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Aggregate the pixel errors, and the metric errors when present, per value of ``key``."""
    aggregations = {
        "mean_error_px": ("error_px", "mean"),
        "std_error_px": ("error_px", "std"),
        "var_error_px": ("error_px", "var"),
        "max_error_px": ("error_px", "max"),
        "count": ("error_px", "count"),
    }
    if "error_mm" in df:
        aggregations.update(
            {
                "mean_error_mm": ("error_mm", "mean"),
                "std_error_mm": ("error_mm", "std"),
                "var_error_mm": ("error_mm", "var"),
                "max_error_mm": ("error_mm", "max"),
                "count_mm": ("error_mm", "count"),
            }
        )
    return df.groupby(key).agg(**aggregations).reset_index()


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
        A DataFrame containing the computed statistics for each object: mean, standard deviation, variance and maximum of the pixel error and count of measurements, plus the same for the metric error (``count_mm`` counting the matches with a 3D point) when the matches carry it.
    """
    return _group_statistics(df, "aruco_name")


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
        A DataFrame containing the computed statistics for each image: mean, standard deviation, variance and maximum of the pixel error and count of measurements, plus the same for the metric error (``count_mm`` counting the matches with a 3D point) when the matches carry it.
    """
    return _group_statistics(df, "image")


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
