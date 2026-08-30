"""Run the evaluation pipeline comparing segmentation output against ArUco ground truth.

The pipeline walks the dataset image by image, matches detected objects to their ArUco
references, aggregates localization and depth errors, and writes CSV tables, a JSON
summary, and the figures used in the report to ``results/``.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .depth_correlation import compute_depth_correlation
from .matching import match_objects
from .metrics import (
    compute_detection_metrics,
    compute_global_metrics,
    image_statistics,
    object_statistics,
    save_summary,
)
from .visualization import create_overlay

RGB_DIR = Path("dataset/rgb")
SEG_DIR = Path("outputs_json_labeled")
ARUCO_DIR = Path("output_aruco")

OUTPUT_DIR = Path("results")
OVERLAY_DIR = OUTPUT_DIR / "overlays"

# Image indices to evaluate. Images whose segmentation or ArUco file is missing are skipped.
IMAGE_IDS = range(1, 105)

FIGURE_DPI = 300


def configure_plot_style() -> None:
    """Apply the report-wide matplotlib style."""
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 12


def load_json(path: Path) -> dict:
    """Read a JSON file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed contents of the file.
    """
    with open(path) as f:
        return json.load(f)


def save_figure(path: Path) -> None:
    """Lay out the current figure, write it to disk, and close it.

    Parameters
    ----------
    path : Path
        Destination file for the figure.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def process_image(image_id: int) -> tuple[dict, list[dict], pd.DataFrame] | None:
    """Evaluate a single image against its ArUco ground truth.

    Writes the overlay figure for the image as a side effect when the RGB file exists.

    Parameters
    ----------
    image_id : int
        Index of the image in the dataset.

    Returns
    -------
    tuple[dict, list[dict], pd.DataFrame] or None
        Detection metrics, per-object matches, and depth correlation rows for the image,
        or ``None`` when the segmentation or ArUco file is missing.
    """
    rgb_file = RGB_DIR / f"rgb_dataset_{image_id}.png"
    seg_file = SEG_DIR / f"output_img{image_id}.json"
    aruco_file = ARUCO_DIR / f"rgb_dataset_aruco_{image_id}" / f"aruco_pos_img{image_id}.json"

    if not seg_file.exists():
        print(f"[MISSING] {seg_file}")
        return None

    if not aruco_file.exists():
        print(f"[MISSING] {aruco_file}")
        return None

    print(f"\nProcessing image {image_id}")

    seg_data = load_json(seg_file)
    aruco_data = load_json(aruco_file)

    detection_metrics = compute_detection_metrics(seg_data, aruco_data)
    detection_metrics["image"] = image_id

    matches = match_objects(seg_data, aruco_data)
    for match in matches:
        match["image"] = image_id

    if rgb_file.exists():
        create_overlay(
            image_path=str(rgb_file),
            matches=matches,
            output_path=str(OVERLAY_DIR / f"overlay_{image_id}.png"),
        )

    depth_df, ignored = compute_depth_correlation(seg_data, aruco_data)
    depth_df["image"] = image_id
    depth_df["ignored"] = ignored

    print(f"Matches found: {len(matches)}")

    return detection_metrics, matches, depth_df


def collect_results(image_ids: range) -> tuple[list[dict], list[dict], list[pd.DataFrame]]:
    """Evaluate every image and gather the per-image results.

    Parameters
    ----------
    image_ids : range
        Image indices to evaluate.

    Returns
    -------
    tuple[list[dict], list[dict], list[pd.DataFrame]]
        Object matches across all images, per-image detection metrics, and per-image
        depth correlation tables.
    """
    all_results: list[dict] = []
    detection_results: list[dict] = []
    all_depth_results: list[pd.DataFrame] = []

    for image_id in image_ids:
        result = process_image(image_id)
        if result is None:
            continue

        detection_metrics, matches, depth_df = result
        detection_results.append(detection_metrics)
        all_results.extend(matches)
        all_depth_results.append(depth_df)

    return all_results, detection_results, all_depth_results


def build_summary(df: pd.DataFrame, detection_df: pd.DataFrame) -> dict:
    """Combine the global localization metrics with the detection totals.

    Parameters
    ----------
    df : pd.DataFrame
        Object matches across all images.
    detection_df : pd.DataFrame
        Per-image detection metrics.

    Returns
    -------
    dict
        Summary metrics for the whole run.
    """
    summary = compute_global_metrics(df)
    summary["mean_detection_recall"] = float(detection_df["recall"].mean())
    summary["total_missed_objects"] = int(detection_df["missed_gt"].sum())
    summary["total_extra_objects"] = int(detection_df["extra_objects"].sum())
    return summary


def depth_statistics_per_image(depth_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the depth errors into one row per image.

    Parameters
    ----------
    depth_results : pd.DataFrame
        Depth correlation rows for every image.

    Returns
    -------
    pd.DataFrame
        Mean, standard deviation, RMSE, and object count per image.
    """
    return (
        depth_results.groupby("image")
        .agg(
            mean_depth_error_mm=("abs_error_mm", "mean"),
            std_depth_error_mm=("abs_error_mm", "std"),
            rmse_depth_mm=("abs_error_mm", lambda x: np.sqrt(np.mean(x**2))),
            n_objects=("abs_error_mm", "count"),
        )
        .reset_index()
    )


def plot_error_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the histogram of localization errors with mean, median, and RMSE markers.

    Parameters
    ----------
    df : pd.DataFrame
        Object matches across all images.
    output_dir : Path
        Directory the figure is written to.
    """
    errors = df["error_px"].dropna()
    mean_error = float(errors.mean())
    median_error = float(errors.median())
    rmse_error = float(np.sqrt((errors**2).mean()))

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=20)
    plt.axvline(mean_error, linestyle="--", linewidth=2, label=f"Mean = {mean_error:.2f}px")
    plt.axvline(median_error, linestyle="-.", linewidth=2, label=f"Median = {median_error:.2f}px")
    plt.axvline(rmse_error, linestyle=":", linewidth=2, label=f"RMSE = {rmse_error:.2f}px")
    plt.xlabel(r"Localization Error ($\it{px}$)")
    plt.ylabel("Count")
    plt.title("Distribution of Localization Errors")
    plt.legend()

    save_figure(output_dir / "error_distribution.png")


def plot_mean_error_per_object(obj_stats: pd.DataFrame, output_dir: Path) -> None:
    """Plot the mean localization error of each object as a bar chart.

    Parameters
    ----------
    obj_stats : pd.DataFrame
        Per-object statistics.
    output_dir : Path
        Directory the figure is written to.
    """
    global_mean = float(obj_stats["mean_error_px"].mean())

    plt.figure(figsize=(12, 5))
    plt.bar(obj_stats["aruco_name"], obj_stats["mean_error_px"])
    plt.axhline(
        global_mean, linestyle="--", linewidth=2, label=f"Global mean = {global_mean:.2f}px"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(r"Mean Error ($\it{px}$)")
    plt.title("Mean Localization Error per Object")
    plt.legend()

    save_figure(output_dir / "mean_error_per_object.png")


def plot_mean_error_per_image(img_stats: pd.DataFrame, output_dir: Path) -> None:
    """Plot the mean localization error per image with a one-sigma band.

    Parameters
    ----------
    img_stats : pd.DataFrame
        Per-image statistics.
    output_dir : Path
        Directory the figure is written to.
    """
    global_mean = float(img_stats["mean_error_px"].mean())
    global_std = float(img_stats["mean_error_px"].std())

    plt.figure(figsize=(10, 5))
    plt.plot(
        img_stats["image"],
        img_stats["mean_error_px"],
        marker="o",
        linewidth=1.5,
        label="Mean error",
    )
    plt.axhline(
        y=global_mean, linestyle="--", linewidth=2, label=f"Global mean = {global_mean:.2f}px"
    )
    plt.axhspan(
        global_mean - global_std,
        global_mean + global_std,
        alpha=0.15,
        label=f"±1σ ({global_std:.2f}px)",
    )
    plt.xticks(rotation=90)
    plt.xlabel("Image")
    plt.ylabel(r"Mean Error ($\it{px}$)")
    plt.title("Mean Localization Error per Image")
    plt.legend()

    save_figure(output_dir / "mean_error_per_image.png")


def plot_error_boxplot_per_object(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the error distribution of each object as a horizontal box plot.

    Parameters
    ----------
    df : pd.DataFrame
        Object matches across all images.
    output_dir : Path
        Directory the figure is written to.
    """
    plt.figure(figsize=(12, 6))
    df.boxplot(column="error_px", by="aruco_name", vert=False)
    plt.xlabel("Localization Error (px)")
    plt.title("Error Distribution per Object")

    save_figure(output_dir / "error_boxplot_per_object.png")


def plot_object_error_ranking(obj_stats: pd.DataFrame, output_dir: Path) -> None:
    """Plot objects ranked from the highest to the lowest mean localization error.

    Parameters
    ----------
    obj_stats : pd.DataFrame
        Per-object statistics.
    output_dir : Path
        Directory the figure is written to.
    """
    sorted_objects = obj_stats.sort_values(by="mean_error_px", ascending=False)

    plt.figure(figsize=(12, 5))
    plt.bar(sorted_objects["aruco_name"], sorted_objects["mean_error_px"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Error (px)")
    plt.xlabel("Object")
    plt.title("Objects Ranked by Mean Localization Error")

    save_figure(output_dir / "object_error_ranking.png")


def plot_detection_recall_per_image(detection_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the object detection recall of each image.

    Parameters
    ----------
    detection_df : pd.DataFrame
        Per-image detection metrics.
    output_dir : Path
        Directory the figure is written to.
    """
    mean_recall = float(detection_df["recall"].mean()) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(detection_df["image"], detection_df["recall"] * 100, marker="o", label="Recall")
    plt.axhline(mean_recall, linestyle="--", linewidth=2, label=f"Mean Recall = {mean_recall:.1f}%")
    plt.xlabel("Image")
    plt.ylabel("Recall (%)")
    plt.title("Object Detection Recall per Image")
    plt.legend()

    save_figure(output_dir / "detection_recall_per_image.png")


def plot_missed_extra_objects(detection_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot missed and extra object counts per image as a stacked bar chart.

    Parameters
    ----------
    detection_df : pd.DataFrame
        Per-image detection metrics.
    output_dir : Path
        Directory the figure is written to.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(detection_df["image"], detection_df["missed_gt"], label="Missed")
    plt.bar(
        detection_df["image"],
        detection_df["extra_objects"],
        bottom=detection_df["missed_gt"],
        label="Extra",
    )
    plt.xlabel("Image")
    plt.ylabel("Count")
    plt.title("Missed and Extra Objects per Image")
    plt.legend()

    save_figure(output_dir / "missed_extra_objects.png")


def plot_depth_error_distribution(depth_results: pd.DataFrame, output_dir: Path) -> None:
    """Plot the histogram of depth errors with mean, RMSE, and one-sigma markers.

    Parameters
    ----------
    depth_results : pd.DataFrame
        Depth correlation rows for every image.
    output_dir : Path
        Directory the figure is written to.
    """
    mean_error = float(depth_results["abs_error_mm"].mean())
    std_error = float(depth_results["abs_error_mm"].std())
    rmse = float(np.sqrt(np.mean(depth_results["abs_error_mm"] ** 2)))

    plt.figure(figsize=(8, 5))
    plt.hist(depth_results["abs_error_mm"], bins=20)
    plt.axvline(
        mean_error, linestyle="--", linewidth=2, color="red", label=f"Mean = {mean_error:.2f} mm"
    )
    plt.axvline(rmse, linestyle="-.", linewidth=2, color="green", label=f"RMSE = {rmse:.2f} mm")
    plt.axvline(
        mean_error - std_error, linestyle=":", linewidth=1.5, label=f"-1σ = {std_error:.2f} mm"
    )
    plt.axvline(
        mean_error + std_error, linestyle=":", linewidth=1.5, label=f"+1σ = {std_error:.2f} mm"
    )
    plt.xlabel("Depth Error (mm)")
    plt.ylabel("Count")
    plt.title("Depth Error Distribution")
    plt.legend()

    save_figure(output_dir / "depth_error_distribution.png")


def plot_mean_depth_error_per_image(depth_img_stats: pd.DataFrame, output_dir: Path) -> None:
    """Plot the mean depth error per image with one-sigma reference lines.

    Parameters
    ----------
    depth_img_stats : pd.DataFrame
        Per-image depth statistics.
    output_dir : Path
        Directory the figure is written to.
    """
    mean_depth = float(depth_img_stats["mean_depth_error_mm"].mean())
    global_std = float(depth_img_stats["mean_depth_error_mm"].std())

    plt.figure(figsize=(10, 5))
    plt.plot(depth_img_stats["image"], depth_img_stats["mean_depth_error_mm"], marker="o")
    plt.axhline(mean_depth, linestyle="--", linewidth=2, label=f"Global mean = {mean_depth:.2f} mm")
    plt.axhline(
        mean_depth + global_std, linestyle=":", linewidth=1.5, label=f"+1σ = {global_std:.2f}"
    )
    plt.axhline(mean_depth - global_std, linestyle=":", linewidth=1.5)
    plt.xlabel("Image")
    plt.ylabel("Mean Depth Error (mm)")
    plt.title("Mean Depth Error per Image")
    plt.legend()

    save_figure(output_dir / "mean_depth_error_per_image.png")


def evaluate_localization(
    df: pd.DataFrame, detection_df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the localization tables and figures.

    Parameters
    ----------
    df : pd.DataFrame
        Object matches across all images.
    detection_df : pd.DataFrame
        Per-image detection metrics.
    output_dir : Path
        Directory the tables and figures are written to.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Per-object and per-image statistics.
    """
    df.to_csv(output_dir / "evaluation.csv", index=False)
    detection_df.to_csv(output_dir / "detection_statistics.csv", index=False)

    obj_stats = object_statistics(df)
    obj_stats.to_csv(output_dir / "object_statistics.csv", index=False)

    img_stats = image_statistics(df)
    img_stats.to_csv(output_dir / "image_statistics.csv", index=False)

    summary = build_summary(df, detection_df)
    save_summary(summary, str(output_dir / "summary.json"))
    print(summary)

    plot_error_distribution(df, output_dir)
    plot_mean_error_per_object(obj_stats, output_dir)
    plot_mean_error_per_image(img_stats, output_dir)
    plot_error_boxplot_per_object(df, output_dir)
    plot_object_error_ranking(obj_stats, output_dir)
    plot_detection_recall_per_image(detection_df, output_dir)
    plot_missed_extra_objects(detection_df, output_dir)

    return obj_stats, img_stats


def evaluate_depth(all_depth_results: list[pd.DataFrame], output_dir: Path) -> None:
    """Write the depth correlation tables and figures.

    Parameters
    ----------
    all_depth_results : list[pd.DataFrame]
        Per-image depth correlation tables.
    output_dir : Path
        Directory the tables and figures are written to.
    """
    depth_results = pd.concat(all_depth_results, ignore_index=True)
    depth_results.to_csv(output_dir / "depth_evaluation.csv", index=False)

    plot_depth_error_distribution(depth_results, output_dir)

    depth_img_stats = depth_statistics_per_image(depth_results)
    depth_img_stats.to_csv(output_dir / "depth_error_per_image.csv", index=False)

    plot_mean_depth_error_per_image(depth_img_stats, output_dir)


def main() -> None:
    """Run the full evaluation and write every table and figure to ``results/``."""
    configure_plot_style()

    OUTPUT_DIR.mkdir(exist_ok=True)
    OVERLAY_DIR.mkdir(exist_ok=True)

    all_results, detection_results, all_depth_results = collect_results(IMAGE_IDS)

    df = pd.DataFrame(all_results)
    detection_df = pd.DataFrame(detection_results)

    evaluate_localization(df, detection_df, OUTPUT_DIR)
    evaluate_depth(all_depth_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
