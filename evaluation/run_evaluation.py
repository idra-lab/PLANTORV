from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matching import match_objects
from visualization import create_overlay
from metrics import (
    compute_global_metrics,
    object_statistics,
    image_statistics,
    save_summary,
    compute_detection_metrics,
)

from depth_correlation import compute_depth_correlation

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 12

RGB_DIR = Path("dataset/rgb")

SEG_DIR = Path("outputs_json_labeled")

ARUCO_DIR = Path("output_aruco")

OUTPUT_DIR = Path("results")
OVERLAY_DIR = OUTPUT_DIR / "overlays"

OUTPUT_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True)

all_results = []
detection_results = []
all_depth_results = []


for i in range(1, 105):
    rgb_file = RGB_DIR / f"rgb_dataset_{i}.png"

    seg_file = SEG_DIR / f"output_img{i}.json"

    aruco_file = ARUCO_DIR / f"rgb_dataset_aruco_{i}" / f"aruco_pos_img{i}.json"

    if not seg_file.exists():
        print(f"[MISSING] {seg_file}")
        continue

    if not aruco_file.exists():
        print(f"[MISSING] {aruco_file}")
        continue

    print(f"\nProcessing image {i}")

    with open(seg_file) as f:
        seg_data = json.load(f)

    with open(aruco_file) as f:
        aruco_data = json.load(f)

    detection_metrics = compute_detection_metrics(seg_data, aruco_data)

    detection_metrics["image"] = i

    detection_results.append(detection_metrics)

    matches = match_objects(seg_data, aruco_data)

    for match in matches:
        match["image"] = i

    all_results.extend(matches)

    if rgb_file.exists():
        create_overlay(
            image_path=str(rgb_file),
            matches=matches,
            output_path=str(OVERLAY_DIR / f"overlay_{i}.png"),
        )

    depth_df, ignored = compute_depth_correlation(seg_data, aruco_data)

    depth_df["image"] = i
    depth_df["ignored"] = ignored

    all_depth_results.append(depth_df)

    print(f"Matches found: {len(matches)}")

df = pd.DataFrame(all_results)

detection_df = pd.DataFrame(detection_results)

df.to_csv(OUTPUT_DIR / "evaluation.csv", index=False)

detection_df.to_csv(OUTPUT_DIR / "detection_statistics.csv", index=False)

obj_stats = object_statistics(df)

obj_stats.to_csv(OUTPUT_DIR / "object_statistics.csv", index=False)

img_stats = image_statistics(df)

img_stats.to_csv(OUTPUT_DIR / "image_statistics.csv", index=False)

summary = compute_global_metrics(df)

summary["mean_detection_recall"] = float(detection_df["recall"].mean())

summary["total_missed_objects"] = int(detection_df["missed_gt"].sum())

summary["total_extra_objects"] = int(detection_df["extra_objects"].sum())

save_summary(summary, OUTPUT_DIR / "summary.json")

print(summary)

plt.figure(figsize=(8, 5))

errors = df["error_px"].dropna()
mean_error = errors.mean()
median_error = errors.median()
rmse_error = np.sqrt((errors**2).mean())

plt.hist(errors, bins=20)

plt.axvline(mean_error, linestyle="--", linewidth=2, label=f"Mean = {mean_error:.2f}px")

plt.axvline(median_error, linestyle="-.", linewidth=2, label=f"Median = {median_error:.2f}px")

plt.axvline(rmse_error, linestyle=":", linewidth=2, label=f"RMSE = {rmse_error:.2f}px")

plt.xlabel(r"Localization Error ($\it{px}$)")
plt.ylabel("Count")
plt.title("Distribution of Localization Errors")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "error_distribution.png", dpi=300, bbox_inches="tight")

plt.close()

plt.figure(figsize=(12, 5))

plt.bar(obj_stats["aruco_name"], obj_stats["mean_error_px"])
global_mean = obj_stats["mean_error_px"].mean()

plt.axhline(global_mean, linestyle="--", linewidth=2, label=f"Global mean = {global_mean:.2f}px")

plt.xticks(rotation=45, ha="right")

plt.ylabel(r"Mean Error ($\it{px}$)")
plt.title("Mean Localization Error per Object")
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "mean_error_per_object.png", dpi=300, bbox_inches="tight")

plt.close()

plt.figure(figsize=(10, 5))

plt.plot(
    img_stats["image"], img_stats["mean_error_px"], marker="o", linewidth=1.5, label="Mean error"
)

global_mean = img_stats["mean_error_px"].mean()

plt.axhline(y=global_mean, linestyle="--", linewidth=2, label=f"Global mean = {global_mean:.2f}px")

global_std = img_stats["mean_error_px"].std()

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

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "mean_error_per_image.png", dpi=300, bbox_inches="tight")

plt.close()

plt.figure(figsize=(12, 6))

df.boxplot(column="error_px", by="aruco_name", vert=False)

plt.xlabel("Localization Error (px)")
plt.title("Error Distribution per Object")


plt.tight_layout()

plt.savefig(OUTPUT_DIR / "error_boxplot_per_object.png", dpi=300, bbox_inches="tight")

plt.close()

sorted_objects = obj_stats.sort_values(by="mean_error_px", ascending=False)

plt.figure(figsize=(12, 5))

plt.bar(sorted_objects["aruco_name"], sorted_objects["mean_error_px"])

plt.xticks(rotation=45, ha="right")

plt.ylabel("Mean Error (px)")
plt.xlabel("Object")

plt.title("Objects Ranked by Mean Localization Error")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "object_error_ranking.png", dpi=300, bbox_inches="tight")

plt.close()

plt.figure(figsize=(10, 5))

plt.plot(detection_df["image"], detection_df["recall"] * 100, marker="o", label="Recall")

mean_recall = detection_df["recall"].mean() * 100

plt.axhline(mean_recall, linestyle="--", linewidth=2, label=f"Mean Recall = {mean_recall:.1f}%")

plt.xlabel("Image")
plt.ylabel("Recall (%)")

plt.title("Object Detection Recall per Image")

plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "detection_recall_per_image.png", dpi=300, bbox_inches="tight")

plt.close()

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

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "missed_extra_objects.png", dpi=300, bbox_inches="tight")

plt.close()


"""Evaluation of depth correlation between RGB-D and ArUco-based measurements"""

depth_results = pd.concat(all_depth_results, ignore_index=True)


depth_results.to_csv(OUTPUT_DIR / "depth_evaluation.csv", index=False)


mean_error = depth_results["abs_error_mm"].mean()

std_error = depth_results["abs_error_mm"].std()

variance = depth_results["abs_error_mm"].var()

rmse = np.sqrt(np.mean(depth_results["abs_error_mm"] ** 2))

plt.figure(figsize=(8, 5))

plt.hist(depth_results["abs_error_mm"], bins=20)


plt.axvline(
    mean_error, linestyle="--", linewidth=2, color="red", label=f"Mean = {mean_error:.2f} mm"
)

plt.axvline(rmse, linestyle="-.", linewidth=2, color="green", label=f"RMSE = {rmse:.2f} mm")

plt.axvline(mean_error - std_error, linestyle=":", linewidth=1.5, label=f"-1σ = {std_error:.2f} mm")

plt.axvline(mean_error + std_error, linestyle=":", linewidth=1.5, label=f"+1σ = {std_error:.2f} mm")


plt.xlabel("Depth Error (mm)")
plt.ylabel("Count")
plt.title("Depth Error Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "depth_error_distribution.png", dpi=300, bbox_inches="tight")

plt.close()


depth_img_stats = (
    depth_results.groupby("image")
    .agg(
        mean_depth_error_mm=("abs_error_mm", "mean"),
        std_depth_error_mm=("abs_error_mm", "std"),
        rmse_depth_mm=("abs_error_mm", lambda x: np.sqrt(np.mean(x**2))),
        n_objects=("abs_error_mm", "count"),
    )
    .reset_index()
)

depth_img_stats.to_csv(OUTPUT_DIR / "depth_error_per_image.csv", index=False)

mean_depth = depth_img_stats["mean_depth_error_mm"].mean()

global_std = depth_img_stats["mean_depth_error_mm"].std()

plt.figure(figsize=(10, 5))

plt.plot(depth_img_stats["image"], depth_img_stats["mean_depth_error_mm"], marker="o")

plt.axhline(mean_depth, linestyle="--", linewidth=2, label=f"Global mean = {mean_depth:.2f} mm")

plt.axhline(mean_depth + global_std, linestyle=":", linewidth=1.5, label=f"+1σ = {global_std:.2f}")

plt.axhline(mean_depth - global_std, linestyle=":", linewidth=1.5)

plt.xlabel("Image")
plt.ylabel("Mean Depth Error (mm)")
plt.title("Mean Depth Error per Image")

plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "mean_depth_error_per_image.png", dpi=300, bbox_inches="tight")

plt.close()
