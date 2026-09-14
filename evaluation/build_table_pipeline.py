"""Build the LaTeX results table by reading the per-run folders under ``results/``.

A run lives in ``results/results_<segmentation>_<vlm>[_<depth>]_<association>`` and is
addressed by the four tools that produced it: the VLM, the segmentation model, the depth
model and the depth-association strategy. This module resolves those four names to a
folder, reads the metrics the evaluation pipeline left there, and prints the table body.

Localization metrics and the semantic-annotation recall come from ``summary.json`` (it is
already restricted to the objects counted in localization): the pixel position error, with
its variance, and the metric position error, the distance in millimetres between the 3D
point of the depth association and the ArUco position. Runs evaluated before those keys
existed fall back to ``evaluation.csv``, or render ``--``. Depth metrics, variance included,
are aggregated here from ``depth_evaluation.csv``, the same way ``run_evaluation`` plots them.

    python -m evaluation.build_table_pipeline --depth sensor --association mask-median
    python -m evaluation.build_table_pipeline --segmentation sam3 --vlm gpt54 --depth sensor \
        --association mask-median
"""

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Friendly name -> folder token. The keys are what the table prints, the values are how
# the runs were named on disk.
SEGMENTATION_DIRS = {
    "sam3": "sam3",
    "sam2-l": "sam21-l",
    "sam21-l": "sam21-l",
    "sam1-h": "sam1-h",
    "fastsam": "fastsam-s",
    "fastsam-s": "fastsam-s",
    "mobile-sam": "mobile-sam",
    "mobilesam": "mobile-sam",
    "only": "only",  # VLM alone, no segmentation stage
}

VLM_DIRS = {
    "sonnet46": "azure_claude-sonnet46",
    "claude-sonnet46": "azure_claude-sonnet46",
    "gpt52": "azure_gpt52",
    "gpt54": "azure_gpt54",
    "gpt54-mini": "azure_gpt54-mini",
    "gpt54-nano": "azure_gpt54-nano",
    "gpt55": "azure_gpt55",
    "qwen3-vl": "hf_Qwen_Qwen3-VL-4B-Instruct",
    "qwen36": "hf_Qwen_Qwen3.6-35B-A3B",
    "qwen3.6": "hf_Qwen_Qwen3.6-35B-A3B",
}

DEPTH_DIRS = {"sensor": "sensor", "rgbd": "sensor", "monocular": "monocular"}
ASSOCIATION_DIRS = {"mask-median": "mask-median", "bbox-center": "bbox-center"}

# How each tool is spelled in the LaTeX table.
SEGMENTATION_LABELS = {
    "sam3": "SAM3",
    "sam21-l": "SAM2-l",
    "sam1-h": "SAM1-h",
    "fastsam-s": "FastSAM",
    "mobile-sam": "MobileSAM",
    "only": "--",
}
VLM_LABELS = {
    "azure_claude-sonnet46": "Sonnet 4.6",
    "azure_gpt52": "GPT 5.2",
    "azure_gpt54": "GPT 5.4",
    "azure_gpt54-mini": "GPT 5.4-Mini",
    "azure_gpt54-nano": "GPT 5.4-Nano",
    "azure_gpt55": "GPT 5.5",
    "hf_Qwen_Qwen3-VL-4B-Instruct": "Qwen 3-VL",
    "hf_Qwen_Qwen3.6-35B-A3B": "Qwen 3.6",
}

# The rows of the table in the report, as (segmentation, vlm) pairs.
DEFAULT_ROWS = [
    ("sam3", "sonnet46"),
    ("sam3", "gpt54"),
    ("sam3", "gpt54-mini"),
    ("sam3", "gpt54-nano"),
    ("sam3", "qwen3-vl"),
    ("sam3", "qwen36"),
    ("sam2-l", "gpt54"),
    ("sam1-h", "gpt54"),
    ("fastsam", "gpt54"),
]


@dataclass
class RunMetrics:
    """Everything one table row needs, plus the folder it was read from."""

    segmentation: str
    vlm: str
    directory: Path
    recall_percent: float
    mean_error_px: float
    median_error_px: float
    rmse_px: float
    var_error_px: float | None
    position_mean_mm: float | None
    position_rmse_mm: float | None
    position_var_mm: float | None
    depth_mean_mm: float | None
    depth_rmse_mm: float | None
    depth_var_mm: float | None


def _resolve(name: str, table: dict[str, str], what: str) -> str:
    key = name.strip().lower()
    if key not in table:
        raise KeyError(f"unknown {what} {name!r}; known: {', '.join(sorted(table))}")
    return table[key]


def finite_values(values: Iterable[float]) -> np.ndarray:
    """Return the finite values of a sequence as a float array."""
    array = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def sample_variance(values: Iterable[float]) -> float | None:
    """Sample variance (``ddof=1``, as in ``summary.json``) of the finite values, or None below two."""
    errors = finite_values(values)
    return float(np.var(errors, ddof=1)) if errors.size > 1 else None


def read_localization_csv(evaluation_dir: Path) -> pd.DataFrame | None:
    """Read the per-match rows of a run that count towards localization.

    Returns None when the run has no ``evaluation.csv``.
    """
    path = evaluation_dir / "evaluation.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "counted_in_localization" in frame:
        frame = frame[frame["counted_in_localization"].astype(str).str.lower() == "true"]
    return frame


def run_directory(
    results_dir: Path, segmentation: str, vlm: str, depth: str, association: str
) -> Path:
    """Locate the folder holding one run.

    The early runs predate the depth model appearing in the folder name: they were all
    produced with the sensor depth, so ``results_<seg>_<vlm>_<association>`` is accepted
    as a fallback when ``--depth sensor`` finds no explicit folder.
    """
    seg = _resolve(segmentation, SEGMENTATION_DIRS, "segmentation model")
    model = _resolve(vlm, VLM_DIRS, "VLM")
    depth_token = _resolve(depth, DEPTH_DIRS, "depth model")
    assoc = _resolve(association, ASSOCIATION_DIRS, "association tool")

    candidates = [results_dir / f"results_{seg}_{model}_{depth_token}_{assoc}"]
    if depth_token == "sensor":
        candidates.append(results_dir / f"results_{seg}_{model}_{assoc}")
    if seg == "only":
        # The VLM-only ablation has neither a depth nor an association stage.
        candidates.append(results_dir / f"results_only_{model}")

    for candidate in candidates:
        if (candidate / "evaluation_results" / "summary.json").exists():
            return candidate
    raise FileNotFoundError(
        "no evaluated run found; looked for "
        + ", ".join(str(c.relative_to(results_dir)) for c in candidates)
    )


def read_metrics(
    results_dir: Path,
    segmentation: str,
    vlm: str,
    depth: str,
    association: str,
    include_ignored_depth: bool = True,
) -> RunMetrics:
    """Read one run's metrics off disk."""
    directory = run_directory(results_dir, segmentation, vlm, depth, association)
    evaluation_dir = directory / "evaluation_results"
    summary = json.loads((evaluation_dir / "summary.json").read_text())
    localization = read_localization_csv(evaluation_dir)

    # A run evaluated before these metrics existed has no key for them in summary.json.
    # Its evaluation.csv still gives the pixel variance, and the metric error if the rows
    # carry it; otherwise the cells stay None and render as --.
    var_error_px = summary.get("var_error_px")
    if "var_error_px" not in summary and localization is not None:
        var_error_px = sample_variance(localization["error_px"])

    position_mean = summary.get("mean_error_mm")
    position_rmse = summary.get("rmse_mm")
    position_var = summary.get("var_error_mm")
    if "mean_error_mm" not in summary and localization is not None and "error_mm" in localization:
        position_errors = finite_values(localization["error_mm"])
        if position_errors.size:
            position_mean = float(position_errors.mean())
            position_rmse = float(np.sqrt((position_errors**2).mean()))
        position_var = sample_variance(position_errors)

    depth_mean = depth_rmse = depth_var = None
    depth_csv = evaluation_dir / "depth_evaluation.csv"
    if depth_csv.exists():
        depth_df = pd.read_csv(depth_csv)
        if not include_ignored_depth and "ignored" in depth_df:
            depth_df = depth_df[depth_df["ignored"] == 0]
        errors = (
            finite_values(depth_df["abs_error_mm"]) if "abs_error_mm" in depth_df else np.array([])
        )
        if errors.size:
            depth_mean = float(errors.mean())
            depth_rmse = float(np.sqrt((errors**2).mean()))
        depth_var = sample_variance(errors)

    seg_token = _resolve(segmentation, SEGMENTATION_DIRS, "segmentation model")
    vlm_token = _resolve(vlm, VLM_DIRS, "VLM")
    return RunMetrics(
        segmentation=SEGMENTATION_LABELS.get(seg_token, seg_token),
        vlm=VLM_LABELS.get(vlm_token, vlm_token),
        directory=directory,
        recall_percent=100.0 * summary["mean_detection_recall"],
        mean_error_px=summary["mean_error_px"],
        median_error_px=summary["median_error_px"],
        rmse_px=summary["rmse_px"],
        var_error_px=var_error_px,
        position_mean_mm=position_mean,
        position_rmse_mm=position_rmse,
        position_var_mm=position_var,
        depth_mean_mm=depth_mean,
        depth_rmse_mm=depth_rmse,
        depth_var_mm=depth_var,
    )


def _cell(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def latex_table(rows: list[RunMetrics], caption_note: str = "") -> str:
    """Render the rows as the full tabular environment used in the report."""
    body = []
    for row in rows:
        cells = [
            f"{row.recall_percent:.1f}\\%",
            _cell(row.mean_error_px),
            _cell(row.median_error_px),
            _cell(row.rmse_px),
            _cell(row.var_error_px),
            _cell(row.position_mean_mm),
            _cell(row.position_rmse_mm),
            _cell(row.position_var_mm),
            _cell(row.depth_mean_mm),
            _cell(row.depth_rmse_mm),
            _cell(row.depth_var_mm),
        ]
        body.append(
            "    {:<12} & {:<12} & ".format(row.segmentation, row.vlm)
            + " & ".join(f"{cell:>7}" for cell in cells)
            + " \\\\"
        )
    header = [
        "    \\begin{tabular}{llccccccccccc}",
        "    \\toprule",
        "    \\multicolumn{2}{c}{Tools}           & Semantic Annotation "
        "& \\multicolumn{4}{c}{Position Error [px]} & \\multicolumn{3}{c}{Position Error [mm]} "
        "& \\multicolumn{3}{c}{Depth Error [mm]} \\\\",
        "    Segmentation & Annotation   & Recall              "
        "& Mean & Median & RMSE & Var & Mean & RMSE & Var & Mean & RMSE & Var \\\\",
        "    \\midrule",
    ]
    footer = ["    \\bottomrule", "    \\end{tabular}"]
    notes = ["variances in px^2 and mm^2"]
    if caption_note:
        notes.insert(0, caption_note)
    footer.append(f"    % {'; '.join(notes)}")
    return "\n".join(header + body + footer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory holding the results_* run folders",
    )
    parser.add_argument(
        "--segmentation",
        help="segmentation model of a single run; omit to render every default row",
    )
    parser.add_argument("--vlm", help="VLM of a single run")
    parser.add_argument(
        "--depth",
        default="sensor",
        help=f"depth model, one of {', '.join(sorted(set(DEPTH_DIRS)))} (default: sensor)",
    )
    parser.add_argument(
        "--association",
        default="mask-median",
        help="depth association tool, mask-median or bbox-center (default: mask-median)",
    )
    parser.add_argument(
        "--exclude-ignored-depth",
        action="store_true",
        help="drop the rows flagged 'ignored' in depth_evaluation.csv before aggregating",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="warn and continue instead of failing when a run has not been evaluated",
    )
    parser.add_argument("--csv", type=Path, help="also write the numbers to this CSV file")
    args = parser.parse_args(argv)
    if bool(args.segmentation) != bool(args.vlm):
        parser.error("--segmentation and --vlm must be given together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = [(args.segmentation, args.vlm)] if args.segmentation else list(DEFAULT_ROWS)

    rows: list[RunMetrics] = []
    for segmentation, vlm in pairs:
        try:
            rows.append(
                read_metrics(
                    args.results_dir,
                    segmentation,
                    vlm,
                    args.depth,
                    args.association,
                    include_ignored_depth=not args.exclude_ignored_depth,
                )
            )
        except (FileNotFoundError, KeyError) as exc:
            message = f"{segmentation} + {vlm}: {exc}"
            if not args.skip_missing:
                raise SystemExit(f"error: {message}") from exc
            print(f"warning: skipping {message}")

    note = f"depth={args.depth}, association={args.association}"
    print(latex_table(rows, caption_note=note))
    print()
    for row in rows:
        print(f"% {row.segmentation} + {row.vlm}: {row.directory.name}")

    if args.csv:
        pd.DataFrame(
            [
                {
                    "segmentation": r.segmentation,
                    "annotation": r.vlm,
                    "recall_percent": r.recall_percent,
                    "mean_error_px": r.mean_error_px,
                    "median_error_px": r.median_error_px,
                    "rmse_px": r.rmse_px,
                    "var_error_px": r.var_error_px,
                    "position_mean_mm": r.position_mean_mm,
                    "position_rmse_mm": r.position_rmse_mm,
                    "position_var_mm": r.position_var_mm,
                    "depth_mean_mm": r.depth_mean_mm,
                    "depth_rmse_mm": r.depth_rmse_mm,
                    "depth_var_mm": r.depth_var_mm,
                    "directory": r.directory.name,
                }
                for r in rows
            ]
        ).to_csv(args.csv, index=False)
        print(f"\n% wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
