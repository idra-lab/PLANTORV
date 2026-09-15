"""Build the LaTeX results table by reading the per-run folders under ``results/``.

A run lives in ``results/results_<segmentation>_<vlm>[_<depth>]_<association>`` and is
addressed by the four tools that produced it: the VLM, the segmentation model, the depth
model and the depth-association strategy. This module resolves those four names to a
folder, reads the metrics the evaluation pipeline left there, and prints the table body.

The table shows the semantic-annotation recall, and the median and RMSE of the pixel position
error and of the depth error. ``--position-mm`` adds the same two statistics of the metric
position error, the distance in millimetres between the 3D point of the depth association
and the ArUco position. Localization
metrics and the recall come from ``summary.json`` (it is already restricted to the objects
counted in localization); runs evaluated before those keys existed fall back to
``evaluation.csv``, or render ``--``. Depth metrics are aggregated here from
``depth_evaluation.csv``, the same way ``run_evaluation`` plots them.

``--confidence-intervals`` follows every error with a column holding its percentile bootstrap
confidence interval, written ``[5th percentile, 95th percentile]`` of the bootstrap
estimates; ``--ci-maths`` writes the same bounds on the value instead, as
``$value^{95th}_{5th}$``, and implies ``--confidence-intervals``. The recall never shows an
interval. The bootstrap resamples images rather than objects, from ``evaluation.csv``,
``depth_evaluation.csv`` and ``detection_statistics.csv``; an error whose rows are missing
gets no interval.
The CSV written by ``--csv`` also keeps the mean and the variance of every error.

    python -m evaluation.build_table_pipeline --depth sensor --association mask-median
    python -m evaluation.build_table_pipeline --position-mm --confidence-intervals
    python -m evaluation.build_table_pipeline --ci-maths
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

from .metrics import BOOTSTRAP_CONFIDENCE, BOOTSTRAP_RESAMPLES, bootstrap_interval

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


# A confidence interval as (lower, upper) bounds, or None when it could not be computed.
Interval = tuple[float, float] | None


@dataclass
class RunMetrics:
    """Everything one table row needs, plus the folder it was read from.

    The ``*_ci`` fields are the confidence intervals of the statistic they follow.
    """

    segmentation: str
    vlm: str
    directory: Path
    recall_percent: float
    recall_ci: Interval
    mean_error_px: float
    mean_px_ci: Interval
    median_error_px: float
    median_px_ci: Interval
    rmse_px: float
    rmse_px_ci: Interval
    var_error_px: float | None
    position_mean_mm: float | None
    position_median_mm: float | None
    position_median_ci: Interval
    position_rmse_mm: float | None
    position_rmse_ci: Interval
    position_var_mm: float | None
    depth_mean_mm: float | None
    depth_mean_ci: Interval
    depth_median_mm: float | None
    depth_median_ci: Interval
    depth_rmse_mm: float | None
    depth_rmse_ci: Interval
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


def image_groups(frame: pd.DataFrame) -> pd.Series | None:
    """Return the image of every row, the unit the bootstrap resamples; None when rows lack it."""
    return frame["image"] if "image" in frame else None


def median_and_rmse_intervals(
    values: pd.Series, groups: pd.Series | None
) -> tuple[Interval, Interval]:
    """Compute the confidence intervals of the median and RMSE of ``values``, resampling ``groups``."""
    return bootstrap_interval(values, "median", groups), bootstrap_interval(values, "rmse", groups)


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

    mean_px_ci = median_px_ci = rmse_px_ci = None
    if localization is not None:
        groups = image_groups(localization)
        mean_px_ci = bootstrap_interval(localization["error_px"], "mean", groups)
        median_px_ci, rmse_px_ci = median_and_rmse_intervals(localization["error_px"], groups)

    position_mean = summary.get("mean_error_mm")
    position_median = summary.get("median_error_mm")
    position_rmse = summary.get("rmse_mm")
    position_var = summary.get("var_error_mm")
    position_median_ci = position_rmse_ci = None
    if localization is not None and "error_mm" in localization:
        if "mean_error_mm" not in summary:
            position_errors = finite_values(localization["error_mm"])
            if position_errors.size:
                position_mean = float(position_errors.mean())
                position_median = float(np.median(position_errors))
                position_rmse = float(np.sqrt((position_errors**2).mean()))
            position_var = sample_variance(position_errors)
        position_median_ci, position_rmse_ci = median_and_rmse_intervals(
            localization["error_mm"], image_groups(localization)
        )

    depth_mean = depth_median = depth_rmse = depth_var = None
    depth_mean_ci = depth_median_ci = depth_rmse_ci = None
    depth_csv = evaluation_dir / "depth_evaluation.csv"
    if depth_csv.exists():
        depth_df = pd.read_csv(depth_csv)
        if not include_ignored_depth and "ignored" in depth_df:
            depth_df = depth_df[depth_df["ignored"] == 0]
        if "abs_error_mm" in depth_df:
            errors = finite_values(depth_df["abs_error_mm"])
            if errors.size:
                depth_mean = float(errors.mean())
                depth_median = float(np.median(errors))
                depth_rmse = float(np.sqrt((errors**2).mean()))
            depth_var = sample_variance(errors)
            depth_groups = image_groups(depth_df)
            depth_mean_ci = bootstrap_interval(depth_df["abs_error_mm"], "mean", depth_groups)
            depth_median_ci, depth_rmse_ci = median_and_rmse_intervals(
                depth_df["abs_error_mm"], depth_groups
            )

    # detection_statistics.csv holds one row per image, so resampling its rows resamples images.
    recall_ci = None
    detection_csv = evaluation_dir / "detection_statistics.csv"
    if detection_csv.exists():
        interval = bootstrap_interval(pd.read_csv(detection_csv)["recall"], "mean")
        if interval is not None:
            recall_ci = (100.0 * interval[0], 100.0 * interval[1])

    seg_token = _resolve(segmentation, SEGMENTATION_DIRS, "segmentation model")
    vlm_token = _resolve(vlm, VLM_DIRS, "VLM")
    return RunMetrics(
        segmentation=SEGMENTATION_LABELS.get(seg_token, seg_token),
        vlm=VLM_LABELS.get(vlm_token, vlm_token),
        directory=directory,
        recall_percent=100.0 * summary["mean_detection_recall"],
        recall_ci=recall_ci,
        mean_error_px=summary["mean_error_px"],
        mean_px_ci=mean_px_ci,
        median_error_px=summary["median_error_px"],
        median_px_ci=median_px_ci,
        rmse_px=summary["rmse_px"],
        rmse_px_ci=rmse_px_ci,
        var_error_px=var_error_px,
        position_mean_mm=position_mean,
        position_median_mm=position_median,
        position_median_ci=position_median_ci,
        position_rmse_mm=position_rmse,
        position_rmse_ci=position_rmse_ci,
        position_var_mm=position_var,
        depth_mean_mm=depth_mean,
        depth_mean_ci=depth_mean_ci,
        depth_median_mm=depth_median,
        depth_median_ci=depth_median_ci,
        depth_rmse_mm=depth_rmse,
        depth_rmse_ci=depth_rmse_ci,
        depth_var_mm=depth_var,
    )


def _cell(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _interval_cell(interval: Interval, digits: int = 1) -> str:
    """Format a confidence interval as ``[lower, upper]``, or ``--`` when there is none."""
    if interval is None:
        return "--"
    low, high = interval
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _maths_cell(value: float | None, interval: Interval, digits: int = 1) -> str:
    """Format a value with its confidence interval as ``$value^{upper}_{lower}$``."""
    if value is None or interval is None:
        return _cell(value, digits)
    low, high = interval
    return f"${value:.{digits}f}^{{{high:.{digits}f}}}_{{{low:.{digits}f}}}$"


def _cells_with_interval(
    value: float | None, interval: Interval, ci_style: str | None
) -> list[str]:
    """Return the cells of a value, with its confidence interval printed as ``ci_style`` says.

    None prints the value alone, ``"column"`` follows it with a ``[lower, upper]`` cell, and
    ``"maths"`` writes the bounds on the value itself, as ``$value^{upper}_{lower}$``.
    """
    if ci_style == "column":
        return [_cell(value), _interval_cell(interval)]
    if ci_style == "maths":
        return [_maths_cell(value, interval)]
    return [_cell(value)]


def sc(title: str) -> str:
    """Set a column title in small capitals."""
    return f"{{\\sc {title}}}"


def metric_headers(name: str, ci_style: str | None) -> list[str]:
    """Return the header of a metric, followed by the header of its interval column if any."""
    headers = [name, interval_header()] if ci_style == "column" else [name]
    return [sc(header) for header in headers]


def _percentiles() -> tuple[str, str]:
    """Return the percentiles the bounds of a confidence interval are, e.g. 5 and 95."""
    tail = 100 * (1 - BOOTSTRAP_CONFIDENCE) / 2
    return f"{tail:g}", f"{100 - tail:g}"


def interval_header() -> str:
    """Return the header of a confidence-interval column: its two percentiles, e.g. 5 and 95."""
    low, high = _percentiles()
    return f"[{low}\\%, {high}\\%]"


def confidence_note(ci_style: str) -> str:
    """Return the table comment saying how the confidence intervals are written and computed."""
    low, high = _percentiles()
    notation = interval_header() if ci_style == "column" else f"value^{{{high}\\%}}_{{{low}\\%}}"
    return (
        f"{notation}: {100 * BOOTSTRAP_CONFIDENCE:.0f}% percentile bootstrap "
        f"confidence interval, {BOOTSTRAP_RESAMPLES} resamples of the images"
    )


def latex_table(
    rows: list[RunMetrics],
    caption_note: str = "",
    position_mm: bool = False,
    ci_style: str | None = None,
) -> str:
    """Render the rows as the full tabular environment used in the report.

    The metric position error [mm] is shown only when ``position_mm`` is set. The confidence
    intervals of the errors are printed as ``ci_style`` says (see :func:`_cells_with_interval`);
    the recall never shows one, although the CSV still carries it.
    """
    body = []
    for row in rows:
        cells = [_cell(row.recall_percent)]
        cells += _cells_with_interval(row.median_error_px, row.median_px_ci, ci_style)
        cells += _cells_with_interval(row.rmse_px, row.rmse_px_ci, ci_style)
        if position_mm:
            cells += _cells_with_interval(row.position_median_mm, row.position_median_ci, ci_style)
            cells += _cells_with_interval(row.position_rmse_mm, row.position_rmse_ci, ci_style)
        cells += _cells_with_interval(row.depth_median_mm, row.depth_median_ci, ci_style)
        cells += _cells_with_interval(row.depth_rmse_mm, row.depth_rmse_ci, ci_style)
        body.append(
            "    {:<12} & {:<12} & ".format(row.segmentation, row.vlm)
            + " & ".join(f"{cell:>14}" for cell in cells)
            + " \\\\"
        )
    error_groups = ["Position Error [px]"]
    if position_mm:
        error_groups.append("Position Error [mm]")
    error_groups.append("Depth Error [mm]")
    metrics = metric_headers("Median", ci_style) + metric_headers("RMSE", ci_style)
    header = [
        f"    \\begin{{tabular}}{{ll{'c' * (1 + len(metrics) * len(error_groups))}}}",
        "    \\toprule",
        f"    \\multicolumn{{2}}{{c}}{{{sc('Tools')}}} & {sc('Semantic Annotation')} & "
        + " & ".join(
            f"\\multicolumn{{{len(metrics)}}}{{c}}{{{sc(group)}}}" for group in error_groups
        )
        + " \\\\",
        "    "
        + " & ".join(sc(title) for title in ("Segmentation", "Annotation", "Recall [\\%]"))
        + " & "
        + " & ".join(" & ".join(metrics) for _ in error_groups)
        + " \\\\",
        "    \\midrule",
    ]
    footer = ["    \\bottomrule", "    \\end{tabular}"]
    notes = [caption_note] if caption_note else []
    if ci_style is not None:
        notes.append(confidence_note(ci_style))
    if notes:
        footer.append(f"    % {'; '.join(notes)}")
    return "\n".join(header + body + footer)


def interval_columns(name: str, interval: Interval) -> dict[str, float | None]:
    """Return the ``<name>_ci_low`` and ``<name>_ci_high`` CSV columns of a confidence interval."""
    low, high = interval if interval is not None else (None, None)
    return {f"{name}_ci_low": low, f"{name}_ci_high": high}


def add_confidence_interval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags choosing whether, and how, the confidence intervals are printed."""
    parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="print the confidence interval of every error, as [lower, upper] in a column after it",
    )
    parser.add_argument(
        "--ci-maths",
        action="store_true",
        help=(
            "print the confidence intervals as $value^{upper}_{lower}$ in the value's own cell "
            "instead of in a column after it; implies --confidence-intervals"
        ),
    )


def confidence_interval_style(args: argparse.Namespace) -> str | None:
    """Return how the flags ask for the confidence intervals: None, ``"column"`` or ``"maths"``."""
    if args.ci_maths:
        return "maths"
    return "column" if args.confidence_intervals else None


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
        "--position-mm",
        action="store_true",
        help=(
            "also show the metric position error [mm], the distance between the 3D point of "
            "the depth association and the ArUco position"
        ),
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
    add_confidence_interval_arguments(parser)
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
    print(
        latex_table(
            rows,
            caption_note=note,
            position_mm=args.position_mm,
            ci_style=confidence_interval_style(args),
        )
    )
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
                    **interval_columns("recall_percent", r.recall_ci),
                    "mean_error_px": r.mean_error_px,
                    **interval_columns("mean_error_px", r.mean_px_ci),
                    "median_error_px": r.median_error_px,
                    **interval_columns("median_error_px", r.median_px_ci),
                    "rmse_px": r.rmse_px,
                    **interval_columns("rmse_px", r.rmse_px_ci),
                    "var_error_px": r.var_error_px,
                    "position_mean_mm": r.position_mean_mm,
                    "position_median_mm": r.position_median_mm,
                    **interval_columns("position_median_mm", r.position_median_ci),
                    "position_rmse_mm": r.position_rmse_mm,
                    **interval_columns("position_rmse_mm", r.position_rmse_ci),
                    "position_var_mm": r.position_var_mm,
                    "depth_mean_mm": r.depth_mean_mm,
                    **interval_columns("depth_mean_mm", r.depth_mean_ci),
                    "depth_median_mm": r.depth_median_mm,
                    **interval_columns("depth_median_mm", r.depth_median_ci),
                    "depth_rmse_mm": r.depth_rmse_mm,
                    **interval_columns("depth_rmse_mm", r.depth_rmse_ci),
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
