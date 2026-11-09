"""Build the LaTeX depth-ablation table from the per-run result folders.

By default each depth technique is scored on *its own* valid depth estimates. This is
intentional: the RGB-D + mask-median numbers then reproduce the depth columns of the
main pipeline table. Use ``--common-objects`` only when you explicitly want a paired
comparison restricted to object keys that are present in every available technique.

Every technique shows the median and the RMSE of its depth error. ``--confidence-intervals``
follows each with a column holding its percentile bootstrap confidence interval, written
``[5th percentile, 95th percentile]`` of the bootstrap estimates and computed by resampling
images; ``--ci-maths`` writes the bounds on the value instead, as ``$value^{95th}_{5th}$``.
``--position-mm`` adds the same two statistics of the metric position error, the distance
in millimetres between the 3D point of the technique's association and the ArUco position,
read from ``evaluation.csv``. The CSV written by
``--csv`` always holds every number, the mean and the variance of the errors included.

Missing runs are never resolved fuzzily. With ``--skip-missing`` they are rendered as
``--`` cells; otherwise the script fails and reports the exact expected run names.

Examples
--------
    python -m evaluation.build_table_depth_ablation
    python -m evaluation.build_table_depth_ablation --segmentation sam3 --vlm gpt54
    python -m evaluation.build_table_depth_ablation --common-objects
    python -m evaluation.build_table_depth_ablation --skip-missing --coverage
    python -m evaluation.build_table_depth_ablation --position-mm --confidence-intervals
    python -m evaluation.build_table_depth_ablation --ci-maths
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .build_table_pipeline import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_ROWS,
    SEGMENTATION_DIRS,
    SEGMENTATION_LABELS,
    VLM_DIRS,
    VLM_LABELS,
    Interval,
    _cells_with_interval,
    _resolve,
    add_confidence_interval_arguments,
    confidence_interval_style,
    confidence_note,
    image_groups,
    interval_columns,
    median_and_rmse_intervals,
    metric_headers,
    read_localization_csv,
    sc,
)

# (depth source, association) in the order the columns appear.
DEFAULT_TECHNIQUES = [
    ("sensor", "mask-median"),
    ("sensor", "bbox-center"),
    ("monocular", "mask-median"),
    ("monocular", "bbox-center"),
]

DEPTH_LABELS = {"sensor": "RGB-D", "rgbd": "RGB-D", "monocular": "DA3"}
ASSOCIATION_LABELS = {"mask-median": "Mask median", "bbox-center": "BBox center"}

# This is useful only for the optional paired/common-object analysis.  The normal table
# does not need to match rows across independently evaluated techniques.
OBJECT_KEY = ["image", "mask_id", "object_name"]


@dataclass
class TechniqueMetrics:
    depth: str
    association: str
    directory: Path | None
    n_valid: int
    n_scored: int
    n_attempted: int | None
    coverage_percent: float | None
    mean_mm: float | None
    median_mm: float | None
    rmse_mm: float | None
    var_mm: float | None = None
    median_ci: Interval = None
    rmse_ci: Interval = None
    n_position: int = 0
    position_mean_mm: float | None = None
    position_median_mm: float | None = None
    position_rmse_mm: float | None = None
    position_var_mm: float | None = None
    position_median_ci: Interval = None
    position_rmse_ci: Interval = None
    missing: bool = False


@dataclass
class AblationRow:
    segmentation: str
    vlm: str
    techniques: list[TechniqueMetrics]


def _errors_stats(
    frame: pd.DataFrame, column: str
) -> tuple[float | None, float | None, float | None, float | None, Interval, Interval]:
    """Compute the statistics of the finite errors in ``column``.

    Returns the mean, median, RMSE and sample variance (``ddof=1``), then the confidence
    intervals of the median and of the RMSE, resampling the images.
    """
    errors = frame[column].to_numpy(dtype=float)
    errors = errors[np.isfinite(errors)]
    if not errors.size:
        return None, None, None, None, None, None
    return (
        float(errors.mean()),
        float(np.median(errors)),
        float(np.sqrt(np.mean(errors**2))),
        float(np.var(errors, ddof=1)) if errors.size > 1 else None,
        *median_and_rmse_intervals(frame[column], image_groups(frame)),
    )


def _exact_run_directory(
    results_dir: Path,
    segmentation: str,
    vlm: str,
    depth: str,
    association: str,
) -> Path:
    """Resolve a run by exact directory names only.

    Older sensor runs omitted the explicit ``sensor`` token, so both exact historical
    names are supported.  No glob/fuzzy fallback is used: a missing run must not be
    silently replaced by a different configuration.
    """
    seg_token = _resolve(segmentation, SEGMENTATION_DIRS, "segmentation model")
    vlm_token = _resolve(vlm, VLM_DIRS, "VLM")

    if depth in {"sensor", "rgbd"}:
        candidates = [
            results_dir / f"results_{seg_token}_{vlm_token}_sensor_{association}",
            results_dir / f"results_{seg_token}_{vlm_token}_{association}",  # legacy
        ]
    elif depth == "monocular":
        candidates = [results_dir / f"results_{seg_token}_{vlm_token}_monocular_{association}"]
    else:
        candidates = [results_dir / f"results_{seg_token}_{vlm_token}_{depth}_{association}"]

    existing = [path for path in candidates if path.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        # Prefer the explicit modern sensor name if both a legacy and a modern run exist.
        return existing[0]

    expected = " or ".join(path.name for path in candidates)
    raise FileNotFoundError(f"missing run directory: {expected}")


def _read_depth_csv(directory: Path) -> tuple[pd.DataFrame, int | None, float | None]:
    depth_csv = directory / "evaluation_results" / "depth_evaluation.csv"
    if not depth_csv.exists():
        raise FileNotFoundError(f"{depth_csv} is missing")

    frame = pd.read_csv(depth_csv)
    required = set(OBJECT_KEY + ["abs_error_mm"])
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{depth_csv} is missing required columns: {sorted(missing_columns)}")

    frame = frame.copy()
    frame["abs_error_mm"] = pd.to_numeric(frame["abs_error_mm"], errors="coerce")
    frame = frame[np.isfinite(frame["abs_error_mm"])].copy()

    # IMPORTANT: ``ignored`` is NOT a per-row ignore flag.  In run_evaluation.py it is
    # the number of invalid/missing depth estimates for the whole image, repeated on all
    # valid rows from that image.  Therefore we must not filter rows with ignored != 0.
    n_attempted: int | None = None
    coverage_percent: float | None = None
    if "ignored" in frame.columns and "image" in frame.columns:
        ignored = pd.to_numeric(frame["ignored"], errors="coerce").fillna(0)
        tmp = frame.assign(_ignored_count=ignored)
        ignored_total = int(tmp.groupby("image")["_ignored_count"].max().sum())
        n_attempted = len(frame) + ignored_total
        if n_attempted:
            coverage_percent = 100.0 * len(frame) / n_attempted

    return frame, n_attempted, coverage_percent


def _read_position_frame(directory: Path) -> pd.DataFrame | None:
    """Read the metric position errors of a run, keyed like the depth rows.

    Returns None for a run evaluated before ``evaluation.csv`` carried ``error_mm``.
    """
    frame = read_localization_csv(directory / "evaluation_results")
    if frame is None or "error_mm" not in frame:
        return None
    frame = frame.rename(columns={"aruco_name": "object_name"}).copy()
    frame["error_mm"] = pd.to_numeric(frame["error_mm"], errors="coerce")
    return frame[np.isfinite(frame["error_mm"])].copy()


def read_row(
    results_dir: Path,
    segmentation: str,
    vlm: str,
    techniques: list[tuple[str, str]],
    common_objects: bool = False,
    skip_missing: bool = False,
) -> AblationRow:
    """Read the depth and metric position errors of the requested techniques for one model pair."""
    loaded: dict[
        tuple[str, str],
        tuple[Path, pd.DataFrame, int | None, float | None, pd.DataFrame | None],
    ] = {}
    missing: set[tuple[str, str]] = set()

    for depth, association in techniques:
        try:
            directory = _exact_run_directory(results_dir, segmentation, vlm, depth, association)
            frame, n_attempted, coverage_percent = _read_depth_csv(directory)
            loaded[(depth, association)] = (
                directory,
                frame,
                n_attempted,
                coverage_percent,
                _read_position_frame(directory),
            )
        except (FileNotFoundError, ValueError) as exc:
            if not skip_missing:
                raise
            print(f"warning: {segmentation} + {vlm} + {depth}:{association}: {exc}")
            missing.add((depth, association))

    # Optional paired comparison.  This is deliberately opt-in because restricting all
    # techniques to the bbox-center-valid subset changes the headline numbers and biases
    # the mask-median result toward the easier cases.
    shared: pd.DataFrame | None = None
    if common_objects and loaded:
        key_frames = [frame[OBJECT_KEY].drop_duplicates() for _, frame, _, _, _ in loaded.values()]
        shared = key_frames[0]
        for keys in key_frames[1:]:
            shared = shared.merge(keys, on=OBJECT_KEY, how="inner")
        shared = shared.drop_duplicates()

    metrics: list[TechniqueMetrics] = []
    for depth, association in techniques:
        key = (depth, association)
        if key in missing or key not in loaded:
            metrics.append(
                TechniqueMetrics(
                    depth=depth,
                    association=association,
                    directory=None,
                    n_valid=0,
                    n_scored=0,
                    n_attempted=None,
                    coverage_percent=None,
                    mean_mm=None,
                    median_mm=None,
                    rmse_mm=None,
                    missing=True,
                )
            )
            continue

        directory, frame, n_attempted, coverage_percent, position = loaded[key]
        paired = common_objects and shared is not None
        scored = frame.merge(shared, on=OBJECT_KEY, how="inner") if paired else frame
        mean, median, rmse, var, median_ci, rmse_ci = _errors_stats(scored, "abs_error_mm")

        position_stats: tuple = (None, None, None, None, None, None)
        n_position = 0
        if position is not None:
            # The same objects as the depth errors when pairing, so both columns of a
            # technique describe one set of objects.
            scored_position = (
                position.merge(shared, on=OBJECT_KEY, how="inner") if paired else position
            )
            n_position = len(scored_position)
            position_stats = _errors_stats(scored_position, "error_mm")

        metrics.append(
            TechniqueMetrics(
                depth=depth,
                association=association,
                directory=directory,
                n_valid=len(frame),
                n_scored=len(scored),
                n_attempted=n_attempted,
                coverage_percent=coverage_percent,
                mean_mm=mean,
                median_mm=median,
                rmse_mm=rmse,
                var_mm=var,
                median_ci=median_ci,
                rmse_ci=rmse_ci,
                n_position=n_position,
                position_mean_mm=position_stats[0],
                position_median_mm=position_stats[1],
                position_rmse_mm=position_stats[2],
                position_var_mm=position_stats[3],
                position_median_ci=position_stats[4],
                position_rmse_ci=position_stats[5],
            )
        )

    seg_token = _resolve(segmentation, SEGMENTATION_DIRS, "segmentation model")
    vlm_token = _resolve(vlm, VLM_DIRS, "VLM")
    return AblationRow(
        segmentation=SEGMENTATION_LABELS.get(seg_token, seg_token),
        vlm=VLM_LABELS.get(vlm_token, vlm_token),
        techniques=metrics,
    )


def _metric_names(coverage: bool, position: bool, ci_style: str | None) -> list[str]:
    """Return the column headers of one technique, in the order :func:`_technique_cells` fills them."""
    names = metric_headers("Median", ci_style) + metric_headers("RMSE", ci_style)
    if position:
        names += metric_headers("Pos. Median", ci_style) + metric_headers("Pos. RMSE", ci_style)
    if coverage:
        names.append(sc("Valid"))
    return names


def _technique_cells(
    technique: TechniqueMetrics, coverage: bool, position: bool, ci_style: str | None
) -> list[str]:
    cells = _cells_with_interval(technique.median_mm, technique.median_ci, ci_style)
    cells += _cells_with_interval(technique.rmse_mm, technique.rmse_ci, ci_style)
    if position:
        cells += _cells_with_interval(
            technique.position_median_mm, technique.position_median_ci, ci_style
        )
        cells += _cells_with_interval(
            technique.position_rmse_mm, technique.position_rmse_ci, ci_style
        )
    if coverage:
        cells.append(
            "--" if technique.coverage_percent is None else f"{technique.coverage_percent:.1f}\\%"
        )
    return cells


def latex_table(
    rows: list[AblationRow],
    techniques: list[tuple[str, str]],
    coverage: bool = False,
    position: bool = False,
    caption_note: str = "",
    ci_style: str | None = None,
) -> str:
    """Render the rows as a tabular with one column group per depth technique.

    The confidence intervals are printed as ``ci_style`` says: None, ``"column"`` or ``"maths"``.
    """
    metric_names = _metric_names(coverage, position, ci_style)
    per_technique = len(metric_names)
    n_cols = len(techniques) * per_technique

    source_groups: list[tuple[str, int]] = []
    for depth, _ in techniques:
        label = DEPTH_LABELS.get(depth, depth)
        if source_groups and source_groups[-1][0] == label:
            source_groups[-1] = (label, source_groups[-1][1] + 1)
        else:
            source_groups.append((label, 1))

    source_cells, source_rules, col = [], [], 3
    for label, count in source_groups:
        width = count * per_technique
        source_cells.append(f"\\multicolumn{{{width}}}{{c}}{{{sc(label)}}}")
        source_rules.append(f"\\cmidrule(lr){{{col}-{col + width - 1}}}")
        col += width

    assoc_cells, assoc_rules, col = [], [], 3
    for _, association in techniques:
        label = ASSOCIATION_LABELS.get(association, association)
        assoc_cells.append(f"\\multicolumn{{{per_technique}}}{{c}}{{{sc(label)}}}")
        assoc_rules.append(f"\\cmidrule(lr){{{col}-{col + per_technique - 1}}}")
        col += per_technique

    metric_cells = metric_names * len(techniques)

    header = [
        f"    \\begin{{tabular}}{{ll{'c' * n_cols}}}",
        "    \\toprule",
        f"    \\multicolumn{{2}}{{c}}{{{sc('Tools')}}} & " + " & ".join(source_cells) + " \\\\",
        "    " + " ".join(source_rules),
        "    & & " + " & ".join(assoc_cells) + " \\\\",
        "    " + " ".join(assoc_rules),
        f"    {sc('Segmentation')} & {sc('Annotation')} & " + " & ".join(metric_cells) + " \\\\",
        "    \\midrule",
    ]

    body = []
    for row in rows:
        cells: list[str] = []
        for technique in row.techniques:
            cells += _technique_cells(technique, coverage, position, ci_style)
        body.append(
            "    {:<12} & {:<12} & ".format(row.segmentation, row.vlm)
            + " & ".join(f"{c:>14}" for c in cells)
            + " \\\\"
        )

    footer = ["    \\bottomrule", "    \\end{tabular}"]
    if caption_note:
        footer.append(f"    % {caption_note}")
    return "\n".join(header + body + footer)


def _parse_technique(spec: str) -> tuple[str, str]:
    depth, sep, association = spec.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"technique {spec!r} must be <depth>:<association>, e.g. sensor:mask-median"
        )
    return depth, association


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
        help="segmentation model of a single row; omit to render every default row",
    )
    parser.add_argument("--vlm", help="VLM of a single row")
    parser.add_argument(
        "--techniques",
        nargs="+",
        type=_parse_technique,
        default=DEFAULT_TECHNIQUES,
        metavar="DEPTH:ASSOCIATION",
        help=(
            "columns of the table, in order (default: sensor and monocular, each "
            "with mask-median and bbox-center)"
        ),
    )
    parser.add_argument(
        "--common-objects",
        action="store_true",
        help=(
            "paired analysis only: score all available techniques on the intersection "
            "of their object keys. By default each technique is scored on its own valid "
            "depth estimates, matching the main pipeline table."
        ),
    )
    parser.add_argument(
        "--own-objects",
        action="store_true",
        help=argparse.SUPPRESS,  # backwards-compatible no-op; own objects is now default
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "add valid-depth coverage. Coverage is derived from valid rows plus the "
            "per-image ignored-depth counts written by the evaluator."
        ),
    )
    parser.add_argument(
        "--position-mm",
        "--position",
        dest="position",
        action="store_true",
        help=(
            "add the median and RMSE of the metric position error [mm] of every technique: "
            "the distance between the 3D point of its association and the ArUco position"
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="render missing technique runs as -- instead of failing",
    )
    add_confidence_interval_arguments(parser)
    parser.add_argument("--csv", type=Path, help="also write the numbers to this CSV file")
    args = parser.parse_args(argv)
    if bool(args.segmentation) != bool(args.vlm):
        parser.error("--segmentation and --vlm must be given together")
    if args.common_objects and args.own_objects:
        parser.error("--common-objects and --own-objects are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = [(args.segmentation, args.vlm)] if args.segmentation else list(DEFAULT_ROWS)

    # Own-object scoring is the default and is the correct mode for the paper's Table III.
    common_objects = bool(args.common_objects)

    rows: list[AblationRow] = []
    for segmentation, vlm in pairs:
        try:
            rows.append(
                read_row(
                    args.results_dir,
                    segmentation,
                    vlm,
                    args.techniques,
                    common_objects=common_objects,
                    skip_missing=args.skip_missing,
                )
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise SystemExit(f"error: {segmentation} + {vlm}: {exc}") from exc

    note = (
        "depth error [mm] on the intersection of objects available to every technique"
        if common_objects
        else "depth error [mm] on each technique's own valid depth estimates"
    )
    if args.position:
        note += "; Pos. = metric position error [mm]"
    ci_style = confidence_interval_style(args)
    if ci_style is not None:
        note += f"; {confidence_note(ci_style)}"
    print(
        latex_table(
            rows,
            args.techniques,
            coverage=args.coverage,
            position=args.position,
            caption_note=note,
            ci_style=ci_style,
        )
    )
    print()

    for row in rows:
        print(f"% {row.segmentation} + {row.vlm}")
        for t in row.techniques:
            if t.missing:
                print(f"%   {t.depth:<9} {t.association:<11} MISSING")
                continue
            coverage = "n/a" if t.coverage_percent is None else f"{t.coverage_percent:5.1f}%"
            print(
                f"%   {t.depth:<9} {t.association:<11} valid {t.n_valid:>4} "
                f"({coverage}), scored {t.n_scored:>4}, positions {t.n_position:>4}: "
                f"{t.directory.name}"
            )

    if args.csv:
        pd.DataFrame(
            [
                {
                    "segmentation": row.segmentation,
                    "annotation": row.vlm,
                    "depth": t.depth,
                    "association": t.association,
                    "common_objects": common_objects,
                    "missing": t.missing,
                    "n_valid": t.n_valid,
                    "n_scored": t.n_scored,
                    "n_attempted": t.n_attempted,
                    "coverage_percent": t.coverage_percent,
                    "depth_mean_mm": t.mean_mm,
                    "depth_median_mm": t.median_mm,
                    **interval_columns("depth_median_mm", t.median_ci),
                    "depth_rmse_mm": t.rmse_mm,
                    **interval_columns("depth_rmse_mm", t.rmse_ci),
                    "depth_var_mm": t.var_mm,
                    "n_position": t.n_position,
                    "position_mean_mm": t.position_mean_mm,
                    "position_median_mm": t.position_median_mm,
                    **interval_columns("position_median_mm", t.position_median_ci),
                    "position_rmse_mm": t.position_rmse_mm,
                    **interval_columns("position_rmse_mm", t.position_rmse_ci),
                    "position_var_mm": t.position_var_mm,
                    "directory": None if t.directory is None else t.directory.name,
                }
                for row in rows
                for t in row.techniques
            ]
        ).to_csv(args.csv, index=False)
        print(f"\n% wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
