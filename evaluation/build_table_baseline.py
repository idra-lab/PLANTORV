"""Build the LaTeX table of the VLM-only baseline from its per-run result folders.

The baseline asks a single vision LLM to list, locate and range every object of a frame
(see :mod:`evaluation.only_vlm`), and each model's run lives in
``results/results_only_<vlm>``. For every model the table shows the semantic-annotation
recall, and the median and RMSE of the pixel position error and of the depth error, the depth
being the model's own estimate. The numbers are read with
:func:`evaluation.build_table_pipeline.read_metrics`, so they are computed exactly as in the
pipeline table.

``--confidence-intervals`` follows every error with a column holding its percentile bootstrap
confidence interval, written ``[5th percentile, 95th percentile]`` of the bootstrap estimates;
``--ci-maths`` writes the bounds on the value instead, as ``$value^{95th}_{5th}$``, and
implies ``--confidence-intervals``. The recall never shows an interval. The CSV written by
``--csv`` also keeps the means and every interval.

Examples
--------
    python -m evaluation.build_table_baseline
    python -m evaluation.build_table_baseline --models gpt54 gpt55 --confidence-intervals
    python -m evaluation.build_table_baseline --ci-maths --csv baseline.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from .build_table_pipeline import (
    DEFAULT_RESULTS_DIR,
    VLM_DIRS,
    RunMetrics,
    _cells_with_interval,
    add_confidence_interval_arguments,
    confidence_interval_style,
    confidence_note,
    interval_columns,
    metric_headers,
    read_metrics,
    sc,
)

# The rows of the table in the report, in order.
DEFAULT_MODELS = ["sonnet46", "gpt54", "gpt54-mini", "gpt54-nano", "qwen3-vl", "qwen36"]


def read_baseline(results_dir: Path, vlm: str) -> RunMetrics:
    """Read the metrics of the VLM-only run of one model.

    The baseline has neither a depth sensor nor a depth association, so the depth arguments
    passed to :func:`read_metrics` only name folders it also looks for; the one that exists is
    ``results_only_<vlm>``.
    """
    return read_metrics(results_dir, "only", vlm, "sensor", "mask-median")


def latex_table(rows: list[RunMetrics], ci_style: str | None = None) -> str:
    """Render the rows as the tabular environment of the baseline table.

    The confidence intervals of the errors are printed as ``ci_style`` says: None,
    ``"column"`` or ``"maths"``.
    """
    metrics = [*metric_headers("Median", ci_style), *metric_headers("RMSE", ci_style)]
    header = [
        f"    \\begin{{tabular}}{{l{'c' * (1 + 2 * len(metrics))}}}",
        "    \\toprule",
        f"    & {sc('Semantic Annotation')} "
        f"& \\multicolumn{{{len(metrics)}}}{{c}}{{{sc('Position Error')} [px]}} "
        f"& \\multicolumn{{{len(metrics)}}}{{c}}{{{sc('Depth Error')} [mm]}} \\\\",
        "    " + " & ".join([sc("Models"), sc("Recall"), *metrics, *metrics]) + " \\\\",
        "    \\midrule",
    ]

    body = []
    for row in rows:
        cells = [f"{row.recall_percent:.1f}\\%"]
        cells += _cells_with_interval(row.median_error_px, row.median_px_ci, ci_style)
        cells += _cells_with_interval(row.rmse_px, row.rmse_px_ci, ci_style)
        cells += _cells_with_interval(row.depth_median_mm, row.depth_median_ci, ci_style)
        cells += _cells_with_interval(row.depth_rmse_mm, row.depth_rmse_ci, ci_style)
        body.append(f"    {row.vlm:<12} & " + " & ".join(f"{cell:>14}" for cell in cells) + " \\\\")

    footer = ["    \\bottomrule", "    \\end{tabular}"]
    if ci_style is not None:
        footer.append(f"    % {confidence_note(ci_style)}")
    return "\n".join(header + body + footer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory holding the results_only_* run folders",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="VLM",
        help=(
            f"rows of the table, in order, among {', '.join(sorted(VLM_DIRS))} "
            f"(default: {' '.join(DEFAULT_MODELS)})"
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="warn and continue instead of failing when a model has no evaluated run",
    )
    add_confidence_interval_arguments(parser)
    parser.add_argument("--csv", type=Path, help="also write the numbers to this CSV file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    rows: list[RunMetrics] = []
    for vlm in args.models:
        try:
            rows.append(read_baseline(args.results_dir, vlm))
        except (FileNotFoundError, KeyError) as exc:
            message = f"{vlm}: {exc}"
            if not args.skip_missing:
                raise SystemExit(f"error: {message}") from exc
            print(f"warning: skipping {message}")

    print(latex_table(rows, ci_style=confidence_interval_style(args)))
    print()
    for row in rows:
        print(f"% {row.vlm}: {row.directory.name}")

    if args.csv:
        pd.DataFrame(
            [
                {
                    "annotation": r.vlm,
                    "recall_percent": r.recall_percent,
                    "mean_error_px": r.mean_error_px,
                    **interval_columns("mean_error_px", r.mean_px_ci),
                    "median_error_px": r.median_error_px,
                    **interval_columns("median_error_px", r.median_px_ci),
                    "rmse_px": r.rmse_px,
                    **interval_columns("rmse_px", r.rmse_px_ci),
                    "depth_mean_mm": r.depth_mean_mm,
                    **interval_columns("depth_mean_mm", r.depth_mean_ci),
                    "depth_median_mm": r.depth_median_mm,
                    **interval_columns("depth_median_mm", r.depth_median_ci),
                    "depth_rmse_mm": r.depth_rmse_mm,
                    **interval_columns("depth_rmse_mm", r.depth_rmse_ci),
                    "directory": r.directory.name,
                }
                for r in rows
            ]
        ).to_csv(args.csv, index=False)
        print(f"\n% wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
