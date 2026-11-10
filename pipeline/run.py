"""Run the three pipeline stages back to back over a dataset.

This is the whole pipeline in one command, kept for the runs where nothing is
being swapped: it segments every frame, annotates the objects and measures their
depth, into a single ``--output-dir``.

The work itself lives in the three stage scripts, and they are the ones to reach
for while a model is being tried out, since each reads what the previous one left
on disk::

    python3 -m pipeline.segment        --input-dir dataset/rgb --output-dir run
    python3 -m pipeline.annotate       --input-dir run/output_segmentation --output-dir run
    python3 -m pipeline.estimate_depth --input-dir run/output_segmentation --output-dir run

Segmenting is the expensive step and nothing but ``pipeline/segment.py`` does it, so
trying a second annotator, or a second depth backend, costs one run of that stage
alone. Both write to ``<output-dir>/output_img<frame-id>.json``, each filling its
own fields, which is the file ``evaluation/run_evaluation.py`` reads.

Examples
--------
Run everything over the dataset with the default models::

    python3 samgpt.py --input-dir dataset/rgb --output-dir output

Run everything with Describe Anything and estimated depth::

    python3 samgpt.py --annotator dam --depth-source monocular --output-dir output_dam

Each stage writes what it was configured with next to what it produced, as
``segmentation_config.yaml``, ``annotation_config.yaml`` and ``depth_config.yaml``.
"""

import argparse
import sys
import time
from pathlib import Path

from pipeline import annotate, artifacts, cli, estimate_depth, segment
from utility.utility import logger


def main(args: argparse.Namespace) -> None:
    """Run segmentation, annotation and depth estimation in that order.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    cli.setup(args)

    output_dir = Path(args.output_dir)
    segmentation_dir = output_dir / artifacts.SEGMENTATION_SUBDIR

    # The frames were just segmented from --input-dir, so the paths recorded for
    # them are current and the later stages need no override.
    args.images_dir = None

    stages = [
        ("segmentation", segment, args.input_dir),
        ("annotation", annotate, segmentation_dir),
        ("depth_estimation", estimate_depth, segmentation_dir),
    ]

    for name, stage, stage_input in stages:
        logger.info(f"Running {name}")
        start = time.time()
        stage.run(stage_input, output_dir, args)
        logger.info(f"{name} done in {time.time() - start}s")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        "--images-dir",
        dest="input_dir",
        type=str,
        default="dataset/rgb",
        help="Directory holding the RGB frames (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Run directory all three stages write to (default: %(default)s)",
    )
    cli.add_common_arguments(parser)
    cli.add_segmentation_arguments(parser)
    cli.add_annotation_arguments(parser)
    cli.add_depth_arguments(parser)

    args = parser.parse_args()

    cli.require_dir(args.input_dir, "Images directory")

    cli.require_file(args.segmenter_config, "Segmentation configuration file")

    cli.validate_annotation_arguments(args)

    cli.validate_depth_arguments(args)

    cli.validate_device(args.device)

    return args


def run_cli() -> None:
    """Entry point of ``python -m pipeline.run`` and of the ``samgpt.py`` wrapper."""
    start_all = time.time()
    try:
        main(parse_arguments())
    except (FileNotFoundError, ValueError) as error:
        logger.error(str(error))
        sys.exit(1)

    logger.info(f"Total time image process: {time.time() - start_all}s")


if __name__ == "__main__":
    run_cli()
