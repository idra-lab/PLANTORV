"""Run the three pipeline stages back to back over a dataset.

This is the whole pipeline in one command, kept for the runs where nothing is
being swapped: it segments every frame, annotates the objects and measures their
depth, into a single ``--output-dir``.

The work itself lives in the three stage scripts, and they are the ones to reach
for while a model is being tried out, since each reads what the previous one left
on disk::

    python3 segmentation.py     --input-dir dataset/rgb --output-dir run
    python3 annotation.py       --input-dir run/output_segmentation --output-dir run
    python3 depth_estimation.py --input-dir run/output_segmentation --output-dir run

Segmenting is the expensive step and nothing but ``segmentation.py`` does it, so
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
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

from pipeline import artifacts, cli
from utility.utility import logger


def load_stage(name: str) -> ModuleType:
    """Import a stage script by its file name.

    ``segmentation.py`` sits next to the ``segmentation/`` package, and a plain
    ``import segmentation`` resolves to the package, so the stages are loaded from
    their paths rather than by name.

    Parameters
    ----------
    name : str
        The stage file, without its extension.

    Returns
    -------
    ModuleType
        The imported module.
    """
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"stage_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the stage {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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
        ("segmentation", args.input_dir),
        ("annotation", segmentation_dir),
        ("depth_estimation", segmentation_dir),
    ]

    for name, stage_input in stages:
        logger.info(f"Running {name}")
        start = time.time()
        load_stage(name).run(stage_input, output_dir, args)
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

    if args.annotator == "gpt":
        cli.require_file(args.llm_config, "LLM configuration file")
    else:
        from scene_understanding.dam_annotator import looks_like_repo_id

        if not looks_like_repo_id(args.dam_model) and not Path(args.dam_model).exists():
            logger.error(
                f"DAM checkpoint does not exist: {args.dam_model}. Download it with "
                "`python3 scripts/install_models.py dam_3b`."
            )
            sys.exit(1)
        if "<image>" not in args.dam_query:
            logger.error(
                "--dam-query must contain the <image> token DAM substitutes the region into."
            )
            sys.exit(1)

    # Only checked for the sensor backend: the estimators never read --depth-dir, so
    # requiring it would force a depth dataset to exist for a run that ignores it.
    if args.depth_source == "sensor":
        cli.require_dir(args.depth_dir, "Depth images directory")

    cli.validate_device(args.device)

    return args


if __name__ == "__main__":
    start_all = time.time()
    try:
        main(parse_arguments())
    except (FileNotFoundError, ValueError) as error:
        logger.error(str(error))
        sys.exit(1)

    logger.info(f"Total time image process: {time.time() - start_all}s")
