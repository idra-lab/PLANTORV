r"""Segment every frame of a dataset and write the masks for the later stages.

First of the three stages. It reads the RGB frames of ``--input-dir`` and writes,
under ``<output-dir>/output_segmentation``, one directory per frame holding the
object crops, their bounding boxes and their binary masks; see
:mod:`pipeline.artifacts` for the layout.

Nothing else in the pipeline segments, so this is the run that
``annotation.py`` and ``depth_estimation.py`` reuse: pointed at the same
``--output-dir``, either of them can be run again, in any order and as often as a
model is swapped, without segmenting anything a second time.

Examples
--------
Segment the dataset into ``output/``::

    python3 segmentation.py --input-dir dataset/rgb --output-dir output

Segment with SAM 3, whose concepts a VLM names::

    python3 segmentation.py --segmenter-config segmentation/conf/sam3.yaml \\
        --output-dir output_sam3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from pipeline import artifacts, cli
from pipeline.provenance import write_run_config
from segmentation.segmentation_factory import create_segmenter, describe
from utility.utility import logger


def run(
    input_dir: Union[str, Path], output_dir: Union[str, Path], args: argparse.Namespace
) -> None:
    """Segment every frame of ``input_dir``.

    Parameters
    ----------
    input_dir : Union[str, Path]
        Directory holding the RGB frames.
    output_dir : Union[str, Path]
        Run directory. The artefacts go to its ``output_segmentation`` subdirectory.
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    images = artifacts.list_images(input_dir)
    segmentation_dir = Path(output_dir) / artifacts.SEGMENTATION_SUBDIR
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # The backend's own figures stay where they have always been, next to the
    # stage artefacts rather than inside them: --debug-masks and --view-masks are
    # for reading, and nothing downstream parses them.
    model = create_segmenter(
        args.segmenter_config,
        save_dir=Path(output_dir) / "segmentation_outputs",
        device=args.device,
        debug_masks=args.debug_masks,
    )

    # Written before the frames rather than after, so an interrupted run still says
    # what it was doing.
    write_run_config(
        segmentation_dir,
        "segmentation",
        describe(model, args.segmenter_config),
        device=args.device,
    )

    for position, image_path in enumerate(images):
        fid = artifacts.frame_id(image_path)
        logger.info(f"Segmenting image {position + 1}/{len(images)}: {image_path}")

        image = Image.open(image_path)

        start = time.time()
        # The backends name their figures after `idx + 1`, so passing the frame
        # number less one makes those names the frame number.
        masked_rgb, mask_bin = model.obtain_bg(image, idx=fid - 1)
        crops, bboxes = model.individual_mask(
            image, bg_mask_bin=mask_bin, bg_masked_rgb=masked_rgb, idx=fid - 1
        )
        logger.debug(f"Segmentation done in {time.time() - start}s")

        # Shows the masks that were actually kept. Only SAM3Model ships a viewer, so this
        # stays an attribute lookup rather than a direct call: the model above is swappable.
        visualize = getattr(model, "visualize", None)
        if args.view_masks and visualize is not None:
            visualize(image, fid - 1)

        written = artifacts.write_frame(
            segmentation_dir, fid, image_path, crops, bboxes, model.last_masks
        )
        logger.info(f"Wrote {len(bboxes)} objects to {written}")

        # The frame is finished, so nothing on the GPU is still needed. Returning
        # the cached blocks keeps the allocator's reserve from ratcheting up to
        # the high-water mark of the greediest frame and staying there: SAM 3
        # asks for one block per concept in the scene, and a later frame that
        # would otherwise fit can fail to find that block contiguous.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        type=str,
        default="dataset/rgb",
        help="Directory holding the RGB frames to segment (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help=(
            "Run directory. The masks are written to its output_segmentation "
            "subdirectory (default: %(default)s)"
        ),
    )
    cli.add_common_arguments(parser)
    cli.add_segmentation_arguments(parser)

    args = parser.parse_args()

    cli.require_dir(args.input_dir, "Images directory")
    cli.require_file(args.segmenter_config, "Segmentation configuration file")
    cli.validate_device(args.device)

    return args


if __name__ == "__main__":
    parsed = parse_arguments()
    cli.setup(parsed)

    start_all = time.time()
    try:
        run(parsed.input_dir, parsed.output_dir, parsed)
    except (FileNotFoundError, ValueError) as error:
        logger.error(str(error))
        sys.exit(1)

    logger.info(f"Total segmentation time: {time.time() - start_all}s")
