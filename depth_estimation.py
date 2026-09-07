r"""Give every segmented object a position and a distance.

Third of the three stages, and independent of the second. It reads the frame
directories ``segmentation.py`` wrote and fills ``coord_center&depth``,
``object_depth_mm`` and ``depth_association`` for every object of
``<output-dir>/output_img<frame-id>.json``, leaving the ``tag`` and the
``description`` ``annotation.py`` owns untouched. Depth needs the bounding boxes
and the masks, never the annotations, so this can run before annotation, after
it, or again on its own when the depth backend changes.

Depth comes either from the sensor images of ``--depth-dir`` or from a monocular
estimator run on the RGB frame, as ``--depth-source`` selects.

Examples
--------
Read the depth of a segmentation run from the dataset::

    python3 depth_estimation.py --input-dir output/output_segmentation --output-dir output

Estimate it from the RGB frames instead, taking one depth per object mask::

    python3 depth_estimation.py --input-dir output/output_segmentation --output-dir output \\
        --depth-source monocular --depth-association mask-median
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
from PIL import Image

from mapping.depth_anything import (
    DEFAULT_MODEL_ID,
    DEFAULT_V3_MODEL_ID,
    DepthAnythingV2Provider,
    DepthAnythingV3Provider,
)
from mapping.depth_provider import DepthProvider
from mapping.rgbd_mapper import attach_object_depths, main_coords
from pipeline import artifacts, cli
from pipeline.provenance import write_run_config
from utility.utility import logger


def build_depth_provider(args: argparse.Namespace) -> Optional[DepthProvider]:
    """Build the monocular depth backend selected by ``--depth-source``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    DepthProvider or None
        The provider to estimate depth with, or None for ``--depth-source sensor``,
        in which case the depth images in ``--depth-dir`` are used instead.
    """
    if args.depth_source == "depth-anything-v2":
        return DepthAnythingV2Provider(
            model_id=args.depth_model or DEFAULT_MODEL_ID,
            device=args.depth_device,
        )
    if args.depth_source == "monocular":
        return DepthAnythingV3Provider(
            model_id=args.depth_model or DEFAULT_V3_MODEL_ID,
            focal_length_px=args.depth_focal_length_px,
            process_res=args.depth_process_res,
            device=args.depth_device,
        )
    return None


def describe(provider: Optional[DepthProvider], args: argparse.Namespace) -> Dict[str, Any]:
    """Describe the depth backend, for the record written next to the run.

    Parameters
    ----------
    provider : DepthProvider or None
        The estimator, or None when the depth images are read from disk.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Dict[str, Any]
        The effective configuration.
    """
    described: Dict[str, Any] = {
        "DEPTH_SOURCE": args.depth_source,
        "DEPTH_ASSOCIATION": args.depth_association,
    }

    if provider is None:
        described["DEPTH_DIR"] = args.depth_dir
        return described

    described.update(
        {
            "PROVIDER": type(provider).__name__,
            "DEPTH_MODEL": getattr(provider, "model_id", args.depth_model),
            "DEPTH_DEVICE": args.depth_device,
        }
    )
    if args.depth_source == "monocular":
        described["DEPTH_CONFIG"] = {
            "focal_length_px": args.depth_focal_length_px,
            "process_res": args.depth_process_res,
        }

    return described


def resolve_image(recorded: Path, fid: int, images_dir: Optional[str]) -> Path:
    """Return the RGB frame the depth is registered to.

    Parameters
    ----------
    recorded : Path
        The path segmentation recorded for this frame.
    fid : int
        The frame number.
    images_dir : str or None
        Value of ``--images-dir``, used when the dataset has moved since.

    Returns
    -------
    Path
        The frame to read.

    Raises
    ------
    FileNotFoundError
        If the frame cannot be found.
    """
    if images_dir is None:
        if not recorded.is_file():
            raise FileNotFoundError(
                f"Frame {fid} was segmented from {recorded}, which no longer exists. "
                "Pass --images-dir to say where the frames are now."
            )
        return recorded

    candidates = [
        path for path in artifacts.list_images(images_dir) if artifacts.frame_id(path) == fid
    ]
    if not candidates:
        raise FileNotFoundError(f"No frame {fid} in {images_dir}")

    return candidates[0]


def run(
    input_dir: Union[str, Path], output_dir: Union[str, Path], args: argparse.Namespace
) -> None:
    """Attach coordinates and depth to every object of a segmentation directory.

    Parameters
    ----------
    input_dir : Union[str, Path]
        The ``output_segmentation`` directory written by ``segmentation.py``.
    output_dir : Union[str, Path]
        Run directory the per-frame result files are written to.
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    frame_ids = artifacts.list_frame_ids(input_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Depth comes either from the dataset in `--depth-dir` or from a monocular
    # estimator, so only one of the two is prepared.
    provider = build_depth_provider(args)
    if provider is None:
        depth_images = artifacts.index_by_frame(artifacts.list_images(args.depth_dir))
    else:
        depth_images = {}
        logger.info(f"Estimating depth with {type(provider).__name__}")

    # Written before the frames rather than after, so an interrupted run still says
    # what it was doing.
    write_run_config(output_dir, "depth", describe(provider, args), device=args.device)

    for position, fid in enumerate(frame_ids):
        logger.info(f"Measuring frame {position + 1}/{len(frame_ids)}: {fid}")

        # The crops belong to the annotator; depth reads the boxes and the masks.
        frame = artifacts.read_frame(input_dir, fid, crops=False)
        image_path = resolve_image(frame.image_path, fid, args.images_dir)

        # The two mappers below add their fields to the objects they are given, so
        # the boxes segmentation kept are what the frame's objects are seeded with.
        objects = {f"mask_{index}": {"bbox": bbox} for index, bbox in enumerate(frame.bboxes)}

        start = time.time()
        # `masks` reaches both branches so that --depth-association is answered the
        # same way whatever --depth-source is: the masks are the ones
        # `individual_mask` kept, in the order of the objects.
        if provider is None:
            if fid not in depth_images:
                raise FileNotFoundError(f"No depth image for frame {fid} in {args.depth_dir}")
            objects = main_coords(
                Image.open(image_path),
                str(depth_images[fid]),
                objects,
                masks=frame.masks,
                association=args.depth_association,
            )
        else:
            # The providers work on the BGR array OpenCV produces, not on the PIL image.
            color_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if color_bgr is None:
                raise FileNotFoundError(f"RGB image not found at path: {image_path}")
            objects = attach_object_depths(
                objects,
                provider.estimate(color_bgr).depth_mm,
                masks=frame.masks,
                association=args.depth_association,
            )
        logger.debug(f"Coordinates and depth done in {time.time() - start}s")

        written = artifacts.update_results(output_dir, fid, objects, keys=list(objects))
        logger.info(f"Wrote the depth of {len(objects)} objects to {written}")


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
        default=f"output/{artifacts.SEGMENTATION_SUBDIR}",
        help="Segmentation directory written by segmentation.py (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Run directory the depths are written to (default: %(default)s)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help=(
            "Directory holding the RGB frames. Only needed when the dataset moved "
            "since the segmentation run, which recorded where each frame was read from"
        ),
    )
    cli.add_common_arguments(parser)
    cli.add_depth_arguments(parser)

    args = parser.parse_args()

    cli.require_dir(args.input_dir, "Segmentation directory")
    if args.images_dir is not None:
        cli.require_dir(args.images_dir, "Images directory")

    # Only checked for the sensor backend: the estimators never read --depth-dir, so
    # requiring it would force a depth dataset to exist for a run that ignores it.
    if args.depth_source == "sensor":
        cli.require_dir(args.depth_dir, "Depth images directory")

    cli.validate_device(args.device)

    return args


if __name__ == "__main__":
    parsed = parse_arguments()
    cli.setup(parsed)

    start_all = time.time()
    try:
        run(parsed.input_dir, parsed.output_dir, parsed)
    except FileNotFoundError as error:
        logger.error(str(error))
        sys.exit(1)

    logger.info(f"Total depth estimation time: {time.time() - start_all}s")
