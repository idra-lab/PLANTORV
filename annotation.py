r"""Tag and describe the objects a segmentation run left on disk.

Second of the three stages, and independent of the third. It reads the frame
directories ``segmentation.py`` wrote and fills the ``tag`` and ``description`` of
every object in ``<output-dir>/output_img<frame-id>.json``, leaving the fields
``depth_estimation.py`` owns untouched. Changing the annotation model therefore
costs one run of this script and nothing else.

Two annotators are available. ``--annotator gpt`` sends the full frame and the
crop of each object to the remote model named by ``--llm-config``, so switching
model means pointing that flag at another file in ``LLM/conf``. ``--annotator dam``
runs Describe Anything locally over the object masks instead.

Examples
--------
Annotate a segmentation run with the model of the default configuration::

    python3 annotation.py --input-dir output/output_segmentation --output-dir output

Annotate the same run with Claude, without segmenting again::

    python3 annotation.py --input-dir output/output_segmentation --output-dir output \\
        --llm-config LLM/conf/azure_claude-opus46.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from pipeline import artifacts, cli
from pipeline.provenance import write_run_config
from scene_understanding.annotator import Annotator
from utility.utility import logger


def build_annotator(args: argparse.Namespace) -> Annotator:
    """Build the annotator selected by ``--annotator``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Annotator
        The configured annotator.

    Raises
    ------
    FileNotFoundError
        If the checkpoint or the configuration file is missing.
    ValueError
        If the LLM configuration does not describe a usable backend.
    ImportError
        If DAM is selected but the ``dam`` package is not installed.
    """
    if args.annotator == "dam":
        # Imported here rather than at module scope so that a GPT run does not pay
        # for a dependency tree it will not use.
        from scene_understanding.dam_annotator import DAMAnnotator

        return DAMAnnotator(
            args.dam_model,
            query=args.dam_query,
            device=args.dam_device or args.device,
            temperature=args.dam_temperature,
            top_p=args.dam_top_p,
            max_new_tokens=args.dam_max_new_tokens,
        )

    from scene_understanding.gpt_annotator import GPTAnnotator

    # The YAML file selects the model, the endpoint, the credentials and the request
    # parameters, so switching model means pointing --llm-config elsewhere.
    return GPTAnnotator.from_config(args.llm_config)


def describe(annotator: Annotator, args: argparse.Namespace) -> Dict[str, Any]:
    """Describe the annotator, for the record written next to the run.

    Reads the annotator rather than the arguments, so a value its constructor filled
    in is part of the answer.

    Parameters
    ----------
    annotator : Annotator
        The annotator, as returned by :func:`build_annotator`.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Dict[str, Any]
        The effective configuration.
    """
    described: Dict[str, Any] = {"ANNOTATOR": args.annotator}

    llm = getattr(annotator, "llm", None)
    if llm is not None:
        described["LLM_CONFIG_FILE"] = llm.config_file
        described["LLM_VERSION"] = llm.model
        described["LLM_CONFIG"] = llm.request_params()
        # The prompt decides the vocabulary of the tags as much as the model does.
        described["PROMPT"] = getattr(annotator, "prompt", None)
        return described

    described.update(
        {
            "DAM_MODEL": getattr(annotator, "model_path", args.dam_model),
            "DAM_CONFIG": {
                "temperature": getattr(annotator, "temperature", args.dam_temperature),
                "top_p": getattr(annotator, "top_p", args.dam_top_p),
                "num_beams": getattr(annotator, "num_beams", None),
                "max_new_tokens": getattr(annotator, "max_new_tokens", args.dam_max_new_tokens),
            },
            "QUERY": getattr(annotator, "query", args.dam_query),
        }
    )

    return described


def resolve_image(recorded: Path, fid: int, images_dir: Optional[str]) -> Path:
    """Return the RGB frame to annotate.

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
    """Annotate every frame of a segmentation directory.

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

    try:
        annotator = build_annotator(args)
    except (FileNotFoundError, ValueError, ImportError) as error:
        logger.error(f"Unable to configure the annotator: {error}")
        sys.exit(1)

    # Written before the frames rather than after, so an interrupted run still says
    # what it was doing.
    write_run_config(output_dir, "annotation", describe(annotator, args), device=args.device)

    for position, fid in enumerate(frame_ids):
        logger.info(f"Annotating frame {position + 1}/{len(frame_ids)}: {fid}")

        frame = artifacts.read_frame(input_dir, fid)
        image = Image.open(resolve_image(frame.image_path, fid, args.images_dir))

        start = time.time()
        # `masks` goes along whichever annotator is in use: the crop-based one
        # ignores it, the mask-based one reads it instead of the crops.
        objects = annotator.annotate(image, frame.crops, frame.bboxes, masks=frame.masks)
        logger.debug(f"Tagging and description done in {time.time() - start}s")

        written = artifacts.update_results(output_dir, fid, objects, keys=list(objects))
        logger.info(f"Wrote {len(objects)} annotations to {written}")

    close = getattr(annotator, "close", None)
    if close is not None:
        close()


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
        help="Run directory the annotations are written to (default: %(default)s)",
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
    cli.add_annotation_arguments(parser)

    args = parser.parse_args()

    cli.require_dir(args.input_dir, "Segmentation directory")
    if args.images_dir is not None:
        cli.require_dir(args.images_dir, "Images directory")

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

    logger.info(f"Total annotation time: {time.time() - start_all}s")
