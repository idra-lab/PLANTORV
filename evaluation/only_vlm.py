"""VLM-only baseline: one model call replaces the whole segmentation pipeline.

The normal pipeline (``samgpt.py``) segments a frame with SAM, crops every object,
annotates each crop with a vision LLM, and reads the depth of each object centre from
the depth sensor. This script asks a single vision LLM, once per frame, to do all of
that at the same time: to list the objects it sees, to place a bounding box around each
of them, and to estimate how far each object is from the camera.

The artefacts are the same as the normal evaluation, so the two runs are comparable
file by file::

    <output-dir>/output_seg_annotation/output_img<N>.json   the per-frame annotations
    <output-dir>/vlm_raw/img<N>.txt                         the raw model answers
    <output-dir>/evaluation_results/                        tables, figures, overlays

Annotations already on disk are reused, so re-running after an interrupted pass only
pays for the frames that are missing. Pass ``--overwrite`` to query every frame again.

Examples
--------
Run the whole baseline over the 151 frames of the dataset::

    python3 -m evaluation.only_vlm

Re-build the tables and figures from annotations already on disk, without querying::

    python3 -m evaluation.only_vlm --skip-annotation

Re-read the answers of an earlier run into fresh annotations, also without querying::

    python3 -m evaluation.only_vlm --rebuild-from-raw
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

from evaluation import run_evaluation
from LLM.llm_base import BaseLLM, configure_env
from LLM.llm_factory import create_llm
from scene_understanding.gpt_annotator import DEFAULT_LLM_CONFIG_FILE
from utility.json_serialization import to_json_compatible as convert
from utility.utility import logger

RGB_DIR = Path("dataset/rgb")
OUTPUT_DIR = Path("results_only_vlm")
ARUCO_DIR = OUTPUT_DIR / "output_aruco"

# Subdirectories of --output-dir. The first two names match the layout of the results_*
# directories of the SAM runs, so the baseline can be diffed against them directly.
ANNOTATION_SUBDIR = "output_seg_annotation"
EVALUATION_SUBDIR = "evaluation_results"
RAW_SUBDIR = "vlm_raw"

# Every frame of the dataset. The SAM baselines were evaluated on 1-104 only; the ArUco
# ground truth covers all 151, and frames without ground truth are skipped anyway.
IMAGE_IDS = range(1, 152)

# The closed tag list of scene_understanding.gpt_annotator.ANNOTATION_PROMPT, kept
# verbatim -- leading spaces included -- because the ArUco ground truth is matched by
# name and the SAM baselines answered with exactly these strings.
TAGS = (
    '"Wide and large blue Lego block", "Small blue Lego block", "Yellow Lego block", ',
    '"Wide red Lego Block with 4 studs", "Green Lego block", "2x2 Blue and red Lego block", ',
    '"Tall red Lego block", "White and red box", "Blue and white small box", "Big Black Bottle", ',
    '"Big White bottle", "Metallic Wrench", "Orange Lego block", "Orange small box", ',
    '"White and green box", "Blue meter", "Black wallet", "Voltimeter", "Calculator", ',
    '"Blackberry purple and white box", "Ice tea peach white and orange box", ',
    '"Passion fruit white purple yellow box", "Raspberry pink white box", "Green cup", ',
    '"White ping-pong", "Orange ping-pong", "End-effector with grappler", ',
    '"Full robotic arm", "Partial robotic arm", "Unknown object".',
)

TAG_INSTRUCTIONS = f"""- "tag": ONLY one of the following: {TAGS}
  Do not change the tags neither use other tags that are not in the list. If you are not
  sure about the tag, use "Unknown object"."""

# Used by --freeform-tags, mirroring gpt_annotator.FREEFORM_ANNOTATION_PROMPT. The tags it
# produces will rarely equal the ground truth names, so the recall it measures is a lower
# bound on what the model actually saw.
FREEFORM_TAG_INSTRUCTIONS = """- "tag": an ultra-specific name for the object. For example,
  instead of "lego block", say "furthest blue lego block with 4 studs". Add the colour in
  the tag."""

# The vision backends rescale an image so that its shortest side is this many pixels
# before the model ever sees it, and the boxes come back in *that* space no matter what
# size the prompt claims the frame is. Sending a frame that is already at that size makes
# the space the prompt describes and the space the model sees the same one; `prepare_image`
# and `rescale_bbox` are the two halves of that, and a frame already this small is sent
# untouched with a scale of 1.
VISION_SHORT_SIDE = 768

PROMPT_TEMPLATE = """You will receive one image of a robotic workspace, {width} pixels wide
and {height} pixels tall.

Your task is to do in one step what a segmentation model, an annotator and a depth sensor
would do together: find every object of the workspace, name it, place it, and say how far
it is from the camera.

For every object you see, return an entry with these fields:
{tag_instructions}
- "description": where the object is with respect to the other objects of the scene, for
  example on the left of, on the right of, in front of, behind or next to another object.
- "bbox": the bounding box of the object in pixels of the image you received, as
  [x_min, y_min, width, height]. x grows to the right starting from the left border, y
  grows downwards starting from the top border. The box must be tight around the object.
- "depth_mm": your best estimate of the distance, in millimetres, between the camera and
  the centre of the object.

List every object once, and only the objects you can actually see. Do not list the table,
the walls, the floor or the background.

Return ONLY raw JSON.
Do not use markdown code fences.
Do not write ```json.
{{
"objects": [
{{"tag": "string", "description": "string", "bbox": [0, 0, 0, 0], "depth_mm": 0.0}}
]
}}
"""


def sent_size(width: int, height: int) -> tuple[int, int]:
    """Return the size a frame of ``width`` x ``height`` is sent at.

    Parameters
    ----------
    width : int
        Width of the frame in pixels.
    height : int
        Height of the frame in pixels.

    Returns
    -------
    tuple[int, int]
        The size the frame is downscaled to, or its own size when it is already
        small enough that the backend would not touch it.
    """
    shortest = min(width, height)
    if shortest <= VISION_SHORT_SIDE:
        return width, height
    ratio = VISION_SHORT_SIDE / shortest
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def prepare_image(image: Image.Image) -> tuple[Image.Image, tuple[float, float]]:
    """Downscale a frame to the size the backend would rescale it to anyway.

    Parameters
    ----------
    image : Image.Image
        The frame to send.

    Returns
    -------
    tuple[Image.Image, tuple[float, float]]
        The image to send, and the per-axis factors that map a box of the answer
        back to the pixels of ``image``. Both factors are ``1.0`` when the frame is
        sent untouched.
    """
    width, height = image.size
    target = sent_size(width, height)
    if target == (width, height):
        return image, (1.0, 1.0)
    return image.resize(target, Image.Resampling.LANCZOS), (width / target[0], height / target[1])


def rescale_bbox(
    bbox: list[int], scale: tuple[float, float], frame_size: tuple[int, int]
) -> list[int]:
    """Map a box from the size the model answered in back to the frame's own pixels.

    Parameters
    ----------
    bbox : list[int]
        Box as ``[x_min, y_min, width, height]`` in the coordinates of the sent image.
    scale : tuple[float, float]
        Per-axis factors from :func:`prepare_image`.
    frame_size : tuple[int, int]
        Size of the original frame, which the scaled box is clamped to.

    Returns
    -------
    list[int]
        The box in the pixels of the original frame.
    """
    scale_x, scale_y = scale
    if scale_x == 1.0 and scale_y == 1.0:
        return bbox

    x_min, y_min, box_width, box_height = bbox
    frame_width, frame_height = frame_size

    x_max = min(round((x_min + box_width) * scale_x), frame_width)
    y_max = min(round((y_min + box_height) * scale_y), frame_height)
    x_min = min(round(x_min * scale_x), frame_width - 1)
    y_min = min(round(y_min * scale_y), frame_height - 1)

    return [x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min)]


def build_prompt(image: Image.Image, freeform: bool = False) -> str:
    """Fill the prompt template in with the size of the frame and the tagging rules.

    Parameters
    ----------
    image : Image.Image
        The frame that is sent along with the prompt. Its size is written into the
        prompt so that the boxes come back in pixels of the original image rather than
        in a normalized range the model picks on its own.
    freeform : bool
        Ask for ultra-specific free-text tags instead of the closed list.

    Returns
    -------
    str
        The prompt to send with the frame.
    """
    width, height = image.size
    return PROMPT_TEMPLATE.format(
        width=width,
        height=height,
        tag_instructions=FREEFORM_TAG_INSTRUCTIONS if freeform else TAG_INSTRUCTIONS,
    )


def extract_json(raw: str) -> Optional[Any]:
    """Read the JSON payload out of a model answer.

    The prompt asks for raw JSON, but models still wrap it in markdown fences or put a
    sentence around it, and one malformed answer should not cost the whole run.

    Parameters
    ----------
    raw : str
        The answer of the model.

    Returns
    -------
    Any or None
        The parsed payload, or ``None`` when nothing parseable was found.
    """
    text = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the widest {...} or [...] span of the answer, which is what is left
    # when the model prefixed the JSON with a sentence.
    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    return None


def as_object_list(payload: Any) -> list[dict]:
    """Normalize a parsed answer into the list of objects it describes.

    Parameters
    ----------
    payload : Any
        The parsed answer, which is either the list itself, a mapping holding it under
        one of the usual keys, or a single object.

    Returns
    -------
    list[dict]
        The objects of the answer. Empty when the payload has no recognizable shape.
    """
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]

    if isinstance(payload, dict):
        for key in ("objects", "detections", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        if "bbox" in payload:
            return [payload]

    return []


def clean_bbox(value: Any, width: int, height: int) -> Optional[list[int]]:
    """Turn the box of an answer into an ``[x, y, w, h]`` box inside the frame.

    Parameters
    ----------
    value : Any
        The ``bbox`` field of the answer.
    width : int
        Width of the frame in pixels.
    height : int
        Height of the frame in pixels.

    Returns
    -------
    list[int] or None
        The clamped box, or ``None`` when the field is not four numbers or when the box
        has no area once clamped to the frame.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x, y, box_width, box_height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None

    # A negative width or height means the model gave two corners in the other order.
    if box_width < 0:
        x, box_width = x + box_width, -box_width
    if box_height < 0:
        y, box_height = y + box_height, -box_height

    x_min = max(0, min(int(round(x)), width - 1))
    y_min = max(0, min(int(round(y)), height - 1))
    x_max = max(0, min(int(round(x + box_width)), width))
    y_max = max(0, min(int(round(y + box_height)), height))

    if x_max <= x_min or y_max <= y_min:
        return None

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def clean_depth(value: Any) -> Optional[float]:
    """Turn the depth of an answer into millimetres.

    Parameters
    ----------
    value : Any
        The ``depth_mm`` field of the answer.

    Returns
    -------
    float or None
        The depth in millimetres, or ``None`` when it is missing or not a positive
        number. ``None`` is what the depth evaluation counts as an ignored object, which
        is also how a missing sensor reading is reported by the normal pipeline.
    """
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        value = match.group(0) if match else None
        if value is None:
            return None

    try:
        depth = float(value)
    except (TypeError, ValueError):
        return None

    return depth if depth > 0.0 else None


def build_image_dict(
    objects: list[dict],
    width: int,
    height: int,
    scale: tuple[float, float] = (1.0, 1.0),
    frame_size: Optional[tuple[int, int]] = None,
) -> dict:
    """Turn the objects of an answer into the dictionary the evaluation reads.

    The centre is the centre of the box, computed exactly as
    ``mapping.rgbd_mapper.attach_object_depths`` computes it for the normal pipeline, so
    the two runs place their objects the same way.

    Parameters
    ----------
    objects : list[dict]
        The objects of the answer.
    width : int
        Width, in pixels, of the image the model answered about.
    height : int
        Height, in pixels, of the image the model answered about.
    scale : tuple[float, float]
        Per-axis factors mapping the boxes of the answer back to the original frame,
        as returned by :func:`prepare_image`. ``(1.0, 1.0)`` leaves them alone.
    frame_size : tuple[int, int] or None
        Size of the original frame. Defaults to ``(width, height)``, which is right
        whenever ``scale`` is ``(1.0, 1.0)``.

    Returns
    -------
    dict
        One entry per object, keyed ``"mask_0"``, ``"mask_1"``, ..., each holding
        ``"tag"``, ``"description"``, ``"bbox"`` and ``"coord_center&depth"``. The
        boxes are in the pixels of the original frame.
    """
    image_dict: dict[str, dict] = {}

    for entry in objects:
        bbox = clean_bbox(entry.get("bbox"), width, height)
        if bbox is None:
            logger.warning(f"Dropping an object with an unusable bbox: {entry.get('bbox')!r}")
            continue

        bbox = rescale_bbox(bbox, scale, frame_size or (width, height))

        x_min, y_min, box_width, box_height = bbox
        center_x = (x_min + x_min + box_width) // 2
        center_y = (y_min + y_min + box_height) // 2

        tag = entry.get("tag")
        description = entry.get("description")

        image_dict[f"mask_{len(image_dict)}"] = {
            "tag": tag if isinstance(tag, str) and tag.strip() else "Unknown object",
            "description": description if isinstance(description, str) else "",
            "bbox": bbox,
            "coord_center&depth": [center_x, center_y, clean_depth(entry.get("depth_mm"))],
        }

    return image_dict


def annotate_image(llm: BaseLLM, image_path: Path, freeform: bool = False) -> tuple[dict, str]:
    """Ask the model to describe one frame, and turn the answer into annotations.

    Parameters
    ----------
    llm : BaseLLM
        The vision backend to query.
    image_path : Path
        The frame to send.
    freeform : bool
        Ask for free-text tags instead of the closed list.

    Returns
    -------
    tuple[dict, str]
        The annotations of the frame, and the raw answer of the model. The annotations
        are empty when the model failed or answered with something unparseable, which is
        counted by the evaluation as a frame where nothing was detected.
    """
    image = Image.open(image_path)
    width, height = image.size

    # The prompt states the size of the image that is actually sent, so that the boxes
    # come back in a space this script can map to the frame rather than in whatever
    # space the backend happened to downscale to.
    sent, scale = prepare_image(image)
    sent_width, sent_height = sent.size

    success, raw = llm.query(build_prompt(sent, freeform), images=[sent])

    if not success or not raw:
        logger.error(f"No response for {image_path.name}")
        return {}, raw or ""

    payload = extract_json(raw)
    if payload is None:
        logger.error(f"Could not decode the answer for {image_path.name}")
        return {}, raw

    objects = as_object_list(payload)
    if not objects:
        logger.error(f"No object in the answer for {image_path.name}")
        return {}, raw

    return build_image_dict(objects, sent_width, sent_height, scale, (width, height)), raw


def annotate(
    llm: BaseLLM,
    image_ids: Iterable[int],
    rgb_dir: Path,
    annotation_dir: Path,
    raw_dir: Path,
    overwrite: bool = False,
    freeform: bool = False,
) -> int:
    """Annotate every frame and write one JSON file per frame.

    Parameters
    ----------
    llm : BaseLLM
        The vision backend to query.
    image_ids : Iterable[int]
        Indices of the frames to annotate.
    rgb_dir : Path
        Directory containing the RGB frames.
    annotation_dir : Path
        Directory the per-frame JSON files are written to.
    raw_dir : Path
        Directory the raw answers of the model are written to.
    overwrite : bool
        Query the frames that already have an annotation file instead of reusing it.
    freeform : bool
        Ask for free-text tags instead of the closed list.

    Returns
    -------
    int
        How many frames were queried.
    """
    annotation_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    queried = 0
    for image_id in image_ids:
        image_path = rgb_dir / f"rgb_dataset_{image_id}.png"
        if not image_path.exists():
            logger.warning(f"[MISSING] {image_path}")
            continue

        output_path = annotation_dir / f"output_img{image_id}.json"
        if output_path.exists() and not overwrite:
            logger.info(f"Reusing {output_path}")
            continue

        logger.info(f"Querying the model for {image_path.name}")
        start = time.time()
        image_dict, raw = annotate_image(llm, image_path, freeform)
        logger.debug(f"Answer for {image_path.name} in {time.time() - start}s")

        (raw_dir / f"img{image_id}.txt").write_text(raw)
        with open(output_path, "w") as output_file:
            json.dump(image_dict, output_file, indent=4, default=convert)

        logger.info(f"{image_path.name}: {len(image_dict)} object(s)")
        queried += 1

    return queried


def rebuild_from_raw(
    image_ids: Iterable[int], rgb_dir: Path, annotation_dir: Path, raw_dir: Path
) -> int:
    """Rebuild the annotations from the answers already saved under ``raw_dir``.

    The raw answers are the expensive part of a run, and they stay valid when only the
    way an answer is turned into annotations changes. Parsing them again costs nothing
    and rewrites every ``output_img<N>.json`` from the text the model already returned.

    Parameters
    ----------
    image_ids : Iterable[int]
        Indices of the frames to rebuild.
    rgb_dir : Path
        Directory containing the RGB frames, read for their size.
    annotation_dir : Path
        Directory the per-frame JSON files are rewritten to.
    raw_dir : Path
        Directory holding the raw answers of an earlier run.

    Returns
    -------
    int
        How many frames were rebuilt.
    """
    annotation_dir.mkdir(parents=True, exist_ok=True)

    rebuilt = 0
    for image_id in image_ids:
        raw_path = raw_dir / f"img{image_id}.txt"
        image_path = rgb_dir / f"rgb_dataset_{image_id}.png"
        if not raw_path.exists():
            logger.warning(f"[MISSING] {raw_path}")
            continue
        if not image_path.exists():
            logger.warning(f"[MISSING] {image_path}")
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        sent_width, sent_height = sent_size(width, height)
        scale = (width / sent_width, height / sent_height)

        payload = extract_json(raw_path.read_text())
        objects = as_object_list(payload) if payload is not None else []
        if not objects:
            logger.error(f"No object in the saved answer for {raw_path.name}")

        image_dict = build_image_dict(objects, sent_width, sent_height, scale, (width, height))
        with open(annotation_dir / f"output_img{image_id}.json", "w") as output_file:
            json.dump(image_dict, output_file, indent=4, default=convert)

        logger.info(f"{raw_path.name}: {len(image_dict)} object(s)")
        rebuilt += 1

    return rebuilt


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments of the baseline run.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=RGB_DIR,
        help=f"Directory containing the RGB frames (default: {RGB_DIR}).",
    )
    parser.add_argument(
        "--aruco-dir",
        type=Path,
        default=ARUCO_DIR,
        help=(
            "Directory containing the ArUco annotations written by "
            f"aruco/aruco_detector.py (default: {ARUCO_DIR})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            f"Directory holding {ANNOTATION_SUBDIR}/, {RAW_SUBDIR}/ and "
            f"{EVALUATION_SUBDIR}/ (default: {OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=Path(DEFAULT_LLM_CONFIG_FILE),
        help="Path to the LLM YAML configuration file (see LLM/conf).",
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to the environment file")
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help="Flag to indicate not to load the environment file",
    )
    parser.add_argument(
        "--images",
        type=run_evaluation.parse_image_ids,
        default=list(IMAGE_IDS),
        help=(
            "Frames to process, as a comma-separated list of indices and inclusive ranges, "
            f"e.g. 1-151 or 1,4,7-9 (default: {IMAGE_IDS.start}-{IMAGE_IDS.stop - 1})."
        ),
    )
    parser.add_argument(
        "--freeform-tags",
        action="store_true",
        help=(
            "Ask for ultra-specific free-text tags instead of the closed list the SAM "
            "baselines use. The tags will rarely match the ArUco names, so the measured "
            "recall becomes a lower bound"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Query every frame again instead of reusing the annotations already on disk",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Do not query the model; evaluate the annotations already on disk",
    )
    parser.add_argument(
        "--rebuild-from-raw",
        action="store_true",
        help=(
            f"Do not query the model; rebuild the annotations from the answers saved in "
            f"{RAW_SUBDIR}/ by an earlier run, then evaluate them. Use it after a change "
            "to the way an answer is read, to avoid paying for the frames again"
        ),
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only query the model, without writing the tables and the figures",
    )

    args = parser.parse_args()

    if not args.skip_annotation and not args.rebuild_from_raw:
        if not args.images_dir.is_dir():
            parser.error(f"Images directory does not exist: {args.images_dir}")
        if not args.llm_config.is_file():
            parser.error(f"LLM configuration file does not exist: {args.llm_config}")

    if args.rebuild_from_raw:
        if not args.images_dir.is_dir():
            parser.error(f"Images directory does not exist: {args.images_dir}")
        raw_dir = args.output_dir / RAW_SUBDIR
        if not raw_dir.is_dir():
            parser.error(f"No saved answers to rebuild from: {raw_dir}")

    if not args.skip_evaluation and not args.aruco_dir.is_dir():
        parser.error(
            f"ArUco directory does not exist: {args.aruco_dir}. Run "
            f"aruco/aruco_detector.py --out_dir {args.aruco_dir}, or point --aruco-dir at "
            f"the directory that holds the ArUco output of this dataset."
        )

    return args


def main(args: argparse.Namespace) -> None:
    """Run the VLM-only baseline and write its artefacts.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments.
    """
    annotation_dir = args.output_dir / ANNOTATION_SUBDIR
    evaluation_dir = args.output_dir / EVALUATION_SUBDIR
    raw_dir = args.output_dir / RAW_SUBDIR

    if args.rebuild_from_raw:
        rebuilt = rebuild_from_raw(args.images, args.images_dir, annotation_dir, raw_dir)
        logger.info(f"Rebuilt {rebuilt} annotation file(s) from the saved answers")
    elif not args.skip_annotation:
        # Also tells the LLM layer which file to read, so `--env-file` reaches the backend
        # instead of it falling back to the project root's `.env`.
        configure_env(args.env_file, load=not args.no_env_file)

        try:
            llm = create_llm(args.llm_config)
        except (FileNotFoundError, ValueError) as error:
            logger.error(f"Unable to configure the model: {error}")
            sys.exit(1)

        if not llm.SUPPORTS_IMAGES:
            logger.error(f"{type(llm).__name__} does not support images and cannot be used here.")
            sys.exit(1)

        try:
            queried = annotate(
                llm,
                args.images,
                args.images_dir,
                annotation_dir,
                raw_dir,
                overwrite=args.overwrite,
                freeform=args.freeform_tags,
            )
        finally:
            llm.close()

        logger.info(f"Queried the model for {queried} frame(s)")

    if args.skip_evaluation:
        return

    run_evaluation.run(
        seg_dir=annotation_dir,
        aruco_dir=args.aruco_dir,
        output_dir=evaluation_dir,
        rgb_dir=args.images_dir,
        image_ids=args.images,
    )


if __name__ == "__main__":
    start_all = time.time()
    main(parse_arguments())
    logger.info(f"Total time: {time.time() - start_all}s")
