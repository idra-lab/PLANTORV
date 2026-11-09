"""The on-disk contract between the pipeline stages.

``segmentation.py`` writes one directory per frame under
``<output-dir>/output_segmentation``::

    output_segmentation/
        frame_0001/
            meta.json           the source frame, its size, and one entry per object
            masks.npz           the object masks, stacked (N, H, W) and compressed
            crops/object_00.png the cropped objects, in the order of meta.json

``annotation.py`` and ``depth_estimation.py`` both read that directory and both
write ``<output-dir>/output_img<frame-id>.json``, the file
``evaluation/run_evaluation.py`` reads. Neither owns it: annotation fills the
``tag`` and ``description`` of every object, depth fills its coordinates and
distance, and :func:`update_results` merges into whatever is already there. That
is what lets the two run in any order, and either of them run again on its own.

Frames are identified by the number their file name ends with -- ``frame_id`` of
``rgb_dataset_7.png`` is ``7`` -- and not by their position in the directory. The
stages run separately, over directories that need not hold the same frames, so a
positional index would silently pair the wrong artefacts.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from utility.json_serialization import to_json_compatible as convert

# Subdirectory of --output-dir that segmentation.py writes to, and that the other
# two stages read by default.
SEGMENTATION_SUBDIR = "output_segmentation"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass
class FrameArtifacts:
    """What segmentation left on disk for one frame.

    Attributes
    ----------
    frame_id : int
        The number the source file name ends with.
    image_path : Path
        The RGB frame the masks were computed on.
    crops : list[numpy.ndarray]
        One cropped object per mask, in mask order. Empty when they were not read.
    bboxes : list[list[int]]
        One ``[x_min, y_min, width, height]`` per mask, in mask order.
    masks : list[numpy.ndarray]
        One binary mask per object, in the order the annotators and the depth
        association expect. Empty when they were not read.
    """

    frame_id: int
    image_path: Path
    crops: List[np.ndarray] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        """Return how many objects the frame holds."""
        return len(self.bboxes)


def frame_id(path: Union[str, Path]) -> int:
    """Return the number a frame file name ends with.

    Parameters
    ----------
    path : Union[str, Path]
        A frame file or a frame directory, e.g. ``rgb_dataset_7.png`` or ``frame_0007``.

    Returns
    -------
    int
        The trailing number.

    Raises
    ------
    ValueError
        If the name ends with no number, since the stages have then no way to pair
        this frame with the same frame of another directory.
    """
    stem = Path(path).stem
    tail = stem.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise ValueError(
            f"Cannot tell which frame {stem!r} is: the name must end with a number, "
            "as in rgb_dataset_7.png, so the stages can pair their artefacts."
        )
    return int(tail)


def list_images(images_dir: Union[str, Path]) -> List[Path]:
    """List the images of a directory, ordered by frame number.

    Parameters
    ----------
    images_dir : Union[str, Path]
        Directory holding the frames.

    Returns
    -------
    List[Path]
        The image files, ordered by :func:`frame_id`.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or holds no image.
    """
    directory = Path(images_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Images directory not found: {directory}")

    images = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise FileNotFoundError(f"No image found in {directory}")

    return sorted(images, key=frame_id)


def index_by_frame(paths: Sequence[Path]) -> Dict[int, Path]:
    """Index paths by the number their name ends with.

    Parameters
    ----------
    paths : Sequence[Path]
        Files whose names end with a frame number.

    Returns
    -------
    Dict[int, Path]
        The frame number of each file mapped to the file.
    """
    return {frame_id(path): path for path in paths}


def frame_dir(segmentation_dir: Union[str, Path], fid: int) -> Path:
    """Return the directory holding the artefacts of one frame.

    Parameters
    ----------
    segmentation_dir : Union[str, Path]
        The ``output_segmentation`` directory.
    fid : int
        The frame number.

    Returns
    -------
    Path
        ``<segmentation_dir>/frame_<fid>``, zero-padded so the directories sort.
    """
    return Path(segmentation_dir) / f"frame_{fid:04d}"


def write_frame(
    segmentation_dir: Union[str, Path],
    fid: int,
    image_path: Union[str, Path],
    crops: Sequence[np.ndarray],
    bboxes: Sequence[Sequence[int]],
    masks: Sequence[np.ndarray],
) -> Path:
    """Write the segmentation artefacts of one frame.

    Parameters
    ----------
    segmentation_dir : Union[str, Path]
        The ``output_segmentation`` directory.
    fid : int
        The frame number.
    image_path : Union[str, Path]
        The RGB frame the masks were computed on, recorded so that the later
        stages need not be told where the dataset is.
    crops : Sequence[numpy.ndarray]
        One cropped object per mask, in mask order.
    bboxes : Sequence[Sequence[int]]
        One ``[x_min, y_min, width, height]`` per mask, in mask order.
    masks : Sequence[numpy.ndarray]
        One binary mask per object, in mask order.

    Returns
    -------
    Path
        The directory that was written.

    Raises
    ------
    ValueError
        If the three sequences do not have the same length, which would leave the
        crops, the boxes and the masks of an object out of step.
    """
    if not len(crops) == len(bboxes) == len(masks):
        raise ValueError(
            f"Frame {fid}: {len(crops)} crops, {len(bboxes)} boxes and {len(masks)} masks. "
            "The three describe the same objects and must be in the same order."
        )

    directory = frame_dir(segmentation_dir, fid)
    crops_dir = directory / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # A stage that runs again over a frame that now holds fewer objects must not
    # leave the crops of the previous run behind for the annotator to read.
    for stale in crops_dir.glob("object_*.png"):
        stale.unlink()

    crop_names = []
    for index, crop in enumerate(crops):
        name = f"object_{index:02d}.png"
        Image.fromarray(np.asarray(crop, dtype=np.uint8)).save(crops_dir / name)
        crop_names.append(f"crops/{name}")

    # Bool masks compress to a few kilobytes each, and stacking them keeps them in
    # the one order everything downstream relies on. A label image would not do:
    # the masks overlap.
    stacked = (
        np.stack([np.asarray(mask, dtype=bool) for mask in masks])
        if masks
        else np.zeros((0, 0, 0), dtype=bool)
    )
    np.savez_compressed(directory / "masks.npz", masks=stacked)

    meta = {
        "frame_id": fid,
        "image": str(image_path),
        "objects": [
            {"index": index, "bbox": bbox, "crop": name}
            for index, (bbox, name) in enumerate(zip(bboxes, crop_names))
        ],
    }
    with open(directory / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=4, default=convert)

    return directory


def list_frame_ids(segmentation_dir: Union[str, Path]) -> List[int]:
    """List the frames a segmentation directory holds, in numeric order.

    Parameters
    ----------
    segmentation_dir : Union[str, Path]
        The ``output_segmentation`` directory.

    Returns
    -------
    List[int]
        The frame numbers, ordered.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or holds no frame.
    """
    directory = Path(segmentation_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory not found: {directory}. Run segmentation.py first."
        )

    ids = sorted(frame_id(p) for p in directory.glob("frame_*") if (p / "meta.json").is_file())
    if not ids:
        raise FileNotFoundError(
            f"No frame found in {directory}. It should hold one frame_<n>/ directory per "
            "frame, as written by segmentation.py."
        )

    return ids


def read_frame(
    segmentation_dir: Union[str, Path],
    fid: int,
    *,
    crops: bool = True,
    masks: bool = True,
) -> FrameArtifacts:
    """Read the segmentation artefacts of one frame.

    Parameters
    ----------
    segmentation_dir : Union[str, Path]
        The ``output_segmentation`` directory.
    fid : int
        The frame number.
    crops : bool
        Whether to read the cropped objects. The depth stage does not need them.
    masks : bool
        Whether to read the object masks. Only the "mask-median" depth association
        and the mask-based annotators need them.

    Returns
    -------
    FrameArtifacts
        The frame's objects, in the order everything downstream expects.

    Raises
    ------
    FileNotFoundError
        If the frame, its metadata or an artefact it names is missing.
    """
    directory = frame_dir(segmentation_dir, fid)
    meta_file = directory / "meta.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"No segmentation metadata for frame {fid}: {meta_file}")

    with open(meta_file) as handle:
        meta = json.load(handle)

    objects = meta.get("objects", [])
    artifacts = FrameArtifacts(
        frame_id=int(meta.get("frame_id", fid)),
        image_path=Path(meta["image"]),
        bboxes=[list(entry["bbox"]) for entry in objects],
    )

    if crops:
        artifacts.crops = [
            np.asarray(Image.open(directory / entry["crop"]).convert("RGB")) for entry in objects
        ]

    if masks:
        masks_file = directory / "masks.npz"
        if not masks_file.is_file():
            raise FileNotFoundError(f"No masks for frame {fid}: {masks_file}")
        with np.load(masks_file) as handle:
            stacked = handle["masks"]
        if len(stacked) != len(objects):
            raise ValueError(
                f"Frame {fid}: {len(stacked)} masks for {len(objects)} objects in "
                f"{meta_file}. The frame directory was written by two different runs."
            )
        artifacts.masks = [stacked[index] for index in range(len(stacked))]

    return artifacts


def result_path(output_dir: Union[str, Path], fid: int) -> Path:
    """Return the per-frame result file the two later stages share.

    Parameters
    ----------
    output_dir : Union[str, Path]
        The run directory.
    fid : int
        The frame number.

    Returns
    -------
    Path
        ``<output_dir>/output_img<fid>.json``, the name
        ``evaluation/run_evaluation.py`` expects.
    """
    return Path(output_dir) / f"output_img{fid}.json"


def read_results(output_dir: Union[str, Path], fid: int) -> Dict[str, Any]:
    """Read the per-frame result file, or return an empty one.

    Parameters
    ----------
    output_dir : Union[str, Path]
        The run directory.
    fid : int
        The frame number.

    Returns
    -------
    Dict[str, Any]
        What the other stage has written so far, keyed ``"mask_0"``, ``"mask_1"``, ...
        Empty when the file does not exist yet.
    """
    path = result_path(output_dir, fid)
    if not path.is_file():
        return {}

    with open(path) as handle:
        return json.load(handle)


def update_results(
    output_dir: Union[str, Path],
    fid: int,
    objects: Dict[str, Dict[str, Any]],
    *,
    keys: Optional[Sequence[str]] = None,
) -> Path:
    """Merge one stage's fields into the per-frame result file.

    The file belongs to no stage in particular: annotation writes the tag and the
    description of each object, depth writes its coordinates and distance, and
    whichever runs second must not drop what the first wrote. Fields of the same
    object are therefore merged rather than replaced.

    Parameters
    ----------
    output_dir : Union[str, Path]
        The run directory.
    fid : int
        The frame number.
    objects : Dict[str, Dict[str, Any]]
        The fields to write, keyed ``"mask_0"``, ``"mask_1"``, ...
    keys : Sequence[str] or None
        The objects the frame currently holds. Entries outside this set are
        dropped, so that re-segmenting a frame into fewer objects does not leave
        the annotations of objects that no longer exist behind. Defaults to the
        keys of ``objects``.

    Returns
    -------
    Path
        The file that was written.
    """
    kept = set(objects if keys is None else keys)

    merged = {key: value for key, value in read_results(output_dir, fid).items() if key in kept}
    for key, fields in objects.items():
        merged.setdefault(key, {}).update(fields)

    # "mask_10" must not sort before "mask_2" in the file the evaluation reads.
    ordered = {key: merged[key] for key in sorted(merged, key=lambda name: (len(name), name))}

    path = result_path(output_dir, fid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(ordered, handle, indent=4, default=convert)

    return path
