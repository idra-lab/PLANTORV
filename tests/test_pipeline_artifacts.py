import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline import artifacts


def _segment(tmp_path: Path, fid: int = 1, objects: int = 2) -> Path:
    """Write the segmentation artefacts of one frame and return the directory."""
    image_path = tmp_path / f"rgb_dataset_{fid}.png"
    Image.new("RGB", (8, 6)).save(image_path)

    segmentation_dir = tmp_path / artifacts.SEGMENTATION_SUBDIR
    masks = []
    for index in range(objects):
        mask = np.zeros((6, 8), dtype=bool)
        mask[index, index] = True
        masks.append(mask)

    artifacts.write_frame(
        segmentation_dir,
        fid,
        image_path,
        crops=[np.full((2, 2, 3), index, dtype=np.uint8) for index in range(objects)],
        bboxes=[[index, index, 2, 2] for index in range(objects)],
        masks=masks,
    )

    return segmentation_dir


def test_frame_id_reads_the_trailing_number_not_the_position() -> None:
    assert artifacts.frame_id("dataset/rgb/rgb_dataset_7.png") == 7
    assert artifacts.frame_id(Path("out/output_segmentation/frame_0007")) == 7


def test_frame_id_rejects_a_name_without_a_number() -> None:
    with pytest.raises(ValueError, match="must end with a number"):
        artifacts.frame_id("dataset/rgb/frame.png")


def test_images_are_listed_in_numeric_order(tmp_path: Path) -> None:
    for number in (10, 2, 1):
        Image.new("RGB", (2, 2)).save(tmp_path / f"rgb_dataset_{number}.png")

    assert [artifacts.frame_id(p) for p in artifacts.list_images(tmp_path)] == [1, 2, 10]


def test_a_frame_survives_the_round_trip(tmp_path: Path) -> None:
    segmentation_dir = _segment(tmp_path, fid=3, objects=2)

    frame = artifacts.read_frame(segmentation_dir, 3)

    assert artifacts.list_frame_ids(segmentation_dir) == [3]
    assert len(frame) == 2
    assert frame.bboxes == [[0, 0, 2, 2], [1, 1, 2, 2]]
    assert frame.image_path.name == "rgb_dataset_3.png"
    assert [mask.sum() for mask in frame.masks] == [1, 1]
    assert frame.masks[1][1, 1]
    assert [int(crop[0, 0, 0]) for crop in frame.crops] == [0, 1]


def test_writing_a_frame_rejects_artefacts_out_of_step(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same order"):
        artifacts.write_frame(
            tmp_path,
            1,
            tmp_path / "rgb_dataset_1.png",
            crops=[np.zeros((2, 2, 3), dtype=np.uint8)],
            bboxes=[[0, 0, 2, 2], [1, 1, 2, 2]],
            masks=[np.zeros((4, 4), dtype=bool)],
        )


def test_the_two_stages_do_not_overwrite_each_other(tmp_path: Path) -> None:
    keys = ["mask_0", "mask_1"]

    artifacts.update_results(
        tmp_path,
        1,
        {"mask_0": {"tag": "block", "description": "a red block"}, "mask_1": {"tag": "arm"}},
        keys=keys,
    )
    artifacts.update_results(
        tmp_path,
        1,
        {
            "mask_0": {"bbox": [0, 0, 2, 2], "object_depth_mm": 1200.0},
            "mask_1": {"bbox": [1, 1, 2, 2]},
        },
        keys=keys,
    )

    written = json.loads(artifacts.result_path(tmp_path, 1).read_text())

    assert written["mask_0"] == {
        "tag": "block",
        "description": "a red block",
        "bbox": [0, 0, 2, 2],
        "object_depth_mm": 1200.0,
    }
    assert written["mask_1"] == {"tag": "arm", "bbox": [1, 1, 2, 2]}


def test_objects_that_no_longer_exist_are_dropped(tmp_path: Path) -> None:
    artifacts.update_results(tmp_path, 1, {"mask_0": {"tag": "a"}, "mask_1": {"tag": "b"}})
    artifacts.update_results(tmp_path, 1, {"mask_0": {"object_depth_mm": 1.0}}, keys=["mask_0"])

    assert list(json.loads(artifacts.result_path(tmp_path, 1).read_text())) == ["mask_0"]


def test_objects_are_written_in_numeric_order(tmp_path: Path) -> None:
    artifacts.update_results(tmp_path, 1, {f"mask_{index}": {"tag": "x"} for index in range(11)})

    written = json.loads(artifacts.result_path(tmp_path, 1).read_text())

    assert list(written)[:3] == ["mask_0", "mask_1", "mask_2"]
    assert list(written)[-1] == "mask_10"


def test_re_segmenting_into_fewer_objects_leaves_no_stale_crop(tmp_path: Path) -> None:
    segmentation_dir = _segment(tmp_path, fid=1, objects=3)
    _segment(tmp_path, fid=1, objects=1)

    frame = artifacts.read_frame(segmentation_dir, 1)

    assert len(frame) == 1
    assert len(list((artifacts.frame_dir(segmentation_dir, 1) / "crops").glob("*.png"))) == 1
