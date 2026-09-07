"""Tests for interchangeable metric-depth pipeline components."""

from types import SimpleNamespace

import numpy as np

from mapping.camera_model import Intrinsics
from mapping.depth_anything import DepthAnythingV2Provider, DepthAnythingV3Provider
from mapping.depth_provider import DepthResult, SensorDepthProvider
from mapping.rgbd_mapper import attach_object_depths
from mapping.rgbd_pointcloud import create_point_cloud_from_aligned_depth


class _FakeMapper:
    def __init__(self, aligned_depth: np.ndarray) -> None:
        self.aligned_depth = aligned_depth
        self.calibration = SimpleNamespace(rgb_intrinsic=None)

    def align_depth_to_color(
        self,
        depth_raw: np.ndarray,
        *,
        depth_unit_scale: float,
    ) -> np.ndarray:
        assert depth_raw.shape == (2, 2)
        assert depth_unit_scale == 1.0
        return self.aligned_depth.copy()


class _FakeProcessor:
    def __call__(self, *, images: np.ndarray, return_tensors: str) -> dict:
        assert images.shape == (2, 3, 3)
        assert return_tensors == "pt"
        return {"pixel_values": images}

    def post_process_depth_estimation(
        self,
        outputs: object,
        *,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict[str, np.ndarray]]:
        assert target_sizes == [(2, 3)]
        return [{"predicted_depth": np.asarray([[1.0, 2.0, np.nan], [0.5, -1.0, 3.0]])}]


class _FakeMetricModel:
    config = SimpleNamespace(depth_estimation_type="metric")

    def to(self, device: str) -> "_FakeMetricModel":
        self.device = device
        return self

    def eval(self) -> "_FakeMetricModel":
        return self

    def __call__(self, **inputs: np.ndarray) -> object:
        assert "pixel_values" in inputs
        return object()


class _FakeDA3Model:
    def to(self, *, device: str) -> "_FakeDA3Model":
        self.device = device
        return self

    def eval(self) -> "_FakeDA3Model":
        return self

    def inference(
        self,
        images: list[np.ndarray],
        *,
        process_res: int,
        process_res_method: str,
    ) -> SimpleNamespace:
        assert images[0].shape == (4, 6, 3)
        assert images[0][0, 0].tolist() == [30, 20, 10]
        assert process_res == 504
        assert process_res_method == "upper_bound_resize"
        return SimpleNamespace(
            depth=np.asarray([[[1.0, 2.0, np.nan], [0.5, -1.0, 3.0]]]),
            conf=np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
            sky=np.asarray([[[False, False, True], [False, False, False]]]),
        )


def test_attach_object_depths_preserves_center_and_nearest_fallback() -> None:
    depth = np.asarray(
        [[500.0, 0.0, 700.0], [0.0, 0.0, 900.0], [600.0, 800.0, 1000.0]],
        dtype=np.float32,
    )
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    result = attach_object_depths(objects, depth)

    assert result["mask_0"]["coord_center&depth"] == [1, 1, 500.0]


def test_attach_object_depths_defaults_to_bbox_center_when_masks_are_passed() -> None:
    """Handing the masks over does not change what the default association returns."""
    depth = np.asarray(
        [[500.0, 0.0, 700.0], [0.0, 0.0, 900.0], [600.0, 800.0, 1000.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[2, :] = True

    result = attach_object_depths({"mask_0": {"bbox": [0, 0, 2, 2]}}, depth, masks=[mask])

    assert result["mask_0"]["coord_center&depth"] == [1, 1, 500.0]
    assert result["mask_0"]["object_depth_mm"] == 500.0
    assert result["mask_0"]["depth_association"] == "bbox-center"


def test_mask_median_takes_the_median_of_the_depths_under_the_mask() -> None:
    depth = np.asarray(
        [[100.0, 200.0, 900.0], [300.0, 400.0, 900.0], [900.0, 900.0, 900.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[0:2, 0:2] = True
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    result = attach_object_depths(objects, depth, masks=[mask], association="mask-median")

    # The centre pixel holds 400.0; the median over the mask is (200 + 300) / 2.
    assert result["mask_0"]["coord_center&depth"] == [1, 1, 250.0]
    assert result["mask_0"]["object_depth_mm"] == 250.0
    assert result["mask_0"]["depth_association"] == "mask-median"


def test_mask_median_ignores_zero_negative_nan_and_inf_depths() -> None:
    depth = np.asarray(
        [[0.0, -50.0, np.nan], [np.inf, 600.0, 800.0], [1000.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[0:3, 0:3] = True
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    result = attach_object_depths(objects, depth, masks=[mask], association="mask-median")

    # Only 600, 800 and 1000 survive the validity test.
    assert result["mask_0"]["object_depth_mm"] == 800.0


def test_mask_median_ignores_depths_outside_the_mask() -> None:
    depth = np.asarray(
        [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0], [700.0, 800.0, 900.0]],
        dtype=np.float32,
    )
    inside = np.zeros((3, 3), dtype=bool)
    inside[0, 0:3] = True
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    result = attach_object_depths(objects, depth, masks=[inside], association="mask-median")

    # The whole image would have median 500.0; the first row alone has 200.0.
    assert result["mask_0"]["object_depth_mm"] == 200.0


def test_mask_median_falls_back_to_bbox_center_without_valid_depth() -> None:
    depth = np.asarray(
        [[500.0, 0.0, 700.0], [0.0, 0.0, 900.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    empty = np.zeros((3, 3), dtype=bool)
    empty[2, :] = True
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    result = attach_object_depths(objects, depth, masks=[empty], association="mask-median")

    # Identical to what bbox-center returns on the same frame.
    assert result["mask_0"]["coord_center&depth"] == [1, 1, 500.0]
    assert result["mask_0"]["object_depth_mm"] == 500.0
    assert result["mask_0"]["depth_association"] == "bbox-center"


def test_mask_median_rejects_a_mask_that_is_not_the_size_of_the_depth_image() -> None:
    depth = np.full((3, 3), 500.0, dtype=np.float32)
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    try:
        attach_object_depths(
            objects,
            depth,
            masks=[np.ones((4, 4), dtype=bool)],
            association="mask-median",
        )
    except ValueError as exc:
        assert "mask_0" in str(exc)
        assert "(4, 4)" in str(exc)
        assert "(3, 3)" in str(exc)
    else:
        raise AssertionError("attach_object_depths accepted a mask of the wrong shape")


def test_mask_median_rejects_a_mask_count_that_does_not_match_the_objects() -> None:
    depth = np.full((3, 3), 500.0, dtype=np.float32)
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}, "mask_1": {"bbox": [1, 1, 2, 2]}}

    try:
        attach_object_depths(
            objects,
            depth,
            masks=[np.ones((3, 3), dtype=bool)],
            association="mask-median",
        )
    except ValueError as exc:
        assert "1 masks for 2 objects" in str(exc)
    else:
        raise AssertionError("attach_object_depths accepted a truncated mask list")


def test_mask_median_matches_each_mask_to_its_own_object() -> None:
    depth = np.asarray(
        [[100.0, 100.0, 900.0], [100.0, 100.0, 900.0], [700.0, 700.0, 700.0]],
        dtype=np.float32,
    )
    first = np.zeros((3, 3), dtype=bool)
    first[0:2, 0:2] = True
    second = np.zeros((3, 3), dtype=bool)
    second[2, :] = True
    third = np.zeros((3, 3), dtype=bool)
    third[0:2, 2] = True
    objects = {
        "mask_0": {"bbox": [0, 0, 2, 2]},
        "mask_1": {"bbox": [0, 2, 3, 1]},
        "mask_2": {"bbox": [2, 0, 1, 2]},
    }

    result = attach_object_depths(
        objects,
        depth,
        masks=[first, second, third],
        association="mask-median",
    )

    assert result["mask_0"]["object_depth_mm"] == 100.0
    assert result["mask_1"]["object_depth_mm"] == 700.0
    assert result["mask_2"]["object_depth_mm"] == 900.0


def test_mask_median_requires_masks() -> None:
    depth = np.full((3, 3), 500.0, dtype=np.float32)

    try:
        attach_object_depths({"mask_0": {"bbox": [0, 0, 2, 2]}}, depth, association="mask-median")
    except ValueError as exc:
        assert "requires the per-object masks" in str(exc)
    else:
        raise AssertionError("attach_object_depths accepted mask-median without masks")


def test_attach_object_depths_rejects_an_unknown_association() -> None:
    depth = np.full((3, 3), 500.0, dtype=np.float32)

    try:
        attach_object_depths({"mask_0": {"bbox": [0, 0, 2, 2]}}, depth, association="centroid")
    except ValueError as exc:
        assert "unknown depth association" in str(exc)
    else:
        raise AssertionError("attach_object_depths accepted an unknown association")


def test_sensor_provider_matches_legacy_alignment_and_resize() -> None:
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    raw_depth = np.ones((2, 2), dtype=np.uint16)
    aligned = np.asarray([[100.0, 200.0], [0.0, np.nan]], dtype=np.float32)
    provider = SensorDepthProvider(
        color_size=(4, 4),
        depth_size=(2, 2),
        mapper=_FakeMapper(aligned),
    )

    result = provider.estimate(color, raw_depth)

    expected = np.asarray(
        [
            [100.0, 100.0, 200.0, 200.0],
            [100.0, 100.0, 200.0, 200.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.depth_mm, expected)
    np.testing.assert_array_equal(result.valid_mask, expected > 0.0)
    assert result.metric
    assert result.metadata["source"] == "sensor"


def test_depth_result_rejects_inconsistent_contract() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    invalid_mask = np.ones((1, 2), dtype=bool)

    try:
        DepthResult(depth_mm=depth, valid_mask=invalid_mask)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("DepthResult accepted an invalid mask")


def test_depth_anything_converts_metric_metres_to_millimetres() -> None:
    provider = DepthAnythingV2Provider(
        processor=_FakeProcessor(),
        model=_FakeMetricModel(),
        device="cpu",
    )
    color = np.zeros((2, 3, 3), dtype=np.uint8)

    result = provider.estimate(color)

    np.testing.assert_array_equal(
        result.depth_mm,
        np.asarray([[1000.0, 2000.0, 0.0], [500.0, 0.0, 3000.0]], dtype=np.float32),
    )
    assert result.metadata["source"] == "depth_anything_v2"


def test_depth_anything_v3_scales_metric_depth_and_aligns_to_rgb() -> None:
    provider = DepthAnythingV3Provider(
        model=_FakeDA3Model(),
        focal_length_px=600.0,
        device="cpu",
    )
    color = np.zeros((4, 6, 3), dtype=np.uint8)
    color[0, 0] = [10, 20, 30]

    result = provider.estimate(color)

    assert result.depth_mm.shape == (4, 6)
    assert result.depth_mm.dtype == np.float32
    assert result.confidence is not None
    assert result.confidence.shape == (4, 6)
    # The prediction is half-sized, so focal=300 px and 1 raw unit = 1 metre.
    assert result.depth_mm[0, 0] == 1000.0
    assert result.depth_mm[0, 4] == 0.0
    assert not result.valid_mask[0, 4]
    assert result.metadata["source"] == "depth_anything_v3"


def test_point_cloud_accepts_depth_already_aligned_to_rgb() -> None:
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    aligned_depth = np.asarray([[1000.0, 0.0], [2000.0, 3000.0]], dtype=np.float32)
    intrinsic = Intrinsics(cx=0.5, cy=0.5, fx=1.0, fy=1.0, width=2, height=2)

    cloud = create_point_cloud_from_aligned_depth(
        color,
        aligned_depth,
        intrinsic,
        depth_trunc_m=4.0,
    )

    assert len(cloud.points) == 3
