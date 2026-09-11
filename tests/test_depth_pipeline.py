"""Tests for interchangeable metric-depth pipeline components."""

from types import SimpleNamespace

import numpy as np

from mapping.camera_model import (
    Distortion,
    Intrinsics,
    rgb_calibration_for_size,
    scale_intrinsics,
)
from mapping.depth_anything import DepthAnythingV2Provider, DepthAnythingV3Provider
from mapping.depth_provider import DepthResult, SensorDepthProvider
from mapping.rgbd_mapper import (
    _mask_median_reference_pixel,
    attach_object_depths,
    backproject_pixel_to_camera_m,
)
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


_PINHOLE = Intrinsics(cx=1.0, cy=1.0, fx=100.0, fy=200.0, width=3, height=3)
_NO_DISTORTION = Distortion(k1=0.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0, p1=0.0, p2=0.0)


def test_backprojection_puts_the_principal_point_on_the_optical_axis() -> None:
    point = backproject_pixel_to_camera_m(1, 1, 2000.0, _PINHOLE, _NO_DISTORTION)

    assert point == [0.0, 0.0, 2.0]


def test_backprojection_scales_the_offset_from_the_principal_point_by_the_depth() -> None:
    point = backproject_pixel_to_camera_m(2, 3, 1000.0, _PINHOLE, _NO_DISTORTION)

    # One pixel right of cx over fx=100, two pixels below cy over fy=200, at 1 m.
    assert point[0] == 0.01
    assert point[1] == 0.01
    assert point[2] == 1.0


def test_objects_get_no_camera_point_without_a_calibrated_frame_size() -> None:
    depth = np.full((3, 3), 1000.0, dtype=np.float32)

    result = attach_object_depths({"mask_0": {"bbox": [0, 0, 2, 2]}}, depth)

    assert result["mask_0"]["object_point_camera_m"] is None
    assert result["mask_0"]["object_point_pixel"] == [1, 1]


def test_objects_get_a_camera_point_from_the_calibration_of_the_frame_size() -> None:
    intrinsic, _ = rgb_calibration_for_size(640, 480)
    depth = np.full((480, 640), 1000.0, dtype=np.float32)
    centre_u = int(round(intrinsic.cx))
    centre_v = int(round(intrinsic.cy))
    objects = {"mask_0": {"bbox": [centre_u - 1, centre_v - 1, 2, 2]}}

    result = attach_object_depths(objects, depth)

    point = result["mask_0"]["object_point_camera_m"]
    assert point is not None
    assert point[2] == 1.0
    # The bounding box is centred on the principal point, so the ray is the optical axis
    # up to the rounding of the principal point to a whole pixel.
    assert abs(point[0]) < 0.01
    assert abs(point[1]) < 0.01
    assert result["mask_0"]["object_point_pixel"] == [centre_u, centre_v]


def test_objects_get_no_camera_point_without_a_depth() -> None:
    depth = np.zeros((480, 640), dtype=np.float32)

    result = attach_object_depths({"mask_0": {"bbox": [100, 100, 2, 2]}}, depth, neighborhood=0)

    assert result["mask_0"]["object_depth_mm"] is None
    assert result["mask_0"]["object_point_camera_m"] is None


def test_an_explicit_calibration_overrides_the_lookup_by_frame_size() -> None:
    depth = np.full((3, 3), 2000.0, dtype=np.float32)

    result = attach_object_depths(
        {"mask_0": {"bbox": [0, 0, 2, 2]}},
        depth,
        rgb_calibration=(_PINHOLE, _NO_DISTORTION),
    )

    assert result["mask_0"]["object_point_camera_m"] == [0.0, 0.0, 2.0]


def test_the_reference_pixel_is_the_centroid_when_it_carries_the_median_depth() -> None:
    depth = np.full((3, 3), 500.0, dtype=np.float32)
    mask = np.ones((3, 3), dtype=bool)

    assert _mask_median_reference_pixel(depth, mask, 500.0) == (1, 1)


def test_the_reference_pixel_leaves_a_centroid_whose_depth_is_not_the_median() -> None:
    depth = np.asarray(
        [[500.0, 500.0, 500.0], [500.0, 4000.0, 500.0], [500.0, 500.0, 500.0]],
        dtype=np.float32,
    )
    mask = np.ones((3, 3), dtype=bool)

    # The centroid (1, 1) sits in a spike, so the nearest pixel at the median is taken.
    pixel = _mask_median_reference_pixel(depth, mask, 500.0)

    assert pixel != (1, 1)
    assert depth[pixel[1], pixel[0]] == 500.0
    # Nearest of the four pixels one step away from the centroid, by column order.
    assert pixel == (1, 0)


def test_the_reference_pixel_tolerates_a_depth_a_tenth_of_a_millimetre_off() -> None:
    depth = np.full((3, 3), 500.05, dtype=np.float32)
    mask = np.ones((3, 3), dtype=bool)

    assert _mask_median_reference_pixel(depth, mask, 500.0) == (1, 1)


def test_the_reference_pixel_falls_back_to_the_closest_depth_to_the_median() -> None:
    # An even number of depths: the median is 1000.0, which no pixel carries.
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[0, 0] = 900.0
    depth[2, 2] = 1100.0
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True

    pixel = _mask_median_reference_pixel(depth, mask, 1000.0)

    # Both candidates are 100 mm from the median and equidistant from the centroid,
    # so the first of them in raster order is taken.
    assert pixel == (0, 0)


def test_the_reference_pixel_stays_inside_a_mask_whose_centroid_falls_outside_it() -> None:
    depth = np.full((3, 3), 500.0, dtype=np.float32)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True
    # The centroid is (1, 1), which the mask does not cover.

    pixel = _mask_median_reference_pixel(depth, mask, 500.0)

    assert mask[pixel[1], pixel[0]]


def test_mask_median_back_projects_through_the_reference_pixel_not_the_bbox_centre() -> None:
    depth = np.full((480, 640), 0.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:110, 200:210] = True
    depth[100:110, 200:210] = 800.0
    objects = {"mask_0": {"bbox": [0, 0, 4, 4]}}

    result = attach_object_depths(objects, depth, masks=[mask], association="mask-median")

    # The bounding box is nowhere near the mask, so the two pixels cannot be confused.
    assert result["mask_0"]["coord_center&depth"][:2] == [2, 2]
    assert result["mask_0"]["object_point_pixel"] == [204, 104]
    assert result["mask_0"]["object_point_camera_m"][2] == 0.8


def test_intrinsics_scale_with_the_sampling_density() -> None:
    scaled = scale_intrinsics(_PINHOLE, 6, 6)

    assert scaled.fx == 200.0
    assert scaled.fy == 400.0
    # The centre of source pixel 1 covers destination pixels 2 and 3, so the principal
    # point lands between them, at (1 + 0.5) * 2 - 0.5.
    assert scaled.cx == 2.5
    assert scaled.cy == 2.5
    assert (scaled.width, scaled.height) == (6, 6)


def test_intrinsics_refuse_a_size_of_another_aspect_ratio() -> None:
    try:
        scale_intrinsics(_PINHOLE, 6, 3)
    except ValueError as error:
        assert "aspect ratio" in str(error)
    else:
        raise AssertionError("scale_intrinsics accepted a change of aspect ratio")


def test_the_calibration_of_a_720p_frame_is_the_360p_one_scaled() -> None:
    recorded, _ = rgb_calibration_for_size(640, 360)
    scaled, distortion = rgb_calibration_for_size(1280, 720)

    assert scaled.fx == recorded.fx * 2.0
    assert scaled.fy == recorded.fy * 2.0
    assert (scaled.width, scaled.height) == (1280, 720)
    # Distortion acts on normalised coordinates, so it is the same at any frame size.
    assert distortion is rgb_calibration_for_size(640, 360)[1]


def test_a_frame_of_an_uncalibrated_aspect_ratio_has_no_calibration() -> None:
    assert rgb_calibration_for_size(1000, 777) is None


def test_a_720p_frame_and_a_360p_frame_give_the_same_ray() -> None:
    intrinsic_360, distortion = rgb_calibration_for_size(640, 360)
    intrinsic_720, _ = rgb_calibration_for_size(1280, 720)

    coarse = backproject_pixel_to_camera_m(200, 100, 1000.0, intrinsic_360, distortion)
    fine = backproject_pixel_to_camera_m(400, 200, 1000.0, intrinsic_720, distortion)

    # Pixel (400, 200) is half a pixel off the centre of pixel (200, 100), which at
    # fx = 747 px and 1 m is 0.7 mm.
    assert coarse[2] == fine[2] == 1.0
    assert abs(coarse[0] - fine[0]) < 0.001
    assert abs(coarse[1] - fine[1]) < 0.001


def test_objects_in_a_720p_frame_get_a_camera_point() -> None:
    depth = np.full((720, 1280), 1500.0, dtype=np.float32)

    result = attach_object_depths({"mask_0": {"bbox": [600, 340, 40, 40]}}, depth)

    point = result["mask_0"]["object_point_camera_m"]
    assert point is not None
    assert point[2] == 1.5
