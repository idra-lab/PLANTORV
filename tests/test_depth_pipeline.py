"""Tests for interchangeable metric-depth pipeline components."""

from types import SimpleNamespace

import numpy as np

from mapping.camera_model import Intrinsics
from mapping.depth_anything import DepthAnythingV2Provider
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


def test_attach_object_depths_preserves_center_and_nearest_fallback() -> None:
    depth = np.asarray(
        [[500.0, 0.0, 700.0], [0.0, 0.0, 900.0], [600.0, 800.0, 1000.0]],
        dtype=np.float32,
    )
    objects = {"mask_0": {"bbox": [0, 0, 2, 2]}}

    result = attach_object_depths(objects, depth)

    assert result["mask_0"]["coord_center&depth"] == [1, 1, 500.0]


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
