"""Reusable RGB-D to coloured point-cloud conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from mapping.camera_model import Intrinsics
from mapping.rgbd_mapper import RGBDMapper


@dataclass(frozen=True)
class RGBDPointCloud:
    """Point cloud generated from one RGB-D frame pair."""

    point_cloud: Any
    aligned_depth_mm: np.ndarray


class RGBDPointCloudGenerator:
    """Generate aligned coloured point clouds from repeated RGB-D frame pairs.

    The expensive/static calibration setup is created once in ``__init__`` and
    reused for every subsequent frame. This is preferable to rebuilding the
    ``RGBDMapper`` each time when processing a stream or a sequence of images.
    """

    def __init__(
        self,
        color_size: tuple[int, int],
        depth_size: tuple[int, int],
        *,
        depth_unit_scale: float = 1.0,
        depth_trunc_m: float = 3.0,
    ) -> None:
        if depth_unit_scale <= 0.0:
            raise ValueError("depth_unit_scale must be positive")
        if depth_trunc_m <= 0.0:
            raise ValueError("depth_trunc_m must be positive")

        self._color_size = color_size
        self._depth_size = depth_size
        self._depth_unit_scale = depth_unit_scale
        self._depth_trunc_m = depth_trunc_m

        self._mapper = RGBDMapper.from_hardcoded(
            color_size=color_size,
            depth_size=depth_size,
        )

    @classmethod
    def from_frames(
        cls,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
        *,
        depth_unit_scale: float = 1.0,
        depth_trunc_m: float = 3.0,
    ) -> "RGBDPointCloudGenerator":
        """Create a generator using the dimensions of one RGB-D frame pair."""
        _validate_images(color_bgr, depth_raw)
        return cls(
            color_size=(color_bgr.shape[1], color_bgr.shape[0]),
            depth_size=(depth_raw.shape[1], depth_raw.shape[0]),
            depth_unit_scale=depth_unit_scale,
            depth_trunc_m=depth_trunc_m,
        )

    @property
    def depth_trunc_m(self) -> float:
        return self._depth_trunc_m

    def generate(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
    ) -> RGBDPointCloud:
        """Generate a coloured Open3D cloud and RGB-aligned depth image.

        ``depth_raw`` is never resized. The RGB image is resized only to the
        calibrated RGB resolution, matching the behaviour of ``RGBDMapper``.
        """
        _validate_images(color_bgr, depth_raw)
        self._validate_frame_sizes(color_bgr, depth_raw)

        aligned_depth_mm = self._mapper.align_depth_to_color(
            depth_raw,
            depth_unit_scale=self._depth_unit_scale,
        )

        point_cloud = create_point_cloud_from_aligned_depth(
            color_bgr,
            aligned_depth_mm,
            self._mapper.calibration.rgb_intrinsic,
            depth_trunc_m=self._depth_trunc_m,
        )

        return RGBDPointCloud(
            point_cloud=point_cloud,
            aligned_depth_mm=aligned_depth_mm,
        )

    def _validate_frame_sizes(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
    ) -> None:
        color_size = (color_bgr.shape[1], color_bgr.shape[0])
        depth_size = (depth_raw.shape[1], depth_raw.shape[0])

        if color_size != self._color_size:
            raise ValueError(f"RGB frame size changed from {self._color_size} to {color_size}")
        if depth_size != self._depth_size:
            raise ValueError(f"Depth frame size changed from {self._depth_size} to {depth_size}")


def create_aligned_point_cloud(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    *,
    depth_unit_scale: float = 1.0,
    depth_trunc_m: float = 3.0,
) -> tuple[Any, np.ndarray]:
    """One-shot compatibility wrapper around :class:`RGBDPointCloudGenerator`."""
    generator = RGBDPointCloudGenerator.from_frames(
        color_bgr,
        depth_raw,
        depth_unit_scale=depth_unit_scale,
        depth_trunc_m=depth_trunc_m,
    )
    result = generator.generate(color_bgr, depth_raw)
    return result.point_cloud, result.aligned_depth_mm


def create_point_cloud_from_aligned_depth(
    color_bgr: np.ndarray,
    aligned_depth_mm: np.ndarray,
    rgb_intrinsic: Intrinsics,
    *,
    depth_trunc_m: float = 3.0,
) -> Any:
    """Create a coloured point cloud from depth already registered to RGB.

    This is the common downstream path for both sensor-aligned and model-inferred
    depth. Intrinsics are scaled when the aligned map uses a different resolution.
    """
    _validate_images(color_bgr, aligned_depth_mm)
    if depth_trunc_m <= 0.0:
        raise ValueError("depth_trunc_m must be positive")

    depth_height, depth_width = aligned_depth_mm.shape
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    if color_rgb.shape[:2] != aligned_depth_mm.shape:
        color_rgb = cv2.resize(
            color_rgb,
            (depth_width, depth_height),
            interpolation=cv2.INTER_AREA,
        )

    scale_x = depth_width / rgb_intrinsic.width
    scale_y = depth_height / rgb_intrinsic.height
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        depth_width,
        depth_height,
        rgb_intrinsic.fx * scale_x,
        rgb_intrinsic.fy * scale_y,
        rgb_intrinsic.cx * scale_x,
        rgb_intrinsic.cy * scale_y,
    )
    depth_clean = np.asarray(aligned_depth_mm, dtype=np.float32).copy()
    depth_clean[~np.isfinite(depth_clean) | (depth_clean <= 0.0)] = 0.0
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_rgb, dtype=np.uint8))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_clean))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return point_cloud


def _validate_images(color_bgr: np.ndarray, depth_raw: np.ndarray) -> None:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 image")
    if depth_raw.ndim != 2:
        raise ValueError("depth_raw must be a single-channel image")
