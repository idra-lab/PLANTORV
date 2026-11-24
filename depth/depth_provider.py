"""Interchangeable sources of RGB-aligned metric depth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from mapping.camera_model import CameraToWorldTransform
from mapping.rgbd_mapper import DEFAULT_DEPTH_ASSOCIATION, RGBDMapper, attach_object_depths


@dataclass(frozen=True)
class DepthResult:
    """A depth map registered to an RGB image.

    ``depth_mm`` always contains axial depth in millimetres. Invalid pixels are
    represented by zero and excluded by ``valid_mask``.
    """

    depth_mm: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray | None = None
    metric: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth_mm.ndim != 2:
            raise ValueError("depth_mm must be a two-dimensional array")
        if self.depth_mm.dtype != np.float32:
            raise ValueError("depth_mm must have dtype float32")
        if self.valid_mask.shape != self.depth_mm.shape:
            raise ValueError("valid_mask must have the same shape as depth_mm")
        if self.valid_mask.dtype != np.bool_:
            raise ValueError("valid_mask must have dtype bool")
        if self.confidence is not None and self.confidence.shape != self.depth_mm.shape:
            raise ValueError("confidence must have the same shape as depth_mm")
        if np.any(self.valid_mask & (~np.isfinite(self.depth_mm) | (self.depth_mm <= 0.0))):
            raise ValueError("valid depth pixels must be finite and positive")


class DepthAligner(Protocol):
    """Minimal surface of :class:`~mapping.rgbd_mapper.RGBDMapper` used here.

    Typing the collaborator structurally lets tests inject a lightweight stub
    instead of a fully calibrated mapper.
    """

    calibration: Any

    def align_depth_to_color(
        self,
        depth_image: np.ndarray,
        /,
        *,
        depth_unit_scale: float,
    ) -> np.ndarray:
        """Project a raw depth frame into the RGB image plane."""
        ...


class DepthProvider(Protocol):
    """Interface implemented by RGB-aligned metric-depth backends."""

    def estimate(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray | None = None,
    ) -> DepthResult:
        """Estimate or align metric depth for one RGB frame."""
        ...


class SensorDepthProvider:
    """Adapt raw Femto Mega depth frames to the common depth contract."""

    def __init__(
        self,
        color_size: tuple[int, int],
        depth_size: tuple[int, int],
        *,
        depth_unit_scale: float = 1.0,
        mapper: DepthAligner | None = None,
    ) -> None:
        if depth_unit_scale <= 0.0:
            raise ValueError("depth_unit_scale must be positive")
        self._color_size = color_size
        self._depth_size = depth_size
        self._depth_unit_scale = depth_unit_scale
        self._mapper = mapper or RGBDMapper.from_hardcoded(
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
    ) -> "SensorDepthProvider":
        """Construct a provider using a representative RGB-D frame pair."""
        _validate_color(color_bgr)
        _validate_raw_depth(depth_raw)
        return cls(
            color_size=(color_bgr.shape[1], color_bgr.shape[0]),
            depth_size=(depth_raw.shape[1], depth_raw.shape[0]),
            depth_unit_scale=depth_unit_scale,
        )

    @property
    def rgb_intrinsic(self) -> Any:
        """Return the calibrated RGB intrinsics used for projection."""
        return self._mapper.calibration.rgb_intrinsic

    def estimate(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray | None = None,
    ) -> DepthResult:
        """Align a raw sensor depth frame with the supplied RGB frame."""
        _validate_color(color_bgr)
        if depth_raw is None:
            raise ValueError("SensorDepthProvider requires depth_raw")
        _validate_raw_depth(depth_raw)

        color_size = (color_bgr.shape[1], color_bgr.shape[0])
        depth_size = (depth_raw.shape[1], depth_raw.shape[0])
        if color_size != self._color_size:
            raise ValueError(f"RGB frame size changed from {self._color_size} to {color_size}")
        if depth_size != self._depth_size:
            raise ValueError(f"Depth frame size changed from {self._depth_size} to {depth_size}")

        depth_mm = self._mapper.align_depth_to_color(
            depth_raw,
            depth_unit_scale=self._depth_unit_scale,
        )
        if depth_mm.shape != color_bgr.shape[:2]:
            depth_mm = cv2.resize(depth_mm, color_size, interpolation=cv2.INTER_NEAREST)

        depth_mm = _normalize_depth(depth_mm)
        return DepthResult(
            depth_mm=depth_mm,
            valid_mask=depth_mm > 0.0,
            metric=True,
            metadata={
                "source": "sensor",
                "alignment": "femto_mega_depth_to_rgb",
                "depth_unit_scale": self._depth_unit_scale,
            },
        )


def _normalize_depth(depth_mm: np.ndarray) -> np.ndarray:
    result = np.asarray(depth_mm, dtype=np.float32).copy()
    result[~np.isfinite(result) | (result <= 0.0)] = 0.0
    return np.ascontiguousarray(result)


def _validate_color(color_bgr: np.ndarray) -> None:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 image")


def _validate_raw_depth(depth_raw: np.ndarray) -> None:
    if depth_raw.ndim != 2:
        raise ValueError("depth_raw must be a single-channel image")


def main_coords(
    image_path: Union[str, Path, Image.Image, np.ndarray],
    depth_path: Union[str, Path, Image.Image, np.ndarray],
    dict_objects: dict,
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    association: str = DEFAULT_DEPTH_ASSOCIATION,
    camera_to_world_transform: Optional[CameraToWorldTransform] = None,
    camera_to_world_file: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Given the paths to an RGB image and a depth image, along with a dictionary of objects containing their bounding boxes, this function aligns the depth image to the RGB image and retrieves the depth information for each object.

    Parameters
    ----------
    image_path : Union[str, Path, Image.Image, np.ndarray]
        The file path to the RGB image, or the image itself as a PIL Image or NumPy array.
    depth_path : Union[str, Path, Image.Image, np.ndarray]
        The file path to the depth image.
    dict_objects : dict
        A dictionary where each key is an object identifier and each value is another dictionary containing at least a "bbox" key with the bounding box coordinates [x_min, y_min, width, height].
    masks : Sequence[np.ndarray] or None
        One binary mask per object, in the same order as the objects, as left in
        ``SegmentationModel.last_masks``. Required by the "mask-median" association
        and ignored by "bbox-center".
    association : str
        Which depth-association strategy to use; see :func:`attach_object_depths`.
    camera_to_world_transform : CameraToWorldTransform or None
        Optional already-loaded transform from the colour camera optical frame
        into ``world``.
    camera_to_world_file : str or Path or None
        Optional YAML file written by ``save_camera_world_transform``. It is
        used when ``camera_to_world_transform`` is not supplied.

    Returns
    -------
    dict
        The input dictionary of objects, updated for each object with "coord_center&depth"
        (a list [center_x, center_y, depth_mm]), "object_depth_mm", "depth_association",
        "object_point_camera_m" (a list [x, y, z] in metres in the colour camera frame) and
        "object_point_world_m" (the same coordinates in world, when a saved
        camera-to-world transform is available), and "object_point_pixel" (the
        pixel that point was back-projected through).
        Under "mask-median" the centre is only the object's image-space reference centre:
        the depth is estimated over the mask rather than sampled at that pixel.
    """
    if isinstance(image_path, str) or isinstance(image_path, Path):
        if isinstance(image_path, str):
            image_path = Path(image_path)
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    elif isinstance(image_path, Image.Image):
        rgb = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    elif isinstance(image_path, np.ndarray):
        if image_path.ndim == 3 and image_path.shape[2] == 3:
            rgb = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("NumPy array must be a 3-channel RGB image")
    else:
        raise TypeError("image_path must be a str, Path, PIL.Image.Image, or np.ndarray")

    if isinstance(depth_path, str) or isinstance(depth_path, Path):
        if isinstance(depth_path, str):
            depth_path = Path(depth_path)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)  # si es PNG de depth visual
    elif isinstance(depth_path, Image.Image):
        depth = np.array(depth_path)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path must be a str, Path, PIL.Image.Image, or np.ndarray")

    if rgb is None:
        raise ValueError(f"Failed to read RGB image from {image_path}")
    if depth is None:
        raise ValueError(f"Failed to read depth image from {depth_path}")

    provider = SensorDepthProvider.from_frames(rgb, depth, depth_unit_scale=1.0)
    result = provider.estimate(rgb, depth)
    return attach_object_depths(
        dict_objects,
        result.depth_mm,
        masks=masks,
        association=association,
        camera_to_world_transform=camera_to_world_transform,
        camera_to_world_file=camera_to_world_file,
    )
