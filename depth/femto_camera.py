"""Depth-to-RGB projection utilities for Orbbec Femto calibration data."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Camera intrinsic parameters.

    Attributes
    ----------
    cx, cy : float
        Principal point coordinates in pixels.
    fx, fy : float
        Focal lengths in pixels.
    width, height : int
        Image size associated with the intrinsics.
    """

    cx: float
    cy: float
    fx: float
    fy: float
    width: int
    height: int


@dataclass(frozen=True)
class Distortion:
    """Camera distortion parameters.

    Attributes
    ----------
    k1, k2, k3, k4, k5, k6 : float
        Radial distortion coefficients.
    p1, p2 : float
        Tangential distortion coefficients.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float


@dataclass(frozen=True)
class CalibrationSet:
    """Calibration parameters for one depth/RGB resolution pair.

    Attributes
    ----------
    depth_distortion, rgb_distortion : Distortion
        Distortion parameters for the depth and RGB cameras.
    depth_intrinsic, rgb_intrinsic : Intrinsics
        Intrinsic parameters for the depth and RGB cameras.
    rot : numpy.ndarray
        Rotation matrix from depth coordinates to RGB coordinates, with shape
        ``(3, 3)``.
    trans : numpy.ndarray
        Translation vector from depth coordinates to RGB coordinates, with
        shape ``(3,)``.
    """

    depth_distortion: Distortion
    depth_intrinsic: Intrinsics
    rgb_distortion: Distortion
    rgb_intrinsic: Intrinsics
    rot: np.ndarray  # 3x3
    trans: np.ndarray  # 3,


@dataclass(frozen=True)
class AlignProfile:
    """Resolution-specific depth/RGB alignment profile.

    Attributes
    ----------
    align_type : int
        Alignment profile type.
    color_width, color_height : int
        RGB resolution.
    depth_width, depth_height : int
        Depth resolution.
    param_index : int
        Index into the hardcoded calibration table.
    align_left, align_top, align_right, align_bottom : int
        Alignment crop or offset metadata from the camera profile.
    depth_scale : float
        Profile depth scaling value.
    """

    align_type: int
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    param_index: int
    align_left: int
    align_top: int
    align_right: int
    align_bottom: int
    depth_scale: float


_ROT = np.asarray(
    [
        [0.994558, -0.00445435, 0.00197944],
        [0.00422393, 0.99455, 0.104173],
        [-0.00243268, -0.104163, 0.994557],
    ],
    dtype=np.float64,
)
_TRANS = np.asarray([-32.6072, -0.835282, 1.99768], dtype=np.float64)

_DEPTH_DIST = Distortion(
    k1=20.449,
    k2=9.65474,
    k3=0.311488,
    k4=20.7575,
    k5=16.556,
    k6=2.11015,
    p1=5.12616e-05,
    p2=-8.77438e-06,
)

_RGB_DIST = Distortion(
    k1=0.0767264,
    k2=-0.104236,
    k3=0.0419684,
    k4=0.0,
    k5=0.0,
    k6=0.0,
    p1=0.000112286,
    p2=-6.58385e-05,
)

_HARDCODED_CALIBRATIONS: List[CalibrationSet] = [
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(
            cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024
        ),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(
            cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(
            cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024
        ),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(
            cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(
            cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576
        ),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(
            cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(
            cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576
        ),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(
            cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
]

_HARDCODED_PROFILES: List[AlignProfile] = [
    AlignProfile(1, 3840, 2160, 1024, 1024, 0, 0, 0, 0, -1680, 3.75),
    AlignProfile(1, 2560, 1440, 1024, 1024, 0, 0, 0, 0, -1120, 2.5),
    AlignProfile(1, 1920, 1080, 1024, 1024, 0, 0, 0, 0, -840, 1.875),
    AlignProfile(1, 1280, 720, 1024, 1024, 0, 0, 0, 0, -560, 1.25),
    AlignProfile(1, 3840, 2160, 512, 512, 0, 0, 0, 0, -1680, 7.5),
    AlignProfile(1, 2560, 1440, 512, 512, 0, 0, 0, 0, -1120, 5.0),
    AlignProfile(1, 1920, 1080, 512, 512, 0, 0, 0, 0, -840, 3.75),
    AlignProfile(1, 1280, 720, 512, 512, 0, 0, 0, 0, -560, 2.5),
    AlignProfile(1, 3840, 2160, 640, 576, 2, 0, 0, 0, -1296, 6.0),
    AlignProfile(1, 2560, 1440, 640, 576, 2, 0, 0, 0, -864, 4.0),
    AlignProfile(1, 1920, 1080, 640, 576, 2, 0, 0, 0, -648, 3.0),
    AlignProfile(1, 1280, 720, 640, 576, 2, 0, 0, 0, -432, 2.0),
    AlignProfile(1, 3840, 2160, 320, 288, 2, 0, 0, 0, -1296, 12.0),
    AlignProfile(1, 2560, 1440, 320, 288, 2, 0, 0, 0, -864, 8.0),
    AlignProfile(1, 1920, 1080, 320, 288, 2, 0, 0, 0, -648, 6.0),
    AlignProfile(1, 1280, 720, 320, 288, 2, 0, 0, 0, -432, 4.0),
    AlignProfile(1, 1280, 960, 1024, 1024, 1, 0, 0, 0, -320, 1.25),
    AlignProfile(1, 1280, 960, 512, 512, 1, 0, 0, 0, -320, 2.5),
    AlignProfile(1, 1280, 960, 640, 576, 3, 0, 0, 0, -192, 2.0),
    AlignProfile(1, 1280, 960, 320, 288, 3, 0, 0, 0, -192, 4.0),
    AlignProfile(2, 3840, 2160, 1024, 1024, 0, 0, 0, 0, 0, 6.0),
]


def _distort_normalized(
    x: np.ndarray, y: np.ndarray, d: Distortion
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply radial and tangential distortion to normalized points.

    Parameters
    ----------
    x, y : numpy.ndarray
        Undistorted normalized coordinates.
    d : Distortion
        Distortion coefficients.

    Returns
    -------
    x_distorted, y_distorted : tuple[numpy.ndarray, numpy.ndarray]
        Distorted normalized coordinates.
    """
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    num = 1.0 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6
    den = 1.0 + d.k4 * r2 + d.k5 * r4 + d.k6 * r6
    radial = num / den
    x_tan = 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x)
    y_tan = d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y
    return x * radial + x_tan, y * radial + y_tan


def _undistort_pixels_to_normalized(
    u: np.ndarray,
    v: np.ndarray,
    intr: Intrinsics,
    dist: Distortion,
    iters: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert distorted pixels to undistorted normalized coordinates.

    Parameters
    ----------
    u, v : numpy.ndarray
        Distorted pixel coordinates.
    intr : Intrinsics
        Camera intrinsic parameters.
    dist : Distortion
        Camera distortion parameters.
    iters : int, optional
        Number of fixed-point refinement iterations.

    Returns
    -------
    x, y : tuple[numpy.ndarray, numpy.ndarray]
        Undistorted normalized coordinates.
    """
    xd = (u - intr.cx) / intr.fx
    yd = (v - intr.cy) / intr.fy

    # Fixed-point refinement for inverse distortion.
    xu = xd.copy()
    yu = yd.copy()
    for _ in range(iters):
        x_est, y_est = _distort_normalized(xu, yu, dist)
        xu += xd - x_est
        yu += yd - y_est
    return xu, yu


def _project_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    intr: Intrinsics,
    dist: Distortion,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project normalized coordinates to distorted image pixels.

    Parameters
    ----------
    x, y : numpy.ndarray
        Undistorted normalized coordinates.
    intr : Intrinsics
        Camera intrinsic parameters.
    dist : Distortion
        Camera distortion parameters.

    Returns
    -------
    u, v : tuple[numpy.ndarray, numpy.ndarray]
        Distorted pixel coordinates.
    """
    xd, yd = _distort_normalized(x, y, dist)
    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy
    return u, v


class DepthRgbMapper:
    """Map depth pixels into the RGB camera frame."""

    def __init__(self, calibration: CalibrationSet, profile: Optional[AlignProfile] = None):
        """Initialize a depth/RGB mapper.

        Parameters
        ----------
        calibration : CalibrationSet
            Calibration parameters used for projection.
        profile : AlignProfile, optional
            Resolution-specific alignment profile.
        """
        self.calibration = calibration
        self.profile = profile

    @classmethod
    def from_hardcoded(
        cls,
        color_size: Tuple[int, int],
        depth_size: Tuple[int, int],
        align_type_preference: Sequence[int] = (1, 2),
    ) -> "DepthRgbMapper":
        """Create a mapper from the built-in calibration tables.

        Parameters
        ----------
        color_size : tuple[int, int]
            RGB image size as ``(width, height)``.
        depth_size : tuple[int, int]
            Depth image size as ``(width, height)``.
        align_type_preference : sequence[int], optional
            Preferred alignment profile types, checked in order.

        Returns
        -------
        DepthRgbMapper
            Mapper configured for the requested resolution pair.

        Raises
        ------
        ValueError
            If no built-in calibration or profile matches the requested sizes.
        """
        calibrations = _HARDCODED_CALIBRATIONS
        profiles = _HARDCODED_PROFILES
        cw, ch = color_size
        dw, dh = depth_size

        matched_profile: Optional[AlignProfile] = None
        for pref in align_type_preference:
            for p in profiles:
                if (
                    p.align_type == pref
                    and p.color_width == cw
                    and p.color_height == ch
                    and p.depth_width == dw
                    and p.depth_height == dh
                ):
                    matched_profile = p
                    break
            if matched_profile is not None:
                break

        if matched_profile is not None:
            idx = matched_profile.param_index
            if idx < 0 or idx >= len(calibrations):
                raise ValueError(f"Profile paramIndex={idx} out of range for calibration list")
            return cls(calibrations[idx], matched_profile)

        for c in calibrations:
            if (
                c.rgb_intrinsic.width == cw
                and c.rgb_intrinsic.height == ch
                and c.depth_intrinsic.width == dw
                and c.depth_intrinsic.height == dh
            ):
                return cls(c, None)

        raise ValueError(
            "No matching hardcoded calibration/profile found for requested color/depth resolution pair"
        )

    def align_depth_to_color_with_correspondence(
        self,
        depth_image: np.ndarray,
        depth_unit_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project a raw depth image into the RGB camera plane.

        Parameters
        ----------
        depth_image : numpy.ndarray
            Two-dimensional depth image from the depth sensor.
        depth_unit_scale : float, optional
            Scale that converts depth image units to millimeters. Use ``1.0``
            when the image is already in millimeters.

        Returns
        -------
        aligned_depth_mm : numpy.ndarray
            RGB-aligned depth image in millimeters, with shape
            ``(color_height, color_width)`` and dtype ``float32``.
        src_u_map : numpy.ndarray
            Source depth-column map for each RGB pixel, with ``-1`` where no
            source pixel is valid.
        src_v_map : numpy.ndarray
            Source depth-row map for each RGB pixel, with ``-1`` where no
            source pixel is valid.

        Raises
        ------
        ValueError
            If ``depth_image`` is not a 2D image or its shape does not match
            the selected calibration.
        """
        c = self.calibration
        if depth_image.ndim != 2:
            raise ValueError("depth_image must be a 2D array")

        h, w = depth_image.shape
        if w != c.depth_intrinsic.width or h != c.depth_intrinsic.height:
            raise ValueError(
                f"depth_image shape {w}x{h} does not match calibration depth size "
                f"{c.depth_intrinsic.width}x{c.depth_intrinsic.height}"
            )

        v_grid, u_grid = np.indices((h, w), dtype=np.float64)
        z_mm = depth_image.astype(np.float64) * float(depth_unit_scale)
        valid = z_mm > 0.0
        if not np.any(valid):
            out_shape = (c.rgb_intrinsic.height, c.rgb_intrinsic.width)
            return (
                np.zeros(out_shape, dtype=np.float32),
                np.full(out_shape, -1, dtype=np.int32),
                np.full(out_shape, -1, dtype=np.int32),
            )

        u = u_grid[valid]
        v = v_grid[valid]
        z = z_mm[valid]

        x_d, y_d = _undistort_pixels_to_normalized(u, v, c.depth_intrinsic, c.depth_distortion)

        xyz_d = np.vstack((x_d * z, y_d * z, z))
        xyz_c = (c.rot @ xyz_d) + c.trans.reshape(3, 1)

        zc = xyz_c[2]
        positive = zc > 1e-6
        if not np.any(positive):
            out_shape = (c.rgb_intrinsic.height, c.rgb_intrinsic.width)
            return (
                np.zeros(out_shape, dtype=np.float32),
                np.full(out_shape, -1, dtype=np.int32),
                np.full(out_shape, -1, dtype=np.int32),
            )

        x_c = xyz_c[0, positive] / zc[positive]
        y_c = xyz_c[1, positive] / zc[positive]
        z_c_mm = zc[positive]
        u_src = u[positive].astype(np.int64)
        v_src = v[positive].astype(np.int64)

        u_c, v_c = _project_to_pixels(x_c, y_c, c.rgb_intrinsic, c.rgb_distortion)
        u_i = np.rint(u_c).astype(np.int64)
        v_i = np.rint(v_c).astype(np.int64)

        in_bounds = (
            (u_i >= 0) & (u_i < c.rgb_intrinsic.width) & (v_i >= 0) & (v_i < c.rgb_intrinsic.height)
        )
        if not np.any(in_bounds):
            out_shape = (c.rgb_intrinsic.height, c.rgb_intrinsic.width)
            return (
                np.zeros(out_shape, dtype=np.float32),
                np.full(out_shape, -1, dtype=np.int32),
                np.full(out_shape, -1, dtype=np.int32),
            )

        u_i = u_i[in_bounds]
        v_i = v_i[in_bounds]
        z_c_mm = z_c_mm[in_bounds]
        u_src = u_src[in_bounds]
        v_src = v_src[in_bounds]

        out_h = c.rgb_intrinsic.height
        out_w = c.rgb_intrinsic.width
        zbuf = np.full(out_h * out_w, np.inf, dtype=np.float64)
        src_u_flat = np.full(out_h * out_w, -1, dtype=np.int32)
        src_v_flat = np.full(out_h * out_w, -1, dtype=np.int32)
        lin = v_i * out_w + u_i

        # Keep the nearest depth sample per RGB pixel and remember source depth pixel.
        for idx in range(lin.size):
            li = int(lin[idx])
            z_val = float(z_c_mm[idx])
            if z_val < zbuf[li]:
                zbuf[li] = z_val
                src_u_flat[li] = int(u_src[idx])
                src_v_flat[li] = int(v_src[idx])

        aligned = zbuf.reshape(out_h, out_w)
        aligned[np.isinf(aligned)] = 0.0
        src_u_map = src_u_flat.reshape(out_h, out_w)
        src_v_map = src_v_flat.reshape(out_h, out_w)
        return aligned.astype(np.float32), src_u_map, src_v_map

    def align_depth_to_color(
        self,
        depth_image: np.ndarray,
        depth_unit_scale: float = 1.0,
    ) -> np.ndarray:
        """Project a raw depth image into the RGB camera plane.

        Parameters
        ----------
        depth_image : numpy.ndarray
            Two-dimensional depth image from the depth sensor.
        depth_unit_scale : float, optional
            Scale that converts depth image units to millimeters.

        Returns
        -------
        numpy.ndarray
            RGB-aligned depth image in millimeters, with shape
            ``(color_height, color_width)`` and dtype ``float32``.
        """
        aligned, _, _ = self.align_depth_to_color_with_correspondence(
            depth_image,
            depth_unit_scale=depth_unit_scale,
        )
        return aligned

    def get_depth_at_rgb(
        self,
        depth_image: np.ndarray,
        rgb_u: int,
        rgb_v: int,
        depth_unit_scale: float = 1.0,
        neighborhood: int = 1,
    ) -> Optional[float]:
        """Return the depth at an RGB pixel after depth-to-color projection.

        Parameters
        ----------
        depth_image : numpy.ndarray
            Two-dimensional depth image from the depth sensor.
        rgb_u, rgb_v : int
            RGB pixel coordinates.
        depth_unit_scale : float, optional
            Scale that converts depth image units to millimeters.
        neighborhood : int, optional
            Radius of the square neighborhood searched when the exact RGB pixel
            has no valid depth.

        Returns
        -------
        float or None
            Depth in millimeters, or ``None`` when no valid depth is found.
        """
        aligned = self.align_depth_to_color(depth_image, depth_unit_scale=depth_unit_scale)

        h, w = aligned.shape
        if rgb_u < 0 or rgb_u >= w or rgb_v < 0 or rgb_v >= h:
            return None

        d = float(aligned[rgb_v, rgb_u])
        if d > 0.0:
            return d

        if neighborhood <= 0:
            return None

        u0 = max(0, rgb_u - neighborhood)
        u1 = min(w - 1, rgb_u + neighborhood)
        v0 = max(0, rgb_v - neighborhood)
        v1 = min(h - 1, rgb_v + neighborhood)

        patch = aligned[v0 : v1 + 1, u0 : u1 + 1]
        nonzero = patch[patch > 0.0]
        if nonzero.size == 0:
            return None
        return float(np.min(nonzero))


def _find_depth_and_source(
    aligned_depth_mm: np.ndarray,
    src_u_map: np.ndarray,
    src_v_map: np.ndarray,
    rgb_u: int,
    rgb_v: int,
    neighborhood: int,
) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """Find depth and source depth pixel for an RGB coordinate.

    Parameters
    ----------
    aligned_depth_mm : numpy.ndarray
        RGB-aligned depth image in millimeters.
    src_u_map, src_v_map : numpy.ndarray
        Source depth pixel maps returned by
        :meth:`DepthRgbMapper.align_depth_to_color_with_correspondence`.
    rgb_u, rgb_v : int
        RGB pixel coordinates.
    neighborhood : int
        Radius of the square neighborhood searched when the exact RGB pixel has
        no valid depth.

    Returns
    -------
    depth_mm : float or None
        Depth in millimeters, or ``None`` when no valid depth is found.
    source_uv : tuple[int, int] or None
        Source depth pixel coordinates, or ``None`` when no valid depth is
        found.
    """
    h, w = aligned_depth_mm.shape
    if rgb_u < 0 or rgb_u >= w or rgb_v < 0 or rgb_v >= h:
        return None, None

    d = float(aligned_depth_mm[rgb_v, rgb_u])
    su = int(src_u_map[rgb_v, rgb_u])
    sv = int(src_v_map[rgb_v, rgb_u])
    if d > 0.0 and su >= 0 and sv >= 0:
        return d, (su, sv)

    if neighborhood <= 0:
        return None, None

    u0 = max(0, rgb_u - neighborhood)
    u1 = min(w - 1, rgb_u + neighborhood)
    v0 = max(0, rgb_v - neighborhood)
    v1 = min(h - 1, rgb_v + neighborhood)

    best_d = None
    best_uv = None
    for vv in range(v0, v1 + 1):
        for uu in range(u0, u1 + 1):
            d_val = float(aligned_depth_mm[vv, uu])
            if d_val <= 0.0:
                continue
            su = int(src_u_map[vv, uu])
            sv = int(src_v_map[vv, uu])
            if su < 0 or sv < 0:
                continue
            if best_d is None or d_val < best_d:
                best_d = d_val
                best_uv = (su, sv)

    return best_d, best_uv


def _depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    """Convert a depth image to a JET colormap visualization.

    Parameters
    ----------
    depth_image : numpy.ndarray
        Two-dimensional depth image.

    Returns
    -------
    numpy.ndarray
        Three-channel BGR colormap image.

    Raises
    ------
    ValueError
        If ``depth_image`` is not two-dimensional.
    """
    if depth_image.ndim != 2:
        raise ValueError("Depth image for visualization must be single-channel")

    depth_f = depth_image.astype(np.float32)
    valid = depth_f > 0
    vis = np.zeros_like(depth_f, dtype=np.uint8)
    if np.any(valid):
        vals = depth_f[valid]
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((depth_f - lo) * (255.0 / (hi - lo)), 0, 255)
        vis = scaled.astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def main_coords(rgb_path, depth_path, dict_objects):
    """Attach center-pixel coordinates and depth to detected objects.

    Parameters
    ----------
    rgb_path : str
        Path to the RGB image.
    depth_path : str
        Path to the depth image.
    dict_objects : dict
        Object dictionary keyed by mask ID. Each object must contain a
        ``"bbox"`` entry in ``[x_min, y_min, width, height]`` format.

    Returns
    -------
    dict
        Input object dictionary with ``"coord_center&depth"`` added to each
        object.
    """
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # si es PNG de depth visual

    color_size = (rgb.shape[1], rgb.shape[0])
    depth_size = (depth.shape[1], depth.shape[0])

    mapper = DepthRgbMapper.from_hardcoded(color_size=color_size, depth_size=depth_size)
    aligned_depth_mm, src_u_map, src_v_map = mapper.align_depth_to_color_with_correspondence(
        depth,
        depth_unit_scale=1,  # Scale from depth pixel units to milimeters
    )

    rgb_h, rgb_w = rgb.shape[:2]
    if aligned_depth_mm.shape[1] != rgb_w or aligned_depth_mm.shape[0] != rgb_h:
        aligned_depth_mm = cv2.resize(
            aligned_depth_mm, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST
        )
        src_u_map = cv2.resize(src_u_map, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        src_v_map = cv2.resize(src_v_map, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

    for mask_id in dict_objects.keys():
        coords = dict_objects[mask_id]["bbox"]
        ix, iy, delta_x, delta_y = coords
        fin_x = ix + delta_x
        fin_y = iy + delta_y
        cx = (ix + fin_x) // 2
        cy = (iy + fin_y) // 2

        depth_mm, src_uv = _find_depth_and_source(
            aligned_depth_mm,
            src_u_map,
            src_v_map,
            cx,
            cy,
            max(0, 1),
        )
        dict_objects[mask_id]["coord_center&depth"] = [cx, cy, depth_mm]

    return dict_objects
