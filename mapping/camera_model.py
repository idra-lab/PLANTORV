from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

"""Depth Estimation"""


@dataclass(frozen=True)
class Intrinsics:
    cx: float
    cy: float
    fx: float
    fy: float
    width: int
    height: int


@dataclass(frozen=True)
class Distortion:
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
    depth_distortion: Distortion
    depth_intrinsic: Intrinsics
    rgb_distortion: Distortion
    rgb_intrinsic: Intrinsics
    rot: np.ndarray  # 3x3
    trans: np.ndarray  # 3,


@dataclass(frozen=True)
class AlignProfile:
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

_DEPTH_DISTORTION = Distortion(
    k1=20.449,
    k2=9.65474,
    k3=0.311488,
    k4=20.7575,
    k5=16.556,
    k6=2.11015,
    p1=5.12616e-05,
    p2=-8.77438e-06,
)

_RGB_DISTORTION = Distortion(
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
        depth_distortion=_DEPTH_DISTORTION,
        depth_intrinsic=Intrinsics(
            cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024
        ),
        rgb_distortion=_RGB_DISTORTION,
        rgb_intrinsic=Intrinsics(
            cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DISTORTION,
        depth_intrinsic=Intrinsics(
            cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024
        ),
        rgb_distortion=_RGB_DISTORTION,
        rgb_intrinsic=Intrinsics(
            cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DISTORTION,
        depth_intrinsic=Intrinsics(
            cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576
        ),
        rgb_distortion=_RGB_DISTORTION,
        rgb_intrinsic=Intrinsics(
            cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360
        ),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DISTORTION,
        depth_intrinsic=Intrinsics(
            cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576
        ),
        rgb_distortion=_RGB_DISTORTION,
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
    """
    Apply distortion to normalized coordinates (x, y) using the provided distortion parameters.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates in normalized space.
    y : np.ndarray
        The y-coordinates in normalized space.
    d : Distortion
        The distortion parameters to apply.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The distorted x and y coordinates as a tuple of numpy arrays.
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
    """
    Convert pixel coordinates (u, v) to normalized coordinates (x, y) by applying the inverse of the distortion model.

    Parameters
    ----------
    u : np.ndarray
        The x-coordinates in pixel space.
    v : np.ndarray
        The y-coordinates in pixel space.
    intr : Intrinsics
        The camera intrinsic parameters.
    dist : Distortion
        The distortion parameters to apply.
    iters : int, optional
        The number of iterations for the fixed-point refinement (default is 8).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The undistorted x and y coordinates as a tuple of numpy arrays.
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
    """
    Project normalized coordinates (x, y) to pixel coordinates (u, v) using the provided camera intrinsics and distortion parameters.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates in normalized space.
    y : np.ndarray
        The y-coordinates in normalized space.
    intr : Intrinsics
        The camera intrinsic parameters.
    dist : Distortion
        The distortion parameters to apply.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The projected x and y coordinates as a tuple of numpy arrays.
    """
    xd, yd = _distort_normalized(x, y, dist)
    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy
    return u, v


def scale_intrinsics(intrinsic: Intrinsics, width: int, height: int) -> Intrinsics:
    """Return intrinsics of the same sensor sampled onto a frame of another size.

    The focal length in pixels scales with the sampling density, and the
    principal point follows the convention OpenCV resizes with: the centre of a
    destination pixel sits at ``(x + 0.5) / scale - 0.5`` of the source, so the
    principal point of the scaled frame is ``(c + 0.5) * scale - 0.5``. The
    distortion coefficients act on normalised coordinates and do not scale.

    Parameters
    ----------
    intrinsic : Intrinsics
        Intrinsics recorded at their own resolution.
    width : int
        Width of the frame the intrinsics are wanted for.
    height : int
        Height of that frame.

    Returns
    -------
    Intrinsics
        The same optics expressed in the pixels of the requested frame size.

    Raises
    ------
    ValueError
        If the requested size is not a uniform scaling of the recorded one,
        which would mean a different field of view rather than a resampling.
    """
    scale_x = width / intrinsic.width
    scale_y = height / intrinsic.height
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError(
            f"{width}x{height} is not a uniform scaling of the calibrated "
            f"{intrinsic.width}x{intrinsic.height}: the aspect ratio differs"
        )

    return Intrinsics(
        cx=(intrinsic.cx + 0.5) * scale_x - 0.5,
        cy=(intrinsic.cy + 0.5) * scale_y - 0.5,
        fx=intrinsic.fx * scale_x,
        fy=intrinsic.fy * scale_y,
        width=width,
        height=height,
    )


def rgb_calibration_for_size(width: int, height: int) -> Optional[Tuple[Intrinsics, Distortion]]:
    """Return the colour-sensor calibration of a frame of that size.

    The colour intrinsics are recorded at 640x360 and 640x480 only, but the
    camera streams colour at larger sizes of the same two aspect ratios, and
    `RGBDMapper` itself aligns depth into the recorded size and leaves the
    result to be resampled up to the frame. A frame whose size is a uniform
    scaling of a recorded one is therefore the same optics sampled more finely,
    and its intrinsics are the recorded ones scaled by :func:`scale_intrinsics`.
    A size of neither aspect ratio has no calibration.

    Parameters
    ----------
    width : int
        Width of the colour frame in pixels.
    height : int
        Height of the colour frame in pixels.

    Returns
    -------
    Optional[Tuple[Intrinsics, Distortion]]
        The colour intrinsics for that frame size and the distortion
        coefficients, which are independent of the size, or None when no
        calibration shares the frame's aspect ratio.
    """
    for calibration in _HARDCODED_CALIBRATIONS:
        intrinsic = calibration.rgb_intrinsic
        if intrinsic.width == width and intrinsic.height == height:
            return intrinsic, calibration.rgb_distortion

    for calibration in _HARDCODED_CALIBRATIONS:
        intrinsic = calibration.rgb_intrinsic
        if width * intrinsic.height == height * intrinsic.width:
            return scale_intrinsics(intrinsic, width, height), calibration.rgb_distortion

    return None
