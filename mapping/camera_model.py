from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass
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
        depth_intrinsic=Intrinsics(cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480),
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


def _distort_normalized(x: np.ndarray, y: np.ndarray, d: Distortion) -> Tuple[np.ndarray, np.ndarray]:
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
    xd, yd = _distort_normalized(x, y, dist)
    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy
    return u, v
