from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from mapping.camera_model import (
    _HARDCODED_CALIBRATIONS,
    _HARDCODED_PROFILES,
    AlignProfile,
    CalibrationSet,
    _project_to_pixels,
    _undistort_pixels_to_normalized,
)


class RGBDMapper:
    """Depth<->RGB utility built from hardcoded Femto Mega calibration data."""

    def __init__(self, calibration: CalibrationSet, profile: Optional[AlignProfile] = None) -> None:
        """Initialize the RGBDMapper with calibration and optional alignment profile."""
        self.calibration = calibration
        self.profile = profile

    @classmethod
    def from_hardcoded(
        cls,
        color_size: Tuple[int, int],
        depth_size: Tuple[int, int],
        align_type_preference: Sequence[int] = (1, 2),
    ) -> "RGBDMapper":
        """Create an RGBDMapper from hardcoded calibration data.

        Parameters
        ----------
        color_size : Tuple[int, int]
            The (width, height) of the color image.
        depth_size : Tuple[int, int]
            The (width, height) of the depth image.
        align_type_preference : Sequence[int], optional
            Preferred alignment types to search for in the hardcoded profiles. Default is (1, 2).

        Returns
        -------
        RGBDMapper
            An instance of RGBDMapper initialized with the matching calibration and profile.
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
        """Project raw depth image into the RGB camera image plane.

        Parameters
        ----------
        depth_image : np.ndarray
            HxW depth array from depth sensor.
        depth_unit_scale : float, optional
            Converts depth_image units to millimeters (mm).
            Example: 1.0 if already in mm, 0.1 if each unit is 0.1 mm.

        Returns
        -------
        aligned_depth_mm : np.ndarray
            Hc x Wc float32 depth image in millimeters, aligned to RGB.
        src_u_map : np.ndarray
            Hc x Wc int32 map of source depth-u for each RGB pixel (-1 if invalid).
        src_v_map : np.ndarray
            Hc x Wc int32 map of source depth-v for each RGB pixel (-1 if invalid).
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
        """
        Project raw depth image into the RGB camera image plane.

        Parameters
        ----------
        depth_image : np.ndarray
            HxW depth array from depth sensor.
        depth_unit_scale : float, optional
            Converts depth_image units to millimeters (mm).
            Example: 1.0 if already in mm, 0.1 if each unit is 0.1 mm.

        Returns
        -------
        aligned : np.ndarray
            Hc x Wc float32 depth image in millimeters, aligned to RGB.
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
        """
        Return depth in mm at RGB pixel after D2C reprojection.

        If exact pixel has no value, searches a small square neighborhood.

        Parameters
        ----------
        depth_image : np.ndarray
            HxW depth array from depth sensor.
        rgb_u : int
            The x-coordinate (column) in the RGB image.
        rgb_v : int
            The y-coordinate (row) in the RGB image.
        depth_unit_scale : float, optional
            Converts depth_image units to millimeters (mm).
            Example: 1.0 if already in mm, 0.1 if each unit is 0.1 mm.
        neighborhood : int, optional
            The radius of the square neighborhood to search for a valid depth value if the exact pixel has no value. A value of 0 means no neighborhood search.

        Returns
        -------
        Optional[float]
            The depth in millimeters at the specified RGB pixel, or None if no valid depth is found within the specified neighborhood.
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
    """
    Find the depth in millimeters and the corresponding source depth pixel coordinates for a given RGB pixel.

    Parameters
    ----------
    aligned_depth_mm : np.ndarray
        Hc x Wc float32 depth image in millimeters, aligned to RGB.
    src_u_map : np.ndarray
        Hc x Wc int32 map of source depth-u for each RGB pixel (-1 if invalid).
    src_v_map : np.ndarray
        Hc x Wc int32 map of source depth-v for each RGB pixel (-1 if invalid).
    rgb_u : int
        The x-coordinate (column) in the RGB image.
    rgb_v : int
        The y-coordinate (row) in the RGB image.
    neighborhood : int
        The radius of the square neighborhood to search for a valid depth value if the exact pixel has no value. A value of 0 means no neighborhood search.

    Returns
    -------
    Tuple[Optional[float], Optional[Tuple[int, int]]]
        A tuple containing:
        - The depth in millimeters at the specified RGB pixel, or None if no valid depth is found within the specified neighborhood.
        - A tuple of (source_u, source_v) coordinates in the depth image corresponding to the found depth value, or None if no valid depth is found.
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
    """
    Convert a single-channel depth image to a color-mapped image for visualization.

    Parameters
    ----------
    depth_image : np.ndarray
        HxW single-channel depth image.

    Returns
    -------
    np.ndarray
        HxWx3 color-mapped image suitable for visualization.
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


def main_coords(rgb_path: str, depth_path: str, dict_objects: dict) -> dict:
    """
    Given the paths to an RGB image and a depth image, along with a dictionary of objects containing their bounding boxes, this function aligns the depth image to the RGB image and retrieves the depth information for each object's center pixel.

    Parameters
    ----------
    rgb_path : str
        The file path to the RGB image.
    depth_path : str
        The file path to the depth image.
    dict_objects : dict
        A dictionary where each key is an object identifier and each value is another dictionary containing at least a "bbox" key with the bounding box coordinates [x_min, y_min, width, height].

    Returns
    -------
    dict
        The input dictionary of objects, updated with an additional key "coord_center&depth" for each object, containing a list [center_x, center_y, depth_mm] representing the center pixel coordinates and the corresponding depth in millimeters.
    """
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # si es PNG de depth visual

    if rgb is None:
        raise FileNotFoundError(f"RGB image not found at path: {rgb_path}")
    if depth is None:
        raise FileNotFoundError(f"Depth image not found at path: {depth_path}")

    color_size = (rgb.shape[1], rgb.shape[0])
    depth_size = (depth.shape[1], depth.shape[0])

    mapper = RGBDMapper.from_hardcoded(color_size=color_size, depth_size=depth_size)
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
        # logger.debug(f"Object {mask_id}: depth={depth_mm} mm, src_uv={src_uv}")
        dict_objects[mask_id]["coord_center&depth"] = [cx, cy, depth_mm]

    return dict_objects
