from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from mapping.camera_model import (
    _HARDCODED_CALIBRATIONS,
    _HARDCODED_PROFILES,
    AlignProfile,
    CalibrationSet,
    Distortion,
    Intrinsics,
    _project_to_pixels,
    _undistort_pixels_to_normalized,
    rgb_calibration_for_size,
)
from utility.utility import logger

# How `attach_object_depths` turns the aligned depth image into one depth per object.
# `bbox-center` samples the centre of the bounding box and is the historical behaviour,
# so it stays the default and keeps earlier runs reproducible; `mask-median` takes the
# median of the depths under the object's segmentation mask.
DEPTH_ASSOCIATIONS = ("bbox-center", "mask-median")
DEFAULT_DEPTH_ASSOCIATION = "bbox-center"

# How far the depth of a candidate reference pixel may sit from the depth reported for the
# object and still be considered the same measurement, in millimetres. Under
# `mask-median` the reported depth is a median, which is not necessarily a depth any
# single pixel carries, so the pixel the 3D point is back-projected from is chosen among
# those that agree with it to within this tolerance. 0.1 mm is the 1e-4 m the depth of two
# pixels has to differ by before the difference matters to anything downstream.
DEPTH_MATCH_TOLERANCE_MM = 0.1


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


def attach_object_depths(
    dict_objects: dict,
    aligned_depth_mm: np.ndarray,
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    association: str = DEFAULT_DEPTH_ASSOCIATION,
    neighborhood: int = 1,
    rgb_calibration: Optional[Tuple[Intrinsics, Distortion]] = None,
) -> dict:
    """Attach RGB centre coordinates and aligned depth to detected objects.

    Two ways of turning the aligned depth image into one number per object are
    available, selected by ``association``:

    ``"bbox-center"``
        The depth is sampled at the centre of the bounding box, falling back to
        the smallest valid depth in a ``neighborhood``-radius square when that
        pixel carries no measurement. This is the original behaviour and the
        default, so existing results stay reproducible.
    ``"mask-median"``
        The depth is the median of the valid depth pixels covered by the
        object's segmentation mask. Zero, negative, NaN and infinite depths are
        excluded. When a mask covers no valid depth at all the object falls back
        to ``"bbox-center"``.

    Every object gets five keys:

    - ``"coord_center&depth"``, kept for backwards compatibility, as
      ``[cx, cy, depth_mm]``;
    - ``"object_depth_mm"``, the same depth under a name that does not claim the
      value was read at ``(cx, cy)``;
    - ``"depth_association"``, the strategy that actually produced the value,
      which is ``"bbox-center"`` for an object that fell back;
    - ``"object_point_camera_m"``, the object as ``[x, y, z]`` metres in the
      colour camera's optical frame (x right, y down, z along the optical axis),
      or None when the depth is missing or the frame size has no colour
      calibration;
    - ``"object_point_pixel"``, the ``[u, v]`` the 3D point was back-projected
      through, which says where in the image the point belongs.

    Under ``"mask-median"`` ``(cx, cy)`` is only the object's image-space
    reference centre: the depth is estimated over the whole mask rather than
    sampled at that pixel. The 3D point does not use that centre; it is
    back-projected through the pixel :func:`_mask_median_reference_pixel`
    chooses, which is the mask's centroid when the centroid's own depth matches
    the median and the nearest matching pixel to it otherwise.

    Parameters
    ----------
    dict_objects : dict
        One entry per object, keyed ``"mask_0"``, ``"mask_1"``, ..., each holding
        at least a ``"bbox"`` of ``[x_min, y_min, width, height]``.
    aligned_depth_mm : np.ndarray
        HxW depth image in millimetres, already registered to the RGB frame.
    masks : Sequence[np.ndarray] or None
        One binary mask per object, in the same order as the objects, as left in
        ``SegmentationModel.last_masks`` by ``individual_mask``. Required by
        ``"mask-median"`` and ignored by ``"bbox-center"``.
    association : str
        ``"bbox-center"`` (default) or ``"mask-median"``.
    neighborhood : int
        Radius of the square searched around the centre pixel when it holds no
        valid depth. ``0`` disables the search.
    rgb_calibration : Tuple[Intrinsics, Distortion] or None
        Colour intrinsics and distortion the 3D points are back-projected with.
        By default they are looked up from the size of ``aligned_depth_mm``,
        which is the size of the RGB frame it is registered to; a frame size the
        colour sensor was never calibrated at leaves
        ``"object_point_camera_m"`` at None rather than assuming intrinsics.

    Returns
    -------
    dict
        The same dictionary, with the five depth keys added to every object.

    Raises
    ------
    ValueError
        If ``association`` is unknown, if ``neighborhood`` is negative, if
        ``aligned_depth_mm`` is not two-dimensional, or if ``"mask-median"`` is
        asked for without masks matching the objects in number and shape.
    """
    if aligned_depth_mm.ndim != 2:
        raise ValueError("aligned_depth_mm must be a two-dimensional array")
    if neighborhood < 0:
        raise ValueError("neighborhood cannot be negative")
    if association not in DEPTH_ASSOCIATIONS:
        raise ValueError(
            f"unknown depth association {association!r}; expected one of "
            f"{', '.join(DEPTH_ASSOCIATIONS)}"
        )

    object_masks: Sequence[np.ndarray] = ()
    if association == "mask-median":
        if masks is None:
            raise ValueError("association 'mask-median' requires the per-object masks")
        if len(masks) != len(dict_objects):
            raise ValueError(
                f"got {len(masks)} masks for {len(dict_objects)} objects; the masks must be "
                "the ones `individual_mask` left in `last_masks`, in the same order"
            )
        object_masks = masks

    # The depth image is registered to the RGB frame, so its shape is the size of the
    # frame the pixels are to be back-projected through. A resolution the colour sensor
    # was never calibrated at leaves the 3D point unset rather than guessing intrinsics.
    height, width = aligned_depth_mm.shape
    calibration = (
        rgb_calibration if rgb_calibration is not None else rgb_calibration_for_size(width, height)
    )
    if calibration is None:
        logger.debug(
            f"no colour calibration for a {width}x{height} frame; "
            "the objects get no 'object_point_camera_m'"
        )

    for position, mask_id in enumerate(dict_objects.keys()):
        coords = dict_objects[mask_id]["bbox"]
        ix, iy, delta_x, delta_y = coords
        cx = (ix + ix + delta_x) // 2
        cy = (iy + iy + delta_y) // 2

        depth_mm: Optional[float] = None
        used = "bbox-center"
        # The pixel the 3D point is back-projected through. It is the centre of the
        # bounding box unless the mask median moves it; see `_mask_median_reference_pixel`.
        reference_pixel: Tuple[int, int] = (int(cx), int(cy))
        if object_masks:
            mask = object_masks[_mask_index(mask_id, position, len(object_masks))]
            depth_mm = _median_depth_over_mask(aligned_depth_mm, mask, mask_id)
            if depth_mm is None:
                logger.debug(
                    f"{mask_id}: no valid depth under the mask, falling back to bbox-center"
                )
            else:
                used = "mask-median"
                reference_pixel = _mask_median_reference_pixel(
                    aligned_depth_mm,
                    mask,
                    depth_mm,
                )

        if depth_mm is None:
            depth_mm = _find_depth_at_rgb(
                aligned_depth_mm,
                cx,
                cy,
                neighborhood,
            )

        point_camera_m: Optional[list] = None
        if depth_mm is not None and calibration is not None:
            point_camera_m = backproject_pixel_to_camera_m(
                reference_pixel[0],
                reference_pixel[1],
                depth_mm,
                calibration[0],
                calibration[1],
            )

        dict_objects[mask_id]["coord_center&depth"] = [cx, cy, depth_mm]
        dict_objects[mask_id]["object_depth_mm"] = depth_mm
        dict_objects[mask_id]["depth_association"] = used
        dict_objects[mask_id]["object_point_camera_m"] = point_camera_m
        dict_objects[mask_id]["object_point_pixel"] = [reference_pixel[0], reference_pixel[1]]

    return dict_objects


def backproject_pixel_to_camera_m(
    u: int,
    v: int,
    depth_mm: float,
    intrinsic: Intrinsics,
    distortion: Distortion,
) -> list:
    """Back-project one RGB pixel and its depth into the colour camera frame.

    The pixel is undistorted to normalised image coordinates and scaled by the
    axial depth, so the result is the 3D point of the colour camera's optical
    frame: x to the right of the image, y down it, z along the optical axis.

    Parameters
    ----------
    u : int
        Column of the pixel in the RGB frame.
    v : int
        Row of the pixel in the RGB frame.
    depth_mm : float
        Axial depth at that pixel, in millimetres.
    intrinsic : Intrinsics
        Colour intrinsics of the frame the pixel belongs to.
    distortion : Distortion
        Colour distortion coefficients of the same frame.

    Returns
    -------
    list
        ``[x, y, z]`` in metres, in the colour camera frame.
    """
    x_n, y_n = _undistort_pixels_to_normalized(
        np.asarray([float(u)], dtype=np.float64),
        np.asarray([float(v)], dtype=np.float64),
        intrinsic,
        distortion,
    )
    z_m = float(depth_mm) / 1000.0
    return [float(x_n[0]) * z_m, float(y_n[0]) * z_m, z_m]


def _mask_median_reference_pixel(
    aligned_depth_mm: np.ndarray,
    mask: np.ndarray,
    median_mm: float,
    tolerance_mm: float = DEPTH_MATCH_TOLERANCE_MM,
) -> Tuple[int, int]:
    """Return the pixel a mask-median depth is back-projected through.

    The median is a property of the whole mask, not of any one pixel, so pairing
    it with the centroid of the mask would describe a 3D point the scene does not
    contain whenever the centroid lies at a different distance -- a mask spanning
    a depth discontinuity, or one whose centroid falls in a hole. The centroid is
    therefore kept only when its own depth agrees with the median to within
    ``tolerance_mm``; otherwise the nearest pixel to the centroid that does agree
    is taken instead.

    When no pixel of the mask is within the tolerance, which the median of an even
    number of depths allows, the pixels whose depth is closest to the median are
    used as the candidates rather than giving up on a reference pixel.

    Parameters
    ----------
    aligned_depth_mm : np.ndarray
        HxW depth image in millimetres, registered to the RGB frame.
    mask : np.ndarray
        HxW binary mask of the object, already checked against the depth image.
    median_mm : float
        The depth reported for the object, as returned by
        :func:`_median_depth_over_mask`.
    tolerance_mm : float
        How far a candidate's depth may sit from ``median_mm``.

    Returns
    -------
    Tuple[int, int]
        The ``(u, v)`` pixel to back-project, guaranteed to be a pixel of the mask
        carrying a valid depth.

    Raises
    ------
    ValueError
        If the mask covers no valid depth, which means ``median_mm`` did not come
        from this mask.
    """
    mask_array = np.asarray(mask).astype(bool)
    valid = mask_array & np.isfinite(aligned_depth_mm) & (aligned_depth_mm > 0.0)
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        raise ValueError("the mask covers no valid depth, so it has no reference pixel")

    centroid_u = float(np.mean(columns))
    centroid_v = float(np.mean(rows))

    # The centroid of a concave or split mask can fall outside it, so it is only
    # usable when it is one of the mask's own valid pixels.
    rounded_u = int(round(centroid_u))
    rounded_v = int(round(centroid_v))
    height, width = aligned_depth_mm.shape
    if (
        0 <= rounded_u < width
        and 0 <= rounded_v < height
        and valid[rounded_v, rounded_u]
        and abs(float(aligned_depth_mm[rounded_v, rounded_u]) - median_mm) <= tolerance_mm
    ):
        return rounded_u, rounded_v

    difference = np.abs(aligned_depth_mm[valid].astype(np.float64) - median_mm)
    candidates = np.nonzero(difference <= tolerance_mm)[0]
    if candidates.size == 0:
        candidates = np.nonzero(difference == difference.min())[0]

    distance_squared = (columns[candidates] - centroid_u) ** 2 + (
        rows[candidates] - centroid_v
    ) ** 2
    nearest = candidates[int(np.argmin(distance_squared))]
    return int(columns[nearest]), int(rows[nearest])


def _mask_index(mask_id: object, position: int, mask_count: int) -> int:
    """Return the index in the mask list of the object stored under ``mask_id``.

    The annotators key their objects ``"mask_0"``, ``"mask_1"``, ... in the order
    of the masks, so the number in the key is the index. Anything else falls back
    to the position of the object in the dictionary.

    Parameters
    ----------
    mask_id : object
        The key the object is stored under.
    position : int
        The position of the object in the dictionary.
    mask_count : int
        How many masks were passed in.

    Returns
    -------
    int
        The index of the mask belonging to this object.

    Raises
    ------
    ValueError
        If the index falls outside the mask list.
    """
    index = position
    if isinstance(mask_id, str) and mask_id.startswith("mask_"):
        suffix = mask_id[len("mask_") :]
        if suffix.isdigit():
            index = int(suffix)

    if index < 0 or index >= mask_count:
        raise ValueError(f"no mask for object {mask_id!r}: only {mask_count} masks were given")
    return index


def _median_depth_over_mask(
    aligned_depth_mm: np.ndarray,
    mask: np.ndarray,
    mask_id: object,
) -> Optional[float]:
    """Return the median of the valid depths covered by one object mask.

    The depth image is registered to the RGB frame the mask was found in, so the
    two are compared pixel by pixel and a mask of a different size is an error
    rather than something to resize.

    Parameters
    ----------
    aligned_depth_mm : np.ndarray
        HxW depth image in millimetres, registered to the RGB frame.
    mask : np.ndarray
        HxW binary mask of the object.
    mask_id : object
        The key of the object, used in the error message.

    Returns
    -------
    Optional[float]
        The median of the depths that are inside the mask, finite and positive,
        or None when the mask covers none.

    Raises
    ------
    ValueError
        If the mask is not two-dimensional or does not have the shape of the
        depth image.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or mask_array.shape != aligned_depth_mm.shape:
        raise ValueError(
            f"mask of object {mask_id!r} has shape {mask_array.shape}, expected the shape of "
            f"the aligned depth image {aligned_depth_mm.shape}; masks and depth must come from "
            "the same RGB frame"
        )

    selected = aligned_depth_mm[mask_array.astype(bool)]
    valid = selected[np.isfinite(selected) & (selected > 0.0)]
    return None if valid.size == 0 else float(np.median(valid))


def _find_depth_at_rgb(
    aligned_depth_mm: np.ndarray,
    rgb_u: int,
    rgb_v: int,
    neighborhood: int,
) -> Optional[float]:
    height, width = aligned_depth_mm.shape
    if rgb_u < 0 or rgb_u >= width or rgb_v < 0 or rgb_v >= height:
        return None

    depth = float(aligned_depth_mm[rgb_v, rgb_u])
    if np.isfinite(depth) and depth > 0.0:
        return depth
    if neighborhood <= 0:
        return None

    u0 = max(0, rgb_u - neighborhood)
    u1 = min(width - 1, rgb_u + neighborhood)
    v0 = max(0, rgb_v - neighborhood)
    v1 = min(height - 1, rgb_v + neighborhood)
    patch = aligned_depth_mm[v0 : v1 + 1, u0 : u1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return None if valid.size == 0 else float(np.min(valid))


def main_coords(
    image_path: Union[str, Path, Image.Image, np.ndarray],
    depth_path: Union[str, Path, Image.Image, np.ndarray],
    dict_objects: dict,
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    association: str = DEFAULT_DEPTH_ASSOCIATION,
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

    Returns
    -------
    dict
        The input dictionary of objects, updated for each object with "coord_center&depth"
        (a list [center_x, center_y, depth_mm]), "object_depth_mm", "depth_association",
        "object_point_camera_m" (a list [x, y, z] in metres in the colour camera frame) and
        "object_point_pixel" (the pixel that point was back-projected through).
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

    # Local import avoids a module cycle: providers reuse RGBDMapper itself.
    from mapping.depth_provider import SensorDepthProvider

    provider = SensorDepthProvider.from_frames(rgb, depth, depth_unit_scale=1.0)
    result = provider.estimate(rgb, depth)
    return attach_object_depths(
        dict_objects,
        result.depth_mm,
        masks=masks,
        association=association,
    )
