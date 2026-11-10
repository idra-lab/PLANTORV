"""Mask helpers shared by the segmentation backends and the annotators."""

import numpy as np
from PIL import Image


def binary_mask_to_pil(mask_bin: np.ndarray) -> Image.Image:
    """Convert a binary mask to a savable grayscale PIL image.

    Parameters
    ----------
    mask_bin : numpy.ndarray
        Binary mask with shape ``(H, W)`` or ``(H, W, 1)``. Values may be
        ``0`` and ``1`` or ``0`` and ``255``.

    Returns
    -------
    PIL.Image.Image
        Two-dimensional grayscale mask image with values ``0`` and ``255``.

    Raises
    ------
    ValueError
        If ``mask_bin`` is a three-dimensional array with more than one
        channel.
    """
    if mask_bin.ndim == 3:
        if mask_bin.shape[2] != 1:
            raise ValueError(f"Expected a single-channel mask, got shape {mask_bin.shape}")
        mask_bin = mask_bin[:, :, 0]

    mask_uint8 = (mask_bin > 0).astype(np.uint8) * 255
    return Image.fromarray(mask_uint8)
