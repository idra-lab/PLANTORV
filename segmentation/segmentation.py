from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image


class SegmentationModel(ABC):
    def __init__(self, save_dir: Optional[Union[str, Path]]) -> None:
        """
        Abstract base class for segmentation models.

        Parameters
        ----------
        save_dir : str or Path or None
            Directory where the output images will be saved if not None.

        Attributes
        ----------
        last_masks : list[numpy.ndarray]
            Per-object binary masks kept by the most recent
            :meth:`individual_mask` call, in the same order as the crops it
            returned. That method returns crops and boxes, which is all a
            crop-based annotator needs, but an annotator that describes a
            masked region (see ``scene_understanding/dam_annotator.py``) needs
            the masks themselves. Empty until ``individual_mask`` has run.
        """
        self.last_masks: list[np.ndarray] = []

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    @abstractmethod
    def obtain_bg(
        self, image: Union[Image.Image, str, Path], **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method to perform segmentation on the input image.

        Parameters
        ----------
        image : (Union[Image.Image, str, Path])
            The input image to be segmented.
        **kwargs : Any
            Additional keyword arguments for segmentation.

        Returns
        -------
            Tuple[np.ndarray, np.ndarray]: A tuple containing the masked background and the segmented mask.
        """
        pass

    @abstractmethod
    def individual_mask(
        self, image: Union[Image.Image, str, Path], **kwargs: Any
    ) -> tuple[list[np.ndarray], list[list[int]]]:
        """
        Abstract method to generate and filter individual object masks inside the kept region.

        Parameters
        ----------
        image : (Union[Image.Image, str, Path])
            The input image to be segmented.
        **kwargs : Any
            Additional keyword arguments for segmentation.

        Returns
        -------
        tuple[list[np.ndarray], list[list[int]]]:
            A tuple containing the cropped RGB images for accepted object masks and their bounding boxes.
        """
        pass


def mask_to_pil(mask_bool: np.ndarray) -> Image.Image:
    """Convert a SAM boolean mask to a grayscale PIL image.

    Parameters
    ----------
    mask_bool : numpy.ndarray
        Two-dimensional boolean SAM mask with shape ``(H, W)``.

    Returns
    -------
    PIL.Image.Image
        Grayscale mask image with pixel values ``0`` and ``255``.
    """
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    return Image.fromarray(mask_uint8)


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
