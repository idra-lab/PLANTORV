"""Abstract base for the object annotators.

An annotator turns the objects a segmentation model found into text: one tag and
one description per object. The concrete implementations differ in where that
text comes from -- :class:`~scene_understanding.gpt_annotator.GPTAnnotator` asks
a remote vision LLM, :class:`~scene_understanding.dam_annotator.DAMAnnotator`
runs Describe Anything locally -- but they answer with the same dictionary, so
the pipeline scripts can swap one for the other without changing anything else.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


class Annotator(ABC):
    """Turn segmented objects into tags and descriptions."""

    @staticmethod
    def to_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Convert any supported image input into a PIL image.

        Parameters
        ----------
        image : Union[str, Path, Image.Image, np.ndarray]
            The path of an image, the image itself, or an array holding it.
            Floating point arrays are assumed to be in the [0, 1] range.

        Returns
        -------
        Image.Image
            The image as a PIL object.

        Raises
        ------
        FileNotFoundError
            If a path is given but no file exists there.
        TypeError
            If the input is of an unsupported type.
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            return Image.open(image_path)

        if isinstance(image, Image.Image):
            return image

        if isinstance(image, np.ndarray):
            array = image
            if np.issubdtype(array.dtype, np.floating):
                array = (array * 255).clip(0, 255)
            return Image.fromarray(array.astype(np.uint8))

        raise TypeError("image must be a str, Path, PIL.Image.Image, or np.ndarray")

    @abstractmethod
    def annotate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        segments: list[np.ndarray],
        bboxes: list[list[int]],
        masks: Optional[list[np.ndarray]] = None,
    ) -> dict:
        """
        Tag and describe every object found in an image.

        Both of the shapes a segmentation model produces are accepted, because
        the implementations need different ones: an annotator that works from a
        cropped object reads ``segments``, one that describes a region of the
        full image reads ``masks``. A caller that passes both can swap one
        annotator for the other without touching the call::

            rgb_masks, bboxes = sam.individual_mask(image, mask_bin, masked_rgb, idx)
            image_dict = annotator.annotate(image, rgb_masks, bboxes, masks=sam.last_masks)

        Parameters
        ----------
        image : Union[str, Path, Image.Image, np.ndarray]
            The original RGB image, or a path to it.
        segments : list[np.ndarray]
            The cropped objects, as returned by ``individual_mask``.
        bboxes : list[list[int]]
            One bounding box per object, as ``[x_min, y_min, width, height]``.
        masks : list[np.ndarray] or None
            One binary mask per object, in the same order as ``segments``. These
            are the masks a segmentation model leaves in
            ``SegmentationModel.last_masks``. An implementation that needs them
            raises when they are missing.

        Returns
        -------
        dict
            One entry per object, keyed ``"mask_0"``, ``"mask_1"``, ... Each
            value holds ``"tag"``, ``"description"`` and ``"bbox"``, which is the
            shape ``mapping.rgbd_mapper.main_coords`` expects.
        """

    @abstractmethod
    def close(self) -> None:
        """Release whatever the annotator holds.

        Abstract rather than a no-op default because what is held is worth being
        deliberate about: a network connection for the remote backends, and
        several gigabytes of VRAM for the local ones.
        """
