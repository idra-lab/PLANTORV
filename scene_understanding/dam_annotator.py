"""Describe Anything (DAM) annotator.

The counterpart of :mod:`scene_understanding.gpt_annotator`: it turns the objects
found by a segmentation model into a ``{"mask_i": {"tag", "description", "bbox"}}``
dictionary, so the two are interchangeable in the pipeline scripts and both feed
``mapping.rgbd_mapper`` unchanged.

The two differ in what they look at. The GPT annotator sends the whole scene plus
a *cropped* object to a remote vision model. DAM runs locally and describes a
*masked region* of the full image, so it needs the per-object binary masks rather
than the crops. Those masks are the ones a segmentation model leaves in
``SegmentationModel.last_masks`` after :meth:`individual_mask`.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

from segmentation.segmentation import binary_mask_to_pil
from utility.utility import logger

# Where `scripts/install_models.py` puts the DAM checkpoint. A Hugging Face
# repository id such as "nvidia/DAM-3B" works here too: DAM's loader accepts
# either, and falls back to downloading into the Hugging Face cache.
DEFAULT_DAM_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dam" / "DAM-3B"

# DAM has no notion of a tag, so the tag is asked for in the prompt: the answer
# opens with the object, then a comma, then the description. `_parse_answer`
# splits the reply back apart on that comma.
DAM_QUERY = (
    "<image>\nDescribe the masked region in detail. The first two words must define "
    "the object. Then use a comma and give the rest of the description."
)

# DAM builds its visual prompt from two crops, named as "<full>+<focal>". This is
# the mode the DAM-3B release is evaluated with.
DEFAULT_PROMPT_MODE = "full+focal_crop"
DEFAULT_CONV_MODE = "v1"


class DAMAnnotator:
    """Tag and describe segmented objects with a local Describe Anything model."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_DAM_MODEL_PATH,
        *,
        query: str = DAM_QUERY,
        device: str = "cuda",
        conv_mode: str = DEFAULT_CONV_MODE,
        prompt_mode: str = DEFAULT_PROMPT_MODE,
        temperature: float = 0.6,
        top_p: float = 0.5,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        model: Any | None = None,
    ) -> None:
        """
        Load a Describe Anything model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Directory holding the checkpoint, or a Hugging Face repository id.
            Defaults to the location ``scripts/install_models.py`` downloads
            ``dam_3b`` into.
        query : str
            The instructions sent with every masked region. It must contain the
            ``<image>`` token DAM substitutes the visual features into, and it is
            expected to ask for ``"<object>, <description>"`` so the answer can be
            split into a tag and a description. See :data:`DAM_QUERY`.
        device : str
            Device the model is moved to. DAM's generation path is CUDA-only, so
            ``"cpu"`` loads but cannot generate.
        conv_mode : str
            Conversation template DAM wraps the query in.
        prompt_mode : str
            How DAM crops the image around the mask, as ``"<full>+<focal>"``.
        temperature, top_p, num_beams, max_new_tokens
            Generation parameters, applied to every call of :meth:`main_dam`.
        model : Any or None
            An already-built model, used instead of loading one. Intended for
            tests; when given, ``model_path`` is ignored.

        Raises
        ------
        ImportError
            If the ``dam`` package is not installed.
        FileNotFoundError
            If ``model_path`` looks like a path (rather than a repository id) and
            does not exist.
        """
        self.model_path = str(model_path)
        self.query = query
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        if model is None:
            try:
                # Imported here rather than at module scope so importing this
                # module costs nothing until a DAM annotator is actually built.
                # `dam` is in requirements.txt, but it installs from git and
                # pulls a large dependency tree, so an environment that only
                # runs the GPT annotator is expected not to have it; the ignore
                # keeps that expected absence from being reported.
                from dam.describe_anything_model import (  # pyright: ignore[reportMissingImports]
                    DescribeAnythingModel,
                )
            except ImportError as exc:
                raise ImportError(
                    "Describe Anything support requires the 'dam' package; see the "
                    "Describe Anything section in README.md"
                ) from exc

            # A repository id has no separator and is never an existing path, so
            # only a path-looking argument is checked. This keeps the failure at
            # construction time rather than inside DAM's loader.
            candidate = Path(self.model_path)
            if candidate.parent != Path(".") and not candidate.exists():
                raise FileNotFoundError(
                    f"DAM checkpoint not found at {candidate}. Download it with "
                    "`python3 scripts/install_models.py dam_3b`."
                )

            model = DescribeAnythingModel(
                model_path=self.model_path,
                conv_mode=conv_mode,
                prompt_mode=prompt_mode,
            )

        self._model: Any = model
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)

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

    def main_dam(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        masks: list[np.ndarray],
        bboxes: list[list[int]],
    ) -> dict:
        """
        Query DAM for the tag and description of every masked object.

        Unlike :meth:`scene_understanding.gpt_annotator.GPTAnnotator.main_gpt`,
        which is given the cropped objects, this takes the binary masks: DAM
        describes a region of the full image. The masks are the ones the
        segmentation model left in ``last_masks``, so a caller does::

            rgb_masks, bboxes = sam.individual_mask(image, mask_bin, masked_rgb, idx)
            image_dict = dam.main_dam(image, sam.last_masks, bboxes)

        Parameters
        ----------
        image : Union[str, Path, Image.Image, np.ndarray]
            The original RGB image, or a path to it.
        masks : list[np.ndarray]
            One binary mask per object, shape ``(H, W)`` or ``(H, W, 1)``, with
            values ``0``/``1`` or ``0``/``255``.
        bboxes : list[list[int]]
            One bounding box per object, as ``[x_min, y_min, width, height]``.

        Returns
        -------
        dict
            One entry per object, keyed ``"mask_0"``, ``"mask_1"``, ... Each value
            holds ``"tag"``, ``"description"`` and ``"bbox"``, which is the shape
            ``mapping.rgbd_mapper.main_coords`` expects.

        Raises
        ------
        ValueError
            If ``masks`` and ``bboxes`` have different lengths.
        """
        if len(masks) != len(bboxes):
            raise ValueError(
                f"Got {len(masks)} masks but {len(bboxes)} bounding boxes; "
                "they must describe the same objects."
            )

        full_image = self.to_image(image).convert("RGB")
        dict_outputs: dict[str, dict] = {}

        for index, mask in enumerate(masks):
            mask_pil = binary_mask_to_pil(mask)

            try:
                raw = self._model.get_description(
                    full_image,
                    mask_pil,
                    self.query,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as error:
                # One object failing should not lose the descriptions of the rest,
                # which on a 151-image run would mean redoing the whole thing.
                logger.error(f"DAM failed on mask_{index}: {error}")
                raw = ""

            dict_outputs[f"mask_{index}"] = self._parse_answer(raw, index)
            dict_outputs[f"mask_{index}"]["bbox"] = bboxes[index]

        return dict_outputs

    @staticmethod
    def _parse_answer(raw: str, index: int) -> dict:
        """
        Split a DAM answer into a tag and a description.

        The prompt asks for ``"<object>, <rest of the description>"``, so the tag
        is what precedes the first comma. An answer that does not follow the
        format still yields something usable: the opening two words become the
        tag and the whole answer the description.

        Parameters
        ----------
        raw : str
            The raw answer of the model.
        index : int
            The index of the object, used for logging.

        Returns
        -------
        dict
            ``{"tag": str, "description": str}``, both ``"unknown"`` when the
            answer is empty.
        """
        text = (raw or "").strip()
        if not text:
            logger.error(f"No description for mask_{index}. Setting default values.")
            return {"tag": "unknown", "description": "unknown"}

        tag, separator, description = text.partition(",")
        if not separator:
            logger.warning(
                f"DAM answer for mask_{index} has no comma to split on: {text!r}. "
                "Using the first two words as the tag."
            )
            return {"tag": " ".join(text.split()[:2]), "description": text}

        return {"tag": tag.strip(), "description": description.strip()}

    def close(self) -> None:
        """Release the model and the memory it holds on the GPU."""
        self._model = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
