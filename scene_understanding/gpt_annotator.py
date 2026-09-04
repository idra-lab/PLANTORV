import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from LLM.llm_base import BaseLLM
from LLM.llm_factory import create_llm
from scene_understanding.annotator import Annotator
from utility.utility import logger

"""GPT Model for tagging and description"""

DEFAULT_LLM_CONFIG_FILE = (
    Path(__file__).resolve().parent.parent / "LLM" / "conf" / "azure_gpt52.yaml"
)

# LABELED PROMPT
ANNOTATION_PROMPT = """You will receive:
1) Two images of the same scene. The first image shows the whole scene, and the second image is a cropped region of the image.
The second image shows the object and the first one gives the context of the image.
Your task:
- Describe the main object from the SECOND image, using the first one to consider the context of the workspace. Tell me the relative positions with respect the other objects that are seen in the first image, for example, specifying if they are on the left, on the rigth or next to another object.
- The tags should be ONLY one of the following ones: "Wide and large blue Lego block", "Small blue Lego block", "Yellow Lego block", "Wide red Lego Block with 4 studs", "Green Lego block", "2x2 Blue and red Lego block", "Tall red Lego block", " White and red box","Blue and white small box", "Big Black Bottle","Big White bottle", "Metallic Wrench", "Orange Lego block", "Orange small box", " White and green box", "Full robotic arm", "Partial robotic arm", "Unknown object".
- Do not change the tags neither use other tags that are not in the list. If you are not sure about the tag, use "Unknown object". For the detection, you can use the context of the whole image.
Return ONLY raw JSON.
Do not use markdown code fences.
Do not write ```json.
{
"tag": "string",
"description": "string"
}
"""

# UNLABELED TEXT PROMPT, kept for the runs that do not use a fixed tag list.
FREEFORM_ANNOTATION_PROMPT = """You will receive:
1) Two images of the same scene. The first image shows the whole scene, and the second image is a cropped region of the image.
The second image shows the object and the first one gives the context of the image.
Your task:
- Describe the main object from the SECOND image, using the first one to consider the context of the workspace. Tell me the relative positions with respect the other objects that are seen in the first image, for example, specifying if they are on the left, on the rigth or next to another object.
- The tagging should be ultra-specific. For example, instead of saying "lego block", say "furthest blue lego block with 4 studs ". Add the colour in the tag.
Return ONLY raw JSON.
Do not use markdown code fences.
Do not write ```json.
{
"tag": "string",
"description": "string"
}
"""


class GPTAnnotator(Annotator):
    def __init__(self, llm: BaseLLM, prompt: str = ANNOTATION_PROMPT) -> None:
        """
        Initialize the GPTAnnotator with an LLM backend.

        Parameters
        ----------
        llm : BaseLLM
            The backend used to annotate the objects. It must support images, which is the case
            for the Azure OpenAI, OpenAI, Anthropic, Gemini and GLM backends.
        prompt : str
            The instructions sent with every pair of images.

        Raises
        ------
        ValueError
            If the backend does not support images.
        """
        if not llm.SUPPORTS_IMAGES:
            raise ValueError(
                f"{type(llm).__name__} does not support images and cannot be used for annotation."
            )

        self.llm = llm
        self.prompt = prompt

    @classmethod
    def from_config(
        cls,
        llm_config_file: Union[str, Path] = DEFAULT_LLM_CONFIG_FILE,
        prompt: str = ANNOTATION_PROMPT,
        **overrides: Any,
    ) -> "GPTAnnotator":
        """
        Build a GPTAnnotator from an LLM YAML configuration file.

        The configuration file carries the model, the endpoint, the credentials and the request
        parameters (see ``LLM/conf``), and also selects the backend: pointing this at
        ``claude-46-opus.yaml`` instead of ``azure_gpt52.yaml`` swaps the provider.

        Parameters
        ----------
        llm_config_file : Union[str, Path]
            The path of the YAML configuration file. Defaults to ``LLM/conf/azure_gpt52.yaml``.
        prompt : str
            The instructions sent with every pair of images.
        **overrides : Any
            Forwarded to the backend's ``from_config``, e.g. ``params={"seed": 7}``.

        Returns
        -------
        GPTAnnotator
            An annotator configured from the file.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist or is not a YAML file.
        ValueError
            If the selected backend does not support images.
        """
        return cls(create_llm(llm_config_file, **overrides), prompt=prompt)

    def annotate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        segments: list[np.ndarray],
        bboxes: list[list[int]],
        masks: Optional[list[np.ndarray]] = None,
    ) -> dict:
        """
        Query the model for the tag and description of the objects passed as inputs.

        It applies the model over the original RGB image and the cropped images of the objects obtained with SAM, and then it returns a dictionary with the tagging and description of each object.

        Parameters
        ----------
        image : Union[str, Path, Image.Image, np.ndarray]
            The path of the original RGB image or the image itself.
        segments : list[np.ndarray]
            The cropped images of the objects obtained with SAM.
        bboxes : list[list[int]]
            The bounding boxes of the objects obtained by SAM. Each bounding box is represented as a list of 4 integers [x_min, y_min, width, height].
        masks : list[np.ndarray] or None
            Unused. Accepted so that this annotator and the mask-based ones share
            one call; see :meth:`scene_understanding.annotator.Annotator.annotate`.

        Returns
        -------
        dict
            A dictionary with the tagging and description of each object. Each key is the name of the object (for example, "mask_0") and each value is another dictionary with the following keys:
                - "tag": the tag of the object obtained by the model. String.
                - "description": the description of the object obtained by the model. String.
                - "bbox": the bounding box of the object obtained by SAM. List of 4 integers [x_min, y_min, width, height].

        Raises
        ------
        TypeError
            If the image is not a string, Path, PIL.Image.Image, or np.ndarray.
        """
        full_image = self.to_image(image)
        dict_outputs = {}

        for index, segment in enumerate(segments):
            crop = self.to_image(segment)

            success, raw = self.llm.query(
                f"{self.prompt}\nThe first image is the full scene, the second one is the cropped object.",
                images=[full_image, crop],
            )

            dict_outputs[f"mask_{index}"] = self._parse_answer(success, raw, index)
            dict_outputs[f"mask_{index}"]["bbox"] = bboxes[index]

        return dict_outputs

    @staticmethod
    def _parse_answer(success: bool, raw: str, index: int) -> dict:
        """
        Turn a model answer into the annotation dictionary of one object.

        Parameters
        ----------
        success : bool
            Whether the query reached the model.
        raw : str
            The raw answer of the model.
        index : int
            The index of the object, used for logging.

        Returns
        -------
        dict
            The parsed annotation, or a placeholder when the answer is missing or malformed.
        """
        unknown = {"tag": "unknown", "description": "unknown", "full_object": False}

        if not success or not raw:
            logger.error(f"No response for mask_{index}. Setting default values.")
            return dict(unknown)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON for mask_{index}: {raw}")
            return dict(unknown)

        if not isinstance(parsed, dict):
            logger.error(f"Unexpected JSON payload for mask_{index}: {raw}")
            return dict(unknown)

        return parsed

    def close(self) -> None:
        """Release the backend connection."""
        self.llm.close()
