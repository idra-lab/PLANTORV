import base64
import json
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from openai import AzureOpenAI
from PIL import Image

from utility.utility import logger

"""GPT Model for tagging and description"""


class GPTAnnotator:
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        deployment: str,
        subscription_key: str,
        api_version: str,
        max_completion_tokens: int = 16384,
    ) -> None:
        """
        Initialize the GPTAnnotator with Azure OpenAI client.

        Parameters
        ----------
        endpoint : str
            The Azure OpenAI endpoint URL.
        model_name : str
            The name of the GPT model to use.
        deployment : str
            The deployment name for the GPT model.
        subscription_key : str
            The subscription key for the Azure OpenAI service.
        api_version : str
            The API version for the Azure OpenAI service.
        max_completion_tokens : int
            The maximum number of tokens to generate in the completion. Default is 16384.
        """
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.deployment = deployment
        self.max_completion_tokens = max_completion_tokens

    def encode_image_data_url(self, image_path: Path) -> str:
        """
        Encode an image file as a base64 data URL.

        Parameters
        ----------
        image_path : Path
            The path to the image file to be encoded.

        Returns
        -------
        str
            A base64-encoded data URL representing the image.

        Raises
        ------
        FileNotFoundError
            If the specified image file does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def encode_image_data_from_array(self, image_array: np.ndarray) -> str:
        """
        Encode a NumPy array representing an image as a base64 data URL.

        Parameters
        ----------
        image_array : np.ndarray
            The NumPy array representing the image to be encoded.

        Returns
        -------
        str
            A base64-encoded data URL representing the image.
        """
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        mime_type = "image/png"
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def main_gpt(
        self,
        image_path: Union[str, Path, Image.Image, np.ndarray],
        segments: list[np.ndarray],
        bboxes: list[list[int]],
    ) -> dict:
        """
        Query GPT for the tag and description of the objects passed as inputs.

        It applies GPT over the original RGB image and the cropped images of the objects obtained with SAM, and then it returns a dictionary with the tagging and description of each object.

        Parameters
        ----------
        image : Union[str, Path, Image.Image, np.ndarray]
            The path of the original RGB image or the image itself.
        segments : list[np.ndarray]
            The cropped images of the objects obtained with SAM.
        bboxes : list[list[int]]
            The bounding boxes of the objects obtained by SAM. Each bounding box is represented as a list of 4 integers [x_min, y_min, width, height].

        Returns
        -------
        dict
            A dictionary with the tagging and description of each object. Each key is the name of the object (for example, "mask_0") and each value is another dictionary with the following keys:
                - "tag": the tag of the object obtained by GPT. String.
                - "description": the description of the object obtained by GPT. String.
                - "mask": the path of the mask obtained for the object. String.
                - "bbox": the bounding box of the object obtained by SAM. List of 4 integers [x_min, y_min, width, height].

        Raises
        ------
        TypeError
            If the image_path is not a string, Path, or PIL.Image.Image.
        """
        if isinstance(image_path, str) or isinstance(image_path, Path):
            if isinstance(image_path, str):
                image_path = Path(image_path)
            image_encoded = self.encode_image_data_url(image_path)
        elif isinstance(image_path, Image.Image):
            image_encoded = self.encode_image_data_from_array(np.array(image_path))
        elif isinstance(image_path, np.ndarray):
            image_encoded = self.encode_image_data_from_array(image_path)
        else:
            raise TypeError("image_path must be a str, Path, PIL.Image.Image, or np.ndarray")
        dict_outputs = {}

        # UNLABELED TEXT PROMPT
        # question_2 = """You will receive:
        # 1) Two images of the same scene. The first image shows the whole scene, and the second image is a cropped region of the image.
        # The second image shows the object and the first one gives the context of the image.
        # Your task:
        # - Describe the main object from the SECOND image, using the first one to consider the context of the workspace. Tell me the relative positions with respect the other objects that are seen in the first image, for example, specifying if they are on the left, on the rigth or next to another object.
        # - The tagging should be ultra-specific. For example, instead of saying "lego block", say "furthest blue lego block with 4 studs ". Add the colour in the tag.
        # Return ONLY raw JSON.
        # Do not use markdown code fences.
        # Do not write ```json.
        # {
        # "tag": "string",
        # "description": "string"
        # }
        # """

        # LABELED PROMPT
        question_2 = """You will receive:
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
        for p in range(len(segments)):
            crop_url = self.encode_image_data_from_array(segments[p])
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question_2},
                            {"type": "text", "text": "Full image:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_encoded, "detail": "auto"},
                            },
                            {"type": "text", "text": "Cropped image:"},
                            {"type": "image_url", "image_url": {"url": crop_url, "detail": "auto"}},
                        ],
                    },
                ],
                max_completion_tokens=self.max_completion_tokens,
                model=self.deployment,
            )
            raw = response.choices[0].message.content
            if raw is None:
                logger.error(f"GPT response is None for mask_{p}. Setting default values.")
                dict_outputs[f"mask_{p}"] = {
                    "tag": "unknown",
                    "description": "unknown",
                    "full_object": False,
                }
            try:
                dict_outputs[f"mask_{p}"] = json.loads(str(raw))
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON for mask_{p}: {raw}")
                dict_outputs[f"mask_{p}"] = {
                    "tag": "unknown",
                    "description": "unknown",
                    "full_object": False,
                }
            dict_outputs[f"mask_{p}"]["bbox"] = bboxes[p]

        return dict_outputs
