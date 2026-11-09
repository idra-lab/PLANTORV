"""Vision-language tagging and description with Azure OpenAI."""

import base64
import json
import mimetypes
from pathlib import Path

from openai import AzureOpenAI


class GPTModel:
    """Tag and describe cropped objects with Azure OpenAI."""

    def __init__(self, endpoint, model_name, deployment, subscription_key, api_version):
        """Initialize the Azure OpenAI client.

        Parameters
        ----------
        endpoint : str
            Azure OpenAI endpoint URL.
        model_name : str
            Model name associated with the deployment. This parameter is kept
            for caller compatibility.
        deployment : str
            Azure OpenAI deployment name used for chat completions.
        subscription_key : str
            Azure OpenAI API key.
        api_version : str
            Azure OpenAI API version.
        """
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.deployment = deployment

    def encode_image_data_url(self, image_path) -> str:
        """Encode an image file as a data URL.

        Parameters
        ----------
        image_path : pathlib.Path
            Path to the image file.

        Returns
        -------
        str
            Data URL containing the image MIME type and base64-encoded bytes.

        Raises
        ------
        FileNotFoundError
            If ``image_path`` does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def main_gpt(self, image, mask_path, masks, bboxes):
        """Tag and describe segmented objects with GPT.

        Parameters
        ----------
        image : str
            Path to the original RGB image.
        mask_path : list[str]
            Paths to cropped object images produced by SAM.
        masks : list[numpy.ndarray] or None
            Binary object masks. This parameter is currently unused but kept for
            caller compatibility.
        bboxes : list[list[int]] or numpy.ndarray
            SAM bounding boxes for the cropped objects. Each box is
            ``[x_min, y_min, width, height]``.

        Returns
        -------
        dict
            Object metadata keyed by ``"mask_{i}"``. Each value contains the
            GPT ``"tag"``, ``"description"``, ``"full_object"`` flag, saved
            ``"mask"`` path, and ``"bbox"``.
        """
        image_path = Path(image)
        image_data_url = self.encode_image_data_url(image_path)
        dict_outputs = {}
        question_2 = """You will receive:
        1) Two images of the same scene. The first image shows the whole scene, and the second image is a cropped region of the image. 
        The second image shows the object and the first one gives the context of the image.
        Your task:
        - Describe the object from the SECOND image, using the first one to consider the context of the workspace. Tell me the relative positions with respect the other objects that are seen in the first image, for example, specifying if they are on the left, on the rigth or next to another object.
        - The tagging should be ultra-specific. For example, instead of saying "lego block", say "blue lego block with 4 studs". Add the colour in the tag.
        - Express me if the cropped image shows the full object or not.
        {
        "tag": "string",
        "description": "string",
        "full_object:"boolean"
        }"""
        print(len(mask_path))
        for p in range(len(mask_path)):
            print(f"Processing mask_{p}...")
            crop_url = self.encode_image_data_url(Path(mask_path[p]))
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
                                "image_url": {"url": image_data_url},
                                "detail": "auto",
                            },
                            {"type": "text", "text": "Cropped image:"},
                            {"type": "image_url", "image_url": {"url": crop_url}, "detail": "auto"},
                        ],
                    },
                ],
                max_completion_tokens=16384,
                model=self.deployment,
            )
            raw = response.choices[0].message.content
            print(f"Raw GPT output for mask_{p}: {raw}")
            try:
                dict_outputs[f"mask_{p}"] = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for mask_{p}: {raw}")
                dict_outputs[f"mask_{p}"] = {
                    "tag": "unknown",
                    "description": "unknown",
                    "full_object": False,
                }
            dict_outputs[f"mask_{p}"]["mask"] = mask_path[p]
            dict_outputs[f"mask_{p}"]["bbox"] = bboxes[p]

        return dict_outputs
