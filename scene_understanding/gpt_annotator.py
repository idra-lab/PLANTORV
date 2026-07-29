from openai import AzureOpenAI
import mimetypes
import base64
from pathlib import Path
import json


"""GPT Model for tagging and description"""

class GPTAnnotator:
    def __init__(self, endpoint, model_name, deployment, subscription_key, api_version):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.deployment = deployment

    def encode_image_data_url(self,image_path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def main_gpt(self,image,mask_path, bboxes):
        """
        This function is defined to obtain the tagging and description of the objects that we are looking for.
        It applies GPT over the original RGB image and the cropped images of the objects obtained with SAM, and then it returns a dictionary with the tagging and description of each object.
        Inputs:
            - image: the original RGB image. String. Path of the original RGB image.
            - mask_path: the paths of the masks obtained. List of strings. Output is a list of length N where each element is the path of the mask obtained for each object.
            - bboxes: the bounding boxes of the objects obtained by SAM.
        Outputs:
            - dict_outputs: the dictionary with the tagging and description of each object. Dictionary. Output is a dictionary where each key is the name of the object (for example, "mask_0") and each value is another dictionary with the following keys:
                - "tag": the tag of the object obtained by GPT. String.
                - "description": the description of the object obtained by GPT. String.
                - "mask": the path of the mask obtained for the object. String.
                - "bbox": the bounding box of the object obtained by SAM. List of 4 integers [x_min, y_min, width, height].
        """
        image_path = Path(image)
        image_data_url = self.encode_image_data_url(image_path)
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
        for p in range(len(mask_path)):
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
                            {"type": "image_url", "image_url": {"url": image_data_url}, "detail": "auto"},
                            {"type": "text", "text": "Cropped image:"},
                            {"type": "image_url", "image_url": {"url": crop_url}, "detail": "auto"},
                        ],

                    }
                ],
                max_completion_tokens =16384,
                model=self.deployment
            )
            raw = response.choices[0].message.content	
            try:
                dict_outputs[f"mask_{p}"] = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for mask_{p}: {raw}")
                dict_outputs[f"mask_{p}"] = {"tag": "unknown", "description": "unknown", "full_object": False}
            dict_outputs[f"mask_{p}"]["mask"]=mask_path[p]
            dict_outputs[f"mask_{p}"]["bbox"]=bboxes[p]

        return dict_outputs
