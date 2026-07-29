# plantorv

Object segmentation, VLM tagging, and depth estimation pipeline for a Femto Mega
RGB-D camera. Originally a single `samgpt.py` script, now split into three modules:

- **`segmentation/`** — `SAMModel`, wraps Segment Anything (SAM) to produce a
  background mask and per-object crops.
- **`vlm/`** — `GPTModel`, sends each crop plus the full scene to an Azure OpenAI
  deployment for an ultra-specific tag + description.
- **`depth/`** — `DepthRgbMapper` / `main_coords`, aligns the depth image to the RGB
  frame (hardcoded Femto Mega calibration) and reads the depth at each object center.

`plantorv.py` wires the three together.

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA (SAM is loaded with `device="cuda"` in
  [segmentation/sam.py](segmentation/sam.py))
- An Azure OpenAI resource with a vision-capable chat deployment

## 1. Install dependencies

Create and activate a virtual environment, then install the Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Install the segmentation model (SAM checkpoint)

The pipeline uses the **ViT-H** SAM checkpoint. `SAMModel` defaults to
`model_type="vit_h"` and `plantorv.py` loads it from `sam_vit_h_4b8939.pth` in the
project root, so download the file there:

```bash
# from the project root
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

(`curl -LO <url>` works too.) The file is ~2.4 GB and is git-ignored (`*.pth`).

If you want a smaller/faster model, download a different checkpoint and pass the
matching `model_type` when constructing `SAMModel`:

| model_type | checkpoint file            | download                                                                 |
|------------|----------------------------|--------------------------------------------------------------------------|
| `vit_h`    | `sam_vit_h_4b8939.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth     |
| `vit_l`    | `sam_vit_l_0b3195.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth     |
| `vit_b`    | `sam_vit_b_01ec64.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth     |

```python
# example: use the smaller vit_b model
SAMModel("sam_vit_b_01ec64.pth", model_type="vit_b")
```

## 3. Configure the `.env` file

The VLM step reads Azure OpenAI credentials from environment variables via
`python-dotenv`. Create a `.env` file in the project root (it is git-ignored):

```dotenv
# .env
AZURE_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_KEY=<your-azure-openai-api-key>
```

| Variable         | Used in                     | Description                                          |
|------------------|-----------------------------|------------------------------------------------------|
| `AZURE_ENDPOINT` | [plantorv.py](plantorv.py)  | Base URL of your Azure OpenAI resource               |
| `AZURE_KEY`      | [plantorv.py](plantorv.py)  | API key for that resource                            |

The deployment name (`gpt-5.2-chat`) and API version (`2024-12-01-preview`) are set
in [plantorv.py](plantorv.py) — edit them there to match your Azure deployment.

## 4. Run

Set the input RGB and depth image paths near the bottom of
[plantorv.py](plantorv.py) (`images` and `depth_path`), then:

```bash
source venv/bin/activate
python plantorv.py
```

For each object the pipeline outputs a tag, a description, whether the crop shows the
full object, the mask crop path, the bbox, and the object center with its depth in mm.

## Tests

Headless unit tests (no GPU, checkpoint, or network needed) live in [tests/](tests/):

```bash
./venv/bin/python -m unittest discover -s tests -v
```

See [tests/README.md](tests/README.md) for details and known issues.
