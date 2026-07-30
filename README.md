# PLANTORV

Pipeline for segmenting tabletop RGB-D scenes (Femto Mega camera), labeling detected
objects with Azure OpenAI, projecting depth coordinates, and evaluating results against
ArUco marker annotations.

## Code Structure

- `samgpt.py` - main pipeline: runs SAM segmentation, GPT labeling, RGB-D coordinate mapping, and writes JSON outputs.
- `segmentation/` - `SAMModel`, wraps Segment Anything (SAM) to produce a background mask and per-object crops.
- `scene_understanding/` - `GPTAnnotator`, sends each crop plus the full scene to an Azure OpenAI deployment for an ultra-specific tag + description.
- `mapping/` - RGB-D camera calibration (hardcoded Femto Mega intrinsics) and depth-to-color projection utilities.
- `aruco/` - ArUco marker detection, pose estimation, camera/config YAML files, and marker generation.
- `evaluation/` - matching, metrics, depth correlation, and result visualization.
- `utility/` - shared logging helpers (compact console + rotating file output, via `loguru`).
- `dataset/` - expected input images: `rgb/`, `depth/`, and `rgb_aruco/`.
- `models/` - local model checkpoints, including the SAM checkpoint expected at `models/sam/sam_vit_h_4b8939.pth`.
- `scripts/PBS/` - cluster job scripts (`generation.sh`, `evaluation.sh`, `aruco_detector.sh`).
- `outputs_json_labeled/`, `output_aruco/`, `results/`, `ppt_outputs/` - generated pipeline outputs.

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA (SAM is loaded with `device="cuda"` in
  [segmentation/sam_model.py](segmentation/sam_model.py))
- An Azure OpenAI resource with a vision-capable chat deployment

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install the segmentation model (SAM checkpoint)

The pipeline uses the **ViT-H** SAM checkpoint by default (`SAMModel(..., model_type="vit_h")`,
loaded by `samgpt.py` from `models/sam/sam_vit_h_4b8939.pth`). Download it there:

```bash
mkdir -p models/sam
wget -O models/sam/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

(`curl -Lo <path> <url>` works too.) The file is ~2.4 GB and is git-ignored (`*.pth`).

If you want a smaller/faster model, download a different checkpoint and pass the
matching `model_type` when constructing `SAMModel`:

| model_type | checkpoint file            | download                                                                 |
|------------|----------------------------|--------------------------------------------------------------------------|
| `vit_h`    | `sam_vit_h_4b8939.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth     |
| `vit_l`    | `sam_vit_l_0b3195.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth     |
| `vit_b`    | `sam_vit_b_01ec64.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth     |

```python
# example: use the smaller vit_b model
SAMModel("models/sam/sam_vit_b_01ec64.pth", model_type="vit_b")
```

### 3. Configure the `.env` file

`samgpt.py` reads Azure OpenAI credentials from environment variables via `python-dotenv`.
Create a `.env` file in the project root (it is git-ignored):

```dotenv
# .env
AZURE_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_API_KEY=<your-azure-openai-api-key>
```

| Variable         | Used in                 | Description                              |
|------------------|-------------------------|-------------------------------------------|
| `AZURE_ENDPOINT` | [samgpt.py](samgpt.py)  | Base URL of your Azure OpenAI resource    |
| `AZURE_API_KEY`  | [samgpt.py](samgpt.py)  | API key for that resource                 |

The deployment name (`gpt-5.2-chat`) and API version (`2024-12-01-preview`) are set
in [samgpt.py](samgpt.py) — edit them there to match your Azure deployment.

## Useful Commands

Run the main segmentation, labeling, and RGB-D coordinate pipeline:

```bash
python3 samgpt.py
```

Generate ArUco annotations from the configured RGB/marker image folders:

```bash
python3 aruco/aruco_detector.py \
  --clean_dir dataset/rgb_aruco \
  --tag_dir dataset/rgb_aruco \
  --out_dir output_aruco \
  --camera_yaml aruco/camera.yaml \
  --config_yaml aruco/config.yaml
```

Run the evaluation after `outputs_json_labeled/` and `output_aruco/` exist:

```bash
python3 evaluation/run_evaluation.py
```

Submit the existing cluster jobs:

```bash
qsub scripts/PBS/generation.sh
qsub scripts/PBS/aruco_detector.sh
qsub scripts/PBS/evaluation.sh
```

Clean generated outputs if you want a fresh run:

```bash
rm -rf outputs_json_labeled output_aruco results ppt_outputs
```

Format, lint, and type-check (via the `Makefile`):

```bash
make check      # format-check + lint + typecheck
make format     # ruff format .
make lint       # ruff check .
make typecheck  # pyright
```
