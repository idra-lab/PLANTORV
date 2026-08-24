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
- `.dev-config/` - Ruff and Pyright settings, referenced from `pyproject.toml` (see [Development](#development)).
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
make install
```

For development, use `install-dev` instead. It adds the linting/type-checking tools
and enables the git hook in one step:

```bash
make install-dev
```

Both targets are thin wrappers, so the raw equivalents work too:

| target | runs |
| --- | --- |
| `make install` | `pip install -e .` |
| `make install-dev` | `pip install -e ".[dev]"` then `pre-commit install` |


### 2. Install the segmentation model (SAM checkpoint)

The pipeline uses the **ViT-H** SAM checkpoint by default (`SAMModel(..., model_type="vit_h")`,
loaded by `samgpt.py` from `models/sam/sam_vit_h_4b8939.pth`). Download it there:

```bash
mkdir -p models/sam
wget -O models/sam/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
If you want a smaller/faster model, download a different checkpoint and pass the
matching `model_type` when constructing `SAMModel`:

| model_type | checkpoint file            | download                                                                 |
|------------|----------------------------|--------------------------------------------------------------------------|
| `vit_h`    | `sam_vit_h_4b8939.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth     |
| `vit_l`    | `sam_vit_l_0b3195.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth     |
| `vit_b`    | `sam_vit_b_01ec64.pth`     | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth     |

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

### PCD Utils

Run the main segmentation, labeling, and RGB-D coordinate pipeline:

```bash
python3 samgpt.py
```

Create a coloured point cloud from a Femto Mega RGB/depth pair. The command aligns
the raw depth frame to the RGB camera before passing both images and the calibrated
RGB intrinsics to Open3D:

```bash
python3 cluster_pcd.py \
  dataset/rgb/rgb_dataset_1.png \
  dataset/depth/depth_dataset_1.png \
  --output results/scene_1.ply \
  --cluster-output results/scene_1_clusters.ply \
  --labels-output results/scene_1_geometric_labels.npy
```

Use `--no-visualize` on a headless machine. If the stored depth values are not
millimetres, pass their conversion factor with `--depth-unit-scale`. Plane RANSAC
and DBSCAN can be tuned with `--plane-distance`, `--min-plane-points`,
`--cluster-eps`, and `--cluster-min-points`. In the saved label image, clusters are
numbered from `0`, unclustered/background pixels are `-1`, and planes start at `-2`.


#### Segmentation with Models

With Rand-LA net: 

```bash
python segment_pcd.py \
    dataset/rgb/rgb_dataset_1.png \
    dataset/depth/depth_dataset_1.png \
    --config models/randla_net/randlanet_s3dis.yml \
    --checkpoint models/randla_net/randlanet_s3dis_202201071330utc.pth \
    --semantic-output semantic.ply \
    --labels-output labels.npy \
    --confidence-output confidence.npy
```
With Pointtrasnformer (not yet well interfaced): 
```bash
python segment_pcd.py \
    dataset/rgb/rgb_dataset_1.png \
    dataset/depth/depth_dataset_1.png \
    --config models/pointtrasformer/pointtransformer_s3dis.yml \
    --checkpoint models/pointtrasformer/pointtransformer_s3dis_202109241350utc.pth \
    --semantic-output semantic.ply \
    --labels-output labels.npy \
    --confidence-output confidence.npy
```
### Aruco Generation

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

## Development

Format, lint, and type-check (via the `Makefile`):

```bash
make check      # format-check + lint + typecheck
make format     # ruff format .
make lint       # ruff check .
make typecheck  # pyright
```

The same checks also run on commit via [pre-commit](https://pre-commit.com/).
`make install-dev` enables the git hook for you; to enable it in an existing clone:

```bash
pre-commit install
```

Note the two are deliberately different, not redundant:

- `make check` is **read-only** and covers the **whole repo** — use it to verify.
- the commit hook **auto-fixes** (`ruff --fix`, `ruff-format`) and, apart from `pyright`,
  only looks at **staged files**.

Because of the auto-fix, `pre-commit run --all-files` will rewrite files; use
`make check` when you want to inspect without changing anything.

`pyright` runs against the environment that's active when `git commit` is run (not an
isolated pre-commit env), so make sure `make install-dev` was run in the venv you
commit from.

### Tool configuration

Ruff and Pyright settings live in [.dev-config/](.dev-config/) rather than in
`pyproject.toml`, which only holds two pointers:

```toml
[tool.ruff]
extend = ".dev-config/ruff.toml"

[tool.pyright]
extends = ".dev-config/pyrightconfig.json"
```

Both tools follow these on their own, so `ruff`, `pyright`, the `Makefile`, pre-commit,
and editor language servers all work with no extra flags.

⚠️ The two files resolve relative paths **differently**:

| file | paths resolve against | so `include`/`exclude` are written as |
| --- | --- | --- |
| `.dev-config/ruff.toml` | the project root | `dataset`, `venv`, … |
| `.dev-config/pyrightconfig.json` | **its own directory** | `../dataset`, `../venv`, … |

If you drop the `../` prefixes in the Pyright config it will match nothing, analyze
**zero files, and still exit 0** — a passing check that verified nothing. Keep the
prefixes when editing that file.
