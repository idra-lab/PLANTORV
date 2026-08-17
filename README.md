# PLANTORV

Pipeline for segmenting tabletop RGB-D scenes (Femto Mega camera), labeling detected
objects with Azure OpenAI, projecting depth coordinates, and evaluating results against
ArUco marker annotations.

## Code Structure

- `samgpt.py` - main pipeline: runs SAM segmentation, GPT labeling, RGB-D coordinate mapping, and writes JSON outputs.
- `segmentation/` - `SegmentationModel` base class, `SAMModel` (SAM 1, MobileSAM, SAM 2 and SAM 2.1) and `FastSAMModel`, all run through Ultralytics to produce a background mask and per-object crops.
- `scene_understanding/` - `GPTAnnotator`, sends each crop plus the full scene to an Azure OpenAI deployment for an ultra-specific tag + description.
- `mapping/` - RGB-D camera calibration (hardcoded Femto Mega intrinsics) and depth-to-color projection utilities.
- `aruco/` - ArUco marker detection, pose estimation, camera/config YAML files, and marker generation.
- `evaluation/` - matching, metrics, depth correlation, and result visualization.
- `utility/` - shared logging helpers (compact console + rotating file output, via `loguru`).
- `dataset/` - expected input images: `rgb/`, `depth/`, and `rgb_aruco/`.
- `models/` - local model checkpoints, including the SAM checkpoint expected at `models/sam/sam_h.pt`.
- `scripts/install_models.py` - downloads the checkpoints into `models/` (see [Setup](#2-install-the-segmentation-model-sam-checkpoint)).
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

Segmentation runs through [Ultralytics](https://docs.ultralytics.com/models/sam/), which
picks the SAM architecture from the **checkpoint file name**. The name must therefore end
with one of the names in `SAM_CHECKPOINTS`
([segmentation/sam_model.py](segmentation/sam_model.py)); `SAMModel` rejects anything else
with an explanatory error. The path in front of it is free.

The same class drives SAM 1 and SAM 2 — the checkpoint name also selects the matching
Ultralytics predictor, so there is no separate SAM 2 class to import.

The pipeline defaults to **ViT-H**, loaded by `samgpt.py` from `models/sam/sam_h.pt`.
Download it with the installer script:

```bash
python3 scripts/install_models.py sam_h
```

It only uses the standard library, so it can be run before `make install`, and an
interrupted download resumes where it stopped. The other entry points:

```bash
python3 scripts/install_models.py --list   # show what is available
python3 scripts/install_models.py --all    # download everything
python3 scripts/install_models.py          # choose interactively
```

#### SAM 1

| model             | key / file             | size     | downloaded from |
|-------------------|------------------------|----------|-----------------|
| ViT-H (default)   | `sam_h.pt`             | ~2.4 GB  | Meta            |
| ViT-L             | `sam_l.pt`             | ~1.2 GB  | Ultralytics     |
| ViT-B             | `sam_b.pt`             | ~360 MB  | Ultralytics     |
| MobileSAM         | `mobile_sam.pt`        | ~39 MB   | Ultralytics     |

#### SAM 2 and SAM 2.1

Same four sizes in both generations, all from Ultralytics. SAM 2.1 is the later release of
the same architecture and is preferred; SAM 2 is kept so older runs stay reproducible.

| variant | SAM 2       | SAM 2.1       | size    |
|---------|-------------|---------------|---------|
| tiny    | `sam2_t.pt` | `sam2.1_t.pt` | ~75 MB  |
| small   | `sam2_s.pt` | `sam2.1_s.pt` | ~88 MB  |
| base+   | `sam2_b.pt` | `sam2.1_b.pt` | ~154 MB |
| large   | `sam2_l.pt` | `sam2.1_l.pt` | ~428 MB |

SAM 2 checkpoints are much smaller than their SAM 1 counterparts for two reasons: the
hierarchical Hiera encoder needs roughly a third of the parameters of SAM 1's plain ViT
(224 M vs 641 M for the largest of each), and they are stored in fp16 rather than fp32.

#### FastSAM

| variant | file             | size    | downloaded from |
|---------|------------------|---------|-----------------|
| s       | `FastSAM-s.pt`   | ~23 MB  | Ultralytics     |
| x       | `FastSAM-x.pt`   | ~138 MB | Ultralytics     |

FastSAM is not a SAM architecture: it is a YOLOv8-seg model that produces all its masks in
one forward pass, driven by Ultralytics' `FastSAMPredictor` rather than the SAM predictor.
It has no `points_stride` "segment everything" mode, and `build_sam` does not recognise its
checkpoints, so `SAMModel` cannot load it. It has its own class instead, with the same
interface:

```python
from segmentation.fastsam_model import FastSAMModel

sam = FastSAMModel("models/fastsam/FastSAM-s.pt", save_dir=..., device="cuda")
```

`FastSAMModel` implements `SegmentationModel` directly and owns its whole pipeline rather
than reusing `SAMModel`, so its thresholds can move independently — the two models produce
very differently shaped mask sets. They are module constants at the top of
[segmentation/fastsam_model.py](segmentation/fastsam_model.py). FastSAM has no
`points_stride`; use `conf`, `iou` and `imgsz` instead.

> **`obtain_bg` works differently here, by necessity.** FastSAM is a *thing* detector: it
> proposes object instances and never emits "stuff" regions like walls or tables. On
> `rgb_dataset_1.png` its 79 masks cover only **18% of the image** (SAM 1 ViT-H covers
> 97%) and its largest is 9.2%, which is part of an object rather than the background. So
> SAM's rule — background is the *biggest* masks — finds nothing at any threshold. FastSAM
> instead takes the background to be the **complement of every mask**, which is exactly
> right for a model that only masks objects, and needs no threshold to tune.
>
> With that, FastSAM-s finds the same four objects as SAM 1 ViT-H on that frame, with
> bounding boxes within ~30 px. It still splits the robot arm into three masks where SAM
> returns one; raising `conf` (0.6 roughly halves the mask count) reduces that splitting.

To use any of them, pass the name or path when constructing `SAMModel` — there is no
separate model-type argument, and no separate SAM 2 class:

```python
sam = SAMModel("models/sam/sam_b.pt", save_dir=..., device="cuda")  # SAM 1
sam = SAMModel("models/sam/sam2.1_l.pt", save_dir=..., device="cuda")  # SAM 2.1
```

Ultralytics fetches everything except `sam_h.pt` itself if it is missing, so for those the
script is optional — `SAMModel("sam2.1_l.pt")` works with no setup. Note that Ultralytics
downloads into the **current working directory**, not into `models/`, so using the script
keeps the checkpoints together and makes runs independent of where they are launched from.

ViT-H is the exception — Ultralytics publishes no `sam_h.pt` in any release, so the script
takes Meta's original checkpoint and saves it under that name. The weights are unchanged:
`.pth` and `.pt` are both `torch.save` archives, and the extension is only a naming
convention.

> **Heads-up before switching models.** The area and IoU thresholds in `individual_mask`
> were tuned against SAM 1 ViT-H. On the same tabletop frame, SAM 2.1 and MobileSAM return
> far fewer, much coarser masks (whole table regions rather than objects), so swapping the
> checkpoint alone will change the pipeline output noticeably. Run with `--debug-masks`
> (see [Useful Commands](#useful-commands)) and re-check those thresholds first.

### 3. Configure the `.env` file

`samgpt.py` reads Azure OpenAI credentials from environment variables via `python-dotenv`.
Create a `.env` file in the project root (it is git-ignored):

```dotenv
# .env
AZURE_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_API_KEY=<your-azure-openai-api-key>
```

| Variable         | Used in                 | Description                              |
|------------------|-------------------------|------------------------------------------|
| `AZURE_ENDPOINT` | [samgpt.py](samgpt.py)  | Base URL of your Azure OpenAI resource   |
| `AZURE_API_KEY`  | [samgpt.py](samgpt.py)  | API key for that resource                |

The deployment name (`gpt-5.2-chat`) and API version (`2024-12-01-preview`) are set
in [samgpt.py](samgpt.py) — edit them there to match your Azure deployment.

## Useful Commands

Run the main segmentation, labeling, and RGB-D coordinate pipeline:

```bash
python3 samgpt.py
```

Inspect what the segmentation model is actually producing. This saves every mask SAM
returns *before* any filtering to `<output-dir>/segmentation_outputs/debug/` — a numbered
colour-coded overlay per pass plus one PNG per mask — and logs the area, bounding box, and
the reason each mask was kept or dropped. Use it to tell "SAM never found this object"
apart from "the area/IoU thresholds discarded it":

```bash
python3 samgpt.py --debug-masks
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
rm -rf output
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
