<p align='center'>
<a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" target="_blank" />
</a>
</a href="https://pytorch.org/" target="_blank">
    <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" target="_blank" />
</a>
</a href="https://huggingface.co/" target="_blank">
    <img src="https://img.shields.io/badge/huggingface-FB8C00?style=for-the-badge&logo=huggingface&logoColor=white" target="_blank" />
</a>
</a href="https://www.ultralytics.com/" target="_blank">
    <img src="https://img.shields.io/badge/ultralytics-FF6C37?style=for-the-badge&logo=ultralytics&logoColor=white" target="_blank" />
</a>
</p>

<p align='center'>
    <h1 align="center">PLanning with Natural language for Task-Oriented Robots with Vision (PLANTORV)</h1>
</p>

Pipeline for segmenting tabletop RGB-D scenes (Femto Mega camera), labeling detected
objects with a vision LLM, projecting depth coordinates, and evaluating results against
ArUco marker annotations.

The pipeline is four stages, each with its own section below:

| stage | what it does | section |
|-------|--------------|---------|
| Segmentation | turns a frame into per-object masks and crops | [Segmentation](#segmentation) |
| Scene Understanding | gives each crop a tag and a description | [Scene Understanding](#scene-understanding) |
| Depth estimation | attaches a 3D position to each object | [Depth estimation](#depth-estimation) |
| Evaluation | scores the result against ArUco ground truth | [Useful Commands](#useful-commands) |

Stages 1 and 2 both talk to an LLM through a shared, backend-agnostic layer, described in
[LLM Backbone](#llm-backbone).

----

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Install dependencies](#install-dependencies)
  - [Install segmentation models](#install-segmentation-models)
  - [Configure the `.env` file](#configure-the-env-file)
- [Code Structure](#code-structure)
- [LLM Backbone](#llm-backbone)
- [Segmentation](#segmentation)
  - [Install script](#install-script)
  - [SAMModel](#sammodel)
    - [SAM 1](#sam-1)
    - [SAM 2 and SAM 2.1](#sam-2-and-sam-21)
  - [FastSAMModel](#fastsammodel)
  - [SAM 3](#sam-3)
    - [Getting the weights](#getting-the-weights)
- [Scene Understanding](#scene-understanding)
- [Depth estimation](#depth-estimation)
- [Useful Commands](#useful-commands)
- [Development](#development)
  - [Tool configuration](#tool-configuration)



## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA (`--device` defaults to `cuda`; pass `--device cpu` to run
  without one, slowly)
- An Azure OpenAI resource with a vision-capable chat deployment, or any other backend
  from [LLM Backbone](#llm-backbone)
- For SAM 3 only: a Hugging Face account with approved access to the gated weights (see
  [Getting the weights](#getting-the-weights))

## Setup

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
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

### Install segmentation models

Checkpoints are downloaded by `scripts/install_models.py` into `models/`. `samgpt.py`
currently builds a SAM 3 model, whose weights are gated and need a Hugging Face token:

```bash
python3 scripts/install_models.py sam3
```

See [Segmentation → Install script](#install-script) for the other models and the full set
of options, and [Getting the weights](#getting-the-weights) for the SAM 3 access steps.

### Configure the `.env` file

Credentials are read from environment variables via `python-dotenv`. Create a `.env` file
in the project root (it is git-ignored):

```dotenv
# .env
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
HF_TOKEN=<your-hugging-face-token>
```

| Variable                | Read by                                                | Description                                       |
|-------------------------|--------------------------------------------------------|---------------------------------------------------|
| `AZURE_OPENAI_ENDPOINT` | [LLM/LLMAzureOpenAI/](LLM/LLMAzureOpenAI/LLMAzureOpenAI.py) | Base URL of your Azure OpenAI resource        |
| `AZURE_OPENAI_API_KEY`  | [LLM/LLMAzureOpenAI/](LLM/LLMAzureOpenAI/LLMAzureOpenAI.py) | API key for that resource                     |
| `HF_TOKEN`              | [scripts/install_models.py](scripts/install_models.py) | Hugging Face token, only needed to download SAM 3 |

Those two Azure names are not hardcoded — the YAML config defines which variables to read, so
a different backend means different variables. See [LLM Backbone](#llm-backbone).

`HF_TOKEN` is optional: it is read only when downloading the gated `sam3.pt`, and only when
no `hf auth login` token is already stored. See
[Getting the weights](#getting-the-weights).

Both `samgpt.py` and [LLM/llm_base.py](LLM/llm_base.py) load the environment variables from
the `.env` file that sits in the root. Two flags are made available from `samgpt.py`:

| flag | effect |
| ------ | -------- |
| *(none)* | the project root's `.env` is used |
| `--env-file other.env` | `other.env` is read instead by `samgpt.py` **and** by the LLM backends |
| `--no-env-file` | no file is read at all; only the shell environment is used |

## Code Structure

- `samgpt.py` - main pipeline: segmentation, LLM labeling, RGB-D coordinate mapping, and JSON outputs. The segmentation model and the LLM config are chosen here; the LLM one is overridable with `--llm-config`.
- `segmentation/` - `SegmentationModel` base class, `SAMModel` (SAM 1, MobileSAM, SAM 2 and SAM 2.1) and `FastSAMModel`, all run through Ultralytics to produce a background mask and per-object crops. `SAM3Model` sits alongside them and works differently: a VLM names the things in the scene and SAM 3 segments those concepts, so its masks arrive already labelled.
- `scene_understanding/` - `GPTAnnotator`, sends each crop plus the full scene to a vision LLM for a tag + description. The backend is whichever one the YAML config selects, not Azure specifically.
- `LLM/` - backend-agnostic LLM layer (`BaseLLM`, one package per provider, YAML configs in `LLM/conf/`), plus `LLM/examples/SAM3/concept_prompts.yaml`, the phrasing examples used to steer SAM 3's concepts.
- `mapping/` - RGB-D camera calibration (hardcoded Femto Mega intrinsics) and depth-to-color projection utilities.
- `aruco/` - ArUco marker detection, pose estimation, camera/config YAML files, and marker generation.
- `evaluation/` - matching, metrics, depth correlation, and result visualization.
- `utility/` - shared logging helpers (compact console + rotating file output, via `loguru`).
- `models/` - local model checkpoints, including the SAM checkpoint expected at `models/sam/sam_*.pt` and, for SAM 3, `models/sam/sam3.pt`.
- `scripts/install_models.py` - downloads the checkpoints into `models/` (see [Install script](#install-script)).
- `scripts/PBS/` - cluster job scripts (`generation.sh`, `evaluation.sh`, `aruco_detector.sh`).
- `.dev-config/` - Ruff and Pyright settings, referenced from `pyproject.toml` (see [Development](#development)).

## LLM Backbone

Everything that talks to a language model goes through [LLM/](LLM/).

- [`LLM/llm_base.py`](LLM/llm_base.py) — `BaseLLM`, the interface every backend implements.
  The public surface is `from_config`, `query(message, images=...)` and `prepare`; a
  backend fills in how to build its client, send messages and read the reply. Backends that
  accept images set `SUPPORTS_IMAGES = True`, which both the annotator and `SAM3Model`
  require and check at construction. `configure_env(path, load=...)` in `llm_base` — chooses
  the environment file the whole layer reads. `samgpt.py` calls it so `--env-file` and
  `--no-env-file` reach the backends;
  see [Configure the `.env` file](#configure-the-env-file).
- [`LLM/llm_factory.py`](LLM/llm_factory.py) — `create_llm(config_file)`, which picks the
  class. An explicit `PROVIDER` key wins; otherwise the provider is inferred from the
  config.
- `LLM/<Provider>/` — one package per backend: `azure_openai`, `openai`, `anthropic`,
  `gemini`, `glm`, `huggingface`, `vllm`.
- [`LLM/conf/`](LLM/conf/) — one YAML file per model, containing all the necessary configuration.

**The YAML file carries everything the model needs**, including which environment variables
hold the credentials, through its `ENDPOINT_ENV` and `API_KEY_NAME` keys — see
[LLM/conf/azure_gpt52.yaml](LLM/conf/azure_gpt52.yaml):

```yaml
LLM_VERSION : "gpt-5.2-chat"           # the deployment name
API_KEY_NAME : "AZURE_OPENAI_API_KEY"  # which env var holds the key
ENDPOINT_ENV : "AZURE_OPENAI_ENDPOINT" # which env var holds the endpoint
API_VERSION : "2024-12-01-preview"

# The gpt-5 family rejects temperature/top_p and renamed the token cap.
LLM_CONFIG:
  max_completion_tokens: 16384
  seed: 42
```

Whatever sits under `LLM_CONFIG` is forwarded to the provider request as-is, which is why a
model wanting `max_completion_tokens` and one wanting `max_tokens` differ only by their
config file rather than by a flag in the code.

So the deployment name, the API version, the credentials and the request parameters all
live in configuration, and switching model or provider is a matter of pointing
`--llm-config` somewhere else:

```bash
python3 samgpt.py --llm-config LLM/conf/gemini_2-0_flash.yaml
```

## Segmentation

Every segmentation model implements `SegmentationModel`
([segmentation/segmentation.py](segmentation/segmentation.py)), which is two calls:
`obtain_bg` returns the background-masked frame plus its binary mask, and `individual_mask`
returns one cropped RGB image and bounding box per object. `samgpt.py` calls them in that
order, so the models are interchangeable from the pipeline's point of view.

`SAM3Model` also implements them, but its real interface is `segment`, which additionally
returns the concept each mask was found by — see [SAM 3](#sam-3).

### Install script

Checkpoints are fetched by [scripts/install_models.py](scripts/install_models.py) into
`models/`. It only uses the standard library, so it can be run before `make install`, and
an interrupted download resumes where it stopped.

```bash
python3 scripts/install_models.py sam3 sam_h   # download by name
python3 scripts/install_models.py --list       # show what is available
python3 scripts/install_models.py --all        # download everything
python3 scripts/install_models.py              # choose interactively
```

Ultralytics fetches its own checkpoints on first use if they are missing, so for those the
script is optional — `SAMModel("sam2.1_l.pt")` works with no setup. Note that Ultralytics
downloads into the **current working directory**, not into `models/`, so using the script
keeps the checkpoints together and makes runs independent of where they are launched from.

Two checkpoints are exceptions Ultralytics cannot fetch at all: `sam_h.pt`, which it
publishes in no release, and the gated `sam3.pt`.

### SAMModel

[`SAMModel`](segmentation/sam_model.py) drives SAM 1 and SAM 2 through
[Ultralytics](https://docs.ultralytics.com/models/sam/), which picks the architecture from
the **checkpoint file name**. The name must therefore end with one of the names in
`SAM_CHECKPOINTS` ([segmentation/sam_model.py](segmentation/sam_model.py)); `SAMModel`
rejects anything else with an explanatory error. The path in front of it is free.

The same class drives both generations — the checkpoint name also selects the matching
Ultralytics predictor, so there is no separate SAM 2 class to import:

```python
sam = SAMModel("models/sam/sam_b.pt", save_dir=..., device="cuda")     # SAM 1
sam = SAMModel("models/sam/sam2.1_l.pt", save_dir=..., device="cuda")  # SAM 2.1
```

#### SAM 1

Docs: [SAM](https://docs.ultralytics.com/models/sam/),
[MobileSAM](https://docs.ultralytics.com/models/mobile-sam/).

| model             | key / file             | size     | downloaded from |
|-------------------|------------------------|----------|-----------------|
| ViT-H (default)   | `sam_h.pt`             | ~2.4 GB  | [Meta](https://github.com/facebookresearch/segment-anything) |
| ViT-L             | `sam_l.pt`             | ~1.2 GB  | [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |
| ViT-B             | `sam_b.pt`             | ~360 MB  | [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |
| MobileSAM         | `mobile_sam.pt`        | ~39 MB   | [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |

ViT-H is the exception — Ultralytics publishes no `sam_h.pt` in any release, so the script
takes Meta's original checkpoint and saves it under that name. The weights are unchanged:
`.pth` and `.pt` are both `torch.save` archives, and the extension is only a naming
convention.

#### SAM 2 and SAM 2.1

Docs: [SAM 2](https://docs.ultralytics.com/models/sam-2/). Same four sizes in both
generations, all from
[Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0). SAM 2.1 is the
later release of the same architecture and is preferred; SAM 2 is kept so older runs stay
reproducible.

| variant | SAM 2       | SAM 2.1       | size    |
|---------|-------------|---------------|---------|
| tiny    | `sam2_t.pt` | `sam2.1_t.pt` | ~75 MB  |
| small   | `sam2_s.pt` | `sam2.1_s.pt` | ~88 MB  |
| base+   | `sam2_b.pt` | `sam2.1_b.pt` | ~154 MB |
| large   | `sam2_l.pt` | `sam2.1_l.pt` | ~428 MB |

> **Heads-up before switching models.** The area and IoU thresholds in `individual_mask`
> were tuned against SAM 1 ViT-H. On the same tabletop frame, SAM 2.1 and MobileSAM return
> far fewer, much coarser masks (whole table regions rather than objects), so swapping the
> checkpoint alone will change the pipeline output noticeably. Run with `--debug-masks`
> (see [Useful Commands](#useful-commands)) and re-check those thresholds first.

### FastSAMModel

Docs: [FastSAM](https://docs.ultralytics.com/models/fast-sam/).

| variant | file             | size    | downloaded from |
|---------|------------------|---------|-----------------|
| s       | `FastSAM-s.pt`   | ~23 MB  | [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |
| x       | `FastSAM-x.pt`   | ~138 MB | [Ultralytics](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |

FastSAM is not a SAM architecture: it is a YOLOv8-seg model that produces all its masks in
one forward pass, driven by Ultralytics' `FastSAMPredictor` rather than the SAM predictor.
Since it has no "segment everything" mode, and `build_sam` does not recognise its
checkpoints, FastSAM has its own class instead, with the same interface:

```python
from segmentation.fastsam_model import FastSAMModel

sam = FastSAMModel("models/fastsam/FastSAM-s.pt", save_dir=..., device="cuda")
```

`FastSAMModel` implements `SegmentationModel` directly so its thresholds can move
independently. They are module constants at the top of
[segmentation/fastsam_model.py](segmentation/fastsam_model.py). FastSAM has no
`points_stride`; use `conf`, `iou` and `imgsz` instead.

> **`obtain_bg` works differently here, by necessity.** FastSAM is a *thing* detector: it
> proposes object instances and never emits "stuff" regions like walls or tables. So
> SAM's rule — background is the *biggest* masks — finds nothing at any threshold. FastSAM
> instead takes the background to be the **complement of every mask**, which is exactly
> right for a model that only masks objects, and needs no threshold to tune.

### SAM 3

Docs: [SAM 3](https://docs.ultralytics.com/models/sam-3/). Weights:
[facebook/sam3](https://huggingface.co/facebook/sam3).

| model | key / file | size    | downloaded from        |
|-------|------------|---------|------------------------|
| SAM 3 | `sam3.pt`  | ~3.2 GB | [Meta](https://huggingface.co/facebook/sam3), **gated** (see below) |

SAM 3 is not a drop-in replacement for the others, and it does not go through `SAMModel`.
The models above are asked to segment *everything* and hand back an unlabelled pile of
masks that `individual_mask` then filters down with area, border and IoU heuristics. SAM 3
is asked for **concepts**: it takes short noun phrases and returns every region matching
each one, already labelled with the phrase that found it. It has its own class,
[segmentation/sam3_model.py](segmentation/sam3_model.py), which drives Ultralytics'
`SAM3SemanticPredictor` and takes a VLM in its constructor — the VLM looks at the scene and
decides what the phrases should be:

```python
from LLM.llm_factory import create_llm
from segmentation.sam3_model import SAM3Model

sam = SAM3Model(create_llm("LLM/conf/azure_gpt52.yaml"), "models/sam/sam3.pt",
                save_dir=..., device="cuda")
masks, bboxes, labels = sam.segment(image)   # labels[i] is the concept that found masks[i]
```

`obtain_bg` and `individual_mask` still work, so `samgpt.py` runs unchanged, but they throw
the labels away — `segment` is the real interface. SAM 3 needs **`ultralytics >= 8.4.114`**;
the feature landed in 8.3.237, but `SAM3SemanticPredictor` crashed on every call until
8.4.114. `requirements.txt` pins the working floor.

#### Getting the weights

Meta distributes `sam3.pt` only through a **gated** Hugging Face repository, so Ultralytics
cannot fetch it on first use the way it fetches the SAM 2 and FastSAM checkpoints. Three
one-off steps:

1. **Request access.** Sign in to [huggingface.co](https://huggingface.co), open
   [facebook/sam3](https://huggingface.co/facebook/sam3), and submit the access form (name,
   affiliation, and accepting Meta's licence). Approval is manual, so it is not instant.
2. **Create a token.** Once approved, go to
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a
   token with permission to read gated repositories.
3. **Make the token available**, either by logging in once, which stores it in
   `~/.cache/huggingface/token`:

   ```bash
   hf auth login
   ```

   or by putting it in the project's `.env` as `HF_TOKEN` (see
   [Configure the `.env` file](#configure-the-env-file)).

Then download it like any other checkpoint:

```bash
python3 scripts/install_models.py sam3
```

The installer looks for `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the environment and in
`.env` first, then falls back to a stored `hf auth login`, so a machine that is already
logged in needs no further setup. It distinguishes the two failures: no token at all, and a
token whose account was never granted access. Please notice that `--all` skips SAM3 unless a 
token is present.

> **Concept phrasing decides the mask quality**, far more than any SAM 3 setting. A phrase
> has to settle *what counts as one thing*: `"lego block"` matches a single brick, a stack
> of two, and the whole tower they form, and SAM 3 returns all of those, nested. Naming the
> assembly instead — `"tall green tower"`, `"long blue object"` — lowers the number of masks
> in the same frame. The prompt that steers the VLM lives in `sam3_model.py`, and the
> good/bad phrasing examples it injects are in
> [LLM/examples/SAM3/concept_prompts.yaml](LLM/examples/SAM3/concept_prompts.yaml).
>
> As a safety net, `SAM3Model` merges masks that share a concept and touch, which puts a
> row of bricks returned brick-by-brick back together. Tune it with `merge_gap` (measured
> in pixels, `0` disables the functionality).

## Scene Understanding

[`GPTAnnotator`](scene_understanding/gpt_annotator.py) turns the crops from segmentation
into text. For each object it sends **two images** — the full scene and the crop — so the
model can describe the object using the rest of the workspace as context, and place it
relative to its neighbours:

```python
from scene_understanding.gpt_annotator import GPTAnnotator

gpt = GPTAnnotator.from_config("LLM/conf/azure_gpt52.yaml")
image_dict = gpt.main_gpt(image, rgb_masks, bboxes)
```

The result is one entry per mask, keyed `mask_0`, `mask_1`, …:

```json
{
  "mask_0": {
    "tag": "Metallic Wrench",
    "description": "a wrench lying to the left of the red box",
    "bbox": [829, 431, 129, 210]
  }
}
```

Two prompts live at the top of the module and are chosen by the `prompt` argument:

| prompt | behaviour |
|--------|-----------|
| `ANNOTATION_PROMPT` (default) | `tag` must come from a fixed list of workspace objects, with `"Unknown object"` as the fallback |
| `FREEFORM_ANNOTATION_PROMPT` | no fixed list; the tag is free text and asked to be ultra-specific |

A malformed or missing reply is logged and becomes a placeholder entry rather than raising,
so one bad answer costs one object rather than the whole run. The backend is whatever
`--llm-config` selects, and it must support images.

## Depth estimation

[`mapping/`](mapping/) attaches a 3D position to each annotated object. The depth and colour
sensors of the Femto Mega do not share a viewpoint, so a depth pixel and a colour pixel with
the same coordinates are not the same point in the scene: the depth frame has to be
reprojected into the colour frame first.

- [`mapping/camera_model.py`](mapping/camera_model.py) holds the calibration as module
  constants — per-sensor intrinsics, Brown-Conrady distortion coefficients, and the
  rotation/translation between the two sensors. They are hardcoded for the Femto Mega, so a
  different camera means editing this file.
- [`mapping/rgbd_mapper.py`](mapping/rgbd_mapper.py) does the reprojection: undistort depth
  pixels to normalised coordinates, transform them into the colour frame, project back
  through the colour intrinsics, and keep the points that land inside the image.

`main_coords` is the entry point the pipeline uses. It takes the RGB frame, the depth frame
and the annotation dictionary, and adds one key per object:

```python
from mapping.rgbd_mapper import main_coords

image_dict = main_coords(image, depth_path, image_dict)
# image_dict["mask_0"]["coord_center&depth"] == [center_x, center_y, depth_mm]
```

The centre is the middle of the object's bounding box, and the depth is read from the
aligned depth image at that pixel, in millimetres. `samgpt.py` writes the enriched
dictionary to `<output-dir>/output_img<N>.json`.

## Useful Commands

Run the main segmentation, labeling, and RGB-D coordinate pipeline:

```bash
python3 samgpt.py
```

Full set of flags, all optional:

| Flag | Default | Purpose |
|------|---------|---------|
| `--images-dir` | `dataset/rgb` | RGB input frames |
| `--depth-dir` | `dataset/depth` | Matching depth frames |
| `--output-dir` | `output` | Where JSON and segmentation outputs are written |
| `--llm-config` | `LLM/conf/azure_gpt52.yaml` | LLM YAML config; selects model, backend and credentials |
| `--env-file` | `.env` | Environment file to load |
| `--no-env-file` | off | Skip loading it, e.g. when the variables are already exported |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--log-level` | `DEBUG` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--view-masks` | off | Write and open an HTML page of the masks that were kept |
| `--debug-masks` | off | Dump every mask of every pass for inspection |

Inspect what the segmentation model is actually producing. This saves every mask SAM
returns *before* any filtering to `<output-dir>/segmentation_outputs/debug/` — a numbered
colour-coded overlay per pass plus one PNG per mask — and logs the area, bounding box, and
the reason each mask was kept or dropped. Use it to tell "SAM never found this object"
apart from "the area/IoU thresholds discarded it":

```bash
python3 samgpt.py --debug-masks
```

See which masks the run actually kept. This writes a single self-contained HTML page to
`<output-dir>/segmentation_outputs/` and opens it — the outlined overlay, then one card per
mask with its concept, area and bounding box, and a note of any concept that matched
nothing. Unlike `--debug-masks` it shows the finished result rather than every intermediate
pass, and because the images are embedded the page can be copied off a cluster node and
opened anywhere:

```bash
python3 samgpt.py --view-masks
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
isolated pre-commit env), so make sure `make install-dev` was run in the `.venv` you
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

**NOTICE** that the two files resolve relative paths **differently**:

| file | paths resolve against | so `include`/`exclude` are written as |
| --- | --- | --- |
| `.dev-config/ruff.toml` | the project root | `dataset`, `.venv`, … |
| `.dev-config/pyrightconfig.json` | **its own directory** | `../dataset`, `../.venv`, … |

If you drop the `../` prefixes in the Pyright config it will match nothing, analyze
**zero files, and still exit 0** — a passing check that verified nothing. Keep the
prefixes when editing that file.
