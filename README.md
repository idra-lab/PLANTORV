# PLANTORV

Pipeline for segmenting tabletop RGB-D scenes, labeling detected objects with Azure OpenAI, projecting depth coordinates, and evaluating results against ArUco marker annotations.

## Code Structure

- `samgpt.py` - main pipeline: runs SAM segmentation, GPT labeling, RGB-D coordinate mapping, and writes JSON outputs.
- `segmentation/` - Segment Anything wrapper and mask post-processing.
- `scene_understanding/` - Azure OpenAI image annotation logic.
- `mapping/` - RGB-D camera calibration and depth-to-color projection utilities.
- `aruco/` - ArUco marker detection, pose estimation, camera/config YAML files, and marker generation.
- `evaluation/` - matching, metrics, depth correlation, and result visualization.
- `dataset/` - expected input images: `rgb/`, `depth/`, and `rgb_aruco/`.
- `models/` - local model checkpoints, including the SAM checkpoint expected at `models/sam/sam_vit_h_4b8939.pth`.
- `outputs_json_labeled/`, `output_aruco/`, `results/`, `ppt_outputs/` - generated pipeline outputs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the Azure OpenAI credentials used by `samgpt.py`:

```bash
AZURE_ENDPOINT=...
AZURE_API_KEY=...
```

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
qsub pbs_script.sh
qsub aruco_detector.sh
qsub verification_script.sh
```

Clean generated outputs if you want a fresh run:

```bash
rm -rf outputs_json_labeled output_aruco results ppt_outputs
```
