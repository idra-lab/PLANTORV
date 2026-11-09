# Point-cloud semantic segmentation

This guide covers giving every 3D point of a Femto Mega RGB-D frame a semantic class.
There are two independent ways to do it. Pick one, then follow only that section.

|                    | Route A: Open3D-ML                              | Route B: Pointcept                                    |
|--------------------|-------------------------------------------------|-------------------------------------------------------|
| Models             | RandLA-Net or Point Transformer v1, S3DIS       | Point Transformer V3 (PTv3), S3DIS                    |
| Environment        | plantorv `.venv`                                | `.venv` to export, conda env to run the model         |
| Input              | one RGB + depth image pair                      | whole dataset folders exported to Pointcept's format  |
| Output             | label image, confidence image, coloured `.ply`  | label image, confidence, coloured `.ply`, `summary.csv` |
| Speed (RTX 4070 Ti)| seconds per frame                               | ~0.2 s per frame single pass; ~47 s for the full test |
| Status             | works, but see [known problems](#known-problems) | works, but see [known problems](#known-problems)      |

Both routes use models trained on **S3DIS**, which has 13 indoor room classes: `ceiling`,
`floor`, `wall`, `beam`, `column`, `window`, `door`, `table`, `chair`, `sofa`, `bookcase`,
`board`, `clutter`. Plants have no class of their own: at best they come out as
`clutter`.

- [Which environment runs what](#which-environment-runs-what)
- [Route A: Open3D-ML](#route-a-open3d-ml)
- [Route B: Pointcept PTv3](#route-b-pointcept-ptv3)
  - [B1. Create the Pointcept environment (once)](#b1-create-the-pointcept-environment-once)
  - [B2. Download the PTv3 weights (once)](#b2-download-the-ptv3-weights-once)
  - [B3. Export your frames](#b3-export-your-frames)
  - [B4. Run the model](#b4-run-the-model)
  - [B5. Use the predictions](#b5-use-the-predictions)
- [Known problems](#known-problems)
- [Troubleshooting the Pointcept install](#troubleshooting-the-pointcept-install)
- [File map](#file-map)

## Which environment runs what

| Step                                 | Environment                          | Activate with                                   |
|--------------------------------------|--------------------------------------|-------------------------------------------------|
| Route A (`segment_pcd.py`)           | plantorv `.venv`                     | `source .venv/bin/activate`                     |
| Route B export (`export_pointcept_scenes.py`) | plantorv `.venv`            | `source .venv/bin/activate`                     |
| Route B model run (`segment_pointcept_scenes.py` or `tools/test.py`) | conda `pointcept-torch2.5.0-cu12.4` | `conda activate pointcept-torch2.5.0-cu12.4` |

The two environments don't mix: `.venv` has no Pointcept CUDA extensions, and the conda
env has no plantorv camera calibration. Always set `PYTHONNOUSERSITE=1` when using the
conda env. Otherwise Python picks up the pip in `~/.local/lib/python3.10/site-packages`
ahead of the environment's own.

## Route A: Open3D-ML

Run from the repository root in `.venv`.

```bash
python3 scripts/install_models.py pointtransformer_s3dis   # or randlanet_s3dis
python3 segment_pcd.py \
  dataset/rgb/rgb_dataset_1.png \
  dataset/depth/depth_dataset_1.png \
  --config models/point_transformer/pointtransformer_s3dis.yml \
  --checkpoint models/point_transformer/pointtransformer_s3dis_202109241350utc.pth \
  --device gpu \
  --semantic-output results/pointtransformer_semantic.ply \
  --labels-output results/pointtransformer_labels.npy \
  --confidence-output results/pointtransformer_confidence.npy \
  --no-visualize
```

For RandLA-Net, swap in `models/randla_net/randlanet_s3dis.yml` and
`randlanet_s3dis_202201071330utc.pth`; it also runs with `--device cpu`. The other flags
are described under *Point-cloud semantic segmentation* in the [main README](README.md).

Before trusting these results, read [known problem 4](#known-problems): the z-up rotation
in this script turns the cloud upside down.

## Route B: Pointcept PTv3

Steps B1 and B2 are done once. B3 to B5 are repeated for every dataset. On this machine,
B1 and B2 are already done, and B3 has been run for `dataset/` and `dataset_rw/`.

### B1. Create the Pointcept environment (once)

Pointcept lives in `models/Pointcept`, a clone of
[Pointcept/Pointcept](https://github.com/Pointcept/Pointcept) at commit `1342eda`.
Upstream's `environment.yml` doesn't install as is. The copy in `models/Pointcept` has
these fixes:

- **CUDA channel:** changed to `nvidia/label/cuda-12.4.1`. The upstream
  `nvidia/label/cuda-12.4.4` channel is empty.
- **CUDA pins:** added `cuda=12.4.1` and `cuda-version=12.4`. Without them, cuDNN 9.26 pulls
  in CUDA 13 packages and compiling `pointops` fails with `nv/target: No such file`.
- **PyG packages:** `torch-cluster`, `torch-scatter` and `torch-sparse` point at prebuilt
  wheel URLs. pip silently ignores `-f` on a requirement line and tries to build from
  source instead.
- **flash-attn:** installed from a prebuilt wheel,
  `flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310`.
- **Local extensions:** `./libs/pointops` and `./libs/pointgroup_ops` are removed from the
  file and installed by hand below.

```bash
cd models/Pointcept
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate pointcept-torch2.5.0-cu12.4
PYTHONNOUSERSITE=1 pip install --no-build-isolation ./libs/pointops ./libs/pointgroup_ops
```

Check that it works, still inside `models/Pointcept`; every line should print `OK` and CUDA
should be available:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=./ python -c "
import importlib, torch
print('cuda', torch.cuda.is_available())
for m in ['pointops', 'pointgroup_ops', 'spconv.pytorch', 'flash_attn', 'torch_scatter', 'pointcept']:
    importlib.import_module(m); print('OK', m)"
```

A failed `conda env create` leaves a half-built environment behind. Remove it with
`conda env remove -n pointcept-torch2.5.0-cu12.4` before trying again.

### B2. Download the PTv3 weights (once)

Only the S3DIS RPE experiment is needed. Run in `.venv` from the repository root:

```bash
hf download Pointcept/PointTransformerV3 \
  s3dis-semseg-pt-v3m1-0-rpe/config.py \
  s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth \
  --revision 5e25cfaf759289b03db1f90f143b3dc5172929d3 \
  --cache-dir models
```

This writes `model_best.pth` (555 MB) to
`models/models--Pointcept--PointTransformerV3/snapshots/5e25cfaf.../s3dis-semseg-pt-v3m1-0-rpe/model/`,
which is the path the config expects. Pinning `--revision` keeps that path stable. The
other experiments in that repository (ScanNet, nuScenes, Waymo, PPT) aren't used, and
their folders can be deleted.

This model is `PT-v3m1` with 13 classes, relative position encoding on and FlashAttention
off. Its inputs are **colour and surface normals**, sampled on a 2 cm grid. It loads into
Pointcept v1.7.0 without any key conversion.

### B3. Export your frames

Pointcept can't read PNGs. It reads one folder of `.npy` arrays per scene.
`export_pointcept_scenes.py` builds those folders with the same point-cloud generator as
`segment_pcd.py`. Run in `.venv` from the repository root:

```bash
python3 export_pointcept_scenes.py dataset                  # 151 frames, ~36 s
python3 export_pointcept_scenes.py dataset_rw               # 4 frames
python3 export_pointcept_scenes.py dataset --frames 1 85 --split smoke   # a quick subset
```

The dataset folder must contain `rgb/` and `depth/`. Images are paired by the trailing
number in their filenames, so `rgb_dataset_85.png` pairs with `depth_dataset_85.png` and
`img_0.png` with `img_0.png`. The output goes to
`models/Pointcept/data/plantorv/<split>/frame_XXXX/`:

| File          | Contents                                                      | Read by Pointcept |
|---------------|---------------------------------------------------------------|-------------------|
| `coord.npy`   | float32 (N, 3), metres, z up                                   | yes               |
| `color.npy`   | uint8 (N, 3), RGB 0–255                                        | yes               |
| `normal.npy`  | float32 (N, 3), unit normals pointing towards the camera       | yes               |
| `pixel.npy`   | int32 (N, 2), (row, column) of each point in the aligned depth grid | no           |
| `meta.json`   | source images, grid size, depth settings, coordinate frame     | no                |

Options:

| Flag                  | Default                          | Meaning                                                  |
|-----------------------|----------------------------------|----------------------------------------------------------|
| `--split`             | name of the dataset folder       | folder under the output root; the config's `data.test.split` |
| `--output-root`       | `models/Pointcept/data/plantorv` | the config's `data_root`, seen from the repository root  |
| `--frames`            | all paired frames                | export only these frame numbers                          |
| `--camera-to-world`   | none (camera assumed level)      | recorded `camera_to_world_transform.yaml`, for true gravity-aligned z |
| `--depth-unit-scale`, `--depth-trunc` | `1.0`, `3.0` m    | same as `segment_pcd.py`                                  |
| `--normal-radius`, `--normal-max-nn`  | `0.05` m, `30`    | neighbourhood used for normal estimation                  |

### B4. Run the model

There are two ways to run it. Start with the fast one.

#### Fast: one pass per frame

Run in the conda env from the **repository root**:

```bash
conda activate pointcept-torch2.5.0-cu12.4
PYTHONNOUSERSITE=1 python segment_pointcept_scenes.py dataset      # 151 frames
PYTHONNOUSERSITE=1 python segment_pointcept_scenes.py dataset_rw   # 4 frames
```

`segment_pointcept_scenes.py` works the way Pointcept's own validation does: it keeps one
point per 2 cm voxel, runs the model once, and gives every point the prediction of its
voxel. The model runs in about 0.1–0.3 s per frame. It writes to
`results/pointcept/<split>/`:

| File                        | Contents                                                        |
|-----------------------------|-----------------------------------------------------------------|
| `frame_XXXX_pred.npy`       | int64 (N,), class id per point, same format as `tools/test.py`   |
| `frame_XXXX_confidence.npy` | float32 (N,), top softmax score per point                        |
| `frame_XXXX_labels.npy`     | int32 label image in the aligned depth grid, -1 without depth    |
| `frame_XXXX_semantic.ply`   | cloud coloured by class                                          |
| `summary.csv`               | share of points per class, one row per frame                     |

| Flag            | Default                                           | Meaning                                              |
|-----------------|---------------------------------------------------|------------------------------------------------------|
| `--frames`      | every scene in the split                          | only these frame numbers                             |
| `--output`      | `results/pointcept/<split>`                       | output directory                                     |
| `--predictions` | none (run the model)                              | skip the model, read `<frame>_pred.npy` from this folder ([B5](#b5-use-the-predictions)) |
| `--no-ply`      | write clouds                                      | skip the `.ply` files                                |
| `--config`, `--weight`, `--data-root` | the plantorv config and its values | use another Pointcept config or checkpoint      |
| `--seed`        | `0`                                               | which point represents each voxel                    |

Compared with the full test on frames 1 and 85, the fast pass gave the same label to
89–91% of points. Most points that differ are ones the model was unsure about: mean
confidence 0.37–0.39, against 0.67 where the two agree.

#### Full test: `tools/test.py`

This uses every point and 10 scale/flip variants per frame, so it's about 200× slower.
Run in the conda env from `models/Pointcept`:

```bash
cd models/Pointcept
conda activate pointcept-torch2.5.0-cu12.4
export PYTHONNOUSERSITE=1 PYTHONPATH=./

# dataset/  (split "dataset" is the config default)
python tools/test.py --config-file configs/plantorv/semseg-pt-v3m1-0-rpe-s3dis.py

# any other split: override the split and where results go
python tools/test.py --config-file configs/plantorv/semseg-pt-v3m1-0-rpe-s3dis.py \
  --options data.test.split=dataset_rw save_path=exp/plantorv/dataset_rw
```

The config
[`configs/plantorv/semseg-pt-v3m1-0-rpe-s3dis.py`](models/Pointcept/configs/plantorv/semseg-pt-v3m1-0-rpe-s3dis.py)
inherits the S3DIS PTv3 config unchanged and replaces only the test dataset.

What to expect:

- **Time:** about 47 s per frame, so about 2 hours for the 151 frames of `dataset/` (see [known problem 1](#known-problems)).
- **Resuming:** frames that already have a `_pred.npy` are skipped on a rerun, so an interrupted run picks up where it stopped. Delete the result folder to recompute.
- **Metrics:** the log prints `Accuracy 0.0000` and `mIoU nan`, plus a `Mean of empty slice` warning. Ignore all of it: your frames have no ground-truth labels.

### B5. Use the predictions

The fast pass already writes label images, coloured clouds and `summary.csv`. The full
test writes only `<save_path>/result/frame_XXXX_pred.npy`. To get the same outputs from
those without running the model again, run from the repository root:

```bash
PYTHONNOUSERSITE=1 python segment_pointcept_scenes.py dataset \
  --predictions models/Pointcept/exp/plantorv/semseg-pt-v3m1-0-rpe-s3dis/result \
  --output results/pointcept/dataset_fulltest
```

Every `frame_XXXX_pred.npy` is an int array with one S3DIS class id per point, in the same
order as `coord.npy`. To work with them directly, `pixel.npy` and `meta.json` hold what's
needed to put them into an image:

```python
import json
import numpy as np

names = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
         "table", "chair", "sofa", "bookcase", "board", "clutter"]
scene = "models/Pointcept/data/plantorv/dataset/frame_0085"
pred = np.load("models/Pointcept/exp/plantorv/semseg-pt-v3m1-0-rpe-s3dis/result/frame_0085_pred.npy")

pixel = np.load(f"{scene}/pixel.npy")
height, width = json.load(open(f"{scene}/meta.json"))["aligned_shape"]
label_image = np.full((height, width), -1, dtype=np.int32)   # -1 where there is no depth
label_image[pixel[:, 0], pixel[:, 1]] = pred

for class_id, count in zip(*np.unique(pred, return_counts=True)):
    print(f"{names[class_id]:10s} {100 * count / pred.size:5.1f}%")
```

The label image is in the **aligned depth grid** (`aligned_shape`, 640×360 for
`dataset/`), not at the RGB image's full resolution.

## Known problems

1. **The full test is slow.** RGB-D clouds are dense: up to about 188 points fall in one
   2 cm voxel. In test mode Pointcept runs one pass per point in the fullest voxel, repeated
   for 10 scale/flip variants, which is 1,884 forward passes and about 47 s per frame. Use
   `segment_pointcept_scenes.py` ([B4](#b4-run-the-model)) unless you need the last few
   percent of agreement.
2. **The exported clouds are tilted.** Without `--camera-to-world`, the camera is assumed
   level. The camera actually points downwards, so the table comes out tilted about
   15–17° from horizontal, and S3DIS models expect true z-up. Pass `--camera-to-world`
   only with a transform recorded for the pose the frames were captured from. The
   `camera_to_world_transform.yaml` in the repository root was recorded on 2026-09-14,
   after both `dataset/` (June) and `dataset_rw/` (2026-09-11) were captured.
3. **S3DIS classes don't fit tabletop scenes.** On frames 1 and 85, PTv3 labelled about
   72% of points `wall` and 15–21% `bookcase`. The table surface came out mostly
   `bookcase` and `floor`.
4. **`segment_pcd.py` turns clouds upside down (Route A).** `RGBDPointCloudGenerator`
   already negates y and z, but `camera_to_z_up()` assumes the raw optical frame
   (x right, y down, z forward). Together they map camera-down to +z and camera-forward
   to −x, so Open3D-ML gets an upside-down cloud. `export_pointcept_scenes.py` undoes the
   generator's flip first and doesn't have this problem. `segment_pcd.py` hasn't been
   changed.
5. **Part of the Route B setup isn't in git.** The fixed `models/Pointcept/environment.yml`
   and `models/Pointcept/configs/plantorv/` sit inside the Pointcept clone, and plantorv
   ignores `models/*`. A fresh clone of either repository won't have them.

## Troubleshooting the Pointcept install

| Error                                                          | Cause                                                        | Fix                                                     |
|----------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------|
| `ModuleNotFoundError: No module named 'torch'` while building a package | pip builds in an isolated environment without torch | use a prebuilt wheel, or `pip install --no-build-isolation` |
| same error for `torch-cluster` / `torch-scatter` / `torch-sparse` | `-f <url>` on a requirement line is ignored              | pin the direct wheel URLs (already in `environment.yml`) |
| traceback paths under `~/.local/lib/python3.10/site-packages/pip` | the user-site pip shadows the environment's pip        | prefix commands with `PYTHONNOUSERSITE=1`               |
| `fatal error: nv/target: No such file or directory`            | CUDA 13 packages mixed with nvcc 12.4                        | `cuda=12.4.1` + `cuda-version=12.4`, then recreate the env |
| `undefined symbol` when importing `flash_attn`                 | wheel C++ ABI doesn't match torch                            | check `torch._C._GLIBCXX_USE_CXX11_ABI`, pick the `cxx11abiTRUE`/`FALSE` wheel to match |
| `[PointROPE] CUDA implementation unavailable`                  | optional extension for LitePT only                           | ignore for PTv3                                         |

## File map

| Path                                                               | What it is                                             |
|--------------------------------------------------------------------|--------------------------------------------------------|
| `segment_pcd.py`                                                   | Route A: RGB-D pair → Open3D-ML segmentation           |
| `export_pointcept_scenes.py`                                       | Route B: RGB-D dataset → Pointcept scene folders        |
| `segment_pointcept_scenes.py`                                      | Route B: fast single-pass run, or outputs from existing predictions |
| `results/pointcept/<split>/`                                       | Route B outputs of `segment_pointcept_scenes.py`        |
| `mapping/rgbd_pointcloud.py`                                       | point-cloud generator shared by both routes            |
| `models/point_transformer/`, `models/randla_net/`                  | Route A weights and configs                            |
| `models/Pointcept/`                                                | Pointcept clone (untracked)                            |
| `models/Pointcept/environment.yml`                                 | fixed conda environment (untracked edit)               |
| `models/Pointcept/configs/plantorv/semseg-pt-v3m1-0-rpe-s3dis.py`  | Route B test config (untracked)                        |
| `models/models--Pointcept--PointTransformerV3/`                    | Route B weights (Hugging Face cache layout)            |
| `models/Pointcept/data/plantorv/<split>/`                          | exported scenes                                        |
| `models/Pointcept/exp/plantorv/<run>/result/`                      | Route B predictions and `test.log`                      |
