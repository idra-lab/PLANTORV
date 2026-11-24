# Tests

Unit tests run with `pytest` and are fully headless: no GPU, model checkpoint or
network access is needed (models and APIs are bypassed or faked).

## Run

```bash
make test                                   # whole suite
.venv/bin/python -m pytest -q tests/test_depth.py   # one file
```

`test_pointcloud_segmentation.py` needs Open3D-ML, whose PyTorch ops only load
under torch 2.2.*; in the main `.venv` (torch 2.5) two of its tests fail with
"Version mismatch: Open3D needs PyTorch version 2.2.*".

## Coverage

- **test_depth.py**: Femto camera model and depth-to-RGB alignment in `mapping`
  (calibration lookup, distortion round trip, alignment, neighbourhood depth
  lookup, `main_coords` end to end on synthetic PNGs).
- **test_depth_pipeline.py**: depth providers (sensor, Depth Anything V2/V3),
  object depth association, back-projection, camera-to-world transform, point clouds.
- **test_segmentation_config.py**: segmentation factory and YAML provenance.
- **test_pipeline_artifacts.py**: the on-disk contract between pipeline stages.
- **test_huggingface_images.py**: multimodal detection in the Hugging Face LLM backend.
- **test_json_serialization.py**: `utility.json_serialization.to_json_compatible`.
- **test_pointcloud_segmentation.py**: the Open3D-ML point-cloud segmentation interface.
