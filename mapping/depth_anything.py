"""Depth Anything V2 metric-depth provider."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np

from mapping.depth_provider import DepthResult, _normalize_depth, _validate_color

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthAnythingV2Provider:
    """Infer indoor metric depth using a Transformers-compatible checkpoint.

    The model is loaded once during construction. Transformers is an optional
    dependency so sensor-only use of PLANTORV remains unchanged.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()

        if (processor is None) != (model is None):
            raise ValueError("processor and model must either both be supplied or both omitted")
        if processor is None:
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            except ImportError as exc:
                raise ImportError(
                    "Depth Anything support requires the 'depth-anything' optional "
                    "dependencies: pip install -e '.[depth-anything]'"
                ) from exc
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)

        assert processor is not None and model is not None
        self._processor: Any = processor
        self._model: Any = model
        config = getattr(self._model, "config", None)
        estimation_type = getattr(config, "depth_estimation_type", None)
        if estimation_type is not None and estimation_type != "metric":
            raise ValueError(f"Checkpoint {model_id!r} is {estimation_type!r}, not metric depth")
        if hasattr(self._model, "to"):
            self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def estimate(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray | None = None,
    ) -> DepthResult:
        """Estimate an RGB-aligned depth map and convert metres to millimetres."""
        _validate_color(color_bgr)
        if depth_raw is not None:
            raise ValueError("DepthAnythingV2Provider does not accept sensor depth")

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=color_rgb, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, dict):
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }

        try:
            import torch

            inference_context = torch.inference_mode()
        except ImportError:
            # Useful for lightweight injected test doubles. A real model requires torch.
            inference_context = nullcontext()

        with inference_context:
            outputs = self._model(**inputs)

        target_size = color_bgr.shape[:2]
        post_processed = self._processor.post_process_depth_estimation(
            outputs,
            target_sizes=[target_size],
        )
        predicted_depth = post_processed[0]["predicted_depth"]
        if hasattr(predicted_depth, "detach"):
            predicted_depth = predicted_depth.detach()
        if hasattr(predicted_depth, "cpu"):
            predicted_depth = predicted_depth.cpu()
        if hasattr(predicted_depth, "numpy"):
            predicted_depth = predicted_depth.numpy()

        depth_m = np.asarray(predicted_depth, dtype=np.float32).squeeze()
        if depth_m.shape != target_size:
            raise RuntimeError(
                f"Depth Anything returned shape {depth_m.shape}; expected {target_size}"
            )
        depth_mm = _normalize_depth(depth_m * 1000.0)
        return DepthResult(
            depth_mm=depth_mm,
            valid_mask=depth_mm > 0.0,
            metric=True,
            metadata={
                "source": "depth_anything_v2",
                "model_id": self.model_id,
                "device": self.device,
                "output_units": "mm",
            },
        )


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
