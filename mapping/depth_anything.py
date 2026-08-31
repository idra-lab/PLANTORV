"""Depth Anything metric-depth providers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np

from mapping.depth_provider import DepthResult, _normalize_depth, _validate_color

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_V3_MODEL_ID = "depth-anything/da3metric-large"

# Mean of fx and fy in aruco/camera.yaml for the 1920x1080 Femto RGB stream.
DEFAULT_FOCAL_LENGTH_PX = (1144.083 + 1132.134) / 2.0


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


class DepthAnythingV3Provider:
    """Infer monocular metric depth with Depth Anything 3.

    ``DA3METRIC-LARGE`` predicts focal-normalized depth. Its documented metric
    conversion is ``depth_m = focal_px * prediction / 300``. The focal length
    is scaled to the model's processing resolution before the conversion, and
    the result is then resized back to the input RGB resolution.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_V3_MODEL_ID,
        *,
        focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
        process_res: int = 504,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if focal_length_px <= 0.0:
            raise ValueError("focal_length_px must be positive")
        if process_res <= 0:
            raise ValueError("process_res must be positive")
        if model is None and "metric" not in model_id.lower():
            raise ValueError("DepthAnythingV3Provider requires a DA3 metric checkpoint")

        self.model_id = model_id
        self.focal_length_px = float(focal_length_px)
        self.process_res = process_res
        self.device = device or _default_device()

        if model is None:
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError as exc:
                raise ImportError(
                    "Depth Anything 3 support requires the depth-anything-3 package; "
                    "see the Monocular depth section in README.md"
                ) from exc
            model = DepthAnything3.from_pretrained(model_id)

        self._model: Any = model
        if hasattr(self._model, "to"):
            self._model.to(device=self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def estimate(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray | None = None,
    ) -> DepthResult:
        """Estimate RGB-aligned metric depth and return it in millimetres."""
        _validate_color(color_bgr)
        if depth_raw is not None:
            raise ValueError("DepthAnythingV3Provider does not accept sensor depth")

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        prediction = self._model.inference(
            [color_rgb],
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
        )

        raw_depth = _single_prediction_map(prediction.depth, "depth")
        input_h, input_w = color_bgr.shape[:2]
        predicted_h, predicted_w = raw_depth.shape
        focal_scale = 0.5 * (predicted_w / input_w + predicted_h / input_h)
        processed_focal_px = self.focal_length_px * focal_scale
        depth_mm = raw_depth * (processed_focal_px / 300.0) * 1000.0

        target_size = (input_w, input_h)
        if depth_mm.shape != (input_h, input_w):
            depth_mm = cv2.resize(depth_mm, target_size, interpolation=cv2.INTER_LINEAR)

        sky = getattr(prediction, "sky", None)
        if sky is not None:
            sky_mask = _single_prediction_map(sky, "sky").astype(bool)
            if sky_mask.shape != (input_h, input_w):
                sky_mask = cv2.resize(
                    sky_mask.astype(np.uint8),
                    target_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            depth_mm[sky_mask] = 0.0

        depth_mm = _normalize_depth(depth_mm)
        confidence = _resize_optional_prediction(
            getattr(prediction, "conf", None), target_size, "confidence"
        )
        return DepthResult(
            depth_mm=depth_mm,
            valid_mask=depth_mm > 0.0,
            confidence=confidence,
            metric=True,
            metadata={
                "source": "depth_anything_v3",
                "model_id": self.model_id,
                "device": self.device,
                "focal_length_px": self.focal_length_px,
                "process_res": self.process_res,
                "output_units": "mm",
            },
        )


def _single_prediction_map(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 3 or result.shape[0] != 1:
        raise RuntimeError(
            f"Depth Anything 3 returned {name} shape {result.shape}; expected (1, H, W)"
        )
    return np.ascontiguousarray(result[0])


def _resize_optional_prediction(
    value: Any | None,
    target_size: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    result = _single_prediction_map(value, name)
    target_w, target_h = target_size
    if result.shape != (target_h, target_w):
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(result, dtype=np.float32)


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
