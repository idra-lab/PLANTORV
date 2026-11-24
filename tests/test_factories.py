"""Tests for the depth and annotator factories (no model weights loaded)."""

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from depth import depth_factory
from scene_understanding import annotator_factory


def test_sensor_depth_source_has_no_provider() -> None:
    assert depth_factory.create_depth_provider("sensor") is None


def test_unknown_depth_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown depth source"):
        depth_factory.create_depth_provider("lidar")


def test_monocular_source_builds_depth_anything_3_with_its_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: Dict[str, Any] = {}
    monkeypatch.setattr(
        depth_factory, "DepthAnythingV3Provider", lambda **kwargs: built.update(kwargs) or "da3"
    )

    provider = depth_factory.create_depth_provider("monocular", device="cpu", process_res=256)

    assert provider == "da3"
    assert built == {
        "model_id": depth_factory.DEFAULT_V3_MODEL_ID,
        "focal_length_px": depth_factory.DEFAULT_FOCAL_LENGTH_PX,
        "process_res": 256,
        "device": "cpu",
    }


def test_unknown_annotator_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown annotator"):
        annotator_factory.create_annotator("clip")


def test_describe_reads_the_dam_settings_from_the_annotator() -> None:
    dam = SimpleNamespace(
        model_path="models/dam/DAM-3B",
        query="<image> Describe",
        temperature=0.6,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=512,
    )

    described = annotator_factory.describe(dam, "dam")  # type: ignore[arg-type]

    assert described == {
        "ANNOTATOR": "dam",
        "DAM_MODEL": "models/dam/DAM-3B",
        "DAM_CONFIG": {"temperature": 0.6, "top_p": 0.5, "num_beams": 1, "max_new_tokens": 512},
        "QUERY": "<image> Describe",
    }
