from pathlib import Path

import pytest
import yaml

from pipeline.provenance import write_run_config
from segmentation.fastsam_model import FastSAMModel
from segmentation.sam_model import SAMModel
from segmentation.segmentation_factory import (
    create_segmenter,
    infer_model,
    normalize_model,
    resolve_class,
)


def _config(tmp_path: Path, **keys: object) -> Path:
    """Write a segmentation configuration file and return its path."""
    path = tmp_path / "segmenter.yaml"
    path.write_text(yaml.safe_dump(keys))

    return path


def test_the_backend_is_read_from_the_model_key() -> None:
    assert infer_model({"MODEL": "SAM-2"}) == "sam"
    assert infer_model({"MODEL": "fast_sam"}) == "fastsam"
    assert infer_model({"MODEL": "sam3"}) == "sam3"


def test_the_backend_falls_back_to_the_checkpoint_name() -> None:
    assert infer_model({"CHECKPOINT": "models/fastsam/FastSAM-s.pt"}) == "fastsam"
    assert infer_model({"CHECKPOINT": "models/sam/sam3.pt"}) == "sam3"
    assert infer_model({"CHECKPOINT": "models/sam/sam2.1_l.pt"}) == "sam"
    assert infer_model({"CHECKPOINT": "models/sam/mobile_sam.pt"}) == "sam"


def test_a_config_naming_neither_is_rejected() -> None:
    with pytest.raises(ValueError, match="neither MODEL nor CHECKPOINT"):
        infer_model({"SEGMENTATION_CONFIG": {"points_stride": 12}})


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported segmentation backend"):
        normalize_model("sam4")


def test_each_backend_resolves_to_its_class() -> None:
    assert resolve_class("sam") is SAMModel
    assert resolve_class("fastsam") is FastSAMModel


def test_a_config_without_a_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing CHECKPOINT"):
        create_segmenter(_config(tmp_path, MODEL="sam"))


def test_a_misspelt_filter_is_rejected_rather_than_ignored() -> None:
    # A filter silently dropped would be recorded as configured while changing nothing,
    # which is exactly what the written configuration is meant to rule out.
    with pytest.raises(ValueError, match="has no filter object_min_are_pct"):
        SAMModel.merge_filters({"object_min_are_pct": 1.0})


def test_configured_filters_win_over_the_defaults() -> None:
    merged = SAMModel.merge_filters({"object_min_area_pct": 1.0})

    assert merged["object_min_area_pct"] == 1.0
    assert merged["object_max_area_pct"] == SAMModel.DEFAULT_FILTERS["object_max_area_pct"]


def test_every_shipped_config_names_a_backend_and_a_checkpoint() -> None:
    for path in sorted(
        (Path(__file__).resolve().parent.parent / "segmentation" / "conf").glob("*.yaml")
    ):
        config = yaml.safe_load(path.read_text())

        assert infer_model(config) in ("sam", "fastsam", "sam3"), path
        assert config.get("CHECKPOINT"), path

        backend = resolve_class(infer_model(config))
        # Raises when the file carries a threshold the backend does not have.
        backend.merge_filters(config.get("FILTERS") or {})


def test_the_written_record_holds_the_stage_and_the_values(tmp_path: Path) -> None:
    path = write_run_config(
        tmp_path,
        "segmentation",
        {"SOURCE_CONFIG": Path("segmentation/conf/sam21_l.yaml"), "FILTERS": {"a": 0.5}},
        device="cuda",
    )
    written = yaml.safe_load(path.read_text())

    assert path.name == "segmentation_config.yaml"
    assert written["STAGE"] == "segmentation"
    assert written["DEVICE"] == "cuda"
    # Paths and NumPy scalars have to survive the dump; nothing reads this file back,
    # but a crash while writing it would take the run down with it.
    assert written["SOURCE_CONFIG"] == "segmentation/conf/sam21_l.yaml"
    assert written["FILTERS"] == {"a": 0.5}
    assert "RUN_AT" in written


def test_sam3_receives_the_vlm_before_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SAM3Model takes `llm` as its first positional parameter, so passing the checkpoint
    # positionally and the VLM by keyword binds the checkpoint to `llm` and raises.
    recorded = {}

    class _FakeBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded["args"] = args
            recorded["kwargs"] = kwargs

    import segmentation.segmentation_factory as factory

    monkeypatch.setattr(factory, "resolve_class", lambda model: _FakeBackend)
    monkeypatch.setattr(factory, "create_llm", lambda path: f"llm<{path}>")

    factory.create_segmenter(
        _config(
            tmp_path,
            MODEL="sam3",
            CHECKPOINT="models/sam/sam3.pt",
            LLM_CONFIG_FILE="LLM/conf/azure_gpt52.yaml",
        )
    )

    assert recorded["args"] == ("llm<LLM/conf/azure_gpt52.yaml>", "models/sam/sam3.pt")
    assert "llm" not in recorded["kwargs"]


def test_the_other_backends_take_only_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = {}

    class _FakeBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            recorded["args"] = args

    import segmentation.segmentation_factory as factory

    monkeypatch.setattr(factory, "resolve_class", lambda model: _FakeBackend)

    factory.create_segmenter(_config(tmp_path, MODEL="sam", CHECKPOINT="models/sam/sam_l.pt"))

    assert recorded["args"] == ("models/sam/sam_l.pt",)
