from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from LLM.LLMHuggingFace.LLMHuggingFace import LLMHuggingFace, _looks_multimodal


def _backend(**config: object) -> LLMHuggingFace:
    """Build the backend without loading any weights.

    SUPPORTS_IMAGES is answered from the configuration here, so `_setup` never reaches
    the network, and the model itself is only loaded by `connect()`.
    """
    return LLMHuggingFace(model="fake/model", config=dict(config))


def test_a_vision_tower_marks_the_model_multimodal() -> None:
    assert _looks_multimodal(SimpleNamespace(vision_config={"hidden_size": 8}))
    assert _looks_multimodal(SimpleNamespace(image_token_id=151655))


def test_an_image_text_architecture_marks_the_model_multimodal() -> None:
    assert _looks_multimodal(SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]))
    assert _looks_multimodal(SimpleNamespace(architectures=["Idefics3ForConditionalGeneration"]))


def test_a_causal_language_model_is_text_only() -> None:
    assert not _looks_multimodal(SimpleNamespace(architectures=["LlamaForCausalLM"]))
    assert not _looks_multimodal(SimpleNamespace())


def test_the_configuration_can_answer_the_question_outright() -> None:
    # The escape hatch for a model whose config cannot be reached, e.g. offline.
    assert _backend(SUPPORTS_IMAGES=True).SUPPORTS_IMAGES
    assert not _backend(SUPPORTS_IMAGES=False).SUPPORTS_IMAGES


def test_a_text_only_model_refuses_an_image(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(path)
    llm = _backend(SUPPORTS_IMAGES=False)

    with pytest.raises(NotImplementedError, match="does not support images"):
        llm.image_part(path)

    # query() refuses before reaching any backend code at all.
    with pytest.raises(NotImplementedError):
        llm.query("describe", images=[path])


def test_an_image_becomes_a_content_part(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8)).save(path)
    llm = _backend(SUPPORTS_IMAGES=True)

    part = llm.image_part(path)

    assert part["type"] == "image"
    # Kept as an image rather than base64: the processor reads it when it renders the
    # chat template, which is what puts the image tokens in the prompt.
    assert isinstance(part["image"], Image.Image)
    assert part["image"].mode == "RGB"


def test_messages_are_recognised_as_carrying_images(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(path)
    llm = _backend(SUPPORTS_IMAGES=True)

    with_image = llm.build_messages("what is this?", images=[path])
    without = llm.build_messages("what is this?")

    assert llm._carries_images(with_image)
    assert not llm._carries_images(without)


def test_plain_text_is_wrapped_as_a_text_part() -> None:
    shaped = LLMHuggingFace._as_template_messages(
        [{"role": "user", "content": "hello"}],
    )

    assert shaped == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_structured_content_reaches_the_template_unflattened(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(path)
    llm = _backend(SUPPORTS_IMAGES=True)

    shaped = LLMHuggingFace._as_template_messages(llm.build_messages("what?", images=[path]))
    parts = shaped[0]["content"]

    # The text path drops non-text parts; this one must not.
    assert [part["type"] for part in parts] == ["text", "image"]
