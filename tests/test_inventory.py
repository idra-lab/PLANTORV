"""Tests for the VLM scene inventory that feeds SAM 3 (no network, no weights)."""

from pathlib import Path
from typing import Any, List, Tuple

import pytest
from PIL import Image

from scene_understanding.inventory import NO_TASK_NOTE, ConceptInventory


class _FakeLLM:
    """Answers every query with a fixed string and records what it was sent."""

    def __init__(self, answer: str, success: bool = True, supports_images: bool = True) -> None:
        self.SUPPORTS_IMAGES = supports_images
        self.answer = answer
        self.success = success
        self.sent: List[Tuple[str, Any]] = []
        self.closed = False

    def query(self, prompt: str, images: Any = None) -> Tuple[bool, str]:
        self.sent.append((prompt, images))
        return self.success, self.answer

    def close(self) -> None:
        self.closed = True


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8))


def test_concepts_are_read_from_the_vlm_answer() -> None:
    llm = _FakeLLM(
        '{"objects": ["red block", " red block", "blue tower", ""], '
        '"task_relevant": ["blue tower", "green cup"]}'
    )
    inventory = ConceptInventory(llm, task="stack the blocks")  # type: ignore[arg-type]

    objects, task_relevant = inventory.concepts_for(_image())

    # Deduplicated and stripped, order kept; a task-relevant phrase that is not an
    # object is dropped, since it was never segmented.
    assert objects == ["red block", "blue tower"]
    assert task_relevant == ["blue tower"]
    prompt, images = llm.sent[0]
    assert "stack the blocks" in prompt
    assert "%%TASK%%" not in prompt and "%%EXAMPLES%%" not in prompt
    assert len(images) == 1


@pytest.mark.parametrize(
    "answer, success",
    [("not json", True), ('["a list"]', True), ('{"objects": ["x"]}', False), ("", True)],
)
def test_an_unusable_answer_is_an_empty_scene(answer: str, success: bool) -> None:
    inventory = ConceptInventory(_FakeLLM(answer, success))  # type: ignore[arg-type]

    assert inventory.concepts_for(_image()) == ([], [])


def test_without_a_task_the_prompt_says_so() -> None:
    inventory = ConceptInventory(_FakeLLM("{}"), examples_file=None)  # type: ignore[arg-type]

    assert NO_TASK_NOTE in inventory.build_prompt()
    assert inventory.examples == ""


def test_the_default_examples_are_rendered_into_the_prompt() -> None:
    inventory = ConceptInventory(_FakeLLM("{}"))  # type: ignore[arg-type]

    assert inventory.examples.startswith("Phrases that work")
    assert inventory.examples in inventory.build_prompt()


def test_a_vlm_without_images_is_rejected() -> None:
    llm = _FakeLLM("{}", supports_images=False)

    with pytest.raises(ValueError, match="does not support images"):
        ConceptInventory(llm)  # type: ignore[arg-type]


def test_the_sent_prompt_is_saved_when_a_save_dir_is_set(tmp_path: Path) -> None:
    inventory = ConceptInventory(_FakeLLM('{"objects": []}'), save_dir=tmp_path)  # type: ignore[arg-type]

    inventory.concepts_for(_image())

    assert len(list(tmp_path.glob("sent_query_*.txt"))) == 1


def test_narrow_keeps_only_replacements_of_ambiguous_phrases() -> None:
    llm = _FakeLLM('{"replacements": {"lego block": ["red brick", "lego block"], "cup": ["mug"]}}')
    inventory = ConceptInventory(llm)  # type: ignore[arg-type]

    assert inventory.narrow(_image(), {"lego block": 3}) == {"lego block": ["red brick"]}


def test_close_releases_the_vlm() -> None:
    llm = _FakeLLM("{}")
    ConceptInventory(llm).close()  # type: ignore[arg-type]

    assert llm.closed
