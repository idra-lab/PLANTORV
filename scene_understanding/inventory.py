"""Scene inventory: ask a VLM which things a scene holds, as phrases SAM 3 can ground.

This is the first half of concept segmentation (see
:class:`segmentation.sam3_model.SAM3Model`, the second half). A VLM -- any
:class:`~LLM.llm_base.BaseLLM` that accepts images -- is shown the whole scene and
asked for the things in it as short noun phrases. When a task is configured it is
also asked which of those the task needs. SAM 3 then segments exactly those phrases.

Mask quality is governed by how the phrases are worded, so the prompt and the
few-shot phrasing pairs of :data:`DEFAULT_EXAMPLES_FILE` live here, next to the call
that uses them. Describing a mask once it has been cut out is a separate job, done by
the annotators of this package.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from PIL import Image

from LLM.llm_base import BaseLLM
from LLM.llm_factory import create_llm
from utility.utility import logger

# Few-shot pairs steering how the VLM phrases the concepts. Data rather than
# code, so the phrasing can be tuned without touching this file.
DEFAULT_EXAMPLES_FILE = Path(__file__).resolve().parent / "conf" / "concept_prompts.yaml"

# STAGE 1 PROMPT: SCENE INVENTORY.
#
# Asks for the noun phrases SAM 3 is prompted with, rather than for a description
# of anything. `%%EXAMPLES%%` is replaced by the pairs rendered from
# `DEFAULT_EXAMPLES_FILE`, and `%%TASK%%` by the task the robot has to carry out.
SCENE_INVENTORY_PROMPT = """You will receive one image of a robotic workspace.

List the things visible in it as short noun phrases. Those phrases are passed to
a concept segmentation model (SAM3), which finds the things you name and nothing 
else: whatever you leave out is not segmented and does not reach the rest of the
pipeline.

Name each thing at the granularity the work happens at: the whole object or
assembly, never the pieces it is built from. If several identical pieces are put
together into one structure, name the structure. Do not name a part of something
you have already named.

Make each phrase specific enough that only one kind of thing in the scene can
match it, and specific enough to settle what counts as *one* of them. Colour,
size and overall shape are what to use for that.

Never make a phrase specific by saying where a thing is, which one of them you
mean, or how many there are. Every region matching a phrase is returned
automatically, so position and order single out nothing and ground against
nothing. The same goes for what a thing happens to be doing right now, which will
have changed by the next frame.

%%EXAMPLES%%

What to leave out:
- The work surface, the walls, the floor, and anything else that is scenery
  rather than a thing in the scene.

%%TASK%%

Return ONLY raw JSON.
Do not use markdown code fences.
Do not write ```json.
{
"objects": ["string"],
"task_relevant": ["string"]
}

- "objects" lists every visible thing that is not background, whether or not a
  task was given.
- "task_relevant" lists the objects needed to carry out the task, copied
  verbatim from "objects". Do not rephrase them, and do not name anything that is
  not already in "objects". Leave it empty when no task was given. 
"""

# Rendered into `%%TASK%%` when no task is configured.
NO_TASK_NOTE = 'No task is given, so leave "task_relevant" empty.'

# REFINEMENT PROMPT.
#
# Sent when a phrase from the inventory matched several regions at once, which is
# the symptom of it not having settled what counts as one thing. `%%AMBIGUOUS%%`
# is replaced by the offending phrases and how many regions each matched, and
# `%%EXAMPLES%%` by the same pairs the inventory prompt uses.
REFINEMENT_PROMPT = """You named the things in this workspace earlier, and some of
those phrases turned out to be ambiguous.

Each phrase below matched several separate regions of the image, which means it
did not settle what counts as one thing:

%%AMBIGUOUS%%

A phrase does that when it names a part rather than a whole. "lego block" matches
a single brick, a stack of two bricks, and the whole tower they are built into,
and nothing in the phrase says which of those was meant.

Replace each of them with one or more phrases that each name a whole object or
assembly, specific enough that only one kind of thing in the scene can match.
Use colour, size and overall shape. Do not say where a thing is, which one of
them you mean, or how many there are.

%%EXAMPLES%%

Return ONLY raw JSON.
Do not use markdown code fences.
Do not write ```json.
{
"replacements": {"the ambiguous phrase": ["a narrower phrase", "another one"]}
}

Every ambiguous phrase must appear as a key. Give at least one replacement for
each, and several when the phrase was covering several different structures.
"""


class ConceptInventory:
    """Ask a VLM for the things in a scene, as phrases for SAM 3 to segment.

    Attributes
    ----------
    llm : BaseLLM
        Backend asked for the scene inventory. Must accept images.
    task : str or None
        The task the robot has to carry out, if any.
    prompt : str
        Scene inventory instructions, with ``%%EXAMPLES%%`` and ``%%TASK%%`` sentinels.
    refinement_prompt : str
        Instructions used by the parked refinement loop of ``SAM3Model``.
    examples : str
        The few-shot phrasing pairs, rendered as they are injected in the prompts.
    save_dir : Path or None
        Where every prompt sent is written, for inspection. None writes nothing.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        task: Optional[str] = None,
        prompt: str = SCENE_INVENTORY_PROMPT,
        refinement_prompt: str = REFINEMENT_PROMPT,
        examples_file: Optional[Union[str, Path]] = DEFAULT_EXAMPLES_FILE,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the inventory.

        Parameters
        ----------
        llm : BaseLLM
            Backend used to name the things in the scene. It must support images,
            which is the case for the Azure OpenAI, OpenAI, Anthropic, Gemini and
            GLM backends.
        task : str or None
            The task the robot has to carry out. When given, the VLM also reports
            which objects it needs. It never restricts what gets segmented: the full
            inventory is always returned, because an object irrelevant to the task
            can still be an obstacle.
        prompt : str
            Instructions sent with the scene image. Defaults to
            :data:`SCENE_INVENTORY_PROMPT`.
        refinement_prompt : str
            Instructions sent when narrowing an ambiguous phrase. Defaults to
            :data:`REFINEMENT_PROMPT`.
        examples_file : str or Path or None
            YAML file of good/bad phrasing pairs rendered into the prompt. None
            leaves the examples section empty.
        save_dir : str or Path or None
            Directory every prompt sent is written to. None writes nothing.

        Raises
        ------
        ValueError
            If the backend does not support images.
        """
        if not llm.SUPPORTS_IMAGES:
            raise ValueError(
                f"{type(llm).__name__} does not support images and cannot name the things in a "
                "scene it cannot see."
            )

        self.llm = llm
        self.task = task
        self.prompt = prompt
        self.refinement_prompt = refinement_prompt
        self.examples = self._render_examples(examples_file)
        self.save_dir = Path(save_dir) if save_dir is not None else None

    @classmethod
    def from_config(
        cls,
        llm_config_file: Union[str, Path],
        llm_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ConceptInventory":
        """Build an inventory from an LLM YAML configuration file.

        Parameters
        ----------
        llm_config_file : str or Path
            Path of the YAML configuration file describing the VLM (see ``LLM/conf``).
        llm_overrides : dict or None
            Forwarded to the backend's ``from_config``, e.g. ``{"params": {"seed": 7}}``.
        kwargs : dict
            Forwarded to :meth:`__init__`, e.g. ``task`` or ``examples_file``.

        Returns
        -------
        ConceptInventory
            An inventory asking the configured VLM.
        """
        return cls(create_llm(llm_config_file, **(llm_overrides or {})), **kwargs)

    @staticmethod
    def _render_examples(examples_file: Optional[Union[str, Path]]) -> str:
        """Render the few-shot phrasing pairs as the text injected in the prompt.

        Parameters
        ----------
        examples_file : str or Path or None
            YAML file holding a ``pairs`` list of ``good``/``bad``/``why``
            mappings. None renders nothing.

        Returns
        -------
        str
            The rendered block, or the empty string when there is nothing to show.
        """
        if examples_file is None:
            return ""

        path = Path(examples_file)
        if not path.is_file():
            logger.warning(f"Concept examples file not found: {path}. Continuing without examples.")
            return ""

        with open(path) as handle:
            content = yaml.safe_load(handle) or {}

        pairs = content.get("pairs") or []
        if not pairs:
            logger.warning(f"No pairs in {path}. Continuing without examples.")
            return ""

        lines = ["Phrases that work, next to phrasings of the same thing that do not:"]
        for pair in pairs:
            lines.append(f'\n  yes: "{pair["good"]}"')
            lines.append(f'  no:  "{pair["bad"]}"')
            lines.append(f"       ({pair['why']})")

        return "\n".join(lines)

    def build_prompt(self) -> str:
        """Return the scene inventory prompt with its sentinels filled in.

        Returns
        -------
        str
            :attr:`prompt` with the examples and the task substituted.
        """
        task_note = (
            f"The robot has to carry out the following task:\n{self.task}"
            if self.task
            else NO_TASK_NOTE
        )
        return self.prompt.replace("%%EXAMPLES%%", self.examples).replace("%%TASK%%", task_note)

    def concepts_for(self, image: Image.Image) -> Tuple[List[str], List[str]]:
        """Ask the VLM what is in the scene.

        Parameters
        ----------
        image : PIL.Image.Image
            The whole scene.

        Returns
        -------
        objects : list[str]
            Every visible thing that is not background.
        task_relevant : list[str]
            The subset of ``objects`` the task needs, empty when no task is set.
        """
        start = time.time()
        prompt = self.build_prompt()
        success, raw = self.llm.query(prompt, images=[image])

        if self.save_dir is not None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(Path(self.save_dir) / f"sent_query_{stamp}.txt", "w") as handle:
                handle.write(prompt)

        objects, task_relevant = self._parse_inventory(success, raw)

        logger.debug(f"Scene inventory obtained in {time.time() - start}s")
        logger.info(f"Concepts: {objects}")
        if task_relevant:
            logger.info(f"Task-relevant: {task_relevant}")

        return objects, task_relevant

    @staticmethod
    def _parse_inventory(success: bool, raw: str) -> Tuple[List[str], List[str]]:
        """Turn the VLM answer into the two concept lists.

        A malformed answer is logged and treated as an empty scene rather than
        raised: one unusable answer should cost one frame, not the whole run.

        Parameters
        ----------
        success : bool
            Whether the query reached the model.
        raw : str
            The raw answer of the model.

        Returns
        -------
        objects : list[str]
            Phrases under the ``objects`` key, deduplicated and stripped.
        task_relevant : list[str]
            Phrases under the ``task_relevant`` key, restricted to those that also
            appear in ``objects``.
        """
        if not success or not raw:
            logger.error("No response for the scene inventory. Segmenting nothing.")
            return [], []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Error decoding the scene inventory JSON: {raw}")
            return [], []

        if not isinstance(parsed, dict):
            logger.error(f"Unexpected scene inventory payload: {raw}")
            return [], []

        def clean(key: str) -> List[str]:
            values = parsed.get(key) or []
            if not isinstance(values, list):
                logger.error(f"Scene inventory field {key!r} is not a list: {values!r}")
                return []
            # dict.fromkeys rather than set(): duplicated phrases would each be
            # sent to SAM 3 and produce the same masks twice, and the order the
            # VLM chose is worth keeping in the logs.
            return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))

        objects = clean("objects")
        task_relevant = clean("task_relevant")

        # The prompt asks for task_relevant to be copied verbatim out of objects.
        # When it is not, the phrase was never segmented, so flagging it as
        # task-relevant would point at masks that do not exist.
        unknown = [phrase for phrase in task_relevant if phrase not in objects]
        if unknown:
            logger.warning(f"Dropping task-relevant phrases absent from the inventory: {unknown}")
            task_relevant = [phrase for phrase in task_relevant if phrase in objects]

        return objects, task_relevant

    def narrow(self, image: Image.Image, ambiguous: Dict[str, int]) -> Dict[str, List[str]]:
        """Ask the VLM to replace ambiguous phrases with narrower ones.

        Parameters
        ----------
        image : PIL.Image.Image
            The whole scene, shown again so the replacements describe what is
            actually there.
        ambiguous : dict[str, int]
            Phrases that matched several regions, and how many each matched.

        Returns
        -------
        dict[str, list[str]]
            Mapping from each ambiguous phrase to its replacements. Empty when the
            answer was unusable.
        """
        listing = "\n".join(
            f'- "{concept}" matched {count} separate regions'
            for concept, count in ambiguous.items()
        )
        message = self.refinement_prompt.replace("%%AMBIGUOUS%%", listing).replace(
            "%%EXAMPLES%%", self.examples
        )

        success, raw = self.llm.query(message, images=[image])
        return self._parse_replacements(success, raw, ambiguous)

    @staticmethod
    def _parse_replacements(
        success: bool, raw: str, ambiguous: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """Turn the VLM answer into a mapping of phrase to narrower phrases.

        Parameters
        ----------
        success : bool
            Whether the query reached the model.
        raw : str
            The raw answer of the model.
        ambiguous : dict[str, int]
            The phrases that were asked about, used to reject replacements for
            anything else.

        Returns
        -------
        dict[str, list[str]]
            The usable replacements, possibly empty.
        """
        if not success or not raw:
            logger.error("No response for the refinement query.")
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Error decoding the refinement JSON: {raw}")
            return {}

        proposed = parsed.get("replacements") if isinstance(parsed, dict) else None
        if not isinstance(proposed, dict):
            logger.error(f"Unexpected refinement payload: {raw}")
            return {}

        replacements: Dict[str, List[str]] = {}
        for phrase, values in proposed.items():
            if phrase not in ambiguous:
                logger.warning(f"Ignoring a replacement for {phrase!r}, which was not ambiguous.")
                continue

            values = [values] if isinstance(values, str) else values
            if not isinstance(values, list):
                logger.warning(f"Replacements for {phrase!r} are not a list: {values!r}")
                continue

            # A replacement identical to the phrase it replaces is what an
            # unhelpful answer looks like, and would loop until the budget runs
            # out without changing anything.
            cleaned = list(
                dict.fromkeys(
                    str(v).strip() for v in values if str(v).strip() and str(v).strip() != phrase
                )
            )
            if cleaned:
                replacements[phrase] = cleaned

        return replacements

    def close(self) -> None:
        """Release the VLM connection."""
        self.llm.close()
