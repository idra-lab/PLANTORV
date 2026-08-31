"""SAM 3 concept segmentation for the pipeline.

SAM 3 is not driven the way SAM 1, SAM 2 and FastSAM are. Those three are asked
to segment *everything* and hand back an unlabelled pile of masks, which
``SAMModel`` and ``FastSAMModel`` then filter down to plausible objects with
area, border and containment heuristics. SAM 3 is asked for *concepts*: it takes
short noun phrases and returns every instance matching each one, already
attributed to the phrase that found it.

That inverts the pipeline. The heuristics exist to compensate for masks having no
semantics, so most of them have nothing left to do here; what matters instead is
the quality of the phrases. This class therefore owns both halves:

1. A VLM (any :class:`~LLM.llm_base.BaseLLM` that accepts images) is shown the
   scene and asked for the things in it, as phrases SAM 3 can ground. When a task
   is configured it is also asked which of those the task needs.
2. Those phrases are passed to Ultralytics' ``SAM3SemanticPredictor``, and the
   masks come back labelled with the concept that produced them.

The labels are the point: :meth:`SAM3Model.segment` returns them alongside the
masks, so the annotator downstream no longer has to work out *what* each crop is
and can spend its call on describing it instead.

Only the whole-scene inventory prompt lives here. Describing a mask once it has
been cut out is a separate job and stays in
``scene_understanding/gpt_annotator.py``.

Requires ``ultralytics >= 8.3.237`` and the gated ``sam3.pt`` checkpoint; see
``scripts/install_models.py`` for how to obtain it.
"""

import base64
import html
import json
import tempfile
import time
import webbrowser
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor

from LLM.llm_base import BaseLLM
from LLM.llm_factory import create_llm
from segmentation.segmentation import SegmentationModel, binary_mask_to_pil
from utility.utility import logger

# The only checkpoint SAM 3 ships. Kept as a tuple for symmetry with
# `SAM_CHECKPOINTS` and `FASTSAM_CHECKPOINTS`, and because the check below is a
# suffix match, so a local path such as "models/sam/sam3.pt" works too.
SAM3_CHECKPOINTS = ("sam3.pt",)

# Few-shot pairs steering how the VLM phrases the concepts. Data rather than
# code, so the phrasing can be tuned without touching this file.
DEFAULT_EXAMPLES_FILE = (
    Path(__file__).resolve().parent.parent / "LLM" / "examples" / "SAM3" / "concept_prompts.yaml"
)

# Detection score below which an instance is dropped. SAM 3 answers every phrase
# it is given, including phrases for things that are not in the scene, so this is
# the knob that decides how readily it hallucinates one.
DEFAULT_CONF = 0.25

# SAM 3's own default is 1008; the pipeline's frames are 1920x1080.
DEFAULT_IMGSZ = 1024

# Half precision by default: the checkpoint is ~3.2 GB in fp32, which is a large
# share of a single GPU next to the VLM and the rest of the pipeline.
DEFAULT_QUANTIZE = 16

# NMS IoU threshold. Far below the Ultralytics default of 0.7, because the masks
# that need suppressing here overlap partially rather than nearly exactly: a
# concept naming a part matches the part *and* the whole it sits in. Note this
# only works alongside phrases specific enough to name one structure each; with a
# broad phrase like "lego block" it would also merge genuinely separate bricks
# whose boxes happen to touch.
DEFAULT_IOU = 0.1

# How many times a scene inventory may be sent back to the VLM to be narrowed.
# Zero: the refinement loop is parked and not wired into `segment`, see the
# section at the end of this file.
DEFAULT_MAX_REFINEMENTS = 0

# How close, in pixels, two masks of the same concept have to be for them to be
# treated as one thing. Zero disables merging. A row of bricks comes back from
# SAM 3 as one mask per brick, all carrying the same concept and all touching, so
# a small gap is enough to put the row back together. Keep it small: the rule
# cannot tell a row of bricks from two separate towers that happen to touch.
DEFAULT_MERGE_GAP = 5

# Thickness of the mask outlines in the debug overlay.
OUTLINE_WIDTH = 3

# Longest side, in pixels, of the images embedded in the HTML view. The page
# carries them base64-encoded so that it stays a single self-contained file that
# can be copied off a cluster node, which makes their size the page's size.
VIEW_OVERLAY_MAX = 1600
VIEW_CROP_MAX = 360

# Styling for the page written by `SAM3Model.visualize`. Kept out of the builder
# so that the braces of the CSS do not have to be escaped inside an f-string.
VIEWER_CSS = """
body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; background: #14161a; color: #e6e8ec; }
h1 { font-size: 19px; margin: 0 0 8px; font-weight: 600; }
p.meta { color: #9aa1ac; font-size: 13px; margin: 0 0 4px; }
p.meta b { color: #e6e8ec; font-weight: 600; }
p.missing b { color: #f0a868; }
img.overlay { width: 100%; border-radius: 8px; display: block; margin: 20px 0; }
.grid { display: grid; gap: 16px;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
figure { margin: 0; background: #1c1f25; border: 1px solid #2a2f38;
         border-radius: 8px; padding: 10px; }
figure img { width: 100%; display: block; background: #000; border-radius: 4px; }
figcaption { font-size: 12px; margin-top: 8px; }
figcaption b { display: block; font-size: 13px; margin-bottom: 2px; }
figcaption span { color: #9aa1ac; }
.swatch { display: inline-block; width: 9px; height: 9px;
          border-radius: 2px; margin-right: 6px; }
"""

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


class SAM3Model(SegmentationModel):
    """Segment a scene by concept, using a VLM to decide what the concepts are.

    The VLM runs first and names the things in the image; SAM 3 then segments
    exactly those. Both steps are hidden behind :meth:`segment`::

        model = SAM3Model(llm, "models/sam/sam3.pt", task="pick up the wrench")
        masks, bboxes, labels = model.segment(image)

    Pass ``concepts=`` to the constructor, or call :meth:`set_concepts`, to skip
    the VLM entirely and segment a fixed vocabulary.

    Attributes
    ----------
    llm : BaseLLM
        Backend asked for the scene inventory. Must accept images.
    concepts : list[str] or None
        Phrases SAM 3 is prompted with. ``None`` until the VLM has been asked, or
        for as long as it is asked afresh for every image.
    task_relevant : list[str]
        Subset of :attr:`concepts` the VLM considers necessary for the task.
        Empty when no task is configured.
    last_concepts : list[str]
        The phrases the most recent :meth:`segment` actually used. Differs from
        :attr:`concepts` once the refinement loop has narrowed something.
    """

    def __init__(
        self,
        llm: BaseLLM,
        sam3_checkpoint: Union[str, Path] = "sam3.pt",
        device: str = "cuda",
        save_dir: Optional[Union[str, Path]] = None,
        debug_masks: bool = False,
        task: Optional[str] = None,
        concepts: Optional[Sequence[str]] = None,
        merge_gap: int = DEFAULT_MERGE_GAP,
        max_refinements: int = DEFAULT_MAX_REFINEMENTS,
        prompt: str = SCENE_INVENTORY_PROMPT,
        refinement_prompt: str = REFINEMENT_PROMPT,
        examples_file: Optional[Union[str, Path]] = DEFAULT_EXAMPLES_FILE,
        **predict_kwargs: Any,
    ) -> None:
        """Initialize the SAM 3 model and the VLM that feeds it.

        Parameters
        ----------
        llm : BaseLLM
            Backend used to name the things in the scene. It must support images,
            which is the case for the Azure OpenAI, OpenAI, Anthropic, Gemini and
            GLM backends.
        sam3_checkpoint : str or Path
            Checkpoint to load. Either ``"sam3.pt"`` or a path to a local file
            whose name ends with it. Unlike the SAM 2 and FastSAM checkpoints this
            one is gated and is never downloaded automatically; run
            ``scripts/install_models.py sam3`` to fetch it.
        device : str
            Device used for inference. Common values are ``"cuda"`` and ``"cpu"``.
            Default is ``"cuda"``.
        save_dir : str or Path or None
            Directory where the output images will be saved if not None.
        debug_masks : bool
            If True, dump every mask under ``save_dir/debug/``, after touching
            regions of one concept have been merged but before any refinement.
            Requires ``save_dir``. Default is False.
        task : str or None
            The task the robot has to carry out. When given, the VLM also reports
            which objects it needs, in :attr:`task_relevant`. It never restricts
            what gets segmented: the full inventory is always returned, because an
            object irrelevant to the task can still be an obstacle.
        concepts : sequence of str or None
            Fixed phrases to segment. When given, the VLM is never called, and no
            refinement is attempted either: an explicit list is taken at its word.
        merge_gap : int
            How close, in pixels, two masks carrying the same concept have to be
            for them to be merged into one. Applied to every prediction, so it
            happens before the refinement loop looks at anything. ``0`` disables
            merging. Default is :data:`DEFAULT_MERGE_GAP`.
        max_refinements : int
            Kept for the parked refinement loop at the end of this file, which
            ``segment`` does not currently call. Setting it above zero warns and
            otherwise does nothing. Default is :data:`DEFAULT_MAX_REFINEMENTS`.
        prompt : str
            Instructions sent with the scene image. Defaults to
            :data:`SCENE_INVENTORY_PROMPT`.
        refinement_prompt : str
            Instructions sent when narrowing an ambiguous phrase. Defaults to
            :data:`REFINEMENT_PROMPT`.
        examples_file : str or Path or None
            YAML file of good/bad phrasing pairs rendered into the prompt. None
            leaves the examples section empty.
        predict_kwargs : dict
            Additional overrides for the Ultralytics predictor, e.g. ``conf``,
            ``iou`` or ``imgsz``. Check the documentation of
            ``ultralytics.models.sam.SAM3SemanticPredictor`` for the full list.

        Raises
        ------
        ValueError
            If ``sam3_checkpoint`` is not a SAM 3 checkpoint, if the backend does
            not support images, or if ``debug_masks`` is set without a
            ``save_dir``.
        """
        super().__init__(save_dir=save_dir)

        checkpoint = str(sam3_checkpoint)
        if not checkpoint.endswith(SAM3_CHECKPOINTS):
            raise ValueError(
                f"Unsupported SAM 3 checkpoint {checkpoint!r}. The file name must end with one of "
                f"{', '.join(SAM3_CHECKPOINTS)}. For SAM 1 and SAM 2 use SAMModel, for FastSAM "
                "use FastSAMModel."
            )

        if not llm.SUPPORTS_IMAGES:
            raise ValueError(
                f"{type(llm).__name__} does not support images and cannot name the things in a "
                "scene it cannot see."
            )

        if debug_masks and self.save_dir is None:
            raise ValueError("debug_masks=True requires save_dir to be set.")
        self.debug_masks = debug_masks

        self.llm = llm
        self.task = task
        self.prompt = prompt
        self.refinement_prompt = refinement_prompt
        self.merge_gap = max(0, int(merge_gap))
        self.max_refinements = max(0, int(max_refinements))
        if self.max_refinements:
            logger.warning(
                f"max_refinements={self.max_refinements} has no effect: the refinement loop is "
                "parked at the end of segmentation/sam3_model.py and segment() does not call it."
            )
        self.concepts: Optional[List[str]] = list(concepts) if concepts is not None else None
        self.task_relevant: List[str] = []
        self.examples = self._render_examples(examples_file)

        predict_kwargs.setdefault("conf", DEFAULT_CONF)
        predict_kwargs.setdefault("iou", DEFAULT_IOU)
        predict_kwargs.setdefault("imgsz", DEFAULT_IMGSZ)
        predict_kwargs.setdefault("quantize", DEFAULT_QUANTIZE)
        predict_kwargs["model"] = checkpoint
        predict_kwargs["device"] = device
        predict_kwargs["task"] = "segment"
        predict_kwargs["mode"] = "predict"
        predict_kwargs["verbose"] = False
        # Ultralytics otherwise writes annotated copies to runs/ on every call.
        predict_kwargs["save"] = False

        self.predictor = SAM3SemanticPredictor(overrides=predict_kwargs)

        # One-entry cache of the last segmentation, so that `obtain_bg` and
        # `individual_mask` called on the same image do not each pay for a VLM
        # call and a SAM 3 forward pass. Keyed on the image index the caller
        # passes, so reusing an index for a different image serves a stale
        # result; `segment(..., refresh=True)` forces the work to be redone.
        self._cache_idx: Optional[int] = None
        self._cache: Optional[Tuple[np.ndarray, List[List[int]], List[str]]] = None
        self.last_concepts: List[str] = []

    @classmethod
    def from_config(
        cls,
        llm_config_file: Union[str, Path],
        sam3_checkpoint: Union[str, Path] = "sam3.pt",
        llm_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SAM3Model":
        """Build a SAM3Model from an LLM YAML configuration file.

        The configuration file carries the model, the endpoint, the credentials
        and the request parameters (see ``LLM/conf``), and also selects the
        backend, exactly as it does for the annotator.

        Parameters
        ----------
        llm_config_file : str or Path
            Path of the YAML configuration file describing the VLM.
        sam3_checkpoint : str or Path
            Checkpoint to load.
        llm_overrides : dict or None
            Forwarded to the backend's ``from_config``, e.g. ``{"params": {"seed": 7}}``.
        kwargs : dict
            Forwarded to :meth:`__init__`, e.g. ``save_dir`` or ``task``.

        Returns
        -------
        SAM3Model
            A model configured from the file.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist or is not a YAML file.
        ValueError
            If the selected backend does not support images.
        """
        llm = create_llm(llm_config_file, **(llm_overrides or {}))
        return cls(llm, sam3_checkpoint, **kwargs)

    ## Concepts ########################################################################################################

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

    def set_concepts(self, concepts: Sequence[str]) -> None:
        """Fix the phrases SAM 3 is prompted with, bypassing the VLM.

        Parameters
        ----------
        concepts : sequence of str
            Phrases to segment from now on.
        """
        self.concepts = list(concepts)
        self._invalidate()
        logger.debug(f"Concepts set manually: {self.concepts}")

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

    ## Segmentation ####################################################################################################

    def _invalidate(self) -> None:
        """Drop the cached segmentation."""
        self._cache_idx = None
        self._cache = None

    @staticmethod
    def _to_pil(image: Union[Image.Image, str, Path]) -> Image.Image:
        """Return the input as a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            RGB image, or a path to one.

        Returns
        -------
        PIL.Image.Image
            The image.

        Raises
        ------
        FileNotFoundError
            If the specified image file does not exist.
        TypeError
            If the input image is not a file path, Path object, or PIL Image.
        """
        if isinstance(image, (str, Path)):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            return Image.open(image)
        if isinstance(image, Image.Image):
            return image
        raise TypeError("Input image must be a file path, Path object, or PIL Image.")

    def segment(
        self,
        image: Union[Image.Image, str, Path],
        idx: int = 0,
        refresh: bool = False,
    ) -> Tuple[np.ndarray, List[List[int]], List[str]]:
        """Name the things in the scene, then segment every instance of each.

        This is the model's real interface. :meth:`obtain_bg` and
        :meth:`individual_mask` exist so that a caller written against
        ``SAMModel`` or ``FastSAMModel`` keeps working, and both are served from
        the result of this method.

        Regions carrying the same concept that touch are merged first, so a row of
        bricks comes back as one object rather than as one per brick.
        :attr:`last_concepts` holds the phrases that were used.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            RGB image, or a path to one.
        idx : int
            Image index, used to key the cache and to name the debug dumps.
        refresh : bool
            Segment again even when a cached result for ``idx`` is available.

        Returns
        -------
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``, at the resolution of the input
            image. Empty with shape ``(0, H, W)`` when nothing is found.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        labels : list[str]
            The concept each mask was found by, one per mask. Several masks share
            a label when a concept has several instances.
        """
        if not refresh and self._cache is not None and self._cache_idx == idx:
            return self._cache

        image_pil = self._to_pil(image)
        image_np = np.array(image_pil)

        concepts = self.concepts
        if concepts is None:
            concepts, self.task_relevant = self.concepts_for(image_pil)

        if not concepts:
            logger.warning(f"No concepts to segment for image {idx + 1}.")
            empty = (np.zeros((0, *image_np.shape[:2]), dtype=bool), [], [])
            self.last_concepts = []
            self._cache_idx, self._cache = idx, empty
            return empty

        masks, bboxes, labels = self._predict(image_np, concepts, tag=f"image_{idx + 1}")
        self.last_concepts = list(concepts)

        self._cache_idx, self._cache = idx, (masks, bboxes, labels)
        return masks, bboxes, labels

    ## Merging #########################################################################################################

    @staticmethod
    def _adjacent(
        mask_a: np.ndarray,
        box_a: List[int],
        mask_b: np.ndarray,
        box_b: List[int],
        gap: int,
    ) -> bool:
        """Return whether two masks come within ``gap`` pixels of each other.

        Only the window where the two boxes, each grown by ``gap``, overlap is
        examined. Any pair of pixels closer than ``gap`` has to lie inside it, so
        the test is exact while touching a few thousand pixels instead of the two
        million in a full frame.

        Parameters
        ----------
        mask_a, mask_b : numpy.ndarray
            Boolean masks with shape ``(H, W)``.
        box_a, box_b : list[int]
            Their bounding boxes as ``[x_min, y_min, width, height]``.
        gap : int
            Largest distance, in pixels, still counted as touching.

        Returns
        -------
        bool
            True when the masks touch or are within ``gap`` pixels.
        """
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        height, width = mask_a.shape[:2]
        left = max(0, max(ax, bx) - gap)
        top = max(0, max(ay, by) - gap)
        right = min(width, min(ax + aw, bx + bw) + gap)
        bottom = min(height, min(ay + ah, by + bh) + gap)
        if left >= right or top >= bottom:
            return False

        window_a = mask_a[top:bottom, left:right].astype(np.uint8)
        window_b = mask_b[top:bottom, left:right]
        if not window_a.any() or not window_b.any():
            return False

        kernel = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
        return bool(np.logical_and(cv2.dilate(window_a, kernel).astype(bool), window_b).any())

    @classmethod
    def _groups(cls, masks: List[np.ndarray], boxes: List[List[int]], gap: int) -> List[List[int]]:
        """Group masks into runs of mutually reachable neighbours.

        Adjacency is transitive here on purpose: the bricks at the two ends of a
        row never touch each other, but each touches the next, so the whole row
        has to come out as one group.

        Parameters
        ----------
        masks : list[numpy.ndarray]
            Boolean masks with shape ``(H, W)``.
        boxes : list[list[int]]
            Their bounding boxes as ``[x_min, y_min, width, height]``.
        gap : int
            Largest distance, in pixels, still counted as touching.

        Returns
        -------
        list[list[int]]
            Indices into ``masks``, one list per group, each in ascending order
            and the groups ordered by their first member.
        """
        parent = list(range(len(masks)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if find(i) != find(j) and cls._adjacent(
                    masks[i], boxes[i], masks[j], boxes[j], gap
                ):
                    parent[find(j)] = find(i)

        grouped: Dict[int, List[int]] = {}
        for i in range(len(masks)):
            grouped.setdefault(find(i), []).append(i)
        return list(grouped.values())

    def _merge_adjacent(
        self, masks: np.ndarray, bboxes: List[List[int]], labels: List[str]
    ) -> Tuple[np.ndarray, List[List[int]], List[str]]:
        """Merge masks that share a concept and touch into a single mask.

        SAM 3 answers a phrase with every region matching it, and a phrase naming
        an assembly matches the pieces as readily as the whole, so a row of bricks
        arrives as one mask per brick under one concept. Those pieces are touching
        by construction, which is what separates them from two genuinely distinct
        objects the phrase happens to cover.

        Runs on every prediction, so the refinement loop only ever sees merged
        masks: a concept whose pieces have been put back together is no longer
        ambiguous, and is not sent back to the VLM to be narrowed.

        Parameters
        ----------
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        labels : list[str]
            The concept each mask was found by.

        Returns
        -------
        masks : numpy.ndarray
            Boolean masks with shape ``(M, H, W)``, ``M <= N``, grouped by concept
            in order of first appearance.
        bboxes : list[list[int]]
            The box enclosing each merged mask.
        labels : list[str]
            The concept of each merged mask.
        """
        if not self.merge_gap or len(masks) < 2:
            return masks, bboxes, labels

        merged_masks: List[np.ndarray] = []
        merged_boxes: List[List[int]] = []
        merged_labels: List[str] = []

        for concept in dict.fromkeys(labels):
            members = [i for i, label in enumerate(labels) if label == concept]
            groups = self._groups(
                [masks[i] for i in members], [bboxes[i] for i in members], self.merge_gap
            )

            for group in groups:
                indices = [members[g] for g in group]
                merged_masks.append(np.any(masks[indices], axis=0))
                merged_labels.append(concept)
                # Built from the member boxes rather than from the union mask, so
                # that a merged box follows the same convention as an unmerged one.
                lefts = [bboxes[i][0] for i in indices]
                tops = [bboxes[i][1] for i in indices]
                rights = [bboxes[i][0] + bboxes[i][2] for i in indices]
                bottoms = [bboxes[i][1] + bboxes[i][3] for i in indices]
                merged_boxes.append(
                    [min(lefts), min(tops), max(rights) - min(lefts), max(bottoms) - min(tops)]
                )

        if len(merged_masks) < len(masks):
            collapsed = Counter(labels) - Counter(merged_labels)
            logger.info(
                f"Merged {len(masks)} masks into {len(merged_masks)} by joining touching regions "
                f"of the same concept: {', '.join(f'{c!r} -{n}' for c, n in collapsed.items())}"
            )

        return np.array(merged_masks), merged_boxes, merged_labels

    def _predict(
        self, image: np.ndarray, concepts: List[str], tag: Optional[str] = None
    ) -> Tuple[np.ndarray, List[List[int]], List[str]]:
        """Run SAM 3 over an image for a list of concepts.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image with shape ``(H, W, 3)``.
        concepts : list[str]
            Noun phrases to look for.
        tag : str or None
            Name used for the debug dump when ``debug_masks`` is enabled.

        Returns
        -------
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        labels : list[str]
            The concept each mask was found by.
        """
        start = time.time()

        # Ultralytics assumes a numpy source is BGR (its preprocess step flips
        # the channels back before normalisation), while the rest of this file
        # works in RGB. Handing it RGB feeds the network channel-swapped images.
        bgr = np.ascontiguousarray(image[..., ::-1])

        # Typed as Any because Ultralytics' annotations do not match runtime here:
        # `masks.data` is a torch.Tensor despite being annotated as an ndarray.
        self.predictor.set_image(bgr)
        result: Any = self.predictor(text=list(concepts))[0]

        if result.masks is None or len(result.boxes) == 0:
            logger.warning(f"SAM 3 found none of {concepts} in {tag or 'image'}")
            return np.zeros((0, *image.shape[:2]), dtype=bool), [], []

        masks = result.masks.data.cpu().numpy().astype(bool)

        # Ultralytics reports boxes as xyxy; `boxes.xywh` is centre-based, while
        # the rest of the pipeline (GPT annotator, evaluation) expects the corner
        # based [x_min, y_min, width, height].
        xyxy = result.boxes.xyxy.cpu().numpy()
        bboxes = [
            [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            for x_min, y_min, x_max, y_max in xyxy
        ]

        # `boxes.cls` indexes the list of phrases the predictor was given, which
        # it also exposes as `names`. Going through `names` rather than through
        # `concepts` keeps the labels right if Ultralytics ever reorders them.
        names = result.names
        classes = result.boxes.cls.cpu().numpy().astype(int)
        labels = [
            str(names[int(c)] if isinstance(names, (list, tuple)) else names.get(int(c), c))
            for c in classes
        ]

        logger.debug(
            f"SAM 3 returned {len(masks)} masks for {len(concepts)} concepts "
            f"in {time.time() - start}s"
        )

        # Merged before anything else sees the masks, so the debug dumps, the
        # refinement loop and the caller all agree on what counts as one object.
        masks, bboxes, labels = self._merge_adjacent(masks, bboxes, labels)

        if self.debug_masks and tag is not None:
            self._dump_masks(image, masks, bboxes, labels, tag)

        return masks, bboxes, labels

    @staticmethod
    def _slug(label: str) -> str:
        """Return a concept label usable in a file name.

        Parameters
        ----------
        label : str
            The concept.

        Returns
        -------
        str
            The label with everything but letters and digits replaced by
            underscores, so that a phrase containing a slash or a space cannot
            turn into a nested path.
        """
        return "".join(c if c.isalnum() else "_" for c in label)

    @staticmethod
    def _colours(count: int) -> List[np.ndarray]:
        """Return one distinct RGB colour per mask.

        Parameters
        ----------
        count : int
            How many colours are needed.

        Returns
        -------
        list[numpy.ndarray]
            Colours spread over the hue circle in fixed steps, which keeps
            neighbouring masks visually distinct.
        """
        return [
            cv2.cvtColor(
                np.array([[[int(180 * i / max(count, 1)), 255, 255]]], dtype=np.uint8),
                cv2.COLOR_HSV2RGB,
            )[0, 0].astype(np.uint16)
            for i in range(count)
        ]

    @classmethod
    def _overlay(cls, image: np.ndarray, masks: np.ndarray, labels: List[str]) -> np.ndarray:
        """Draw every mask on the image as a coloured outline with its concept.

        Outlines only, never a fill. Colour is what the concept phrases are built
        on, and a fill repaints the objects in a hue picked per mask index rather
        than per object, which makes the one question worth asking of the picture
        - did "yellow structure" land on something yellow? - unanswerable.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image with shape ``(H, W, 3)``.
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        labels : list[str]
            The concept each mask was found by.

        Returns
        -------
        numpy.ndarray
            A copy of ``image`` with the outlines and labels drawn on it.
        """
        overlay = image.copy()
        colours = cls._colours(len(masks))

        for mask, colour in zip(masks, colours):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colour.tolist(), OUTLINE_WIDTH)

        # Labels go on after every outline, so that an outline drawn later cannot
        # run through the text of a mask drawn earlier.
        for i, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            cv2.putText(
                overlay,
                f"{i}:{labels[i]}",
                (int(xs.mean()), int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    def _dump_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        bboxes: List[List[int]],
        labels: List[str],
        tag: str,
    ) -> None:
        """Write the masks of one prediction to ``save_dir/debug/`` for inspection.

        Produces one colour-coded overlay of all masks at once, one binary PNG per
        mask named after the concept that found it, and a log line per mask, so a
        concept the VLM never named can be told apart from one SAM 3 could not
        find.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image the masks were generated from, shape ``(H, W, 3)``.
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        labels : list[str]
            The concept each mask was found by.
        tag : str
            Sub-directory name, e.g. ``"image_1"``.
        """
        debug_dir = Path(cast(Path, self.save_dir)) / "debug" / tag
        debug_dir.mkdir(parents=True, exist_ok=True)

        H, W = image.shape[:2]

        logger.debug(f"[{tag}] SAM 3 returned {len(masks)} masks")
        for i, mask in enumerate(masks):
            area_pct = np.sum(mask) * 100 / (H * W)
            logger.debug(
                f"[{tag}]   mask {i:03d}: {labels[i]!r} area {area_pct:6.2f}% bbox {bboxes[i]}"
            )
            binary_mask_to_pil(mask.astype(np.uint8)).save(
                debug_dir / f"mask_{i:03d}_{self._slug(labels[i])}.png"
            )

        Image.fromarray(self._overlay(image, masks, labels)).save(
            debug_dir.parent / f"{tag}_all_masks.png"
        )
        logger.debug(f"[{tag}] wrote overlay and {len(masks)} masks to {debug_dir}")

    ## Visualisation ###################################################################################################

    @staticmethod
    def _data_uri(array: np.ndarray, max_side: int) -> str:
        """Encode an image as a PNG ``data:`` URI, downscaled to fit.

        Parameters
        ----------
        array : numpy.ndarray
            RGB image with shape ``(H, W, 3)``.
        max_side : int
            Longest side the image is shrunk to before encoding.

        Returns
        -------
        str
            The image as ``data:image/png;base64,...``.
        """
        img = Image.fromarray(array.astype(np.uint8))
        img.thumbnail((max_side, max_side))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _render_view(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        bboxes: List[List[int]],
        labels: List[str],
        idx: int,
    ) -> str:
        """Build the self-contained HTML page showing one segmentation.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image the masks were produced from, shape ``(H, W, 3)``.
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        labels : list[str]
            The concept each mask was found by.
        idx : int
            Image index, shown in the heading.

        Returns
        -------
        str
            The complete HTML document.
        """
        height, width = image.shape[:2]
        colours = self._colours(len(masks))

        header = [
            f"<h1>image {idx + 1} &mdash; {len(masks)} masks "
            f"from {len(self.last_concepts)} concepts</h1>"
        ]
        if self.task:
            header.append(f"<p class='meta'>task: <b>{html.escape(self.task)}</b></p>")
        if self.task_relevant:
            relevant = ", ".join(html.escape(c) for c in self.task_relevant)
            header.append(f"<p class='meta'>task-relevant: <b>{relevant}</b></p>")

        # A phrase that matched nothing is invisible in the picture, and is
        # exactly what a badly chosen concept looks like, so it is named here.
        missing = [c for c in self.last_concepts if c not in set(labels)]
        if missing:
            names = ", ".join(html.escape(c) for c in missing)
            header.append(f"<p class='meta missing'>found nothing for: <b>{names}</b></p>")

        cards = []
        for i, (mask, colour) in enumerate(zip(masks, colours)):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            top, bottom = int(ys.min()), int(ys.max()) + 1
            left, right = int(xs.min()), int(xs.max()) + 1

            # Deliberately not `_crop`: that boosts contrast for the annotator,
            # which shifts the colours this page exists to let you check.
            cut = image[top:bottom, left:right] * mask[top:bottom, left:right, None]

            swatch = "#{:02x}{:02x}{:02x}".format(*(int(c) for c in colour))
            area = mask.sum() * 100 / (height * width)
            cards.append(
                "<figure>"
                f"<img src='{self._data_uri(cut, VIEW_CROP_MAX)}'>"
                f"<figcaption><b><span class='swatch' style='background:{swatch}'></span>"
                f"{i}: {html.escape(labels[i])}</b>"
                f"<span>{area:.2f}% of frame &middot; bbox {bboxes[i]}</span>"
                "</figcaption></figure>"
            )

        overlay = self._data_uri(self._overlay(image, masks, labels), VIEW_OVERLAY_MAX)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>SAM 3 &mdash; image {idx + 1}</title>"
            f"<style>{VIEWER_CSS}</style></head><body>"
            + "".join(header)
            + f"<img class='overlay' src='{overlay}'>"
            + "<div class='grid'>"
            + "".join(cards)
            + "</div></body></html>"
        )

    def visualize(
        self,
        image: Union[Image.Image, str, Path],
        idx: int = 0,
        path: Optional[Union[str, Path]] = None,
        open_browser: bool = True,
    ) -> Path:
        """Write an HTML page showing the final masks, and open it in a browser.

        Renders whatever :meth:`segment` returned, from the cache, so it shows the
        finished result by construction rather than one of the intermediate
        predictions the debug dumps hold.

        The page embeds its images, so it is a single file that can be copied off
        a cluster node and opened anywhere.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            The image to show. Segmented on the spot if it has not been already.
        idx : int
            Image index, used for the cache, the heading and the file name.
        path : str or Path or None
            Where to write the page. Defaults to ``save_dir``, or the system
            temporary directory when no ``save_dir`` is configured.
        open_browser : bool
            Open the page once written. Default is True.

        Returns
        -------
        pathlib.Path
            The file that was written.
        """
        image_np = np.array(self._to_pil(image))
        masks, bboxes, labels = self.segment(image, idx=idx)

        if path is not None:
            destination = Path(path)
        elif self.save_dir is not None:
            destination = Path(self.save_dir) / f"image_{idx + 1}_masks.html"
        else:
            destination = Path(tempfile.gettempdir()) / f"sam3_image_{idx + 1}_masks.html"
        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(
            self._render_view(image_np, masks, bboxes, labels, idx), encoding="utf-8"
        )
        logger.info(f"Wrote the mask view for image {idx + 1} to {destination}")

        if open_browser:
            # `webbrowser.open` reports "nothing could open it" by returning False
            # rather than by raising, so a headless machine or a cluster node
            # would otherwise fail here in complete silence.
            opened = False
            try:
                opened = webbrowser.open(destination.as_uri())
            except Exception as error:
                logger.warning(f"Could not open a browser: {error}")
            if not opened:
                logger.warning(f"No browser was opened. Open {destination} by hand.")

        return destination

    ## SegmentationModel interface #####################################################################################

    def obtain_bg(
        self, image: Union[Image.Image, str, Path], idx: int = 0, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove the background region from an RGB image.

        Takes the background to be everything no concept mask covers. Where the
        SAM and FastSAM versions have to infer which masks are scenery from their
        size and their position, here the question does not arise: a mask exists
        only because something was named, and the prompt asks for things rather
        than for scenery, so the union of the masks is the foreground by
        construction.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            RGB image, or a path to one.
        idx : int
            Image index used when saving the debug and background images.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        masked_rgb : numpy.ndarray
            RGB image with everything unnamed zeroed, shape ``(H, W, 3)``.
        mask_bin : numpy.ndarray
            Binary keep-mask with shape ``(H, W, 1)`` and values ``0`` and ``1``.
        """
        start = time.time()

        image_np = np.array(self._to_pil(image))
        masks, _, _ = self.segment(image, idx=idx)

        keep = np.any(masks, axis=0) if len(masks) else np.zeros(image_np.shape[:2], dtype=bool)
        mask_bin = keep.astype(np.uint8)[..., None]
        masked_rgb = (image_np * mask_bin).astype(np.uint8)

        if self.save_dir is not None:
            logger.debug(
                f"Saving applied background mask for image {idx} as "
                f"{self.save_dir}/figure_{idx}_bg_masked_rgb.png"
            )
            Image.fromarray(masked_rgb).save(f"{self.save_dir}/figure_{idx}_bg_masked_rgb.png")
            binary_mask_to_pil(mask_bin).save(f"{self.save_dir}/figure_{idx}_bg_mask_bin.png")

        logger.debug(f"BG mask obtained in {time.time() - start}s")

        return masked_rgb, mask_bin

    def individual_mask(
        self,
        image: Union[Image.Image, str, Path],
        bg_mask_bin: np.ndarray = np.asarray([]),
        bg_masked_rgb: np.ndarray = np.asarray([]),
        idx: int = 0,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], list[list[int]]]:
        """Return one cropped RGB image per segmented instance.

        Kept signature-compatible with ``SAMModel`` and ``FastSAMModel`` so the
        existing pipeline runs unchanged, which is why ``bg_mask_bin`` and
        ``bg_masked_rgb`` are accepted and ignored: SAM 3 does not need the
        background removed first, since it never proposes a mask nobody asked for.

        The concept labels are dropped here, because the two-method interface has
        nowhere to put them. Call :meth:`segment` instead to keep them, and read
        :attr:`task_relevant` to know which of them the task needs.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            The original RGB image, or a path to it.
        bg_mask_bin : numpy.ndarray
            Unused. Present for interface compatibility.
        bg_masked_rgb : numpy.ndarray
            Unused. Present for interface compatibility.
        idx : int
            Image index used in saved crop names.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        rgb_masks : list[numpy.ndarray]
            Cropped RGB images, one per instance.
        bboxes : list[list[int]]
            Bounding boxes, each ``[x_min, y_min, width, height]``.
        """
        start = time.time()

        image_np = np.array(self._to_pil(image))
        masks, bboxes, labels = self.segment(image, idx=idx)

        rgb_masks = [self._crop(image_np, mask) for mask in masks]

        if self.save_dir is not None and rgb_masks:
            save_path = Path(self.save_dir) / f"image_{idx + 1}" / "objects"
            save_path.mkdir(parents=True, exist_ok=True)
            for i, (crop, label) in enumerate(zip(rgb_masks, labels)):
                safe = "".join(c if c.isalnum() else "_" for c in label)
                Image.fromarray(crop).save(save_path / f"object_{i:03d}_{safe}.png")

        logger.debug(f"{len(rgb_masks)} individual masks obtained in {time.time() - start}s")

        return rgb_masks, bboxes

    @staticmethod
    def _crop(
        rgb: np.ndarray, mask: np.ndarray, alpha: float = 1.4, beta: float = 25
    ) -> np.ndarray:
        """Cut one instance out of the image, then upscale and sharpen it.

        Unlike the FastSAM and SAM versions this does no morphological cleaning.
        Those masks come from a blind everything-pass and arrive ragged; a SAM 3
        mask is a single instance of a named concept, so eroding and dilating it
        would only cost detail the annotator uses.

        Parameters
        ----------
        rgb : numpy.ndarray
            RGB image with shape ``(H, W, 3)``.
        mask : numpy.ndarray
            Boolean mask with shape ``(H, W)``.
        alpha : float, optional
            Contrast multiplier passed to ``cv2.convertScaleAbs``.
        beta : float, optional
            Brightness offset passed to ``cv2.convertScaleAbs``.

        Returns
        -------
        numpy.ndarray
            The cropped, upscaled, contrast-adjusted and sharpened instance.
        """
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1

        # Zero everything outside the instance, so the crop shows the object
        # rather than the object plus whatever shares its bounding box.
        cut = rgb[top:bottom, left:right] * mask[top:bottom, left:right, None]

        crop = cv2.convertScaleAbs(cut, alpha=alpha, beta=beta)
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(crop, -1, KERNEL)

    def close(self) -> None:
        """Release the VLM connection."""
        self.llm.close()

    ## Parked: concept refinement ######################################################################################
    #
    # `segment` used to send every concept that matched several regions back to
    # the VLM to be narrowed. Merging touching regions of one concept covers the
    # case it was built for, and the loop had a habit of answering an ambiguous
    # phrase with synonyms of it rather than with distinct objects, which grew the
    # concept list without improving the masks.
    #
    # Kept because splitting a concept into genuine sub-concepts is still worth
    # trying. To wire it back in, restore this at the end of `segment`, before the
    # cache is written, with `derived = concepts is None` recorded before the
    # inventory call:
    #
    #     if derived and self.max_refinements:
    #         masks, bboxes, labels = self._refine(
    #             image_pil, image_np, concepts, (masks, bboxes, labels), idx
    #         )
    #
    # One thing to fix first: `_ambiguity` scores surplus masks per concept, so
    # splitting one ambiguous phrase into two synonyms lowers the score without
    # improving anything. That is the incentive that made it run away. Scoring
    # overlapping masks instead is not gameable that way.

    @staticmethod
    def _ambiguous(labels: List[str]) -> Dict[str, int]:
        """Return the concepts that matched more than one region, and how many.

        Several regions for one phrase usually mean the phrase never settled what
        counts as one thing, so SAM 3 matched it against a part, against the whole
        containing that part, and against everything in between. It is also what
        several genuine copies of the same object look like, which no rephrasing
        can reduce; the loop in :meth:`_refine` is bounded so that case costs a
        fixed number of queries instead of running forever.

        Parameters
        ----------
        labels : list[str]
            The concept each mask was found by.

        Returns
        -------
        dict[str, int]
            Concepts with more than one mask, mapped to their mask count, most
            ambiguous first.
        """
        counts = Counter(labels)
        return dict(
            sorted(
                ((c, n) for c, n in counts.items() if n > 1),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    @classmethod
    def _ambiguity(cls, labels: List[str]) -> int:
        """Return how many masks are surplus to one per concept.

        Counting *how many* phrases are ambiguous would rate a phrase matching
        nine regions the same as one matching two, so a refinement that halved the
        over-segmentation without eliminating it would not register as progress.

        Parameters
        ----------
        labels : list[str]
            The concept each mask was found by.

        Returns
        -------
        int
            Sum over ambiguous concepts of their mask count minus one. Zero when
            every concept matched exactly one region.
        """
        return sum(count - 1 for count in cls._ambiguous(labels).values())

    @staticmethod
    def _apply_replacements(
        concepts: Sequence[str], replacements: Dict[str, List[str]]
    ) -> List[str]:
        """Swap every replaced phrase for the phrases replacing it.

        Parameters
        ----------
        concepts : sequence of str
            Phrases to rewrite.
        replacements : dict[str, list[str]]
            Mapping from a phrase to the narrower phrases taking its place.

        Returns
        -------
        list[str]
            The rewritten list, order preserved and deduplicated.
        """
        rewritten: List[str] = []
        for concept in concepts:
            rewritten.extend(replacements.get(concept, [concept]))
        return list(dict.fromkeys(rewritten))

    def _narrow(self, image: Image.Image, ambiguous: Dict[str, int]) -> Dict[str, List[str]]:
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

    def _refine(
        self,
        image_pil: Image.Image,
        image_np: np.ndarray,
        concepts: List[str],
        result: Tuple[np.ndarray, List[List[int]], List[str]],
        idx: int,
    ) -> Tuple[np.ndarray, List[List[int]], List[str]]:
        """Narrow the phrases that matched several regions, and segment again.

        Runs at most :attr:`max_refinements` rounds, and keeps the attempt that
        left the fewest ambiguous phrases rather than the last one, so a round
        that makes matters worse cannot be what the caller ends up with.

        Parameters
        ----------
        image_pil : PIL.Image.Image
            The whole scene, for the VLM.
        image_np : numpy.ndarray
            The same image as an array, for SAM 3.
        concepts : list[str]
            Phrases the first attempt used.
        result : tuple
            What the first attempt produced, as ``(masks, bboxes, labels)``.
        idx : int
            Image index, used to name the debug dumps.

        Returns
        -------
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        labels : list[str]
            The concept each mask was found by.
        """
        masks, bboxes, labels = result
        best = result
        best_concepts = list(concepts)
        best_task_relevant = list(self.task_relevant)
        best_score = self._ambiguity(labels)

        for attempt in range(1, self.max_refinements + 1):
            ambiguous = self._ambiguous(labels)
            if not ambiguous:
                break

            logger.info(
                f"Refinement {attempt}/{self.max_refinements}: narrowing "
                + ", ".join(f"{c!r} ({n} regions)" for c, n in ambiguous.items())
            )

            replacements = self._narrow(image_pil, ambiguous)
            if not replacements:
                logger.warning("No usable replacement was proposed; keeping the current phrases.")
                break

            refined = self._apply_replacements(concepts, replacements)
            if set(refined) == set(concepts):
                logger.warning("Refinement left the phrases unchanged; stopping.")
                break

            concepts = refined
            self.task_relevant = self._apply_replacements(self.task_relevant, replacements)
            logger.debug(f"Refined concepts: {concepts}")

            masks, bboxes, labels = self._predict(
                image_np, concepts, tag=f"image_{idx + 1}_refined_{attempt}"
            )

            score = self._ambiguity(labels)
            if score < best_score:
                best = (masks, bboxes, labels)
                best_concepts = list(concepts)
                best_task_relevant = list(self.task_relevant)
                best_score = score
            if score == 0:
                break

        self.last_concepts = best_concepts
        self.task_relevant = best_task_relevant

        # The dumps are written once per round and the best round wins, which is
        # not necessarily the last, so the log has to say which one it kept.
        logger.info(
            f"Refinement kept {len(best[2])} masks from {len(best_concepts)} concepts: "
            f"{best_concepts}"
        )

        if best_score:
            still = self._ambiguous(best[2])
            logger.warning(
                f"After {self.max_refinements} refinement(s), {len(still)} phrase(s) still match "
                f"several regions: {', '.join(f'{c!r} ({n})' for c, n in still.items())}. "
                "Genuine copies of the same object look identical to an ambiguous phrase."
            )

        return best
