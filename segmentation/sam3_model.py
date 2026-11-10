"""SAM 3 concept segmentation for the pipeline.

SAM 3 is not driven the way SAM 1, SAM 2 and FastSAM are. Those three are asked
to segment *everything* and hand back an unlabelled pile of masks, which
``SAMModel`` and ``FastSAMModel`` then filter down to plausible objects with
area, border and containment heuristics. SAM 3 is asked for *concepts*: it takes
short noun phrases and returns every instance matching each one, already
attributed to the phrase that found it.

That inverts the pipeline. The heuristics exist to compensate for masks having no
semantics, so most of them have nothing left to do here; what matters instead is
the quality of the phrases. The work comes in two halves:

1. A VLM is shown the scene and asked for the things in it, as phrases SAM 3 can
   ground. That is :class:`scene_understanding.inventory.ConceptInventory`, which
   owns the prompt and the phrasing examples.
2. Those phrases are passed to Ultralytics' ``SAM3SemanticPredictor``, and the
   masks come back labelled with the concept that produced them. That is this
   class, which takes the inventory as its first argument.

The labels are the point: :meth:`SAM3Model.segment` returns them alongside the
masks, so the annotator downstream no longer has to work out *what* each crop is
and can spend its call on describing it instead.

The debug dumps and the HTML viewer live in :mod:`segmentation.sam3_report`.

Requires ``ultralytics >= 8.3.237`` and the gated ``sam3.pt`` checkpoint; see
``scripts/install_models.py`` for how to obtain it.
"""

import gc
import tempfile
import time
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor

from scene_understanding.inventory import DEFAULT_EXAMPLES_FILE, ConceptInventory
from segmentation import sam3_report
from segmentation.segmentation import SegmentationModel
from utility.masks import binary_mask_to_pil
from utility.utility import logger

# The only checkpoint SAM 3 ships. Kept as a tuple for symmetry with
# `SAM_CHECKPOINTS` and `FASTSAM_CHECKPOINTS`, and because the check below is a
# suffix match, so a local path such as "models/sam/sam3.pt" works too.
SAM3_CHECKPOINTS = ("sam3.pt",)

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


class SAM3Model(SegmentationModel):
    """Segment a scene by concept, using a VLM to decide what the concepts are.

    The VLM runs first and names the things in the image; SAM 3 then segments
    exactly those. Both steps are hidden behind :meth:`segment`::

        inventory = ConceptInventory(llm, task="pick up the wrench")
        model = SAM3Model(inventory, "models/sam/sam3.pt")
        masks, bboxes, labels = model.segment(image)

    Pass ``concepts=`` to the constructor, or call :meth:`set_concepts`, to skip
    the VLM entirely and segment a fixed vocabulary; the inventory may then be None.

    Attributes
    ----------
    inventory : ConceptInventory or None
        Asks the VLM for the scene inventory. None when only fixed concepts are used.
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
        inventory: Optional[ConceptInventory],
        sam3_checkpoint: Union[str, Path] = "sam3.pt",
        device: str = "cuda",
        save_dir: Optional[Union[str, Path]] = None,
        debug_masks: bool = False,
        concepts: Optional[Sequence[str]] = None,
        merge_gap: int = DEFAULT_MERGE_GAP,
        max_refinements: int = DEFAULT_MAX_REFINEMENTS,
        filters: Optional[Mapping[str, float]] = None,
        **predict_kwargs: Any,
    ) -> None:
        """Initialize the SAM 3 model.

        Parameters
        ----------
        inventory : ConceptInventory or None
            Asks the VLM what is in each scene. None is only allowed together with
            ``concepts``.
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
        filters : Mapping[str, float] or None
            Overrides of :attr:`DEFAULT_FILTERS`. SAM 3 declares none: it segments what
            it was asked for, so there is nothing to filter by size or overlap.
        predict_kwargs : dict
            Additional overrides for the Ultralytics predictor, e.g. ``conf``,
            ``iou`` or ``imgsz``. Check the documentation of
            ``ultralytics.models.sam.SAM3SemanticPredictor`` for the full list.

        Raises
        ------
        ValueError
            If ``sam3_checkpoint`` is not a SAM 3 checkpoint, if neither an
            inventory nor concepts are given, or if ``debug_masks`` is set without
            a ``save_dir``.
        """
        super().__init__(save_dir=save_dir, filters=filters)

        # Release what earlier models left behind before loading SAM 3. Collecting
        # first drops unreferenced tensors, so the cache they sat in can be freed.
        # This only returns memory cached by this process: whatever another
        # process holds, such as a vLLM server, stays where it is.
        if str(device).startswith("cuda") and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        checkpoint = str(sam3_checkpoint)
        if not checkpoint.endswith(SAM3_CHECKPOINTS):
            raise ValueError(
                f"Unsupported SAM 3 checkpoint {checkpoint!r}. The file name must end with one of "
                f"{', '.join(SAM3_CHECKPOINTS)}. For SAM 1 and SAM 2 use SAMModel, for FastSAM "
                "use FastSAMModel."
            )

        if inventory is None and concepts is None:
            raise ValueError("SAM3Model needs a ConceptInventory unless fixed concepts are given.")

        if debug_masks and self.save_dir is None:
            raise ValueError("debug_masks=True requires save_dir to be set.")
        self.debug_masks = debug_masks

        self.inventory = inventory
        self.merge_gap = max(0, int(merge_gap))
        self.max_refinements = max(0, int(max_refinements))
        if self.max_refinements:
            logger.warning(
                f"max_refinements={self.max_refinements} has no effect: the refinement loop is "
                "parked at the end of segmentation/sam3_model.py and segment() does not call it."
            )
        self.concepts: Optional[List[str]] = list(concepts) if concepts is not None else None
        self.task_relevant: List[str] = []

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

        self.predict_kwargs = dict(predict_kwargs)
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
        task: Optional[str] = None,
        examples_file: Optional[Union[str, Path]] = DEFAULT_EXAMPLES_FILE,
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
        task : str or None
            The task the robot has to carry out; see :class:`ConceptInventory`.
        examples_file : str or Path or None
            Phrasing pairs rendered into the inventory prompt.
        kwargs : dict
            Forwarded to :meth:`__init__`, e.g. ``save_dir``.

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
        inventory = ConceptInventory.from_config(
            llm_config_file,
            llm_overrides,
            task=task,
            examples_file=examples_file,
            save_dir=kwargs.get("save_dir"),
        )
        return cls(inventory, sam3_checkpoint, **kwargs)

    @property
    def llm(self) -> Any:
        """The inventory's VLM, or None without an inventory."""
        return self.inventory.llm if self.inventory is not None else None

    @property
    def task(self) -> Optional[str]:
        """The inventory's task, or None without an inventory."""
        return self.inventory.task if self.inventory is not None else None

    def effective_params(self) -> Dict[str, Any]:
        """Return the arguments forwarded to the SAM 3 predictor.

        Returns
        -------
        Dict[str, Any]
            The prediction arguments, including the defaults filled in here.
        """
        return dict(self.predict_kwargs)

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
            assert self.inventory is not None  # enforced by __init__
            concepts, self.task_relevant = self.inventory.concepts_for(image_pil)

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
            sam3_report.dump_masks(cast(Path, self.save_dir), image, masks, bboxes, labels, tag)

        return masks, bboxes, labels

    ## Visualisation ###################################################################################################

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
            sam3_report.render_view(
                image_np,
                masks,
                bboxes,
                labels,
                idx,
                concepts=self.last_concepts,
                task=self.task,
                task_relevant=self.task_relevant,
            ),
            encoding="utf-8",
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

        # Kept for the annotators that describe a masked region rather than a crop.
        self.last_masks = list(masks)

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
        # rather than the object plus whatever shares its bounding box. The mask
        # is cast rather than left boolean because `cv2.convertScaleAbs` below
        # takes no boolean source: numpy already widens the product to uint8,
        # this only makes that explicit.
        cut = rgb[top:bottom, left:right] * mask[top:bottom, left:right, None].astype(np.uint8)

        crop = cv2.convertScaleAbs(cut, alpha=alpha, beta=beta)
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(crop, -1, KERNEL)

    def close(self) -> None:
        """Release the VLM connection."""
        if self.inventory is not None:
            self.inventory.close()

    ## Parked: concept refinement ######################################################################################
    #
    # `segment` used to send every concept that matched several regions back to
    # the VLM to be narrowed. Merging touching regions of one concept covers the
    # case it was built for, and the loop had a habit of answering an ambiguous
    # phrase with synonyms of it rather than with distinct objects, which grew the
    # concept list without improving the masks.
    #
    # The VLM half of the loop (`ConceptInventory.narrow`, `REFINEMENT_PROMPT`) moved
    # to scene_understanding/inventory.py with the rest of the inventory.
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

            assert self.inventory is not None  # refinement only runs on VLM concepts
            replacements = self.inventory.narrow(image_pil, ambiguous)
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
