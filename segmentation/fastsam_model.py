"""FastSAM segmentation for the pipeline.

FastSAM is not a SAM architecture: it is a YOLOv8-seg model that emits every mask
in one forward pass.

"""

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
from PIL import Image
from skimage import measure
from skimage.morphology import dilation, disk, erosion, remove_small_objects
from ultralytics import FastSAM

from segmentation.segmentation import SegmentationModel, binary_mask_to_pil
from utility.utility import logger

# FastSAM checkpoints published by Ultralytics.
FASTSAM_CHECKPOINTS = ("FastSAM-s.pt", "FastSAM-x.pt")

# FastSAM's own default is 640.
DEFAULT_IMGSZ = 1024

# Area band, in percent of the image, for a mask to be a plausible object. The
# upper bound is generous because the robot arm spans a large part of the frame;
# anything above it is a region (wall, table, whole frame) rather than a thing.
OBJECT_MIN_AREA_PCT = 0.2
OBJECT_MAX_AREA_PCT = 35.0

# A mask touching this many image borders spans the frame and is scenery rather
# than an object.
BORDER_TOUCH_LIMIT = 3

# Fraction of a mask that must fall inside the region `obtain_bg` kept for the
# mask to count as an object there. This is a containment test, not an IoU: the
# keep-region is far larger than any single object, so IoU against it is
# dominated by the size difference and says little about overlap.
MIN_INSIDE_KEEP_FRACTION = 0.5

# During de-duplication a mask is dropped when this much of it is already covered
# by a larger kept mask. This is what collapses FastSAM's per-face masks (the top
# and front of a brick) into the single whole-brick mask that contains them.
MAX_CONTAINED_FRACTION = 0.7

# Connected components smaller than this many pixels are dropped while cleaning
# masks up.
MIN_REGION_SIZE = 3500


class FastSAMModel(SegmentationModel):
    """Perform segmentation using FastSAM via Ultralytics."""

    def __init__(
        self,
        fastsam_checkpoint: Union[str, Path] = "FastSAM-s.pt",
        device: str = "cuda",
        save_dir: Optional[Union[str, Path]] = None,
        debug_masks: bool = False,
        **predict_kwargs: Any,
    ) -> None:
        """Initialize the FastSAM model.

        Parameters
        ----------
        fastsam_checkpoint : str or Path
            Checkpoint to load, as a bare name from :data:`FASTSAM_CHECKPOINTS`
            (downloaded by Ultralytics on first use) or a path to a local file
            whose name ends with one of them.
        device : str
            Device used for inference. Common values are ``"cuda"`` and
            ``"cpu"``. Default is ``"cuda"``.
        save_dir : str or Path or None
            Directory where the output images will be saved if not None.
        debug_masks : bool
            If True, dump every mask FastSAM returns, before any filtering, under
            ``save_dir/debug/``. Requires ``save_dir``. Default is False.
        predict_kwargs : dict
            Additional keyword arguments for the FastSAM prediction call. FastSAM
            has no ``points_stride``; the equivalent knobs are ``conf``, ``iou``
            and ``imgsz``.

        Raises
        ------
        ValueError
            If ``fastsam_checkpoint`` is not a FastSAM checkpoint, or if
            ``debug_masks`` is set without a ``save_dir``.
        """
        super().__init__(save_dir=save_dir)

        checkpoint = str(fastsam_checkpoint)
        if not checkpoint.endswith(FASTSAM_CHECKPOINTS):
            raise ValueError(
                f"Unsupported FastSAM checkpoint {checkpoint!r}. The file name must end with one "
                f"of {', '.join(FASTSAM_CHECKPOINTS)}. For SAM checkpoints, use SAMModel instead."
            )

        if debug_masks and self.save_dir is None:
            raise ValueError("debug_masks=True requires save_dir to be set.")
        self.debug_masks = debug_masks

        # Without retina_masks the masks come back at the letterboxed model size
        # (576x1024 for a 1920x1080 frame) instead of the original resolution,
        # which the rest of this class assumes. Silently misaligned masks are
        # worse than a slower call, so this is not left to the caller.
        if predict_kwargs.get("retina_masks") is False:
            logger.warning(
                "retina_masks=False would return masks at the letterboxed size instead of the "
                "original resolution; forcing it back to True."
            )
        predict_kwargs["retina_masks"] = True

        predict_kwargs.setdefault("imgsz", DEFAULT_IMGSZ)
        predict_kwargs["device"] = device
        predict_kwargs["verbose"] = False
        # Ultralytics otherwise writes annotated copies to runs/ on every call.
        predict_kwargs["save"] = False
        self.predict_kwargs = predict_kwargs

        self.model = FastSAM(checkpoint)

    def _generate(
        self, image: np.ndarray, tag: Optional[str] = None
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Run FastSAM over a whole image and return every mask it produces.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image with shape ``(H, W, 3)``.
        tag : str or None
            Name used for the debug dump when ``debug_masks`` is enabled. Ignored
            otherwise.

        Returns
        -------
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``, at the resolution of
            ``image``. Empty with shape ``(0, H, W)`` when nothing is found.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        """
        # Ultralytics assumes a numpy source is BGR (its preprocess step flips
        # the channels back before normalisation), while the rest of this file
        # works in RGB. Handing it RGB feeds the network channel-swapped images.
        bgr = np.ascontiguousarray(image[..., ::-1])
        # The call is typed as possibly returning a generator (it does when
        # stream=True, which is not used here), so materialise it first. Typed as
        # Any because Ultralytics' annotations do not match runtime here: the
        # union includes Tensor, and `masks.data` is a torch.Tensor despite being
        # annotated as an ndarray.
        result: Any = list(self.model(bgr, **self.predict_kwargs))[0]

        if result.masks is None:
            logger.warning(f"FastSAM returned no masks for {tag or 'image'}")
            return np.zeros((0, *image.shape[:2]), dtype=bool), []

        masks = result.masks.data.cpu().numpy().astype(bool)

        # Ultralytics reports boxes as xyxy; `boxes.xywh` is centre-based, while
        # the rest of the pipeline (GPT annotator, evaluation) expects the corner
        # based [x_min, y_min, width, height].
        xyxy = result.boxes.xyxy.cpu().numpy()
        bboxes = [
            [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            for x_min, y_min, x_max, y_max in xyxy
        ]

        if self.debug_masks and tag is not None:
            self._dump_masks(image, masks, bboxes, tag)

        return masks, bboxes

    def _dump_masks(
        self, image: np.ndarray, masks: np.ndarray, bboxes: List[List[int]], tag: str
    ) -> None:
        """Write every raw FastSAM mask to ``save_dir/debug/`` for inspection.

        Produces one colour-coded overlay of all masks at once, one binary PNG
        per mask, and a log line per mask with its area and bounding box, so an
        object FastSAM never produced can be told apart from one the later
        filtering removed.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image the masks were generated from, shape ``(H, W, 3)``.
        masks : numpy.ndarray
            Boolean masks with shape ``(N, H, W)``.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        tag : str
            Sub-directory name, e.g. ``"image_1_bg"``.
        """
        debug_dir = Path(cast(Path, self.save_dir)) / "debug" / tag
        debug_dir.mkdir(parents=True, exist_ok=True)

        H, W = image.shape[:2]
        overlay = image.copy()
        # Fixed hue steps keep neighbouring masks visually distinct.
        colours = [
            cv2.cvtColor(
                np.array([[[int(180 * i / max(len(masks), 1)), 255, 255]]], dtype=np.uint8),
                cv2.COLOR_HSV2RGB,
            )[0, 0].astype(np.uint16)
            for i in range(len(masks))
        ]

        logger.debug(f"[{tag}] FastSAM returned {len(masks)} raw masks")
        for i, (mask, colour) in enumerate(zip(masks, colours)):
            area_pct = np.sum(mask) * 100 / (H * W)
            logger.debug(f"[{tag}]   mask {i:03d}: area {area_pct:6.2f}%  bbox {bboxes[i]}")

            overlay[mask] = (overlay[mask] // 2 + colour // 2).astype(np.uint8)
            binary_mask_to_pil(mask.astype(np.uint8)).save(debug_dir / f"mask_{i:03d}.png")

        # Number each mask at its centroid so the overlay maps back to the files.
        for i, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            cv2.putText(
                overlay,
                str(i),
                (int(xs.mean()), int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        Image.fromarray(overlay).save(debug_dir.parent / f"{tag}_all_masks.png")
        logger.debug(f"[{tag}] wrote overlay and {len(masks)} masks to {debug_dir}")

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Smooth a binary mask and drop its small connected components.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask with shape ``(H, W)``.

        Returns
        -------
        numpy.ndarray
            Cleaned binary mask with shape ``(H, W)`` and values ``0`` and ``1``.
        """
        mask_bin = (mask > 0).astype(np.uint8)
        mask_blur = cv2.GaussianBlur(mask_bin * 255, (7, 7), 4)
        mask_blur = (mask_blur > 0).astype(np.uint8)

        # Find groups of connected non-zero pixels and drop the small ones.
        label_image = cast(np.ndarray, measure.label(mask_blur))
        label_image = remove_small_objects(label_image > 0, min_size=MIN_REGION_SIZE)

        label_image = erosion(label_image, disk(9))
        label_image = dilation(label_image, disk(3))

        label_image = cast(np.ndarray, measure.label(label_image))
        return (label_image > 0).astype(np.uint8)

    def _preprocess_mask(
        self, mask: np.ndarray, rgb: Image.Image, f: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clean the background mask and knock it out of the RGB image.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary background mask with shape ``(H, W)``.
        rgb : PIL.Image.Image
            RGB image.
        f : int, optional
            Index of the image, used for saving the masked RGB for visualization.

        Returns
        -------
        masked_rgb : numpy.ndarray
            The RGB image with the background zeroed, shape ``(H, W, 3)``.
        mask_bin : numpy.ndarray
            The keep-mask applied to the RGB, shape ``(H, W, 1)`` with values
            ``0`` and ``1``.

        Raises
        ------
        ValueError
            If ``f`` is given but no ``save_dir`` was configured.
        """
        mask_bin = self._clean_mask(mask)[..., None]
        # Invert: what was background becomes the region to zero out.
        mask_bin = 1 - mask_bin
        masked_rgb = np.asarray(rgb) * mask_bin

        if f is not None:
            if self.save_dir is None:
                raise ValueError("save_dir must be specified to save the masked RGB image.")
            save_path = Path(self.save_dir) / f"image_{f + 1}"
            save_path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(masked_rgb.astype("uint8")).save(save_path / "masked_rgb.png")

        return masked_rgb, mask_bin

    def _cropping_mask(
        self, masks: np.ndarray, rgb: np.ndarray, alpha: float = 1.4, beta: float = 25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop a mask and matching RGB region, then upscale and sharpen it.

        Parameters
        ----------
        masks : numpy.ndarray
            Binary mask used to find the crop bounds.
        rgb : numpy.ndarray
            RGB image from which the matching crop is extracted.
        alpha : float, optional
            Contrast multiplier passed to ``cv2.convertScaleAbs``.
        beta : float, optional
            Brightness offset passed to ``cv2.convertScaleAbs``.

        Returns
        -------
        mask_crop : numpy.ndarray
            Cropped and upscaled binary mask with values ``0`` and ``255``.
        rgb_crop : numpy.ndarray
            Cropped, upscaled, contrast-adjusted, and sharpened RGB image.
        """
        cleaned = self._clean_mask(np.array(masks))

        rgb = np.array(rgb)
        ys, xs = np.where(cleaned > 0)
        top_y = ys.min() + 3
        bot_y = ys.max() + 3
        left_x = xs.min() + 3
        right_x = xs.max() + 3

        mask_crop = cleaned[top_y:bot_y, left_x:right_x].astype(np.uint8) * 255
        mask_crop = cv2.resize(mask_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255  # for being binary

        mask_rgb = rgb[top_y:bot_y, left_x:right_x, :]
        rgb_crop = cv2.convertScaleAbs(mask_rgb, alpha=alpha, beta=beta)
        KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        rgb_crop = cv2.resize(rgb_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        rgb_crop = cv2.filter2D(rgb_crop, -1, KERNEL)

        return mask_crop, rgb_crop

    @staticmethod
    def _border_touches(mask: np.ndarray) -> int:
        """Count how many image borders a mask reaches.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask with shape ``(H, W)``.

        Returns
        -------
        int
            Number of the four image edges the mask has pixels on, 0 to 4.
        """
        return int(mask[0, :].any() + mask[-1, :].any() + mask[:, 0].any() + mask[:, -1].any())

    def _suppress_contained(self, masks: List[np.ndarray]) -> List[int]:
        """Drop masks that are largely contained in a bigger kept one.

        FastSAM emits a mask per visible face as well as one for the whole
        object, so plain IoU de-duplication keeps both: a brick's top face and
        the whole brick overlap little relative to their union. Containment is
        the right test - the face sits almost entirely inside the whole - and
        working from the largest mask down keeps the whole and drops the faces.

        Parameters
        ----------
        masks : list[numpy.ndarray]
            Candidate boolean masks with shape ``(H, W)``.

        Returns
        -------
        list[int]
            Indices into ``masks`` that survive, largest first.
        """
        order = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)

        keep: List[int] = []
        for i in order:
            area = masks[i].sum()
            if area == 0:
                continue
            contained = any(
                np.logical_and(masks[i], masks[j]).sum() / area > MAX_CONTAINED_FRACTION
                for j in keep
            )
            if not contained:
                keep.append(i)

        return keep

    def obtain_bg(
        self, image: Union[Image.Image, str, Path], idx: int = 0, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove the background region from an RGB image.

        Takes the background to be everything **no** mask covers, rather than the
        largest masks as SAM's version does. FastSAM is a "thing" detector: it
        proposes object instances and never emits stuff regions like walls or
        tables, so on a typical frame its masks cover under a fifth of the image
        and none of them is the background. Their complement, on the other hand,
        is exactly the background.

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
            RGB image after applying the inverted keep-mask, with shape
            ``(H, W, 3)``.
        mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)`` and values
            ``0`` and ``1``.

        Raises
        ------
        FileNotFoundError
            If the specified image file does not exist.
        TypeError
            If the input image is not a file path, Path object, or PIL Image.
        """
        start = time.time()

        if isinstance(image, (str, Path)):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            image_file = Image.open(image)
            image_np = np.array(image_file)
        elif isinstance(image, Image.Image):
            image_file = image
            image_np = np.array(image)
        else:
            raise TypeError("Input image must be a file path, Path object, or PIL Image.")

        H, W = image_np.shape[:2]
        all_masks, _ = self._generate(image_np, tag=f"image_{idx + 1}_bg")

        # Split the masks by size, then treat as background everything that is
        # either a large region or unmasked entirely, minus anything that looks
        # like an object. The subtraction is what makes this work across FastSAM
        # variants: the small model emits no region masks at all (so background
        # is essentially the complement), while the large one emits region masks
        # that swallow the objects standing on them (so the objects have to be
        # carved back out, or they get removed along with the table).
        object_union = np.zeros((H, W), dtype=bool)
        region_union = np.zeros((H, W), dtype=bool)
        covered = np.zeros((H, W), dtype=bool)
        n_objects = n_regions = 0

        for mask in all_masks:
            covered |= mask
            area_pct = mask.sum() * 100 / (H * W)
            if area_pct > OBJECT_MAX_AREA_PCT:
                region_union |= mask
                n_regions += 1
            elif area_pct >= OBJECT_MIN_AREA_PCT:
                object_union |= mask
                n_objects += 1

        background = ((region_union | ~covered) & ~object_union).astype(np.uint8)
        bg_pct = background.sum() * 100 / (H * W)

        logger.debug(
            f"BG pass: {len(all_masks)} masks ({n_regions} region-sized, {n_objects} "
            f"object-sized) cover {covered.sum() * 100 / (H * W):.1f}% of the image; "
            f"background is {bg_pct:.1f}%"
        )
        if len(all_masks) == 0:
            logger.warning(
                "FastSAM found no masks, so the whole image is treated as background and the "
                "object pass will have nothing to work with."
            )
        elif bg_pct < 1.0:
            logger.warning(
                f"Only {bg_pct:.1f}% of the image was identified as background, so almost "
                "nothing will be removed. Raising `conf` thins the mask set if this is "
                "unexpected."
            )

        masked_rgb, mask_bin = self._preprocess_mask(background, image_file, idx)

        if self.save_dir is not None:
            logger.debug(
                f"Saving applied background mask for image {idx} as "
                f"{self.save_dir}/figure_{idx}_bg_masked_rgb.png"
            )
            Image.fromarray(masked_rgb).save(f"{self.save_dir}/figure_{idx}_bg_masked_rgb.png")
            logger.debug(
                f"Saving binary mask for background for image {idx} as "
                f"{self.save_dir}/figure_{idx}_bg_mask_bin.png"
            )
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
        """Generate and filter individual object masks inside the kept region.

        Parameters
        ----------
        image : PIL.Image.Image or str or Path
            The original RGB image, or a path to it.
        bg_mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)``.
        bg_masked_rgb : numpy.ndarray
            RGB image produced by :meth:`obtain_bg`, with shape ``(H, W, 3)``.
        idx : int
            Image index used in saved crop names.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        rgb_masks : list[numpy.ndarray]
            Cropped RGB images for accepted object masks.
        bboxes : list[list[int]]
            Bounding boxes for accepted masks, each ``[x_min, y_min, width, height]``.

        Raises
        ------
        FileNotFoundError
            If the specified image file does not exist.
        TypeError
            If the input image is not a file path, Path object, or PIL Image.
        """
        start = time.time()

        if isinstance(image, (str, Path)):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            rgb_image = Image.open(Path(image))
        elif isinstance(image, Image.Image):
            rgb_image = image
        else:
            raise TypeError(
                f"Input image must be a file path, Path object, or PIL Image, got {type(image)}."
            )
        rgb = np.asarray(rgb_image)

        H, W = bg_masked_rgb.shape[:2]
        # Drop the last channel of bg_mask_bin if it has 3 channels
        if bg_mask_bin.ndim == 3:
            bg_mask_bin = bg_mask_bin[..., 0]

        masks_fastsam, bboxes_fastsam = self._generate(
            bg_masked_rgb, tag=f"image_{idx + 1}_objects"
        )

        keep_region = bg_mask_bin.astype(bool)
        candidate_ids: List[int] = []
        candidates: List[np.ndarray] = []

        for i, segment in enumerate(masks_fastsam):
            area = segment.sum()
            area_pct = area * 100 / (H * W)
            # Fraction of the mask sitting in the region obtain_bg kept. A mask
            # lying mostly on the removed background is scenery, whatever its size.
            inside = np.logical_and(segment, keep_region).sum() / area if area else 0.0
            borders = self._border_touches(segment)

            if area_pct < OBJECT_MIN_AREA_PCT:
                reason = f"area {area_pct:.2f}% < {OBJECT_MIN_AREA_PCT}%"
            elif area_pct > OBJECT_MAX_AREA_PCT:
                reason = f"area {area_pct:.2f}% > {OBJECT_MAX_AREA_PCT}%, a region not an object"
            elif borders >= BORDER_TOUCH_LIMIT:
                reason = f"touches {borders} image borders, spans the frame"
            elif inside < MIN_INSIDE_KEEP_FRACTION:
                reason = f"only {inside:.0%} of it lies inside the kept region"
            else:
                candidate_ids.append(i)
                candidates.append(segment)
                continue

            if self.debug_masks:
                logger.debug(f"[image_{idx + 1}_objects]   mask {i:03d} dropped: {reason}")

        valid = self._suppress_contained(candidates)

        masks_filtered = [(candidate_ids[i], candidates[i]) for i in valid]
        bboxes_filtered = [bboxes_fastsam[candidate_ids[i]] for i in valid]

        logger.debug(
            f"Object pass: {len(masks_fastsam)} raw masks -> {len(candidates)} plausible "
            f"objects -> {len(masks_filtered)} after containment removal"
        )
        if self.debug_masks:
            dropped = sorted(set(candidate_ids) - {orig for orig, _ in masks_filtered})
            logger.debug(f"Containment removal discarded raw mask ids: {dropped}")

        rgb_masks = []
        for orig_idx, segment in masks_filtered:
            _, rgb_crop = self._cropping_mask(segment, rgb)
            rgb_masks.append(rgb_crop)

            if self.save_dir is not None:
                save_dir = Path(self.save_dir) / f"image_{idx + 1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb_crop).save(save_dir / f"segment_{orig_idx}.png")

        logger.debug(f"Individual masks obtained in {time.time() - start}s")

        return rgb_masks, bboxes_filtered
