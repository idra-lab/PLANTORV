import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
from PIL import Image
from skimage import measure
from skimage.morphology import dilation, disk, erosion, remove_small_objects
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM2Predictor

from segmentation.segmentation import SegmentationModel, binary_mask_to_pil
from utility.utility import logger

# Checkpoint names Ultralytics knows how to build a SAM model from. `build_sam`
# selects the architecture by matching the *end* of the checkpoint path against
# these, so "models/sam/sam_h.pt" works but Meta's original "sam_vit_h_4b8939.pth"
# does not. Of the SAM 1 names, only sam_b.pt and sam_l.pt are downloaded
# automatically; sam_h.pt and mobile_sam.pt have to be present locally. Every
# SAM 2 checkpoint is published by Ultralytics and downloads on first use.
SAM1_CHECKPOINTS = ("sam_h.pt", "sam_l.pt", "sam_b.pt", "mobile_sam.pt")
SAM2_CHECKPOINTS = (
    "sam2_t.pt",
    "sam2_s.pt",
    "sam2_b.pt",
    "sam2_l.pt",
    "sam2.1_t.pt",
    "sam2.1_s.pt",
    "sam2.1_b.pt",
    "sam2.1_l.pt",
)
SAM_CHECKPOINTS = SAM1_CHECKPOINTS + SAM2_CHECKPOINTS


class SAMModel(SegmentationModel):
    """Perform segmentation using the Segment Anything Model via Ultralytics.

    Handles both SAM 1 and SAM 2: the checkpoint name selects the architecture
    and the matching Ultralytics predictor, and both expose the same "segment
    everything" call and result format, so no SAM 2 specific subclass is needed::

        SAMModel("models/sam/sam_h.pt")  # SAM 1
        SAMModel("sam2.1_l.pt")  # SAM 2, downloaded on first use
    """

    def __init__(
        self,
        sam_checkpoint: Union[str, Path] = "sam_h.pt",
        device: str = "cuda",
        save_dir: Optional[Union[str, Path]] = None,
        debug_masks: bool = False,
        **sam_kwargs: Any,
    ) -> None:
        """Initialize the SAM mask generator.

        Parameters
        ----------
        sam_checkpoint : str or Path
            Checkpoint to load. Either a bare name from :data:`SAM_CHECKPOINTS`
            (everything except ``sam_h.pt`` and ``mobile_sam.pt`` is downloaded
            by Ultralytics on first use) or a path to a local file whose name
            ends with one of them. The name selects both the architecture and
            the SAM 1 / SAM 2 predictor, so there is no separate model type
            argument.
        device : str
            Device used for inference. Common values are ``"cuda"`` and
            ``"cpu"``. Default is ``"cuda"``.
        save_dir : str or Path or None
            Directory where the output images will be saved if not None.
        debug_masks : bool
            If True, dump every mask SAM returns, before any filtering, under
            ``save_dir/debug/``. Useful to tell apart "SAM never produced this
            object" from "the filtering in :meth:`individual_mask` discarded it".
            Requires ``save_dir``. Default is False.
        sam_kwargs : dict
            Additional keyword arguments for the Ultralytics "segment everything"
            pass, forwarded to ``Predictor.generate``. For example,
            ``points_stride`` controls the number of points sampled per side of
            the image (the equivalent of ``points_per_side`` in Meta's
            ``SamAutomaticMaskGenerator``).

        Raises
        ------
        ValueError
            If ``sam_checkpoint`` does not end with a name Ultralytics can map to
            a SAM architecture.
        """
        super().__init__(save_dir=save_dir)

        checkpoint = str(sam_checkpoint)
        if not checkpoint.endswith(SAM_CHECKPOINTS):
            raise ValueError(
                f"Unsupported SAM checkpoint {checkpoint!r}. The file name must end with one of "
                f"{', '.join(SAM_CHECKPOINTS)}. Meta's original checkpoints are compatible once "
                "renamed (for example sam_vit_h_4b8939.pth -> sam_h.pt)."
            )

        if "points_stride" not in sam_kwargs:
            sam_kwargs["points_stride"] = 32
        self.generate_kwargs = sam_kwargs

        if debug_masks and self.save_dir is None:
            raise ValueError("debug_masks=True requires save_dir to be set.")
        self.debug_masks = debug_masks

        # SAM 2 needs its own predictor: it overrides how image features are
        # computed. Everything after that - "segment everything" mode, the
        # Results contract, mask post-processing - is inherited unchanged from
        # the SAM 1 predictor, so the rest of this class does not care which is
        # in use.
        predictor_cls = SAM2Predictor if checkpoint.endswith(SAM2_CHECKPOINTS) else SAMPredictor
        self.predictor = predictor_cls(
            overrides={
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "imgsz": 1024,
                "model": checkpoint,
                "device": device,
                "save": False,  # prevent Ultralytics from saving annotated images to runs/
                "verbose": False,
            }
        )

    def _generate(
        self, image: np.ndarray, tag: Optional[str] = None
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Run SAM in "segment everything" mode over a whole image.

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
            Boolean masks with shape ``(N, H, W)``, already at the resolution of
            ``image``. Empty with shape ``(0, H, W)`` when nothing is found.
        bboxes : list[list[int]]
            One box per mask as ``[x_min, y_min, width, height]``.
        """
        # Ultralytics assumes a numpy source is BGR (its preprocess step flips
        # the channels back before normalisation), while the rest of this file
        # works in RGB. Handing it RGB feeds the network channel-swapped images.
        bgr = np.ascontiguousarray(image[..., ::-1])
        result = self.predictor(source=bgr, **self.generate_kwargs)[0]

        if result.masks is None:
            logger.warning(f"SAM returned no masks for {tag or 'image'}")
            return np.zeros((0, *image.shape[:2]), dtype=bool), []

        masks = result.masks.data.cpu().numpy().astype(bool)

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
        """Write every raw SAM mask to ``save_dir/debug/`` for inspection.

        Produces one colour-coded overlay of all masks at once, one binary PNG
        per mask, and a log line per mask with its area and bounding box, so an
        object that never made it out of SAM can be told apart from one that the
        later filtering removed.

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

        logger.debug(f"[{tag}] SAM returned {len(masks)} raw masks")
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

    def _preprocess_mask(
        self, mask: np.ndarray, rgb: Image.Image, f: Optional[Union[int, None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the RGB image before applying SAM for the second time.

        This is done to obtain a better segmentation of the objects that we are looking for.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask to be applied over the RGB image.
        rgb : PIL.Image.Image
            RGB image.
        f : int, optional
            Index of the image, used for saving the masked RGB for visualization.

        Returns
        -------
        numpy.ndarray
            The RGB image with the mask applied. Output is a 3-channel uint8 image (H,W,3)
        - mask_bin: the binary mask that is applied over the RGB. Numpy array. Output is a 3-channel uint8 image (H,W,3) where each channel is the same binary mask.
        """
        mask = mask.astype(np.uint8) * 255
        mask_bin = (mask > 0).astype(np.uint8)
        mask_blur = cv2.GaussianBlur(mask_bin * 255, (7, 7), 4)
        mask_blur = (mask_blur > 0).astype(np.uint8)

        # Find groups of connected non-zero pixels in the mask and remove small objects
        label_image = measure.label(mask_blur)
        label_image = remove_small_objects(label_image, min_size=3500)

        label_image = erosion(label_image, disk(9))
        label_image = dilation(label_image, disk(3))

        label_image = cast(np.ndarray, measure.label(label_image))
        mask_clean = (label_image > 0).astype(np.uint8) * 255
        mask_bin = (mask_clean > 0).astype(np.uint8)[..., None]
        mask_bin = 1 - mask_bin
        masked_rgb = np.asarray(rgb) * mask_bin
        ref_img = Image.fromarray(masked_rgb.astype("uint8"))
        if f is not None:
            if self.save_dir is None:
                raise ValueError("save_dir must be specified to save the masked RGB image.")
            save_path = Path(self.save_dir) / f"image_{f + 1}"
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "masked_rgb.png"
            ref_img.save(f"{save_path}")
        return masked_rgb, mask_bin

    def _cropping_mask(
        self, masks: np.ndarray, rgb: np.ndarray, alpha: float = 1.4, beta: float = 25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop a mask and matching RGB region, then upscale and sharpen it.

        Parameters
        ----------
        masks : numpy.ndarray or PIL.Image.Image
            Binary mask used to find the crop bounds.
        rgb : numpy.ndarray or PIL.Image.Image
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
        masks = np.array(masks)
        mask = masks.astype(np.uint8) * 255
        mask_bin = (mask > 0).astype(np.uint8)
        mask_blur = cv2.GaussianBlur(mask_bin * 255, (7, 7), 4)
        mask_blur = (mask_blur > 0).astype(np.uint8)

        label_image = cast(np.ndarray, measure.label(mask_blur))
        # Convert label_image to a binary mask and remove small objects
        label_image = remove_small_objects((label_image > 0), min_size=3500)

        label_image = erosion(label_image, disk(9))
        label_image = dilation(label_image, disk(3))

        label_image = cast(np.ndarray, measure.label(label_image))
        mask_clean = (label_image > 0).astype(np.uint8) * 255
        masks = (mask_clean > 0).astype(np.uint8)

        rgb = np.array(rgb)
        ys, xs = np.where(masks > 0)
        top_y = ys.min() + 3
        bot_y = ys.max() + 3
        left_x = xs.min() + 3
        right_x = xs.max() + 3

        mask_crop = masks[top_y:bot_y, left_x:right_x]
        mask_crop = mask_crop.astype(np.uint8) * 255
        mask_crop = cv2.resize(mask_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255  # for being binary
        mask_rgb = rgb[top_y:bot_y, left_x:right_x, :]
        rgb_crop = cv2.convertScaleAbs(mask_rgb, alpha=alpha, beta=beta)
        KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        rgb_crop = cv2.resize(rgb_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        rgb_crop = cv2.filter2D(rgb_crop, -1, KERNEL)

        return mask_crop, rgb_crop

    def obtain_bg(
        self, image: Union[Image.Image, str, Path], idx: int = 0, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove the coarse SAM background region from an RGB image.

        The method generates SAM masks on the full image, keeps masks whose
        area is at least 15 percent of the image, unions them, cleans the union,
        inverts the result, and applies that keep-mask to the RGB image.

        Parameters
        ----------
        image_path : str
            Path to the RGB image.
        idx : int
            Image index used when saving ``masked_rgb{idx}.png`` and
            ``mask_bin{idx}.png``.
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

        # Instantiate the image
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

        # Keep only the masks covering at least 15 percent of the image, i.e. the
        # coarse background regions rather than the objects standing on them.
        union_mask = np.zeros((H, W), dtype=np.uint8)
        kept = 0
        for mask in all_masks:
            area_mask = np.sum(mask) * 100 / (H * W)
            if area_mask >= 15:
                union_mask |= mask.astype(np.uint8)
                kept += 1
        logger.debug(f"BG pass: {kept}/{len(all_masks)} masks over the 15% area threshold")

        masked_rgb, mask_bin = self._preprocess_mask(union_mask, image_file, idx)
        end = time.time()

        if self.save_dir is not None:
            logger.debug(
                f"Saving applied background mask for image {idx} as {self.save_dir}/figure_{idx}_bg_masked_rgb.png"
            )
            Image.fromarray(masked_rgb).save(f"{self.save_dir}/figure_{idx}_bg_masked_rgb.png")
            logger.debug(
                f"Saving binary mask for background for image {idx} as {self.save_dir}/figure_{idx}_bg_mask_bin.png"
            )
            binary_mask_to_pil(mask_bin).save(f"{self.save_dir}/figure_{idx}_bg_mask_bin.png")

        logger.debug(f"BG mask obtained in {end - start}s")

        return masked_rgb, mask_bin

    def _filter_masks_by_iou(
        self,
        numbered_masks: List[np.ndarray],
        robot_ids: List[int],
        iou_threshold: float = 0.01,
        iou_2objectthreshold: float = 0.4,
        iou_maxthreshold: float = 0.6,
        iou_robot_threshold: float = 0.01,
    ) -> List[int]:
        """Erase the redundant masks: if a mask is almost contained in another, the smaller one is removed."""
        keep = []
        removed = set()

        n_masks = len(numbered_masks)
        areas = [m[1].sum() for m in numbered_masks]

        for i in range(n_masks):
            if i in removed:
                continue

            for j in range(i + 1, n_masks):
                if j in removed:
                    continue

                inter = np.logical_and(numbered_masks[i][1], numbered_masks[j][1]).sum()
                union = np.logical_or(numbered_masks[i][1], numbered_masks[j][1]).sum()
                iou = inter / union if union > 0 else 0
                if i in robot_ids:
                    if j in robot_ids:
                        if areas[i] > areas[j]:
                            removed.add(j)
                    elif iou > iou_robot_threshold:
                        if areas[i] >= areas[j]:
                            removed.add(j)
                        else:
                            removed.add(i)
                            break
                else:
                    if iou > iou_2objectthreshold:
                        if iou > iou_maxthreshold:
                            if areas[j] > areas[i]:
                                removed.add(j)
                            else:
                                removed.add(i)
                                break
                        else:
                            if areas[i] < areas[j]:
                                removed.add(j)
                            else:
                                removed.add(i)
                                break
                    elif iou > iou_threshold:
                        if areas[i] >= areas[j]:
                            removed.add(j)
                        else:
                            removed.add(i)
                            break

            if i not in removed:
                keep.append(i)

        return keep

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
        bg_mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)``.
        bg_masked_rgb : numpy.ndarray
            RGB image produced by :meth:`remove_bg`, with shape ``(H, W, 3)``.
        rgb_path : str
            Path to the original RGB image.
        idx : int
            Image index used in saved crop names.

        Returns
        -------
        rgb_masks : list[numpy.ndarray]
            Cropped RGB images for accepted object masks.
        bboxes : numpy.ndarray
            Bounding boxes for accepted masks. Each row is
            ``[x_min, y_min, width, height]``.
        """
        start = time.time()

        if isinstance(image, (str, Path)):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            rgb_path = Image.open(Path(image))
        elif isinstance(image, Image.Image):
            rgb_path = image
        else:
            raise TypeError(
                f"Input image must be a file path, Path object, or PIL Image, got {type(image)}."
            )
        rgb = np.asarray(rgb_path)

        H, W = bg_masked_rgb.shape[:2]
        # Drop the last channel of bg_mask_bin if it has 3 channels
        if bg_mask_bin.ndim == 3:
            bg_mask_bin = bg_mask_bin[..., 0]

        masks_sam, bboxes_sam = self._generate(bg_masked_rgb, tag=f"image_{idx + 1}_objects")

        numbered_masks = []
        bboxes = []
        robot_ids = []

        for i, segment in enumerate(masks_sam):
            intersection = np.logical_and(segment, bg_mask_bin)
            union = np.logical_or(segment, bg_mask_bin)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            num_pixels = np.sum(intersection > 0)
            area_mask = num_pixels * 100 / (H * W)
            if (0.35 < area_mask < 6.5 or area_mask > 10) and iou > 0.02:
                if area_mask > 10:
                    robot_ids.append(i)
                numbered_masks.append((i, segment))
                bboxes.append(bboxes_sam[i])
            elif self.debug_masks:
                reason = (
                    f"area∩bg {area_mask:.2f}% outside (0.35, 6.5) and <= 10"
                    if not (0.35 < area_mask < 6.5 or area_mask > 10)
                    else f"iou {iou:.4f} <= 0.02"
                )
                logger.debug(f"[image_{idx + 1}_objects]   mask {i:03d} dropped: {reason}")

        valid = self._filter_masks_by_iou(numbered_masks, robot_ids, iou_threshold=0.01)

        masks_filtered = [numbered_masks[i] for i in valid]
        bboxes_filtered = [bboxes[i] for i in valid]

        logger.debug(
            f"Object pass: {len(masks_sam)} raw masks -> {len(numbered_masks)} after the "
            f"area/IoU thresholds -> {len(masks_filtered)} after overlap removal"
        )
        if self.debug_masks:
            dropped = {i for i, _ in numbered_masks} - {orig for orig, _ in masks_filtered}
            logger.debug(f"Overlap removal discarded raw mask ids: {sorted(dropped)}")

        rgb_masks = []

        for i, (orig_idx, segment) in enumerate(masks_filtered):
            _, rgb_crop = self._cropping_mask(segment, rgb)
            rgb_masks.append(rgb_crop)

            if self.save_dir is not None:
                save_dir = Path(self.save_dir) / f"image_{idx + 1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"segment_{orig_idx}.png"
                Image.fromarray(rgb_crop).save(save_path)

        end = time.time()
        logger.debug(f"Individual masks obtained in {end - start}s")

        # Kept for the annotators that describe a masked region rather than a crop.
        self.last_masks = [segment for _, segment in masks_filtered]

        return rgb_masks, bboxes_filtered
