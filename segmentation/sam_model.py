import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import measure
from skimage.morphology import dilation, disk, erosion, remove_small_objects

from segmentation.segmentation import SegmentationModel, binary_mask_to_pil, mask_to_pil
from utility.utility import logger


class SAMModel(SegmentationModel):
    """Perform segmentation using the Segment Anything Model (SAM)."""

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        save_dir: Optional[Union[str, Path]] = None,
        **sam_kwargs: Any,
    ) -> None:
        """Initialize the SAM mask generator.

        Parameters
        ----------
        sam_checkpoint : str
            Path to the SAM checkpoint.
        model_type : str
            SAM model type. Expected values are ``"vit_b"``, ``"vit_l"``, or
            ``"vit_h"``.
        device : str
            Device used for inference. Common values are ``"cuda"`` and
            ``"cpu"``. Default is ``"cuda"``.
        save_dir : str or Path or None
            Directory where the output images will be saved if not None.
        sam_kwargs : dict
            Additional keyword arguments for the SAM mask generator. For example,
            ``points_per_side`` can be specified to control the number of points
            sampled per side of the image. See the SAM documentation for more details.
        """
        super().__init__(save_dir=save_dir)
        self.sam = sam_model_registry[model_type](sam_checkpoint)
        self.sam.to(device=device)

        if "points_per_side" not in sam_kwargs:
            sam_kwargs["points_per_side"] = 32  # Default value if not provided

        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **sam_kwargs)

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
            ref_img.save(f"ppt_outputs/image{f + 1}/masked_rgb.png")
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
        masks_sam = self.mask_generator.generate(image_np)
        all_masks = []
        all_bboxes = []
        del_id = []
        for m in masks_sam:
            all_masks.append(m["segmentation"])
            all_bboxes.append(m["bbox"])

        for i, mask in enumerate(all_masks):
            masked = mask_to_pil(mask)
            masked = masked.resize((W, H))
            masked_np = np.array(masked)
            num_pixels = np.sum(masked_np > 0)
            area_mask = num_pixels * 100 / (H * W)
            if area_mask < 15:
                del_id.append(i)
        masks = np.delete(all_masks, del_id, axis=0)
        h, w = masks[0].shape
        union_mask = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            union_mask |= m

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
    ) -> tuple[list[np.ndarray], list[list[int]], list[str]]:
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
        masks_path : list[str]
            Paths where accepted cropped object images were saved.
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

        masks_sam = self.mask_generator.generate(bg_masked_rgb)

        numbered_masks = []
        bboxes = []
        robot_ids = []

        for i in range(len(masks_sam)):
            segment = mask_to_pil(masks_sam[i]["segmentation"])
            segment = np.asarray(segment.resize((W, H)))

            intersection = np.logical_and(segment, bg_mask_bin)
            union = np.logical_or(segment, bg_mask_bin)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            num_pixels = np.sum(intersection > 0)
            area_mask = num_pixels * 100 / (H * W)
            if (0.35 < area_mask < 6.5 or area_mask > 10) and iou > 0.02:
                if area_mask > 10:
                    robot_ids.append(i)
                numbered_masks.append((i, masks_sam[i]["segmentation"]))
                bboxes.append(masks_sam[i]["bbox"])

        valid = self._filter_masks_by_iou(numbered_masks, robot_ids, iou_threshold=0.01)

        masks_filtered = [numbered_masks[i] for i in valid]
        bboxes_filtered = [bboxes[i] for i in valid]

        rgb_masks = []
        segment_paths = []

        for i, (orig_idx, segment) in enumerate(masks_filtered):
            mask_crop, rgb_crop = self._cropping_mask(segment, rgb)
            rgb_masks.append(rgb_crop)

            if self.save_dir is not None:
                save_dir = Path(self.save_dir) / f"image_{idx + 1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"segment_{orig_idx}.png"
                segment_paths.append(save_path)
                Image.fromarray(rgb_crop).save(save_path)

        end = time.time()
        logger.debug(f"Individual masks obtained in {end - start}s")

        return rgb_masks, bboxes_filtered, segment_paths
