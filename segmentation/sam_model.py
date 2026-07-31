import time
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import measure
from skimage.morphology import dilation, disk, erosion, remove_small_objects

from utility.utility import logger


class SAMModel:
    """Perform segmentation using the Segment Anything Model (SAM)."""

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_h",
        device: str = "cuda",
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
        sam_kwargs : dict
            Additional keyword arguments for the SAM mask generator. For example,
            ``points_per_side`` can be specified to control the number of points
            sampled per side of the image. See the SAM documentation for more details.
        """
        self.sam = sam_model_registry[model_type](sam_checkpoint)
        self.sam.to(device=device)

        if "points_per_side" not in sam_kwargs:
            sam_kwargs["points_per_side"] = 32  # Default value if not provided

        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **sam_kwargs)

    def sam_mask_to_pil(self, mask_bool: np.ndarray) -> Image.Image:
        """Convert a SAM boolean mask to a grayscale PIL image.

        Parameters
        ----------
        mask_bool : numpy.ndarray
            Two-dimensional boolean SAM mask with shape ``(H, W)``.

        Returns
        -------
        PIL.Image.Image
            Grayscale mask image with pixel values ``0`` and ``255``.
        """
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        return Image.fromarray(mask_uint8)

    def binary_mask_to_pil(self, mask_bin: np.ndarray) -> Image.Image:
        """Convert a binary mask to a savable grayscale PIL image.

        Parameters
        ----------
        mask_bin : numpy.ndarray
            Binary mask with shape ``(H, W)`` or ``(H, W, 1)``. Values may be
            ``0`` and ``1`` or ``0`` and ``255``.

        Returns
        -------
        PIL.Image.Image
            Two-dimensional grayscale mask image with values ``0`` and ``255``.

        Raises
        ------
        ValueError
            If ``mask_bin`` is a three-dimensional array with more than one
            channel.
        """
        if mask_bin.ndim == 3:
            if mask_bin.shape[2] != 1:
                raise ValueError(f"Expected a single-channel mask, got shape {mask_bin.shape}")
            mask_bin = mask_bin[:, :, 0]

        mask_uint8 = (mask_bin > 0).astype(np.uint8) * 255
        return Image.fromarray(mask_uint8)

    def preprocess_mask(
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

    def cropping_mask(
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
        self, image: str, idx: int, save_dir: Optional[str] = None
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
        save_dir : str or None
            Directory where the output images will be saved if not None.

        Returns
        -------
        masked_rgb : numpy.ndarray
            RGB image after applying the inverted keep-mask, with shape
            ``(H, W, 3)``.
        mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)`` and values
            ``0`` and ``1``.
        """
        start = time.time()
        image_read = Image.open(image)
        image_np = np.array(image_read)
        H, W, D = image_np.shape
        masks_sam = self.mask_generator.generate(image_np)
        all_masks = []
        all_bboxes = []
        del_id = []
        for m in masks_sam:
            all_masks.append(m["segmentation"])
            all_bboxes.append(m["bbox"])

        for i, mask in enumerate(all_masks):
            masked = self.sam_mask_to_pil(mask)
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

        masked_rgb, mask_bin = self.preprocess_mask(union_mask, image_read, idx)
        end = time.time()

        if save_dir is not None:
            logger.debug(
                f"Saving applied background mask for image {idx} as {save_dir}/figure_{idx}_masked_rgb.png"
            )
            Image.fromarray(masked_rgb).save(f"{save_dir}/figure_{idx}_masked_rgb.png")

        if save_dir is not None:
            logger.debug(
                f"Saving the binary background mask for image {idx} as {save_dir}/figure_{idx}_mask_bin.png"
            )

        logger.debug(f"BG mask obtained in {end - start}s")

        return masked_rgb, mask_bin

    def filter_masks_by_iou(
        self,
        masks: List[np.ndarray],
        index: List[int],
        robot_id: List[int],
        iou_threshold: float = 0.01,
        iou_2objectthreshold: float = 0.4,
        iou_maxthreshold: float = 0.6,
        iou_robot_threshold: float = 0.95,
    ) -> List[int]:
        """Erase the redundant masks: if a mask is almost contained in another, the smaller one is removed."""
        keep = []
        removed = set()

        n = len(masks)

        areas = [m.sum() for m in masks]

        for i in range(n):
            if i in removed:
                continue

            for j in range(i + 1, n):
                if j in removed:
                    continue

                inter = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                iou = inter / union if union > 0 else 0
                if i in robot_id:
                    if (
                        j in robot_id
                    ):  # erase for more than one robot, erase this if. This filters extra masks for an unique robot
                        if areas[i] > areas[j]:
                            removed.add(j)
                    elif iou > iou_threshold:
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
        mask_bin: np.ndarray,
        mask_rgb: np.ndarray,
        rgb_path: str,
        idx: int,
        save_dir: Union[str, None] = None,
    ) -> tuple[list[np.ndarray], list[list[int]], list[str]]:
        """Generate and filter individual object masks inside the kept region.

        Parameters
        ----------
        mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)``.
        mask_rgb : numpy.ndarray
            RGB image produced by :meth:`remove_bg`, with shape ``(H, W, 3)``.
        rgb_path : str
            Path to the original RGB image.
        idx : int
            Image index used in saved crop names.
        save_dir : str or None
            Directory where the output images will be saved if not None.

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
        H, W = mask_rgb.shape[:2]
        masks_sam = self.mask_generator.generate(mask_rgb)

        all_masks = []
        all_bboxes = []

        keep = []
        robot_id = []
        rgb = np.asarray(Image.open(rgb_path))
        mask_bin = mask_bin[..., 0]

        for m in masks_sam:
            all_masks.append(m["segmentation"])
            all_bboxes.append(m["bbox"])

        for i, masked in enumerate(all_masks):
            masked = self.sam_mask_to_pil(masked)
            masked = masked.resize((W, H))
            masked = np.asarray(masked)

            intersection = np.logical_and(masked, mask_bin)
            union = np.logical_or(masked, mask_bin)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            num_pixels = np.sum(intersection > 0)
            area_mask = num_pixels * 100 / (H * W)
            if (0.35 < area_mask < 6.5 or area_mask > 10) and iou > 0.02:  # area min estaba 0.35
                if area_mask > 10:
                    robot_id.append(i)
                keep.append(i)

        masks = [(i, all_masks[i]) for i in keep]
        bboxes = [all_bboxes[i] for i in keep]
        masks_only = [m[1] for m in masks]
        index = [m[0] for m in masks]
        valid = self.filter_masks_by_iou(masks_only, index, robot_id, iou_threshold=0.01)

        masks_filtered = [masks[i] for i in valid]
        bboxes_filtered = [bboxes[i] for i in valid]

        rgb_masks = []
        masks_path = []

        for i, (orig_idx, masked) in enumerate(masks_filtered):
            save_path = f"ppt_outputs/image{idx + 1}/crop_{orig_idx}.png"
            masks_path.append(save_path)
            mask_crop, rgb_crop = self.cropping_mask(masked, rgb)
            rgb_masks.append(rgb_crop)
            Image.fromarray(rgb_crop).save(save_path)

        end = time.time()
        logger.debug(f"Individual masks obtained in {end - start}s")

        return rgb_masks, bboxes_filtered, masks_path
