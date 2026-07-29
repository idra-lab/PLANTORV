"""SAM-based background removal and object crop extraction."""

import time
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import measure
from skimage.morphology import dilation, disk, erosion, remove_small_objects

from utility.logger import logger


class SAMModel:
    """Segment scenes with SAM and prepare object crops for VLM tagging."""

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str,
        device: Optional[str] = "cuda",
        points_per_side: Optional[int] = 16,
    ):
        """Initialize the SAM mask generator.

        Parameters
        ----------
        sam_checkpoint : str
            Path to the SAM checkpoint.
        model_type : str
            SAM model type. Expected values are ``"vit_b"``, ``"vit_l"``, or
            ``"vit_h"``.
        device : str, optional
            Device used for inference. Common values are ``"cuda"`` and
            ``"cpu"``.
        points_per_side : int, optional
            Number of prompt points sampled per image side by the automatic
            mask generator.
        """
        self.sam = sam_model_registry[model_type](sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam, 
            # points_per_side=points_per_side,
            # crop_n_layers=2,
        )

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

    def preprocess_mask(self, mask, rgb, f) -> np.ndarray:
        """Clean and invert a coarse mask before the second SAM pass.

        Parameters
        ----------
        mask : numpy.ndarray
            Coarse binary mask to clean, with shape ``(H, W)``.
        rgb : numpy.ndarray or PIL.Image.Image
            RGB image to multiply by the cleaned mask.
        f : int
            Image index used when saving the visualization.

        Returns
        -------
        masked_rgb : numpy.ndarray
            RGB image after applying the inverted keep-mask, with shape
            ``(H, W, 3)``.
        mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)`` and values
            ``0`` and ``1``.
        """
        mask = mask.astype(np.uint8) * 255
        mask_bin = (mask > 0).astype(np.uint8)
        mask_blur = cv2.GaussianBlur(mask_bin * 255, (7, 7), 4)
        mask_blur = (mask_blur > 0).astype(np.uint8)

        label_image = measure.label(mask_blur)

        label_image = remove_small_objects(label_image, min_size=3501)

        label_image = erosion(label_image, disk(9))
        label_image = dilation(label_image, disk(3))

        label_image = measure.label(label_image)
        mask_clean = (label_image > 0).astype(np.uint8) * 255
        mask_bin = (mask_clean > 0).astype(np.uint8)[..., None]
        mask_bin = 1 - mask_bin
        masked_rgb = rgb * mask_bin
        ref_img = Image.fromarray(masked_rgb.astype("uint8"))
        ref_img.save(f"masked_rgb{f}.png")
        return masked_rgb, mask_bin

    def cropping_mask(self, masks, rgb, alpha=1.4, beta=25):
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
        rgb = np.array(rgb)
        ys, xs = np.where(masks > 0)
        top_y = ys.min()
        bot_y = ys.max() + 1
        left_x = xs.min()
        right_x = xs.max() + 1

        mask_crop = masks[top_y:bot_y, left_x:right_x]
        mask_crop = cv2.resize(mask_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255  # for being binary
        mask_rgb = rgb[top_y:bot_y, left_x:right_x, :]
        rgb_crop = cv2.convertScaleAbs(mask_rgb, alpha=alpha, beta=beta)
        KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        rgb_crop = cv2.resize(rgb_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        rgb_crop = cv2.filter2D(rgb_crop, -1, KERNEL)

        return mask_crop, rgb_crop

    def remove_bg(
        self, image_path: str, idx: int, save_dir: Union[str, None] = None
    ) -> tuple[np.ndarray, np.ndarray]:
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

        image = Image.open(image_path)
        image_np = np.array(image)
        H, W, _ = image_np.shape
        masks_sam = self.mask_generator.generate(image_np)
        all_masks = [m["segmentation"] for m in masks_sam]
        del_id = []

        for i, mask in enumerate(all_masks):
            masked = self.sam_mask_to_pil(mask)
            masked = masked.resize((W, H))
            masked_np = np.array(masked)
            num_pixels = np.sum(masked_np > 0)
            area_mask = num_pixels * 100 / (H * W)
            logger.debug(f"mask_{i}:{area_mask}")
            if area_mask < 15:
                logger.debug(f"Deleting mask_{i} with area {area_mask}%")
                del_id.append(i)
        masks = np.delete(all_masks, del_id, axis=0)
        h, w = masks[0].shape
        union_mask = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            union_mask |= m

        masked_rgb, mask_bin = self.preprocess_mask(union_mask, image, idx)

        if save_dir is not None:
            logger.debug(
                f"Saving applied background mask for image {idx} as {save_dir}/figure_{idx}_masked_rgb.png"
            )
            Image.fromarray(masked_rgb).save(f"{save_dir}/figure_{idx}_masked_rgb.png")

        if save_dir is not None:
            logger.debug(
                f"Saving the binary background mask for image {idx} as {save_dir}/figure_{idx}_mask_bin.png"
            )
            self.binary_mask_to_pil(mask_bin).save(f"{save_dir}/figure_{idx}_mask_bin.png")

        end = time.time()
        logger.debug(f"BG mask obtained in {end - start}s")

        return masked_rgb, mask_bin

    def individual_mask(
        self,
        mask_bin: np.ndarray,
        mask_rgb: np.ndarray,
        rgb: str,
        idx: int,
        save_dir: Union[str, None] = None,
    ) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        """Generate and filter individual object masks inside the kept region.

        Parameters
        ----------
        mask_bin : numpy.ndarray
            Inverted binary keep-mask with shape ``(H, W, 1)``.
        mask_rgb : numpy.ndarray
            RGB image produced by :meth:`remove_bg`, with shape ``(H, W, 3)``.
        rgb : str
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

        rgb = Image.open(rgb)
        mask_bin = mask_bin[..., 0]
        
        H, W = mask_rgb.shape[:2]
        masks_sam = self.mask_generator.generate(mask_rgb)

        # Save an RGB figure with all the SAM masks overlaid on the original image
        overlay = mask_rgb.copy()
        for m in masks_sam:
            mask = m["segmentation"]
            color = np.random.randint(0, 255, size=3)
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
        overlay_image = Image.fromarray(overlay)
        if save_dir is not None:
            overlay_image.save(f"{save_dir}/figure_{idx}_overlay_masks.png")
            logger.debug(f"Overlay of SAM masks saved as {save_dir}/figure_{idx}_overlay_masks.png")

        all_masks = []
        all_bboxes = []
        rgb_masks = []
        masks_path = []
        del_id = []
        
        for m in masks_sam:
            all_masks.append(m["segmentation"])
            all_bboxes.append(m["bbox"])

        for i, masked in enumerate(all_masks):
            masked = self.sam_mask_to_pil(masked)
            masked = masked.resize((W, H))

            intersection = np.logical_and(masked, mask_bin)
            union = np.logical_or(masked, mask_bin)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

            logger.debug(f"mask_{i}: IoU with keep-mask = {iou:.5f}")

            num_pixels = np.sum(intersection > 0)
            # Image.fromarray(intersection).save(f"intersection_{idx}_{i}.png")
            area_mask = num_pixels * 100 / (H * W)
            logger.debug(f"mask_{i}:{np.round(area_mask, 5)}%, iou: {np.round(iou, 5)}")

            if False and (area_mask < 0.3 or area_mask > 2 or iou < 0.065):
                del_id.append(i)
                logger.debug(f"mask_{i} deleted with area {area_mask}% and IoU {iou}")
            else:
                save_path = f"{save_dir}/figure_{idx}_mask_overlay_{i}.png"
                masks_path.append(save_path)
                _, rgb_crop = self.cropping_mask(masked, rgb)
                rgb_masks.append(rgb_crop)
                Image.fromarray(rgb_crop).save(save_path)
                if save_dir is not None:
                    logger.debug(
                        f"Saving cropped mask for image {idx} as {save_dir}/figure_{idx}_mask_overlay_{i}.png"
                    )
                    Image.fromarray(rgb_crop).save(f"{save_dir}/figure_{idx}_mask_overlay_{i}.png")

        bboxes = np.delete(all_bboxes, del_id, axis=0)
        end = time.time()
        logger.debug(f"Individual masks obtained in {end - start}s")

        return rgb_masks, bboxes, masks_path
