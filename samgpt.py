import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.morphology import erosion,dilation,remove_small_objects, disk
from skimage import measure
from skimage.measure import regionprops
from ultralytics import settings
import pathlib
import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import time
import os
import argparse
import base64
import mimetypes
from pathlib import Path
import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass
import re

def convert(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

"""SEGMENTATION"""
class SAMModel:
    def __init__(self, sam_checkpoint, model_type="vit_h", device="cuda", points_per_side=32):#points_per_side=32
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        self.sam = sam_model_registry[model_type](sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=points_per_side)

    def sam_mask_to_pil(self,mask_bool) -> Image.Image:
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        return Image.fromarray(mask_uint8)

    def preprocess_mask(self,mask,rgb,f) -> np.ndarray: 
        """
            This function is to preprocess the RGB image before applying SAM for the second time.
            This is done to obtain a better segmentation of the objects that we are looking for.
            Inputs:
            - mask: the mask that we want to apply over the RGB
            - rgb: RGB image
            - f: index of the image, used for saving the masked RGB for visualization.
            Outputs:
            - masked_rgb: the RGB image with the mask applied. Numpy array. Output is a 3-channel uint8 image (H,W,3) 
            - mask_bin: the binary mask that is applied over the RGB. Numpy array. Output is a 3-channel uint8 image (H,W,3) where each channel is the same binary mask.
        """
        mask = mask.astype(np.uint8) * 255
        mask_bin = (mask > 0).astype(np.uint8)
        mask_blur = cv2.GaussianBlur(mask_bin * 255, (7, 7), 4)
        mask_blur = (mask_blur > 0).astype(np.uint8)

        label_image = measure.label(mask_blur)

        label_image = remove_small_objects(label_image, max_size=3500)

        label_image = erosion(label_image, disk(9))
        label_image = dilation(label_image, disk(3))

        label_image = measure.label(label_image)
        mask_clean = (label_image > 0).astype(np.uint8) * 255
        mask_bin = (mask_clean > 0).astype(np.uint8)[..., None]
        mask_bin = 1-mask_bin
        masked_rgb = rgb * mask_bin 
        ref_img = Image.fromarray(masked_rgb.astype("uint8"))
        return masked_rgb, mask_bin

    def cropping_mask(self,masks,rgb, alpha = 1.4, beta = 25):
        """
            This funciton is defined to crop and improve the masks out of the first filter.
            Inputs:
            - masks: filtered masks. #Three channels (1920,1080,3)
            - rgb: rgb image. 
            - alpha: contrast factor for improving the visualization of rgb
            - beta: brightness factor
            Outputs:
            - mask_crop: cropped mask
            - rgb_crop: cropped rgb
        """
        
        masks = np.array(masks)
        mask = masks.astype(np.uint8) * 255
        mask_bin = (mask > 0).astype(np.uint8)
        mask_blur = cv2.GaussianBlur(mask_bin * 255, (7, 7), 4)
        mask_blur = (mask_blur > 0).astype(np.uint8)

        label_image = measure.label(mask_blur)

        label_image = remove_small_objects(label_image, max_size=3500)

        label_image = erosion(label_image, disk(9))
        label_image = dilation(label_image, disk(3))

        label_image = measure.label(label_image)
        mask_clean = (label_image > 0).astype(np.uint8) * 255
        masks = (mask_clean > 0).astype(np.uint8)

        rgb=np.array(rgb)
        ys,xs = np.where(masks > 0)
        top_y = ys.min()+3
        bot_y = ys.max()+3
        left_x = xs.min()+3
        right_x = xs.max()+3

        mask_crop = masks[top_y:bot_y, left_x:right_x]
        mask_crop = mask_crop.astype(np.uint8) * 255
        mask_crop = cv2.resize(mask_crop,None, fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4)
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255 #for being binary
        mask_rgb = rgb[top_y:bot_y, left_x:right_x, :]
        rgb_crop = cv2.convertScaleAbs(mask_rgb, alpha=alpha, beta=beta)
        KERNEL = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
        rgb_crop = cv2.resize(rgb_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        rgb_crop = cv2.filter2D(rgb_crop,-1,KERNEL)

        return mask_crop, rgb_crop

    def obtain_bg(self,image,idx):
        """
            This function is defined to obtain the background mask of the image.
            It applies SAM over the original RGB image and then filters the masks obtained by area.
            Inputs:
            - image: the original RGB image.
            - idx: index of the image, used for saving the masked RGB for visualization.
            Outputs:
            - masked_rgb: the RGB image with the background mask applied. Numpy array. Output is a 3-channel uint8 image (H,W,3)
            - mask_bin: the binary background mask that is applied over the RGB. Numpy array. Output is a 3-channel uint8 image (H,W,3) where each channel is the same binary mask."""
        start=time.time()
        image_read = Image.open(image)
        image_np = np.array(image_read)
        H,W,D = image_np.shape
        masks_sam = self.mask_generator.generate(image_np)
        all_masks = []
        all_bboxes = []
        del_id = []
        for m in masks_sam:
            all_masks.append(m["segmentation"])
            all_bboxes.append(m["bbox"])

        for i,mask in enumerate(all_masks):
            masked = self.sam_mask_to_pil(mask)
            masked = masked.resize((W,H))
            masked_np = np.array(masked)
            num_pixels = np.sum(masked_np > 0)
            area_mask = num_pixels*100/(H*W)
            if area_mask<15:
                del_id.append(i)
        masks = np.delete(all_masks, del_id, axis=0)
        h, w = masks[0].shape
        union_mask = np.zeros((h, w), dtype=np.uint8)

        for m in masks:
            union_mask |= m

        masked_rgb,mask_bin = self.preprocess_mask(union_mask,image_read,idx)
        end = time.time()
        print(f"BG mask obtained in {end-start}s")
        return masked_rgb,mask_bin

    def filter_masks_by_iou(self,masks,index, robot_id, iou_threshold=0.01, iou_2objectthreshold=0.4, iou_maxthreshold=0.6, iou_robot_threshold = 0.95):#iou_maxthreshold=0.65 #iou_2objectthreshold=0.35 
        """
        Erases the redundant masks: if a mask is almost contained in another, the smaller one is removed.
        """
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
                    if j in robot_id:
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


    def individual_mask(self,mask_bin,mask_rgb,rgb,idx):
        """
            This function is defined to obtain the individual masks of the objects that we are looking for. 
            It applies SAM over the masked RGB image and then filters the masks obtained by area and IoU with the original mask.
            Inputs:
            - mask_bin: the binary mask that is applied over the RGB. Numpy array. Output is a 3-channel uint8 image (H,W,3) where each channel is the same binary mask.
            - mask_rgb: the RGB image with the mask applied. Numpy array. Output is a 3-channel uint8 image (H,W,3)
            - rgb: the original RGB image.
            - idx: index of the image, used for saving the masked RGB for visualization.
            Outputs:
            - rgb_crop: the cropped RGB image of the object. Numpy array. Output is a 3-channel uint8 image (H',W',3) where H' and W' are the height and width of the cropped image.
            - bboxes: the bounding boxes of the objects. Numpy array. Output is a Nx4 array where N is the number of objects and each row is [x_min, y_min, width, height].
            - masks_path: the paths of the masks obtained. List of strings. Output is a list of length N where each element is the path of the mask obtained for each object.
        """
        start = time.time()
        H,W = mask_rgb.shape[:2]
        masks_sam =self.mask_generator.generate(mask_rgb)

        all_masks = []
        all_bboxes = []

        keep = []
        robot_id = []
        rgb = Image.open(rgb)
        mask_bin = mask_bin[...,0]


        for m in masks_sam:
            all_masks.append(m["segmentation"])
            all_bboxes.append(m["bbox"])

        for i,masked in enumerate(all_masks):
            masked = self.sam_mask_to_pil(masked)
            masked = masked.resize((W,H))
            
            intersection = np.logical_and(masked, mask_bin)
            union = np.logical_or(masked, mask_bin)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            num_pixels = np.sum(intersection > 0)
            area_mask = num_pixels*100/(H*W)         
            if (0.35<area_mask<6.5 or area_mask>10) and iou>0.02: #area min estaba 0.35
                if area_mask>10: 
                    robot_id.append(i)
                keep.append(i)  
                 

        masks = [(i,all_masks[i]) for i in keep]
        bboxes = [all_bboxes[i] for i in keep]
        masks_only = [m[1] for m in masks]
        index = [m[0] for m in masks]
        valid = self.filter_masks_by_iou(masks_only,index, robot_id, iou_threshold=0.01)

        masks_filtered = [masks[i] for i in valid]
        bboxes_filtered = [bboxes[i] for i in valid]

        rgb_masks = []
        masks_path = []

        for i,(orig_idx,masked) in enumerate(masks_filtered):
            save_path = f"ppt_outputs/image{idx+1}/crop_{orig_idx}.png"
            masks_path.append(save_path)
            mask_crop, rgb_crop = self.cropping_mask(masked,rgb)
            rgb_masks.append(rgb_crop)
            Image.fromarray(rgb_crop).save(save_path)


        end = time.time()
        print(f"Individual masks obtained in {end-start}s")

        return rgb_masks, bboxes_filtered,masks_path


"""Depth Estimation"""
@dataclass(frozen=True)
class Intrinsics:
    cx: float
    cy: float
    fx: float
    fy: float
    width: int
    height: int

@dataclass(frozen=True)
class Distortion:
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float

@dataclass(frozen=True)
class CalibrationSet:
    depth_distortion: Distortion
    depth_intrinsic: Intrinsics
    rgb_distortion: Distortion
    rgb_intrinsic: Intrinsics
    rot: np.ndarray  # 3x3
    trans: np.ndarray  # 3,


@dataclass(frozen=True)
class AlignProfile:
    align_type: int
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    param_index: int
    align_left: int
    align_top: int
    align_right: int
    align_bottom: int
    depth_scale: float

_ROT = np.asarray(
    [
        [0.994558, -0.00445435, 0.00197944],
        [0.00422393, 0.99455, 0.104173],
        [-0.00243268, -0.104163, 0.994557],
    ],
    dtype=np.float64,
)
_TRANS = np.asarray([-32.6072, -0.835282, 1.99768], dtype=np.float64)

_DEPTH_DIST = Distortion(
    k1=20.449,
    k2=9.65474,
    k3=0.311488,
    k4=20.7575,
    k5=16.556,
    k6=2.11015,
    p1=5.12616e-05,
    p2=-8.77438e-06,
)

_RGB_DIST = Distortion(
    k1=0.0767264,
    k2=-0.104236,
    k3=0.0419684,
    k4=0.0,
    k5=0.0,
    k6=0.0,
    p1=0.000112286,
    p2=-6.58385e-05,
)

_HARDCODED_CALIBRATIONS: List[CalibrationSet] = [
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=516.94, cy=519.187, fx=504.676, fy=504.768, width=1024, height=1024),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.734, cy=176.424, fx=373.497, fy=373.414, width=640, height=360),
        rot=_ROT,
        trans=_TRANS,
    ),
    CalibrationSet(
        depth_distortion=_DEPTH_DIST,
        depth_intrinsic=Intrinsics(cx=324.94, cy=339.187, fx=504.676, fy=504.768, width=640, height=576),
        rgb_distortion=_RGB_DIST,
        rgb_intrinsic=Intrinsics(cx=320.978, cy=235.232, fx=497.996, fy=497.886, width=640, height=480),
        rot=_ROT,
        trans=_TRANS,
    ),
]

_HARDCODED_PROFILES: List[AlignProfile] = [
    AlignProfile(1, 3840, 2160, 1024, 1024, 0, 0, 0, 0, -1680, 3.75),
    AlignProfile(1, 2560, 1440, 1024, 1024, 0, 0, 0, 0, -1120, 2.5),
    AlignProfile(1, 1920, 1080, 1024, 1024, 0, 0, 0, 0, -840, 1.875),
    AlignProfile(1, 1280, 720, 1024, 1024, 0, 0, 0, 0, -560, 1.25),
    AlignProfile(1, 3840, 2160, 512, 512, 0, 0, 0, 0, -1680, 7.5),
    AlignProfile(1, 2560, 1440, 512, 512, 0, 0, 0, 0, -1120, 5.0),
    AlignProfile(1, 1920, 1080, 512, 512, 0, 0, 0, 0, -840, 3.75),
    AlignProfile(1, 1280, 720, 512, 512, 0, 0, 0, 0, -560, 2.5),
    AlignProfile(1, 3840, 2160, 640, 576, 2, 0, 0, 0, -1296, 6.0),
    AlignProfile(1, 2560, 1440, 640, 576, 2, 0, 0, 0, -864, 4.0),
    AlignProfile(1, 1920, 1080, 640, 576, 2, 0, 0, 0, -648, 3.0),
    AlignProfile(1, 1280, 720, 640, 576, 2, 0, 0, 0, -432, 2.0),
    AlignProfile(1, 3840, 2160, 320, 288, 2, 0, 0, 0, -1296, 12.0),
    AlignProfile(1, 2560, 1440, 320, 288, 2, 0, 0, 0, -864, 8.0),
    AlignProfile(1, 1920, 1080, 320, 288, 2, 0, 0, 0, -648, 6.0),
    AlignProfile(1, 1280, 720, 320, 288, 2, 0, 0, 0, -432, 4.0),
    AlignProfile(1, 1280, 960, 1024, 1024, 1, 0, 0, 0, -320, 1.25),
    AlignProfile(1, 1280, 960, 512, 512, 1, 0, 0, 0, -320, 2.5),
    AlignProfile(1, 1280, 960, 640, 576, 3, 0, 0, 0, -192, 2.0),
    AlignProfile(1, 1280, 960, 320, 288, 3, 0, 0, 0, -192, 4.0),
    AlignProfile(2, 3840, 2160, 1024, 1024, 0, 0, 0, 0, 0, 6.0),
]

def _distort_normalized(x: np.ndarray, y: np.ndarray, d: Distortion) -> Tuple[np.ndarray, np.ndarray]:
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    num = 1.0 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6
    den = 1.0 + d.k4 * r2 + d.k5 * r4 + d.k6 * r6
    radial = num / den
    x_tan = 2.0 * d.p1 * x * y + d.p2 * (r2 + 2.0 * x * x)
    y_tan = d.p1 * (r2 + 2.0 * y * y) + 2.0 * d.p2 * x * y
    return x * radial + x_tan, y * radial + y_tan


def _undistort_pixels_to_normalized(
    u: np.ndarray,
    v: np.ndarray,
    intr: Intrinsics,
    dist: Distortion,
    iters: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    xd = (u - intr.cx) / intr.fx
    yd = (v - intr.cy) / intr.fy

    # Fixed-point refinement for inverse distortion.
    xu = xd.copy()
    yu = yd.copy()
    for _ in range(iters):
        x_est, y_est = _distort_normalized(xu, yu, dist)
        xu += xd - x_est
        yu += yd - y_est
    return xu, yu


def _project_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    intr: Intrinsics,
    dist: Distortion,
) -> Tuple[np.ndarray, np.ndarray]:
    xd, yd = _distort_normalized(x, y, dist)
    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy
    return u, v


class DepthRgbMapper():
    """Depth<->RGB utility built from hardcoded Femto Mega calibration data."""

    def __init__(self, calibration: CalibrationSet, profile: Optional[AlignProfile] = None):
        self.calibration = calibration
        self.profile = profile

    @classmethod
    def from_hardcoded(
        cls,
        color_size: Tuple[int, int],
        depth_size: Tuple[int, int],
        align_type_preference: Sequence[int] = (1, 2),
    ) -> "DepthRgbMapper":
        calibrations = _HARDCODED_CALIBRATIONS
        profiles = _HARDCODED_PROFILES
        cw, ch = color_size
        dw, dh = depth_size

        matched_profile: Optional[AlignProfile] = None
        for pref in align_type_preference:
            for p in profiles:
                if (
                    p.align_type == pref
                    and p.color_width == cw
                    and p.color_height == ch
                    and p.depth_width == dw
                    and p.depth_height == dh
                ):
                    matched_profile = p
                    break
            if matched_profile is not None:
                break

        if matched_profile is not None:
            idx = matched_profile.param_index
            if idx < 0 or idx >= len(calibrations):
                raise ValueError(f"Profile paramIndex={idx} out of range for calibration list")
            return cls(calibrations[idx], matched_profile)

        for c in calibrations:
            if (
                c.rgb_intrinsic.width == cw
                and c.rgb_intrinsic.height == ch
                and c.depth_intrinsic.width == dw
                and c.depth_intrinsic.height == dh
            ):
                return cls(c, None)

        raise ValueError(
            "No matching hardcoded calibration/profile found for requested color/depth resolution pair"
        )


    def align_depth_to_color_with_correspondence(
        self,
        depth_image: np.ndarray,
        depth_unit_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project raw depth image into the RGB camera image plane.

        Args:
            depth_image: HxW depth array from depth sensor.
            depth_unit_scale: Converts depth_image units to millimeters (mm).
                Example: 1.0 if already in mm, 0.1 if each unit is 0.1 mm.

        Returns:
            aligned_depth_mm: Hc x Wc float32 depth image in millimeters, aligned to RGB.
            src_u_map: Hc x Wc int32 map of source depth-u for each RGB pixel (-1 if invalid).
            src_v_map: Hc x Wc int32 map of source depth-v for each RGB pixel (-1 if invalid).
        """
        c = self.calibration
        if depth_image.ndim != 2:
            raise ValueError("depth_image must be a 2D array")

        h, w = depth_image.shape
        if w != c.depth_intrinsic.width or h != c.depth_intrinsic.height:
            raise ValueError(
                f"depth_image shape {w}x{h} does not match calibration depth size "
                f"{c.depth_intrinsic.width}x{c.depth_intrinsic.height}"
            )

        v_grid, u_grid = np.indices((h, w), dtype=np.float64)
        z_mm = depth_image.astype(np.float64) * float(depth_unit_scale)
        valid = z_mm > 0.0
        if not np.any(valid):
            out_shape = (c.rgb_intrinsic.height, c.rgb_intrinsic.width)
            return (
                np.zeros(out_shape, dtype=np.float32),
                np.full(out_shape, -1, dtype=np.int32),
                np.full(out_shape, -1, dtype=np.int32),
            )

        u = u_grid[valid]
        v = v_grid[valid]
        z = z_mm[valid]

        x_d, y_d = _undistort_pixels_to_normalized(u, v, c.depth_intrinsic, c.depth_distortion)

        xyz_d = np.vstack((x_d * z, y_d * z, z))
        xyz_c = (c.rot @ xyz_d) + c.trans.reshape(3, 1)

        zc = xyz_c[2]
        positive = zc > 1e-6
        if not np.any(positive):
            out_shape = (c.rgb_intrinsic.height, c.rgb_intrinsic.width)
            return (
                np.zeros(out_shape, dtype=np.float32),
                np.full(out_shape, -1, dtype=np.int32),
                np.full(out_shape, -1, dtype=np.int32),
            )

        x_c = xyz_c[0, positive] / zc[positive]
        y_c = xyz_c[1, positive] / zc[positive]
        z_c_mm = zc[positive]
        u_src = u[positive].astype(np.int64)
        v_src = v[positive].astype(np.int64)

        u_c, v_c = _project_to_pixels(x_c, y_c, c.rgb_intrinsic, c.rgb_distortion)
        u_i = np.rint(u_c).astype(np.int64)
        v_i = np.rint(v_c).astype(np.int64)

        in_bounds = (
            (u_i >= 0)
            & (u_i < c.rgb_intrinsic.width)
            & (v_i >= 0)
            & (v_i < c.rgb_intrinsic.height)
        )
        if not np.any(in_bounds):
            out_shape = (c.rgb_intrinsic.height, c.rgb_intrinsic.width)
            return (
                np.zeros(out_shape, dtype=np.float32),
                np.full(out_shape, -1, dtype=np.int32),
                np.full(out_shape, -1, dtype=np.int32),
            )

        u_i = u_i[in_bounds]
        v_i = v_i[in_bounds]
        z_c_mm = z_c_mm[in_bounds]
        u_src = u_src[in_bounds]
        v_src = v_src[in_bounds]

        out_h = c.rgb_intrinsic.height
        out_w = c.rgb_intrinsic.width
        zbuf = np.full(out_h * out_w, np.inf, dtype=np.float64)
        src_u_flat = np.full(out_h * out_w, -1, dtype=np.int32)
        src_v_flat = np.full(out_h * out_w, -1, dtype=np.int32)
        lin = v_i * out_w + u_i

        # Keep the nearest depth sample per RGB pixel and remember source depth pixel.
        for idx in range(lin.size):
            li = int(lin[idx])
            z_val = float(z_c_mm[idx])
            if z_val < zbuf[li]:
                zbuf[li] = z_val
                src_u_flat[li] = int(u_src[idx])
                src_v_flat[li] = int(v_src[idx])

        aligned = zbuf.reshape(out_h, out_w)
        aligned[np.isinf(aligned)] = 0.0
        src_u_map = src_u_flat.reshape(out_h, out_w)
        src_v_map = src_v_flat.reshape(out_h, out_w)
        return aligned.astype(np.float32), src_u_map, src_v_map

    def align_depth_to_color(
        self,
        depth_image: np.ndarray,
        depth_unit_scale: float = 1.0,
    ) -> np.ndarray:
        aligned, _, _ = self.align_depth_to_color_with_correspondence(
            depth_image,
            depth_unit_scale=depth_unit_scale,
        )
        return aligned

    def get_depth_at_rgb(
        self,
        depth_image: np.ndarray,
        rgb_u: int,
        rgb_v: int,
        depth_unit_scale: float = 1.0,
        neighborhood: int = 1,
    ) -> Optional[float]:
        """Return depth in mm at RGB pixel after D2C reprojection.

        If exact pixel has no value, searches a small square neighborhood.
        """
        aligned = self.align_depth_to_color(depth_image, depth_unit_scale=depth_unit_scale)

        h, w = aligned.shape
        if rgb_u < 0 or rgb_u >= w or rgb_v < 0 or rgb_v >= h:
            return None

        d = float(aligned[rgb_v, rgb_u])
        if d > 0.0:
            return d

        if neighborhood <= 0:
            return None

        u0 = max(0, rgb_u - neighborhood)
        u1 = min(w - 1, rgb_u + neighborhood)
        v0 = max(0, rgb_v - neighborhood)
        v1 = min(h - 1, rgb_v + neighborhood)

        patch = aligned[v0 : v1 + 1, u0 : u1 + 1]
        nonzero = patch[patch > 0.0]
        if nonzero.size == 0:
            return None
        return float(np.min(nonzero))

def _find_depth_and_source(
    aligned_depth_mm: np.ndarray,
    src_u_map: np.ndarray,
    src_v_map: np.ndarray,
    rgb_u: int,
    rgb_v: int,
    neighborhood: int,
) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    h, w = aligned_depth_mm.shape
    if rgb_u < 0 or rgb_u >= w or rgb_v < 0 or rgb_v >= h:
        return None, None

    d = float(aligned_depth_mm[rgb_v, rgb_u])
    su = int(src_u_map[rgb_v, rgb_u])
    sv = int(src_v_map[rgb_v, rgb_u])
    if d > 0.0 and su >= 0 and sv >= 0:
        return d, (su, sv)

    if neighborhood <= 0:
        return None, None

    u0 = max(0, rgb_u - neighborhood)
    u1 = min(w - 1, rgb_u + neighborhood)
    v0 = max(0, rgb_v - neighborhood)
    v1 = min(h - 1, rgb_v + neighborhood)

    best_d = None
    best_uv = None
    for vv in range(v0, v1 + 1):
        for uu in range(u0, u1 + 1):
            d_val = float(aligned_depth_mm[vv, uu])
            if d_val <= 0.0:
                continue
            su = int(src_u_map[vv, uu])
            sv = int(src_v_map[vv, uu])
            if su < 0 or sv < 0:
                continue
            if best_d is None or d_val < best_d:
                best_d = d_val
                best_uv = (su, sv)

    return best_d, best_uv

def _depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    if depth_image.ndim != 2:
        raise ValueError("Depth image for visualization must be single-channel")

    depth_f = depth_image.astype(np.float32)
    valid = depth_f > 0
    vis = np.zeros_like(depth_f, dtype=np.uint8)
    if np.any(valid):
        vals = depth_f[valid]
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((depth_f - lo) * (255.0 / (hi - lo)), 0, 255)
        vis = scaled.astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def main_coords(rgb_path,depth_path, dict_objects):

    rgb   = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # si es PNG de depth visual

    color_size = (rgb.shape[1],rgb.shape[0])
    depth_size = (depth.shape[1],depth.shape[0])

    mapper = DepthRgbMapper.from_hardcoded(color_size = color_size , depth_size=depth_size)
    aligned_depth_mm, src_u_map, src_v_map = mapper.align_depth_to_color_with_correspondence(
        depth,
        depth_unit_scale=1 #Scale from depth pixel units to milimeters
    )

    rgb_h,rgb_w = rgb.shape[:2]
    if aligned_depth_mm.shape[1] != rgb_w or aligned_depth_mm.shape[0] != rgb_h:
        aligned_depth_mm = cv2.resize(aligned_depth_mm, (rgb_w, rgb_h),interpolation = cv2.INTER_NEAREST)
        src_u_map = cv2.resize(src_u_map,(rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        src_v_map = cv2.resize(src_v_map,(rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

   
    for mask_id in dict_objects.keys():
        coords = dict_objects[mask_id]["bbox"]
        ix,iy,delta_x,delta_y= coords
        fin_x = ix+delta_x
        fin_y = iy+delta_y
        cx = (ix+fin_x)//2
        cy = (iy+fin_y)//2 

        depth_mm, src_uv = _find_depth_and_source(
            aligned_depth_mm,
            src_u_map,
            src_v_map,
            cx,
            cy,
            max(0,1),
        )
        # print(f"Object {mask_id}: depth={depth_mm} mm, src_uv={src_uv}")
        dict_objects[mask_id]["coord_center&depth"]=[cx,cy,depth_mm]

    return dict_objects


"""GPT Model for tagging and description"""

class GPTModel:
    def __init__(self, endpoint, model_name, deployment, subscription_key, api_version):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        self.deployment = deployment

    def encode_image_data_url(self,image_path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def main_gpt(self,image,mask_path, bboxes):
        """
        This function is defined to obtain the tagging and description of the objects that we are looking for.
        It applies GPT over the original RGB image and the cropped images of the objects obtained with SAM, and then it returns a dictionary with the tagging and description of each object.
        Inputs:
            - image: the original RGB image. String. Path of the original RGB image.
            - mask_path: the paths of the masks obtained. List of strings. Output is a list of length N where each element is the path of the mask obtained for each object.
            - bboxes: the bounding boxes of the objects obtained by SAM.
        Outputs:
            - dict_outputs: the dictionary with the tagging and description of each object. Dictionary. Output is a dictionary where each key is the name of the object (for example, "mask_0") and each value is another dictionary with the following keys:
                - "tag": the tag of the object obtained by GPT. String.
                - "description": the description of the object obtained by GPT. String.
                - "mask": the path of the mask obtained for the object. String.
                - "bbox": the bounding box of the object obtained by SAM. List of 4 integers [x_min, y_min, width, height].
        """
        image_path = Path(image)
        image_data_url = self.encode_image_data_url(image_path)
        dict_outputs = {}
        # question_2 = """You will receive:
        # 1) Two images of the same scene. The first image shows the whole scene, and the second image is a cropped region of the image. 
        # The second image shows the object and the first one gives the context of the image.
        # Your task:
        # - Describe the main object from the SECOND image, using the first one to consider the context of the workspace. Tell me the relative positions with respect the other objects that are seen in the first image, for example, specifying if they are on the left, on the rigth or next to another object.
        # - The tagging should be ultra-specific. For example, instead of saying "lego block", say "furthest blue lego block with 4 studs ". Add the colour in the tag.
        # Return ONLY raw JSON.
        # Do not use markdown code fences.
        # Do not write ```json.
        # {
        # "tag": "string",
        # "description": "string"
        # }
        # """
        question_2 = """You will receive:
        1) Two images of the same scene. The first image shows the whole scene, and the second image is a cropped region of the image. 
        The second image shows the object and the first one gives the context of the image.
        Your task:
        - Describe the main object from the SECOND image, using the first one to consider the context of the workspace. Tell me the relative positions with respect the other objects that are seen in the first image, for example, specifying if they are on the left, on the rigth or next to another object.
        - The tags should be ONLY one of the following ones: "Wide and large blue Lego block", "Small blue Lego block", "Yellow Lego block", "Wide red Lego Block with 4 studs", "Green Lego block", "2x2 Blue and red Lego block", "Tall red Lego block", " White and red box","Blue and white small box", "Big Black Bottle","Big White bottle", "Metallic Wrench", "Orange Lego block", "Orange small box", " White and green box", "Full robotic arm", "Partial robotic arm", "Unknown object". 
        - Do not change the tags neither use other tags that are not in the list. If you are not sure about the tag, use "Unknown object". For the detection, you can use the context of the whole image.
        Return ONLY raw JSON.
        Do not use markdown code fences.
        Do not write ```json.
        {
        "tag": "string",
        "description": "string"
        }
        """
        for p in range(len(mask_path)):
            crop_url = self.encode_image_data_url(Path(mask_path[p]))
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",

                        "content": [
                            {"type": "text", "text": question_2},
                            {"type": "text", "text": "Full image:"},
                            {"type": "image_url", "image_url": {"url": image_data_url}, "detail": "auto"},
                            {"type": "text", "text": "Cropped image:"},
                            {"type": "image_url", "image_url": {"url": crop_url}, "detail": "auto"},
                        ],

                    }
                ],
                max_completion_tokens =16384,
                model=self.deployment
            )
            raw = response.choices[0].message.content	
            try:
                dict_outputs[f"mask_{p}"] = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for mask_{p}: {raw}")
                dict_outputs[f"mask_{p}"] = {"tag": "unknown", "description": "unknown", "full_object": False}
            dict_outputs[f"mask_{p}"]["mask"]=mask_path[p]
            dict_outputs[f"mask_{p}"]["bbox"]=bboxes[p]

        return dict_outputs


"""Main Function"""
def main(images,depth_path):

    load_dotenv()

    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_API_KEY")

    masks_dic = {}
    bboxes_dic = {}
    full_dict = {}

    sam = SAMModel("models/sam/sam_vit_h_4b8939.pth")

    endpoint = azure_endpoint
    model_name = "gpt-5.2-chat"
    deployment = "gpt-5.2-chat"

    subscription_key = azure_key
    api_version = "2024-12-01-preview"

    gpt = GPTModel(endpoint, model_name, deployment, subscription_key, api_version)

    for f,image in enumerate(images):
        print(f"Processing image {f+1}/{len(images)}: {image}")
        rute = f"ppt_outputs/image{f+1}"
        os.makedirs(rute, exist_ok=True)

        masked_rgb,mask_bin = sam.obtain_bg(image,f)
        rgb_masks, bboxes, masks_path= sam.individual_mask(mask_bin,masked_rgb,image,f)

        # start_gpt = time.time()
        # full_dict[f"Image_{f}"]=gpt.main_gpt(image,masks_path,bboxes)
        # end_gpt = time.time()
        # print(f"GPT tagging and description for image {f+1} obtained in {end_gpt-start_gpt}s")


        # start_coords = time.time()
        # full_dict[f"Image_{f}"]=main_coords(image,depth_path[f],full_dict[f"Image_{f}"])
        # end_coords = time.time()
        # print(f"Coordinates and depth for image {f+1} obtained in {end_coords-start_coords}s")


        # # print(f"Image {f+1}: {full_dict[f"Image_{f}"]}")
    
        # with open(f"outputs_json_labeled/output_img{f+1}.json","w") as k:
        #     json.dump(full_dict[f"Image_{f}"], k, indent=4, default=convert)

if __name__=="__main__":
    
    start_all = time.time()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    rute = f"outputs_json_labeled"
    os.makedirs(rute, exist_ok=True)

    images = ["dataset/rgb/rgb_dataset_1.png"]
    depth = ["dataset/depth/depth_dataset_1.png"]
    
    # path_img = Path.cwd() / "dataset/rgb"
    # path_depth = Path.cwd() / "dataset/depth"

    # images = sorted(path_img.glob("*.png"), key = lambda x: int(x.stem.split("_")[-1]))
    # depth = sorted(path_depth.glob("*.png"), key = lambda x: int(x.stem.split("_")[-1]))
   
    main(images,depth)
    end_all=time.time()

    print(f"Total time image process: {end_all-start_all}s")





