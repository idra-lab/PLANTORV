from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import cv2

from skimage.morphology import erosion,dilation,remove_small_objects, disk
from skimage import measure
from skimage.measure import regionprops
import time




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

        label_image = remove_small_objects(label_image, min_size=3500)

        label_image = erosion( label_image, disk(9))
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

        label_image = remove_small_objects(label_image, min_size=3500)

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
            - mask_bin: the binary background mask that is applied over the RGB. Numpy array. Output is a 3-channel uint8 image (H,W,3) where each channel is the same binary mask.
        """
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

    def filter_masks_by_iou(self,masks,index, robot_id, iou_threshold=0.01, iou_2objectthreshold=0.4, iou_maxthreshold=0.6, iou_robot_threshold = 0.95): 
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
                    if j in robot_id:  #erase for more than one robot, erase this if. This filters extra masks for an unique robot
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


