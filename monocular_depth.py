import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
model = model.to(device=device)

intrinsic = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])  # [3, 3] float32

intrinsics = np.array([intrinsic, intrinsic])  # [N, 3, 3] float32

# Run inference on images
images = ["dataset/rgb/rgb_dataset_1.png"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
    export_dir="output",
    export_format="npz",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
print(prediction.depth.shape)  # Depth maps: [N, H, W] float32
# print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
# print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
# print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
