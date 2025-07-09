import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class MaizeLeafDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to folder containing input images.
            mask_dir (str): Path to folder containing segmentation masks.
            transform (albumentations.Compose): Albumentations transformations.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect all image paths in the image directory
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        mask_rgb = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        # Create class-index mask: 0=background, 1=Blight, 2=Rust, 3=Gray Leaf Spot
        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
        mask[(mask_rgb == [255, 0, 0]).all(axis=2)] = 1  # Blight
        mask[(mask_rgb == [0, 255, 0]).all(axis=2)] = 2  # Rust
        mask[(mask_rgb == [0, 0, 255]).all(axis=2)] = 3  # Gray Leaf Spot

        # Compute dominant class label from mask (excluding background)
        flat = mask.flatten()
        if np.all(flat == 0):
            label = 0  # Healthy / background only
        else:
            label = np.bincount(flat[flat != 0]).argmax()

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask, label
