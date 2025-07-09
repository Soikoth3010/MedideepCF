#part3_transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Standard ImageNet mean
                    std=(0.229, 0.224, 0.225)),  # Standard ImageNet std
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
