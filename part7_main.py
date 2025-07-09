#part7_main.py
import os
import torch
import cv2
import matplotlib.pyplot as plt
from collections import Counter

from part6_train_eval import run_k_fold
from part2_dataset import MaizeLeafDataset
from part3_transforms import get_train_transforms
from part4_model import MediDeepCFNet
from part1_setup import decode_segmap

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths
    dataset_root = r"D:\Thesis_ML_Maize"
    image_dir = os.path.join(dataset_root, "Model_dataset(copy)")
    mask_dir = os.path.join(dataset_root, "Mask_dataset(copy)")
    output_dir = os.path.join(dataset_root, "output-4")
    os.makedirs(output_dir, exist_ok=True)

    # Class index to name
    class_names = ["Corn_Health", "Corn_Blight", "Corn_Common_Rust", "Corn_Gray_Spot"]

    # Load raw dataset for label extraction
    raw_dataset = MaizeLeafDataset(image_dir=image_dir, mask_dir=mask_dir, transform=None)

    # Extract labels from masks using dominant class (exclude class 0 if others exist)
    class_labels = []
    for i in range(len(raw_dataset)):
        _, mask, _ = raw_dataset[i]
        flat = torch.tensor(mask).view(-1)
        if (flat == 0).all():
            label = 0
        else:
            label = torch.mode(flat[flat != 0])[0].item()
        class_labels.append(label)

    # Print class distribution
    label_dist = Counter(class_labels)
    print("Total Class Distribution:", dict(label_dist))

    # Save class distribution to file
    with open(os.path.join(output_dir, "overall_class_distribution.txt"), "w") as f:
        for cls, count in label_dist.items():
            f.write(f"{class_names[cls]}: {count}\n")

    # Visual check of 3 samples only before training
    for i in range(3):
        image, mask, label = raw_dataset[i]  # Unpack all three

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)  # image is already RGB from dataset
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(decode_segmap(mask))
        plt.title("Decoded Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_visual_check_{i}.png"))
        plt.close()

    # Load dataset with transformations
    dataset = MaizeLeafDataset(image_dir=image_dir, mask_dir=mask_dir, transform=get_train_transforms())

    # Run K-Fold cross-validation training
    run_k_fold(
        model_fn=lambda: MediDeepCFNet(num_classes=len(class_names)),
        dataset=dataset,
        device=device,
        class_names=class_names,
        output_dir=output_dir,
        folds=4,
        epochs=10,
        batch_size=8,
        stratified_labels=class_labels
    )

if __name__ == "__main__":
    main()
