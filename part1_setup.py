#part1_setup.py
import os
import random
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_COLORS = {
    0: [0, 0, 0],       # Background - black
    1: [255, 0, 0],     # Blight - red
    2: [0, 255, 0],     # Common Rust - green
    3: [0, 0, 255],     # Gray Leaf Spot - blue
}

def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        color_mask[mask == cls_idx] = color
    return color_mask

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def plot_confusion_matrix(cm, class_names, normalize=False, title="", save_path=""):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_over_epochs(all_metrics, output_dir):
    """
    Plots average Train/Val Accuracy, Precision, Recall, F1 over epochs across all folds.

    all_metrics: list of dicts with keys like 'train_acc', 'val_acc', etc.
    """
    print("Plotting average metric curves over epochs...")

    metrics_keys = ['acc', 'prec', 'rec', 'f1']
    epochs = len(all_metrics[0]['train_acc'])

    avg = {}
    for key in metrics_keys:
        avg[f'train_{key}'] = np.mean([m[f'train_{key}'] for m in all_metrics], axis=0)
        avg[f'val_{key}'] = np.mean([m[f'val_{key}'] for m in all_metrics], axis=0)

    plt.figure(figsize=(12, 8))
    for key in metrics_keys:
        plt.plot(range(1, epochs + 1), avg[f'train_{key}'], label=f"Train {key.upper()}")
        plt.plot(range(1, epochs + 1), avg[f'val_{key}'], label=f"Val {key.upper()}")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Average Train/Val Metrics Across Folds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_metrics_over_epochs.png"))
    plt.close()
