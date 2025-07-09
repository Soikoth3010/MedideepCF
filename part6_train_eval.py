# part6_train_eval.py
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold
import os
import csv


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, _, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        class_outputs = outputs[1]

        loss = criterion(class_outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(class_outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, _, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            class_outputs = outputs[1]

            loss = criterion(class_outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(class_outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1, all_labels, all_preds


def plot_history_metrics(history_json_path, output_dir, fold_num):
    with open(history_json_path, "r") as f:
        history = json.load(f)

    metrics = ['loss', 'acc', 'prec', 'rec', 'f1']
    for metric in metrics:
        train = history.get(f"train_{metric}", [])
        val = history.get(f"val_{metric}", [])
        plt.figure()
        plt.plot(train, label="Train")
        plt.plot(val, label="Validation")
        plt.title(f"{metric.upper()} over Epochs - Fold {fold_num}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"{metric}_plot_fold{fold_num}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, fold_idx, model_save_path):
    best_f1 = 0.0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} / {num_epochs}")

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_labels, val_preds = validate_one_epoch(
            model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  ➤ Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_prec'].append(train_prec)
        history['val_prec'].append(val_prec)
        history['train_rec'].append(train_rec)
        history['val_rec'].append(val_rec)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved Best Model at Epoch {epoch + 1} with F1: {best_f1:.4f}")

    # Save metrics to JSON
    output_dir = os.path.dirname(model_save_path)
    metrics_path = os.path.join(output_dir, f"metrics_fold{fold_idx + 1}.json")
    with open(metrics_path, 'w') as f:
        json.dump(history, f)

    # Save predictions and labels for confusion matrix
    pred_path = os.path.join(output_dir, f"predictions_fold{fold_idx + 1}.pkl")
    with open(pred_path, 'wb') as f:
        pickle.dump({"labels": val_labels, "preds": val_preds}, f)

    # Plot and save confusion matrix
    labels = [0, 1, 2, 3]
    target_names = ['Corn_Health', 'Blight', 'Common_Rust', 'Gray_Leaf_Spot']
    try:
        cm = confusion_matrix(val_labels, val_preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
        cm_path = os.path.join(output_dir, f"confusion_matrix_fold{fold_idx + 1}.png")
        plt.savefig(cm_path)
        plt.close()
    except Exception as e:
        print(f"Confusion matrix plotting failed: {e}")

    # Plot all metric graphs
    plot_history_metrics(metrics_path, output_dir, fold_num=fold_idx + 1)

    # Classification report
    try:
        report = classification_report(val_labels, val_preds, labels=labels, target_names=target_names, zero_division=0)
        print(f"\nClassification Report (Fold {fold_idx + 1}):\n{report}")
    except Exception as e:
        print(f"Classification report failed: {e}")

    return history


def run_k_fold(model_fn, dataset, device, class_names, output_dir, folds=4, epochs=5, batch_size=8, stratified_labels=None):
    assert stratified_labels is not None, "stratified_labels must be provided for StratifiedKFold"

    max_attempts = 20
    valid_splits = []

    print("Finding valid StratifiedKFold splits with all classes in each validation set...")

    for attempt in range(max_attempts):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42 + attempt)
        temp_splits = []
        all_folds_valid = True

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dataset, stratified_labels)):
            val_labels = [stratified_labels[i] for i in val_idx]
            unique_val_classes = set(val_labels)

            if len(unique_val_classes) < len(class_names):
                all_folds_valid = False
                break

            temp_splits.append((train_idx, val_idx))

        if all_folds_valid:
            valid_splits = temp_splits
            print(f"✔ Found valid fold split on attempt {attempt + 1}")
            break

    if not valid_splits:
        raise ValueError("❌ Could not find valid folds with all classes in validation sets after multiple attempts.")

    fold_histories = []
    fold_summary = []

    for fold_idx, (train_idx, val_idx) in enumerate(valid_splits):
        print(f"\n--- Fold {fold_idx + 1} / {folds} ---")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        train_labels = [stratified_labels[i] for i in train_idx]
        val_labels = [stratified_labels[i] for i in val_idx]
        print("Train distribution:", Counter(train_labels))
        print("Val distribution:", Counter(val_labels))

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        model_save_path = os.path.join(fold_output_dir, f"best_model_fold{fold_idx + 1}.pth")

        history = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=epochs,
            fold_idx=fold_idx,
            model_save_path=model_save_path
        )

        fold_histories.append(history)

        # Save best val metrics of this fold
        best_val_idx = int(np.argmax(history["val_f1"]))
        fold_summary.append({
            "Fold": fold_idx + 1,
            "Val_Accuracy": round(history["val_acc"][best_val_idx], 4),
            "Val_Precision": round(history["val_prec"][best_val_idx], 4),
            "Val_Recall": round(history["val_rec"][best_val_idx], 4),
            "Val_F1": round(history["val_f1"][best_val_idx], 4)
        })

    # Save fold summary as CSV
    csv_path = os.path.join(output_dir, "fold_summary.csv")
    with open(csv_path, "w", newline='') as f:
        fieldnames = ["Fold", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_summary)

        # Compute averages
        avg_metrics = {
            "Fold": "Average",
            "Val_Accuracy": round(np.mean([row["Val_Accuracy"] for row in fold_summary]), 4),
            "Val_Precision": round(np.mean([row["Val_Precision"] for row in fold_summary]), 4),
            "Val_Recall": round(np.mean([row["Val_Recall"] for row in fold_summary]), 4),
            "Val_F1": round(np.mean([row["Val_F1"] for row in fold_summary]), 4),
        }
        writer.writerow(avg_metrics)

    print(f"\n✅ Fold summary with averages saved to: {csv_path}")
    return fold_histories
