#!/usr/bin/env python3
"""
Evaluate binary segmentation metrics (IoU, F1, Precision, Recall, Accuracy)
using scikit-learn on folders of predicted and ground-truth masks.

Assumes:
  - Foreground pixels = 1 (white)
  - Background pixels = 0 (black)
"""

import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
#import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
)

# ---------------- CONFIG ----------------
PRED_DIR = r"..PATH../test_set/pred_UNet_4m/"
GT_DIR   = r"..PATH../test_set/masks_4m/"


EXTS = [".png", ".jpg", ".tif", ".bmp", ".npy"]
THRESH = 0.5  # if predictions are probability maps (0–1)
# ----------------------------------------

def list_files(folder):
    files = []
    for ext in EXTS:
        files.extend(glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def load_image(path):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path).convert("L"))  # grayscale
    # normalize if needed
    if arr.max() > 1:
        arr = arr / 255.0
    return arr

def binarize(arr):
    """Convert probability or grayscale image to binary mask."""
    return (arr >= THRESH).astype(np.uint8)

def main():
    pred_files = list_files(PRED_DIR)
    gt_files = list_files(GT_DIR)
    pred_map = {os.path.splitext(os.path.basename(p))[0]: p for p in pred_files}
    gt_map   = {os.path.splitext(os.path.basename(g))[0]: g for g in gt_files}
    keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not keys:
        raise RuntimeError("No matching filenames found between predictions and masks.")

    print(f"Found {len(keys)} images for evaluation.")

    all_preds = []
    all_gts = []
    per_image = []

    for k in tqdm(keys, desc="Evaluating"):
        gt = load_image(gt_map[k])
        pred = load_image(pred_map[k])

        # resize if shapes differ
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) / 255.0

        gt_bin = binarize(gt)
        pred_bin = binarize(pred)

        # flatten for sklearn
        y_true = gt_bin.ravel()
        y_pred = pred_bin.ravel()

        iou = jaccard_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        per_image.append({
            "name": k,
            "IoU": iou,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "Accuracy": acc
        })

        all_preds.append(y_pred)
        all_gts.append(y_true)

    # concatenate all pixels
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # global metrics
    global_iou = jaccard_score(all_gts, all_preds, zero_division=0)
    global_f1 = f1_score(all_gts, all_preds, zero_division=0)
    global_prec = precision_score(all_gts, all_preds, zero_division=0)
    global_rec = recall_score(all_gts, all_preds, zero_division=0)
    global_acc = accuracy_score(all_gts, all_preds)

    print("\n=== Global metrics (all pixels across all images) ===")
    print(f"IoU:        {global_iou:.4f}")
    print(f"F1 (Dice):  {global_f1:.4f}")
    print(f"Precision:  {global_prec:.4f}")
    print(f"Recall:     {global_rec:.4f}")
    print(f"Accuracy:   {global_acc:.4f}")

if __name__ == "__main__":
    main()
