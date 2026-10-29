from pathlib import Path
import random
import numpy as np

import cv2
import matplotlib.pyplot as plt

from src.annotations import read_yolo_label, draw_boxes
from src.train import read_results

PLOT_FILES = {
    "Confusion matrix (counts)": ("confusion_matrix.png",),
    "Confusion matrix (normalized)": ("confusion_matrix_normalized.png",),
    "Precision-recall curve": ("BoxPR_curve.png", "PR_curve.png"),      # name depends on the Ultralytics version
    "F1 curve": ("BoxF1_curve.png", "F1_curve.png"),
}

def _pick_samples(pairs, n, seed):
    """Pick n (class_folder, image, label) triplets, spread evenly over the classes."""
    rng = random.Random(seed)
    pools = {}
    for class_name in sorted(pairs):
        items = sorted(pairs[class_name], key=lambda p: p[0].name) # fixed order first
        rng.shuffle(items)
        pools[class_name] = items

    picked = []
    while len(picked) < n and any(pools.values()):
        for class_name, items in pools.items():
            if items and len(picked) < n:
                img, label = items.pop()
                picked.append((class_name, img, label))
    return picked


def show_samples_grid(pairs, n=9, seed=42, class_names=(), cols=3, figsize_per_cell=4):
    """Show n images with their ground-truth boxes and class names.
    """
    samples = _pick_samples(pairs, n, seed)
    rows = -(-len(samples) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_cell, rows * figsize_per_cell))
    axes = np.asarray(axes).reshape(-1)

    for ax, (folder, img_path, lbl_path) in zip(axes, samples):
        image  = cv2.imread(str(img_path))
        if image is None:                                            # unreadable file
            ax.set_title(f"{folder}: unreadable", fontsize=9)
            ax.axis("off")
            continue
        boxes = read_yolo_label(lbl_path)
        ax.imshow(cv2.cvtColor(draw_boxes(image, boxes, class_names), cv2.COLOR_BGR2RGB))
        names = sorted({class_names[b[0]] if b[0] < len(class_names) else str(b[0]) for b in boxes})
        ax.set_title(f"{folder}\nboxes: {len(boxes)} ({', '.join(names) or 'none'})", fontsize=9)
        ax.axis("off")

    for ax in axes[len(samples):]:                                   # hide unused cells
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_class_distribution(df):
    """
    takes the table from class_distribution and draws two bar charts side by side
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ("images", "boxes")):
        bars = ax.bar(df["class"], df[col])
        ax.bar_label(bars)
        ax.set_title(f"{col} per class")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()

def plot_training_curves(run_dir):
    """Plot train vs validation losses, precision/recall, mAP and the generalization gap."""
    df = read_results(run_dir)
    epochs = df["epoch"]
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    best_epoch = int(df.loc[fitness.idxmax(), "epoch"])  # epoch saved as best.pt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes[0], ("box_loss", "cls_loss", "dfl_loss")):
        ax.plot(epochs, df[f"train/{name}"], label="train")
        ax.plot(epochs, df[f"val/{name}"], label="validation")
        ax.set_title(name)

    ax = axes[1][0]
    ax.plot(epochs, df["metrics/precision(B)"], label="precision")
    ax.plot(epochs, df["metrics/recall(B)"], label="recall")
    ax.set_title("precision and recall (validation)")

    ax = axes[1][1]
    ax.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5")
    ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    ax.set_title("mAP (validation)")

    ax = axes[1][2]
    ax.plot(epochs, df["val/box_loss"] - df["train/box_loss"], color="tab:red", label="val - train")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("box_loss gap (a growing gap means overfitting)")

    for ax in axes.ravel():
        ax.axvline(best_epoch, color="green", ls="--", lw=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    val_min_epoch = int(df.loc[df["val/box_loss"].idxmin(), "epoch"])
    last_epoch = int(epochs.iloc[-1])
    print(
        f"Epochs run: {last_epoch} | best epoch (best.pt): {best_epoch} | lowest val box_loss at epoch {val_min_epoch}")
    if last_epoch > val_min_epoch and df["val/box_loss"].iloc[-1] > 1.1 * df["val/box_loss"].min():
        print("Warning: validation box_loss rose more than 10% above its minimum while training continued: "
              "possible overfitting.")

def show_run_plots(run_dir):
    """Show the confusion matrix, PR curve and F1 curve saved by Ultralytics in run_dir."""
    run_dir = Path(run_dir)
    found, missing = {}, []
    for title, candidates in PLOT_FILES.items():
        path = next((run_dir / n for n in candidates if (run_dir / n).exists()), None)
        if path:
            found[title] = path
        else:
            missing.append(title)

    if not found:
        available = sorted(p.name for p in run_dir.glob("*.png")) if run_dir.exists() else []
        raise FileNotFoundError(f"No evaluation plots in {run_dir}. PNG files found: {available or 'none'}")

    cols = 2
    rows = -(-len(found) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, path) in zip(axes, found.items()):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    for ax in axes[len(found):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    if missing:
        print("Not found in this folder:", ", ".join(missing))