import random

import cv2
import matplotlib.pyplot as plt

from src.annotations import read_yolo_label, draw_boxes

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

        pairs: {class_folder: [(image_path, label_path), ...]} from collect_pairs().
        Images are spread evenly over the classes and the choice is reproducible (seed).
    """
    samples = _pick_samples(pairs, n, seed)
    rows = -(-len(samples) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_cell, rows * figsize_per_cell))
    axes = axes.reval() if hasattr(axes, "reval") else [axes]

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


