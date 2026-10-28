import os
import sys
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import hashlib
from itertools import combinations

import yaml
import pandas as pd
from tqdm.auto import tqdm

from src.annotations import validate_label
from src.config import cfg
 
SPLITS = ("train", "valid", "test")

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "valid", "valid": "valid", "validation": "valid",
    "test": "test", "testing": "test",
}

DATASET_SLUG = "ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes"

def download_dataset(raw_dir: Path = cfg.RAW_DIR, slug: str = DATASET_SLUG) -> Path:
    """Download the Kaggle dataset into raw_dir. Skipped if raw_dir already has files."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check whether the dataset has already been downloaded
    if any(raw_dir.iterdir()):
        print(f"Dataset already exists, skipping download: {raw_dir}")
        return raw_dir

    try:
        import kagglehub
    except ImportError as e:
        raise ImportError("kagglehub is missing: pip install kagglehub") from e

    print("Downloading dataset from Kaggle...")
    source = Path(kagglehub.dataset_download(slug))  # goes to the kagglehub cache first

    for item in source.iterdir():  # then copy into data/raw
        dest = raw_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"Dataset copied to: {raw_dir}")
    return raw_dir

def class_distribution(pairs) -> pd.DataFrame:
    """counts, for each tumor class, how many images it has and how many boxes. It returns a table that"""
    rows = []
    for folder in sorted(pairs):
        n_boxes = n_empty = 0
        for _, label in pairs[folder]:
            lines = [l for l in label.read_text().splitlines() if l.strip()]
            n_boxes += len(lines)
            n_empty += not lines
        rows.append({"class": folder, "images": len(pairs[folder]),
                     "boxes": n_boxes, "empty_labels": n_empty})
    return pd.DataFrame(rows)

def audit_dataset(pairs, raw_dir: Path = cfg.RAW_DIR):
    """Find every problem before training. Returns a dict of lists (empty list = no problem)."""
    # 1. Non-conforming image extensions (collect_pairs ignores them silently)
    bad_extensions = [p for p in raw_dir.rglob("images/*") if p.is_file() and p.suffix.lower() not in cfg.IMG_EXTS]
    corrupted, invalid, empty = [], [], []
    for class_name, items in pairs.items():
        for img, label in tqdm(items, desc=class_name):
            # 2. Corrupted images: cannot be read
            try:
                if cv2.imread(str(img)) is None:
                    raise ValueError("unreadable")
            except Exception:
                corrupted.append(img)

            # 3. Annotation format: 5 values, valid class, coordinates in 0..1
            errors = validate_label(label, num_classes=len(cfg.CLASS_NAMES))
            if errors:
                invalid.append((label, errors))

            # 4. Empty annotation: no tumor, or forgotten
            if not label.read_text().strip():
                empty.append(label)

    return {
        "bad_extensions": bad_extensions,
        "corrupted": corrupted,
        "invalid_annotations": invalid,
        "empty_annotations": empty,
    }


# collect pairs 
def collect_pairs(raw_dir: Path = cfg.RAW_DIR):
    """Collect (image, label) pairs by class from the original Train and Val folders.

    Returns (pairs_by_class, orphans). Images without a label are set aside and counted.
    """
    pairs, orphans, seen = defaultdict(list), [], set()
    for old_split in ("Train", "Val"):
        for class_dir in sorted((raw_dir / old_split).iterdir()):
            if not class_dir.is_dir():
                continue
            for img in sorted((class_dir / "images").iterdir()):
                if not img.is_file() or img.suffix.lower() not in cfg.IMG_EXTS:
                    continue
                label = class_dir / "labels" / f"{img.stem}.txt"
                if not label.exists():
                    orphans.append(img)
                    continue
                key = (class_dir.name, img.stem)
                if key in seen:
                    raise ValueError(f"Duplicate image name: {key}")
                seen.add(key)
                pairs[class_dir.name].append((img, label))
    return pairs, orphans

# split per class
def split_pairs(pairs, out: Path = cfg.RAW_SPLIT_DIR,
                val_ratio: float = cfg.VAL_RATIO, test_ratio: float = cfg.TEST_RATIO,
                seed: int = cfg.SEED):
    """Copy pairs into out/<Train|Val|Test>/<class>/{images,labels}. Returns a Counter."""
    shutil.rmtree(out, ignore_errors=True)  # rebuild only the output folder
    rng = random.Random(seed)
    counts = Counter()
    for class_name, items in pairs.items():
        for sp in ("Train", "Val", "Test"):
            for sub in ("images", "labels"):
                (out / sp / class_name / sub).mkdir(parents=True, exist_ok=True)
 
        items = sorted(items, key=lambda p: p[0].name) # fixed order, then seeded shuffle
        rng.shuffle(items)
        n_val = round(len(items) * val_ratio)
        n_test = round(len(items) * test_ratio)
        parts = {
            "Val": items[:n_val],
            "Test": items[n_val:n_val + n_test],
            "Train": items[n_val + n_test:],
        }
 
        prefix = class_name.replace(" ", "_") # unique names after flattening
        for sp, chunk in parts.items():
            for img, label in chunk:
                stem = f"{prefix}__{img.stem}"
                shutil.copy2(img, out / sp / class_name / "images" / f"{stem}{img.suffix.lower()}")
                shutil.copy2(label, out / sp / class_name / "labels" / f"{stem}.txt")
                counts[(sp, class_name)] += 1
    return counts

# organize in YOLO format
def detect_split(path: Path, root: Path):
    """Infer the split from the closest parent folder name (Train, Val, Test...)."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return None

def index_files(root: Path):
    """Index images and labels by (split, file name without extension)."""
    images, labels, duplicates = {}, {}, 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        key = (detect_split(p, root), p.stem)
        ext = p.suffix.lower()
        if ext in cfg.IMG_EXTS:
            duplicates += key in images
            images[key] = p
        elif ext == ".txt":
            duplicates += key in labels
            labels[key] = p
    return images, labels, duplicates


def organize(root: Path = cfg.RAW_SPLIT_DIR, out: Path = cfg.DATA_DIR):
    """Copy every image+label PAIR into out/images/<split> and out/labels/<split>."""
    images, labels, duplicates = index_files(root)
    pairs = sorted(set(images) & set(labels), key=lambda k: (str(k[0]), k[1]))

    for sub in ("images", "labels"):                     # rebuild from scratch: no stale files
        shutil.rmtree(out / sub, ignore_errors=True)
        for split in SPLITS:
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    copied, unsplit = Counter(), 0
    for split, stem in tqdm(pairs, desc="Copying pairs"):
        if split is None:
            unsplit += 1
            continue
        img = images[(split, stem)]
        shutil.copy2(img, out / "images" / split / img.name)
        shutil.copy2(labels[(split, stem)], out / "labels" / split / f"{stem}.txt")
        copied[split] += 1

    return {
        "pairs_copied": dict(copied),
        "images_without_label": len(set(images) - set(labels)),
        "labels_without_image": len(set(labels) - set(images)),
        "pairs_without_split": unsplit,
        "duplicate_names": duplicates,
    }


# ── Step 4: counts, data.yaml, validation ─────────────────────────────────────
def class_id_map(raw_dir: Path = cfg.RAW_DIR):
    """Which class ids appear in the labels of each class folder? Used to check cfg.CLASS_NAMES."""
    found = defaultdict(Counter)
    for label in raw_dir.rglob("labels/*.txt"):
        folder = label.parent.parent.name
        for line in label.read_text().splitlines():
            if line.strip():
                found[folder][int(line.split()[0])] += 1
    return dict(found)


def count_split(data_dir: Path = cfg.DATA_DIR):
    """Count images and labels per split. Returns {split: (n_images, n_labels, match)}."""
    result = {}
    for split in SPLITS:
        imgs = {p.stem for p in (data_dir / "images" / split).iterdir()
                if p.suffix.lower() in cfg.IMG_EXTS}
        lbls = {p.stem for p in (data_dir / "labels" / split).iterdir() if p.suffix == ".txt"}
        result[split] = (len(imgs), len(lbls), imgs == lbls)
    return result


def write_data_yaml(path: Path = cfg.DATA_YAML, data_dir: Path = cfg.DATA_DIR,
                    names=cfg.CLASS_NAMES):
    """Write data.yaml (paths, nc, names) for YOLOv8."""
    data = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(names),
        "names": dict(enumerate(names)),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return data


def validate_data_yaml(path: Path = cfg.DATA_YAML):
    """Check that data.yaml is valid. Returns a list of problems (empty list = valid)."""
    problems = []
    data = yaml.safe_load(path.read_text())
    root = Path(data["path"])
    for key in ("train", "val", "test"):
        if not (root / data[key]).is_dir():
            problems.append(f"{key}: folder not found ({root / data[key]})")
    if data["nc"] != len(data["names"]):
        problems.append(f"nc={data['nc']} but {len(data['names'])} names")
    if sorted(data["names"]) != list(range(data["nc"])):
        problems.append("class ids must be 0..nc-1 without gaps")
    for split, (n_img, n_lbl, match) in count_split(root).items():
        if not match:
            problems.append(f"{split}: {n_img} images vs {n_lbl} labels, names differ")
    return problems

def _split_images(split: str, data_dir: Path):
    return sorted(p for p in (data_dir / "images" / split).iterdir() if p.suffix.lower() in cfg.IMG_EXTS)

def _dominant_class(label_path: Path) -> int:
    """Most frequent class id in a label file (-1 if the file is empty)."""
    ids = [int(l.split()[0]) for l in label_path.read_text().splitlines() if l.strip()]
    return Counter(ids).most_common(1)[0][0] if ids else -1

def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def check_leakage(class_names=cfg.CLASS_NAMES, data_dir: Path = cfg.DATA_DIR):
    """
    Check class balance and data leakage between splits.
    """
    names = dict(enumerate(class_names))
    names[-1] = "no box"

    counts, stems, hashes = {}, {}, {}
    for split in SPLITS:
        imgs = _split_images(split, data_dir)
        classes = (_dominant_class(data_dir / "labels" / split / f"{p.stem}.txt") for p in imgs)
        counts[split] = Counter(names.get(c, f"unknown id {c}") for c in classes)
        stems[split] = {p.stem for p in imgs}
        hashes[split] = {_md5(p) for p in tqdm(imgs, desc=f"Hashing {split}")}

    balance = pd.DataFrame(counts).fillna(0).astype(int)
    balance = (balance / balance.sum() * 100).round(1)

    leakage = pd.DataFrame(
        [{"splits": f"{a} / {b}",
          "same_name": len(stems[a] & stems[b]),
          "same_content": len(hashes[a] & hashes[b])}
         for a, b in combinations(SPLITS, 2)]
    ).set_index("splits")
    return balance, leakage

if __name__ == "__main__":

    pairs, orphans = collect_pairs()
    print("Pairs per class:", {c: len(v) for c, v in pairs.items()})
    print("Total pairs    :", sum(len(v) for v in pairs.values()))
    print("Orphan images  :", len(orphans))

    counts = split_pairs(pairs)
    report = organize()
    print(json.dumps(report, indent=2))