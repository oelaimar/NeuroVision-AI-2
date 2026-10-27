import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
 
from tqdm.auto import tqdm
 
from src.config import cfg
 
SPLITS = ("train", "valid", "test")

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "valid", "valid": "valid", "validation": "valid",
    "test": "test", "testing": "test",
}

# collect pairs 
def collect_pairs(raw_dir: Path, Path = cfg.RAW_DIR):
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

def index_files(root Path):
    """Index images and labels by (split, file name without extension)."""
    images, labels, duplicates = {}, {}, 0
    for p in root.rglob("*"):
        if not p.is_files():
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
    pairs = sorted(set(images) & set(labels), key= lambda k: (str(k[0]), k[1]))

    for sub in ("images", "labels"):# rebuild from scratch: no stale files
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

def prepare_dataset():
    """Run collect -> split -> organize. Used by notebooks and by the Airflow task."""
    pairs, orphans = collect_pairs()
    counts = split_pairs(pairs)
    report = organize()
    report["orphan_images"] = len(orphans)
    return report, counts


if __name__ == "__main__":
    report, _ = prepare_dataset()
    print(json.dumps(report, indent=2))
