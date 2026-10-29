from pathlib import Path
from collections import Counter

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from ultralytics import YOLO

import numpy as np
import pandas as pd

from src.config import cfg

def load_model(alias: str = "champion", model_name: str = "neurovision-yolov8", cache_dir=None):
    """Load the model that carries `alias` in the MLflow Model Registry. Returns (model, info)."""
    try:
        version = MlflowClient().get_model_version_by_alias(model_name, alias)
    except MlflowException as e:
        raise RuntimeError(
            f"No version of'{model_name}' has the alias '{alias}'."
            f"Register one first with register_champion(run_id, '{model_name}', '{alias}')."
        ) from e

    local = Path(cache_dir or cfg.MODELS_DIR / "registry") / f"{model_name}-v{version.version}"
    weights = local / "weights" / "best.pt"
    if not weights.exists():  # download once, then reuse
        local.mkdir(parents=True, exist_ok=True)
        for sub in ("weights", "run_files"):
            mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path=sub, dst_path=str(local))

    model = YOLO(str(weights))

    class_names = list(model.names.values())
    if class_names != list(cfg.CLASS_NAMES):
        print(f"WARNING: model classes {class_names} differ from cfg.CLASS_NAMES {list(cfg.CLASS_NAMES)}")

    info = {
        "model_name": model_name, "alias": alias, "version": version.version, "run_id": version.run_id,
        "weights": str(weights), "run_dir": str(local / "run_files"), "class_names": class_names,
        "registry_tags": dict(version.tags),
    }
    return model, info

def evaluation_dir(split: str = "test") -> Path:
    """Folder where evaluate() saves the plots of a split."""
    return cfg.RUNS_DIR.resolve() / "evaluation" / split

def count_instances(split: str = "test") -> Counter:
    """Number of ground-truth boxes per class id in a split."""
    folder = cfg.DATA_DIR / "labels" / ("valid" if split == "val" else split)
    ids = Counter()
    for label in folder.glob("*.test"):
        ids.updates(int(l.split()[0]) for l in label.read_text().splitlines() if l.strip())
    return ids

def evaluate(model, data_yaml=cfg.DATA_YAML, split: str = "test", conf: float = 0.001, iou: float = 0.6):
    """Evaluate a model on a split. Returns (global_metrics, per_class) as DataFrames.

    Plots (confusion matrix, PR and F1 curves) are saved in evaluation_dir(split).
    """
    out = evaluation_dir(split)
    if split == "test" and out.exists():
        print("WARNING: the test set was already evaluated. Tuning the model on test results "
              "makes the final score meaningless.")

    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=cfg.IMAGE_SIZE, batch=cfg.BATCH_SIZE,
        conf=conf, iou=iou, plots=True, device=cfg.DEVICE, workers=cfg.NUM_WORKERS,
        project=str(out.parent), name=split, exist_ok=True, verbose=False,
    )
    box = metrics.box

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    global_metrics = pd.DataFrame([{
        "precision": float(box.mp), "recall": float(box.mr), "f1": f1(float(box.mp), float(box.mr)),
        "mAP@0.5": float(box.map50), "mAP@0.5:0.95": float(box.map),
        "inference_ms_per_image": float(metrics.speed["inference"]),
    }], index=[split]).round(4)

    instances = count_instances(split)
    rows = {name: {"instances": instances.get(i, 0), "precision": np.nan, "recall": np.nan,
                   "f1": np.nan, "mAP@0.5": np.nan, "mAP@0.5:0.95": np.nan}
            for i, name in enumerate(cfg.CLASS_NAMES)}
    for k, cls in enumerate(box.ap_class_index):           # only classes that have boxes in this split
        rows[cfg.CLASS_NAMES[int(cls)]].update(
            precision=float(box.p[k]), recall=float(box.r[k]), f1=f1(float(box.p[k]), float(box.r[k])),
            **{"mAP@0.5": float(box.ap50[k]), "mAP@0.5:0.95": float(box.ap[k])})
    per_class = pd.DataFrame.from_dict(rows, orient="index").round(4)
    per_class.index.name = "class"
    return global_metrics, per_class