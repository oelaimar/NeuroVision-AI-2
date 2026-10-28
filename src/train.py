import torch
from pathlib import Path
import time
from contextlib import nullcontext
import matplotlib.pyplot as plt

import mlflow
import pandas as pd


from src.config import cfg, AUGMENTATION

GB = 1024 ** 3

def check_gpu(raise_if_missing: bool = False) -> dict:
    """Check that PyTorch sees a CUDA GPU. Returns a JSON-friendly dict."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": 0,
        "device_name": None,
        "total_memory_gb": None,
        "free_memory_gb": None,
    }
    if info["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        info.update(
            device_count=torch.cuda.device_count(),
            device_name=torch.cuda.get_device_name(0),
            total_memory_gb=round(total / GB, 1),
            free_memory_gb=round(free / GB, 1),
        )
    elif raise_if_missing:
        raise RuntimeError("No CUDA GPU found: check the Vast instance and the PyTorch/CUDA install.")
    else:
        print("WARNING: no CUDA GPU found, training would run on CPU (very slow).")
    return info

def preview_augmentation(n: int = 6, model_name: str = cfg.MODEL):
    """Train for a moment on a small subset and show the augmented training images with their boxes."""
    from ultralytics import YOLO

    total = sum(1 for p in (cfg.DATA_DIR / "images" / "train").iterdir()
                if p.suffix.lower() in cfg.IMG_EXTS)
    fraction = min(1.0, max(n * 3 / total, 0.01))          # tiny subset, enough for one batch
    project = cfg.RUNS_DIR.resolve() / "augmentation_preview"

    model = YOLO(model_name)
    model.train(
        data=str(cfg.DATA_YAML), epochs=1, imgsz=cfg.IMAGE_SIZE, batch=n, fraction=fraction,
        close_mosaic=0, plots=True, seed=cfg.SEED, device=cfg.DEVICE, workers=cfg.NUM_WORKERS,
        project=str(project), name="preview", exist_ok=True, verbose=False, **AUGMENTATION,
    )

    batch_file = Path(model.trainer.save_dir) / "train_batch0.jpg"   # saved by Ultralytics
    if not batch_file.exists():
        raise FileNotFoundError(f"{batch_file} not found: the preview run did not produce its plots")

    plt.figure(figsize=(10, 10))
    plt.imshow(plt.imread(batch_file))
    plt.axis("off")
    plt.title(f"Augmented training images ({n} per batch, boxes drawn)")
    plt.show()
    return batch_file

def read_results(run_dir) -> pd.DataFrame:
    """Read results.csv written by Ultralytics (one row per epoch, clean column names)."""
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def summarize_results(run_dir) -> dict:
    """Best-epoch metrics (same fitness rule as best.pt) and last-epoch losses."""
    df = read_results(run_dir)
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    best, last = df.loc[fitness.idxmax()], df.iloc[-1]
    return {
        "best_epoch": int(best["epoch"]),
        "epochs_run": int(len(df)),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "train_box_loss": float(last["train/box_loss"]),
        "train_cls_loss": float(last["train/cls_loss"]),
        "val_box_loss": float(last["val/box_loss"]),
        "val_cls_loss": float(last["val/cls_loss"]),
    }

def train_yolo(model_name: str = cfg.MODEL, epochs: int = cfg.EPOCHS, run_name=None,
               imgsz: int = cfg.IMAGE_SIZE, batch: int = cfg.BATCH_SIZE,
               patience: int = cfg.PATIENCE, augmentation=None,
               mlflow_log: bool = True, resume_from=None) -> dict:
    """Train YOLOv8 and (optionally) log the run to MLflow."""
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})          # we log to MLflow ourselves, avoid duplicate runs
    aug = dict(AUGMENTATION if augmentation is None else augmentation)
    run_name = run_name or f"{Path(model_name).stem}-{time.strftime('%Y%m%d-%H%M%S')}"

    params = {"model": model_name, "epochs": epochs, "imgsz": imgsz, "batch": batch,
              "patience": patience, "seed": cfg.SEED, "device": cfg.DEVICE,
              **{f"aug_{k}": v for k, v in aug.items()}}

    def fit() -> Path:
        if resume_from:                                   # continue an interrupted run
            model = YOLO(str(resume_from))
            model.train(resume=True)
        else:
            model = YOLO(model_name)                      # COCO pre-trained weights
            model.train(
                data=str(cfg.DATA_YAML), epochs=epochs, imgsz=imgsz, batch=batch,
                patience=patience, seed=cfg.SEED, device=cfg.DEVICE, workers=cfg.NUM_WORKERS,
                project=str(cfg.RUNS_DIR.resolve() / "detect"), name=run_name, **aug,
            )
        return Path(model.trainer.save_dir)

    run_id = None
    with (mlflow.start_run(run_name=run_name) if mlflow_log else nullcontext()) as active:
        if mlflow_log:
            mlflow.log_params(params)                     # logged first: a failed run keeps its settings
            run_id = active.info.run_id

        start = time.perf_counter()
        run_dir = fit()
        duration = time.perf_counter() - start
        metrics = summarize_results(run_dir)

        if mlflow_log:
            mlflow.log_metrics({**metrics, "duration_s": duration})
            for f in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
                if f.exists():
                    mlflow.log_artifact(str(f), "weights")
            for f in [run_dir / "results.csv", run_dir / "args.yaml", cfg.DATA_YAML, *run_dir.glob("*.png")]:
                if f.exists():
                    mlflow.log_artifact(str(f), "run_files")

    return {"run_dir": run_dir, "duration_s": duration, "metrics": metrics,
            "mlflow_run_id": run_id, "best_weights": run_dir / "weights" / "best.pt"}
