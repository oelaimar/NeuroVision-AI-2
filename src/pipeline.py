"""
    python -m src.pipeline prepare
    python -m src.pipeline train    --run-name airflow-20261030      # prints the MLflow run id (last line)
    python -m src.pipeline evaluate --run-id RUN_ID
    python -m src.pipeline register --run-id RUN_ID                  # prints the new model version (last line)
    python -m src.pipeline promote  --run-id RUN_ID --version VERSION
 
Run from the project root. Each command exits with an error code on failure, so Airflow can retry it.
"""

import argparse
import json
import os
 
from dotenv import load_dotenv
 
from src.config import cfg

load_dotenv(cfg.PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv("MODEL_NAME", "neurovision-yolov8")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MAX_MAP_DROP = float(os.getenv("MAX_MAP_DROP", "0.05")) 

def _tracking() -> None:
    from src.mlflow_utils import start_tracking
    start_tracking()

def cmd_prepare(args) -> None:
    """Download the dataset (skipped if present), split it, organize it and write data.yaml."""
    from src.dataset import download_dataset, prepare_dataset
    download_dataset()
    report, _ = prepare_dataset()
    print(json.dumps(report))

def cmd_train(args) -> None:
    """Train YOLOv8 with MLflow logging. The LAST line printed is the MLflow run id (read by Airflow)."""
    from src.train import train_yolo
    _tracking()
    result = train_yolo(model_name=args.model or cfg.MODEL, epochs=args.epochs or cfg.EPOCHS,
                        run_name=args.run_name)
    print(result["mlflow_run_id"], flush=True)

def cmd_evaluate(args) -> None:
    """Evaluate the new model on the VALIDATION split (never on test) and log the metrics to its run."""
    import mlflow
    from src.evaluate import evaluate, load_run_model
    _tracking()
    model = load_run_model(args.run_id)
    global_metrics, _ = evaluate(model, split="val", name="pipeline-new")
    row = global_metrics.iloc[0]
    metrics = {"eval_precision": float(row["precision"]), "eval_recall": float(row["recall"]),
               "eval_map50": float(row["mAP@0.5"]), "eval_map50_95": float(row["mAP@0.5:0.95"])}
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics(metrics)
    print(json.dumps(metrics))

def cmd_register(args) -> None:
    """Add the model to the registry as a new version (no alias yet). Prints the version."""
    from src.mlflow_utils import register_version
    _tracking()
    print(register_version(args.run_id, MODEL_NAME), flush=True)

def cmd_promote(args) -> None:
    """Move the alias to the new version ONLY if it beats the current champion on the same validation split."""
    from mlflow import MlflowClient
    from src.evaluate import evaluate, load_model
    from src.mlflow_utils import is_better, set_alias
    _tracking()
 
    run_metrics = MlflowClient().get_run(args.run_id).data.metrics
    new = {"recall": run_metrics["eval_recall"], "map50": run_metrics["eval_map50"]}
 
    try:
        champion, _ = load_model(alias=MODEL_ALIAS, model_name=MODEL_NAME)
    except RuntimeError:                                         # no champion yet
        set_alias(MODEL_NAME, args.version, MODEL_ALIAS)
        print(f"No champion yet: version {args.version} is now '{MODEL_ALIAS}'.")
        return

    champ_global, _ = evaluate(champion, split="val", name="pipeline-champion")
    row = champ_global.iloc[0]
    current = {"recall": float(row["recall"]), "map50": float(row["mAP@0.5"])}
 
    print(f"new      : recall={new['recall']:.4f} map50={new['map50']:.4f}")
    print(f"champion : recall={current['recall']:.4f} map50={current['map50']:.4f}")
    if is_better(new, current, MAX_MAP_DROP):
        set_alias(MODEL_NAME, args.version, MODEL_ALIAS)
        print(f"PROMOTED: version {args.version} is now '{MODEL_ALIAS}'.")
    else:
        print(f"KEPT: the current champion stays (version {args.version} not promoted).")

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroVision pipeline steps")
    sub = parser.add_subparsers(dest="command"ssh -p 12526 root@79.160.189.79 -L 8080:localhost:8080, required=True)
 
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
 
    p = sub.add_parser("train")
    p.add_argument("--run-name", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.set_defaults(func=cmd_train)
 
    for name, func in (("evaluate", cmd_evaluate), ("register", cmd_register), ("promote", cmd_promote)):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
        if name == "promote":
            p.add_argument("--version", required=True)
        p.set_defaults(func=func)
 
    args = parser.parse_args()
    args.func(args)
 
 
if __name__ == "__main__":
    main()


