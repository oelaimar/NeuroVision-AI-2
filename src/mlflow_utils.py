import os

import mlflow
import requests

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

DEFAULT_URI = "http://localhost:5000"

START_CMD = (
    "mlflow server "
    "--host 0.0.0.0 "
    "--port 5000 "
    "--allowed-hosts 'localhost:*,127.0.0.1:*,Tunnel domain' "
    "--cors-allowed-origins 'Tunnel url'"
)


def start_tracking(experiment: str = "neurovision-yolov8", tracking_uri=None, timeout: int = 3) -> None:
    """Connect to the MLflow server and select the experiment (created if missing)."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI)

    if uri.startswith("http"):
        try:
            requests.get(f"{uri.rstrip('/')}/health", timeout=timeout).raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(
                f"MLflow server not reachable at {uri}. Start it in a terminal:\n  {START_CMD}"
            ) from e
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(experiment)
    print(f"MLflow tracking: {uri} | experiment: {experiment} (id {exp.experiment_id})")
    return exp.experiment_id

def compare_runs(experiment:str = "neurovision-yolov8") -> pd.DataFrame:
    """Finished runs of an experiment, sorted by recall. Only the settings that DIFFER between runs are shown."""
    raw = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.recall DESC"],
    )
    if raw.empty:
        return raw

    params = [c for c in raw.columns if c.startswith("params.")]
    varying = [c for c in params if raw[c].nunique(dropna=False) > 1]

    out = pd.DataFrame({"run_id": raw["run_id"],  "run_name": raw["tags.mlflow.runName"]})
    for col in varying:
        out[col.removeprefix("params.")] = raw[col]
    for metric in ("precision", "recall", "map50", "map50_95", "best_epoch"):
        if f"metrics.{metric}" in raw:
            out[metric] = raw[f"metrics.{metric}"].round(4)
    if "metrics.duration_s" in raw:
        out["duration_min"] = (raw["metrics.duration_s"] / 60).round(1)
        return out

def pick_champion(runs: pd.DataFrame, max_map_drop: float = 0.05) -> str:
    """Medical rule: highest recall among runs whose mAP@0.5 is within max_map_drop of the best mAP@0.5."""
    eligible = runs[runs["map50"] >= runs["map50"].max() - max_map_drop]
    return eligible.sort_values("recall", ascending=False).iloc[0]["run_id"]

def register_champion(run_id: str, model_name: str = "neurovision-yolov8", alias: str = "champion") -> str:
    """Register the weights of a run in the Model Registry and point the alias to the new version."""
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except MlflowException:                      # already exists
        pass

    version = client.create_model_version(
        name=model_name, source=f"runs:/{run_id}/weights", run_id=run_id
    ).version

    metrics = client.get_run(run_id).data.metrics
    for key in ("recall", "precision", "map50", "map50_95"):
        if key in metrics:
            client.set_model_version_tag(model_name, version, key, f"{metrics[key]:.4f}")

    client.set_registered_model_alias(model_name, alias, version)
    print(f"{model_name} version {version} is now '{alias}' (run {run_id[:8]})")
    return version

