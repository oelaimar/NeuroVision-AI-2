"""
prepare_data >> train_model >> evaluate_model >> register_model >> promote_if_better
 
Each task calls one command of src/pipeline.py, run with the PROJECT's Python (the environment that has
torch and ultralytics), not the Airflow venv. A task starts only if the previous one succeeded;
a failing task is retried, then the run stops with an error.
 
Environment variables (set before starting Airflow):
    PROJECT_ROOT    absolute path of the project
    PROJECT_PYTHON  absolute path of the project's python (find it BEFORE activating the Airflow venv: `which python`)
"""
import os
from datetime import datetime, timedelta
 
from airflow import DAG
from airflow.operators.bash import BashOperator
 
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/home/user/NeuroVision-AI-2")
PROJECT_PYTHON = os.getenv("PROJECT_PYTHON", "python3")
RUN = f"cd {PROJECT_ROOT} && {PROJECT_PYTHON} -m src.pipeline"

RUN_ID = "{{ ti.xcom_pull(task_ids='train_model') }}"          # last line printed by the train task
VERSION = "{{ ti.xcom_pull(task_ids='register_model') }}"      # last line printed by the register task
 
default_args = {
    "owner": "neurovision",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="neurovision_pipeline",
    description="Prepare data, train YOLOv8, evaluate, register, promote if better than the champion",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,                       # never two trainings at the same time on one GPU
    default_args=default_args,
    tags=["neurovision", "yolov8"],
    doc_md=__doc__,
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"{RUN} prepare",
    )
 
    train_model = BashOperator(
        task_id="train_model",
        bash_command=RUN + " train --run-name airflow-{{ ds_nodash }}",
        do_xcom_push=True,                   # the last stdout line (MLflow run id) goes to XCom
        retries=1,                           # a retry restarts training from scratch: keep it low
        execution_timeout=timedelta(hours=8),
    )
 
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"{RUN} evaluate --run-id {RUN_ID}",
    )
 
    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"{RUN} register --run-id {RUN_ID}",
        do_xcom_push=True,                   # the new model version number
    )
 
    promote_if_better = BashOperator(
        task_id="promote_if_better",
        bash_command=f"{RUN} promote --run-id {RUN_ID} --version {VERSION}",
    )
 
    prepare_data >> train_model >> evaluate_model >> register_model >> promote_if_better
