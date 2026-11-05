"""DAG `xgboost_training_pipeline`.

Airflow only ORCHESTRATES: every heavy step runs inside the Cloud Run Job
`mlops-pipeline-job` (same image, different sub-command: see src/cli.py).
Configuration comes from Composer environment variables (see README).
"""
import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
REGION = os.getenv("GCP_REGION", "europe-west1")
JOB_NAME = os.getenv("PIPELINE_JOB_NAME", "mlops-pipeline-job")
API_URL = os.getenv("API_URL", "").rstrip("/")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _log(event: str, severity: str = "INFO", **fields):
    print(json.dumps({"severity": severity, "message": event, "event": event,
                      "component": "airflow", **fields}, default=str))


def job_task(task_id: str, command: str) -> CloudRunExecuteJobOperator:
    return CloudRunExecuteJobOperator(
        task_id=task_id,
        project_id=PROJECT_ID,
        region=REGION,
        job_name=JOB_NAME,
        overrides={"container_overrides": [{"args": [command, "--run-id", "{{ ts_nodash }}"]}]},
    )


def reload_api(**_):
    """Tell the prediction API to load the (possibly new) champion model."""
    import google.auth.transport.requests
    import google.oauth2.id_token
    import requests

    if not API_URL:
        _log("api_reload_skipped", severity="WARNING", reason="API_URL not set")
        return
    request = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(request, API_URL)
    r = requests.post(f"{API_URL}/reload", headers={"Authorization": f"Bearer {token}"},
                      timeout=180)
    r.raise_for_status()
    _log("api_reloaded", response=r.json())


def pipeline_report(**context):
    dag_run = context["dag_run"]
    states = {ti.task_id: ti.state for ti in dag_run.get_task_instances()}
    failed = [t for t, st in states.items() if st == "failed"]
    _log("pipeline_report", severity="ERROR" if failed else "INFO",
         dag_run_id=dag_run.run_id, task_states=states, failed_tasks=failed)


with DAG(
    dag_id="xgboost_training_pipeline",
    description="Data pipeline + XGBoost training + MLflow registration",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,  # manual trigger to keep Composer costs low; e.g. "@weekly" later
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "xgboost", "mlflow"],
) as dag:
    extract = job_task("extract_data", "extract")
    validate = job_task("validate_data", "validate")
    preprocess = job_task("preprocess_data", "preprocess")
    load_bq = job_task("load_bigquery", "load")
    train = job_task("trigger_training", "train")
    evaluate = job_task("evaluate_model", "evaluate")
    register = job_task("register_model", "register")
    reload_task = PythonOperator(task_id="reload_api", python_callable=reload_api)
    report = PythonOperator(task_id="pipeline_report", python_callable=pipeline_report,
                            trigger_rule="all_done")

    (extract >> validate >> preprocess >> load_bq >> train >> evaluate >> register
     >> reload_task >> report)
    # report also runs when something failed, to log the final status
    [extract, validate, preprocess, load_bq, train, evaluate, register] >> report
