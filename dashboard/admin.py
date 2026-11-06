import os
from datetime import datetime, timezone

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import run_v2

from dashboard.auth import require_admin
from src.config import get_settings
from src.utils.logging import log_event

DAG_ID = "xgboost_training_pipeline"


def run_job_step(step: str, run_id: str) -> str:
    admin = require_admin()
    s = get_settings()
    job = os.getenv("PIPELINE_JOB_NAME", "mlops-pipeline-job")
    name = f"projects/{s.project_id}/locations/{s.region}/jobs/{job}"
    override = run_v2.RunJobRequest.Overrides.ContainerOverride(args=[step, "--run-id", run_id])
    req = run_v2.RunJobRequest(
        name=name, overrides=run_v2.RunJobRequest.Overrides(container_overrides=[override]))
    op = run_v2.JobsClient().run_job(request=req)
    log_event("admin_job_triggered", component="dashboard", user=admin, step=step, run_id=run_id)
    return op.operation.name


def _airflow():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds), os.environ["AIRFLOW_URL"].rstrip("/"), os.getenv("AIRFLOW_API", "v1")


def trigger_dag() -> dict:
    admin = require_admin()
    session, base, ver = _airflow()
    body = {"conf": {}}
    if ver == "v2":   # Airflow 3 needs a logical date
        body["logical_date"] = datetime.now(timezone.utc).isoformat()
    r = session.post(f"{base}/api/{ver}/dags/{DAG_ID}/dagRuns", json=body, timeout=90)
    r.raise_for_status()
    log_event("admin_dag_triggered", component="dashboard", user=admin)
    return r.json()


def set_dag_paused(paused: bool) -> None:
    admin = require_admin()
    session, base, ver = _airflow()
    r = session.patch(f"{base}/api/{ver}/dags/{DAG_ID}", params={"update_mask": "is_paused"},
                      json={"is_paused": paused}, timeout=90)
    r.raise_for_status()
    log_event("admin_dag_paused", component="dashboard", user=admin, paused=paused)