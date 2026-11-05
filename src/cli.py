"""Pipeline CLI. Each sub-command is one Airflow task, executed in a Cloud Run Job:

    python -m src.cli <extract|validate|preprocess|load|train|evaluate|register> --run-id X
"""
import argparse
import os
import re
import sys
from datetime import datetime, timezone

from src.config import get_settings
from src.data.ingestion import read_raw_csv, sample_rows
from src.data.validation import assert_valid, validate
from src.features.preprocessing import add_split, clean
from src.utils import bigquery, gcp_storage
from src.utils.logging import log_event

EXTRACTED = "processed/extracted.csv"
CLEAN = "processed/clean.csv"


def _run_id(value: str | None) -> str:
    default = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    raw = value or os.getenv("PIPELINE_RUN_ID") or default
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def step_extract(s, run_id):
    df = read_raw_csv_from_gcs(s)
    sample = sample_rows(df, s.sample_rows, s.seed)
    gcp_storage.write_csv(sample, s.gcs(EXTRACTED))
    log_event("data_extracted", component="data", rows_raw=len(df), rows_sampled=len(sample))


def read_raw_csv_from_gcs(s):
    import io

    from google.cloud import storage

    bucket, blob = gcp_storage.parse_uri(s.gcs(s.raw_blob))
    data = storage.Client().bucket(bucket).blob(blob).download_as_bytes()
    return read_raw_csv(io.BytesIO(data))


def step_validate(s, run_id):
    df = gcp_storage.read_csv(s.gcs(EXTRACTED), sep=None, engine="python")
    report = validate(df)
    gcp_storage.write_json(report, s.gcs(f"reports/validation_{run_id}.json"))
    log_event("data_validated", component="data", **report)
    assert_valid(report)


def step_preprocess(s, run_id):
    df = gcp_storage.read_csv(s.gcs(EXTRACTED), sep=None, engine="python")
    out = clean(df)
    gcp_storage.write_csv(out, s.gcs(CLEAN))
    log_event("data_preprocessed", component="data", rows_in=len(df), rows_out=len(out))


def step_load(s, run_id):
    df = gcp_storage.read_csv(s.gcs(CLEAN))
    df = add_split(df, 0.2, s.seed)
    n = bigquery.load_dataframe(df, s.training_table_id, s.project_id)
    log_event("data_loaded_bigquery", component="data", table=s.training_table_id, rows=n)


def step_train(s, run_id):
    from src.models.train import run_training

    run_training(run_id)


def step_evaluate(s, run_id):
    from src.models.evaluate import run

    run("evaluate", run_id)


def step_register(s, run_id):
    from src.models.evaluate import run

    run("register", run_id)


STEPS = {
    "extract": step_extract, "validate": step_validate, "preprocess": step_preprocess,
    "load": step_load, "train": step_train, "evaluate": step_evaluate, "register": step_register,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="src.cli")
    p.add_argument("step", choices=STEPS)
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)
    run_id, s = _run_id(args.run_id), get_settings()
    log_event("pipeline_step_started", component="pipeline", step=args.step, pipeline_run_id=run_id)
    try:
        STEPS[args.step](s, run_id)
    except Exception as exc:  # noqa: BLE001
        log_event("pipeline_failure", severity="ERROR", component="pipeline",
                  step=args.step, pipeline_run_id=run_id, error=repr(exc))
        return 1
    log_event("pipeline_step_completed", component="pipeline", step=args.step,
              pipeline_run_id=run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
