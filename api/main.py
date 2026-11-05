"""Prediction API: serves the MLflow model that currently holds the `champion` alias."""
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.models.predict import load_champion, predict_proba
from src.utils.logging import log_event

state = {"model": None, "version": None, "run_id": None}


def reload_model() -> None:
    s = get_settings()
    model, version, run_id = load_champion(s)
    state.update(model=model, version=version, run_id=run_id)
    log_event("model_loaded", component="api", model_name=s.model_name, model_version=version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        reload_model()
    except Exception as exc:  # noqa: BLE001 - the API must start even without a model yet
        log_event("model_load_failed", severity="WARNING", component="api", error=repr(exc))
    yield


app = FastAPI(title="Bank Marketing Prediction API", version="1.0.0", lifespan=lifespan)


class Customer(BaseModel):
    age: float = Field(..., ge=16, le=110)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: float = Field(..., ge=1)
    pdays: float
    previous: float = Field(..., ge=0)
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


class Prediction(BaseModel):
    prediction: int
    probability: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_version": state["version"]}


@app.get("/model-info")
def model_info():
    s = get_settings()
    return {"model_name": s.model_name, "alias": s.model_alias,
            "model_version": state["version"], "run_id": state["run_id"]}


@app.post("/reload")
def reload():
    try:
        reload_model()
    except Exception as exc:  # noqa: BLE001
        log_event("model_load_failed", severity="ERROR", component="api", error=repr(exc))
        raise HTTPException(
            status_code=502, detail=f"Could not load champion model: {exc}"
        ) from exc
    return {"status": "reloaded", "model_version": state["version"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded yet (run the pipeline).")
    record = customer.model_dump()
    proba = predict_proba(state["model"], [record])[0]
    pred = int(proba >= 0.5)
    log_event("model_prediction", component="api", model="xgboost",
              model_version=state["version"], prediction=pred, probability=round(proba, 4))
    if os.getenv("LOG_PREDICTIONS_TO_BQ", "false").lower() == "true":
        _store_prediction(record, pred, proba)
    return Prediction(prediction=pred, probability=proba, model_version=state["version"])


def _store_prediction(record: dict, pred: int, proba: float) -> None:
    try:
        from src.utils.bigquery import insert_rows

        s = get_settings()
        errors = insert_rows(s.prediction_table_id, [{
            "prediction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": s.model_name,
            "model_version": state["version"],
            "prediction": pred,
            "probability": proba,
            "features": json.dumps(record),
        }], s.project_id)
        if errors:
            log_event("prediction_store_failed", severity="WARNING", component="api",
                      errors=errors)
    except Exception as exc:  # noqa: BLE001
        log_event("prediction_store_failed", severity="WARNING", component="api", error=repr(exc))
