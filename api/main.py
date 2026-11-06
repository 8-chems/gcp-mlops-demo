"""Prediction API: serves the MLflow model that currently holds the `champion` alias."""
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
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


app = FastAPI(title="Bank Marketing Prediction API", version="1.1.0", lifespan=lifespan)


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
def predict(
    customer: Customer,
    x_submitted_by: Annotated[str | None, Header()] = None,
):
    """Predict for one customer.

    The optional `X-Submitted-By` header carries the signed-in user's e-mail. It is sent by the
    dashboard (the only caller allowed on this private service) and stored with the prediction.
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded yet (run the pipeline).")
    submitted_by = (x_submitted_by or "").strip()[:254] or None
    record = customer.model_dump()
    proba = predict_proba(state["model"], [record])[0]
    pred = int(proba >= 0.5)
    log_event("model_prediction", component="api", model="xgboost",
              model_version=state["version"], prediction=pred, probability=round(proba, 4),
              submitted_by=submitted_by)
    if os.getenv("LOG_PREDICTIONS_TO_BQ", "false").lower() == "true":
        _store_prediction(record, pred, proba, submitted_by)
    return Prediction(prediction=pred, probability=proba, model_version=state["version"])


def _store_prediction(record: dict, pred: int, proba: float, submitted_by: str | None) -> None:
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
            "submitted_by": submitted_by,
        }], s.project_id)
        if errors:
            log_event("prediction_store_failed", severity="WARNING", component="api",
                      errors=errors)
    except Exception as exc:  # noqa: BLE001
        log_event("prediction_store_failed", severity="WARNING", component="api", error=repr(exc))