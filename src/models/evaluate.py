"""Model gating and registration: select the best run, check threshold, promote champion."""
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import Settings, get_settings
from src.utils.logging import log_event


class ThresholdNotMet(Exception):
    pass


def select_best_run(client: MlflowClient, experiment_name: str, pipeline_run_id: str,
                    metric: str = "roc_auc", only_candidates: bool = False):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    flt = f"tags.pipeline_run_id = '{pipeline_run_id}'"
    if only_candidates:
        flt += " and tags.candidate = 'true'"
    runs = client.search_runs([exp.experiment_id], filter_string=flt,
                              order_by=[f"metrics.{metric} DESC"], max_results=20)
    return runs[0] if runs else None


def evaluate(client: MlflowClient, s: Settings, pipeline_run_id: str) -> dict:
    best = select_best_run(client, s.experiment_name, pipeline_run_id)
    if best is None:
        raise RuntimeError(f"No MLflow runs found for pipeline_run_id={pipeline_run_id}")
    auc = best.data.metrics["roc_auc"]
    info = {"run_id": best.info.run_id, "roc_auc": auc, "threshold": s.min_roc_auc}
    if auc < s.min_roc_auc:
        log_event("model_evaluation_failed", severity="ERROR", component="evaluation", **info)
        raise ThresholdNotMet(f"roc_auc {auc:.4f} < threshold {s.min_roc_auc}")
    client.set_tag(best.info.run_id, "candidate", "true")
    log_event("model_evaluation_passed", component="evaluation", **info)
    return info


def _champion(client: MlflowClient, s: Settings):
    try:
        return client.get_model_version_by_alias(s.model_name, s.model_alias)
    except MlflowException:
        return None


def register(client: MlflowClient, s: Settings, pipeline_run_id: str) -> dict:
    best = select_best_run(client, s.experiment_name, pipeline_run_id, only_candidates=True)
    if best is None:
        raise RuntimeError("No candidate run: run the `evaluate` step first.")
    auc = best.data.metrics["roc_auc"]
    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", s.model_name)
    client.set_model_version_tag(s.model_name, mv.version, "roc_auc", f"{auc:.6f}")

    champ = _champion(client, s)
    champ_auc = client.get_run(champ.run_id).data.metrics.get("roc_auc", 0.0) if champ else -1.0
    promoted = auc > champ_auc
    if promoted:
        client.set_registered_model_alias(s.model_name, s.model_alias, mv.version)
    info = {"model_name": s.model_name, "version": str(mv.version), "roc_auc": auc,
            "previous_champion_roc_auc": champ_auc, "promoted": promoted}
    log_event("model_registered", component="registry", **info)
    return info


def run(step: str, pipeline_run_id: str) -> dict:
    from src.utils.auth import configure_mlflow_auth

    s = get_settings()
    configure_mlflow_auth()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    client = MlflowClient()
    return evaluate(client, s, pipeline_run_id) if step == "evaluate" \
        else register(client, s, pipeline_run_id)
