import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import Settings
from src.features.preprocessing import FEATURES, ensure_types
from src.utils.auth import configure_mlflow_auth


def load_champion(s: Settings):
    """Load the model currently holding the champion alias. Returns (model, version, run_id)."""
    configure_mlflow_auth()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(s.model_name, s.model_alias)
    model = mlflow.sklearn.load_model(f"models:/{s.model_name}/{mv.version}")
    return model, str(mv.version), mv.run_id


def predict_proba(model, records: list[dict]) -> list[float]:
    df = ensure_types(pd.DataFrame(records)[FEATURES])
    return [float(p) for p in model.predict_proba(df)[:, 1]]
