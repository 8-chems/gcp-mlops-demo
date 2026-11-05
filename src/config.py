"""Central configuration, read from environment variables at call time."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project_id: str
    region: str
    bucket: str
    bq_dataset: str
    training_table: str
    prediction_table: str
    mlflow_tracking_uri: str
    experiment_name: str
    model_name: str
    model_alias: str
    min_roc_auc: float
    sample_rows: int
    raw_blob: str
    seed: int = 42

    @property
    def training_table_id(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.{self.training_table}"

    @property
    def prediction_table_id(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.{self.prediction_table}"

    def gcs(self, path: str) -> str:
        return f"gs://{self.bucket}/{path}"


def get_settings() -> Settings:
    return Settings(
        project_id=os.getenv("GCP_PROJECT_ID", ""),
        region=os.getenv("GCP_REGION", "europe-west1"),
        bucket=os.getenv("GCS_BUCKET", ""),
        bq_dataset=os.getenv("BQ_DATASET", "mlops_demo"),
        training_table=os.getenv("BQ_TRAINING_TABLE", "training_data"),
        prediction_table=os.getenv("BQ_PREDICTION_TABLE", "prediction_results"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "bank-marketing-xgboost"),
        model_name=os.getenv("MODEL_NAME", "bank-marketing-xgb"),
        model_alias=os.getenv("MODEL_ALIAS", "champion"),
        min_roc_auc=float(os.getenv("MIN_ROC_AUC", "0.75")),
        sample_rows=int(os.getenv("SAMPLE_ROWS", "15000")),
        raw_blob=os.getenv("RAW_BLOB", "raw/bank-additional-full.csv"),
    )
