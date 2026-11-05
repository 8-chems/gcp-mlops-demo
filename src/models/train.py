"""XGBoost training with MLflow tracking (3 hyper-parameter experiments by default)."""
import json
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.features.preprocessing import CATEGORICAL, FEATURES, NUMERIC, TARGET  # noqa: E402
from src.utils.logging import log_event  # noqa: E402

DEFAULT_CONFIGS = [
    {"name": "exp1", "max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
    {"name": "exp2", "max_depth": 5, "learning_rate": 0.05, "n_estimators": 150},
    {"name": "exp3", "max_depth": 7, "learning_rate": 0.01, "n_estimators": 200},
]


def build_pipeline(cfg: dict, seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = XGBClassifier(
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        subsample=0.8,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _save_artifacts(pipe: Pipeline, X_test, y_test, y_pred, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=110)
    plt.close(fig)

    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("Top 20 feature importances")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=110)
    plt.close(fig)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    (out / "classification_report.json").write_text(json.dumps(report, indent=2))


def train_and_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    configs: list[dict] | None = None,
    experiment_name: str = "bank-marketing-xgboost",
    pipeline_run_id: str = "manual",
    seed: int = 42,
) -> list[dict]:
    """Train one model per config, log everything to MLflow, return run summaries."""
    configs = configs or DEFAULT_CONFIGS
    mlflow.set_experiment(experiment_name)
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    results = []

    for cfg in configs:
        with mlflow.start_run(run_name=f"{cfg['name']}-{pipeline_run_id}") as run:
            log_event("model_training_started", component="training",
                      run_id=run.info.run_id, config=cfg, pipeline_run_id=pipeline_run_id)
            t0 = time.time()
            mlflow.set_tags({"pipeline_run_id": pipeline_run_id, "model_family": "xgboost"})
            mlflow.log_params({
                "max_depth": cfg["max_depth"], "learning_rate": cfg["learning_rate"],
                "n_estimators": cfg["n_estimators"], "subsample": 0.8, "random_state": seed,
                "train_rows": len(train_df), "test_rows": len(test_df),
            })
            pipe = build_pipeline(cfg, seed).fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = compute_metrics(y_test, pred, proba)
            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as tmp:
                _save_artifacts(pipe, X_test, y_test, pred, Path(tmp))
                mlflow.log_artifacts(tmp, artifact_path="evaluation")

            mlflow.sklearn.log_model(
                pipe, artifact_path="model",
                signature=infer_signature(X_test.head(50), proba[:50]),
                input_example=X_test.head(3),
            )
            duration = round(time.time() - t0, 2)
            log_event("model_training_completed", component="training",
                      run_id=run.info.run_id, duration_s=duration,
                      mlflow_experiment_id=run.info.experiment_id, **metrics)
            results.append({"run_id": run.info.run_id, "config": cfg, "metrics": metrics})
    return results


def run_training(pipeline_run_id: str) -> list[dict]:
    """Entry point used by the Cloud Run Job: read from BigQuery, train, log to MLflow."""
    from src.utils.auth import configure_mlflow_auth
    from src.utils.bigquery import read_table

    s = get_settings()
    configure_mlflow_auth()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    df = read_table(s.training_table_id, s.project_id)
    train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]
    return train_and_log(train_df, test_df, experiment_name=s.experiment_name,
                         pipeline_run_id=pipeline_run_id, seed=s.seed)
