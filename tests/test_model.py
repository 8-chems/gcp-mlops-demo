import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient

from src.config import get_settings
from src.features.preprocessing import FEATURES
from src.models.evaluate import ThresholdNotMet, evaluate, register
from src.models.predict import predict_proba
from src.models.train import train_and_log

TINY = [
    {"name": "a", "max_depth": 2, "learning_rate": 0.3, "n_estimators": 10},
    {"name": "b", "max_depth": 3, "learning_rate": 0.3, "n_estimators": 20},
]


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    return MlflowClient()


def _train(split_df, run_id="t1"):
    return train_and_log(split_df[split_df.split == "train"], split_df[split_df.split == "test"],
                         configs=TINY, experiment_name="test-exp", pipeline_run_id=run_id)


def test_training_logs_runs_metrics_and_model(split_df, tracking):
    results = _train(split_df)
    assert len(results) == 2
    run = tracking.get_run(results[0]["run_id"])
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert m in run.data.metrics
    model = mlflow.sklearn.load_model(f"runs:/{results[0]['run_id']}/model")
    proba = model.predict_proba(split_df[FEATURES].head(5))[:, 1]
    assert len(proba) == 5 and ((proba >= 0) & (proba <= 1)).all()
    arts = {a.path for a in tracking.list_artifacts(results[0]["run_id"], "evaluation")}
    assert "evaluation/confusion_matrix.png" in arts


def test_evaluate_threshold_and_register_promotion(split_df, tracking, monkeypatch):
    _train(split_df, "t1")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "test-exp")
    monkeypatch.setenv("MIN_ROC_AUC", "0.99")
    with pytest.raises(ThresholdNotMet):
        evaluate(tracking, get_settings(), "t1")
    monkeypatch.setenv("MIN_ROC_AUC", "0.5")
    s = get_settings()
    evaluate(tracking, s, "t1")
    info = register(tracking, s, "t1")
    assert info["promoted"] and info["version"] == "1"
    champ = tracking.get_model_version_by_alias(s.model_name, s.model_alias)
    assert str(champ.version) == "1"


def test_predict_proba_helper(split_df, tracking):
    res = _train(split_df)
    model = mlflow.sklearn.load_model(f"runs:/{res[0]['run_id']}/model")
    rec = split_df[FEATURES].head(2).to_dict("records")
    assert len(predict_proba(model, rec)) == 2
