"""Streamlit dashboard: data overview, experiments, model performance, live predictions."""
import os
import tempfile

import pandas as pd
import requests
import streamlit as st
from mlflow.tracking import MlflowClient

from src.config import get_settings
from src.features.preprocessing import FEATURES, TARGET
from src.utils.auth import auth_headers, configure_mlflow_auth
from src.utils.logging import log_event

st.set_page_config(page_title="MLOps demo - Bank Marketing", layout="wide")


from dashboard.auth import require_login, role_of

email = require_login()
role = role_of(email)
if role is None:
    st.error("Access restricted. Ask an admin to add your e-mail.")
    st.button("Sign out", on_click=st.logout)
    st.stop()
st.sidebar.write(f"{email} ({role})")
st.sidebar.button("Sign out", on_click=st.logout)

names = ["Predictions"] if role == "user" else [
    "Data overview", "Model experiments", "Model performance", "Predictions", "Pipeline admin"]
T = dict(zip(names, st.tabs(names), strict=True))

if "Data overview" in T:
    with T["Data overview"]:
        ...   # existing block
# same pattern for the other existing tabs

with T["Predictions"]:
    ...       # existing form, but send the e-mail to the API:
    # headers={**auth_headers(API_URL), "X-Submitted-By": email}

if "Pipeline admin" in T:
    with T["Pipeline admin"]:
        from dashboard import admin
        step = st.selectbox("Job step", ["extract", "validate", "preprocess", "load",
                                          "train", "evaluate", "register"])
        run_id = st.text_input("Run id", value="manual")
        if st.button("Run this step"):
            st.success(f"Started: {admin.run_job_step(step, run_id)}")
        if st.button("Trigger Airflow DAG"):
            st.json(admin.trigger_dag())
        c1, c2 = st.columns(2)
        if c1.button("Pause DAG"):
            admin.set_dag_paused(True)
        if c2.button("Unpause DAG"):
            admin.set_dag_paused(False)
            

S = get_settings()
API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")

CHOICES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"], "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"], "contact": ["cellular", "telephone"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}


@st.cache_data(ttl=600)
def load_training_data() -> pd.DataFrame:
    from src.utils.bigquery import read_table

    return read_table(S.training_table_id, S.project_id)


def mlflow_client() -> MlflowClient:
    configure_mlflow_auth()
    return MlflowClient(tracking_uri=S.mlflow_tracking_uri)


@st.cache_data(ttl=120)
def load_runs() -> pd.DataFrame:
    client = mlflow_client()
    exp = client.get_experiment_by_name(S.experiment_name)
    if exp is None:
        return pd.DataFrame()
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"],
                              max_results=100)
    return pd.DataFrame([{
        "run": r.info.run_name, "run_id": r.info.run_id[:8],
        "pipeline_run_id": r.data.tags.get("pipeline_run_id"),
        **{k: r.data.params.get(k) for k in ("max_depth", "learning_rate", "n_estimators")},
        **{k: round(v, 4) for k, v in r.data.metrics.items()},
    } for r in runs])


st.title("MLOps demo - Bank Marketing (XGBoost)")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data overview", "Model experiments", "Model performance", "Predictions"])

with tab1:
    try:
        df = load_training_data()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", len(FEATURES))
        c3.metric("Missing values", int(df.isna().sum().sum()))
        c4.metric("Positive rate", f"{df[TARGET].mean():.1%}")
        st.subheader("Target distribution")
        st.bar_chart(df[TARGET].value_counts().rename({0: "no", 1: "yes"}))
        st.subheader("Data quality")
        st.write(f"Duplicate rows: {int(df.drop(columns=['split']).duplicated().sum())}")
        st.write(df["split"].value_counts().rename("rows per split"))
        st.dataframe(df.head(50))
    except Exception as exc:  # noqa: BLE001
        log_event("dashboard_error", severity="ERROR", component="dashboard", tab="data",
                  error=repr(exc))
        st.error(f"Could not load BigQuery data: {exc}")

with tab2:
    try:
        runs = load_runs()
        if runs.empty:
            st.info("No MLflow runs yet. Execute the pipeline first.")
        else:
            st.dataframe(runs, use_container_width=True)
            st.bar_chart(runs.set_index("run")[["roc_auc", "f1", "accuracy"]])
    except Exception as exc:  # noqa: BLE001
        log_event("dashboard_error", severity="ERROR", component="dashboard", tab="experiments",
                  error=repr(exc))
        st.error(f"Could not reach MLflow: {exc}")

with tab3:
    try:
        client = mlflow_client()
        mv = client.get_model_version_by_alias(S.model_name, S.model_alias)
        run = client.get_run(mv.run_id)
        st.write(f"Champion: **{S.model_name} v{mv.version}** (run `{mv.run_id[:8]}`)")
        cols = st.columns(5)
        for col, m in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"], strict=True):
            col.metric(m, f"{run.data.metrics.get(m, float('nan')):.3f}")
        with tempfile.TemporaryDirectory() as tmp:
            img_cols = st.columns(3)
            names = ["confusion_matrix", "roc_curve", "feature_importance"]
            for col, name in zip(img_cols, names, strict=True):
                path = client.download_artifacts(mv.run_id, f"evaluation/{name}.png", tmp)
                col.image(path, caption=name.replace("_", " "))
    except Exception as exc:  # noqa: BLE001
        log_event("dashboard_error", severity="ERROR", component="dashboard", tab="performance",
                  error=repr(exc))
        st.warning(f"No champion model available yet: {exc}")

with tab4:
    st.write("Will this customer subscribe to a term deposit?")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        payload = {}
        for i, (name, opts) in enumerate(CHOICES.items()):
            payload[name] = [c1, c2, c3][i % 3].selectbox(name, opts)
        payload["age"] = c1.number_input("age", 16, 110, 40)
        payload["campaign"] = c2.number_input("campaign", 1, 60, 2)
        payload["pdays"] = c3.number_input("pdays (999 = never)", 0, 999, 999)
        payload["previous"] = c1.number_input("previous", 0, 20, 0)
        payload["emp_var_rate"] = c2.number_input("emp_var_rate", -4.0, 2.0, 1.1)
        payload["cons_price_idx"] = c3.number_input("cons_price_idx", 90.0, 96.0, 93.9)
        payload["cons_conf_idx"] = c1.number_input("cons_conf_idx", -55.0, -25.0, -36.4)
        payload["euribor3m"] = c2.number_input("euribor3m", 0.5, 5.5, 4.8)
        payload["nr_employed"] = c3.number_input("nr_employed", 4900.0, 5300.0, 5191.0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, headers=auth_headers(API_URL),
                              timeout=60)
            r.raise_for_status()
            out = r.json()
            label = "WILL subscribe" if out["prediction"] else "will NOT subscribe"
            st.success(f"Prediction: customer {label} (probability {out['probability']:.1%}, "
                       f"model v{out['model_version']})")
        except Exception as exc:  # noqa: BLE001
            log_event("api_connection_failure", severity="ERROR", component="dashboard",
                      error=repr(exc))
            st.error(f"API call failed: {exc}")
