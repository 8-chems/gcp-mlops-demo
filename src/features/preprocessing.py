"""Cleaning and feature definitions for the UCI Bank Marketing dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC = [
    "age", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "target"
RAW_TARGET = "y"
# `duration` is only known after the call ends: dropped to avoid target leakage.
LEAKY_COLUMNS = ["duration"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(".", "_") for c in df.columns]
    return df


def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC:
        df[col] = df[col].astype("float64")
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names, drop duplicates and leaky columns, encode target as 0/1."""
    df = normalize_columns(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    if TARGET not in df.columns:
        df[TARGET] = (df[RAW_TARGET].astype(str).str.lower() == "yes").astype(int)
    df = ensure_types(df[FEATURES + [TARGET]])
    df[TARGET] = df[TARGET].astype(int)
    return df


def add_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Stratified, reproducible 80/20 split stored in a `split` column."""
    train_idx, _ = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    out = df.copy()
    out["split"] = "test"
    out.loc[train_idx, "split"] = "train"
    return out
