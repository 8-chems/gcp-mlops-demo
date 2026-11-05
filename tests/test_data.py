import pandas as pd
import pytest

from src.data.ingestion import sample_rows
from src.data.validation import DataValidationError, assert_valid, validate
from src.features.preprocessing import FEATURES, TARGET, add_split, clean


def test_validation_ok(raw_df):
    report = validate(raw_df)
    assert report["ok"] and report["rows"] == len(raw_df)


def test_validation_detects_missing_columns(raw_df):
    report = validate(raw_df.drop(columns=["job", "y"]))
    assert not report["ok"]
    assert {"job", "y"} <= set(report["missing_required_columns"])
    with pytest.raises(DataValidationError):
        assert_valid(report)


def test_validation_counts_duplicates_and_missing(raw_df):
    df = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    df.loc[100, "age"] = None
    report = validate(df)
    assert report["duplicate_rows"] >= 5 and report["missing_values"]["age"] == 1


def test_clean_output_schema(raw_df):
    out = clean(pd.concat([raw_df, raw_df.head(10)], ignore_index=True))
    assert list(out.columns) == FEATURES + [TARGET]
    assert "duration" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}
    assert len(out) == len(raw_df)  # duplicates removed


def test_split_is_reproducible_and_stratified(raw_df):
    df = clean(raw_df)
    a, b = add_split(df, seed=7), add_split(df, seed=7)
    assert a["split"].equals(b["split"])
    assert abs((a["split"] == "test").mean() - 0.2) < 0.01
    rates = a.groupby("split")[TARGET].mean()
    assert abs(rates["train"] - rates["test"]) < 0.02


def test_sample_rows(raw_df):
    assert len(sample_rows(raw_df, 100, seed=1)) == 100
    assert len(sample_rows(raw_df, 10**6)) == len(raw_df)
