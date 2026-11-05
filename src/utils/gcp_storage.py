import io
import json

import pandas as pd
from google.cloud import storage


def parse_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    bucket, _, blob = uri[5:].partition("/")
    return bucket, blob


def read_csv(uri: str, **kwargs) -> pd.DataFrame:
    bucket, blob = parse_uri(uri)
    data = storage.Client().bucket(bucket).blob(blob).download_as_bytes()
    return pd.read_csv(io.BytesIO(data), **kwargs)


def write_csv(df: pd.DataFrame, uri: str) -> None:
    bucket, blob = parse_uri(uri)
    storage.Client().bucket(bucket).blob(blob).upload_from_string(
        df.to_csv(index=False), content_type="text/csv"
    )


def write_json(obj: dict, uri: str) -> None:
    bucket, blob = parse_uri(uri)
    storage.Client().bucket(bucket).blob(blob).upload_from_string(
        json.dumps(obj, indent=2, default=str), content_type="application/json"
    )
