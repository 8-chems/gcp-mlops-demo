import pandas as pd
from google.cloud import bigquery


def load_dataframe(df: pd.DataFrame, table_id: str, project: str) -> int:
    """Replace the content of `table_id` with `df` (WRITE_TRUNCATE)."""
    client = bigquery.Client(project=project)
    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    return client.get_table(table_id).num_rows


def read_table(table_id: str, project: str) -> pd.DataFrame:
    client = bigquery.Client(project=project)
    return client.query(f"SELECT * FROM `{table_id}`").to_dataframe()


def insert_rows(table_id: str, rows: list[dict], project: str) -> list:
    client = bigquery.Client(project=project)
    return client.insert_rows_json(table_id, rows)
