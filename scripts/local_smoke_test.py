"""Run clean -> split -> train -> evaluate -> register locally (SQLite MLflow, no GCP)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.data.ingestion import read_raw_csv, sample_rows  # noqa: E402
from src.data.synthetic import make_synthetic_bank  # noqa: E402
from src.data.validation import assert_valid, validate  # noqa: E402
from src.features.preprocessing import add_split, clean  # noqa: E402
from src.models.evaluate import evaluate, register  # noqa: E402
from src.models.train import train_and_log  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="mlops-smoke-"))
os.chdir(work)
mlflow.set_tracking_uri(f"sqlite:///{work}/mlflow.db")

csv = Path(__file__).resolve().parent.parent / "data" / "bank-additional-full.csv"
if csv.exists():
    print("Using real UCI data")
    raw = sample_rows(read_raw_csv(csv), 15000)
else:
    print("data/bank-additional-full.csv not found -> synthetic data (run `make data` for UCI)")
    raw = make_synthetic_bank(3000)

assert_valid(validate(raw))
df = add_split(clean(raw))
res = train_and_log(df[df.split == "train"], df[df.split == "test"], pipeline_run_id="smoke")
for r in res:
    print(r["config"]["name"], {k: round(v, 4) for k, v in r["metrics"].items()})

s = get_settings()
client = MlflowClient()
print("evaluate:", evaluate(client, s, "smoke"))
print("register:", register(client, s, "smoke"))
print(f"OK - MLflow data in {work} (browse: mlflow ui --backend-store-uri sqlite:///{work}/mlflow.db)")
