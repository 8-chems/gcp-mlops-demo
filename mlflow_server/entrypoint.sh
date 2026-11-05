#!/bin/sh
set -e
# Required env: DB_USER, DB_PASSWORD, DB_NAME, CLOUDSQL_CONNECTION_NAME, ARTIFACT_ROOT (gs://...)
BACKEND="postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@/${DB_NAME}?host=/cloudsql/${CLOUDSQL_CONNECTION_NAME}"
exec mlflow server \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  --backend-store-uri "${BACKEND}" \
  --serve-artifacts \
  --artifacts-destination "${ARTIFACT_ROOT}"
