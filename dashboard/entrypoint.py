"""
Container entrypoint — fetches the Streamlit secrets from GCP Secret Manager,
writes them to /app/.streamlit/secrets.toml, then exec's Streamlit.

Required env vars
-----------------
GOOGLE_CLOUD_PROJECT   : GCP project id (set automatically on Cloud Run)
AUTH_SECRET_NAME       : Secret Manager secret name  (default: dashboard-auth-secrets)
PORT                   : HTTP port                   (default: 8080, set by Cloud Run)
"""
import os
import pathlib
import subprocess
import sys


def _fetch_secret() -> str:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    secret  = os.environ.get("AUTH_SECRET_NAME", "dashboard-auth-secrets")
    # Import here so the library is only required at container start-up
    from google.cloud import secretmanager  # noqa: PLC0415

    client  = secretmanager.SecretManagerServiceClient()
    name    = f"projects/{project}/secrets/{secret}/versions/latest"
    payload = client.access_secret_version(request={"name": name}).payload.data.decode()
    return payload


def main() -> None:
    secrets_dir  = pathlib.Path("/app/.streamlit")
    secrets_file = secrets_dir / "secrets.toml"

    secrets_dir.mkdir(parents=True, exist_ok=True)

    try:
        toml_content = _fetch_secret()
        secrets_file.write_text(toml_content, encoding="ascii")
        print(
            f"[entrypoint] secrets.toml written from "
            f"{os.environ.get('AUTH_SECRET_NAME', 'dashboard-auth-secrets')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        # Fail fast — Streamlit would crash anyway without valid auth config
        print(f"[entrypoint] FATAL: could not fetch auth secret: {exc}", file=sys.stderr)
        sys.exit(1)

    port = os.environ.get("PORT", "8080")
    cmd  = [
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]
    os.execv(sys.executable, cmd)  # replace this process — no orphan PIDs


if __name__ == "__main__":
    main()
