"""Authentication helpers for service-to-service calls on Cloud Run.

MLflow and the prediction API are deployed as *private* Cloud Run services, so callers
must send a Google-signed ID token whose audience is the service URL."""
import os
import subprocess


def get_id_token(audience: str) -> str:
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        return google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), audience
        )
    except Exception:
        # Local development with user credentials: fall back to the gcloud CLI.
        out = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()


def auth_headers(url: str) -> dict:
    if url.startswith("https://"):
        return {"Authorization": f"Bearer {get_id_token(url)}"}
    return {}


def configure_mlflow_auth() -> None:
    """Expose an ID token to the MLflow client when tracking against a private Cloud Run URL."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri.startswith("https://"):
        os.environ["MLFLOW_TRACKING_TOKEN"] = get_id_token(uri)
