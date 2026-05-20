from __future__ import annotations

import os
from functools import lru_cache

import requests

from .models import ValidationError

_SERVER_INFO_PATH = "/api/3.0/mlflow/server-info"
_SERVER_INFO_TIMEOUT_SECONDS = 10


def _workspace_name() -> str:
    return str(os.environ.get("MLFLOW_WORKSPACE", "")).strip()


def _workspace_store_uri() -> str | None:
    value = str(os.environ.get("MLFLOW_WORKSPACE_STORE_URI", "")).strip()
    return value or None


def _tracking_uri_value(mlflow_tracking_uri: str | None = None) -> str:
    return str(mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip()


def _tracking_tls_verify() -> bool:
    return (
        str(os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "false")).strip().lower()
        != "true"
    )


def _tracking_auth() -> tuple[str, str] | None:
    username = str(os.environ.get("MLFLOW_TRACKING_USERNAME", "")).strip()
    password = str(os.environ.get("MLFLOW_TRACKING_PASSWORD", "")).strip()
    if not username or not password:
        return None
    return username, password


@lru_cache(maxsize=16)
def _server_supports_workspaces(
    tracking_uri: str,
    username: str,
    password: str,
    verify_tls: bool,
) -> bool:
    if not tracking_uri.startswith(("http://", "https://")):
        return False

    response = requests.get(
        f"{tracking_uri.rstrip('/')}{_SERVER_INFO_PATH}",
        auth=((username, password) if username and password else None),
        timeout=_SERVER_INFO_TIMEOUT_SECONDS,
        verify=verify_tls,
    )
    if response.status_code == 404:
        return False
    if not response.ok:
        raise ValidationError(
            "failed to query MLflow server workspace support: "
            f"{response.status_code} {response.text.strip() or response.reason}"
        )

    payload = response.json() if response.content else {}
    return bool((payload or {}).get("workspaces_enabled"))


def configure_mlflow_tracking(mlflow_tracking_uri: str | None = None) -> str:
    import mlflow

    tracking_uri = _tracking_uri_value(mlflow_tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    workspace = _workspace_name()
    if not workspace:
        return tracking_uri

    if not tracking_uri:
        raise ValidationError(
            "MLFLOW_WORKSPACE requires MLFLOW_TRACKING_URI to point to the MLflow server"
        )
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValidationError(
            "MLFLOW_WORKSPACE requires an HTTP(S) MLflow tracking server"
        )
    if not hasattr(mlflow, "set_workspace"):
        raise ValidationError(
            "MLFLOW_WORKSPACE requires an MLflow client with native workspace support; "
            "install mlflow>=3.10"
        )

    auth = _tracking_auth()
    if not _server_supports_workspaces(
        tracking_uri,
        auth[0] if auth else "",
        auth[1] if auth else "",
        _tracking_tls_verify(),
    ):
        raise ValidationError(
            f"MLFLOW_WORKSPACE={workspace!r} was requested, but the MLflow server at "
            f"{tracking_uri} does not advertise workspace support"
        )

    mlflow.set_workspace(workspace)
    return tracking_uri


def create_mlflow_client(mlflow_tracking_uri: str | None = None):
    import mlflow

    tracking_uri = configure_mlflow_tracking(mlflow_tracking_uri)
    workspace_store_uri = _workspace_store_uri()
    try:
        return mlflow.tracking.MlflowClient(
            tracking_uri=tracking_uri or None,
            workspace_store_uri=workspace_store_uri,
        )
    except TypeError:
        return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri or None)
