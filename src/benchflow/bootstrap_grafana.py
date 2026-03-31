from __future__ import annotations

import secrets
import time
from typing import Any

from .cluster import CommandError
from .ui import detail, step, success


def install_grafana_if_needed(installer: Any) -> None:
    if not installer.options.install_grafana:
        detail(
            "Skipping Grafana install because install_grafana=false for this bootstrap run"
        )
        return
    success(
        f"Grafana will be installed directly in namespace {installer.grafana_namespace}"
    )


def discover_grafana_route_host(installer: Any) -> str | None:
    if not installer._resource_exists(
        "get", "route", "-n", installer.grafana_namespace
    ):
        return None
    routes = installer._oc_json(
        "get",
        "route",
        "-n",
        installer.grafana_namespace,
        retry=True,
        description="reading routes",
    )
    for item in routes.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        host = item.get("spec", {}).get("host", "")
        if "grafana" in name and host:
            return str(host)
    return None


def wait_for_grafana_route(installer: Any, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        host = discover_grafana_route_host(installer)
        if host:
            return
        time.sleep(5)
    raise CommandError("timed out waiting for the Grafana route")


def wait_for_grafana_ready(installer: Any, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not installer._resource_exists(
            "get", "deployment", "grafana", "-n", installer.grafana_namespace
        ):
            time.sleep(5)
            continue
        deployment = installer._oc_json(
            "get",
            "deployment",
            "grafana",
            "-n",
            installer.grafana_namespace,
            retry=True,
            description="reading deployment/grafana",
        )
        status = deployment.get("status", {}) or {}
        ready = int(status.get("readyReplicas", 0) or 0)
        desired = int(status.get("replicas", 0) or 0)
        if desired > 0 and ready >= desired:
            return
        time.sleep(5)
    raise CommandError("timed out waiting for deployment/grafana to become ready")


def apply_grafana_stack(installer: Any) -> None:
    if not installer.options.install_grafana:
        return

    step("Applying Grafana monitoring RBAC")
    installer._apply_asset_documents(
        "operators/grafana/rbac.yaml",
        namespace=None,
        description="applying Grafana service account resources",
        variables=installer._base_asset_variables(),
    )
    installer._wait_for_secret_key(
        name=installer.grafana_datasource_token_secret,
        key="token",
        namespace=installer.grafana_namespace,
        timeout_seconds=300,
    )
    if not installer._resource_exists(
        "get",
        "secret",
        installer.grafana_admin_secret_name,
        "-n",
        installer.grafana_namespace,
    ):
        installer._apply_asset_documents(
            "operators/grafana/admin-secret.yaml",
            namespace=installer.grafana_namespace,
            description="applying Grafana admin credentials",
            variables={
                **installer._base_asset_variables(),
                "GRAFANA_ADMIN_USER": "admin",
                "GRAFANA_ADMIN_PASSWORD": secrets.token_urlsafe(24),
            },
        )

    step("Applying Grafana deployment, route, datasource, and dashboards")
    grafana_stack_variables = {
        **installer._base_asset_variables(),
        "GRAFANA_DATASOURCES_YAML": installer._asset_text(
            "operators/grafana/datasources.yaml"
        ),
        "GRAFANA_DASHBOARDS_YAML": installer._asset_text(
            "operators/grafana/dashboard-providers.yaml"
        ),
        "GRAFANA_LIVE_DASHBOARD_JSON": installer._asset_text(
            "operators/grafana/benchflow-live-dashboard.json"
        ),
    }
    installer._apply_documents(
        [
            *installer._render_asset_documents(
                "operators/grafana/provisioning-configmap.yaml",
                grafana_stack_variables,
            ),
            *installer._render_asset_documents(
                "operators/grafana/dashboards-configmap.yaml",
                grafana_stack_variables,
            ),
            *installer._render_asset_documents(
                "operators/grafana/deployment.yaml",
                grafana_stack_variables,
            ),
            *installer._render_asset_documents(
                "operators/grafana/service.yaml",
                grafana_stack_variables,
            ),
            *installer._render_asset_documents(
                "operators/grafana/route.yaml",
                grafana_stack_variables,
            ),
        ],
        namespace=installer.grafana_namespace,
        description="applying Grafana stack",
    )
    step("Waiting for Grafana route")
    wait_for_grafana_route(installer, timeout_seconds=600)
    step("Waiting for Grafana to become ready")
    wait_for_grafana_ready(installer, timeout_seconds=600)
