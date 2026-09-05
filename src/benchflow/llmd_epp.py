from __future__ import annotations

from dataclasses import dataclass

from .cluster import CommandError, run_json_command


@dataclass(frozen=True)
class LlmdEppIdentity:
    helm_release_name: str
    deployment_name: str
    selectors: tuple[str, ...]

    @property
    def release_aliases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.helm_release_name, self.deployment_name)))


def _chart_label_key(gateway_mode: str) -> str:
    return (
        "llm-d-router-standalone"
        if gateway_mode == "standalone"
        else "llm-d-router-gateway"
    )


def _fallback_selectors(release_name: str, gateway_mode: str) -> tuple[str, ...]:
    epp_name = f"gaie-{release_name}-epp"
    selectors = [f"{_chart_label_key(gateway_mode)}={epp_name}"]
    for label_key in ("llm-d-router-gateway", "llm-d-router-standalone"):
        selector = f"{label_key}={epp_name}"
        if selector not in selectors:
            selectors.append(selector)
    return tuple(selectors)


def resolve_llmd_epp_identity(
    namespace: str,
    release_name: str,
    gateway_mode: str,
    kubectl_cmd: str,
) -> LlmdEppIdentity:
    """Resolve the Helm-truncated EPP Deployment and its pod selectors."""
    helm_release_name = f"gaie-{release_name}"
    fallback_selectors = _fallback_selectors(release_name, gateway_mode)
    fallback = LlmdEppIdentity(
        helm_release_name=helm_release_name,
        deployment_name=f"{helm_release_name}-epp",
        selectors=fallback_selectors,
    )
    try:
        payload = run_json_command(
            [kubectl_cmd, "get", "deployments", "-n", namespace, "-o", "json"]
        )
    except CommandError:
        return fallback

    for deployment in payload.get("items", []):
        metadata = deployment.get("metadata") or {}
        annotations = metadata.get("annotations") or {}
        if (
            annotations.get("meta.helm.sh/release-name") != helm_release_name
            or annotations.get("meta.helm.sh/release-namespace") != namespace
        ):
            continue
        deployment_name = str(metadata.get("name") or "").strip()
        if not deployment_name:
            continue
        match_labels = (
            deployment.get("spec", {}).get("selector", {}).get("matchLabels") or {}
        )
        selector_parts = []
        if isinstance(match_labels, dict):
            selector_parts = [
                f"{key}={value}"
                for key, value in match_labels.items()
                if isinstance(key, str) and isinstance(value, (str, int, float, bool))
            ]
        selectors = (
            tuple([",".join(selector_parts), *fallback_selectors])
            if selector_parts
            else fallback_selectors
        )
        return LlmdEppIdentity(
            helm_release_name=helm_release_name,
            deployment_name=deployment_name,
            selectors=selectors,
        )
    return fallback
