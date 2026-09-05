from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from benchflow.artifacts import _matches_release
from benchflow.llmd_epp import LlmdEppIdentity, resolve_llmd_epp_identity
from benchflow.metrics.prometheus import (
    _promql_regex_escape,
    _resolved_scheduler_pod_regex,
)


RELEASE_NAME = "qwen36-35b-offloading-scalabili-41420c6911"
HELM_RELEASE_NAME = f"gaie-{RELEASE_NAME}"
EPP_DEPLOYMENT_NAME = "gaie-qwen36-35b-offloading-scalabili-414-epp"


@patch("benchflow.llmd_epp.run_json_command")
def test_resolves_helm_truncated_epp_identity(run_json_command) -> None:
    run_json_command.return_value = {
        "items": [
            {
                "metadata": {
                    "name": EPP_DEPLOYMENT_NAME,
                    "annotations": {
                        "meta.helm.sh/release-name": HELM_RELEASE_NAME,
                        "meta.helm.sh/release-namespace": "benchflow",
                    },
                },
                "spec": {
                    "selector": {
                        "matchLabels": {"llm-d-router-gateway": EPP_DEPLOYMENT_NAME}
                    }
                },
            }
        ]
    }

    identity = resolve_llmd_epp_identity("benchflow", RELEASE_NAME, "istio", "oc")

    assert identity.deployment_name == EPP_DEPLOYMENT_NAME
    assert identity.release_aliases == (HELM_RELEASE_NAME, EPP_DEPLOYMENT_NAME)
    assert identity.selectors[0] == (f"llm-d-router-gateway={EPP_DEPLOYMENT_NAME}")


def test_release_matching_includes_helm_resources_and_epp_pods() -> None:
    aliases = (HELM_RELEASE_NAME, EPP_DEPLOYMENT_NAME)

    assert _matches_release(
        {
            "name": "generated-epp-config",
            "annotations": {"meta.helm.sh/release-name": HELM_RELEASE_NAME},
        },
        RELEASE_NAME,
        aliases,
    )
    assert _matches_release(
        {
            "name": f"{EPP_DEPLOYMENT_NAME}-795dcdd785-7ds99",
            "labels": {"llm-d-router-gateway": EPP_DEPLOYMENT_NAME},
        },
        RELEASE_NAME,
        aliases,
    )
    assert not _matches_release(
        {
            "name": "gaie-unrelated-epp-12345",
            "labels": {"llm-d-router-gateway": "gaie-unrelated-epp"},
        },
        RELEASE_NAME,
        aliases,
    )


@patch("benchflow.metrics.prometheus.resolve_llmd_epp_identity")
def test_scheduler_metrics_use_resolved_epp_deployment_name(resolve_identity) -> None:
    resolve_identity.return_value = LlmdEppIdentity(
        helm_release_name=HELM_RELEASE_NAME,
        deployment_name=EPP_DEPLOYMENT_NAME,
        selectors=(f"llm-d-router-gateway={EPP_DEPLOYMENT_NAME}",),
    )
    plan = SimpleNamespace(
        deployment=SimpleNamespace(
            platform="llm-d",
            repo_ref="v0.9.0",
            namespace="benchflow",
            gateway="istio",
        )
    )

    pod_regex = _resolved_scheduler_pod_regex(plan, RELEASE_NAME, "oc")

    assert pod_regex == f"{_promql_regex_escape(EPP_DEPLOYMENT_NAME)}.*"
    resolve_identity.assert_called_once_with("benchflow", RELEASE_NAME, "istio", "oc")
