from __future__ import annotations

import json
from typing import Any

from ..models import ResolvedRunPlan


def render_pipelinerun(
    plan: ResolvedRunPlan, pipeline_name: str = "benchflow-e2e"
) -> dict[str, Any]:
    run_plan_json = json.dumps(plan.to_dict(), separators=(",", ":"), sort_keys=True)

    return {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {
            "generateName": f"{plan.metadata.name}-",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": plan.metadata.name,
                "benchflow.io/platform": plan.deployment.platform,
                "benchflow.io/mode": plan.deployment.mode,
            },
        },
        "spec": {
            "pipelineRef": {"name": pipeline_name},
            "taskRunTemplate": {"serviceAccountName": plan.service_account},
            "ttlSecondsAfterFinished": plan.ttl_seconds_after_finished,
            "params": [
                {"name": "RUN_PLAN", "value": run_plan_json},
            ],
            "workspaces": [
                {
                    "name": "results",
                    "persistentVolumeClaim": {"claimName": "benchmark-results"},
                },
                {
                    "name": "models-storage",
                    "persistentVolumeClaim": {
                        "claimName": plan.deployment.model_storage.pvc_name
                    },
                },
                {"name": "source", "emptyDir": {}},
            ],
        },
    }
