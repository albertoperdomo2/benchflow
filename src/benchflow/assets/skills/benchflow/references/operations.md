# Operations

## Inspect and diagnose

Start with BenchFlow's own surface and then use cluster-native evidence:

```bash
bflow experiment list --namespace benchflow
bflow logs <execution-name> --namespace benchflow --all
```

Use the relevant command's `--help`; select a single failed step when full logs
are noisy. Correlate the management-cluster PipelineRun and TaskRuns with the
target-cluster Jobs, serving resources, Pods, events, scheduling state, and
endpoint readiness. Separate the primary failure from downstream collection,
upload, or cleanup failures.

For matrix runs, distinguish the parent supervisor from child executions.
Count succeeded, failed, running, and queued children rather than inferring the
matrix state from the parent alone.

## Mutating recovery

Do not cancel, delete, clean up, patch, or relaunch resources unless the user
has authorized that action. Resolve exact names first and avoid broad label
selectors when several matrices or users share a namespace.

Use BenchFlow commands when they cover the operation:

```bash
bflow experiment cancel <execution-name>
bflow experiment cleanup <experiment.yaml>
```

Confirm exact syntax with `--help`. After cancellation or cleanup, verify both
management-cluster orchestration state and target-cluster resources.

## MLflow evidence

BenchFlow treats benchmark results, reports, metrics, logs, and rendered
manifests as first-class outputs. Prefer the pre-authorized `mlflow` CLI for
read-only run and artifact inspection. Ask before deleting, renaming, or
otherwise mutating MLflow runs, experiments, or artifacts.

When reporting a failure or comparison, include the run link or run ID, profile
names, resolved model/runtime shape, benchmark load, validity caveats, and the
specific artifacts or metrics supporting the conclusion.
