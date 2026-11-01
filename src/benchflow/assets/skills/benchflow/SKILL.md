---
name: benchflow
description: Use BenchFlow to author, validate, run, monitor, diagnose, cancel, or clean up profile-based LLM inference benchmark experiments and inspect their MLflow outputs.
---

# BenchFlow

Use the installed `bflow` CLI as the source of truth. Check `bflow --help` and
the relevant subcommand's `--help` before constructing commands; do not invent
flags from memory.

BenchFlow experiments select deployment, benchmark, and metrics profiles and
resolve them into an internal RunPlan. Prefer a new profile when behavior must
change. Keep experiments small and do not reproduce toolbox behavior in shell
scripts or Tekton definitions.

## Choose the workflow

- For authoring, validating, resolving, or launching an experiment, read
  [references/experiments.md](references/experiments.md).
- For status, logs, failure diagnosis, cancellation, cleanup, or MLflow
  evidence, read [references/operations.md](references/operations.md).

## Operating boundaries

- Read-only inspection and validation are safe defaults.
- Launch, bootstrap, cancellation, cleanup, deletion, and changes to live
  clusters require clear user intent. A request to diagnose does not authorize
  a fix or cleanup.
- Before a live mutation, identify the management cluster, target cluster,
  namespace, experiment or execution name, and exact affected resources.
- Preserve user changes in dirty repositories. Do not edit profiles or
  experiments unrelated to the requested run.
- Treat kubeconfigs, tokens, registry credentials, and MLflow credentials as
  secrets. Never copy them into experiments, profiles, logs, or reports.
- Report unsupported paths plainly instead of presenting partial behavior as
  working support.
