# BenchFlow Working Principles

BenchFlow should stay narrow, explicit, and operationally clear. If a change makes the system more flexible but harder to understand, it is probably the wrong change.

## Product Principles

- Prefer one blessed path over many configurable paths.
- Prefer profiles over per-run overrides.
- Prefer removing options over adding knobs.
- Prefer a real working slice over a broad scaffold.
- If something does not work, say so plainly.
- Do not present partial support as real support.

## Architecture Principles

- The user-facing entrypoint is `bflow`.
- The install path is `bflow install`.
- The default namespace is `benchflow`, unless explicitly overridden.
- Tekton is the execution engine, not the place for business logic.
- Python owns the control logic.
- Tekton YAML should stay orchestration-focused and thin.
- The internal contract is the `RunPlan`.
- Tasks and internal commands should consume the run plan, not a wide flat list of parameters.
- Keep one real implementation path. Do not keep duplicate paths alive.

## Configuration Principles

- Experiments should stay small.
- Deployment, benchmark, and metrics behavior should come from profiles.
- Users should not override profile internals inline.
- If a new scenario is needed, create a new profile.
- Keep directory structure flat when extra nesting adds no value.
- Experiments should always have a CLI equivalent.

## Installation Principles

- Installation should be as close to one-click as the cluster allows.
- Tekton and Grafana are part of the default install story.
- MLflow is part of the product, not an optional afterthought.
- Do not make users choose between multiple install modes unless there is a strong reason.
- Prefer one clear cluster model over trying to support every environment shape.

## Benchmarking Principles

- Benchmarking, report generation, metrics, logs, and manifests are all part of the same workflow.
- Artifacts, metrics and logs are first-class outputs, not debugging extras.
- The benchmark runtime should live inside the main package structure.
- Dependency choices must preserve the working benchmark runtime inside the shipped container image.

## Repo Hygiene

- Keep the README concise, operational, and current.
- Keep examples only for supported flows.
- Remove dead code when it stops earning its keep.
- If a future feature is not implemented, leave an explicit Python placeholder rather than a misleading example or half-wired path.
- Avoid duplicate directory trees that express the same concept twice.

## Working Style

- Read the repository before proposing architecture.
- Make the narrow path work first.
- Simplify brittle code before extending it.
- Question changes that add knobs, alternate modes, or extra layers.
- Ask when ambiguity is real, but do not stop for avoidable uncertainty.
- Do not call something done until it has been validated at the right level.

## Definition of Done

A change is in good shape when:

- there is one clear way to use it
- the code path is real
- unsupported paths are explicit
- the docs match the implementation
- the packaging and build story are coherent
- there is no stale duplicate structure left behind

When in doubt, choose the simpler shape.
