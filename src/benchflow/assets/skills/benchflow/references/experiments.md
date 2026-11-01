# Experiments

## Discover the repository contract

1. Locate the repository root and inspect its `AGENTS.md` or equivalent local
   instructions.
2. Use `bflow profiles list` and `bflow profiles show` to discover profiles.
   If the repository uses a non-default profile tree, pass the documented
   `--profiles-dir` option rather than moving or duplicating profiles.
3. Inspect a nearby supported experiment before creating a new one.

An experiment normally contains a model and references to one deployment, one
benchmark, and one metrics profile. Matrix-valued references or overrides
produce the Cartesian product. Put reusable runtime or benchmark behavior in a
profile, not an inline experiment override.

## Validate before execution

Use the CLI stages in increasing order of commitment:

```bash
bflow experiment validate <experiment.yaml>
bflow experiment resolve <experiment.yaml> --format yaml
bflow experiment render-deployment <experiment.yaml> --output-dir <directory>
```

Consult each command's `--help` for repository-root and profile-directory
options. Inspect resolved replica counts, GPU requirements, placement,
benchmark load, target cluster, and cleanup behavior before submission.

## Launch

Run only after the user has authorized execution and supplied or confirmed the
target cluster and image:

```bash
bflow experiment run <experiment.yaml> \
  --cluster-name <target-cluster> \
  --benchflow-image <image>
```

Use `--follow` only when the user wants the local process to stay attached.
Interrupting a local follow normally stops watching; verify live state before
assuming the remote execution was canceled.

Capture the returned PipelineRun or matrix-parent name. It is the stable handle
for subsequent status, logs, cancellation, and result lookup.
