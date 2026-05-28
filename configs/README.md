# Cfg templates

These files are **templates**, not live configs. No script reads from
`configs/` directly — every pipeline script takes a `<run_dir>` positional
argument and reads `<run_dir>/config.toml` exclusively.

To start a new run, copy a template into a fresh run directory:

```bash
RUN=/path/to/runs/<name>
mkdir -p "$RUN"
cp configs/quickstart.toml "$RUN/config.toml"   # then edit "$RUN/config.toml"
bash scripts/run_reverb_mdd.sh "$RUN"
```

That copy is the run's source of truth from then on; editing
`configs/quickstart.toml` afterwards has no effect on the run.

## Files

| File | Purpose |
|---|---|
| `quickstart.toml` | CPU-runnable smoke-test cfg (n=51, ~3 minutes). Used in [`INSTALL.md`](../INSTALL.md) and [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) to verify the pipeline runs end-to-end on a fresh clone. Not for physically meaningful results. |

## Why no `paper.toml` here

The actual paper-scale cfg evolves with the experimental setup and lives
in the run-dir on local data storage, not in the working repo (the in-repo
template would drift from the cfg that produced the published figures).

At release time, `scripts/prepare_release.sh` copies the run-dir
`config.toml` of the canonical paper run into the public archive as
`configs/paper.toml` (pass its path with `--paper-config=<path>`). The
released archive is therefore the only place where a `configs/paper.toml`
exists, and by construction it matches the figures in the paper.
