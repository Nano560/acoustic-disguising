# Reproducibility

This document describes what the pipeline produces, how to verify a run is
healthy, and the practical steps to reproduce the paper's results.

## What "reproducibility" means here

- **Bit-exact** at the FDTD level: the time-stepping kernels are deterministic
  in float32. With identical inputs (same compiler, same backend, same grid
  size), runs reproduce the same intermediate fields up to backend-specific
  reduction order. The CUDA and Threads backends agree to within a small
  multiple of float32 round-off.
- **Workflow-exact** at the framework level: every parameter that affects
  results lives in `configs/*.toml`. There are no random seeds — the FDTD,
  Fibonacci sampling, MDD inversion, and hologram synthesis are all
  deterministic. Re-running with the same TOML and the same code produces
  the same outputs.
- **Statistically equivalent**, not bit-exact, when the grid resolution or
  the angular sampling is changed: the framework converges to the same
  continuous answer at higher resolution.

## Run directory layout

Every pipeline run is associated with a *run directory* that holds its cfg
and all stage outputs. The cfg lives at `<run_dir>/config.toml`; subdirs
are auto-created by the pipeline scripts:

```
<run_dir>/
├── config.toml      # the cfg snapshot for this run
├── greens/          # Stage 1 outputs — methods 1, 2, and 3b (MDD)
├── reverb/          # Stage 1 outputs — method 3a (reverberant FDTD data)
├── holograms/       # Stage 2 outputs + disguise/cloaking holograms
├── scattering/      # far-field scattering acquisition outputs
└── figures/         # matplotlib/Makie diagnostic PNGs
```

Different runs (different `f_3db_hz`, different scatterers, etc.) live in
*separate* run directories — they cannot mix or overwrite each other,
even when the cfg parameters change.

## Pipeline stages and outputs

The pipeline branches. Stage 1 retrieves Green's functions — one of three
interchangeable methods — and three consumers then read those Green's
functions independently:

```
config.toml
    │   STAGE 1 — Green's-function retrieval  (pick ONE method)
    ├──▶ analytical.jl ................. greens/analytical_none.h5     (method 1)
    ├──▶ impulsive.jl .................. greens/impulsive_<sc>.h5      (method 2)
    └──▶ reverb.jl ─▶ mdd_extract.py ... greens/mdd_extracted_<sc>.h5  (method 3)
                                                 │
          the Green's functions feed three consumers
                                                 │
        ┌────────────────────────┬───────────────┴───────────────────┐
        ▼                        ▼                                   ▼
   STAGE 2                   scattering/acquire.jl           disguise/synthesize.jl
   hologram/synthesize.jl    scattering/scattering_*.h5      holograms/..._cloak_*.h5
   holograms/hologram_*.h5   (far-field scattering)          (cloaking / disguising)
```

| Stage | Script | Output (HDF5) | Approximate size at `paper.toml` |
|-------|--------|---------------|----------------------------------|
| 1 · method 1 (analytical) | `scripts/greens/analytical.jl`   | `<run_dir>/greens/analytical_none.h5`           | ~50 MB (scatterer=none only) |
| 1 · method 2 (impulsive)  | `scripts/greens/impulsive.jl`    | `<run_dir>/greens/impulsive_<scatterer>.h5`     | ~50–200 MB / scatterer |
| 1 · method 3a (reverb)    | `scripts/greens/reverb.jl`       | `<run_dir>/reverb/reverb_<scatterer>.h5`        | ~1–4 GB / scatterer |
| 1 · method 3b (MDD)       | `scripts/greens/mdd_extract.py`  | `<run_dir>/greens/mdd_extracted_<scatterer>.h5` | ~50–200 MB / scatterer |
| 2 · hologram synthesis    | `scripts/hologram/synthesize.jl` | `<run_dir>/holograms/hologram_<scatterer>_<gf_label>.h5`           | ~4–5 GB / hologram † |
| + scattering acquisition  | `scripts/scattering/acquire.jl`  | `<run_dir>/scattering/scattering_<scatterer>_<gf_label>.h5`        | ~15–20 MB / pair |
| + disguise synthesis      | `scripts/disguise/synthesize.jl` | `<run_dir>/holograms/hologram_<real>_cloak_<target>_<gf_label>.h5` | ~4–5 GB / hologram † |

† Hologram size is dominated by the wavefield-snapshot tensors written when
`[hologram].snapshot_every > 0` (the figure pipeline needs them). With
snapshots disabled a hologram is ~10–100 MB.

`<scatterer>` is one of `none`, `sphere`, `cube`, `cross` (none = no scatterer,
homogeneous-medium baseline). `<gf_label>` is `real` (bare FDTD, no GF)
when `--scatterer-mode=real` was used, or one of `analytical`, `impulsive`,
`mdd` (ordered by increasing cost) when `--scatterer-mode=hologram` was used.
`scripts/run_reverb_mdd.sh` covers Stage 1 method 3 + Stage 2 and defaults to
`--gf-method=mdd`; scattering and disguise are run separately (see the
README's *Reproducing the paper* section).

## Verifying a quickstart run

```bash
RUN=/tmp/3ddisguising-quickstart
mkdir -p "$RUN"
cp configs/quickstart.toml "$RUN/config.toml"
bash scripts/run_reverb_mdd.sh "$RUN"
```

`run_reverb_mdd.sh` runs the reverb-MDD-hologram path. After it finishes, the
run directory should look like:

```
$RUN/
├── config.toml
├── greens/
│   └── mdd_extracted_none.h5    ~5 MB
├── reverb/
│   └── reverb_none.h5           ~30 MB
├── holograms/
│   └── hologram_none_mdd.h5     ~5 MB
└── figures/                     diagnostic plots from the run
```

Two minimal checks:

- HDF5 files open without errors:
  ```bash
  python -c "import h5py, glob; [print(f, list(h5py.File(f).keys())) \
      for f in sorted(glob.glob('$RUN/**/*.h5', recursive=True))]"
  ```
- The Julia and Python test suites pass on your machine:
  ```bash
  julia --project=. -e 'using Pkg; Pkg.test()'
  pytest python/mdd/tests
  ```

## Verifying a paper-scale run

`paper.toml` runs are large (the paper's took ~1 week on CPU). Once stage 1
finishes, `diagnostics/check_gf_scale.jl` can verify the amplitude scaling of the
extracted Green's functions.

The sanity battery shipped with the archive:

- **Reciprocity** (Julia test): G(rₐ, r_b, t) ≈ G(r_b, rₐ, t) on a tiny
  homogeneous domain. Run `Pkg.test()`.
- **MDD round-trip** (Python test): writing and re-reading the MDD HDF5
  schema preserves the data exactly. Run `pytest python/mdd/tests`.
- **GF / MDD amplitude scale factor** (`diagnostics/check_gf_scale.jl`): linear-fits
  the optimal Green's-function amplitude scale factor via Kirchhoff-Helmholtz
  cancellation. Run with `--gf-method=analytical|impulsive|mdd`; a fitted
  factor near the expected value confirms the GF (or MDD) amplitude scaling.

## Hardware used for the paper

Every result in the paper was produced on CPU — the `Threads` backend — on a
single workstation. No GPU was used.

| Component | Value |
|-----------|-------|
| CPU       | Apple M1 Pro (10 cores) |
| RAM       | 32 GB |
| Backend   | `Threads` (CPU) |
| Julia     | 1.12.6 |
| Python    | 3.11.7 |
| OS        | macOS 26.5 |
| Wall time, full pipeline | ~1 week (all four scatterers, all stages) |

## Re-running on a different machine

The paper-scale pipeline was produced on the CPU (`Threads`) backend — see
the hardware table above. For reference:

- 10-core Apple M1 Pro, Threads backend: ~1 week for the full pipeline.
- A CUDA GPU (`ACOUSTIC_DISGUISING_BACKEND=CUDA`) is supported and substantially
  faster per FDTD step, but was not used for the paper, so no measured wall
  time is given.

If you only need MDD-extracted Green's functions and you already have the
reverb outputs (Stage 1 method 3a) in `<run_dir>/reverb/`, you can skip
straight to:

```bash
python scripts/greens/mdd_extract.py /path/to/run_dir/
julia --project=. --threads=10 scripts/hologram/synthesize.jl /path/to/run_dir/ --gf-method=mdd
```

## Known sources of small numerical drift

- **Backend differences**: CUDA reductions and CPU OpenMP reductions sum
  in different orders. Expect deviations at ~1e-6 relative for individual
  field values; integral quantities (energy, GF norms) agree to ~1e-7.
- **HDF5 compression**: lossless; does not affect bit-exactness.
- **Julia version**: pinned via `Manifest.toml`. Using a different Julia
  may slightly change BLAS behaviour but should not affect the FDTD result.
- **Pylops version**: pinned in `pyproject.toml`. The MDD inversion uses
  damped LSQR with a fixed iteration limit, so the result is fully
  determined by the input + parameters.
