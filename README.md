# Acoustic Disguising — Green's function retrieval and hologram synthesis

Companion code to:

> **Acoustic disguising: a unified framework for cloaking and holography**
> Jonas Müller and Dirk-Jan van Manen, 2026.
> DOI: [TODO](https://doi.org/TODO) <!-- TODO: fill in after acceptance -->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) <!-- TODO: fill in after first Zenodo release -->

This repository reproduces every numerical result and figure in the
accompanying paper. Acoustic disguising has two stages: first the Green's
functions of the cloaked region are obtained, then they are used to
synthesise an acoustic hologram. Three interchangeable methods for the
Green's-function retrieval are implemented:

1. **Analytical** — closed-form free-space monopole/dipole evaluation
   (homogeneous medium only; used as a reference).
2. **Impulsive** — FDTD simulation with an impulsive point source.
3. **Reverberant + MDD** — FDTD simulation with a distributed reverberant
   source, followed by Multi-Dimensional Deconvolution to extract the GFs.

The three GF methods are listed in increasing computational cost.

Hologram synthesis follows the De Hoop / Fokkema–van den Berg source-term
convention (`q` = volume-injection-rate density in s⁻¹, `f` = body-force
density in N/m³) with explicit FDTD injection coefficients `C_f, C_q`
implemented in `C_for_N`
([`diagnostics/_check_inject_kh/injection_coefficients.jl`](diagnostics/_check_inject_kh/injection_coefficients.jl)).
The impulsive-GF hologram path is traced step-by-step in
[`docs/impulsive_gf_hologram_path.md`](docs/impulsive_gf_hologram_path.md).

For installation, see [`INSTALL.md`](INSTALL.md). For the hardware and
toolchain table, wall-time estimates, and end-to-end reproduction
instructions, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository layout

```
acoustic-disguising/
├── src/                          Julia package `AcousticDisguising`
│   ├── AcousticDisguising.jl          module entry-point + exports
│   ├── simulation.jl               Domain, CPML, transceivers, run_fdtd!
│   ├── kernels/                    FDTD kernels (CPML coeffs, derivatives,
│   │                               updates, extrapolation, forward step)
│   ├── geometry.jl                 scatterer shapes + masks
│   ├── illumination.jl             Fibonacci/r2 sphere sampling, wavelets
│   ├── greens.jl                   `impulsive_gfs`
│   ├── reverb.jl                   `build_reverb_data`
│   ├── hologram.jl                 `build_hologram`
│   ├── io.jl                       config, HDF5 I/O, Green's-function loader
│   ├── state.jl                    sidecar HDF5 checkpoint for reverb
│   ├── eta.jl                      ETA benchmark used by the long stages
│   ├── plots.jl                    shared CairoMakie helpers
│   └── resample.jl                 time-axis resampling helper
│
├── python/mdd/                   Python package implementing MDD
│   ├── mdd.py                      core algorithm + MDDParams
│   ├── wavefield_separation.py     SHT-based outer-shell pin / pout split
│   ├── wavefield_separation_local.py   local plane-wave variant
│   ├── io.py                       HDF5 I/O (matches Julia schema)
│   └── cli.py                      `python -m mdd.cli` entry point
│
├── scripts/                      End-to-end pipeline
│   ├── greens/                     stage 1 — Green's-function retrieval
│   │   ├── analytical.jl             method 1: closed form (scatterer=none only)
│   │   ├── impulsive.jl              method 2: FDTD point source
│   │   ├── reverb.jl                 method 3a: reverberant FDTD data
│   │   └── mdd_extract.py            method 3b: MDD on reverb data
│   ├── hologram/                   stage 2 — hologram synthesis
│   │   └── synthesize.jl             GF-driven FDTD + Kirchhoff-Helmholtz
│   ├── scattering/                 far-field scattering acquisition
│   │   └── acquire.jl                samples p, vₙ on a far-field sphere
│   ├── disguise/                   acoustic disguising / cloaking
│   │   └── synthesize.jl             scatterer A driven by target B's GF
│   └── run_reverb_mdd.sh           reverb-MDD GF retrieval + hologram synthesis
│
├── configs/                      cfg templates copied into a run dir (see configs/README.md)
├── test/                         test suite (run via Pkg.test)
├── diagnostics/                  CLI scripts: check_* / compare_* / sweep_* / plot_* / inspect_*
├── docs/                         supplementary notes (physics gotchas etc.)
│
├── Project.toml                  Julia dependencies
├── Manifest.toml                 Julia dependencies (PINNED — do not delete)
├── pyproject.toml                Python dependencies
├── CITATION.cff
├── .zenodo.json
└── LICENSE                       MIT
```

## Requirements

- **Julia** ≥ 1.10
- **Python** ≥ 3.10
- **CPU or CUDA-capable GPU** — the default backend is `Threads` (CPU), and
  the paper-scale results were produced on CPU (see *Hardware used for the
  paper* below). A CUDA GPU via `CUDA.jl` / `ParallelStencil.jl` is selected
  with `ACOUSTIC_DISGUISING_BACKEND=CUDA` before launching Julia; it is
  substantially faster per FDTD step but is not required.

Full pinned dependencies live in `Manifest.toml` (Julia) and are resolved
from `pyproject.toml` (Python). Committing `Manifest.toml` is intentional:
it guarantees the exact package versions used to produce the published
results.

## Reproducing the paper

```bash
# 1. Clone
git clone https://github.com/Nano560/acoustic-disguising.git
cd acoustic-disguising

# 2. Install Julia deps (pinned via Manifest.toml)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Install Python deps
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. Run the full pipeline end-to-end (reverb + MDD path)
RUN=runs/paper
mkdir -p "$RUN" && cp configs/paper.toml "$RUN/config.toml"
bash scripts/run_reverb_mdd.sh "$RUN"
```

`run_reverb_mdd.sh` runs the reverb-MDD method end-to-end. To reproduce the paper
with a different Green's-function method, run one of the alternatives
manually before the hologram step (pass `--threads=N` to match your core
count — the FDTD stages run single-threaded otherwise):

```bash
# Method 1 — analytical closed form (homogeneous medium only)
julia --project=. --threads=10 scripts/greens/analytical.jl   "$RUN"
julia --project=. --threads=10 scripts/hologram/synthesize.jl "$RUN" --gf-method=analytical

# Method 2 — FDTD impulsive point source
julia --project=. --threads=10 scripts/greens/impulsive.jl    "$RUN"
julia --project=. --threads=10 scripts/hologram/synthesize.jl "$RUN" --gf-method=impulsive
```

For a real-scatterer bare-FDTD reference run (no Green's functions):

```bash
julia --project=. --threads=10 scripts/hologram/synthesize.jl "$RUN" --scatterer-mode=real
```

### Reproducing the scattering and disguising results

The far-field scattering data and the disguising (cloaking) holograms are
produced by two further scripts that consume the Stage 1 Green's functions.
Run them once the Green's functions exist for the relevant scatterers:

```bash
# Far-field scattering acquisition — one process per (scatterer-mode,
# gf-method). The postprocess subtracts the matching scatterer=none
# baseline, so always include `none` in the scatterer set.
julia --project=. --threads=10 scripts/scattering/acquire.jl "$RUN" \
    --scatterer-mode=real --scatterer=all
julia --project=. --threads=10 scripts/scattering/acquire.jl "$RUN" \
    --scatterer-mode=hologram --gf-method=mdd --scatterer=all

# Acoustic disguising — the interior scatterer A reads as a target B on
# the outer surface. `--target=none` is the invisibility case.
julia --project=. --threads=10 scripts/disguise/synthesize.jl "$RUN" \
    --real=all --target=sphere --gf-method=mdd
```

This archive reproduces the paper's numerical results — the FDTD fields,
Green's functions, holograms, and scattering data. The figure-generation
code (plots, 3D Blender renders, movie pipeline) is not part of the archive.

## Output storage

Generated data (Green's functions, reverberant data, holograms, figures)
is written under the user-chosen `<run_dir>` (one per experiment). It is `.gitignored`
— the artefacts are regenerable from the code. The paper-scale run
produces of order 100 GB of intermediate HDF5 data (the holograms dominate,
at several GB each); create the run directory on scratch storage.

## Hardware used for the paper

Every result in the paper was produced on CPU — the `Threads` backend — on a
single workstation; no GPU was used:

- Machine: **Apple M1 Pro, 10 cores, 32 GB RAM**
- Toolchain / OS: **Julia 1.12.6, Python 3.11.7, macOS 26.5**
- Wall time, full pipeline: **~1 week** (all four scatterers, all stages)

## Citation

If this code or the associated results contribute to your work, please
cite both the paper and the archived software release:

```bibtex
@article{Muller2026,
  author  = {Müller, Jonas and van Manen, Dirk-Jan},
  title   = {Acoustic disguising: a unified framework for cloaking and holography},
  journal = {Physical Review Research},
  year    = {2026},
  doi     = {TODO}
}

@software{Muller2026_code,
  author    = {Müller, Jonas and van Manen, Dirk-Jan},
  title     = {Acoustic Disguising: Green's function retrieval and hologram synthesis},
  version   = {1.0.0},
  year      = {2026},
  doi       = {10.5281/zenodo.XXXXXXX},
  publisher = {Zenodo}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
