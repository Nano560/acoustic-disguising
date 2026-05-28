# Installation

This project has two halves — Julia for the FDTD + hologram synthesis, and
Python for the MDD inversion. Both are installed in-place inside the cloned
repository; no system-wide packages are needed.

## 1. Prerequisites

| Tool   | Minimum | Recommended | Notes |
|--------|---------|-------------|-------|
| Julia  | 1.10    | 1.12        | Use [`juliaup`](https://github.com/JuliaLang/juliaup) on Linux/macOS/Windows. |
| Python | 3.10    | 3.11+       | Tested with CPython; conda and venv both work. |
| Git    | any     | —           | `git clone` only. |
| CUDA toolkit | 11.8 | 12.x      | Optional; needed only for the GPU backend. |
| NVIDIA driver | matching toolkit | — | Optional; CPU-only runs are fine without it. |

The CPU backend works on macOS (Apple Silicon and Intel), Linux, and
Windows. The CUDA backend is supported on Linux only — `CUDA.jl` will pick
up the system toolkit automatically once a recent NVIDIA driver is
installed.

## 2. Clone and enter

```bash
git clone https://github.com/Nano560/acoustic-disguising.git
cd acoustic-disguising
```

## 3. Julia side

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This reads `Manifest.toml` and downloads the exact pinned versions of every
Julia dependency (CUDA, ParallelStencil, FFTW, HDF5, CairoMakie, ...).
Expect a few minutes the first time as binary artifacts are pulled.

To verify:

```bash
julia --project=. -e 'using AcousticDisguising; println("OK, backend = ", AcousticDisguising.BACKEND)'
```

You should see `OK, backend = Threads` (the default — safe on every
platform). To use a GPU on a Linux box with a working NVIDIA driver,
set the env var explicitly:

```bash
ACOUSTIC_DISGUISING_BACKEND=CUDA julia --project=. ...
```

The backend is chosen at module load time, so the env var must be set
before Julia starts.

For multi-threaded CPU runs use Julia's `--threads` flag (matching the
number of physical cores is usually best). Pipeline scripts take a *run
directory* as their positional argument; the cfg lives at
`<run_dir>/config.toml` inside it:

```bash
julia --project=. --threads=10 scripts/greens/impulsive.jl /path/to/runs/quickstart/
```

## 4. Python side

In a fresh virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

To verify:

```bash
python -c "import mdd; print('OK, version =', mdd.__version__)"
```

The Python package is installed in editable mode (`-e`) so any local edits
take effect immediately. The MDD inversion is run as a module:

```bash
python -m mdd.cli --help
```

For end-to-end use, the wrapper script `scripts/greens/mdd_extract.py`
auto-locates the reverb h5, drives the inversion, and writes the GF h5 +
diagnostic plots — that's the canonical entry point, not the bare
`mdd.cli` module.

## 5. Run the smoke test

Set up a quickstart run directory and execute the pipeline against it:

```bash
RUN=/tmp/3ddisguising-quickstart
mkdir -p "$RUN"
cp configs/quickstart.toml "$RUN/config.toml"
bash scripts/run_reverb_mdd.sh "$RUN"
```

This runs the reverb-MDD-hologram path on a tiny grid in a few minutes.
If it finishes without error and writes files under `$RUN/`, the
installation is healthy. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for
what to expect.

## 6. Run the test suites

```bash
# Julia
julia --project=. -e 'using Pkg; Pkg.test()'

# Python
pytest python/mdd/tests
```

Both test suites are CPU-only and run in well under a minute on a laptop.

---

## Troubleshooting

**`Pkg.instantiate()` fails on CUDA.jl on macOS / Windows.** That's fine —
the manifest pulls CUDA.jl as a dependency, but it does not require a GPU
to install (the kernels are JIT-compiled at first call). Only setting
`ACOUSTIC_DISGUISING_BACKEND=CUDA` will actually try to use the GPU.

**`InitError: CUDA driver not found`.** You set the CUDA backend on a
machine without a working NVIDIA driver. Either install the driver or
unset the variable to fall back to `Threads`.

**`pylops` install fails on `numpy` ABI.** Re-create the venv with a fresh
`pip` (`pip install -U pip`) and reinstall — pylops needs a recent numpy
wheel.

**`HDF5.jl` complains about a system library.** The pinned `HDF5_jll`
artifact ships its own libhdf5; if a system one is being picked up, unset
`HDF5_PATH` from your environment.

**Out-of-memory during the reverb stage.** Reduce `[grid].n` (or the
`[reverb.grid]` overrides) and `[reverb].nPoints_ill` in the config —
reverb cost is `n³ · nt · n_ill`. The `quickstart.toml` defaults are
conservative; `paper.toml` is sized for a GPU with ≥ 16 GB VRAM.

**Tests pass but pipeline hangs.** Check that you're not silently using a
single-threaded build of FFTW; the FDTD kernels rely on threaded BLAS for
the boundary inversions.
