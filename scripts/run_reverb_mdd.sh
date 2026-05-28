#!/usr/bin/env bash
# =============================================================================
# Run the reverb-MDD Green's-function path end-to-end, then synthesise a
# hologram:
#
#   1. Green's-function retrieval — generate reverberant FDTD data, then
#      deconvolve into Green's functions via MDD.
#   2. Hologram synthesis — drive the FDTD with the extracted GFs.
#
# This covers ONE of the three interchangeable GF methods (analytical /
# impulsive / reverb-MDD). It does not run the scattering or disguise
# stages — see the README's "Reproducing the paper" section for those.
#
# Takes one required argument: path to the run directory. The run dir must
# contain a `config.toml` and is the single root for all stage outputs:
#
#   bash scripts/run_reverb_mdd.sh /path/to/runs/<name>/
#
# The FDTD stages are launched with `--threads` set to the physical core
# count; export JULIA_NUM_THREADS to override.
#
# To use a different GF method, run
#   julia --project=. scripts/greens/analytical.jl  <run_dir>  # closed form (scatterer=none)
#   julia --project=. scripts/greens/impulsive.jl   <run_dir>  # FDTD point source
# and then
#   julia --project=. scripts/hologram/synthesize.jl <run_dir> --gf-method=impulsive
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_dir>" >&2
    exit 1
fi

RUN_DIR="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$RUN_DIR/config.toml" ]; then
    echo "config.toml not found at $RUN_DIR/config.toml" >&2
    exit 1
fi

# Thread count for the FDTD stages. An explicit JULIA_NUM_THREADS wins;
# otherwise detect the physical core count (macOS via sysctl, Linux via
# nproc), falling back to 1. Without this the reverb FDTD — the expensive
# stage — runs single-threaded.
if [ -n "${JULIA_NUM_THREADS:-}" ]; then
    THREADS="$JULIA_NUM_THREADS"
elif sysctl -n hw.physicalcpu >/dev/null 2>&1; then
    THREADS="$(sysctl -n hw.physicalcpu)"
elif command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
else
    THREADS=1
fi

echo "=== Pipeline: run_dir=$RUN_DIR  (julia --threads=$THREADS) ==="

# Graceful-stop sentinel: `touch $RUN_DIR/STOP` from another terminal to
# finish the current FDTD source and halt the pipeline. reverb.jl leaves
# STOP in place when triggered; we check between stages so the surrounding
# pipeline also stops. Clear any leftover STOP so we don't immediately abort.
STOP_FILE="$RUN_DIR/STOP"
rm -f "$STOP_FILE"
check_stop() {
    if [ -f "$STOP_FILE" ]; then
        echo "=== STOP file present at $STOP_FILE — halting pipeline ===" >&2
        exit 0
    fi
}

# Green's-function retrieval (reverb + MDD).
julia --project=. --threads="$THREADS" scripts/greens/reverb.jl "$RUN_DIR"
check_stop
python scripts/greens/mdd_extract.py "$RUN_DIR"
check_stop

# Hologram synthesis.
julia --project=. --threads="$THREADS" scripts/hologram/synthesize.jl "$RUN_DIR" --gf-method=mdd

echo "=== Pipeline complete ==="
