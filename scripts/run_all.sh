#!/usr/bin/env bash
# =============================================================================
# Run the full Green's-function comparison for one run directory: retrieve
# the GFs by all three methods, synthesise holograms from every method, and
# acquire the far-field scattering for every method.
#
# Stages 1-3 ONLY compute Green's functions (in increasing order of cost);
# stage 4 synthesises the holograms and stage 5 the scattering, so both run
# against a complete, calibrated GF set:
#
#   1. analytical  — closed-form free-space GFs (homogeneous only)
#   2. impulsive   — FDTD point-source reference GFs
#   3. reverb-MDD  — reverberant FDTD → MDD deconvolution, then the MDD GF
#                    amplitude is calibrated (diagnostics/check_gf_scale.jl writes an
#                    `empirical_scale` attr into each mdd_extracted_*.h5)
#   4. holograms   — synthesise hologram_<scatterer>_<method>.h5 for all three
#                    methods; the MDD holograms pick up the stage-3 scale
#   5. scattering  — acquire far-field scattering_<scatterer>_<label>.h5 for
#                    the bare-FDTD `real` field and the impulsive / MDD
#                    holograms (consumed by the figure-generation pipeline)
#
# Stages 4-5 give directly comparable outputs across the three GF methods.
# The analytical method is homogeneous-only: its hologram is synthesised for
# `--scatterer=none` alone, and it has no scatterer-specific scattering, so it
# is absent from stage 5. Impulsive and MDD sweep whatever `scatterer` in
# config.toml selects.
#
# Does NOT run the disguise stage, nor regenerate paper figures — see the
# README's "Reproducing the paper" section. For just the reverb-MDD path
# with its hologram, use run_reverb_mdd.sh.
#
# Takes one required argument: path to the run directory (must contain a
# `config.toml`):
#
#   bash scripts/run_all.sh /path/to/runs/<name>/
#
# The FDTD stages are launched with `--threads` set to the physical core
# count; export JULIA_NUM_THREADS to override.
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_dir>" >&2
    exit 1
fi

# Resolve the run dir to an absolute path before the `cd` below, so a relative
# argument keeps working after the working directory changes to the repo root.
RUN_DIR="$1"
if [ ! -d "$RUN_DIR" ]; then
    echo "run dir not found: $RUN_DIR" >&2
    exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$RUN_DIR/config.toml" ]; then
    echo "config.toml not found at $RUN_DIR/config.toml" >&2
    exit 1
fi

# Thread count for the FDTD stages — same detection as run_reverb_mdd.sh.
if [ -z "${JULIA_NUM_THREADS:-}" ]; then
    if sysctl -n hw.physicalcpu >/dev/null 2>&1; then
        JULIA_NUM_THREADS="$(sysctl -n hw.physicalcpu)"
    elif command -v nproc >/dev/null 2>&1; then
        JULIA_NUM_THREADS="$(nproc)"
    else
        JULIA_NUM_THREADS=1
    fi
fi
export JULIA_NUM_THREADS
JULIA=(julia --project=. --threads="$JULIA_NUM_THREADS")

# Graceful-stop sentinel: `touch $RUN_DIR/STOP` from another terminal to
# finish the current FDTD source and halt the pipeline. The impulsive and
# reverb stages leave STOP in place when triggered; we check between every
# stage so the surrounding pipeline also stops. Clear any leftover STOP from
# a previous run so we don't immediately abort.
STOP_FILE="$RUN_DIR/STOP"
rm -f "$STOP_FILE"
check_stop() {
    if [ -f "$STOP_FILE" ]; then
        echo "=== STOP file present at $STOP_FILE — halting run_all ===" >&2
        exit 0
    fi
}

echo "=== run_all: run_dir=$RUN_DIR  (julia --threads=$JULIA_NUM_THREADS) ==="

# --- Stage 1: analytical GF (closed-form, homogeneous only) ------------------
echo "--- [1/5] analytical: Green's functions ---"
"${JULIA[@]}" scripts/greens/analytical.jl "$RUN_DIR"
check_stop

# --- Stage 2: impulsive GF (FDTD point source) -------------------------------
echo "--- [2/5] impulsive: Green's functions ---"
"${JULIA[@]}" scripts/greens/impulsive.jl "$RUN_DIR"
check_stop

# --- Stage 3: reverb-MDD GF + MDD amplitude calibration ----------------------
echo "--- [3/5] reverb-MDD: Green's functions + scale calibration ---"
"${JULIA[@]}" scripts/greens/reverb.jl "$RUN_DIR"
check_stop
python scripts/greens/mdd_extract.py "$RUN_DIR"
check_stop
"${JULIA[@]}" diagnostics/check_gf_scale.jl "$RUN_DIR" --gf-method=mdd --scatterer=all --write-scale
check_stop

# --- Stage 4: holograms for every GF method ----------------------------------
echo "--- [4/5] hologram synthesis (analytical, impulsive, mdd) ---"
"${JULIA[@]}" scripts/hologram/synthesize.jl "$RUN_DIR" --gf-method=analytical --scatterer=none
check_stop
"${JULIA[@]}" scripts/hologram/synthesize.jl "$RUN_DIR" --gf-method=impulsive
check_stop
"${JULIA[@]}" scripts/hologram/synthesize.jl "$RUN_DIR" --gf-method=mdd
check_stop

# --- Stage 5: far-field scattering acquisition for every GF method -----------
# `real` is the bare-FDTD ground truth the scattering figures compare against;
# impulsive / MDD are hologram-mode. Analytical is omitted (homogeneous-only —
# no scatterer-specific far field). acquire.jl reads the GF files directly, so
# this depends on stages 1-3, not on stage 4.
echo "--- [5/5] scattering acquisition (real, impulsive, mdd) ---"
"${JULIA[@]}" scripts/scattering/acquire.jl "$RUN_DIR" --scatterer-mode=real
check_stop
"${JULIA[@]}" scripts/scattering/acquire.jl "$RUN_DIR" --scatterer-mode=hologram --gf-method=impulsive
check_stop
"${JULIA[@]}" scripts/scattering/acquire.jl "$RUN_DIR" --scatterer-mode=hologram --gf-method=mdd

echo "=== run_all complete ==="
