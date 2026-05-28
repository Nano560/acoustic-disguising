#!/usr/bin/env python3
# =============================================================================
# Green's-function retrieval — method 3 (part 2): MDD extraction.
#
# Multi-Dimensional Deconvolution converts reverberant pressure / velocity
# recordings (from `reverb.jl`) into Green's functions of the same shape
# as the impulsive / analytical results. Thin wrapper around `mdd.cli` plus
# diagnostic plots of the input traces and the extracted GFs.
#
# Reads:   <run_dir>/reverb/reverb_<scatterer>.h5
# Writes:  <run_dir>/greens/mdd_extracted_<scatterer>.h5
#          <run_dir>/figures/mdd/<sc>_*.png  (diagnostic per-panel PNGs)
# Usage:   python scripts/greens/mdd_extract.py <run_dir> \\
#              [--scatterer=none|sphere|cube|cross|all]
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from mdd.cli import main as mdd_main

C0 = 1500.0                # Domain default sound speed (water)
ALL_SCATTERERS = ("none", "sphere", "cube", "cross")
PLOT_ISRC = 1              # ill source index used for the diagnostic plots

# Diverging colormap with dark centre — matches Julia's :berlin spirit.
_PLOT_CMAP = LinearSegmentedColormap.from_list(
    "blackcenter", ["#1f77b4", "#000000", "#ff5050"])
plt.style.use("dark_background")


# -----------------------------------------------------------------------------
# Console formatting
# -----------------------------------------------------------------------------
_RULE_WIDTH = 72


def _banner(text: str) -> str:
    """Section divider with `text` centred in a horizontal rule. Used per
    scatterer so a `--scatterer=all` run is easy to scan visually."""
    pad = max(2, (_RULE_WIDTH - len(text) - 2) // 2)
    return "─" * pad + f"  {text}  " + "─" * pad


def _path_rel(p: Path, base: Path) -> str:
    """Display path relative to `base` when possible; otherwise absolute."""
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> tuple[Path, str | None]:
    run_dir: Path | None = None
    scatterer = None
    for a in argv:
        if a.startswith("--scatterer="):
            scatterer = a.split("=", 1)[1]
        elif not a.startswith("--"):
            run_dir = Path(a).resolve()
    if run_dir is None:
        raise SystemExit(
            "usage: mdd_extract.py <run_dir> [--scatterer=none|sphere|cube|cross|all]")
    return run_dir, scatterer


# -----------------------------------------------------------------------------
# Diagnostic plots
# -----------------------------------------------------------------------------
def _saturated_vmax(slab: np.ndarray, q: float = 0.98) -> float:
    """98th-percentile of |slab|, restricted to non-zeros so mostly-zero
    arrays don't render flat-black."""
    nz = np.abs(slab).ravel()
    nz = nz[nz > 0]
    v = float(np.quantile(nz, q)) if nz.size else 0.0
    return v if v > 0 else 1.0


def _save_panel(slab: np.ndarray, t: np.ndarray, out_path: Path) -> None:
    """Axis-less heatmap PNG: image fills the canvas, time top-to-bottom,
    nearest-neighbor (sample-honest)."""
    vmax = _saturated_vmax(slab)
    n_rec = slab.shape[1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(slab, origin="lower", aspect="auto", interpolation="nearest",
              extent=[1, n_rec, t[0] * 1e3, t[-1] * 1e3],
              cmap=_PLOT_CMAP, vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.invert_yaxis()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _read_reverb_for_plot(reverb_h5: Path) -> dict:
    with h5py.File(reverb_h5, "r") as f:
        # Julia → Python: HDF5.jl reverses dim order on write, so 3D arrays
        # come back as (n_ill, n_rec, nt). Transpose to (nt, n_rec, n_ill).
        data = {k: np.array(f[k]).transpose(2, 1, 0)
                for k in ("outer_p", "outer_vnz", "inner_p", "inner_vnz")}
        data["t"] = np.array(f["t"])
        for k in ("inner_positions", "outer_positions", "ill_positions"):
            data[k] = np.array(f[k]).T              # (3, n) → (n, 3)
    return data


def _plot_mdd_input(sc: str, reverb_h5: Path, fig_dir: Path, run_dir: Path,
                    iSrc: int = PLOT_ISRC) -> None:
    """One axis-less PNG per reverb-stage panel (outer_p, outer_vnz,
    inner_p, inner_vnz), receivers sorted by distance to the ill source."""
    d = _read_reverb_for_plot(reverb_h5)
    n_ill = d["outer_p"].shape[2]
    if not (1 <= iSrc <= n_ill):
        print(f"[mdd:{sc}] input plot skipped — iSrc={iSrc} out of range 1..{n_ill}",
              file=sys.stderr)
        return

    ill_pt = d["ill_positions"][iSrc - 1]
    order_o = np.argsort(np.linalg.norm(d["outer_positions"] - ill_pt, axis=1))
    order_i = np.argsort(np.linalg.norm(d["inner_positions"] - ill_pt, axis=1))

    out_dir = fig_dir / "mdd"
    panels = (
        ("outer_p",   d["outer_p"],   order_o),
        ("outer_vnz", d["outer_vnz"], order_o),
        ("inner_p",   d["inner_p"],   order_i),
        ("inner_vnz", d["inner_vnz"], order_i),
    )
    for name, arr, order in panels:
        slab = arr[:, order, iSrc - 1]
        out_path = out_dir / f"{sc}_{name}_input.png"
        _save_panel(slab, d["t"], out_path)
        print(f"[mdd:{sc}] input plot   → {_path_rel(out_path, run_dir)}", file=sys.stderr)


def _plot_mdd_result(sc: str, mdd_h5: Path,
                     fig_dir: Path, run_dir: Path,
                     iSrc: int = PLOT_ISRC) -> None:
    """One axis-less PNG per MDD-extracted kernel (p_p, v_p), inner
    receivers sorted by distance to the chosen outer source.

    Note: the third axis of the MDD output is **n_outer** (not n_ill). The
    inverted GF is independent of the ill geometry by construction.
    """
    # mdd.cli matches Julia's on-disk axis convention (axes reversed from
    # canonical) so the same load rules apply to both Julia- and
    # Python-written GF files.
    with h5py.File(mdd_h5, "r") as f:
        p_p = np.array(f["p_p"]).transpose(2, 1, 0)   # → (nt_out, n_inner, n_outer)
        v_p = np.array(f["v_p"]).transpose(2, 1, 0)
        t = np.array(f["t"])
        rec_positions = np.array(f["rec_positions"]).T  # (3, n_inner) → (n_inner, 3)
        src_positions = np.array(f["src_positions"]).T  # (3, n_outer) → (n_outer, 3)

    n_outer = p_p.shape[2]
    if not (1 <= iSrc <= n_outer):
        print(f"[mdd:{sc}] result plot skipped — iSrc={iSrc} out of range 1..{n_outer}",
              file=sys.stderr)
        return

    src_pt = src_positions[iSrc - 1]
    order  = np.argsort(np.linalg.norm(rec_positions - src_pt, axis=1))

    out_dir = fig_dir / "mdd"
    for name, arr in (("p_p", p_p), ("v_p", v_p)):
        slab = arr[:, order, iSrc - 1]
        out_path = out_dir / f"{sc}_{name}_mdd.png"
        _save_panel(slab, t, out_path)
        print(f"[mdd:{sc}] result plot  → {_path_rel(out_path, run_dir)}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    run_dir, scatterer_cli = _parse_args(argv)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")
    config_path = run_dir / "config.toml"
    if not config_path.is_file():
        raise SystemExit(f"config.toml not found at {config_path}")
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    scatterer = scatterer_cli or cfg.get("scatterer", "cross")
    scatterers = ALL_SCATTERERS if scatterer == "all" else (scatterer,)

    reverb_root = run_dir / "reverb"
    greens_root = run_dir / "greens"
    fig_dir     = run_dir / "figures"
    greens_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    skipped = []
    for sc in scatterers:
        print(file=sys.stderr)
        print(_banner(f"scatterer = {sc}"), file=sys.stderr)

        reverb_h5 = reverb_root / f"reverb_{sc}.h5"
        if not reverb_h5.is_file():
            print(f"[mdd:{sc}] SKIPPED — reverb file not found: {reverb_h5}",
                  file=sys.stderr)
            skipped.append(sc)
            continue

        gfs_h5 = greens_root / f"mdd_extracted_{sc}.h5"

        print(f"[mdd:{sc}] reverb in    ← {_path_rel(reverb_h5, run_dir)}",
              file=sys.stderr)
        print(f"[mdd:{sc}] gfs out      → {_path_rel(gfs_h5, run_dir)}",
              file=sys.stderr)
        rc = mdd_main(["--input",  str(reverb_h5),
                       "--output", str(gfs_h5),
                       "--config", str(config_path)])
        exit_code = exit_code or rc

        # Plot the MDD input regardless of MDD success — useful even on
        # failure. Plot the output only when extraction succeeded.
        try:
            _plot_mdd_input(sc, reverb_h5, fig_dir, run_dir)
        except Exception as e:
            print(f"[mdd:{sc}] input plot ERROR — {e}", file=sys.stderr)
        if rc == 0 and gfs_h5.exists():
            try:
                _plot_mdd_result(sc, gfs_h5, fig_dir, run_dir)
            except Exception as e:
                print(f"[mdd:{sc}] result plot ERROR — {e}", file=sys.stderr)

    print(file=sys.stderr)
    print(_banner("done"), file=sys.stderr)
    completed = [s for s in scatterers if s not in skipped]
    if completed:
        print(f"[mdd] completed ({len(completed)}): {', '.join(completed)}",
              file=sys.stderr)
    if skipped:
        print(f"[mdd] skipped   ({len(skipped)}): {', '.join(skipped)}",
              file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
