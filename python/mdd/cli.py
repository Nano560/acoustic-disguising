"""Command-line entry point for MDD extraction.

The canonical way to run this stage is the wrapper
`scripts/greens/mdd_extract.py`, which auto-locates the reverb h5,
writes the GF h5, and saves diagnostic plots. This module is invoked
by the wrapper in-process via `from mdd.cli import main`.

For debugging the bare inversion (no plots, no auto-locate), invoke
directly:

    python -m mdd.cli --input  <reverb_<scatterer>.h5> \\
                      --output <mdd_extracted_<scatterer>.h5> \\
                      --config configs/paper.toml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from . import io, mdd


# Domain defaults (water) — match Julia `Domain` constructor in
# src/simulation.jl. Used as fallbacks if the reverb attrs don't carry them.
_RHO0_DEFAULT = 1000.0
_C0_DEFAULT   = 1500.0


def _reverb_dom_dx(reverb_attrs: dict, cfg: dict) -> float:
    """Resolve the reverb-stage `dom_dx`.

    Prefer the H5 attr (single source of truth); fall back to deriving it
    from the cfg with `[reverb.grid]` applied (mirrors the Julia
    `_apply_reverb_overrides` logic) so files written before reverb.jl
    started recording dom_dx still get a sensible value.
    """
    if "dom_dx" in reverb_attrs:
        return float(reverb_attrs["dom_dx"])

    root_g = cfg.get("grid", {})
    root_n = int(root_g.get("n", 0))
    if root_n <= 1 or "xmax" not in root_g:
        raise SystemExit("cli: reverb file lacks `dom_dx` and cfg has no usable [grid] block")
    root_dx = 2.0 * float(root_g["xmax"]) / (root_n - 1)

    rev_grid = cfg.get("reverb", {}).get("grid", {})
    if "margin_cells" in rev_grid:
        r_ill = float(cfg.get("reverb", {}).get("radius_ill", 0.6))
        m     = int(rev_grid["margin_cells"])
        xmax  = r_ill + m * root_dx
        n     = round(2.0 * xmax / root_dx) + 1
        n     = n + 1 if n % 2 == 0 else n
    elif "xmax" in rev_grid and "n" not in rev_grid:
        xmax = float(rev_grid["xmax"])
        n    = round(2.0 * xmax / root_dx) + 1
        n    = n + 1 if n % 2 == 0 else n
    else:
        n    = int(rev_grid.get("n", root_n))
        xmax = float(rev_grid.get("xmax", root_g["xmax"]))
    return 2.0 * xmax / (n - 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  required=True, type=Path,
                   help="HDF5 with reverberant data (from scripts/02).")
    p.add_argument("--output", required=True, type=Path,
                   help="HDF5 with extracted Green's functions "
                        "(input to scripts/04).")
    p.add_argument("--config", required=True, type=Path,
                   help="TOML config file (e.g. configs/paper.toml).")
    args = p.parse_args(argv)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    params = mdd.MDDParams.from_cfg(cfg)
    print(f"[mdd] params: {params}", file=sys.stderr)

    reverb = io.load_reverb(args.input)
    rev_attrs = reverb.get("attrs", {})

    # Production layout (= what MDD inverts onto) is a downstream choice
    # read from the cfg, not from the reverb file. The reverb file knows
    # only what was actually recorded.
    surfaces = cfg.get("surfaces", {})
    if "nPoints_inner" not in surfaces or "nPoints_outer" not in surfaces:
        raise SystemExit(
            "cli: cfg is missing [surfaces].nPoints_inner / .nPoints_outer "
            "— required to specify the MDD production layout.")
    n_inner_prod = int(surfaces["nPoints_inner"])
    n_outer_prod = int(surfaces["nPoints_outer"])

    gf = mdd.mdd_extract(reverb, params=params,
                         n_inner_prod=n_inner_prod, n_outer_prod=n_outer_prod)

    # `mdd_extract` attaches the post-resampling positions to the gf dict —
    # use those instead of the reverb file's positions, which describe the
    # (possibly denser) recording layout.
    inner_positions = gf["inner_positions"]
    outer_positions = gf["outer_positions"]
    ill_positions   = gf["ill_positions"]

    n_inner = inner_positions.shape[0]
    n_outer = outer_positions.shape[0]
    n_ill   = ill_positions.shape[0]
    if gf["p_p"].shape[1:] != (n_inner, n_outer):
        raise SystemExit(
            f"cli: GF axis-(1,2) shape {gf['p_p'].shape[1:]} doesn't match "
            f"(n_inner={n_inner}, n_outer={n_outer}) — resampling logic "
            f"is out of sync.")

    # Acoustic impedance for the v_p unit conversion. Match Julia Domain
    # defaults (rho0=1000, c0=1500) unless reverb attrs carry per-run values.
    rho0 = float(rev_attrs.get("dom_rho0", _RHO0_DEFAULT))
    c0   = float(rev_attrs.get("dom_c0",   _C0_DEFAULT))
    z0   = rho0 * c0

    dom_dx = _reverb_dom_dx(rev_attrs, cfg)

    # MDD-native amplitude convention. The pylops MDD inverts
    #     y = √n_t · Δt · Δr · Σ_{i_r} G ⊗ x
    # for G, where Δr is a 1-D spacing (not a surface element). Our `dr` is
    # `2π · radius_inner / n_inner_prod` (matches the original notebook),
    # and the inversion runs over the full nt of the reverb input (before
    # tmax_out truncation). Recording these here lets compare_gfs.py
    # convert MDD output onto the analytical / impulsive surface-area
    # quadrature convention without re-deriving them at compare time.
    r_inner = float(rev_attrs.get("radius_inner", 0.2))
    mdd_dr_pylops = 2.0 * np.pi * r_inner / n_inner_prod
    mdd_n_t_inversion = int(np.asarray(reverb["t"]).size)

    # Build the unified attr set: same keys as
    # scripts/greens/{impulsive,analytical}.jl plus MDD-specific ones.
    def _attr(k, default=None):
        v = rev_attrs.get(k, default)
        # h5py rejects numpy 0-d arrays as attrs in some versions — coerce.
        if hasattr(v, "item"):
            v = v.item()
        return v

    attrs: dict = {
        "scatterer":          _attr("scatterer"),
        "backend":            "mdd",
        "dom_dt":             _attr("dom_dt"),
        "dom_dx":             dom_dx,
        "dom_tmax":           _attr("dom_tmax"),
        "dom_c0":             c0,
        "dom_rho0":           rho0,
        "n":                  _attr("n", _attr("n_grid")),
        "nPoints_outer":      n_outer,
        "nPoints_inner":      n_inner,
        "n_ill":              n_ill,
        "radius_inner":       _attr("radius_inner"),
        "radius_outer":       _attr("radius_outer"),
        "radius_ill":         _attr("radius_ill"),
        "fc_source":          _attr("fc_source"),
        "fs_out":             _attr("fs_out"),
        # python/mdd/io.py::save_gfs applies the full pylops→impulsive
        # conversion (Wapenaar 2011 eqs 20+43; pylops MDC normalization
        # + per-pair dipole→monopole geometric factor + ∂_t), so the
        # on-disk values land in the same physical 1/(m·s²) units as
        # analytical / impulsive `p_p`.
        "units_convention":   "physical",
        "mdd_dr_pylops":      mdd_dr_pylops,
        "mdd_n_t_inversion":  mdd_n_t_inversion,
        "source_file":        str(args.input),
        "config_file":        str(args.config),
        "created_at":         _dt.datetime.now().isoformat(timespec="seconds"),
        "mdd_separation":     params.separation_method,
        "mdd_fmax_hz":        params.fmax_hz,
        "mdd_iter_lim":       params.iter_lim,
        "mdd_damp":           params.damp,
        "mdd_tmax_out":       params.tmax_out,
    }
    # Drop keys that turned out to be None (reverb file didn't carry them).
    attrs = {k: v for k, v in attrs.items() if v is not None}

    io.save_gfs(
        args.output,
        p_p           = gf["p_p"],
        v_p         = gf["v_p"],
        t               = gf["t"],
        outer_positions = outer_positions,
        inner_positions = inner_positions,
        ill_positions   = ill_positions,
        z0              = z0,
        n_t_inversion   = mdd_n_t_inversion,
        dr_pylops       = mdd_dr_pylops,
        attrs           = attrs,
    )
    print(f"[mdd] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
