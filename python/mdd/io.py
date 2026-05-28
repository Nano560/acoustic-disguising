"""HDF5 I/O — must mirror the schema documented in src/reverb.jl and src/io.jl.

If you change the schema on either side, update BOTH files.

Reverb input (from scripts/greens/reverb.jl):
    /inner_p, /inner_vnz             (nt_ill, n_inner, n_ill)    Float32
    /outer_p, /outer_vnz             (nt_ill, n_outer, n_ill)    Float32
    /t                               (nt_ill,)                   Float32
    /ill_positions                   (n_ill, 3)                  Float32
    /inner_positions                 (n_inner, 3)                Float32
    /outer_positions                 (n_outer, 3)                Float32
    attrs: dom_dt, dom_dx, dom_tmax, radius_ill, fc_source, fs_out,
           radius_inner, radius_outer, nPoints_inner, nPoints_outer,
           n, n_done, scatterer, ...
    (The MDD-side production layout is read from cfg[surfaces].nPoints_*
    by python/mdd/cli.py at extract time — not stored here.)

MDD-extracted GF output (for src/io.jl::load_gf_mdd → hologram synthesis;
schema matches the impulsive/analytical files):
    /p_p                     (nt_out, n_inner, n_outer)  Float32
    /v_p                     (nt_out, n_inner, n_outer)  Float32
    /t                       (nt_out,)                   Float32
    /src_positions           (n_outer, 3)                Float32
    /rec_positions           (n_inner, 3)                Float32
    /ill_positions           (n_ill, 3)                  Float32
    attrs: scatterer, backend="mdd", dom_dt, dom_dx, dom_tmax, n,
           nPoints_outer, nPoints_inner, n_ill, radius_inner,
           radius_outer, radius_ill, units_convention, source_file,
           config_file, created_at, mdd_iter_lim, mdd_damp,
           mdd_fmax_hz, mdd_tmax_out
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load_reverb(path: str | Path) -> dict[str, Any]:
    """Read reverberant pressure + normal-velocity data produced by scripts/02.

    Returns a dict with keys `inner_p`, `inner_vnz`, `outer_p`, `outer_vnz`,
    `t`, `ill_positions`, `inner_positions`, `outer_positions`, and `attrs`.
    Arrays keep the shape written by the Julia side: `(nt, n_rec, n_ill)`.

    If the file was flushed mid-run (the `n_done` attr is less than `n_ill`),
    the 3D tensors and `ill_positions` are truncated to the first `n_done`
    ill sources. The remaining slots are zero-filled placeholders left by
    greens/reverb.jl checkpoint flush — feeding them to MDD would invert noise, so
    they are dropped here and a notice is printed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reverb file not found: {path}")

    # HDF5.jl reverses dim order on write, so Julia's `(nt, n_rec, n_ill)`
    # arrays come back via h5py as `(n_ill, n_rec, nt)`. Transpose the 3D
    # tensors and 2D position arrays here so the rest of the MDD pipeline
    # (and downstream consumers) see the documented Julia shapes.
    _3D_KEYS = ("inner_p", "inner_vnz", "outer_p", "outer_vnz")
    _2D_KEYS = ("ill_positions", "inner_positions", "outer_positions")

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in (*_3D_KEYS, "t", *_2D_KEYS):
            if key not in f:
                continue
            arr = np.array(f[key])
            if key in _3D_KEYS:
                arr = arr.transpose(2, 1, 0)   # (n_ill, n_rec, nt) -> (nt, n_rec, n_ill)
            elif key in _2D_KEYS:
                arr = arr.T                    # (3, n) -> (n, 3)
            out[key] = arr
        out["attrs"] = {k: f.attrs[k] for k in f.attrs.keys()}

    # Drop zero-padded ill slots from a partial reverb checkpoint flush.
    n_ill = out[_3D_KEYS[0]].shape[2]
    n_done = int(out["attrs"].get("n_done", n_ill))
    if 0 < n_done < n_ill:
        print(f"[load_reverb] partial reverb file: n_done={n_done} of n_ill={n_ill} "
              f"sources computed — truncating to the first {n_done}.",
              file=sys.stderr)
        for k in _3D_KEYS:
            if k in out:
                out[k] = out[k][:, :, :n_done]
        if "ill_positions" in out:
            out["ill_positions"] = out["ill_positions"][:n_done, :]
    elif n_done == 0:
        raise ValueError(
            f"reverb file at {path} has n_done = 0 (no ill sources computed yet). "
            f"Wait for greens/reverb.jl to flush at least one source.")

    return out


def save_gfs(
    path: str | Path,
    *,
    p_p: np.ndarray,
    v_p: np.ndarray,
    t: np.ndarray,
    outer_positions: np.ndarray,
    inner_positions: np.ndarray,
    ill_positions: np.ndarray,
    z0: float,
    n_t_inversion: int,
    dr_pylops: float,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write MDD-extracted Green's functions to HDF5 in physical units.

    Expected shapes:
      p_p, v_p   : (nt_out, n_inner, n_outer)  float32
      t                : (nt_out,)                   float32
      outer_positions  : (n_outer, 3)                float32  → /src_positions
      inner_positions  : (n_inner, 3)                float32  → /rec_positions
      ill_positions    : (n_ill,   3)                float32

    On-disk schema and amplitude convention mirror
    `scripts/greens/{impulsive,analytical}.jl`, so the file shares the
    same Python (`compare_gfs.py`) and Julia (`load_gf_*`) loaders.

    Conversion from pylops MDD output to impulsive `p_p` convention
    -----------------------------------------------------------------
    MDD (per Wapenaar et al. 2011, GJI 185:1335-1364) inverts for the
    **dipole** Green's function (paper's eq 20):
        G̅_d(x_B, x, ω) = (-2/jωρ) · n_i ∂_i G̅(x_B, x, ω)
    while impulsive `p_p` is the response to a *monopole* pressure
    source. The two are related (free-space far-field) by:
        p_p_impulsive ≈ -(z_0 / (2 cos θ_outer)) · G̅_d
    where `cos θ_outer = n_outer · (x_inner - x_outer) / |x_inner - x_outer|`
    is per-pair geometric and `z_0 = ρc`.

    Pylops' MDC formula
        y = √n_t · Δt · Δr · Σ G_pylops · x
    differs from the paper's eq (43) discretization (`Σ G̅_d · Γ` — no
    Δx weight), so pylops' inverted G is `1/(√n_t · Δt · Δr)` times the
    paper's `G̅_d`. To recover `G̅_d`, multiply by `√n_t · Δt · Δr`.

    Combining: full conversion from pylops MDD output to impulsive `p_p`
    convention (as it lives on disk, ready for the K-H consumer):
        p_p ≈ (-z_0 / (2 cos θ_outer)) · [G_pylops · √n_t · Δt · Δr]

    Note: there are TWO downstream consumers of the saved file and they
    expect DIFFERENT Δt conventions:

      1. A one-shot K-H sum (no Δt in the sum):
         `inner_p[t] ≈ ΔS · Σ_τ G · outer_pin`. At this level, dropping
         `Δt` from pylops_norm appears to fix MDD's amplitude (a test
         reports MDD α matching impulsive's).

      2. The in-FDTD source injection used by
         `src/hologram.jl::build_hologram` with --gf-method=mdd. This
         accumulates the GF-as-source over thousands of FDTD steps; the
         per-step injection mechanics implicitly need a `Δt` baked into
         the GF (otherwise the source amplitude is `1/Δt`× too large
         per step, the FDTD destabilises, and the final-step field is
         many orders of magnitude over the analytical/impulsive
         hologram).

    Consumer #2 (the production hologram pipeline) is load-bearing —
    the K-H test diagnostic is informational only. So `Δt` STAYS in
    pylops_norm. An earlier patch that removed it (chasing the K-H
    test's α=1) made the MDD hologram come out 1e15× over-amplitude
    via FDTD instability; that change was reverted on 2026-05-12.

    Note: an earlier version of this code ALSO applied an `∂_t` based
    on a literal reading of the paper's eq 20 (1/(jωρ) factor → time
    integration). Empirically that produced shape mismatch (NCC ≈ 0
    against impulsive). Pylops' MDC operator absorbs the 1/(jω) itself,
    so no `∂_t` is needed. Separate issue from the `Δt` story above.

    `v_p` (velocity from incoming-pressure dipole source) requires its
    own analytical derivation involving inner-side geometry. Until that
    is worked out, we apply the SAME conversion to `v_p` as to `p_p`
    (using cos θ_outer); the resulting `v_p` is approximate at the
    amplitude level. Shape recovery should still be useful.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    p_p   = np.asarray(p_p,   dtype=np.float64)
    v_p = np.asarray(v_p, dtype=np.float64)
    t       = np.asarray(t,       dtype=np.float64)
    inner_positions = np.asarray(inner_positions, dtype=np.float64)
    outer_positions = np.asarray(outer_positions, dtype=np.float64)

    # ---- Per-pair geometric factor: -z_0 / (2 cos θ_outer) ----
    # Outer normals: positions are on a sphere centered at origin (Julia's
    # fibonacci_sphere produces this), so the outward normal at each outer
    # point is just the unit vector pointing from origin. If a future code
    # path uses non-spherical outer surfaces this needs to be revisited.
    outer_norms = np.linalg.norm(outer_positions, axis=1, keepdims=True)
    outer_normals = outer_positions / np.where(outer_norms > 0, outer_norms, 1.0)
    # Ray from outer to inner: shape (n_inner, n_outer, 3)
    ray = inner_positions[:, None, :] - outer_positions[None, :, :]
    r   = np.linalg.norm(ray, axis=2)               # (n_inner, n_outer)
    # cos θ_outer = n_outer · raŷ. Use einsum for the per-pair dot.
    cos_theta_outer = np.einsum("oj,ioj->io", outer_normals, ray) / np.where(r > 0, r, 1.0)
    # Guard against degenerate cos θ ≈ 0 (tangential rays). For our
    # spherical setup (inner radius < outer radius), cos θ ranges
    # from ~-1 (radial) to ~-(R_inner/R_outer) (tangential). Bounded
    # away from 0 by geometry, but clip to stay safe.
    safe_cos = np.where(np.abs(cos_theta_outer) > 1e-3, cos_theta_outer, np.sign(cos_theta_outer) * 1e-3)
    geom_factor = -float(z0) / (2.0 * safe_cos)     # (n_inner, n_outer)

    # ---- Global pylops-normalization factor: √n_t_inv · Δt · Δr ----
    # The `Δt` here is REQUIRED for the in-FDTD source injection (see
    # docstring); a previous attempt to drop it for the K-H test was
    # reverted after MDD holograms came out 1e15× over-amplitude.
    #
    # An additional empirical amplitude factor (the K-H test gives
    # `s_opt ≈ 19` for fresh MDD vs ≈ 1 for analytical) is NOT applied
    # here. Instead it lives as the `empirical_scale` HDF5 attribute,
    # initialised to 1.0 on every save_gfs call and (optionally)
    # rewritten by `diagnostics/check_gf_scale.jl --write-scale` after a
    # calibration run. `src/io.jl::load_gf_mdd` multiplies the loaded
    # tensor by this attribute at load time. This keeps the calibration
    # data-side, opt-in, and traceable per-file.
    dt_save     = float(t[1] - t[0])
    pylops_norm = float(np.sqrt(int(n_t_inversion)) * dt_save * float(dr_pylops))

    # ---- Apply conversion to bring MDD onto impulsive p_p convention ----
    # Step 1: G̅_d (paper convention) = G_pylops · pylops_norm
    # Step 2: per-pair geom_factor (broadcast over time axis)
    # No ∂_t — see docstring for why pylops' MDC absorbs the 1/(jω) itself.
    p_p_phys = (p_p * pylops_norm) * geom_factor[None, :, :]
    # Same conversion applied to v_p (approximate — see docstring).
    # Note: v_p is z0-multiplied normal velocity from MDD, so we
    # divide by z0 first to get the natural velocity-like quantity, then
    # apply the dipole→monopole conversion as for p.
    v_p_phys = ((v_p / float(z0)) * pylops_norm) * geom_factor[None, :, :]

    p_p = p_p_phys.astype(np.float32)
    v_p = v_p_phys.astype(np.float32)

    # Match Julia's on-disk layout so the shared loaders can use the
    # same `transpose(2,1,0)` / position `.T` rules for both Julia- and
    # Python-written GF files. Julia HDF5.jl writes column-major, so a
    # Julia tensor of shape (nt, n_inner, n_outer) appears in h5py as
    # raw shape
    # (n_outer, n_inner, nt). We mirror that by writing the canonical
    # (nt, n_inner, n_outer) tensor with axes reversed; same for the
    # (N, 3) → (3, N) position arrays.
    with h5py.File(path, "w") as f:
        f.create_dataset("p_p",            data=p_p.transpose(2, 1, 0))
        f.create_dataset("v_p",            data=v_p.transpose(2, 1, 0))
        f.create_dataset("t",              data=t.astype(np.float32))
        f.create_dataset("src_positions",  data=outer_positions.astype(np.float32).T)
        f.create_dataset("rec_positions",  data=inner_positions.astype(np.float32).T)
        f.create_dataset("ill_positions",  data=ill_positions.astype(np.float32).T)
        # Empirical amplitude calibration: 1.0 on fresh extraction.
        # `diagnostics/check_gf_scale.jl --write-scale` can update this; load_gf_mdd
        # multiplies the loaded tensor by this value at load time.
        f.attrs["empirical_scale"] = 1.0
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
