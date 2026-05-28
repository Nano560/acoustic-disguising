"""End-to-end smoke test for `mdd.mdd_extract`.

Builds a tiny synthetic reverb dict, runs the normal-incidence-separation
MDD path (no SHT / local_pw — those need real spherical geometry), and
checks that the extracted Green's functions have the documented shape and
finite values.

Bigger-physics validation lives in the development diagnostics suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdd import MDDParams, mdd_extract


def _synthetic_reverb(*, nt: int = 32, n_inner: int = 6, n_outer: int = 6,
                      n_ill: int = 4, dt: float = 1.0e-5) -> dict:
    """Synthetic reverb dict matching `io.load_reverb`'s output shape.
    All four 3D tensors are (nt, n_rec, n_ill); positions are (N, 3)."""
    rng = np.random.default_rng(7)
    t = np.arange(nt, dtype=np.float32) * np.float32(dt)
    return {
        "inner_p":          rng.standard_normal((nt, n_inner, n_ill)).astype(np.float32),
        "inner_vnz":        rng.standard_normal((nt, n_inner, n_ill)).astype(np.float32),
        "outer_p":          rng.standard_normal((nt, n_outer, n_ill)).astype(np.float32),
        "outer_vnz":        rng.standard_normal((nt, n_outer, n_ill)).astype(np.float32),
        "t":                t,
        "inner_positions":  rng.standard_normal((n_inner, 3)).astype(np.float32),
        "outer_positions":  rng.standard_normal((n_outer, 3)).astype(np.float32),
        "ill_positions":    rng.standard_normal((n_ill,   3)).astype(np.float32),
        "attrs": {
            "dom_dt":        dt,
            "radius_inner":  0.10,
            "radius_outer":  0.20,
            "nPoints_inner": n_inner,
            "nPoints_outer": n_outer,
            "n_ill":         n_ill,
        },
    }


def test_mdd_extract_smoke() -> None:
    """mdd_extract returns p_p / v_p tensors of the documented shape
    `(nt_out, n_inner, n_outer)` (NOT n_ill — the inverted GF is independent
    of the ill geometry by construction) with finite values."""
    reverb = _synthetic_reverb()
    params = MDDParams(
        fmax_hz=20_000.0,
        iter_lim=3,                 # fast for CI; not converged
        damp=1.0e-4,
        tmax_out=2.0e-4,
        separation_method="normal_incidence",
    )

    gf = mdd_extract(reverb, params=params)

    # Documented output keys.
    assert {"p_p", "v_p", "t"}.issubset(set(gf.keys()))

    nt_out  = gf["t"].size
    n_inner = reverb["inner_p"].shape[1]
    n_outer = reverb["outer_p"].shape[1]

    for k in ("p_p", "v_p"):
        assert gf[k].shape == (nt_out, n_inner, n_outer), (
            f"{k} has wrong shape {gf[k].shape}, "
            f"expected {(nt_out, n_inner, n_outer)}")
        assert np.all(np.isfinite(gf[k])), f"{k} contains NaN/Inf"

    # tmax_out should cap the time axis.
    dt = float(reverb["t"][1] - reverb["t"][0])
    assert nt_out <= int(round(params.tmax_out / dt)) + 1


def test_mdd_extract_rejects_mismatched_shapes() -> None:
    """Inner / outer surface tensors must share nt and n_ill; mismatch
    is caught with a clear ValueError, not propagated to pylops."""
    reverb = _synthetic_reverb()
    # Break inner_p / inner_vnz consistency.
    reverb["inner_p"] = reverb["inner_p"][:, :3, :]
    with pytest.raises(ValueError):
        mdd_extract(reverb,
                    params=MDDParams(separation_method="normal_incidence",
                                     iter_lim=2, fmax_hz=10_000.0))


def test_mdd_params_from_cfg() -> None:
    """MDDParams.from_cfg picks up the relevant `[mdd]` keys."""
    cfg = {
        "mdd": {
            "fmax_hz":           50_000.0,
            "iter_lim":          7,
            "damp":              5.0e-5,
            "tmax_out":          1.0e-3,
            "separation_method": "normal_incidence",
        }
    }
    params = MDDParams.from_cfg(cfg)
    assert params.fmax_hz           == pytest.approx(50_000.0)
    assert params.iter_lim          == 7
    assert params.damp              == pytest.approx(5.0e-5)
    assert params.tmax_out          == pytest.approx(1.0e-3)
    assert params.separation_method == "normal_incidence"


def test_mdd_params_rejects_unknown_method() -> None:
    """Unknown separation_method values fail-fast at cfg time."""
    cfg = {"mdd": {"separation_method": "nonsense"}}
    with pytest.raises(ValueError):
        MDDParams.from_cfg(cfg)
