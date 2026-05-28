"""Round-trip tests for the HDF5 schema bridging Julia (reverb stage) and
Python (MDD stage).

These tests don't require the FDTD pipeline — they construct synthetic
arrays in the documented shapes, write them through `save_gfs` /
`load_reverb`, and verify the schema round-trips correctly.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from mdd import io


# ---------------------------------------------------------------------------
# save_gfs round-trip
# ---------------------------------------------------------------------------

def _save_gfs_kwargs(nt_out: int = 7, n_inner: int = 5,
                     n_outer: int = 4, n_ill: int = 3) -> dict:
    """Build a synthetic kwargs bundle for `save_gfs` with all required args.

    Shapes mirror what `mdd_extract` produces:
      - p_p / v_p : (nt_out, n_inner, n_outer)
      - inner / outer / ill positions : (N, 3)
    """
    rng = np.random.default_rng(42)
    return dict(
        p_p           = rng.standard_normal((nt_out, n_inner, n_outer)).astype(np.float32),
        v_p         = rng.standard_normal((nt_out, n_inner, n_outer)).astype(np.float32),
        t               = np.linspace(0.0, 1.0, nt_out, dtype=np.float32),
        outer_positions = rng.standard_normal((n_outer, 3)).astype(np.float32),
        inner_positions = rng.standard_normal((n_inner, 3)).astype(np.float32),
        ill_positions   = rng.standard_normal((n_ill,   3)).astype(np.float32),
        z0              = 1.5e6,                  # ρ·c for water
        n_t_inversion   = nt_out,
        dr_pylops       = 0.05,
    )


def test_save_gfs_roundtrip(tmp_path: Path) -> None:
    """save_gfs writes the canonical Julia-compatible schema: p_p / v_p
    tensors with axes reversed from canonical (matches HDF5.jl), positions
    transposed to (3, N), and attrs preserved."""
    kwargs = _save_gfs_kwargs(nt_out=7, n_inner=5, n_outer=4, n_ill=3)
    attrs = {
        "config_file":      "configs/quickstart.toml",
        "mdd_fmax_hz":      60_000.0,
        "mdd_iter_lim":     5,
        "mdd_damp":         1.0e-4,
        "dom_dt":           1.0e-6,
        "radius_inner":     0.15,
        "radius_outer":     0.25,
        "nPoints_inner":    5,
        "nPoints_outer":    4,
        "scatterer":        "none",
        "units_convention": "physical",
    }

    out_path = tmp_path / "gfs.h5"
    io.save_gfs(out_path, **kwargs, attrs=attrs)
    assert out_path.is_file()

    with h5py.File(out_path, "r") as f:
        # Datasets present.
        assert {"p_p", "v_p", "t", "src_positions", "rec_positions",
                "ill_positions"}.issubset(set(f.keys()))

        # Axis convention: 3D tensors written with axes reversed so that
        # Julia HDF5.jl reads them as (nt, n_inner, n_outer). From Python
        # we see them as (n_outer, n_inner, nt).
        nt_out, n_inner, n_outer = kwargs["p_p"].shape
        assert f["p_p"].shape == (n_outer, n_inner, nt_out)
        assert f["v_p"].shape == (n_outer, n_inner, nt_out)

        # Positions transposed to (3, N).
        assert f["src_positions"].shape == (3, n_outer)
        assert f["rec_positions"].shape == (3, n_inner)
        assert f["ill_positions"].shape == (3, kwargs["ill_positions"].shape[0])

        # t passes through.
        np.testing.assert_array_equal(np.array(f["t"]), kwargs["t"])

        # User attrs preserved.
        assert dict(f.attrs)["config_file"]    == attrs["config_file"]
        assert int(f.attrs["mdd_iter_lim"])    == attrs["mdd_iter_lim"]
        assert float(f.attrs["mdd_fmax_hz"])   == pytest.approx(attrs["mdd_fmax_hz"])
        assert dict(f.attrs)["scatterer"]      == "none"


def test_save_gfs_creates_parent(tmp_path: Path) -> None:
    """save_gfs creates intermediate directories if they don't exist."""
    out_path = tmp_path / "deeper" / "tree" / "gfs.h5"
    io.save_gfs(out_path, **_save_gfs_kwargs(nt_out=4, n_inner=3,
                                              n_outer=2, n_ill=2))
    assert out_path.is_file()


def test_save_gfs_dtypes(tmp_path: Path) -> None:
    """Saved tensors are float32 regardless of input dtype (h5 schema
    declares float32 for compactness + Julia interop)."""
    kwargs = _save_gfs_kwargs(nt_out=4, n_inner=3, n_outer=2, n_ill=2)
    # Promote to float64 inputs.
    kwargs["p_p"]   = kwargs["p_p"].astype(np.float64)
    kwargs["v_p"] = kwargs["v_p"].astype(np.float64)
    out_path = tmp_path / "gfs.h5"
    io.save_gfs(out_path, **kwargs)
    with h5py.File(out_path, "r") as f:
        assert f["p_p"].dtype == np.float32
        assert f["v_p"].dtype == np.float32
        assert f["t"].dtype   == np.float32


# ---------------------------------------------------------------------------
# load_reverb round-trip
# ---------------------------------------------------------------------------

def _write_synthetic_reverb(path: Path, *, nt: int = 6, n_inner: int = 5,
                            n_outer: int = 7, n_ill: int = 3) -> dict:
    """Write a tiny reverb HDF5 in Julia's on-disk axis layout (axes
    reversed from canonical) and return the canonical-form arrays for
    cross-checking after load_reverb's transposes."""
    rng = np.random.default_rng(0)
    inner_p   = rng.standard_normal((nt, n_inner, n_ill)).astype(np.float32)
    inner_vnz = rng.standard_normal((nt, n_inner, n_ill)).astype(np.float32)
    outer_p   = rng.standard_normal((nt, n_outer, n_ill)).astype(np.float32)
    outer_vnz = rng.standard_normal((nt, n_outer, n_ill)).astype(np.float32)
    t         = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    inner_pos = rng.standard_normal((n_inner, 3)).astype(np.float32)
    outer_pos = rng.standard_normal((n_outer, 3)).astype(np.float32)
    ill_pos   = rng.standard_normal((n_ill,   3)).astype(np.float32)

    with h5py.File(path, "w") as f:
        # Mirror Julia HDF5.jl: tensor shape (nt, n_rec, n_ill) → on-disk
        # (n_ill, n_rec, nt); positions (N, 3) → (3, N).
        f.create_dataset("inner_p",         data=inner_p.transpose(2, 1, 0))
        f.create_dataset("inner_vnz",       data=inner_vnz.transpose(2, 1, 0))
        f.create_dataset("outer_p",         data=outer_p.transpose(2, 1, 0))
        f.create_dataset("outer_vnz",       data=outer_vnz.transpose(2, 1, 0))
        f.create_dataset("t",               data=t)
        f.create_dataset("inner_positions", data=inner_pos.T)
        f.create_dataset("outer_positions", data=outer_pos.T)
        f.create_dataset("ill_positions",   data=ill_pos.T)
        f.attrs["dom_dt"]        = 1.0e-6
        f.attrs["nPoints_inner"] = n_inner
        f.attrs["nPoints_outer"] = n_outer
        f.attrs["radius_inner"]  = 0.10
        f.attrs["radius_outer"]  = 0.20
        f.attrs["n_done"]        = n_ill   # mark all ill sources populated

    return dict(inner_p=inner_p, inner_vnz=inner_vnz,
                outer_p=outer_p, outer_vnz=outer_vnz, t=t,
                inner_positions=inner_pos, outer_positions=outer_pos,
                ill_positions=ill_pos)


def test_load_reverb_restores_julia_axes(tmp_path: Path) -> None:
    """load_reverb undoes Julia's HDF5.jl axis reversal: 3D tensors come
    back as (nt, n_rec, n_ill); positions as (N, 3)."""
    path = tmp_path / "reverb.h5"
    expected = _write_synthetic_reverb(path)

    out = io.load_reverb(path)

    for k in ("inner_p", "inner_vnz", "outer_p", "outer_vnz"):
        np.testing.assert_array_equal(out[k], expected[k])

    for k in ("inner_positions", "outer_positions", "ill_positions"):
        np.testing.assert_array_equal(out[k], expected[k])
        assert out[k].shape[1] == 3

    np.testing.assert_array_equal(out["t"], expected["t"])

    assert float(out["attrs"]["dom_dt"])         == pytest.approx(1.0e-6)
    assert int(out["attrs"]["nPoints_inner"])    == 5
    assert int(out["attrs"]["nPoints_outer"])    == 7


def test_load_reverb_missing_file(tmp_path: Path) -> None:
    """load_reverb errors clearly on a missing path."""
    with pytest.raises(FileNotFoundError):
        io.load_reverb(tmp_path / "does_not_exist.h5")
