"""Multi-Dimensional Deconvolution core algorithm.

Uses `pylops.waveeqprocessing.MDD` to recover Green's functions from the
reverberant illumination pressure and normal-velocity recordings produced
by `scripts/greens/reverb.jl`.

Algorithm sketch:

    1. Synthesise the INCOMING pressure on the outer surface:

           outer_pin = (outer_p - outer_vnz) / 2

       (decomposition into one-way components; `outer_vnz` has already been
       multiplied by the impedance z0 on the Julia side, so this is the
       standard p+/- = (p ± z0·vn)/2 split.)

    2. For each inner-surface receiver channel type f ∈ (p, vnz) run

           pylops.waveeqprocessing.MDD(
               incident  = outer_pin,       # (n_ill, n_rec, nt) after permute
               recording = inner_<f>,       # (n_ill, n_rec, nt)
               dt = dt, dr = 2π·r_inner / n_inner, …
           )

       to solve the matrix equation `recording = incident ⊗ G` for G, which
       is the desired Green's function `p_p` (returned in the dict below).

    3. Truncate in time to `tmax_out`, then return the dict
       `{'p_p': …, 'v_p': …, 't': …}` in `(nt_out, n_inner, n_outer)`
       shape — ready to be written through `io.save_gfs`.

    The output's third axis is **n_outer** (not n_ill): pylops.MDD inverts
    `recording = G ⊗ incident` for the kernel `G(outer → inner)`, so the
    inverted model has shape `(n_outer, n_inner, nt)` (per pylops docstring:
    "Inverted model of size [n_r × n_vs × n_t]" with n_r = n_outer here).
    More illumination sources improve conditioning but the resulting GF is,
    by construction, independent of where the ill sources sat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pylops
import scipy.fft as _spfft

from . import wavefield_separation as wfs
from . import wavefield_separation_local as wfs_local


# pylops.MDD takes kernel shape (n_s, n_r, nt) and data (n_s, n_vs, nt), so
# the Julia HDF5 tensors of shape (nt, n_rec, n_ill) need their axes
# reversed: (2, 1, 0) → (n_ill, n_rec, nt). Used for both the input permute
# (Julia → pylops) and the output permute (pylops → save_gfs canonical
# shape (nt, n_inner, n_outer)).
_PERM = (2, 1, 0)

# pylops's `MDD` defaults to fftengine="numpy" (single-threaded, always
# returns complex128 + downcasts → spammy warning). scipy.fft handles
# complex64 natively and threads via `set_workers`. Benchmarked at
# production scale (300×300×400, 400 ill): scipy + 4 workers gives ~1.33×
# wall-time speedup vs numpy; 10 workers plateaus at 1.34× — Fredholm
# matvec / LSQR scalar work bounds the rest. 4 is the sweet spot.
_FFT_ENGINE  = "scipy"
_FFT_WORKERS = 4


@dataclass
class MDDParams:
    """Parameters controlling the MDD inversion."""

    fmax_hz: float = 30_000.0      # frequency-band cap; `from_cfg` derives it
                                   # from [reverb].bandwidth_factor · fc_source
                                   # (the recorded Ricker band) — this literal
                                   # is only the no-cfg fallback
    iter_lim: int  = 50            # LSQR iteration cap
    damp: float    = 1.0e-4        # Tikhonov damping
    atol: float    = 1.0e-12
    btol: float    = 1.0e-12
    tmax_out: float = 3.0e-3       # truncate output GFs in time [s]

    # Wave-field separation on the outer shell:
    #   "normal_incidence" — high-frequency / normal-incidence approximation
    #                p_in ≈ (p − z₀·v_n) / 2  (uses outer p + outer v_n).
    #                Originally called "legacy" — the simplest and oldest
    #                method, exact only for waves arriving normal to the
    #                outer shell.
    #   "sht"      — modally exact via spherical-harmonic projection +
    #                closed-form radial Hankel split (uses outer p + outer
    #                v_n; bandwidth-limited by full-shell angular Nyquist).
    #   "local_pw" — local plane-wave decomposition over a tangent-plane
    #                patch with two-shell support (uses outer p + inner p).
    #                Bandwidth-limited only by per-patch sampling, so it
    #                works above the global SHT angular Nyquist.
    #   "local_pvn"— same local plane-wave framework but using outer p +
    #                outer z₀·v_n at every patch point. Same bandwidth
    #                advantage as local_pw, plus avoids the two-shell
    #                spectral comb at f = n·c₀/(2·Δr) and the 1/r
    #                amplitude mismatch between shells. Reduces exactly to
    #                the normal-incidence formula when n_pw = 1.
    separation_method: str = "sht"
    sht_params: wfs.SHTSeparationParams = field(
        default_factory=wfs.SHTSeparationParams)
    local_pw_params: wfs_local.LocalPWSeparationParams = field(
        default_factory=wfs_local.LocalPWSeparationParams)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "MDDParams":
        mdd = cfg.get("mdd", {})
        method = str(mdd.get("separation_method", cls.separation_method)).lower()
        if method not in ("normal_incidence", "sht", "local_pw", "local_pvn"):
            raise ValueError(
                f"[mdd].separation_method must be one of 'normal_incidence', "
                f"'sht', 'local_pw', 'local_pvn'; got {method!r}")
        sht = wfs.SHTSeparationParams.from_cfg(cfg)
        lpw = wfs_local.LocalPWSeparationParams.from_cfg(cfg)
        # fmax_hz defaults to [reverb].bandwidth_factor · fc_source — the
        # Ricker band the reverb recording covers (= its Nyquist, fs_out/2).
        # The MDD inversion cannot use frequencies the data does not contain.
        # An explicit [mdd].fmax_hz overrides this (e.g. to cap lower for
        # speed when lower-frequency content alone suffices).
        rev = cfg.get("reverb", {})
        bw, fc = rev.get("bandwidth_factor"), rev.get("fc_source")
        fmax_default = (float(bw) * float(fc)
                        if bw is not None and fc is not None else cls.fmax_hz)
        return cls(
            fmax_hz           = float(mdd.get("fmax_hz",  fmax_default)),
            iter_lim          = int  (mdd.get("iter_lim", cls.iter_lim)),
            damp              = float(mdd.get("damp",     cls.damp)),
            atol              = float(mdd.get("atol",     cls.atol)),
            btol              = float(mdd.get("btol",     cls.btol)),
            tmax_out          = float(mdd.get("tmax_out", cls.tmax_out)),
            separation_method = method,
            sht_params        = sht,
            local_pw_params   = lpw,
        )


def replace_sht_fmax(p: wfs.SHTSeparationParams, fmax_hz: float
                     ) -> wfs.SHTSeparationParams:
    """Return a copy of `p` with `fmax_hz` overridden."""
    return wfs.SHTSeparationParams(
        n_max=p.n_max, reg=p.reg, c0=p.c0, fmax_hz=fmax_hz, beta_max=p.beta_max)


def replace_sht_c0(p: wfs.SHTSeparationParams, c0: float
                   ) -> wfs.SHTSeparationParams:
    """Return a copy of `p` with `c0` overridden."""
    return wfs.SHTSeparationParams(
        n_max=p.n_max, reg=p.reg, c0=c0, fmax_hz=p.fmax_hz, beta_max=p.beta_max)


def _idw_resample_axis1(values: np.ndarray, src_pos: np.ndarray,
                        dst_pos: np.ndarray, k: int = 4) -> np.ndarray:
    """Inverse-distance-weighted interpolation along axis 1 (receivers).

    `values` has shape (nt, n_src, n_ill); returns (nt, n_dst, n_ill). The
    K nearest neighbours on the source sphere are weighted by 1/d² (chord
    distance is fine; both source and dest live on the same sphere). Used
    to map dense-recording reverb data onto canonical Fibonacci(N_prod)
    positions.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(src_pos)
    dists, idx = tree.query(dst_pos, k=k)
    weights = 1.0 / (dists ** 2 + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)        # (n_dst, k)
    gathered = values[:, idx, :]                          # (nt, n_dst, k, n_ill)
    return (gathered * weights[None, :, :, None]).sum(axis=2)


def fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    """Mirror src/illumination.jl::fibonacci_sphere(n, radius). Returns
    `(n, 3)` Float64 points on the sphere of given radius. Used to generate
    the canonical production layout that `_maybe_resample_to_production`
    targets when the recording was done at a denser receiver count."""
    i = np.arange(n)
    z = 1.0 - (2.0 * i + 1.0) / n
    r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    pts = np.empty((n, 3), dtype=np.float64)
    pts[:, 0] = radius * r_xy * np.cos(theta)
    pts[:, 1] = radius * r_xy * np.sin(theta)
    pts[:, 2] = radius * z
    return pts


def _maybe_resample_to_production(
    reverb: dict, *,
    n_inner_prod: int | None,
    n_outer_prod: int | None,
    verbose: bool = True,
) -> dict:
    """IDW-interpolate the four 3D tensors and position arrays onto the
    canonical Fibonacci(N_prod) sphere. Mutates a shallow copy of the dict
    and returns it. `None` on either count means "keep recorded layout";
    when both equal the recorded counts this is a no-op.

    The production-layout target positions are generated on the fly via
    `fibonacci_sphere(n_*_prod, radius_*)` — radius comes from the reverb
    file's attrs (intrinsic to the recording geometry). The reverb file
    itself no longer carries the prod count or canonical positions; that's
    a downstream choice made by the caller from the cfg.
    """
    attrs = reverb.get("attrs", {})
    n_outer_rec = int(attrs.get("nPoints_outer", reverb["outer_p"].shape[1]))
    n_inner_rec = int(attrs.get("nPoints_inner", reverb["inner_p"].shape[1]))
    n_outer_prod = n_outer_rec if n_outer_prod is None else int(n_outer_prod)
    n_inner_prod = n_inner_rec if n_inner_prod is None else int(n_inner_prod)
    if n_outer_rec == n_outer_prod and n_inner_rec == n_inner_prod:
        return reverb   # nothing to do

    if "radius_inner" not in attrs or "radius_outer" not in attrs:
        raise KeyError(
            "Resampling to a sparser production layout requires `radius_inner` "
            "and `radius_outer` in the reverb file's attrs (used to generate "
            "the Fibonacci target sphere). Both should be written by "
            "scripts/greens/reverb.jl::_build_reverb_attrs.")
    r_inner = float(attrs["radius_inner"])
    r_outer = float(attrs["radius_outer"])

    out = dict(reverb)
    if n_outer_rec != n_outer_prod:
        canonical_outer = fibonacci_sphere(n_outer_prod, r_outer)
        if verbose:
            print(f"[mdd] resampling outer recordings: {n_outer_rec} → {n_outer_prod} "
                  f"(IDW, k=4)", flush=True)
        out["outer_p"]   = _idw_resample_axis1(reverb["outer_p"],   reverb["outer_positions"], canonical_outer)
        out["outer_vnz"] = _idw_resample_axis1(reverb["outer_vnz"], reverb["outer_positions"], canonical_outer)
        out["outer_positions"] = canonical_outer
    if n_inner_rec != n_inner_prod:
        canonical_inner = fibonacci_sphere(n_inner_prod, r_inner)
        if verbose:
            print(f"[mdd] resampling inner recordings: {n_inner_rec} → {n_inner_prod} "
                  f"(IDW, k=4)", flush=True)
        out["inner_p"]   = _idw_resample_axis1(reverb["inner_p"],   reverb["inner_positions"], canonical_inner)
        out["inner_vnz"] = _idw_resample_axis1(reverb["inner_vnz"], reverb["inner_positions"], canonical_inner)
        out["inner_positions"] = canonical_inner

    # Surface attrs the rest of the pipeline reads — patch them to match.
    new_attrs = dict(attrs)
    new_attrs["nPoints_outer"] = n_outer_prod
    new_attrs["nPoints_inner"] = n_inner_prod
    out["attrs"] = new_attrs
    return out


def _decompose_outer_pin(outer_p: np.ndarray, outer_vnz: np.ndarray) -> np.ndarray:
    """Legacy normal-incidence approximation of the incoming pressure.

    `outer_vnz` on the Julia side is already multiplied by the impedance z0,
    so the standard plane-wave decomposition `p± = (p ± z0·vn)/2` reduces to
    the plain `(outer_p − outer_vnz)/2` for the incoming component. This is
    exact only for waves that are normally incident on the shell (kr → ∞);
    `wavefield_separation.decompose_outer_pin_sht` is the modally exact
    upgrade for moderate kr / curved wavefronts.
    """
    return (outer_p - outer_vnz) / 2.0


def _run_mdd(
    incident: np.ndarray,      # (n_ill, n_rec, nt)
    recording: np.ndarray,     # (n_ill, n_rec, nt)
    *,
    dt: float,
    dr: float,
    params: MDDParams,
) -> np.ndarray:
    """Single pylops.MDD call → Green's-function tensor shaped (nt, n_inner, n_outer).

    pylops.MDD signature: kernel `G` of size [n_s × n_r × n_t] and data `d`
    of size [n_s × n_vs × n_t] → inverted model of size [n_r × n_vs × n_t].
    Here n_s = n_ill, n_r = n_outer, n_vs = n_inner, so the returned tensor
    is (n_outer, n_inner, nt). The final transpose([2, 1, 0]) flips it to
    the canonical (nt, n_inner, n_outer) save shape.

    `nfmax` (maximum frequency bin) is computed from `params.fmax_hz`:

        df     = 1 / (nt * dt)
        nfmax  = round(fmax_hz / df) = round(fmax_hz * nt * dt)

    matching the notebook's `nfmax = round(100e3 / df)` construction.
    """
    nt = incident.shape[-1]
    df = 1.0 / (nt * dt)
    nfmax = max(1, int(round(params.fmax_hz / df)))

    with _spfft.set_workers(_FFT_WORKERS):
        g = pylops.waveeqprocessing.MDD(
            incident,
            recording,
            dt=dt,
            dr=dr,
            nfmax=nfmax,
            wav=None,
            twosided=False,
            add_negative=False,
            causality_precond=True,
            dottest=False,
            damp=params.damp,
            iter_lim=params.iter_lim,
            atol=params.atol,
            btol=params.btol,
            fftengine=_FFT_ENGINE,
            show=1,
        )
    if g.ndim == 2:
        # pylops returns (n_outer, nt) when n_vs=1; restore the (n_vs=1, ...) axis.
        g = np.expand_dims(g, axis=1)

    # Permute (n_outer, n_inner, nt) → (nt, n_inner, n_outer) for saving
    # alongside the impulsive/analytical files (same canonical axis order).
    return g.transpose(_PERM)


def mdd_extract(
    reverb: dict,
    *,
    params: MDDParams,
    dr: float | None = None,
    fields: tuple[str, ...] = ("p_p", "v_p"),
    n_inner_prod: int | None = None,
    n_outer_prod: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract Green's functions from a reverb data dict (see io.load_reverb).

    `dr` is the spatial-quadrature weight pylops.MDD applies along its
    `n_r` summation axis (= our outer axis). Per its docstring,
    `y = √n_t · Δt · Δr · Σ_{i_r} G ⊗ x`, so `dr` is what discretizes the
    K-H surface integral over outer points. Default: `4π·radius_outer² /
    nPoints_outer`, the per-point area on the outer Fibonacci sphere
    (`ΔS_out`). The inner-radius-based default the seismic notebooks use
    (`2π·radius_inner / nPoints_inner`) is a 1-D line-array convention
    that doesn't reflect spherical surface geometry; swapping to ΔS_out
    brings amplitudes onto the physical K-H quadrature scale.

    `n_inner_prod` / `n_outer_prod`: production layout the GF should be
    inverted onto. `None` (the default) means "use whatever's in the reverb
    dict" (no resampling). The cli reads these from `cfg["surfaces"].nPoints_*`
    so the production layout is a downstream choice, not a property of the
    recording.

    Returns a dict with the requested `fields` (`p_p` / `v_p`), the
    time axis `t`, and the post-resampling `inner_positions` / `outer_positions`
    so the cli can save them without re-deriving the Fibonacci layout.
    """
    # If greens/reverb.jl recorded at denser N than the requested production
    # layout, IDW-interpolate the four 3D tensors + position arrays onto
    # canonical Fibonacci(N_prod). No-op when prod counts == recorded counts.
    reverb = _maybe_resample_to_production(
        reverb, n_inner_prod=n_inner_prod, n_outer_prod=n_outer_prod)

    inner_p    = reverb["inner_p"]
    inner_vnz  = reverb["inner_vnz"]
    outer_p    = reverb["outer_p"]
    outer_vnz  = reverb["outer_vnz"]
    t          = reverb["t"]

    # Sanity check shapes. Each surface's p / vnz must match each other
    # (same Fibonacci layout). Inner and outer must share nt (axis 0) and
    # n_ill (axis 2), but their middle axis (n_rec) is independent —
    # pylops.MDD's kernel uses n_r (= n_outer) and the data uses n_vs (= n_inner)
    # as separate axes, so e.g. an n_outer sweep can vary outer alone.
    if inner_p.shape != inner_vnz.shape:
        raise ValueError(
            f"inner_p / inner_vnz shape mismatch: {inner_p.shape} vs {inner_vnz.shape}")
    if outer_p.shape != outer_vnz.shape:
        raise ValueError(
            f"outer_p / outer_vnz shape mismatch: {outer_p.shape} vs {outer_vnz.shape}")
    if (inner_p.shape[0] != outer_p.shape[0]
            or inner_p.shape[2] != outer_p.shape[2]):
        raise ValueError(
            "inner / outer must share nt (axis 0) and n_ill (axis 2); "
            f"got inner {inner_p.shape}, outer {outer_p.shape}")

    nt = int(t.size)
    dt = float(t[1] - t[0])

    if dr is None:
        attrs = reverb.get("attrs", {})
        r_in = float(attrs.get("radius_inner", 0.2))
        n_in = int  (attrs.get("nPoints_inner", inner_p.shape[1]))
        dr = 2.0 * np.pi * r_in / n_in

    # (1) Decompose outer into incoming pressure.
    if params.separation_method == "sht":
        attrs = reverb.get("attrs", {})
        r_outer = float(attrs.get("radius_outer", 0.3))
        outer_positions = reverb.get("outer_positions")
        if outer_positions is None:
            raise KeyError(
                "reverb dict has no `outer_positions` — required by the SHT "
                "wave-field separation. Either set [mdd].separation_method = "
                "'normal_incidence' or rerun greens/reverb.jl to write outer_positions.")
        # Inherit fmax_hz from MDD if the SHT block didn't override it.
        sht_params = params.sht_params
        if sht_params.fmax_hz is None:
            sht_params = replace_sht_fmax(sht_params, params.fmax_hz)
        # Allow c0 from reverb attrs to override the SHT default.
        c0_attr = attrs.get("dom_c0")
        if c0_attr is not None:
            sht_params = replace_sht_c0(sht_params, float(c0_attr))
        print(f"[mdd] separation: SHT (n_max={sht_params.n_max}, "
              f"reg={sht_params.reg:.1g}, c0={sht_params.c0:.1g} m/s, "
              f"r_outer={r_outer:.3g} m)", flush=True)
        outer_pin = wfs.decompose_outer_pin_sht(
            outer_p, outer_vnz, outer_positions,
            radius=r_outer, dt=dt, params=sht_params, verbose=True,
        )
    elif params.separation_method == "local_pw":
        attrs = reverb.get("attrs", {})
        outer_positions = reverb.get("outer_positions")
        inner_positions = reverb.get("inner_positions")
        if outer_positions is None or inner_positions is None:
            raise KeyError(
                "reverb dict needs both `outer_positions` and "
                "`inner_positions` for local_pw separation. Either set "
                "[mdd].separation_method to 'normal_incidence' or rerun greens/reverb.jl.")
        lpw_params = params.local_pw_params
        if lpw_params.fmax_hz is None:
            lpw_params = wfs_local.LocalPWSeparationParams(
                patch_size=lpw_params.patch_size,
                n_pw_radial=lpw_params.n_pw_radial,
                n_pw_azimuth=lpw_params.n_pw_azimuth,
                reg=lpw_params.reg, c0=lpw_params.c0,
                fmax_hz=params.fmax_hz, k_cap=lpw_params.k_cap)
        c0_attr = attrs.get("dom_c0")
        if c0_attr is not None:
            lpw_params = wfs_local.LocalPWSeparationParams(
                patch_size=lpw_params.patch_size,
                n_pw_radial=lpw_params.n_pw_radial,
                n_pw_azimuth=lpw_params.n_pw_azimuth,
                reg=lpw_params.reg, c0=float(c0_attr),
                fmax_hz=lpw_params.fmax_hz, k_cap=lpw_params.k_cap)
        print(
            f"[mdd] separation: local_pw (K={lpw_params.patch_size}, "
            f"n_pw=({lpw_params.n_pw_radial}r,{lpw_params.n_pw_azimuth}a), "
            f"reg={lpw_params.reg:.1g}, c0={lpw_params.c0:.0g} m/s)",
            flush=True)
        outer_pin = wfs_local.decompose_outer_pin_local_pw(
            outer_p, outer_positions, inner_p, inner_positions,
            dt=dt, params=lpw_params, verbose=True,
        )
    elif params.separation_method == "local_pvn":
        attrs = reverb.get("attrs", {})
        outer_positions = reverb.get("outer_positions")
        if outer_positions is None:
            raise KeyError("reverb dict has no `outer_positions`")
        lpw_params = params.local_pw_params
        if lpw_params.fmax_hz is None:
            lpw_params = wfs_local.LocalPWSeparationParams(
                patch_size=lpw_params.patch_size,
                n_pw_radial=lpw_params.n_pw_radial,
                n_pw_azimuth=lpw_params.n_pw_azimuth,
                reg=lpw_params.reg, c0=lpw_params.c0,
                fmax_hz=params.fmax_hz, k_cap=lpw_params.k_cap)
        c0_attr = attrs.get("dom_c0")
        if c0_attr is not None:
            lpw_params = wfs_local.LocalPWSeparationParams(
                patch_size=lpw_params.patch_size,
                n_pw_radial=lpw_params.n_pw_radial,
                n_pw_azimuth=lpw_params.n_pw_azimuth,
                reg=lpw_params.reg, c0=float(c0_attr),
                fmax_hz=lpw_params.fmax_hz, k_cap=lpw_params.k_cap)
        print(
            f"[mdd] separation: local_pvn (K={lpw_params.patch_size}, "
            f"n_pw=({lpw_params.n_pw_radial}r,{lpw_params.n_pw_azimuth}a), "
            f"reg={lpw_params.reg:.1g}, c0={lpw_params.c0:.0g} m/s)",
            flush=True)
        outer_pin = wfs_local.decompose_outer_pin_local_pvn(
            outer_p, outer_vnz, outer_positions,
            dt=dt, params=lpw_params, verbose=True,
        )
    elif params.separation_method == "normal_incidence":
        print("[mdd] separation: normal_incidence ((p − z₀·v_n)/2)",
              flush=True)
        outer_pin = _decompose_outer_pin(outer_p, outer_vnz)
    else:
        raise ValueError(
            f"unknown separation_method {params.separation_method!r}")

    # (2) Permute to (n_ill, n_rec, nt) for pylops.
    incident_perm = outer_pin.transpose(_PERM)

    valid_fields = ("p_p", "v_p")
    field_recording = {"p_p": inner_p, "v_p": inner_vnz}
    for f in fields:
        if f not in valid_fields:
            raise ValueError(f"unknown field {f!r}; must be in {valid_fields}")

    gf: dict[str, np.ndarray] = {}
    for field_name in fields:
        recording = field_recording[field_name]
        print(f"[mdd] deconvolving {field_name}: nt={nt}, dt={dt:.3g}, dr={dr:.3g}, fmax={params.fmax_hz:.0g} Hz", flush=True)
        recording_perm = recording.transpose(_PERM)
        gf[field_name] = _run_mdd(
            incident_perm,
            recording_perm,
            dt=dt, dr=dr, params=params,
        ).astype(np.float32)

    # (3) Truncate in time.
    # pylops.MDD can return nt_out != nt_input depending on `nfmax`/`twosided`.
    # Build the output time axis from the actual GF shape × dt so `t` and
    # the tensors always agree on length.
    first_key = next(iter(fields))
    nt_out_actual = gf[first_key].shape[0]
    t_out = (np.arange(nt_out_actual) * dt).astype(np.float32)

    ntmax = max(1, int(round(params.tmax_out / dt)))
    ntmax = min(ntmax, nt_out_actual)
    for k in fields:
        gf[k] = gf[k][:ntmax, :, :]
    gf["t"] = t_out[:ntmax]

    # Attach the post-resampling positions so downstream consumers (cli,
    # save_gfs) don't need to know the production layout themselves.
    gf["inner_positions"] = np.asarray(reverb["inner_positions"])
    gf["outer_positions"] = np.asarray(reverb["outer_positions"])
    gf["ill_positions"]   = np.asarray(reverb["ill_positions"])

    return gf
