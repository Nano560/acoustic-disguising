"""Local plane-wave wave-field separation using two concentric shells.

3D port of the 2D MATLAB pipeline `curvedArrayDecomTwoLayers.m` (jmuller PhD,
mdd repo). For each outer-shell receiver point j we

    1. Take a tangent-plane patch of K nearest outer-shell neighbours plus
       K nearest inner-shell points, around the outer point j.
    2. Build a local Cartesian frame with +z along the outward normal at j
       and the origin midway between the inner and outer shells at j.
    3. For each frequency ω with wavenumber k = ω/c₀, build a dictionary of
       2·N_pw plane waves: N_pw forward (+k_z) and N_pw backward (−k_z) in
       the local frame, with (k_x, k_y) sampled on the radiating disk
       k_x² + k_y² ≤ k².
    4. Solve a damped least-squares fit
           [F_in_o   F_out_o] [a_in ]   [p_outer_patch]
           [F_in_i   F_out_i] [a_out] = [p_inner_patch]
       for the plane-wave amplitudes (a_in, a_out), then evaluate the
       incoming sum at the outer-shell sample point itself to recover
       p_in(j, ω).

Why this works at higher frequency than the modal SHT split:
the relevant angular Nyquist is set by the local patch sampling
(nearest-neighbour spacing), not by the full-shell sampling. With
nPoints_outer = 300 on r₂ = 0.3 m the global SHT cap is f ≲ 12 kHz, but a
patch of K = 25 nearest neighbours spans about 12 cm of arc with internal
sample spacing ~3 cm — the LS problem just has to fit a small local field
under that, which is fine for an 18 kHz Ricker.

The convention matches NumPy/PyLops e^{-iωt}: a local plane wave
exp(i(k_x x + k_y y − k_z z)), with z = outward radial, has time-domain
form exp(i(k_x x + k_y y − k_z z − ωt)), i.e. its phase fronts move in
the −z direction → INCOMING in the radial sense. Likewise +k_z is OUTGOING.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class LocalPWSeparationParams:
    """Parameters for the local plane-wave / two-shell separation."""

    patch_size: int  = 40            # K nearest neighbours per shell per patch.
                                     # Increase to tighten the LS overdetermination
                                     # (more equations vs unknowns ⇒ more robust).
    n_pw_radial: int = 2             # plane-wave radial samples (rings on disk).
                                     # Together with n_pw_azimuth controls the
                                     # plane-wave dictionary size:
                                     # n_pw = 1 + (n_pw_radial − 1) · n_pw_azimuth.
                                     # Keep small (≲ 1/2 · patch_size) — the
                                     # in/out basis becomes near-degenerate at
                                     # low kr if you let the dictionary grow.
    n_pw_azimuth: int = 4            # plane-wave azimuthal samples per ring
    reg: float       = 1.0e-1        # Tikhonov damping for the per-patch LS solve.
                                     # Must be substantial — the in/out plane-wave
                                     # basis at small kx,ky is nearly degenerate
                                     # since the only z-discrimination comes from
                                     # the two shell sample heights.
    c0: float        = 1500.0        # background sound speed [m/s]
    fmax_hz: float | None = None     # zero-out frequencies above this
    k_cap: float     = 0.95          # cap radial wavenumber at k_cap·k to avoid
                                     # k_z = 0 grazing modes (numerically singular)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "LocalPWSeparationParams":
        """Read parameters from `[mdd.local_pw]` (with `fmax_hz` inherited
        from the parent `[mdd]` table)."""
        mdd = cfg.get("mdd", {})
        lpw = mdd.get("local_pw", {})
        return cls(
            patch_size   = int  (lpw.get("patch_size", cls.patch_size)),
            n_pw_radial  = int  (lpw.get("n_radial",   cls.n_pw_radial)),
            n_pw_azimuth = int  (lpw.get("n_azimuth",  cls.n_pw_azimuth)),
            reg          = float(lpw.get("reg",        cls.reg)),
            k_cap        = float(lpw.get("k_cap",      cls.k_cap)),
            fmax_hz      = (None if mdd.get("fmax_hz") is None
                             else float(mdd["fmax_hz"])),
            # c0 falls through to the dataclass default; the reverb file's
            # `dom_c0` attr overrides it inside `mdd_extract` when present.
        )


# -----------------------------------------------------------------------------
# Geometry: per-patch indices and local frames
# -----------------------------------------------------------------------------

def _build_patches_and_frames(
    outer_pts: np.ndarray,
    inner_pts: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each outer point j:

      * indices of the K nearest outer-shell points (j included as 0-th)
      * indices of the K nearest inner-shell points to outer[j]
      * patch origin = midpoint between outer[j] and the nearest inner point
      * local frame (x_hat, y_hat, z_hat) with z_hat along the outward radial.

    Returns `(out_idx, in_idx, origin, x_hat, y_hat, z_hat)`.
    """
    n_outer = outer_pts.shape[0]
    tree_o = cKDTree(outer_pts)
    tree_i = cKDTree(inner_pts)
    _, out_idx = tree_o.query(outer_pts, k=K)
    _, partner = tree_i.query(outer_pts, k=1)
    # K nearest inner points to outer[j] directly — gives a denser inner
    # patch directly under the outer patch, regardless of asymmetric sampling.
    _, in_idx = tree_i.query(outer_pts, k=K)

    origin = 0.5 * (outer_pts + inner_pts[partner])

    z_hat = outer_pts / np.linalg.norm(outer_pts, axis=1, keepdims=True)

    # Tangent x-axis: project +x_world into the tangent plane; if z_hat is
    # too aligned with +x_world, fall back to +y_world.
    x_world = np.tile(np.array([1.0, 0.0, 0.0]), (n_outer, 1))
    x_proj  = x_world - np.einsum("ij,ij->i", x_world, z_hat)[:, None] * z_hat
    bad = np.linalg.norm(x_proj, axis=1) < 1.0e-3
    if bad.any():
        y_world = np.tile(np.array([0.0, 1.0, 0.0]), (bad.sum(), 1))
        x_proj_alt = (y_world
                      - np.einsum("ij,ij->i", y_world, z_hat[bad])[:, None]
                        * z_hat[bad])
        x_proj[bad] = x_proj_alt

    x_hat = x_proj / np.linalg.norm(x_proj, axis=1, keepdims=True)
    y_hat = np.cross(z_hat, x_hat)
    return out_idx, in_idx, origin, x_hat, y_hat, z_hat


def _patch_local_coords(
    pts: np.ndarray,
    idx: np.ndarray,
    origin: np.ndarray,
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    z_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express patch points in the local Cartesian frame.

    Returns `(x, y, z)` arrays of shape (n_outer, K).
    """
    coords = pts[idx] - origin[:, None, :]                 # (n_outer, K, 3)
    x = np.einsum("jki,ji->jk", coords, x_hat)
    y = np.einsum("jki,ji->jk", coords, y_hat)
    z = np.einsum("jki,ji->jk", coords, z_hat)
    return x, y, z


# -----------------------------------------------------------------------------
# Plane-wave dictionary
# -----------------------------------------------------------------------------

def _build_pw_grid(k: float, n_radial: int, n_azimuth: int, k_cap: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return `((kx, ky), kz)` for a plane-wave dictionary on the radiating
    disk k_x² + k_y² ≤ (k_cap · k)².

    The grid has 1 + (n_radial − 1) · n_azimuth points: one normal-incidence
    direction at the origin plus `n_radial − 1` rings of `n_azimuth`
    samples each. Each direction implicitly carries both ±k_z signs (handled
    by the caller).
    """
    if k <= 0.0:
        return np.zeros((0, 2)), np.zeros(0)
    kmax = k * k_cap
    radial  = np.linspace(0.0, kmax, n_radial)
    azimuth = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    kxs, kys = [0.0], [0.0]                                # normal incidence
    for r in radial[1:]:
        kxs.extend(r * np.cos(azimuth))
        kys.extend(r * np.sin(azimuth))
    kxky = np.stack([np.array(kxs), np.array(kys)], axis=1)
    kz = np.sqrt(np.maximum(0.0, k * k - kxky[:, 0] ** 2 - kxky[:, 1] ** 2))
    return kxky, kz


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def decompose_outer_pin_local_pw(
    outer_p: np.ndarray,             # (nt, n_outer, n_ill), real, [Pa]
    outer_positions: np.ndarray,     # (n_outer, 3) [m]
    inner_p: np.ndarray,             # (nt, n_inner, n_ill), real, [Pa]
    inner_positions: np.ndarray,     # (n_inner, 3) [m]
    *,
    dt: float,                       # sample period [s]
    params: LocalPWSeparationParams,
    verbose: bool = False,
) -> np.ndarray:
    """Local plane-wave / two-shell incoming-pressure separation.

    Pressure on both shells is required; v_n is not used. Returns
    `outer_pin` with the same shape and dtype as `outer_p`.
    """
    if outer_p.shape[0] != inner_p.shape[0]:
        raise ValueError("outer_p and inner_p must agree on the time axis")
    if outer_p.shape[2] != inner_p.shape[2]:
        raise ValueError("outer_p and inner_p must agree on the n_ill axis")
    nt, n_outer, n_ill = outer_p.shape
    n_inner = inner_p.shape[1]
    K = params.patch_size
    if K > min(n_outer, n_inner):
        raise ValueError(f"patch_size {K} > min(n_outer, n_inner)")

    # Geometry — once.
    out_idx, in_idx, origin, x_hat, y_hat, z_hat = _build_patches_and_frames(
        outer_positions, inner_positions, K)
    x_o, y_o, z_o = _patch_local_coords(
        outer_positions, out_idx, origin, x_hat, y_hat, z_hat)
    x_i, y_i, z_i = _patch_local_coords(
        inner_positions, in_idx, origin, x_hat, y_hat, z_hat)
    # Local z of point j itself (where we evaluate p_in) is z_o[j, 0]
    # because out_idx[j, 0] = j (kd-tree returns the query point first).
    z_at_j = z_o[:, 0]                                     # (n_outer,)

    # FFT in time.
    P_o = np.fft.rfft(outer_p, axis=0)                     # (nf, n_outer, n_ill)
    P_i = np.fft.rfft(inner_p, axis=0)
    nf  = P_o.shape[0]
    freqs = np.fft.rfftfreq(nt, d=dt)
    fmax = params.fmax_hz if params.fmax_hz is not None else freqs[-1]

    Pin_fft = np.zeros((nf, n_outer, n_ill), dtype=np.complex128)
    n_active = 0
    n_pw_min = n_pw_max = 0

    eye2pw_cache: dict[int, np.ndarray] = {}
    for fi in range(1, nf):                                # skip DC
        f = freqs[fi]
        if f > fmax:
            continue
        k = 2.0 * np.pi * f / params.c0
        kxky, kz = _build_pw_grid(
            k, params.n_pw_radial, params.n_pw_azimuth, params.k_cap)
        n_pw = kxky.shape[0]
        if n_pw == 0:
            continue
        n_active += 1
        n_pw_min = n_pw if n_pw_min == 0 else min(n_pw_min, n_pw)
        n_pw_max = max(n_pw_max, n_pw)

        # Build F tensor of shape (n_outer, 2K, 2 n_pw) all at once.
        # tangent-plane phase for outer/inner patches:
        phx_o = x_o[..., None] * kxky[None, None, :, 0]    # (n_outer, K, n_pw)
        phy_o = y_o[..., None] * kxky[None, None, :, 1]
        phx_i = x_i[..., None] * kxky[None, None, :, 0]
        phy_i = y_i[..., None] * kxky[None, None, :, 1]
        phz_o = z_o[..., None] * kz[None, None, :]
        phz_i = z_i[..., None] * kz[None, None, :]
        F_in_o  = np.exp(1j * (phx_o + phy_o - phz_o))
        F_in_i  = np.exp(1j * (phx_i + phy_i - phz_i))
        F_out_o = np.exp(1j * (phx_o + phy_o + phz_o))
        F_out_i = np.exp(1j * (phx_i + phy_i + phz_i))
        F_top    = np.concatenate([F_in_o, F_out_o], axis=2)   # (n_outer, K, 2n_pw)
        F_bottom = np.concatenate([F_in_i, F_out_i], axis=2)
        F = np.concatenate([F_top, F_bottom], axis=1)          # (n_outer, 2K, 2n_pw)

        # Batched normal equations: G @ a = FH @ D  per patch j
        FH = np.conj(np.transpose(F, (0, 2, 1)))               # (n_outer, 2n_pw, 2K)
        if 2*n_pw not in eye2pw_cache:
            eye2pw_cache[2*n_pw] = np.eye(2*n_pw, dtype=np.complex128)
        G = FH @ F + params.reg * eye2pw_cache[2*n_pw][None, :, :]   # (n_outer, 2n_pw, 2n_pw)
        # Patch data only at this frequency: (n_outer, 2K, n_ill).
        Do = np.transpose(P_o[fi, out_idx, :], (0, 1, 2))      # already (n_outer, K, n_ill)
        Di = np.transpose(P_i[fi, in_idx,  :], (0, 1, 2))
        D  = np.concatenate([Do, Di], axis=1)                  # (n_outer, 2K, n_ill)
        rhs = FH @ D                                           # (n_outer, 2n_pw, n_ill)
        a = np.linalg.solve(G, rhs)                            # (n_outer, 2n_pw, n_ill)
        a_in = a[:, :n_pw, :]                                  # (n_outer, n_pw, n_ill)

        # Sum incoming amplitudes at the outer-shell point j itself
        # (local position (0, 0, z_at_j[j])).
        phase_at_j = np.exp(-1j * kz[None, :] * z_at_j[:, None])   # (n_outer, n_pw)
        Pin_fft[fi] = np.einsum("jp,jpi->ji", phase_at_j, a_in)

    outer_pin = np.fft.irfft(Pin_fft, n=nt, axis=0)

    if verbose:
        if n_active > 0:
            print(
                f"[lpw-sep] K={K}, n_pw_radial={params.n_pw_radial}, "
                f"n_pw_az={params.n_pw_azimuth}, reg={params.reg:.1g}, "
                f"c0={params.c0:.0g} m/s. Active band has {n_active} bins, "
                f"plane-wave count {n_pw_min}–{n_pw_max} per bin.",
                flush=True,
            )
        else:
            print("[lpw-sep] no active frequencies (check fmax_hz)", flush=True)

    return outer_pin.astype(outer_p.dtype, copy=False)


# -----------------------------------------------------------------------------
# Single-shell variant: p + z0·v_n on the outer shell only
# -----------------------------------------------------------------------------

def decompose_outer_pin_local_pvn(
    outer_p: np.ndarray,             # (nt, n_outer, n_ill), real, [Pa]
    outer_vnz: np.ndarray,           # (nt, n_outer, n_ill), real, z₀·v_n [Pa]
    outer_positions: np.ndarray,     # (n_outer, 3) [m]
    *,
    dt: float,                       # sample period [s]
    params: LocalPWSeparationParams,
    verbose: bool = False,
) -> np.ndarray:
    """Local plane-wave separation using p + z₀·v_n on the outer shell only.

    Per outer-shell receiver j, we take a tangent-plane patch of K nearest
    outer points (no inner shell needed), and at each patch point pair up
    the pressure and impedance-scaled normal velocity. For a plane wave
    of direction m with k_z = √(k² − k_x² − k_y²) evaluated at patch
    point k:

        p_m  (k) = exp(i (k_x x_k + k_y y_k ± k_z z_k))
        z₀·v_n_m(k) = ± cosθ_m · p_m(k),  where cosθ_m = k_z / k.

    Stacking K outer-pressure rows on top of K outer-velocity rows gives a
    2K × 2N_pw system whose per-direction 2×2 in/out determinant is just
    2cosθ_m — non-singular for every non-grazing direction. No comb of
    singular frequencies, and no 1/r amplitude mismatch (both observations
    sit at the same point).

    Reduces exactly to the `normal_incidence` formula `(p − z₀·v_n)/2`
    when N_pw = 1 at normal incidence (cosθ = 1, the only direction).
    Adding more directions on the (k_x, k_y) disk gives the curvature
    correction that `normal_incidence` misses for obliquely-incident
    wavefronts on a curved shell.
    """
    if outer_p.shape != outer_vnz.shape:
        raise ValueError("outer_p and outer_vnz must have the same shape")
    nt, n_outer, n_ill = outer_p.shape
    K = params.patch_size
    if K > n_outer:
        raise ValueError(f"patch_size {K} > n_outer {n_outer}")

    # 1. Patch geometry on the OUTER shell only. Local +z = outward normal at
    # the patch centre, origin at the patch centre point itself.
    tree_o = cKDTree(outer_positions)
    _, out_idx = tree_o.query(outer_positions, k=K)             # (n_outer, K)

    # Origin is the patch-centre point (point j itself), so the patch
    # points have small but nonzero local z due to shell curvature.
    origin = outer_positions.copy()
    z_hat = outer_positions / np.linalg.norm(outer_positions, axis=1, keepdims=True)
    x_world = np.tile(np.array([1.0, 0.0, 0.0]), (n_outer, 1))
    x_proj = x_world - np.einsum("ij,ij->i", x_world, z_hat)[:, None] * z_hat
    bad = np.linalg.norm(x_proj, axis=1) < 1.0e-3
    if bad.any():
        y_world = np.tile(np.array([0.0, 1.0, 0.0]), (bad.sum(), 1))
        x_proj[bad] = (y_world
                       - np.einsum("ij,ij->i", y_world, z_hat[bad])[:, None]
                         * z_hat[bad])
    x_hat = x_proj / np.linalg.norm(x_proj, axis=1, keepdims=True)
    y_hat = np.cross(z_hat, x_hat)

    coords  = outer_positions[out_idx] - origin[:, None, :]     # (n_outer, K, 3)
    x_pat = np.einsum("jki,ji->jk", coords, x_hat)
    y_pat = np.einsum("jki,ji->jk", coords, y_hat)
    z_pat = np.einsum("jki,ji->jk", coords, z_hat)

    # 2. FFT in time.
    P_o = np.fft.rfft(outer_p,   axis=0)                        # (nf, n_outer, n_ill)
    V_o = np.fft.rfft(outer_vnz, axis=0)
    nf  = P_o.shape[0]
    freqs = np.fft.rfftfreq(nt, d=dt)
    fmax = params.fmax_hz if params.fmax_hz is not None else freqs[-1]

    Pin_fft = np.zeros((nf, n_outer, n_ill), dtype=np.complex128)
    n_active = 0
    n_pw_min = n_pw_max = 0
    eye2pw_cache: dict[int, np.ndarray] = {}

    for fi in range(1, nf):
        f = freqs[fi]
        if f > fmax:
            continue
        k = 2.0 * np.pi * f / params.c0
        kxky, kz = _build_pw_grid(
            k, params.n_pw_radial, params.n_pw_azimuth, params.k_cap)
        n_pw = kxky.shape[0]
        if n_pw == 0:
            continue
        n_active += 1
        n_pw_min = n_pw if n_pw_min == 0 else min(n_pw_min, n_pw)
        n_pw_max = max(n_pw_max, n_pw)

        # cos(θ_m) = k_z / k for each direction m
        cos_th = kz / k                                          # (n_pw,)

        # Phase factors at each patch point for each direction
        phx = x_pat[..., None] * kxky[None, None, :, 0]          # (n_outer, K, n_pw)
        phy = y_pat[..., None] * kxky[None, None, :, 1]
        phz = z_pat[..., None] * kz [None, None, :]
        # In and out plane waves at the patch points, evaluated on the curved
        # shell surface so z_pat carries the curvature contribution.
        F_in_p  = np.exp(1j * (phx + phy - phz))                 # (n_outer, K, n_pw)
        F_out_p = np.exp(1j * (phx + phy + phz))
        # Pressure rows (K rows): p
        # Velocity rows (K rows): z₀·v_n = ± cosθ · p (sign carried by in/out)
        F_in_v  = -cos_th[None, None, :] * F_in_p
        F_out_v = +cos_th[None, None, :] * F_out_p
        # Stack to (n_outer, 2K, 2 n_pw): top K rows = pressure, bottom K = velocity.
        F_top    = np.concatenate([F_in_p, F_out_p], axis=2)
        F_bottom = np.concatenate([F_in_v, F_out_v], axis=2)
        F = np.concatenate([F_top, F_bottom], axis=1)

        # Patch data: (n_outer, K, n_ill) for pressure, same for velocity.
        Dp = P_o[fi, out_idx, :]                                  # (n_outer, K, n_ill)
        Dv = V_o[fi, out_idx, :]
        D  = np.concatenate([Dp, Dv], axis=1)                     # (n_outer, 2K, n_ill)

        FH = np.conj(np.transpose(F, (0, 2, 1)))
        if 2*n_pw not in eye2pw_cache:
            eye2pw_cache[2*n_pw] = np.eye(2*n_pw, dtype=np.complex128)
        G = FH @ F + params.reg * eye2pw_cache[2*n_pw][None, :, :]
        rhs = FH @ D
        a = np.linalg.solve(G, rhs)
        a_in = a[:, :n_pw, :]                                     # (n_outer, n_pw, n_ill)

        # Evaluate the incoming-only sum at the outer-shell point j itself.
        # In this single-shell framing, the patch origin IS point j, so j's
        # local position is (0, 0, 0). The phase at the origin is just exp(0) = 1.
        Pin_fft[fi] = a_in.sum(axis=1)

    outer_pin = np.fft.irfft(Pin_fft, n=nt, axis=0)

    if verbose:
        if n_active > 0:
            print(
                f"[lpvn-sep] K={K}, n_pw_radial={params.n_pw_radial}, "
                f"n_pw_az={params.n_pw_azimuth}, reg={params.reg:.1g}, "
                f"c0={params.c0:.0g} m/s. Active band has {n_active} bins, "
                f"plane-wave count {n_pw_min}–{n_pw_max} per bin.",
                flush=True,
            )
        else:
            print("[lpvn-sep] no active frequencies (check fmax_hz)", flush=True)

    return outer_pin.astype(outer_p.dtype, copy=False)
