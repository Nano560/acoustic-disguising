"""Wave-field separation on a closed spherical control surface.

Replaces the high-frequency / normal-incidence approximation

    p_in(Ω, t) ≈ (p(Ω, t) − z₀·v_n(Ω, t)) / 2

used by `mdd._decompose_outer_pin` with a modally exact decomposition based
on the spherical-Hankel form of the exterior Helmholtz solution

    p̂(r, Ω, ω) = Σ_{n,m} [A_nm h_n^{(1)}(kr) + B_nm h_n^{(2)}(kr)] Y_n^m(Ω),

with the e^{-iωt} convention used by NumPy/PyLops, h_n^{(1)} is the *outgoing*
radial mode and h_n^{(2)} the *incoming* one. The incoming pressure on a
shell of radius r₂ is

    p̂_in(r₂, Ω, ω) = Σ_{n,m} B_nm(ω) h_n^{(2)}(k r₂) Y_n^m(Ω).

For each (n, m, ω), `(A_nm, B_nm)` are recovered from the local pressure /
normal-velocity coefficients on the shell by the 2×2 system

    [   h_n^{(1)}(kr)        h_n^{(2)}(kr)     ] [A_nm]   [   p_nm     ]
    [ −i h_n^{(1)′}(kr)    −i h_n^{(2)′}(kr)   ] [B_nm] = [ z₀·v_nm    ]

(the second row is the radial component of v_n = ∂_r p / (iωρ₀) multiplied by
z₀ = ρ₀ c₀; the input `outer_vnz` is already pre-multiplied by z₀ on the
Julia side). Using the spherical-Bessel Wronskian j_n y_n′ − y_n j_n′ = 1/x²,
the closed-form solution is

    p_in_nm = ½ · ( α_n(kr) · p_nm  −  β_n(kr) · vnz_nm ),
        α_n(x) = 1 − i x² (j_n j_n′ + y_n y_n′),
        β_n(x) = x² (j_n² + y_n²).

In the high-frequency limit kr → ∞, j² + y² → 1/x² and j j′ + y y′ → 0,
so α_n → 1 and β_n → 1, and `p_in_nm → (p_nm − vnz_nm) / 2`, recovering the
`normal_incidence` formula (the high-frequency / normal-incidence
plane-wave approximation).

References:
  - Williams, Fourier Acoustics (Academic Press, 1999), Ch. 6 — exterior
    Helmholtz expansion.
  - Rafaely, Fundamentals of Spherical Array Processing (Springer, 2015) —
    spherical-harmonic projection on quasi-uniform sample grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
# SciPy ≥ 1.15 removed `sph_harm`; the replacement is `sph_harm_y` with the
# (n, m) argument order swapped. Adapt at import time so this file works on
# both old and new SciPy.
from scipy.special import spherical_jn, spherical_yn
try:
    from scipy.special import sph_harm  # SciPy < 1.15
except ImportError:
    from scipy.special import sph_harm_y as _sph_harm_y  # SciPy ≥ 1.15
    def sph_harm(m, n, phi, theta):
        return _sph_harm_y(n, m, theta, phi)


def auto_n_max(n_outer: int) -> int:
    """Angular-Nyquist heuristic for the SH truncation degree on a
    quasi-uniform Fibonacci-sphere sampling.

    For N quasi-uniform points on a unit sphere, the maximum
    distinguishable spherical-harmonic degree is bounded by
    `n_max ≲ √N · k` with k ≈ 0.5–0.7 depending on the sampling scheme
    (Saff & Kuijlaars 1997; Fliege & Maier 1999). Empirically k = 0.6
    matches a Fibonacci-sphere sweep
    (n_outer=300 → ~10, n_outer=1000 → ~20, n_outer=4000 → ~38). Below
    this bound the SH projection is safely over-determined; above it,
    high-n modes alias and create banding artefacts.
    """
    return max(1, int(round(0.6 * np.sqrt(n_outer))))


@dataclass
class SHTSeparationParams:
    """Parameters for the SHT-based outgoing/incoming pressure split."""

    n_max: int | None = None     # spherical-harmonic truncation hard cap (max n).
                                 # None ⇒ auto from `auto_n_max(n_outer)`
                                 # (≈ 0.6·√n_outer, the angular-Nyquist
                                 # bound for Fibonacci sampling). Set
                                 # explicitly to override.
    reg: float     = 1.0e-6      # Tikhonov damping for the SH projection
    c0: float      = 1500.0      # background sound speed [m/s]
    fmax_hz: float | None = None # zero-out frequencies above this (None → Nyquist)
    beta_max: float = 20.0       # per-(n, ω) magnitude cap:
                                 #   keep mode (n, m) iff β_n(kr) ≤ beta_max.
                                 # β_n = (kr)²·(j_n²+y_n²) is exactly 1 in
                                 # the radiating regime (kr > n) and grows
                                 # as ~((2n−1)!!/(kr)^n)² in the evanescent
                                 # regime (kr < n). Capping β_n bounds how
                                 # much noise on v_nm the radial split can
                                 # amplify. Equivalent to a per-frequency
                                 # cutoff `n ≲ kr · β_max^(1/(2n))`. No
                                 # closed-form optimum — depends on input
                                 # SNR. Practical defaults:
                                 #   1   only radiating modes (loses signal)
                                 #   10  noisy / real data
                                 #   20  clean simulation data (default)
                                 #   50  very clean data, accept some
                                 #       evanescent-mode amplification

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SHTSeparationParams":
        """Read SHT params from `[mdd.sht]` (with `fmax_hz` inherited from
        the parent `[mdd]` table). `n_max` defaults to None when absent
        from cfg → `decompose_outer_pin_sht` derives it from n_outer."""
        mdd = cfg.get("mdd", {})
        sht = mdd.get("sht", {})
        n_max_raw = sht.get("n_max", None)
        return cls(
            n_max    = (None if n_max_raw is None else int(n_max_raw)),
            reg      = float(sht.get("reg",      cls.reg)),
            beta_max = float(sht.get("beta_max", cls.beta_max)),
            fmax_hz  = (None if mdd.get("fmax_hz") is None
                        else float(mdd["fmax_hz"])),
            # c0 falls through to the dataclass default; the reverb file's
            # `dom_c0` attr overrides it inside `mdd_extract` when present.
        )


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

def _spherical_to_angles(positions: np.ndarray, radius: float, *,
                         rtol: float = 5.0e-3
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian (x, y, z) on a sphere of given radius → (theta, phi).

    `theta` is the polar angle from +z in [0, π]; `phi` the azimuth in
    [-π, π]. Verifies that all sample points lie on r ≈ `radius` (within
    relative tolerance `rtol` — Fibonacci-sphere points sit exactly on the
    radius up to floating-point error).
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    rel_dev = np.max(np.abs(r / radius - 1.0))
    if rel_dev > rtol:
        raise ValueError(
            f"outer positions deviate from r = {radius}: max relative "
            f"deviation = {rel_dev:.3g} > rtol {rtol:.0g}"
        )
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi   = np.arctan2(y, x)
    return theta, phi


def _build_sh_design(theta: np.ndarray, phi: np.ndarray, n_max: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return `(Y, mode_n)`:

    - `Y[j, k] = Y_n^m(Ω_j)` of shape (M, K) with K = (n_max + 1)²,
    - `mode_n[k]` is the degree n for column k.

    Columns are stored in (n=0..n_max, m=-n..+n) row-major order.
    """
    n_modes = (n_max + 1) ** 2
    Y = np.empty((theta.size, n_modes), dtype=np.complex128)
    mode_n = np.empty(n_modes, dtype=np.int64)
    col = 0
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            # scipy.special.sph_harm(m, n, phi, theta): phi=azimuth, theta=polar.
            Y[:, col] = sph_harm(m, n, phi, theta)
            mode_n[col] = n
            col += 1
    return Y, mode_n


def _sh_pinv(Y: np.ndarray, reg: float) -> np.ndarray:
    """Tikhonov-damped least-squares pseudo-inverse: `(YᴴY + reg·I)⁻¹ Yᴴ`.

    With M ≈ 300 quasi-uniform Fibonacci samples and (n_max + 1)² ≲ M, this is
    well-conditioned for n_max ≲ 15; the small `reg` only stabilises the
    high-degree tail.
    """
    K = Y.shape[1]
    G = Y.conj().T @ Y + reg * np.eye(K)
    return np.linalg.solve(G, Y.conj().T)


def _radial_coefficients(n_max: int, kr: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Return `(α_n, β_n)` for n = 0..n_max at given `kr`.

    α_n(x) = 1 − i x² (j_n j_n′ + y_n y_n′)   (complex)
    β_n(x) = x² (j_n² + y_n²)                 (real)

    For `kr ≤ 0` returns zeros (no separation possible at DC).
    """
    if kr <= 0.0:
        return (np.zeros(n_max + 1, dtype=np.complex128),
                np.zeros(n_max + 1, dtype=np.float64))

    n_grid = np.arange(n_max + 1)
    j  = spherical_jn(n_grid, kr)
    yy = spherical_yn(n_grid, kr)
    jp = spherical_jn(n_grid, kr, derivative=True)
    yp = spherical_yn(n_grid, kr, derivative=True)
    alpha = 1.0 - 1j * (kr * kr) * (j * jp + yy * yp)
    beta  = (kr * kr) * (j * j + yy * yy)
    return alpha.astype(np.complex128), beta.astype(np.float64)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def decompose_outer_pin_sht(
    outer_p: np.ndarray,            # (nt, n_outer, n_ill), real, [Pa]
    outer_vnz: np.ndarray,          # (nt, n_outer, n_ill), real, z₀·v_n [Pa]
    outer_positions: np.ndarray,    # (n_outer, 3) [m]
    *,
    radius: float,                  # outer-shell radius [m]
    dt: float,                      # sample period [s]
    params: SHTSeparationParams,
    verbose: bool = False,
) -> np.ndarray:
    """Modally-exact incoming-pressure separation on a closed spherical shell.

    Steps:
      1. Project p, vnz onto a spherical-harmonic basis up to degree
         `params.n_max` via Tikhonov-regularised least-squares.
      2. For each (n, ω), apply the closed-form 2×2 radial split to obtain
         the spherical-harmonic coefficients of the incoming pressure on
         the shell.
      3. Re-evaluate at the original sample points and inverse-FFT in time.

    Returns `outer_pin` of shape `(nt, n_outer, n_ill)` and the same dtype as
    `outer_p`.
    """
    if outer_p.shape != outer_vnz.shape:
        raise ValueError(
            f"shape mismatch: outer_p={outer_p.shape}, outer_vnz={outer_vnz.shape}")
    nt, n_pts, n_ill = outer_p.shape
    if outer_positions.shape != (n_pts, 3):
        raise ValueError(
            f"outer_positions shape {outer_positions.shape} != ({n_pts}, 3)")
    # Resolve n_max: None ⇒ angular-Nyquist heuristic from n_outer.
    if params.n_max is None:
        params = SHTSeparationParams(
            n_max=auto_n_max(n_pts), reg=params.reg, c0=params.c0,
            fmax_hz=params.fmax_hz, beta_max=params.beta_max)
    if params.n_max < 0:
        raise ValueError("n_max must be ≥ 0")
    if (params.n_max + 1) ** 2 > n_pts:
        raise ValueError(
            f"(n_max+1)² = {(params.n_max+1)**2} exceeds n_outer = {n_pts}; "
            "spherical-harmonic projection is under-determined. "
            "Lower n_max or sample more points.")

    # 1. Build SH design matrix and pseudo-inverse on the shell sample points.
    theta, phi = _spherical_to_angles(outer_positions, radius)
    Y, mode_n = _build_sh_design(theta, phi, params.n_max)
    Y_pinv    = _sh_pinv(Y, params.reg)                     # (K, M)

    # 2. Time-domain → frequency-domain (real input → Hermitian spectrum).
    P_fft = np.fft.rfft(outer_p,   axis=0)                  # (nf, M, n_ill)
    V_fft = np.fft.rfft(outer_vnz, axis=0)
    nf    = P_fft.shape[0]
    freqs = np.fft.rfftfreq(nt, d=dt)
    k_arr = 2.0 * np.pi * freqs / params.c0

    # 3. Project to SH coefficients.
    P_nm = np.einsum("kp,fpi->fki", Y_pinv, P_fft)          # (nf, K, n_ill)
    V_nm = np.einsum("kp,fpi->fki", Y_pinv, V_fft)

    # 4. Per-(n, ω) closed-form radial split. Vectorised over m and ill-source.
    #    Magnitude-based truncation: drop modes where β_n(kr) > beta_max,
    #    which simultaneously caps the radial-split noise gain and removes
    #    evanescent modes (β_n ~ (kr)^(-2n) for n > kr).
    fmax_hz = params.fmax_hz if params.fmax_hz is not None else freqs[-1]
    Pin_nm  = np.zeros_like(P_nm)
    n_kept_per_f = np.zeros(nf, dtype=np.int64)
    for f_idx in range(1, nf):                              # skip DC
        f = freqs[f_idx]
        if f > fmax_hz:
            continue
        kr = float(k_arr[f_idx]) * radius
        alpha_n, beta_n = _radial_coefficients(params.n_max, kr)   # (n_max+1,)
        # `n_ok` is the highest contiguous degree with β_n ≤ beta_max
        # (β_n grows monotonically with n for fixed kr in the evanescent
        # regime, so a contiguous keep-band is the natural rule).
        ok = beta_n <= params.beta_max
        keep = ok[mode_n]
        n_kept_per_f[f_idx] = int(keep.sum())
        if not keep.any():
            continue
        a = alpha_n[mode_n[keep]][:, None]                  # (K_keep, 1)
        b = beta_n [mode_n[keep]][:, None]
        Pin_nm[f_idx, keep, :] = 0.5 * (a * P_nm[f_idx, keep, :]
                                        - b * V_nm[f_idx, keep, :])

    # 5. Reconstruct on the spatial sample points.
    Pin_fft = np.einsum("pk,fki->fpi", Y, Pin_nm)           # (nf, M, n_ill)

    # 6. Inverse FFT to time. The radial-split operator is Hermitian-symmetric
    #    in ω (α(-ω) = conj α(ω), β real and even), so the spectrum is
    #    Hermitian and irfft gives a real output.
    outer_pin = np.fft.irfft(Pin_fft, n=nt, axis=0)

    if verbose:
        active = (n_kept_per_f > 0)
        if active.any():
            print(
                f"[sht-sep] n_max={params.n_max}, beta_max={params.beta_max:.0g}, "
                f"reg={params.reg:.1g}, c0={params.c0:.1g} m/s, "
                f"r={radius:.3g} m. "
                f"Active band: {freqs[active][0]:.0g}–{freqs[active][-1]:.0g} Hz, "
                f"#modes ranges {n_kept_per_f[active].min()}–"
                f"{n_kept_per_f[active].max()} (full = {(params.n_max+1)**2}).",
                flush=True,
            )
        else:
            print("[sht-sep] no active frequencies (check fmax_hz)", flush=True)

    return outer_pin.astype(outer_p.dtype, copy=False)
