
# History of α formula iterations (newest last; kept here as a record of
# how we arrived at the closed-form α used in `C_for_N` below — useful
# documentation, not consumed by any code path):
#
#   v1_neg_dSin_e_Cp   C_f = -dS_in · e · C_p,   C_q = ... · z₀²
#       At N=50:  s_opt = -235.6,  rms_B = 2.17e-3
#       At N=500: s_opt = -609.1,  rms_B = 1.35e-3
#       → wrong sign. s_opt scales ~N^0.41 (slab_B nearly converges due
#         to the dS_in ∝ 1/N weighting — correct structure for K-H
#         convergence — but the absolute scale is way off).
#
#   v2_pos_e_Cp        C_f = +e · C_p,           C_q = ... · z₀²
#       Drops dS_in (each source has constant amplitude).
#       At N=300: s_opt = 1.003,  rms_B = 0.810,  suppression 4.66×
#       At N=400: s_opt = 0.750,  rms_B = 1.087,  suppression 4.96×
#       At N=500: s_opt = 0.612,  rms_B = 1.340,  suppression 6.06×
#       → s_opt · N ≈ 301 (constant ± 2%): CLEAN 1/N scaling. Without the
#         dS_in weighting, the radiated field is the SUM (not the integral)
#         over sources, so slab_B grows ∝ N → s_opt has to shrink ∝ 1/N.
#         v2 happens to pass through s_opt=1 at N≈300 by accident. Need to
#         re-add dS_in.
#
#   v3_pos_dSin_e_Cp_C  C_f = +C · dS_in · e · C_p,   C_q = ... · z₀²
#       Re-adds dS_in (correct K-H surface weighting), flips sign back to +.
#       The constant `C` is the "missing factor" we're hunting; iterate
#       from the s_opt feedback to land s_opt → 1 N-independently.
#       Starting `C = 1.0`. Expected starting s_opt ≈ 1/v1_s_opt_at_same_N
#       (around 1/600 ≈ 1.6e-3) if v1's N-scaling structure was right.
#       At N=200: s_opt = 553.7
#       At N=300: s_opt = 598.6
#       At N=400: s_opt = 596.5
#       At N=500: s_opt = 609.1
#       → s_opt nearly N-independent at ~600 (~10 % drift across 200→500; only
#         ~2 % across 300/400/500). Asymptote around 620 ≈ 1/(ρ·c²·dt²).
#         The dimensional analysis closes it: v3's C_f has units m·s but
#         vn[:src]/p_inner requires m²·s/kg → a 1/ρ is missing.
#
#   v4_dSin_dt_over_rho_dx3   C_f = (dS_in · dt) / (ρ · dx³),  C_q = C_f · z₀²
#       Derived analytically from dimensional consistency. The ratio to
#       v3(C=1) is 1/(ρ·c²·dt²) ≈ 620, matching v3's empirical s_opt ≈ 609
#       at N=500 to 1.8 % (K-H quadrature noise). Physical reading:
#           C_f = (dS_in · dt) / (ρ·dx³) = (surface-time element) / (cell mass)
#                  → converts surface pressure into per-step velocity increment.
#           C_q = C_f · z₀² = C_f · ρ²·c² = (dS_in · dt · ρ · c²) / dx³
#                  → converts surface velocity into per-step pressure increment.
#       Full (N, cf, dx) sweep (sorted by cps = source-spacing / dx):
#           (cf, dx,    N)    cps     s_opt    gap to 1
#           (0.5, 0.0088, 500)  3.6   1.011    +1.1 %
#           (0.5, 0.0044, 500)  7.2   0.983    -1.7 %
#           (0.25,0.0044, 500)  7.2   0.981    -1.9 %      ← dt-independence ✓
#           (0.5, 0.0044, 400)  8.1   0.962    -3.8 %
#           (0.5, 0.0044, 300)  9.3   0.966    -3.4 %
#           (0.25,0.0044, 300)  9.3   0.964    -3.6 %      ← dt-independence ✓
#           (0.5, 0.0022,1000) 10.2   0.965    -3.5 %      ← matches dx=0.0044,N=300
#           (0.5, 0.0022, 500) 14.4   0.905    -9.5 %
#           (0.5, 0.0022, 300) 18.6   0.786   -21.4 %
#       ✓ The whole sweep COLLAPSES onto a single parameter:
#             cps = √(4π·R_in²/N) / dx = mean inter-source spacing in cells.
#         cf drops out (canceled by the dt factor). (dx, N) couple via cps.
#         As cps → 0 (FDTD coarser or N denser), s_opt → 1 — the K-H integral
#         converges in the joint (N, dx) continuum limit.
#       Why coarser dx is *better* here: when the FDTD's spatial resolution
#         is comparable to the gap between sources (small cps), the FDTD
#         blurs the discrete sources into an effectively continuous
#         distribution → matches the K-H integral. When dx ≪ source-spacing
#         (large cps), the FDTD resolves the per-source structure and the
#         gaps between sources become visible → quadrature error grows.
#       Rule of thumb for the test: pick (N, dx) so that
#             N ≳ 4π·R_in² / (5·dx)²
#         (i.e., ~5 cells per source-spacing) to get s_opt within a few % of 1.
#       Bottom line: v4 is dimensionally and analytically correct — no fitted
#         constant. Any residual s_opt ≠ 1 is K-H quadrature error governed
#         by cps alone.
#
# Short tag for the α formula, written into HDF5-cache attributes + the
# `sweep_results.csv` row so a multi-run accumulator can tell which formula
# produced each row. Not used in filenames — caches are wiped manually if
# the formula changes (matches the project's no-backcompat policy).
const C_FORMULA = "dSin_dt_over_rho_dx3"

# ============================================================================
# Closed-form analytical α — factor-by-factor breakdown
# ============================================================================
#
# Two channels, one formula. Sources placed at each inner-sphere
# transceiver:
#
#     vn[:src](x_S, t)  =  C_f · p_inner_recorded(x_S, t)        (dipole / f)
#     p[:src](x_S, t)   =  C_q · vn_inner_recorded(x_S, t)       (monopole / q)
#
# with
#
#     C_f = dS_in · dt / (ρ · dx³)                  units: m²·s / kg
#     C_q = C_f · z₀²    (= dS_in · dt · ρ · c² / dx³)
#
# Every factor is analytical: no fitted constant, no empirical knob. Each
# factor is annotated below — what it is, how the (v1→v4) sweep revealed
# it, and what it means physically.
#
# ─── dS_in = 4π · R_in² / N ─────────────────────────────────────────────
# Surface element per Fibonacci-sphere source point.
#  • Origin: the K-H representation is a SURFACE INTEGRAL
#    ∮f(x_S) dS ≈ Σ_i f(x_i) · dS_in.  Each discrete source must carry the
#    fraction of the sphere's area it represents (otherwise the sum scales
#    with N instead of converging).
#  • How found: v2 dropped dS_in → slab_B grew ∝ N (sources just added up)
#    and s_opt scaled cleanly as 1/N. Re-adding dS_in in v3 made slab_B
#    N-independent and s_opt nearly N-flat.
#
# ─── dt ──────────────────────────────────────────────────────────────────
# FDTD time step (= cf · dx / (c·√3) from the CFL condition).
#  • Origin: the FDTD source injection is `field += src[step]` at each
#    step. To match a continuous K-H source density (with units of force
#    per area per time), the per-step amplitude must be the continuous
#    density × dt — this dt is the time-integration weight.
#  • How found: dimensional analysis. v3's α had units m·s but the v→p
#    relation requires m²·s/kg — explicit dt and explicit 1/ρ surfaced
#    only after expanding the v3 (e · C_p) bundle.
#  • Independently verified: Phase A (cf 0.5 → 0.25, halving dt) changed
#    s_opt by < 0.2 % at every N — formula is dt-exact.
#
# ─── 1 / (ρ · dx³) = 1 / m_cell ─────────────────────────────────────────
# Inverse of the mass of fluid in one FDTD cell (ρ · dx³).
#  • Origin: the V-field update is ∂_t V = -∇P/ρ + per-cell-force. The
#    transceiver's source is added to V at one cell, so it acts on one
#    cell's worth of mass. Dividing the integrand (dS_in · dt · p) by
#    the cell mass converts a pressure-driving into a velocity increment.
#  • Physical reading: "the surface-force impulse (dS_in · dt · p) per
#    cell mass equals the velocity increment per step".
#  • How found: v3 matched to ≈ 620 = 1/(ρ · c² · dt²). The cleanest
#    refactor pulled 1/ρ explicitly out of v3's bundled e · C_p, yielding
#    C_f = dS_in · dt / (ρ · dx³) with the right m²·s/kg units.
#  • Phase B verifies dx-independence.
#
# ─── z₀² = (ρ · c)² ─────────────────────────────────────────────────────
# Impedance² — ratio between the two source channels (monopole/dipole).
#  • Origin: the K-H equivalent sources are q = +vn_surf (monopole) and
#    f = -p_surf (dipole). On a plane wave, p = z₀ · vn, so a monopole
#    expressed in pressure units differs from a dipole expressed in
#    velocity units by z₀². Equivalently: C_q · dx³ / (dS_in · dt) = ρc²
#    = κ (bulk modulus) — the natural per-cell-volume conversion that
#    appears in ∂_t P = κ · ∇·V.
#  • How found: empirically the C_q / C_f ratio matched z₀² to numerical
#    precision in every iteration (v1 → v4). Independently confirmed in
#    diagnostics/check_vq_two_way.jl across a (ρ, c) sweep to rel-RMS 1e-7.
#
# ─── Sign convention ─────────────────────────────────────────────────────
# `C_f` and `C_q` are positive scalar coefficients (units only, no sign).
# The two K-H sign sources (linearised Euler `∂_n p = −ρ ∂_t v_n` and the
# De Hoop dipole equivalence `f = −p · n̂`) appear at exactly one place each:
#
#   K-H summation (Part 3 — `run_part3_for_N`, kernel × scale products):
#       scale_pv_q  =  −ρ · dS · dt        ← q-channel, EULER-substitution minus
#       scale_pv_f  =  −1 · dS · dt        ← f-channel, DE-HOOP-dipole minus
#       scale_pin_q =  (1/c) · dS · dt     ← pin q-leg (Euler rewrite, no minus)
#       scale_pin_f =  −1 · dS · dt        ← pin f-leg, DE-HOOP-dipole minus
#   Kernel definitions in `eval_one_kernel` carry no baked-in sign:
#       p_v ≡ +∂_n G,    v_v ≡ +∂_t ∂_n G / ρ
#   So the K-H summation reads, term-by-term in De Hoop form:
#       p_inner  = ∮[ −ρ · p_p · v_n   −   p_v · p ] dS · dt
#                     ╰── (a) Euler ──╯   ╰── (b) De Hoop ─╯
#   Both minuses are physical; neither cancels the other.
#
# Direct injection (Part 2 + Part 3b Run B FDTD) — no explicit minus on
# `C_f`:
#       vn_src(txs_on_grid) = +C_f · p_recorded     (f-channel)
#       p_src( txs_on_grid) = +C_q · vn_recorded    (q-channel, q = +v_n)
# The De Hoop `f = −p · n̂` minus at this site is implicit in the
# combined convention `(outward-pointing inner_nrm) × (FDTD vn-channel
# sign)` — these two conventions together absorb one sign flip, and
# adding ANOTHER explicit `−` on `C_f` empirically breaks cancellation
# (the two injection channels destruct instead of constructively
# radiating −slab_A). The cancellation diagnostic is therefore
# `‖slab_A + slab_B‖` (addition).
#
# Verified at N=300 via a sign-flip sweep:
#       baseline (+C_f):  rms_A = 0.8562  rms_B = 0.8531  rms_residual = 0.04384   suppression = 19.53×
#       flipped  (−C_f):  rms_A = 0.8562  rms_B = 0.5002  rms_residual = 1.121     suppression = 0.7638×
# With `−C_f` the channels destruct (rms_B halves) AND the residual
# *exceeds* rms_A — the field is amplified inside the inner disk
# rather than cancelled.
#
# Production K-H path (`src/greens_io.jl` `load_gf_ref` /
# `extrapolate_pv_from_pv`) carries the K-H representation theorem's
# leading `−∮` as `s[src][rec] *= -1` at greens_io.jl:440 for the
# two-way (`pv_from_pv`) form.
#
# ============================================================================
# Physical-anchor + literature map for every scale constant
# ============================================================================
# Per-constant pointer into the textbook / paper that establishes the factor.
# Use this when arguing in the manuscript that nothing in α / the K-H formula
# is fit-by-eye. Citations are the canonical ones (verify keys against the
# project Zotero before quoting in text).
#
#   dS_in = 4π·R_in² / N
#       Surface element per Fibonacci-sphere source point.
#       Anchor: representation theorem is a SURFACE INTEGRAL (Helmholtz 1860;
#       Kirchhoff 1882; Pierce "Acoustics" §4-5; Morse & Ingard "Theoretical
#       Acoustics" §7.1). Fibonacci/equal-area sphere sampling is standard
#       low-discrepancy quadrature on S² — see e.g. González 2010, "Measurement
#       of Areas on a Sphere Using Fibonacci and Latitude-Longitude Lattices",
#       Math. Geosci. 42.
#
#   dt  (FDTD time step, dt = cf·dx/(c·√3))
#       Anchor: Yee 1966, "Numerical solution of initial boundary value
#       problems involving Maxwell's equations in isotropic media", IEEE
#       Trans. AP-14. Stability bound: CFL (Courant–Friedrichs–Lewy 1928).
#       Standard treatment for acoustics: Taflove & Hagness "Computational
#       Electrodynamics" §4 (analogous wave-equation operator), and
#       Virieux 1986 for the elastic case in geophysics.
#
#   1/(ρ·dx³)  (inverse cell mass — only present in C_f)
#       Anchor: linearised Euler momentum equation ρ·∂_t v = -∇p
#       (Pierce §1.5; Morse & Ingard §6.1). FDTD's per-cell V-update
#       multiplies the integrated force by 1/(ρ·V_cell) = 1/(ρ·dx³) to
#       advance velocity by one step (Yee staggered scheme).
#
#   z₀² = (ρ·c)²  (impedance² — ratio C_q / C_f)
#       Anchor: plane-wave impedance p = ρc·vn — Pierce §1.6; Morse &
#       Ingard §6.4. The factor connects the two K-H equivalent-source
#       channels: q = vn (monopole) carries pressure units after κ = ρc²
#       multiplication; f = -p (dipole) carries velocity units after 1/(ρc²)
#       multiplication.  Independently verified to rel-RMS 1e-7 in
#       diagnostics/check_vq_two_way.jl across a (ρ, c) sweep.
#
#   K-H two-way representation (`pv_from_pv`, p_inner + vn_inner via 4 kernels)
#       Anchor: classical interior Helmholtz integral
#           p(x) = ∮[ G·∂_n p − ∂_n G·p ] dS
#       with ∂_n p = -ρ·∂_t v_n (linearised Euler). Derivations:
#         • Helmholtz 1860 (frequency-domain),
#         • Kirchhoff 1882 (time-domain),
#         • Morse & Ingard §7.1, Pierce §4-5 (modern textbooks),
#         • Wapenaar & Berkhout 1989 §3 (seismic time-domain form),
#         • Fokkema & van den Berg 1993 §3 ("Seismic Applications of
#           Acoustic Reciprocity") — same form, sign conventions matched.
#       This is the form used in the manuscript and the code's primary path.
#
#   K-H one-way representation (`pv_from_pin`, single kernel × incoming p)
#       Anchor: incoming/outgoing decomposition of the surface field —
#       p = p_in + p_out, with the one-way assumption p_out ≈ 0 on the
#       receiving surface.  Derivations in:
#         • Berkhout 1982 "Seismic Migration: Imaging of Acoustic Energy
#           by Wave Field Extrapolation" Vol A §2-3,
#         • Wapenaar & Berkhout 1989 §4 (formal one-way operators),
#         • Wapenaar 1998 "Reciprocity properties of one-way propagators"
#           Geophysics 63 (signed convention check).
#       In a homogeneous medium one-way p_in obeys p_in = (p − z₀·v_n)/2;
#       substituting back into the full K-H formula collapses two
#       convolutions into one. This is `load_gf_ref`'s production default.
#
#   Source-equivalence (q = vn, f = -p)
#       Anchor: Equivalent-source representations of the Kirchhoff–Helmholtz
#       integral, used to convert surface-pressure / surface-velocity data
#       into volume monopoles / dipoles at injection time. Standard
#       derivation:
#         • Schenck 1968 "Improved Integral Formulation for Acoustic
#           Radiation Problems" J. Acoust. Soc. Am. 44 (CHIEF; surface
#           sources),
#         • Burton & Miller 1971 "The application of integral equation
#           methods to the numerical solution of some exterior boundary
#           value problems" Proc. R. Soc. A. 323 (combined surface
#           source / well-posedness on closed surfaces),
#         • Williams "Fourier Acoustics" §8 (modern statement).
#       The "-" on f is what makes our s_opt land at +1 (see sign block
#       above).
#
#   Equal-area / Fibonacci sphere quadrature
#       Anchor: low-discrepancy sampling of S² for surface integrals.
#         • Marques & Bouville 2013 "Spherical Fibonacci point sets for
#           illumination integrals", Computer Graphics Forum 32,
#         • Hardin & Saff 2004 "Discretizing manifolds via minimum energy
#           points", Notices AMS 51,
#         • plus the González 2010 reference cited under dS_in.
#       Convergence rate of the K-H quadrature in N is studied empirically
#       via the cps sweep above (s_opt vs cps).
#
# Cross-check: every constant either (a) appears as-is in a standard textbook
# (Pierce, Morse & Ingard, Yee, Taflove), or (b) was verified independently
# in this codebase's check_* test scripts (check_vq_two_way for z₀²,
# check_gf_scale for K_FDTD, this script for C_f). No factor is fit.
# ============================================================================

# ============================================================================
# Barton equivalence: Eq.(1) of the Letter from the time-domain Helmholtz
# integral via linearised Euler
# (resolves the TODO at lib/sup.tex:~267 of the Overleaf project;
#  cross-checked programmatically in python/mdd/tests/test_barton_units.py)
# ============================================================================
#
# Goal: verify that the Letter's Eq.(1)
#     p(x,t) = ∫dt' ∫dS' [ G^{p|q} ∗ v_n + G^{p|f} ∗ p ]
# is dimensionally consistent with the textbook (Barton 1989 ch. 5)
# time-domain retarded Helmholtz integral, and that the hidden factor
# implied by the linearised-Euler substitution is the same -ρ·∂_t that
# surfaces explicitly as ρc²·Δt in the discrete C_q above. No factor
# missing; no factor double-counted.
#
# ─── 1. Setup — Barton's time-domain retarded Helmholtz integral ────────
#
# For x in the interior of a closed surface S enclosing a source-free
# volume, Barton 1989 ch. 5 (also Pierce §4-5; Morse & Ingard §7.1):
#
#     p(x,t) = ∮_S [ G ∗ ∂_n p  -  ∂_n G ∗ p ] dt' dS'
#
# with G the bare 3D retarded wave-equation Green's function,
#     G(x-x', t-t') = δ(t - t' - |x-x'|/c) / (4π|x-x'|),
# units [G] = 1/(s·m). ∂_n is the outward normal derivative on S;
# ∗ denotes time convolution.
#
# ─── 2. Apply the linearised Euler equation ────────────────────────────
#
# Linearised momentum on the surface, projected onto the outward normal:
#     ∂_n p = -ρ · ∂_t v_n   (v_n = v · n̂, scalar per the surface
#                              convention introduced at main.tex:104)
#
# Substituting into Barton:
#     p(x,t) = ∮_S [ G ∗ (-ρ ∂_t v_n)  -  ∂_n G ∗ p ] dt' dS'
#            = ∮_S [ (-ρ ∂_t G) ∗ v_n  +  (-∂_n G) ∗ p ] dt' dS'
# (the ∂_t moves onto G by parts in the convolution; boundary terms
# at ±∞ vanish for any finite-duration recording).
#
# ─── 3. Match Eq.(1) of the Letter ──────────────────────────────────────
#
# Compare with
#     p(x,t) = ∫dt' ∫dS' [ G^{p|q} ∗ v_n + G^{p|f} ∗ p ]
# →
#     G^{p|q}  =  -ρ · ∂_t G_bare           units: kg / (m^4 · s^2)
#     G^{p|f}  =  -∂_n G_bare                units: 1 / (s · m^2)
#
# The hidden -ρ · ∂_t factor inside G^{p|q} is precisely the ρc² factor
# that surfaces explicitly in the discrete C_q = ρc² · Δt · dS_in/Δx³
# (the time derivative discretises to 1/Δt, then the ρc²·Δt Yee prefactor
# absorbs both).
#
# ─── 4. Dimensional balance — every step LHS = RHS = Pa ─────────────────
#
#   [G_bare]    = 1 / (s · m)                 (retarded GF, time domain)
#   [∂_t G]     = 1 / (s² · m)
#   [∂_n G]     = 1 / (s · m²)
#   [ρ]         = kg / m³
#   [v]         = m / s
#   [p]         = kg / (m · s²)  = Pa
#   [dt']       = s
#   [dS']       = m²
#
# Monopole integrand × dt' × dS':
#   [ρ · ∂_t G · v · dt · dS] = (kg/m³)·(1/(s²·m))·(m/s)·s·m² = kg/(m·s²) = Pa ✓
#
# Dipole integrand × dt' × dS':
#   [∂_n G · p · dt · dS]     = (1/(s·m²))·(kg/(m·s²))·s·m² = kg/(m·s²) = Pa ✓
#
# Eq.(1) LHS = [p] = Pa ✓
#
# ─── 5. Interior form (eq:kh in the SI) ─────────────────────────────────
#
# For x_0 in the interior of S, the canonical exterior representation
# evaluated inside yields -p (see main.tex:107):
#
#     p^IBC(x_0,t)  =  -∮_S [ G^{p,q} ∗ v_n  +  G^{p,f_n} ∗ p ] dt' dS'
#
# Same kernels as Eq.(1) (G^{p,q} ≡ G^{p|q} = -ρ ∂_t G_bare;
# G^{p,f_n} ≡ G^{p|f} · n_i = -∂_n G_bare · n_i). The only difference
# between Eq.(1) and eq:kh is the leading sign from interior-vs-exterior
# evaluation. The hidden -ρ·∂_t factor is accounted for the same way in
# both; nothing is silently changed in the kernels between the two forms.
#
# ─── 6. Discrete bridge — verifies the C_f / C_q closed forms above ─────
#
# Discretise:
#     ∂_t G → multiplication by Δt (in the FDTD time-stepping coupling
#               q to a pressure increment: ∂_t p = ρc²·q ⇒ Δp = ρc² Δt q)
#     ∂_n G → no extra Δ; the surface integral becomes Σ × ΔS × Δt; the
#               surface delta δ_S becomes the per-cell density ΔS/Δx³.
#
# Then
#     C_q = ρc²·Δt · dS_in/Δx³   ← matches G^{p|q}'s hidden ρ·∂_t × surface
#     C_f =  Δt/ρ · dS_in/Δx³    ← matches G^{p|f}'s discrete Yee prefactor
#
# C_q/C_f = ρ²c² = z₀², matching the impedance-squared identity above.
#
# ─── 7. Conclusion ──────────────────────────────────────────────────────
#
# Eq.(1), eq:kh, and the eq:C-coef boxed pair derive consistently from
# Barton 1989 ch. 5 via the linearised Euler substitution. The implicit
# -ρ · ∂_t factor in G^{p|q} (Eq.(1) and eq:kh) corresponds exactly to
# the ρc²·Δt Yee prefactor in C_q on the discrete side. No factor is
# silently missing or double-counted across Barton → Eq.(1) → eq:kh →
# eq:C-coef. The programmatic assertion lives in
# python/mdd/tests/test_barton_units.py.
#
# References:
#   • Barton, "Elements of Green's Functions and Propagation" (1989) ch. 5
#   • Pierce, "Acoustics: An Introduction to its Physical Principles
#       and Applications" §4-5
#   • Morse & Ingard, "Theoretical Acoustics" §7.1
# ============================================================================

# ============================================================================
# Body-force / surface-pressure equivalence (De Hoop / Williams)
# resolves the main-text reference at main.tex line ~167 of the
# Overleaf project: "the body-force/surface-pressure identification on
# f", citing hoopHandbookRadiationScattering1995 and williams_fourier_2000.
# ============================================================================
#
# The Kirchhoff–Helmholtz integral represents the wave field at an
# observation point as a surface integral of two source-strength
# densities on a closed surface S. Translating these surface densities
# into RHS source terms of the linearised wave equations gives two
# identifications, only one of which is intuitive.
#
# ─── Monopole identification (intuitive) ────────────────────────────────
#
# Continuity equation:  ∂_t p + ρc² ∇·v = ρc² q
#                                          ^^^
#                                          volume-injection-rate source [s⁻¹]
#
# A surface monopole layer of normal-velocity strength v_n is exactly a
# volume source density q concentrated on the surface: q_eq = v_n · δ_S.
# Physical picture: a surface layer pushing fluid normally outward at
# rate v_n pumps mass at exactly the rate q would. This identification
# is obvious from continuity alone.
#
# ─── Dipole identification (non-intuitive — De Hoop / Williams) ─────────
#
# Momentum equation:  ∂_t v + ρ⁻¹ ∇p = ρ⁻¹ f
#                                       ^^^
#                                       body-force volume density [N/m³]
#
# A surface pressure layer of strength p is equivalent to a body-force
# volume density along the outward normal: f_eq = p · n̂ · δ_S. Physical
# picture: a pressure layer on the outside of S pushes the medium
# inward, which is exactly what a body-force layer in the -n̂ direction
# does. The "equivalence" is not obvious from first principles -- it
# comes from working out what RHS source in the momentum equation
# reproduces the field a recorded pressure layer would emit.
#
# References:
#   • de Hoop 1995, "Handbook of Radiation and Scattering of Waves" §28
#     (canonical source-equivalence statement in the De Hoop convention)
#   • Williams 2000, "Fourier Acoustics" §8.4 (modern derivation;
#     Helmholtz integral → equivalent dipole layer)
#   • Schenck 1968, "Improved Integral Formulation for Acoustic
#     Radiation Problems", J. Acoust. Soc. Am. 44 (original surface-
#     source treatment for the CHIEF method)
#   • Burton & Miller 1971, "The application of integral equation
#     methods to the numerical solution of some exterior boundary
#     value problems", Proc. R. Soc. A. 323
#
# ─── Sign note ──────────────────────────────────────────────────────────
#
# The textbook (reproduction) form is f_eq = -p · n̂ · δ_S; substituting
# this into the KH integral reproduces the recorded interior field. The
# IBC use case (cancellation) flips this sign: f_eq = +p · n̂ · δ_S, so
# the injection radiates the negative of the recorded field and the
# cancellation residual ‖slab_A + slab_B‖ vanishes inside S^I. See the
# "Sign convention" block earlier in this file for the full story.
#
# In the SCALAR-normal-component surface convention used in the Letter
# (main.tex line 104), the outward normal is implicit and we write
# f = p on the surface (not f = p · n̂). The vector form is recovered by
# multiplying by n̂.
# ============================================================================

"C_f, C_q for a given N. Closed-form analytical scaling — no empirical knob."
function C_for_N(N)
    dS_in = 4π * R_INNER^2 / N
    C_f   = dS_in * dom_pw.dt / (dom_pw.r0 * dom_pw.dx^3)   # vn[:src] ← p_inner
    C_q   = C_f * dom_pw.z0^2                                # p[:src]  ← vn_inner
    return (; C_f, C_q, dS_in)
end

# ---- 1D sanity check: initial wavefield along propagation direction --------
# Plots p(x, y=0, z=0) at t=0 on dom_pw with outer/inner sphere + box-wall
# overlays. Verifies the Ricker fits inside dom_pw (xmax may have been changed
# by --dx snapping).
