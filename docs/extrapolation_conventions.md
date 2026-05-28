# Boundary-extrapolation conventions

This note documents the sign and amplitude conventions of the Green's-function
boundary extrapolation used in hologram synthesis: the FDTD source-injection
scale factors `C_p` / `C_v`, the relationship between the one-way and two-way
extrapolation forms, and how the discrete extrapolation relates to the
Kirchhoff–Helmholtz representation.

> **Naming convention.** All numerical FDTD coefficients use the
> **Schneider C-prefix** (Schneider, *Understanding the FDTD Method*,
> Ch. 12). Two distinct pairs of coefficients live in this codebase:
>
> - `C_p`, `C_v` — *recording-conversion* factors, used here. They
>   convert physical-unit GF kernels into FDTD-recorded amplitudes (or
>   vice versa) at GF save/load time.
> - `C_f`, `C_q` — *injection* coefficients. They convert recorded
>   surface fields into FDTD source increments at hologram injection
>   time.
>
> Both pairs are correct and necessary; they are at different pipeline
> stages.

## The two extrapolation forms

The pipeline records pressure `p` and normal velocity `vₙ` on the outer
control surface Sᴼ and uses pre-computed Green's functions to drive the inner
emitting surface Sᴵ. Two forms are implemented, selected by `load_gf`'s
`extrap_type` keyword (`src/kernels/extrapolation.jl`):

- **`pv_from_pv`** — the two-way Kirchhoff representation
  `Σ[G^{p|q}·vₙ + G^{p|f}·p]·ΔS`, evaluated from the four reference kernels
  `p_p, p_v, v_p, v_v`.
- **`pv_from_pin`** — the one-way form: the recorded field is first reduced to
  its incoming-pressure constituent `p_in = (p − z₀vₙ)/2`, which is then
  propagated with the two combined kernels `p_pin = p_p − p_v/z₀` and
  `v_pin = v_p − v_v/z₀`.

`pv_from_pin` is the **production path**, used by all three GF methods
(analytical, impulsive, MDD). `pv_from_pv` is available for the
analytical/impulsive methods only — MDD extracts the incoming-pressure form
directly, so its on-disk files carry only `p_p, v_p`.

## FDTD source-injection factors `C_p`, `C_v`

The reference Green's functions are stored in physical units; `load_gf_ref`
multiplies them by per-kernel factors to bring them onto the FDTD-recorded
amplitude convention. With `dx` the GF-stage grid spacing, `dt_save` /
`dt_fdtd` the saved / FDTD time steps, `ρ` the density and `c` the sound
speed:

```
C_p =  dx³·dt_save / (dt_fdtd²·c²)     (kernels p_p, v_p — pressure source)
C_v = −ρ·dx³·dt_save / dt_fdtd²        (kernels p_v, v_v — velocity source)
```

The **sign** of `C_v` is not a bookkeeping convention — it follows from the
governing equations. The FDTD integrates the first-order acoustic system

```
continuity:  ∂ₜp = −ρc²·(∇·v)
momentum:    ∂ₜv = −(1/ρ)·∇p
```

A source is injected additively into a field array (`interp_trilinear!`), so a
**pressure** source is a term `S_p` in the *continuity* equation and a
**velocity** source a term `s_v` in the *momentum* equation. Eliminating `v`
gives the second-order pressure wave equation with both source types:

```
∂ₜ²p − c²∇²p  =  ∂ₜS_p  −  ρc²·(∇·s_v)
```

The pressure-source forcing is `+∂ₜS_p`; the velocity-source forcing is
`−ρc²·∇·s_v`. The velocity source inherits the minus sign **and** the `ρc²`
from the `−ρc²∇·v` term of the continuity equation — hence `C_v/C_p = −ρc²`.
The opposite sign of the velocity-source kernels is therefore the
continuity-equation coupling of a momentum-equation source, not an arbitrary
convention. (Pressure-source kernels also carry one extra time derivative,
`∂ₜ`, where velocity-source kernels carry a divergence `∇·` — this is why the
closed-form `p_p` kernel carries `f′` and `p_v` carries `f`; see
`scripts/greens/analytical.jl`.)

## What each scaling factor is — FDTD discretisation vs. physical conversion

`load_gf_ref` (`src/io.jl`) multiplies the physical-unit GF kernels by a chain
of factors before they are convolved with the recorded boundary field to
produce the injected source amplitudes `q` (monopole, into the pressure field)
and `f` (dipole, into the velocity field). For the production path
(`pv_from_pin`) the two consumed kernels factor *exactly* as

```
gf[:v][:pin] = (ΔSᴵ·ΔSᴼ·e·C_p) · z₀     · (v_p + c·v_v)   → drives q (monopole)
gf[:p][:pin] = (ΔSᴵ·ΔSᴼ·e·C_p) · (1/z₀) · (p_p + c·p_v)   → drives f (dipole)
```

using `C_v = −ρc²·C_p` and `ρc²/z₀ = c`, with `p_p, p_v, v_p, v_v` the on-disk
physical kernels resampled to the synthesis time step. Every factor falls into
exactly one of three categories:

| Factor | Role | Category |
|--------|------|----------|
| `C_p`, `C_v` (`C_v/C_p = −ρc²`) | physical → FDTD-recorded amplitude | FDTD discretisation |
| `e` | GF-recording → synthesis-injection bridge | FDTD discretisation |
| `ΔSᴵ·ΔSᴼ` (`4πr²/N` per sphere) | Kirchhoff–Helmholtz surface quadrature | geometry / quadrature |
| **`z₀ = ρc`** (×z₀ on `q`, ÷z₀ on `f`) | **v→q / p→f impedance conversion** | **physical** |
| `½` and the `c`-weighted kernel sum | incoming-field (one-way) decomposition | physical |

`q` is computed as `gf[:v][:pin] ⊛ p_in`. The kernel `gf[:v][:pin]` is the
*normal-velocity* channel — its convolution with `p_in` yields a velocity —
yet `q` is a monopole *added to the pressure field*. The `×z₀` factor closes
that gap (velocity → pressure); symmetrically `÷z₀` converts the
pressure-channel kernel `gf[:p][:pin]` into the velocity-field dipole `f`.
This `z₀` is the **only** factor that differs between the `q`- and `f`-channels
and the only genuinely physical unit conversion in the chain — applied in the
impedance-conversion block of `load_gf_ref` (the loop commented "source-side
impedance conversion"). `C_p`, `C_v`, `e` are FDTD discretisation; `ΔSᴵ·ΔSᴼ`
is the K-H quadrature; neither carries a `z₀`.

**Physical origin of `z₀`.** The factor is not new physics: it is the specific
acoustic impedance `z₀ = ρc` — the plane-wave ratio `p = z₀vₙ` (e.g. Pierce
2019, Ch. 1). In the Kirchhoff–Helmholtz representation theorem (Fokkema & van
den Berg 1993; van Manen et al. 2007, Eq. 3) the same impedance is **implicit**:
the representation drives the monopole layer with normal velocity and the
dipole layer with pressure through two distinct Green's functions — `G^q`
(pressure from a volume-injection `q` source, contracted with `vₙ`) and
`Γ_k^q` (particle motion from a `q` source, contracted with `p`) — whose
differing units silently carry the velocity↔pressure conversion. The IBC
literature inherits this two-Green's-function structure (van Manen et al. 2005,
2007; Broggini et al. 2017) and does not surface `z₀` separately in the
injection formula. The one-way factorisation of the scaling chain used here
makes the same impedance appear explicitly — the same `z₀` that defines the
one-way wavefield decomposition `p_in = ½(p − z₀vₙ)` (Wapenaar & Grimbergen
1996; Wapenaar 1998). Surfacing it is a clarity choice, not a correction.

**For the manuscript.** A clean scalar relation `q = c_q·vₙ`, `f = c_f·p` does
*not* hold — the map from the recorded field to `{q, f}` is a Green's-function
convolution (`q = gf[:v][:pin] ⊛ p_in`, `f = gf[:p][:pin] ⊛ p_in`,
`p_in = ½(p − z₀vₙ)`). What *is* honest in a main-text equation: the
monopole/dipole pair is obtained from the incoming constituent `p_in` by the
extrapolation kernels, and the velocity → monopole and pressure → dipole
conversions carry the impedance `z₀ = ρc`. The discretisation factors `C_p`,
`C_v`, `e` and the quadrature weights `ΔS` belong in a discretisation appendix.

The factorisation is verified numerically by re-deriving the scaling chain
independently and confirming the production kernels equal
`(ΔSᴵ·ΔSᴼ·e·C_p)·z₀^{±1}·(one-way kernel)` to within floating-point tolerance.

## Equivalence of the one-way and two-way forms

Write the one-way combined kernel as `G_pin = G^{p|f} − G^{p|q}/z₀` and the
incoming constituent as `p_in = (p − z₀vₙ)/2`. The one-way extrapolation
output expands to

```
G_pin·p_in = ½[ G^{p|f}·p − z₀·G^{p|f}·vₙ − G^{p|q}·p/z₀ + G^{p|q}·vₙ ]
```

For a **purely incoming** field on Sᴼ the impedance relation `p = −z₀vₙ` holds
(an inward-travelling wave; `vₙ` is the *outward* normal velocity).
Substituting `vₙ = −p/z₀`:

```
G_pin·p_in  →  G^{p|f}·p − G^{p|q}·p/z₀
```

which is **identically** the two-way form `Σ[G^{p|f}·p + G^{p|q}·vₙ]` evaluated
at the same `vₙ = −p/z₀`. Therefore:

> The one-way (`pv_from_pin`) and two-way (`pv_from_pv`) extrapolations produce
> the same field whenever the boundary data is purely incoming. They differ
> only by an outgoing constituent — which the one-way form discards by
> construction. For an immersive boundary that is the intended behaviour: only
> the incoming field is reproduced inside Sᴵ.

`p = −z₀vₙ` is exact for plane waves and in the high-`kr` limit; for curved
wavefronts it holds to `O(1/kr)`. Correspondingly, `p_in = (p − z₀vₙ)/2` is
the normal-incidence / high-`kr` approximation of the exact modal
incoming-pressure projector — the spherical-harmonic separation in
[`wavefield_separation.md`](wavefield_separation.md) is its exact form.

## Cost — why `pv_from_pin` is the production path

The extrapolation runs at every FDTD time step. `pv_from_pv` convolves four GF
kernels per output point; `pv_from_pin` convolves two. So `pv_from_pin` does
**half** the extrapolation arithmetic and holds **half** the GF-kernel memory.
Because the two forms are equivalent for the incoming field, `pv_from_pin` is
the natural production choice — the same hologram at half the extrapolation
cost.

## Relation to the Kirchhoff–Helmholtz representation

For an observation point *interior* to a closed surface, the bounded-domain
Kirchhoff representation theorem (Fokkema & van den Berg 1993, Eq. 7.82)
carries the opposite sign to the exterior case (their Eq. 7.63), when both use
the surface's outward normal. The boundary-extrapolation code is consistent
with this, by the following:

- **Normals.** Both control surfaces are built by `fibonacci_sphere`
  (`src/illumination.jl`) with *outward* unit normals; no inward-normal or
  sign-flipped variant is used anywhere.
- **No explicit sign.** The discrete extrapolation sum
  (`extrapolate_pv_from_pin!`) carries no explicit `−1`.
- **One-way, not two-way.** The production path does not evaluate the two-way
  representation at an interior point. It reduces the recorded field to its
  incoming constituent `p_in` and propagates *that* from Sᴼ inward to Sᴵ.
  Forward propagation of a one-way (incoming) field is causal and
  sign-unambiguous, so the exterior/interior `±` of the two-way representation
  theorem does not enter.
- **Kernels.** `G^{p|q}`, `G^{p|f}` are the FDTD-measured (or MDD-extracted, or
  closed-form analytic) boundary responses, brought to a common amplitude
  convention by `C_p`, `C_v` above — they are not the bare free-space Green's
  function and its normal derivative.

A statement suitable for the manuscript, reconciling Eq. 1 and Eq. 2:

> Equation (2) propagates the incoming-pressure constituent
> `p_in = (p − z₀vₙ)/2` of the recorded boundary field from Sᴼ inward to Sᴵ.
> It is the one-way reduction of the two-way representation, Eq. (1) — exact
> for the incoming field and equal to it whenever the boundary field is purely
> incoming. The bounded-domain sign of the representation theorem (Fokkema &
> van den Berg, Eq. 7.82) does not enter Eq. (2): forward propagation of the
> one-way incoming field carries no exterior/interior sign ambiguity.

## References

- Broggini, F., Vasmel, M., Robertsson, J. O. A., & van Manen, D.-J. (2017).
  Immersive boundary conditions: Theory, implementation, and examples.
  *Geophysics* 82(3), T97–T110. doi:10.1190/geo2016-0458.1.
- Fokkema, J. T., & van den Berg, P. M. (1993). *Seismic Applications of
  Acoustic Reciprocity.* Elsevier, Amsterdam.
- Pierce, A. D. (2019). *Acoustics: An Introduction to Its Physical Principles
  and Applications,* 3rd ed. Springer, Cham.
- van Manen, D.-J., Robertsson, J. O. A., & Curtis, A. (2005). Modeling of wave
  propagation in inhomogeneous media. *Physical Review Letters* 94, 164301.
- van Manen, D.-J., Robertsson, J. O. A., & Curtis, A. (2007). Exact wave field
  simulation for finite-volume scattering problems. *Journal of the Acoustical
  Society of America* 122(4), EL115–EL121. doi:10.1121/1.2771371.
- Wapenaar, C. P. A. (1998). Reciprocity properties of one-way propagators.
  *Geophysics* 63(5), 1795–1798. doi:10.1190/1.1444473.
- Wapenaar, C. P. A., & Grimbergen, J. L. T. (1996). Reciprocity theorems for
  one-way wavefields. *Geophysical Journal International* 127(1), 169–177.
