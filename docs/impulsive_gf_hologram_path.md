# Impulsive-GF hologram path — precise narrative

End-to-end trace of how the impulsive-Green's-function hologram-synthesis
pipeline implements the De Hoop Kirchhoff–Helmholtz representation, with
every scale factor named and dimensions explicit. Companion to
[`extrapolation_conventions.md`](extrapolation_conventions.md)
(the recording-conversion and surface-element scaling chain) and to the
comment block above `C_for_N` in
[`diagnostics/_check_inject_kh/injection_coefficients.jl`](../diagnostics/_check_inject_kh/injection_coefficients.jl)
(the De Hoop convention and the reproduction-vs-cancellation sign
discussion).

The v,p ↔ q,f surface-delta equivalence is stated explicitly at Stage A
and its appearance in the discrete chain is tracked through Stages B–E.

All numerical FDTD coefficients in this document follow the **Schneider
C-prefix convention** (Schneider, *Understanding the FDTD Method*, Ch. 12):

- `C_f`, `C_q` — *injection* coefficients (recorded surface field → FDTD
  source increment).
- `C_p`, `C_v` — *recording-conversion* coefficients (FDTD-recorded GF
  kernel → physical-unit GF kernel).

## Stage A — Continuous Kirchhoff–Helmholtz (no FDTD)

Linearised first-order coupled wave equations in the De Hoop convention
(Fokkema & van den Berg 1993; Vasmel & Robertsson 2016, Eq. 1):

```
∂_t p  +  ρc² ∇·v  =  ρc² · q          (continuity-like; monopole source q in s⁻¹)
∂_t v  +  ρ⁻¹ ∇p   =  ρ⁻¹ · f           (momentum;        body-force f in N/m³)
```

K-H representation of the interior field at `x₀` enclosed by the
emitting surface `S_emt`:

```
p̄^IBC(x₀, t)
   = −∮_{S_emt} [ G^{p,q}(x₀, xₛ, t) ⊛ vₙ(xₛ, t)
                + G^{p,f}_i(x₀, xₛ, t) nᵢ ⊛ p(xₛ, t) ] dS
```

The **v,p ↔ q,f equivalence** is a surface-delta-distribution statement:

```
q_eq(x, t) = +vₙ(xₛ, t) · δ_S(x)        units: (m/s)·(1/m) = s⁻¹    ✓
f_eq(x, t) = −p(xₛ, t) · n̂ · δ_S(x)    units: Pa·(1/m) = N/m³     ✓
```

The recorded surface fields `vₙ` and `p` are **not themselves** the
volume-source densities `q` and `f` — they are *surface* fields. The
equivalent volume-source-density that the wave equation accepts is the
surface field multiplied by the surface delta `δ_S` (and, for `f`, by
`−n̂`). The dimensions of (i) and (ii) above match automatically.

## Stage B — Impulsive-GF generation (forward FDTD)

File: [`scripts/greens/impulsive.jl`](../scripts/greens/impulsive.jl).
Recording-conversion factors: lines 133–148.

For each source point on the outer (recording) surface, run forward
FDTD with one of two source types injected at that point:

- **p-source** (impulse in the pressure register): produces kernels
  `g^p_p, g^p_v` recorded at the inner surface — the response to a
  unit pressure source.
- **v-source** (impulse in a velocity register): produces kernels
  `g^v_p, g^v_v` — response to a unit velocity source.

Raw recordings are in FDTD-stencil-implicit units. To convert to
physical units (so the kernel obeys the continuous K-H form):

```
g^p_p_phys  =  g^p_p_raw  /  C_p         C_p  =  Δx³ · Δt_save / (Δt_fdtd² · c²)
g^p_v_phys  =  g^p_v_raw  /  C_p
g^v_p_phys  =  g^v_p_raw  /  C_v         C_v  = −ρ · Δx³ · Δt_save / Δt_fdtd²
g^v_v_phys  =  g^v_v_raw  /  C_v
```

The minus sign in `C_v` comes from the continuity-equation coupling
(`∂_t p ∝ −ρc²∇·v`): a positive velocity source produces a recorded
pressure of opposite sign. This is the **same minus sign that lives on
`f = −p`** in the surface-delta equivalence (Stage A) — it's the
continuity-equation sign manifesting in the discrete chain.

Saved to disk: 4 physical-units kernels per outer–inner pair.

## Stage C — Hologram-synthesis load chain

File: [`src/io.jl`](../src/io.jl) `load_gf_ref`. Lines 447–563.

The recorded production field is in FDTD units; the GFs on disk are in
physical units. The load chain bridges them in five sub-steps:

### C1 — Multiply by `C_p`, `C_v` (line 447–454)

Convert physical → FDTD-recorded so the kernels match the recorded
outer field's units for convolution:

```julia
gf[:p][:p] *= C_p    gf[:v][:p] *= C_p     # p-source kernels
gf[:p][:v] *= C_v    gf[:v][:v] *= C_v     # v-source kernels
```

### C2 — FDTD-recording → injection bridge `e` (line 495)

```
e  =  c² · Δt_fdtd² · Δt_prod  /  (C_p · Δx³)
```

`e` converts "FDTD-recorded amplitude" → "FDTD source-injection
amplitude for the production grid", absorbing any
(Δt_gf vs Δt_prod, Δx_gf vs Δx_prod) differences between the
GF-generation and production grids.

### C3 — K-H surface elements (line 562, via `scalesIn`)

```
scalesIn[:src]  =  dS_in   =  4π R_in²  / N_in
scalesIn[:rec]  =  dS_out  =  4π R_out² / N_out
scalesIn[:injection]  =  1
```

Discrete quadrature weights on the two surfaces.

### C4 — Impedance conversion z₀ (lines 535–558)

The two K-H channels carry recorded fields in different units (`p` in
Pa, `vₙ` in m/s); to inject either back into the FDTD they have to be
converted to the destination slot's units:

- **v-channel** kernel injecting into the pressure slot:  × z₀
  (the `v → q` surface-delta equivalence in discrete form)
- **p-channel** kernel injecting into the velocity slot:  ÷ z₀
  (the `−p → f` surface-delta equivalence; the minus is absorbed into
  the sign convention in step C5)

For `pv_from_pv`, both receiver-side **and** source-side conversions
apply; for `pv_from_pin` only source-side (the one-way decomposition
has already collapsed the receiver-side z₀ into the
`p_in = (p − z₀·vₙ)/2` combination).

### C5 — Sign convention (line 547–549, two-way path only)

Multiply by `−1` for the bounded-domain K-H representation theorem
sign. This is the `f = −p` minus sign appearing at the kernel level.

### Final per-(src, rec) pre-factor on the GF tensor

```
gf_baked[src][rec]  =  (C_p or C_v) · dS_in · dS_out · e · z₀^±1 · (−1 if pv_from_pv)
```

## Stage D — Convolution + injection

Files: [`src/kernels/extrapolation.jl`](../src/kernels/extrapolation.jl),
[`src/kernels/forward.jl`](../src/kernels/forward.jl).

Convolve the pre-scaled GF tensor with the raw recorded outer field
(FDTD-recorded units) to produce an injection-ready source-time
function per inner point:

```
src_array[t, ir]  =  Σ_outer  gf_baked[ir, is](t')  ⊛_t  outer_recorded[t, is]
```

This array is added **directly** into the appropriate FDTD source slot
(`txs_on_grid[:p][:src]` for the q-channel; `txs_on_grid[:vn][:src]` for the
f-channel) via the trilinear injection kernel
(`src/kernels/derivatives.jl:55`), with no further multiplicative
scaling.

## Stage E — Equivalence to the direct-injection coefficients `C_f, C_q`

The direct-injection path bypasses the GF convolution: it takes the
**already-recorded inner-surface fields** (from a separate FDTD with
the same plane wave) and injects them via the typed accessors `vn_src`,
`p_src` from `AcousticDisguising`:

```
vn_src(txs_on_grid)  =  C_f · p_inner_recorded     (f-channel)
p_src( txs_on_grid)  =  C_q · vn_inner_recorded    (q-channel; q = +v_n)
```

with

```
C_f  =  dS_in · Δt / (ρ · Δx³)
C_q  =  C_f · z₀²  =  dS_in · Δt · ρ · c² / Δx³
```

The De Hoop body-force minus `f = −p · n̂` is **not** applied as an
explicit `−` on `C_f` here. The combined convention `(outward-pointing
inner_nrm) × (FDTD vn-channel sign)` absorbs the sign equivalence at
injection time; adding an explicit `−` on top empirically breaks the
cancellation (verified by direct experiment). The cancellation
diagnostic is `‖slab_A + slab_B‖` (addition) inside the tapered inner
disk — `slab_B` is sign-tuned to ≈ `−slab_A` so the sum goes to zero
in the interior. The De Hoop minus appears explicitly only at the K-H
summation step (the f-channel scale `scale_pv_b = −1`), not at the
direct-injection step.

The Yee × surface-conversion decomposition of these coefficients is

```
C_q  =  (Δt · ρ · c²) · (dS_in / Δx³)
C_f  =  (Δt / ρ)      · (dS_in / Δx³)
```

— i.e., **(Yee stencil prefactor) × (discrete surface-delta density)**.

The impulsive-GF hologram path (Stages B–D) is dimensionally equivalent
to this direct injection for any inner-surface field reconstructable
from the outer recording. Specifically, the load chain's combined
factor `C_p · e · dS_in · dS_out · z₀ (injecting into the pressure
slot)` reproduces `C_q · dS_out` (absorbing the outer-surface
quadrature) up to the K-H extrapolation error.

So the same Yee × surface-delta-density decomposition appears both
ways:

- **Direct injection** (`check_inject_kh.jl`): `C_q = (Δt·ρc²) ·
  dS_in/Δx³` is computed and applied explicitly via `C_for_N(N)`.
- **Impulsive-GF hologram synthesis** (`src/io.jl`): the equivalent
  `(Δt·ρc²) · dS_in/Δx³` structure is **baked into the GF tensor**
  via the `C_p · e · dS_in · z₀` chain, and the recorded outer field
  is convolved against it.

Both paths arrive at the same injection-amplitude per FDTD step
because both are discretisations of the same continuous K-H
representation of Stage A.

## Quick-reference summary table

| stage | what happens | factors that enter | continuous source → discrete? |
|---|---|---|---|
| A | continuous K-H, surface fields recorded | none (physical units) | `vₙ → q_eq = vₙ·δ_S`,  `−p → f_eq = −p·n̂·δ_S` |
| B | forward FDTD generates impulsive GFs | `C_p`, `C_v` (save-time divide) | recorded raw kernel → physical-unit kernel |
| C | load GFs, scale for production injection | `C_p`/`C_v` × `e` × `dS_in` × `dS_out` × `z₀^±1` × (−1) | physical-unit kernel → injection-ready GF tensor |
| D | convolve baked GF with recorded outer field; add to FDTD source slot | none beyond Stage C | one outer-recorded sample becomes one inner-injection increment per step |
| E | equivalent direct-injection path | `C_f = (Δt/ρ)·dS_in/Δx³`,  `C_q = (Δt·ρc²)·dS_in/Δx³` | same overall scaling, different intermediate decomposition |
