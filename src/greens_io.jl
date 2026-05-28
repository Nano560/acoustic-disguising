# -----------------------------------------------------------------------------
# Green's-function I/O: load_gf (dispatch), load_gf_ref (impulsive +
# analytical reference GFs), load_gf_mdd (MDD-extracted GFs), and their HDF5
# readers. The save side of the FDTD-recorded ⇄ physical-units round-trip
# lives in this file too (gf_unit_factors / gf_to_physical / gf_to_raw_fdtd)
# so the two halves can't drift independently.
#
# Pulled out of src/io.jl: ~600 lines, one cohesive concern. Generic I/O
# (load_config, parse_flag, save_h5/load_h5, …) stays in src/io.jl.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Green's-function loader — unified dispatch
# -----------------------------------------------------------------------------

"""
    load_gf(path, dom, cfg;
            gf_method::Symbol = :analytical,
            gf_of::GFContent.T = GFContent.heterogeneous,
            tmax::Real = 0.0,
            homogeneous_path = nothing,
            extrap_type::AbstractString = "pv_from_pin") -> GFMap

Single entry-point for all three Green's-function methods. `gf_method` ∈
(`:analytical`, `:impulsive`, `:mdd`), listed in increasing computational cost:

  - `:analytical` → closed-form GFs from `scripts/greens/analytical.jl`.
                    Same schema as `:impulsive` but in physical units; the
                    per-kernel α factors are applied automatically when the
                    H5 `backend = "analytical"` attribute is present.
  - `:impulsive`  → reference GFs from `scripts/greens/impulsive.jl`
                    (4 components: p_p, p_v, v_p, v_v).
  - `:mdd`        → MDD-extracted GFs from `scripts/greens/mdd_extract.py`
                    (2 components: p_p, v_p — same on-disk schema as
                    impulsive/analytical, but only the pressure-source
                    kernels since MDD inverts the incoming-pressure form).

`gf_of` ∈ (`GFContent.heterogeneous`, `GFContent.scattered`);
`GFContent.scattered` requires `homogeneous_path` (subtracts the
homogeneous-medium GF from the heterogeneous one to give the scatterer's
contribution). Not to be confused
with `scatterer` in `impulsive_gfs` (src/greens.jl), which selects the
physical geometry (`:none`/`:sphere`/`:cube`/`:cross`) the GFs were
generated under.

`extrap_type` selects the boundary-extrapolation form — `"pv_from_pin"`
(one-way incoming-pressure form, the production default) or `"pv_from_pv"`
(two-way Kirchhoff representation; `:analytical`/`:impulsive` only — MDD
extracts the one-way form only). See `docs/extrapolation_conventions.md`.

Returns a `GFMap` (concrete subtype per `extrap_type`) consumed by
`forward_onestep!`. Internally dispatches to `load_gf_ref` (for
`:analytical` and `:impulsive`) or `load_gf_mdd`.
"""
function load_gf(path::AbstractString, dom::Domain, cfg;
                 gf_method::Symbol = :analytical,
                 gf_of::GFContent.T = GFContent.heterogeneous,
                 tmax::Real = 0.0,
                 homogeneous_path = nothing,
                 extrap_type::AbstractString = "pv_from_pin")
    if gf_method === :mdd
        # MDD extracts only the 2-component incoming-pressure form, so the
        # two-way "pv_from_pv" extrapolation is not available for it.
        extrap_type == "pv_from_pin" || error(
            "load_gf: gf_method=:mdd supports only extrap_type=\"pv_from_pin\"; " *
            "got $(repr(extrap_type)).")
        return load_gf_mdd(path, dom, cfg;
                           gf_of = gf_of, tmax = tmax, homogeneous_path = homogeneous_path)
    elseif gf_method === :analytical || gf_method === :impulsive
        # `load_gf_ref` auto-detects analytical mode via the H5 `backend`
        # attribute, so :analytical and :impulsive share the same code path.
        return load_gf_ref(path, dom, cfg;
                           gf_of = gf_of, tmax = tmax, homogeneous_path = homogeneous_path,
                           extrap_type = extrap_type)
    else
        error("load_gf: unknown gf_method=:$gf_method. Expected :analytical, :impulsive, or :mdd.")
    end
end

# -----------------------------------------------------------------------------
# Green's-function loaders (per-format, called by `load_gf`).
#
# `load_gf_ref` reads impulsive / analytical reference GFs (4 components
# per (rec, src) — p_p, p_v, v_p, v_v); `load_gf_mdd` reads the 2-component
# pin form produced by the Python MDD stage. Both apply a surface-area +
# injection + `e` coefficient scaling chain so the FDTD GFs are usable as
# Kirchhoff-Helmholtz weights in `build_hologram`.
# -----------------------------------------------------------------------------

"""
    load_gf_ref(path, dom, cfg; gf_of = GFContent.heterogeneous, tmax = 0.0, homogeneous_path = nothing) -> GFMap

Load impulsive-source reference Green's functions from the HDF5 file at
`path`, interpolate to `dom.dt`, and apply the surface-area / impedance
scalings described in the paper's appendix.

Keyword arguments:
  - `gf_of` ∈ (`GFContent.heterogeneous`, `GFContent.scattered`).
    `GFContent.scattered` subtracts the homogeneous-medium companion GF
    loaded from `homogeneous_path`; if `homogeneous_path === nothing`
    when `GFContent.scattered` is requested the function errors.
  - `tmax`: if > 0, truncate/extend the interpolated time axis to `0:dom.dt:tmax`.
    Defaults to the original GF duration.
  - `extrap_type` ∈ (`"pv_from_pin"`, `"pv_from_pv"`, `"p_from_pin"`).
    Selects which `GFMap` concrete subtype is returned (`PVFromPin`,
    `PVFromPV`, or `PFromPin`); default `"pv_from_pin"` (the one-way
    production form). See `docs/extrapolation_conventions.md`.

The HDF5 input is expected to contain the datasets `p_p`, `p_v`, `v_p`, `v_v`
(each `(nt, nsrc, nrec)`), a `t` time axis, and the attribute `dom_dt`
(time step used when the GFs were generated).

Returns a `GFMap` (concrete subtype per `extrap_type`) consumed by
`forward_onestep!`. The returned struct's `iRec`/`iSrc` fields are empty;
the caller assigns them before kicking off the FDTD loop.
"""

"""
    _scaling_factors(dom, dom_dx, dom_dt, dt_in, cfg) -> (C_p, C_v, dx_gf)

FDTD source-injection convention factors. `interp_trilinear!` deposits a
source additively (`field += w·tf` per step); the cell volume dx³ and the
per-step `/dt` set the continuous source density. A pressure source is a
term S_p in the discrete continuity equation; a velocity source a term s_v
in the momentum equation. Carried into the 2nd-order pressure wave equation,

    ∂ₜ²p − c²∇²p = ∂ₜS_p − ρc²∇·s_v,

the pressure-source forcing is +∂ₜS_p and the velocity-source forcing is
−ρc²∇·s_v — the minus and the ρc² inherited from the continuity equation
∂ₜp = −ρc²∇·v. Free-space GF convolution then gives the FDTD-recorded →
physical amplitude ratios:

    C_p =  dx_gf³ · dt_save_gf / (dt_fdtd_gf² · c²)   (pressure-source kernels)
    C_v = −ρ · dx_gf³ · dt_save_gf / dt_fdtd_gf²      (velocity-source kernels)

so C_v/C_p = −ρc²: the velocity-source kernels' opposite sign is the
continuity-equation coupling, not a bookkeeping convention. Full
derivation in docs/extrapolation_conventions.md.

All on-disk GF files now live in *physical* units (analytical's natural
convention; impulsive divides by α at save time — see
scripts/greens/{analytical,impulsive}.jl). Multiply by the appropriate α
in the loader to bring kernels onto the FDTD-recorded amplitude convention
that the downstream `e` chain expects. K_FDTD = C_p still plays the
cancellation-diagnostic role; the FDTD-vs-analytical verification gives
1-6% per-kernel residual and 0.10% RMS hologram round-trip.
"""
function _scaling_factors(dom::Domain, dom_dx, dom_dt, dt_in::Float64, cfg)
    # `dom_dx` should be present on all files written by the current Julia
    # save path (impulsive/analytical both write it). Cfg fallback retained
    # only as a last resort with a loud warning.
    dx_gf = if isnan(dom_dx)
        cfg_gf  = apply_overrides(cfg, "greens")
        n_gf    = Int(cfg_gf["grid"]["n"])
        xmax_gf = Float64(cfg_gf["grid"]["xmax"])
        dx_cfg  = 2 * xmax_gf / (n_gf - 1)
        @warn "_scaling_factors: H5 file had no `dom_dx` attr; using cfg-derived dx" dx_cfg
        dx_cfg
    else
        dom_dx
    end
    C_p = dx_gf^3 * dt_in / (Float64(dom_dt)^2 * dom.c0^2)
    C_v = -dom.r0 * dx_gf^3 * dt_in / Float64(dom_dt)^2
    return C_p, C_v, dx_gf
end

"""
    gf_unit_factors(dom, dt_save) -> (C_p, C_v)

Save-side equivalent of `_scaling_factors`: when the caller knows the GFs
were generated under `dom` directly (impulsive / analytical scripts), the
factors collapse to a closed form using only the caller's Domain. See the
docstring of `_scaling_factors` for the derivation.

    C_p =  dx³ · dt_save / (dt_fdtd² · c²)
    C_v = −ρ · dx³ · dt_save / dt_fdtd²

Save side divides by these (raw → physical); load side multiplies
(physical → raw).
"""
function gf_unit_factors(dom::Domain, dt_save::Float64)
    C_p = dom.dx^3 * dt_save / (dom.dt^2 * dom.c0^2)
    C_v = -dom.r0 * dom.dx^3 * dt_save / dom.dt^2
    return C_p, C_v
end

"""
    gf_to_physical(state, C_p, C_v) -> NamedTuple

Return a NamedTuple of GFs in physical units (FDTD-recorded → physical).
Divides the four kernel arrays by the appropriate factor and threads the
time / source / receiver arrays through unchanged.
"""
function gf_to_physical(state, C_p::Float64, C_v::Float64)
    return (;
        p_p = state.p_p ./ Float32(C_p),
        p_v = state.p_v ./ Float32(C_v),
        v_p = state.v_p ./ Float32(C_p),
        v_v = state.v_v ./ Float32(C_v),
        t   = state.t,
        src_positions = state.src_positions,
        rec_positions = state.rec_positions,
    )
end

"""
    gf_to_raw_fdtd(loaded, C_p, C_v) -> NamedTuple

Inverse of `gf_to_physical`: physical units → raw FDTD-recorded units.
"""
function gf_to_raw_fdtd(loaded, C_p::Float64, C_v::Float64)
    return (;
        p_p = loaded.p_p .* Float32(C_p),
        p_v = loaded.p_v .* Float32(C_v),
        v_p = loaded.v_p .* Float32(C_p),
        v_v = loaded.v_v .* Float32(C_v),
        t   = loaded.t,
        src_positions = loaded.src_positions,
        rec_positions = loaded.rec_positions,
    )
end

function load_gf_ref(path::AbstractString, dom::Domain, cfg;
                     gf_of::GFContent.T = GFContent.heterogeneous, tmax::Real = 0.0,
                     homogeneous_path = nothing,
                     extrap_type::AbstractString = "pv_from_pin")
    @info "load_gf_ref" path gf_of tmax extrap_type
    extrap_type in ("pv_from_pin", "pv_from_pv", "p_from_pin") || error(
        "load_gf_ref: extrap_type must be \"pv_from_pin\", \"pv_from_pv\", or " *
        "\"p_from_pin\"; got $(repr(extrap_type)).")

    if gf_of === GFContent.scattered && homogeneous_path === nothing
        error("load_gf_ref: gf_of=GFContent.scattered requires `homogeneous_path` to the homogeneous-medium GF file.")
    end

    # Surface-area elements for the Kirchhoff-Helmholtz integral.
    # `:injection` is the FDTD-source → physical-GF rescaling, applied below
    # as `e`.
    s = cfg["surfaces"]
    scalesIn = Dict(
        :src => fib_sphere_area(s["radius_inner"], s["nPoints_inner"]),
        :rec => fib_sphere_area(s["radius_outer"], s["nPoints_outer"]),
        :injection => 1.0,
    )

    gf_heterogeneous, dom_dt, dom_dx, backend, units_convention = _read_gf_ref_h5(path)
    units_convention == "physical" || error(
        "load_gf_ref: $path has units_convention=$(repr(units_convention)); " *
        "expected \"physical\". Files written before the units-convention " *
        "change need to be regenerated (no back-compat path).")

    # Scattered-only: subtract homogeneous companion.
    if gf_of === GFContent.scattered
        gf_homogeneous, _, _, _, _ = _read_gf_ref_h5(homogeneous_path)
        gf = Dict(
            :p => Dict(
                :p => gf_heterogeneous[:p][:p] .- gf_homogeneous[:p][:p],
                :v => gf_heterogeneous[:p][:v] .- gf_homogeneous[:p][:v],
            ),
            :v => Dict(
                :p => gf_heterogeneous[:v][:p] .- gf_homogeneous[:v][:p],
                :v => gf_heterogeneous[:v][:v] .- gf_homogeneous[:v][:v],
            ),
            :t => gf_homogeneous[:t],
        )
    else
        gf = gf_heterogeneous
    end

    # Interpolate onto dom.dt-spaced time axis. HDF5 returns `gf[:t]` as a
    # Vector, but `Interpolations.scale` needs an AbstractRange for each
    # dimension — so reconstruct the range from its endpoints.
    t_raw = gf[:t]
    dt_in = Float64(t_raw[2] - t_raw[1])
    dt_out = dom.dt
    t_end = tmax > 0 ? Float64(tmax) : Float64(t_raw[end])
    tin = range(Float64(t_raw[1]), Float64(t_raw[end]); length = length(t_raw))
    tout = 0:dt_out:t_end

    # FDTD source-injection convention factors C_p / C_v — derivation +
    # docstring in `_scaling_factors`.
    C_p, C_v, dx_gf = _scaling_factors(dom, dom_dx, dom_dt, dt_in, cfg)

    @info "load_gf_ref: scaling physical → FDTD-recorded" C_p C_v dx_gf dt_fdtd_gf=dom_dt dt_save_gf=dt_in backend
    gf[:p][:p] .*= Float32(C_p)   # pressure rec ← p source
    gf[:v][:p] .*= Float32(C_p)   # vn rec ← p source
    gf[:p][:v] .*= Float32(C_v)   # pressure rec ← v source
    gf[:v][:v] .*= Float32(C_v)   # vn rec ← v source

    for i in (:p, :v), j in (:p, :v)
        # Cubic in time; axes 2/3 are evaluated at integer indices only, so
        # their order is immaterial. Cubic reduces the resampling error
        # that contributes to the ~0.7% K-H pipeline volume-RMS floor
        # (Linear was the original choice and bandwidth-limits f'(τ) at
        # the dt_save_gf rate).
        gf[i][j] = resample_time(gf[i][j], tin, tout; kind = :cubic)
    end

    # Sanity check.
    for i in (:p, :v), j in (:p, :v)
        v = Array(gf[i][j])
        if any(isinf, v) || any(isnan, v)
            @warn "load_gf_ref: Inf/NaN in GF tensor" rec=i src=j
        end
    end

    map = Dict{Symbol,Any}(
        :iRec => Int[],
        :iSrc => Int[],
        :gf => Dict(
            :p => Dict(
                :p => Data.Array(gf[:p][:p]),
                :v => Data.Array(gf[:p][:v]),
            ),
            :v => Dict(
                :p => Data.Array(gf[:v][:p]),
                :v => Data.Array(gf[:v][:v]),
            ),
        ),
    )

    gf = map[:gf]

    # FDTD-recording → injection bridge. K_FDTD is C_p (see top of function).
    # The previous empirical calibration (K_FDTD ≈ 1.46e-7) sat within 9% of
    # the analytical value, with the gap attributable to the cancellation
    # diagnostic's structural residual (numerical dispersion + finite-aperture
    # K-H truncation), not to a missing factor.
    e = (dom.c0^2 * dom_dt^2 * dom.dt) / (C_p * dom.dx^3)

    # Per-(src, rec) scale accumulator.
    s = Dict(
        :p => Dict(:p => 1.0, :v => 1.0),
        :v => Dict(:p => 1.0, :v => 1.0),
    )

    # Extrapolation type: "pv_from_pv" (two-way Kirchhoff representation),
    # "pv_from_pin" (one-way injected in-going pressure — the production
    # default), or "p_from_pin". Selected by the `extrap_type` kwarg (already
    # in scope); see docs/extrapolation_conventions.md. The concrete `GFMap`
    # subtype is constructed from the scaled `gf` arrays at the bottom of
    # this function.

    # ------------------------------------------------------------------
    # Impedance conversion: bring each GF kernel channel into the FDTD
    # source-injection units of the field it will drive. This is the only
    # *physical* unit conversion in load_gf_ref's scaling chain — C_p/C_v
    # and e (above) are FDTD discretisation and the surface weights are
    # K-H quadrature, whereas z₀ = ρc here is the genuine v→q / p→f
    # conversion. z₀ is the specific acoustic impedance (plane-wave ratio
    # p = z₀·vₙ); the K-H representation theorem keeps it implicit in the
    # two Green's functions (Fokkema & van den Berg 1993; van Manen et al.
    # 2007, Eq. 3) — the one-way factorisation used here makes it explicit.
    #
    # The GFs were measured outer-source → inner-receiver and are consumed
    # inner-source → outer-receiver by reciprocity (src/greens.jl), so a
    # kernel's two channels act at consumption as (emission source,
    # recording receiver). A velocity (`:v`) channel that drives the
    # monopole q — injected into the PRESSURE field — is ×z₀ (velocity →
    # pressure); a pressure (`:p`) channel that drives the dipole f —
    # injected into the VELOCITY field — is ÷z₀ (pressure → velocity).
    #
    # "pv_from_pv" (two-way) keeps both kernel indices live, so both the
    # receiver-side and source-side channels are converted, and the
    # trailing ×(−1) is the bounded-domain Kirchhoff–Helmholtz sign.
    # "pv_from_pin" (one-way) has already reduced the recorded field to
    # its incoming-pressure constituent p_in (a pressure), so only the
    # source-side conversion remains. See docs/extrapolation_conventions.md.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # pv_from_pv vs pv_from_pin — equivalence  (HYPOTHESIS, 2026-05-26)
    #
    # In the continuum the two forms should produce the SAME interior
    # field from any outer-surface recording, by linearity:
    #   • Pure-incoming (vn_outer = −p_outer/z₀):  both reduce to
    #     ∮[(1/c)·p_p − p_v]·p_outer dS·dt — direct K-H reconstruction.
    #   • Pure-outgoing (vn_outer = +p_outer/z₀):  both must return 0.
    #     pin trivially (p_in = (p − z₀·vn)/2 = 0); pv via the K-H
    #     representation theorem (interior K-H integral over a closed
    #     surface vanishes for fields sourced inside that surface).
    #   • Arbitrary field: linear combination of the above ⇒ equivalence.
    #
    # Empirical (N-sweep at the paper-scale run):
    #     N    | pv:p_err  pv:vn_err | pin:p_err  pin:vn_err
    #     200  | 0.0255    0.289     | 0.178      0.254
    #     400  | 0.0220    0.288     | 0.178      0.254
    #     700  | 0.0219    0.288     | 0.178      0.254
    # vn matches; p-channel shows ~8× pin error. The per-time-step
    # rel-RMS curves (diagnostics/_check_inject_kh/part3_kh_extrapolation.jl)
    # overlay on vn and sit a flat ~10× apart on p during the signal
    # window — a uniform multiplicative offset, not a transient or drift.
    #
    # Hypothesis H1 (sphere-aliasing): expanding p_in into (p − z₀·vn)/2
    # turns pin's outgoing-component cancellation into a 4-term discrete
    # sum vs pv's 2-term sum, so aliasing residual should be larger for
    # pin and should shrink with N.
    # ⟶ FALSIFIED by the N-sweep: pin's p-error is 0.178 at N=200, 400
    #   AND 700 — completely flat. (pv also flattens after N=400; the
    #   N=200 dip on pv is small-N undersampling.) Sphere sampling is
    #   NOT the bottleneck for EITHER form past N ≈ 400.
    #
    # Hypothesis H2 (FDTD-grid floor, current): both pin and pv hit
    # N-independent residual floors set by something other than sphere
    # sampling — most likely FDTD-grid discretization (dx, dt: Yee
    # staggering offsets p vs vn samples; numerical dispersion acts
    # slightly differently on the two channels) and/or analytical-kernel
    # time-grid resolution (σ_T, dt_KH). The 8× gap reflects how pin's
    # combination p_in = (p − z₀·vn)/2 amplifies the relative grid-level
    # error between p and vn by z₀ ≈ 1.5e6, while pv keeps p and vn in
    # separate kernel branches and tolerates the same error differently.
    # The vn-channel sits at the v_v F-pedestal floor in both forms (see
    # Part 3a kernel comment) and is therefore equivalent.
    #
    # Status: H2 not yet tested. Next discriminator would be an
    # FDTD-dx sweep on Part 3a — H2 predicts pin's floor shrinks as
    # dx → 0 while pv's floor shrinks more slowly. If true, the 8× gap
    # is a numerical-dispersion fingerprint, not a fundamental property
    # of the one-way decomposition.
    #
    # Production consequence: MDD (forced to pin via load_gf_mdd) carries
    # the ~0.18 p-channel floor; analytical/impulsive (pv-capable) gets
    # ~0.022. Switching the production default from pin to pv would help
    # only the analytical/impulsive path; MDD stays constrained.
    # ------------------------------------------------------------------
    if extrap_type == "pv_from_pv"
        # receiver-side (recording-receiver channel) impedance conversion
        for src in (:p, :v)
            s[src][:p] *= dom.z0
            s[src][:v] /= dom.z0
        end
        # source-side (emission-source channel) impedance conversion
        for rec in (:p, :v)
            s[:p][rec] /= dom.z0
            s[:v][rec] *= dom.z0
        end
        # bounded-domain Kirchhoff–Helmholtz representation sign
        for src in (:p, :v), rec in (:p, :v)
            s[src][rec] *= -1
        end

    elseif extrap_type == "pv_from_pin"
        # source-side impedance conversion: v-channel ×z₀ → monopole q,
        # p-channel ÷z₀ → dipole f (the v→q / p→f physical conversion)
        for rec in (:p, :v)
            s[:p][rec] /= dom.z0
            s[:v][rec] *= dom.z0
        end
    end

    # Injection scale
    for src in (:p, :v), rec in (:p, :v)
        s[src][rec] *= scalesIn[:src] * scalesIn[:rec] * scalesIn[:injection] * e
    end

    # Apply scales
    for src in (:p, :v), rec in (:p, :v)
        gf[src][rec] .*= s[src][rec]
    end

    if extrap_type == "pv_from_pv"
        return PVFromPV(gf[:p][:p], gf[:p][:v], gf[:v][:p], gf[:v][:v])

    elseif extrap_type == "pv_from_pin"
        p_pin = gf[:p][:p] - gf[:p][:v] / dom.z0
        v_pin = gf[:v][:p] - gf[:v][:v] / dom.z0
        return PVFromPin(p_pin, v_pin)

    elseif extrap_type == "p_from_pin"
        p_pin = gf[:p][:p] - gf[:p][:v] / dom.z0
        v_pin = gf[:v][:p] - gf[:v][:v] / dom.z0

        pos = (p_pin + v_pin * dom.z0) / 2
        neg = (p_pin - v_pin * dom.z0) / 2
        return PFromPin(pos, neg)
    end
end

"""
    _read_gf_ref_h5(path) -> (gf_dict, dom_dt)

Helper for `load_gf_ref`: read a four-component reference GF file into the
nested `Dict(:p=>Dict(:p,:v), :v=>Dict(:p,:v), :t=>...)` shape. Returns the
dict together with the `dom_dt` attribute needed for the scaling constant.

Expected datasets: `p_p`, `p_v`, `v_p`, `v_v`, `t`.
Expected root attribute: `dom_dt`.
"""
function _read_gf_ref_h5(path::AbstractString)
    isfile(path) || error("load_gf_ref: GF file not found: $path")
    local p_p, p_v, v_p, v_v, t, dom_dt, dom_dx, backend, units_convention
    h5open(path, "r") do f
        p_p = read(f["p_p"])
        p_v = read(f["p_v"])
        v_p = read(f["v_p"])
        v_v = read(f["v_v"])
        t   = read(f["t"])
        # HDF5.attrs(f)["name"] returns the scalar directly (no `read` needed).
        dom_dt = haskey(HDF5.attrs(f), "dom_dt") ? HDF5.attrs(f)["dom_dt"] :
                 (length(t) > 1 ? (t[2] - t[1]) : error("load_gf_ref: cannot infer dom_dt"))
        dom_dx = haskey(HDF5.attrs(f), "dom_dx") ? Float64(HDF5.attrs(f)["dom_dx"]) : NaN
        backend = haskey(HDF5.attrs(f), "backend") ? String(HDF5.attrs(f)["backend"]) : ""
        # Asserted by load_gf_ref. Both impulsive and analytical pipelines
        # write `physical` from scripts/greens/{impulsive,analytical}.jl;
        # files predating that change have no attr and are rejected.
        units_convention = haskey(HDF5.attrs(f), "units_convention") ?
                           String(HDF5.attrs(f)["units_convention"]) : ""
    end
    gf = Dict(
        :p => Dict(:p => Float32.(p_p), :v => Float32.(p_v)),
        :v => Dict(:p => Float32.(v_p), :v => Float32.(v_v)),
        :t => t,
    )
    return gf, dom_dt, dom_dx, backend, units_convention
end

"""
    load_gf_mdd(path, dom, cfg; gf_of = GFContent.heterogeneous, tmax = 0.0, homogeneous_path = nothing) -> PVFromPin

Load MDD-extracted Green's functions from the HDF5 file at `path`,
interpolate to `dom.dt`, and apply surface-area / impedance scalings.
The MDD GFs already represent the K-H "incoming-pressure" response, so
the returned `GFMap` is always a `PVFromPin` (`p_pin` and `v_pin`
fields, translated from the on-disk `p_p`/`v_p` datasets — see
`_read_gf_mdd_h5`).

Keyword arguments mirror `load_gf_ref`. `GFContent.scattered` requires `homogeneous_path`.

The HDF5 input is expected to contain `p_p`, `v_p` (each
`(nt, nsrc, nrec)`) and a `t` time axis — same on-disk schema as
`scripts/greens/{analytical,impulsive}.jl`.

The returned struct's `iRec`/`iSrc` fields are empty; the caller assigns
them before kicking off the FDTD loop.
"""
function load_gf_mdd(path::AbstractString, dom::Domain, cfg;
                     gf_of::GFContent.T = GFContent.heterogeneous, tmax::Real = 0.0,
                     homogeneous_path = nothing)
    @info "load_gf_mdd" path gf_of tmax

    if gf_of === GFContent.scattered && homogeneous_path === nothing
        error("load_gf_mdd: gf_of=GFContent.scattered requires `homogeneous_path` to the homogeneous-medium GF file.")
    end

    # Surface-area elements for the K-H integral. Same as load_gf_ref:
    # `scalesIn[:injection] = 1.0` (no FDTD-injection compensation here);
    # `e = 1.0` (no MDD-empirical bridge here). Both former MDD-only
    # factors (`1/dx²` injection and `1/c0` empirical) are now applied at
    # save time in python/mdd/io.py::save_gfs, so this loader carries
    # exactly the same scaling chain as load_gf_ref's pin-form path
    # (modulo the α multiplications and pin extraction that don't apply
    # to MDD's already-pin-form output).
    s = cfg["surfaces"]
    scalesIn = Dict(
        :src => fib_sphere_area(s["radius_inner"], s["nPoints_inner"]),
        :rec => fib_sphere_area(s["radius_outer"], s["nPoints_outer"]),
        :injection => 1.0,
    )

    # Heterogeneous MDD GFs.
    gf_heterogeneous = _read_gf_mdd_h5(path)

    if gf_of === GFContent.scattered
        gf_homogeneous = _read_gf_mdd_h5(homogeneous_path)
        gf = Dict(
            :p_pin => gf_heterogeneous[:p_pin] .- gf_homogeneous[:p_pin],
            :v_pin => gf_heterogeneous[:v_pin] .- gf_homogeneous[:v_pin],
            :t     => gf_homogeneous[:t],
        )
    else
        gf = gf_heterogeneous
    end

    tin  = range(0, gf[:t][end], length = size(gf[:t], 1))
    t_end = tmax > 0 ? Float64(tmax) : Float64(gf[:t][end])
    tout = 0:dom.dt:t_end

    for i in (:p_pin, :v_pin)
        gf[i] = resample_time(gf[i], tin, tout; kind = :linear)
    end

    for i in (:p_pin, :v_pin)
        v = Array(gf[i])
        if any(isinf, v) || any(isnan, v)
            @warn "load_gf_mdd: Inf/NaN in GF tensor" field=i
        end
    end

    # Local scratch Dict; only the scaling math needs the nested-key shape.
    # The final return constructs a `PVFromPin` from the scaled arrays.
    gf = Dict(
        :p => Dict(:pin => Data.Array(gf[:p_pin])),
        :v => Dict(:pin => Data.Array(gf[:v_pin])),
    )

    e = 1.0

    s = Dict(
        :p => Dict(:pin => 1.0),
        :v => Dict(:pin => 1.0),
    )

    # source swap: v-channel ×z₀ → monopole q, p-channel ÷z₀ → dipole f
    s[:p][:pin] /= dom.z0
    s[:v][:pin] *= dom.z0

    # Injection scale
    for src in (:p, :v), rec in (:pin,)
        s[src][rec] *= scalesIn[:src] * scalesIn[:rec] * scalesIn[:injection] * e
    end

    for src in (:p, :v), rec in (:pin,)
        gf[src][rec] .*= s[src][rec]
    end

    return PVFromPin(gf[:p][:pin], gf[:v][:pin])
end

"""
    _read_gf_mdd_h5(path) -> Dict

Helper for `load_gf_mdd`: read an MDD GF file into the two-component
`Dict(:p_pin, :v_pin, :t)` shape. The on-disk schema matches
`scripts/greens/{impulsive,analytical}.jl` (datasets `p_p` and `v_p`,
both in physical units), so this loader just reads them straight in —
the Python `mdd.cli` is responsible for the `vnz / z0` conversion at
save time.

Expected datasets: `p_p`, `v_p`, `t`.
"""
function _read_gf_mdd_h5(path::AbstractString)
    isfile(path) || error("load_gf_mdd: GF file not found: $path")
    local p_p, v_p, t, units_convention, empirical_scale
    h5open(path, "r") do f
        p_p = read(f["p_p"])
        v_p = read(f["v_p"])
        t   = read(f["t"])
        units_convention = haskey(HDF5.attrs(f), "units_convention") ?
                           String(HDF5.attrs(f)["units_convention"]) : ""
        # Empirical amplitude calibration. save_gfs writes 1.0 on every
        # fresh extraction; `diagnostics/check_gf_scale.jl --write-scale` can
        # update this attribute after a calibration pass. Multiplied
        # straight onto the tensors before they leave this function so
        # downstream loaders never need to know about it.
        empirical_scale = haskey(HDF5.attrs(f), "empirical_scale") ?
                          Float64(HDF5.attrs(f)["empirical_scale"]) :
                          error("_read_gf_mdd_h5: $path missing `empirical_scale` attr. " *
                                "Re-extract via scripts/greens/mdd_extract.py to add it.")
    end
    units_convention == "physical" || error(
        "_read_gf_mdd_h5: $path has units_convention=$(repr(units_convention)); " *
        "expected \"physical\". Files written before the pylops→impulsive " *
        "conversion landed in python/mdd/io.py::save_gfs need to be " *
        "regenerated (no back-compat path). The Python save now produces " *
        "p_p/v_p in the same 1/(m·s²) physical convention as analytical / " *
        "impulsive, so this loader matches `_read_gf_ref_h5`'s contract.")
    if empirical_scale != 1.0
        @info "load_gf_mdd: applying empirical_scale" path=relpath(path) empirical_scale
    end
    s = Float32(empirical_scale)
    # Internal dict keys keep the `_pin` suffix to communicate "MDD pin form"
    # to the rest of `load_gf_mdd`; downstream hologram synthesis treats
    # these as the incoming-pressure source variants.
    return Dict(
        :p_pin => Float32.(p_p) .* s,
        :v_pin => Float32.(v_p) .* s,
        :t     => t,
    )
end
