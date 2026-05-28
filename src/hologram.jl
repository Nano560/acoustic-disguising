# -----------------------------------------------------------------------------
# Hologram synthesis from Green's functions.
#
# `build_hologram` drives the FDTD forward solve with the initial plane-wave
# field and the pre-extracted Green's functions (either reference/impulsive
# or MDD) to reconstruct the hologram on the outer surface.
# -----------------------------------------------------------------------------

using LinearAlgebra

# -----------------------------------------------------------------------------
# HologramMode — singleton dispatch tag for build_hologram's three flavours.
# Mirrors the GFMap pattern (src/kernels/extrapolation.jl): construction
# instead of Symbol tagging makes the union closed and enables method
# dispatch on the per-mode logic.
# -----------------------------------------------------------------------------

"""
    HologramMode

Abstract type tagging which extrapolation path `build_hologram` takes.
Three singletons:

  - `RealMode()` — no GF extrapolation; pure FDTD driven by the initial
    plane wave + scatterer mask (`va`).
  - `RefMode()`  — extrapolation using `load_gf_ref` output
    (analytical / impulsive).
  - `MDDMode()`  — extrapolation using `load_gf_mdd` output.

`build_hologram`'s `implementation` kwarg takes a `HologramMode` instance.
CLI / config callsites that have a Symbol should funnel through
`hologram_mode(s::Symbol)` to convert.
"""
abstract type HologramMode end
struct RealMode <: HologramMode end
struct RefMode  <: HologramMode end
struct MDDMode  <: HologramMode end

"""
    hologram_mode(s::Union{Symbol,AbstractString}) -> HologramMode

Parse a Symbol or String into the corresponding `HologramMode` singleton.
Use at the CLI / config parse boundary; downstream code holds the
`HologramMode` value directly.
"""
function hologram_mode(s::Symbol)
    s === :real && return RealMode()
    s === :ref  && return RefMode()
    s === :mdd  && return MDDMode()
    error("hologram_mode: unknown :$s (expected :real, :ref, or :mdd)")
end
hologram_mode(s::AbstractString) = hologram_mode(Symbol(s))

# Canonical stringification — used for HDF5 attribute writes and log
# messages. `String(::HologramMode)` (capital S) also routes here.
Base.string(::RealMode) = "real"
Base.string(::RefMode)  = "ref"
Base.string(::MDDMode)  = "mdd"

# GF-map preparation per mode. Ref and MDD share the index-population logic;
# Real skips extrapolation entirely.
function _prepare_gf_map(::Union{RefMode,MDDMode}, gf, outer_range, inner_range)
    gf === nothing && error("build_hologram: this implementation requires `gf` (use load_gf_ref/load_gf_mdd).")
    gf.iRec = collect(outer_range)
    gf.iSrc = collect(inner_range)
    return gf
end
_prepare_gf_map(::RealMode, _gf, _outer, _inner) = nothing

"""
    build_hologram(dom::Domain, cfg, gf;
                   scatterer::Symbol = :cross,
                   implementation::HologramMode = MDDMode(),
                   gf_of::GFContent.T = GFContent.heterogeneous,
                   fc::Real = 4000.0,
                   initial_distance::Real = 0.65,    # was 0.62 (used historically)
                   initial_amplitude::Real = 1.5,
                   va = nothing,
                   txs_extra::Vector = Transceiver[],
                   snapshot_every::Union{Nothing,Int} = nothing) -> NamedTuple

Synthesise the acoustic hologram by driving the FDTD forward solver with the
pre-computed Green's-function extrapolation operator `gf`.

`gf` is the `GFMap` returned by `load_gf_ref` / `load_gf_mdd`, OR `nothing`
for the `RealMode()` path where no extrapolation is performed and the
surfaces are simply passive receivers.

Keyword arguments:
  - `scatterer`    ∈ (`:none`, `:sphere`, `:cube`, `:cross`). Selects the
                     scatterer geometry referenced by `getPolygons`.
  - `implementation` :: `HologramMode` — `RealMode()` / `RefMode()` /
                     `MDDMode()`. `RefMode` / `MDDMode` use GF extrapolation;
                     `RealMode` runs bare FDTD with the transceiver list
                     acting only as receivers.
  - `gf_of`        ∈ (`GFContent.heterogeneous`, `GFContent.scattered`).
                     Propagated into the metadata; the branching itself happens
                     at `load_gf_*` time.
  - `fc`           : centre frequency of the initial Ricker wavefield [Hz].
  - `initial_distance` : offset of the initial plane wave in +x [m].
  - `initial_amplitude`: amplitude multiplier for the initial field.
  - `va`           : staircase scatterer mask dict (`"update"` key on the
                     device). Required when `implementation = RealMode()` and
                     `scatterer != :none`; ignored otherwise.
  - `txs_extra`    : additional `Transceiver`s to append to the inner/outer
                     lists (e.g. a point-source illumination). Empty by
                     default; was `pointSource` in the original.
  - `snapshot_every` : if set, record three orthogonal pressure slices every
                     `N` time steps into `xSlices` / `ySlices` / `zSlices`.
                     For each offset in `slice_offsets_m`, slice planes pass
                     through `(off, 0, 0)`, `(0, off, 0)`, `(0, 0, off)`.
                     `nothing` disables (zero overhead).
  - `slice_offsets_m`: vector of signed offsets of the cut planes from
                     origin, in metres. Each entry adds one set of
                     orthogonal slices to the snapshot. First entry is the
                     "primary" offset by convention (e.g. used by the
                     paper-figure renderer).

Returns `(; field_final, xSlices, ySlices, zSlices, snapshot_steps,
          slice_offsets_m, slice_indices, times, txs_on_grid, metadata)` where
  - `field_final`     :: `Array{Float32,4}` of the `(p, vx, vy, vz)` field at
                         `t = tmax`,
  - `xSlices`         :: `Vector{Vector{Matrix{Float32}}}`. Outer index =
                         offset (matches `slice_offsets_m`); inner index =
                         snapshot. Each matrix is pressure on the
                         x = offset plane, shape `(ny, nz)`. All-empty
                         when `snapshot_every === nothing`.
  - `ySlices`         :: same on y = offset plane (each `(nx, nz)`),
  - `zSlices`         :: same on z = offset plane (each `(nx, ny)`),
  - `snapshot_steps`  :: `Vector{Int}` of the FDTD step indices at which
                         each snapshot was captured (shared across offsets),
  - `slice_offsets_m` :: the requested offsets (`Vector{Float64}`),
  - `slice_indices`   :: `Vector{(; x, y, z)}` integer voxel indices used,
                         one entry per offset,
  - `times`           :: time axis `collect(t(dom))`,
  - `txs_on_grid`     :: the transceiver-grid Dict after the solve (pressure
                         and normal-velocity traces at each inner/outer point),
  - `metadata`        :: a `NamedTuple` with the keyword-argument values and
                         inner/outer index ranges.
"""
function build_hologram(dom::Domain, cfg, gf;
                        scatterer::Symbol = :cross,
                        implementation::HologramMode = MDDMode(),
                        gf_of::GFContent.T = GFContent.heterogeneous,
                        fc::Real = 4000.0,
                        initial_distance::Real = 0.65,    # was 0.62 (used historically)
                        initial_amplitude::Real = 1.5,
                        va = nothing,
                        txs_extra::Vector = Transceiver[],
                        snapshot_every::Union{Nothing,Int} = nothing,
                        slice_offsets_m::AbstractVector{<:Real} = [-0.3, 0.0])

    # `va` is honored whenever provided. `:real` with a non-`:none` scatterer
    # must have one (bare-FDTD has no other way to express the geometry).
    # `:ref` / `:mdd` accept `va` for the disguising / cloaking path —
    # interior physics from the mask, outer-surface drive from a different
    # scatterer's GF. `forward_onestep!` branches on `va === nothing`
    # independently of the GF extrapolation branch.
    if implementation isa RealMode && scatterer !== :none && va === nothing
        error("build_hologram: implementation=RealMode(), scatterer=:$scatterer requires `va` (build with AcousticDisguising.build_update_mask).")
    end
    va_use = va

    # -------------------------------------------------------------------------
    # Source-time function (empty — the surfaces are driven by the
    # extrapolation kernel, not by an explicit wavelet at t=0).
    # -------------------------------------------------------------------------
    rectf = zeros(dom.nt)

    # -------------------------------------------------------------------------
    # Inner / outer transceiver surfaces (Fibonacci sphere points).
    # -------------------------------------------------------------------------
    ill = illumination_points(cfg)

    directions_outer = hcat(zeros(size(ill.outer, 1)), ill.outer_normals)
    directions_inner = hcat(zeros(size(ill.inner, 1)), ill.inner_normals)

    Inner = Transceiver[]
    for (p, d) in zip(eachrow(ill.inner), eachrow(directions_inner))
        push!(Inner, Transceiver(dom; point=collect(p), direction=collect(d), tf=rectf))
    end

    Outer = Transceiver[]
    for (p, d) in zip(eachrow(ill.outer), eachrow(directions_outer))
        push!(Outer, Transceiver(dom; point=collect(p), direction=collect(d), tf=rectf))
    end

    txs = vcat(Outer, Inner, txs_extra)

    n_outer = Int(cfg["surfaces"]["nPoints_outer"])
    n_inner = Int(cfg["surfaces"]["nPoints_inner"])
    outer_range = 1:n_outer
    inner_range = (1:n_inner) .+ n_outer

    # -------------------------------------------------------------------------
    # CPML + transceiver packing.
    # -------------------------------------------------------------------------
    cpml = Cpml(dom;
                npml=Int(cfg["pml"]["n"]),
                rcoef=Float64(cfg["pml"]["rcoef"]),
                fc=Float64(cfg["pml"]["fc"]))

    txs_on_grid = transceiversToGrid(dom, txs)

    # -------------------------------------------------------------------------
    # Build Green's-function extrapolation map (passed as a kwarg to
    # `run_fdtd!`, not grafted onto `txs_on_grid`). `_prepare_gf_map` returns
    # `nothing` for `RealMode` (no extrapolation) and the index-populated
    # `gf` for `RefMode` / `MDDMode`.
    # -------------------------------------------------------------------------
    gf_map = _prepare_gf_map(implementation, gf, outer_range, inner_range)

    # -------------------------------------------------------------------------
    # Initial plane-wave field.
    # -------------------------------------------------------------------------
    field_host = initialField(dom, Float64(fc), Float64(initial_distance))
    field = Data.Array(field_host) * Float32(initial_amplitude)

    # -------------------------------------------------------------------------
    # Time-stepping loop.
    # -------------------------------------------------------------------------
    # Slice-only snapshots: store just the pressure on three orthogonal cut
    # planes at each world offset in slice_offsets_m. Each snapshot is then
    # ~3·n_offsets·n²·4 bytes ≈ 1.5 MB at n = 251, n_offsets = 2 — small
    # enough for a movie's worth of frames, vs. ~250 MB / snapshot for the
    # full 4·n³ field.
    cs = coords(dom)
    n_off    = length(slice_offsets_m)
    ix_slice = [argmin(abs.(cs[:x][:p] .- Float64(o))) for o in slice_offsets_m]
    iy_slice = [argmin(abs.(cs[:y][:p] .- Float64(o))) for o in slice_offsets_m]
    iz_slice = [argmin(abs.(cs[:z][:p] .- Float64(o))) for o in slice_offsets_m]

    # Outer index = offset, inner = snapshot.
    xSlices = [Matrix{Float32}[] for _ in 1:n_off]   # each (ny, nz)
    ySlices = [Matrix{Float32}[] for _ in 1:n_off]   # each (nx, nz)
    zSlices = [Matrix{Float32}[] for _ in 1:n_off]   # each (nx, ny)
    snapshot_steps = Int[]
    on_step = snapshot_every === nothing ? nothing :
        function (it, f)
            it % snapshot_every == 0 || return
            for k in 1:n_off
                push!(xSlices[k], Array(f[ix_slice[k], :, :, 1]))
                push!(ySlices[k], Array(f[:, iy_slice[k], :, 1]))
                push!(zSlices[k], Array(f[:, :, iz_slice[k], 1]))
            end
            push!(snapshot_steps, it)
        end
    run_fdtd!(dom, field, txs_on_grid, cpml, va_use; on_step = on_step, gf_map = gf_map)

    field_final = Array(field)

    metadata = (
        scatterer       = scatterer,
        implementation  = implementation,
        gf_of           = gf_of,
        fc              = Float64(fc),
        initial_distance = Float64(initial_distance),
        initial_amplitude = Float64(initial_amplitude),
        n_outer         = n_outer,
        n_inner         = n_inner,
        outer_range     = outer_range,
        inner_range     = inner_range,
    )

    return (
        field_final     = field_final,
        xSlices         = xSlices,
        ySlices         = ySlices,
        zSlices         = zSlices,
        snapshot_steps  = snapshot_steps,
        slice_offsets_m = collect(Float64, slice_offsets_m),
        slice_indices   = [(x = ix_slice[k], y = iy_slice[k], z = iz_slice[k])
                           for k in 1:n_off],
        times           = collect(t(dom)),
        txs_on_grid     = txs_on_grid,
        metadata        = metadata,
    )
end
