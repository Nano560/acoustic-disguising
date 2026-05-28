# -----------------------------------------------------------------------------
# FDTD wave-propagation core: Domain, grids, transceivers, CPML, interpolation,
# initial-field construction, and the `run_fdtd!` time-stepping driver.
# -----------------------------------------------------------------------------

using LinearAlgebra

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

"""
    Domain(; tmax=10e-4, xmax=1.0, ymax=1.0, zmax=1.0,
             nx=201, ny=201, nz=201,
             c0=1500.0, r0=1000.0, cf=0.25)

Computational FDTD domain.

Fields derived from the inputs: grid spacings `dx/dy/dz`, time step `dt` from
the CFL condition, number of time steps `nt`, pressure/velocity update factors
`fact_m0`/`fact_m1`, and acoustic impedance `z0 = r0 * c0`.

Default medium is water (c0 = 1500 m/s, ρ = 1000 kg/m³).
"""
struct Domain
    tmax::Float64
    xmax::Float64
    ymax::Float64
    zmax::Float64
    nx::Int
    ny::Int
    nz::Int
    c0::Float64
    r0::Float64
    cf::Float64
    dx::Float64
    dy::Float64
    dz::Float64
    fact_m0::Float64
    fact_m1::Float64
    dt::Float64
    nt::Int
    z0::Float64
end

function Domain(;
        tmax::Real = 10e-4,
        xmax::Real = 1.0,
        ymax::Real = 1.0,
        zmax::Real = 1.0,
        nx::Integer = 201,
        ny::Integer = 201,
        nz::Integer = 201,
        c0::Real = 1500.0,   # air: 340,   water: 1500
        r0::Real = 1000.0,   # air: 1.125, water: 1000
        cf::Real = 0.25,
    )
    tmax, xmax, ymax, zmax = Float64(tmax), Float64(xmax), Float64(ymax), Float64(zmax)
    nx, ny, nz             = Int(nx),       Int(ny),       Int(nz)
    c0, r0, cf             = Float64(c0),   Float64(r0),   Float64(cf)

    dx = 2 * xmax / (nx - 1)
    dy = 2 * ymax / (ny - 1)
    dz = 2 * zmax / (nz - 1)

    dt = cf * min(dx, dy, dz) / c0 / sqrt(3)        # CFL condition
    nt = Int(ceil(tmax / dt))

    fact_m0 = dt * r0 * c0^2
    fact_m1 = dt / r0
    z0      = r0 * c0

    return Domain(tmax, xmax, ymax, zmax,
                  nx, ny, nz,
                  c0, r0, cf,
                  dx, dy, dz,
                  fact_m0, fact_m1,
                  dt, nt, z0)
end

# -----------------------------------------------------------------------------
# Grid coordinate helpers
# -----------------------------------------------------------------------------

"Time axis `0:dt:tmax` as a range of length `nt`."
t(d::Domain) = range(0; stop=d.tmax, length=d.nt)

"Cell-centred x coordinates."
x(d::Domain) = range(-d.xmax; stop=d.xmax, length=d.nx)
"Cell-centred y coordinates."
y(d::Domain) = range(-d.ymax; stop=d.ymax, length=d.ny)
"Cell-centred z coordinates."
z(d::Domain) = range(-d.zmax; stop=d.zmax, length=d.nz)

"Staggered-grid x coordinates (shifted by `-dx/2`)."
xv(d::Domain) = x(d) .- d.dx / 2
"Staggered-grid y coordinates."
yv(d::Domain) = y(d) .- d.dy / 2
"Staggered-grid z coordinates."
zv(d::Domain) = z(d) .- d.dz / 2

"""
    coords(d::Domain)

Per-field grid-coordinate lookup: a nested `Dict` keyed by axis (`:x`, `:y`, `:z`)
and field (`:p`, `:vx`, `:vy`, `:vz`). Pressure lives on the cell-centred grid;
each velocity component is staggered along its own axis.
"""
function coords(d::Domain)
    return Dict(
        :x => Dict(:p => x(d),  :vx => xv(d), :vy => x(d),  :vz => x(d)),
        :y => Dict(:p => y(d),  :vx => y(d),  :vy => yv(d), :vz => y(d)),
        :z => Dict(:p => z(d),  :vx => z(d),  :vy => z(d),  :vz => zv(d)),
    )
end

# -----------------------------------------------------------------------------
# Transceivers (sources / receivers at arbitrary grid positions)
# -----------------------------------------------------------------------------

"""
    InterpolationIndices

Per-channel grid-index vectors for the four staggered field components
(`p`, `vx`, `vy`, `vz`). Each entry is a 3-element `[ix, iy, iz]` for the
lower corner of the trilinear stencil.
"""
struct InterpolationIndices
    p::Vector{Int}
    vx::Vector{Int}
    vy::Vector{Int}
    vz::Vector{Int}
end

"""
    InterpolationWeights

Per-channel `2×2×2` trilinear stencil weights, one entry per staggered field
component (`p`, `vx`, `vy`, `vz`).
"""
struct InterpolationWeights
    p::Array{Float64,3}
    vx::Array{Float64,3}
    vy::Array{Float64,3}
    vz::Array{Float64,3}
end

"""
    InterpolationPoint

A single spatial point plus the trilinear-interpolation indices and weights
needed to read/write the staggered grid fields at that point.
"""
struct InterpolationPoint
    point::Vector{Float64}
    indices::InterpolationIndices
    weights::InterpolationWeights
end

"""
    Transceiver(pInt, tf)
    Transceiver(d::Domain; point, direction, tf)

A source/receiver with its interpolation point, time-function samples, and
direction vector `[q, fx, fy, fz]`. The second constructor computes the
trilinear interpolation weights from a spatial `point` in the `Domain`.
"""
struct Transceiver
    pInt::InterpolationPoint
    tf::Vector{Float64}
    direction::Vector{Float64}

    Transceiver(pInt::InterpolationPoint, tf::Vector{Float64}) =
        new(pInt, tf, [1.0, 0.0, 0.0, 0.0])

    function Transceiver(
        d::Domain;
        point,
        direction,
        tf::Vector{Float64},
    )
        point = vec(copy(point))
        direction = vec(copy(direction))
        pInt = trilinearInterpolation(d, point)
        new(pInt, tf, direction)
    end
end

# -----------------------------------------------------------------------------
# CPML absorbing boundary — memory variables + reset
# -----------------------------------------------------------------------------

"""
    Cpml(d::Domain; npml, rcoef=1e-4, fc=3e3, ndim=3)

Memory variables for the Convolutional Perfectly Matched Layer, plus the
precomputed coefficient arrays (`CPMLCoefficients`). One memory tensor per
field (`:p`, `:v`) per spatial direction (x, y, z).
"""
struct Cpml
    n::Int
    p::Vector{Data.Array}
    v::Vector{Data.Array}
    c::CPMLCoefficients

    function Cpml(
        d::Domain;
        npml::Int,
        rcoef::Float64=1e-4,
        fc::Float64=3e3,
        ndim::Int=3,
    )
        mem_p = Data.Array[]
        mem_v = Data.Array[]
        for i in 1:ndim
            m_ns_v = [d.nx, d.ny, d.nz, 2]; m_ns_v[i] = npml
            m_ns_p = copy(m_ns_v);          m_ns_p[i] = npml + 1
            push!(mem_p, @zeros(m_ns_p...))
            push!(mem_v, @zeros(m_ns_v...))
        end

        c = CPMLCoefficients(npml)
        compute_CPML_coefficients!(c, d.c0, d.dt, npml, rcoef, d.dx * npml, fc)

        new(npml, mem_p, mem_v, c)
    end
end

"""
    resetCPMLmemory!(cpml::Cpml)

Zero the CPML memory variables (used between successive FDTD runs that share
the same `Cpml` allocation).
"""
function resetCPMLmemory!(cpml::Cpml)
    for i in 1:size(cpml.p, 1)
        cpml.p[i] .= 0
        cpml.v[i] .= 0
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Trilinear / bilinear interpolation for off-grid sources & receivers
# -----------------------------------------------------------------------------

"""
    trilinearInterpolation(d::Domain, point)

Return an `InterpolationPoint` holding the 2×2×2 stencil of grid indices and
weights for each staggered field component (`p`, `vx`, `vy`, `vz`), centred
on the 3D position `point`.
"""
function trilinearInterpolation(d::Domain, point::Vector{Float64})
    dz = d.dz
    xp, yp, zp = point
    cs = coords(d)

    # Bilinear x-y stencils for (p, vx, vy). vz's x-y stencil is degenerate
    # with p (vz isn't staggered in x or y), so it reuses p's data below.
    bi_idx, bi_w = bilinearInterpolation(d, point)

    # Lift one channel's 2D (indices, weights) into 3D by adding the z-stencil.
    function _add_z(idx_xy::Vector{Int}, w_xy::Matrix{Float64},
                    zk::AbstractVector)
        iz = findlast(zk .< zp)
        w3 = zeros(2, 2, 2)
        if iz < d.nz
            z1 = zk[iz]
            z2 = zk[iz + 1]
            w3[:, :, 1] = w_xy .* (z2 - zp)
            w3[:, :, 2] = w_xy .* (zp - z1)
            w3 ./= dz
        end
        return vcat(idx_xy, iz), w3
    end

    idx_p,  w_p  = _add_z(bi_idx.p,  bi_w.p,  cs[:z][:p])
    idx_vx, w_vx = _add_z(bi_idx.vx, bi_w.vx, cs[:z][:vx])
    idx_vy, w_vy = _add_z(bi_idx.vy, bi_w.vy, cs[:z][:vy])
    idx_vz, w_vz = _add_z(bi_idx.p,  bi_w.p,  cs[:z][:vz])  # vz uses p's x-y

    return InterpolationPoint(point,
        InterpolationIndices(idx_p, idx_vx, idx_vy, idx_vz),
        InterpolationWeights(w_p,  w_vx,  w_vy,  w_vz),
    )
end

# Per-channel 2D stencil indices/weights produced by `bilinearInterpolation`.
# Private to simulation.jl; consumed only by `trilinearInterpolation`.
struct BilinearIndices
    p::Vector{Int}
    vx::Vector{Int}
    vy::Vector{Int}
end

struct BilinearWeights
    p::Matrix{Float64}
    vx::Matrix{Float64}
    vy::Matrix{Float64}
end

"""
    bilinearInterpolation(d::Domain, point) -> (BilinearIndices, BilinearWeights)

2×2 x-y stencil indices and weights for the three staggered field components
`p`, `vx`, `vy` at `point`. The `vz` channel's bilinear stencil is degenerate
with `p` in x-y, so it's omitted here and grafted on by the caller.
"""
function bilinearInterpolation(d::Domain, point::Vector{Float64})
    dx = d.dx
    dy = d.dy
    ds = dx * dy
    xp, yp, _ = point
    cs = coords(d)

    function _one_channel(xk::AbstractVector, yk::AbstractVector)
        ix = findlast(xk .< xp)
        iy = findlast(yk .< yp)
        w = zeros(2, 2)
        if ix < d.nx && iy < d.ny
            x1 = xk[ix]; x2 = xk[ix + 1]
            y1 = yk[iy]; y2 = yk[iy + 1]
            dx0 = xp - x1; dx1 = x2 - xp
            dy0 = yp - y1; dy1 = y2 - yp
            w[1, 1] = dx1 * dy1
            w[2, 1] = dx0 * dy1
            w[1, 2] = dx1 * dy0
            w[2, 2] = dx0 * dy0
            w ./= ds
        end
        return vec([ix iy]), w
    end

    idx_p,  w_p  = _one_channel(cs[:x][:p],  cs[:y][:p])
    idx_vx, w_vx = _one_channel(cs[:x][:vx], cs[:y][:vx])
    idx_vy, w_vy = _one_channel(cs[:x][:vy], cs[:y][:vy])

    return BilinearIndices(idx_p, idx_vx, idx_vy),
           BilinearWeights(w_p, w_vx, w_vy)
end

# -----------------------------------------------------------------------------
# Collect a list of transceivers into the packed GPU tensors the kernels expect
# -----------------------------------------------------------------------------

"""
    ChannelBuffers

Packed per-field buffers consumed by the FDTD kernels: trilinear stencil
indices `i`, weights `w`, source-time function `src`, and receiver buffer
`rec`. One `ChannelBuffers` per staggered field (`p`, `vx`, `vy`, `vz`).
"""
struct ChannelBuffers
    i::Data.Array
    w::Data.Array
    src::Data.Array
    rec::Data.Array
end

"""
    TransceiverGrid

Device-resident packed form of a list of `Transceiver`s, ready for the FDTD
kernels. One `ChannelBuffers` per field channel, plus the aggregated
`direction` matrix and the flat normal-velocity src/rec buffers. Use the
accessors [`p_src`](@ref), [`p_rec`](@ref), [`vn_src`](@ref), [`vn_rec`](@ref)
for the hot write/read targets.
"""
struct TransceiverGrid
    p::ChannelBuffers
    vx::ChannelBuffers
    vy::ChannelBuffers
    vz::ChannelBuffers
    direction::Data.Array
    vn_src::Data.Array
    vn_rec::Data.Array
end

"""
    transceiversToGrid(d::Domain, txs::Vector{Transceiver}) -> TransceiverGrid

Pack a list of `Transceiver`s into the GPU-resident arrays used by the FDTD
kernels: per-field `ChannelBuffers(i, w, src, rec)` plus the aggregated
`direction` matrix and the flat normal-velocity (`vn_src`, `vn_rec`) buffers.
"""
function transceiversToGrid(d::Domain, txs::Vector{Transceiver})
    channels = Dict{Symbol,ChannelBuffers}()

    for (i, k) in enumerate((:p, :vx, :vy, :vz))
        # Indices
        idx = Array{Int64}(undef, 3, 0)
        for tx in txs
            idx = cat(idx, getfield(tx.pInt.indices, k); dims=2)
        end

        # Weights
        wts = Array{Float64}(undef, 2, 2, 2, 0)
        for tx in txs
            wts = cat(wts, getfield(tx.pInt.weights, k); dims=4)
        end

        # Time functions
        srcbuf = Array{Float64}(undef, d.nt, 0)
        for tx in txs
            s = tx.tf * tx.direction[i]  # scale by direction component
            srcbuf = cat(srcbuf, s; dims=2)
        end

        channels[k] = ChannelBuffers(
            Data.Array(idx),
            Data.Array(wts),
            Data.Array(srcbuf),
            Data.Array(srcbuf * 0),
        )
    end

    # Normal velocity: concatenated directions + receiver/source buffers.
    dirs = Array{Int64}(undef, 4, 0)
    for tx in txs
        dirs = cat(dirs, tx.direction; dims=2)
    end
    nTx = size(dirs, 2)

    return TransceiverGrid(
        channels[:p], channels[:vx], channels[:vy], channels[:vz],
        Data.Array(dirs),
        @zeros(d.nt, nTx),  # vn_src
        @zeros(d.nt, nTx),  # vn_rec
    )
end

"""
    p_src(txs_on_grid)   p_rec(txs_on_grid)
    vn_src(txs_on_grid)  vn_rec(txs_on_grid)

Typed accessors for the four channel buffers that consumers of
`transceiversToGrid` write to (`*_src`) or read from (`*_rec`):

  • `p_src`  — pressure source-time function (q-channel,  monopole)
  • `p_rec`  — pressure receiver buffer
  • `vn_src` — normal-velocity source-time function (f-channel, dipole)
  • `vn_rec` — normal-velocity receiver buffer

Each is a single inlined struct-field fetch — typo-safe, autocompleted,
and devirtualized by the compiler. Pair with `inject_inner_sources!`
helpers in the test scripts to keep injection sites two lines.
"""
@inline p_src(t::TransceiverGrid)  = t.p.src
@inline p_rec(t::TransceiverGrid)  = t.p.rec
@inline vn_src(t::TransceiverGrid) = t.vn_src
@inline vn_rec(t::TransceiverGrid) = t.vn_rec

# -----------------------------------------------------------------------------
# Initial wavefield (plane wave or focusing)
# -----------------------------------------------------------------------------

"""
    initialField(d::Domain, fc::Float64, distance::Float64) -> Array{Float32,4}

Build the initial pressure-velocity state `(nx, ny, nz, 4)` for a Ricker-
wavelet plane wave travelling in +x, offset by `distance`, with centre
frequency `fc`.

(A `:focusing` branch for a spherical converging wave is retained for the
supplementary figures; flip the `waveform` symbol below to activate it.)
"""
function initialField(d::Domain, fc::Float64, distance::Float64)
    c = coords(d)

    k = fc / d.c0
    wn(r) = pi * k .* (r .- distance)
    ricker(wn) = (1 .- 2 .* wn .^ 2) .* exp.(-(wn .^ 2))

    field = zeros(Float32, d.nx, d.ny, d.nz, 4)

    waveform = :planeWave  # :planeWave or :focusing

    for key in (:p, :vx, :vy, :vz)
        cx = reshape(c[:x][key], :, 1, 1)
        cy = reshape(c[:y][key], 1, :, 1)
        cz = reshape(c[:z][key], 1, 1, :)

        if waveform == :focusing
            r = sqrt.(cx .^ 2 .+ cy .^ 2 .+ cz .^ 2)

            if key in (:vx, :vy)
                phi = sign.(cy) .* acos.(cx ./ (sqrt.(cx .^ 2 .+ cy .^ 2)))
            end
            if key in (:vx, :vy, :vz)
                theta = acos.(cz ./ r)
            end

            p = ricker.(wn(r))
            p[r .< 0.1] .= 0

            if key == :p
                field[:, :, :, 1] = p
            elseif key == :vx
                field[:, :, :, 2] = -p .* sin.(theta) .* cos.(phi) ./ d.z0
            elseif key == :vy
                field[:, :, :, 3] = -p .* sin.(theta) .* sin.(phi) ./ d.z0
            elseif key == :vz
                field[:, :, :, 4] = -p .* cos.(theta) ./ d.z0
            end

        elseif waveform == :planeWave
            if key == :p
                field[:, :, :, 1] .= ricker.(pi * k .* (cx .+ distance))
            elseif key == :vx
                field[:, :, :, 2] .= ricker.(pi * k .* (cx .+ distance .- d.c0 * d.dt / 2)) ./ d.z0
            end
            # :vy and :vz are identically zero for a plane wave along +x.
        end
    end

    return field
end

# -----------------------------------------------------------------------------
# Time-stepping driver — single source of truth for the FDTD loop.
# -----------------------------------------------------------------------------

"""
    run_fdtd!(dom, field, txs, cpml, va;
              start_step::Int = 1,
              on_step = nothing) -> field

Advance `field` from `start_step` to `dom.nt` using the pressure/velocity
update kernels in `src/kernels/{updates,extrapolation,forward}.jl`.
Source/receiver traces and CPML memory are updated in place.

`on_step(it, field)` (if non-`nothing`) is called after each FDTD step. Use
it for snapshot capture, live diagnostics, or progress reporting — the
hot-path callers in `greens.jl` / `hologram.jl` pass `nothing`, while
`reverb.jl` uses it to grab pressure-field snapshots at selected steps.

`start_step != 1` is used by `reverb.jl` to resume an iSrc from a saved
final wavefield (the source-time function is zero past `start_step` for
that path, so re-firing the source is unnecessary).
"""
function run_fdtd!(dom::Domain, field, txs, cpml::Cpml, va;
                   start_step::Int = 1,
                   on_step       = nothing,
                   gf_map        = nothing)
    if on_step === nothing
        for it in start_step:dom.nt
            forward_onestep!(dom, field, txs, cpml, va, it; gf_map = gf_map)
        end
    else
        for it in start_step:dom.nt
            forward_onestep!(dom, field, txs, cpml, va, it; gf_map = gf_map)
            on_step(it, field)
        end
    end
    return field
end
