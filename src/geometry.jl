# -----------------------------------------------------------------------------
# Scatterer geometry + volume/area masks + figure-polygon overlays.
#
# The four scatterer values (`:none`, `:sphere`, `:cube`, `:cross`) and
# their dimensions used in the paper. The configs/paper.toml may override
# any of these by setting the matching key (`sphere_radius`, etc.).
# -----------------------------------------------------------------------------

"Default scatterer dimensions used by the paper. Overridable via the same-named keys in `configs/paper.toml`."
const SCATTERER_DEFAULTS = (
    sphere_radius = 0.15,
    cube_length   = 0.20,
    cross_length  = 0.30,
)

@inline function _scatterer_param(cfg, name::Symbol)
    s = get(cfg, "surfaces", Dict{String,Any}())
    key = String(name)
    return haskey(s, key) ? s[key] : getfield(SCATTERER_DEFAULTS, name)
end

# -----------------------------------------------------------------------------
# Scatterer overlay polygons (for the 2D slice plots in figures/)
# -----------------------------------------------------------------------------

"""
    getPolygons(dom::Domain, cfg, cpml::Cpml, scatterer::AbstractString)

Return a vector of `(name, xs, ys)` tuples describing the outlines of the
PML box, the inner and outer control surfaces, and the chosen scatterer
(`"none"`, `"sphere"`, `"cube"`, or `"cross"` — `"none"` produces no
scatterer polygon).

Used by `figures/` scripts that overlay geometry on 2D slices of the field.
"""
function getPolygons(dom::Domain, cfg, cpml::Cpml, scatterer::AbstractString)
    n = cfg["grid"]["n"]
    cn = cpml.n
    theta = range(0, 2π; length=100)
    c = cos.(theta)
    s = sin.(theta)

    polygons = [
        (
            "npml",
            ([cn, cn, n - cn, n - cn, cn] .- n / 2) .* dom.dx,
            ([cn, n - cn, n - cn, cn, cn] .- n / 2) .* dom.dy,
        ),
        ("outer", cfg["surfaces"]["radius_outer"] * c, cfg["surfaces"]["radius_outer"] * s),
        ("inner", cfg["surfaces"]["radius_inner"] * c, cfg["surfaces"]["radius_inner"] * s),
    ]

    if scatterer == "sphere"
        r = _scatterer_param(cfg, :sphere_radius)
        push!(polygons, ("scatterer", r * c, r * s))
    elseif scatterer == "cube"
        l = _scatterer_param(cfg, :cube_length)
        push!(polygons, (
            "scatterer",
            [1, 1, -1, -1, 1] * l / 2,
            [1, -1, -1, 1, 1] * l / 2,
        ))
    elseif scatterer == "cross"
        l = _scatterer_param(cfg, :cross_length)
        push!(polygons, (
            "scatterer",
            [3, 3, 1, 1, -1, -1, -3, -3, -1, -1, 1, 1] / 3 / 2 * l,
            [1, -1, -1, -3, -3, -1, -1, 1, 1, 3, 3, 1] / 3 / 2 * l,
        ))
    end

    return polygons
end

# -----------------------------------------------------------------------------
# Scatterer shape predicates
# -----------------------------------------------------------------------------

"""
    _outside_predicate(scatterer::Symbol, cfg) -> Function

Return a closure `(x, y, z) -> Bool` that is `true` when the point
`(x, y, z)` lies strictly OUTSIDE the scatterer
(`:none`, `:sphere`, `:cube`, or `:cross`).
"""
function _outside_predicate(scatterer::Symbol, cfg)
    if scatterer === :none
        return (x, y, z) -> true
    elseif scatterer === :sphere
        r = _scatterer_param(cfg, :sphere_radius)
        return (x, y, z) -> x^2 + y^2 + z^2 > r^2
    elseif scatterer === :cube
        # 3D axis-aligned cube of side `cube_length`, centred at origin.
        # The 2D `getPolygons` outline traces a square cross-section.
        r = _scatterer_param(cfg, :cube_length)
        return (x, y, z) -> abs(x) > r / 2 || abs(y) > r / 2 || abs(z) > r / 2
    elseif scatterer === :cross
        # 3D Greek cross (plus sign): seven unit cubes — one central + six
        # face-attached arms along ±x/±y/±z. Equivalent terms in the
        # literature: "plus", "plus3d", "Greek cross", or generically
        # "polycube" (for any face-glued cube assembly). Constructed here as
        # the central cube with its 12 corner-edge channels carved out, so
        # axis-aligned slices show a `+` outline (see polygon trace in
        # `getPolygons` above).
        l = _scatterer_param(cfg, :cross_length)
        s = l / 2
        sl = 2.0 * l / 3
        is_inside_cube = (x, y, z) -> abs(x) < s && abs(y) < s && abs(z) < s
        return function (x, y, z)
            # A point is INSIDE the cross if it lies in the central cube and
            # ALSO in one of the six axial-shift copies of that cube. The
            # predicate returns true when the point is OUTSIDE that union.
            if is_inside_cube(x, y, z)
                # shift xy
                if is_inside_cube(x + sl, y + sl, z) ||
                   is_inside_cube(x - sl, y - sl, z) ||
                   is_inside_cube(x + sl, y - sl, z) ||
                   is_inside_cube(x - sl, y + sl, z)
                    return true
                end
                # shift xz
                if is_inside_cube(x + sl, y, z + sl) ||
                   is_inside_cube(x - sl, y, z - sl) ||
                   is_inside_cube(x + sl, y, z - sl) ||
                   is_inside_cube(x - sl, y, z + sl)
                    return true
                end
                # shift yz
                if is_inside_cube(x, y + sl, z + sl) ||
                   is_inside_cube(x, y - sl, z - sl) ||
                   is_inside_cube(x, y + sl, z - sl) ||
                   is_inside_cube(x, y - sl, z + sl)
                    return true
                end
                return false
            else
                return true
            end
        end
    else
        error("Unknown scatterer: $scatterer. Expected one of :none, :sphere, :cube, :cross.")
    end
end

# -----------------------------------------------------------------------------
# Voxelised scatterer mask
# -----------------------------------------------------------------------------

"""
    build_scatterer(dom::Domain, scatterer::Symbol, cfg) -> BitArray{3}

Return a 3D boolean mask marking scatterer voxels inside the domain.
`scatterer` ∈ (`:none`, `:sphere`, `:cube`, `:cross`). `:none` returns an
all-false mask (no scatterer — homogeneous medium baseline).

The mask is evaluated at cell-centre coordinates on the pressure grid
(`coords(dom)[:x][:p]` etc.).
"""
function build_scatterer(dom::Domain, scatterer::Symbol, cfg)
    is_outside = _outside_predicate(scatterer, cfg)

    cs = coords(dom)
    xp = collect(cs[:x][:p])
    yp = collect(cs[:y][:p])
    zp = collect(cs[:z][:p])

    mask = falses(dom.nx, dom.ny, dom.nz)
    if scatterer === :none
        return mask
    end

    @inbounds for k in 1:dom.nz, j in 1:dom.ny, i in 1:dom.nx
        mask[i, j, k] = !is_outside(xp[i], yp[j], zp[k])
    end
    return mask
end

# -----------------------------------------------------------------------------
# Per-voxel volume and face-area weights
# -----------------------------------------------------------------------------

"""
    volume_areas(dom::Domain, mask::BitArray{3}) -> NamedTuple

Cell-resolution (binary) volume + face-area mask: every cell is fully fluid
or fully solid according to `mask[i,j,k]`, and every face is open only when
BOTH neighbouring cells are fluid. The FDTD path doesn't use the float
values — call `build_update_mask` for that. This function is kept for
sanity checks and future per-cell-volume work.

Returns `(; V, Ax, Ay, Az)`.
"""
function volume_areas(dom::Domain, mask::BitArray{3})
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    @assert size(mask) == (nx, ny, nz) "mask size $(size(mask)) does not match domain ($(nx),$(ny),$(nz))"

    dV = dom.dx * dom.dy * dom.dz
    dAx = dom.dy * dom.dz
    dAy = dom.dx * dom.dz
    dAz = dom.dx * dom.dy

    # Fluid volume: 0 inside scatterer, full cell volume outside.
    V = Array{Float64}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        V[i, j, k] = mask[i, j, k] ? 0.0 : dV
    end

    # Face areas are zero at the boundary between a fluid cell and a scatterer
    # cell (and at the two outermost index strips, which have no neighbour).
    Ax = zeros(Float64, nx, ny, nz)
    Ay = zeros(Float64, nx, ny, nz)
    Az = zeros(Float64, nx, ny, nz)

    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        if !mask[i-1, j, k] && !mask[i, j, k]
            Ax[i, j, k] = dAx
        end
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        if !mask[i, j-1, k] && !mask[i, j, k]
            Ay[i, j, k] = dAy
        end
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        if !mask[i, j, k-1] && !mask[i, j, k]
            Az[i, j, k] = dAz
        end
    end

    return (; V, Ax, Ay, Az)
end

# -----------------------------------------------------------------------------
# Update-mask builder for the FDTD kernels (replaces the old _volume_mask
# helper that built the full Float64 V/Ax/Ay/Az tensors and threw them away).
# -----------------------------------------------------------------------------

"""
    build_update_mask(dom::Domain, scatterer::Symbol, cfg) -> Union{Nothing, DeviceArray{Bool,4}}

Return the device-resident boolean mask consumed by `update_p_mask!` and
`update_v*_mask!`. Channel index `1` is pressure (cell-centred) and `2,3,4`
are vx, vy, vz (face-centred). A face channel is open iff BOTH adjacent
pressure cells are fluid; the two outermost index strips along each face's
axis are always closed (no neighbour).

Returns `nothing` when `scatterer === :none` so `forward_onestep!` can
short-circuit to the unmasked fast path.
"""
function build_update_mask(dom::Domain, scatterer::Symbol, cfg)
    scatterer === :none && return nothing

    s  = build_scatterer(dom, scatterer, cfg)
    nx, ny, nz = dom.nx, dom.ny, dom.nz
    u  = fill(false, nx, ny, nz, 4)   # dense Array{Bool}, not a BitArray

    # Pressure: open at every fluid cell.
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        u[i, j, k, 1] = !s[i, j, k]
    end
    # vx face (between i-1 and i): open iff both neighbours are fluid.
    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        u[i, j, k, 2] = !s[i-1, j, k] && !s[i, j, k]
    end
    # vy face.
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        u[i, j, k, 3] = !s[i, j-1, k] && !s[i, j, k]
    end
    # vz face.
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        u[i, j, k, 4] = !s[i, j, k-1] && !s[i, j, k]
    end

    # `DeviceArray` (Array on CPU, CuArray on GPU) keeps the mask `Bool`.
    # `Data.Array` would cast it to the Float32 numbertype set by
    # `@init_parallel_stencil` — 4× the storage and 4× the per-timestep
    # bandwidth for a 0/1 array. The mask kernels fold it in as
    # `coeff = f * mask`, and `Float * Bool` is exact.
    return DeviceArray(u)
end
