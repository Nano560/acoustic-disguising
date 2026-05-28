# -----------------------------------------------------------------------------
# Illumination: source/receiver placement on inner and outer surfaces, plus
# shared source-wavelet helpers (impulsive / Ricker).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Fibonacci-sphere point distribution
# -----------------------------------------------------------------------------

"""
    fibonacci_sphere(n::Int, radius::Real = 1.0) -> (points, normals)

Return `n` near-uniformly distributed points on a sphere of radius `radius`,
together with their outward unit normals. Uses the standard golden-angle
Fibonacci-spiral construction.

Returns `(points, normals)` where both are `n × 3` `Matrix{Float64}`.
"""
function fibonacci_sphere(n::Int, radius::Real = 1.0)
    @assert n > 0 "fibonacci_sphere requires n > 0"
    points = Matrix{Float64}(undef, n, 3)
    normals = Matrix{Float64}(undef, n, 3)

    golden = π * (3.0 - sqrt(5.0))  # golden-angle increment

    for i in 1:n
        # Evenly-spaced z in (-1, 1); the (2i-1)/n convention avoids the poles.
        z = 1.0 - (2.0 * (i - 1) + 1.0) / n
        r_xy = sqrt(max(0.0, 1.0 - z * z))
        θ = golden * (i - 1)

        nx = r_xy * cos(θ)
        ny = r_xy * sin(θ)
        nz = z

        normals[i, 1] = nx
        normals[i, 2] = ny
        normals[i, 3] = nz

        points[i, 1] = radius * nx
        points[i, 2] = radius * ny
        points[i, 3] = radius * nz
    end

    return points, normals
end

"""
    r2_sphere(n::Int, radius::Real = 1.0) -> (points, normals)

R₂ (Roberts' generalised golden-ratio) low-discrepancy sequence on a Lambert
equal-area sphere mapping. Plastic constant `ϕ = real positive root of
x³ = x + 1 ≈ 1.32472`; sample i uses `(u, v) = ((i/ϕ) mod 1, (i/ϕ²) mod 1)`,
mapped via `(φ = 2π u, z = 1 − 2v)` to a sphere point.

**Property**: position `i` depends only on `i`, not on `n`. Going from `n`
points to `n+k` points keeps the first `n` positions fixed and only adds new
ones — useful for incrementally refining sweeps without invalidating
previously-computed data (in contrast to `fibonacci_sphere`, where every
position depends on the total count).

Uniformity is moderate: stricter than Halton, looser than Fibonacci. For
`n = 300` the nearest-neighbour chord-distance std is ~6× larger than
Fibonacci's, but the angular coverage is still well below the K-H
quadrature residual at the resolutions used here.

Returns `(points, normals)` of shape `(n, 3)` matching `fibonacci_sphere`.
"""
function r2_sphere(n::Int, radius::Real = 1.0)
    @assert n > 0 "r2_sphere requires n > 0"
    # ϕ = positive real root of x³ = x + 1 (the "plastic constant").
    ϕ = (9 + sqrt(69))^(1/3) / cbrt(18) + (9 - sqrt(69))^(1/3) / cbrt(18)
    α1 = 1.0 / ϕ
    α2 = 1.0 / ϕ^2

    points  = Matrix{Float64}(undef, n, 3)
    normals = Matrix{Float64}(undef, n, 3)

    for i in 1:n
        u = mod(i * α1, 1.0)
        v = mod(i * α2, 1.0)
        # Lambert equal-area mapping (preserves uniform density).
        φ    = 2π * u
        z    = 1.0 - 2.0 * v
        r_xy = sqrt(max(0.0, 1.0 - z * z))

        nx = r_xy * cos(φ)
        ny = r_xy * sin(φ)
        nz = z
        normals[i, 1] = nx
        normals[i, 2] = ny
        normals[i, 3] = nz
        points[i, 1]  = radius * nx
        points[i, 2]  = radius * ny
        points[i, 3]  = radius * nz
    end

    return points, normals
end

# -----------------------------------------------------------------------------
# Illumination-point assembly (inner and outer Fibonacci spheres)
# -----------------------------------------------------------------------------

"""
    illumination_points(cfg) -> NamedTuple

Fibonacci-sphere source/receiver positions on the inner sphere (radius
`cfg["surfaces"]["radius_inner"]`) and outer sphere
(radius `cfg["surfaces"]["radius_outer"]`), using
`cfg["surfaces"]["nPoints_inner"]` and `["nPoints_outer"]` points respectively.

Returns `(; inner, outer, inner_normals, outer_normals)` where `inner` /
`outer` are `N × 3` matrices of xyz positions [m] and `*_normals` are the
matching outward unit normals. The normals are exposed so the callers in
`greens.jl` / `reverb.jl` can build `Transceiver` direction vectors for the
normal-velocity receivers.
"""
function illumination_points(cfg)
    s     = cfg["surfaces"]
    r_in  = Float64(s["radius_inner"])
    r_out = Float64(s["radius_outer"])
    n_in  = Int(s["nPoints_inner"])
    n_out = Int(s["nPoints_outer"])

    inner,  inner_normals  = fibonacci_sphere(n_in,  r_in)
    outer,  outer_normals  = fibonacci_sphere(n_out, r_out)

    return (; inner, outer, inner_normals, outer_normals)
end

# -----------------------------------------------------------------------------
# Source wavelets
# -----------------------------------------------------------------------------
#
# `gfs.jl` (impulsive GFs) used a compact Gaussian pulse (length 241 samples,
#   centre at 121, std 24.1), normalised so that its discrete integral is 1.
#   A `timeshift = pulse_center * dt` was then subtracted at interpolation
#   time so the recorded GFs start at t=0.
#
# `illumination.jl` (reverberant run) used a single Ricker wavelet at
#   fc = 18 kHz, time-offset by 1/fc (its centre), then cumulatively summed
#   and scaled by `dt / dx` — the original comment called that the "injection
#   scaling".
#
# Both helpers below return `Vector{Float64}` of length `dom.nt`, ready to
# be wrapped in a `Transceiver`.
# -----------------------------------------------------------------------------

"""
    impulsive_wavelet(dom::Domain; f_3db_hz::Real) -> (tf, timeshift)

Compact Gaussian source-time function used by `impulsive_gfs` to approximate
a band-limited Dirac. Bandwidth is parameterised by the spectrum's −3 dB
cutoff frequency `f_3db_hz`; the time-domain std is

    σ_t = √(ln 2) / (2π · f_3db_hz)

The pulse is windowed to ±5σ_t (10·σ_t total samples) so the truncated
Gaussian tails are below ~10⁻⁶ of the peak. Centred inside that window and
normalised so `sum(tf) * dom.dt == 1`.

Returns `(tf, timeshift)` where `tf::Vector{Float64}` has length `dom.nt`
and `timeshift` is the centre of the pulse in seconds (subtract from the
FDTD time axis when resampling to the GF time grid).
"""
function impulsive_wavelet(dom::Domain; f_3db_hz::Real)
    σ_t = sqrt(log(2)) / (2π * Float64(f_3db_hz))
    pulse_length = round(Int, 10 * σ_t / dom.dt)
    if pulse_length > dom.nt
        @warn "impulsive_wavelet: pulse_length > dom.nt; clamping (truncates Gaussian tails, slightly widens spectrum)" pulse_length dom_nt=dom.nt f_3db_hz
    end
    pl = min(pulse_length, dom.nt)
    pulse_center = pl / 2
    pulse_width  = pl / 10
    pulse = exp.(-((1:pl) .- pulse_center) .^ 2 ./ (2 * pulse_width^2))

    tf = zeros(Float64, dom.nt)
    tf[1:pl] .= pulse
    tf ./= (sum(tf) * dom.dt)

    timeshift = pulse_center * dom.dt
    return tf, timeshift
end

"""
    ricker_wavelet(dom::Domain; fc::Real = 18_000.0, injection::Bool = true) -> Vector{Float64}

Ricker wavelet at centre frequency `fc` Hz, evaluated on the Domain's time
axis (length `dom.nt`). Time-shifted by `1/fc` so the pulse is fully
contained within `t >= 0`.

If `injection = true`, apply the original "injection scaling" from
`illumination.jl`: cumulative sum times `dt / dx`. That scaling is required
when the wavelet is fed into the pressure source via the velocity-free
branch of `transceiversToGrid`. Pass `injection = false` if you need the
raw Ricker pulse itself (for analysis/plotting).
"""
function ricker_wavelet(dom::Domain; fc::Real = 18_000.0, injection::Bool = true)
    wn = π .* fc .* (collect(t(dom)) .- 1.0 / fc)
    tf = (1.0 .- 2.0 .* wn .^ 2) .* exp.(-(wn .^ 2))

    if injection
        tf = cumsum(tf) .* (dom.dt / dom.dx)
    end
    return collect(Float64, tf)
end
