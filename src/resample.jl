# -----------------------------------------------------------------------------
# Time-axis resampling helper.
#
# Both stages 1 and 2, plus both Green's-function loaders (load_gf_ref,
# load_gf_mdd), interpolate a 3-D tensor `A[t, channel, channel2]` from a
# raw FDTD time grid onto a target grid, then rescale by `dt_target /
# dt_source` to preserve impulse-response energy density across the sample
# rate change. This file is the single source of truth for that operation.
# -----------------------------------------------------------------------------

using Interpolations

"""
    resample_time(A, t_in::AbstractRange, t_out::AbstractRange;
                  kind::Symbol = :cubic,
                  scale_amplitude::Bool = true) -> Array

Resample `A` along axis 1 (time) from `t_in` to `t_out`. The remaining
axes are evaluated at integer indices only.

  - `kind` ∈ (`:cubic`, `:linear`). `:cubic` uses `BSpline(Cubic(Line(OnGrid())))`
    and is appropriate for FDTD output traces (bandwidth-limited at dt_in).
    `:linear` is the original choice for MDD-extracted GFs.
  - `scale_amplitude = true` multiplies the output by `step(t_out) / step(t_in)`,
    matching the impulse-response convention used everywhere downstream.

`A` may be 1-D, 2-D, or 3-D. Time is always axis 1.

Returns an `Array` of the same `eltype` as `A` (after the optional rescale).
Use `Float32.(resample_time(...))` if you need to narrow the dtype.
"""
function resample_time(A::AbstractArray, t_in::AbstractRange, t_out::AbstractRange;
                       kind::Symbol = :cubic,
                       scale_amplitude::Bool = true)
    spline = kind === :cubic  ? BSpline(Cubic(Line(OnGrid()))) :
             kind === :linear ? BSpline(Linear()) :
             error("resample_time: kind must be :cubic or :linear, got :$kind")

    itp = interpolate(A, spline)
    itp = extrapolate(itp, Flat())

    # 1D evaluation has no parallelism axis; 2D/3D thread over the
    # non-time receiver/channel axes. Interpolations.jl evaluation is
    # thread-safe (read-only reads from the precomputed coefficients).
    if ndims(A) == 1
        itp = scale(itp, t_in)
        out = itp(t_out)
    elseif ndims(A) == 2
        itp = scale(itp, t_in, 1:size(A, 2))
        n_t = length(t_out)
        n_r = size(A, 2)
        T   = typeof(itp(first(t_out), 1))
        out = Array{T, 2}(undef, n_t, n_r)
        Threads.@threads for r in 1:n_r
            @inbounds for ti in 1:n_t
                out[ti, r] = itp(t_out[ti], r)
            end
        end
    elseif ndims(A) == 3
        itp = scale(itp, t_in, 1:size(A, 2), 1:size(A, 3))
        n_t = length(t_out)
        n_r = size(A, 2)
        n_c = size(A, 3)
        T   = typeof(itp(first(t_out), 1, 1))
        out = Array{T, 3}(undef, n_t, n_r, n_c)
        Threads.@threads for r in 1:n_r
            @inbounds for c in 1:n_c, ti in 1:n_t
                out[ti, r, c] = itp(t_out[ti], r, c)
            end
        end
    else
        error("resample_time: only 1D, 2D, 3D arrays supported (got $(ndims(A))D)")
    end

    return scale_amplitude ? out .* (step(t_out) / step(t_in)) : out
end
