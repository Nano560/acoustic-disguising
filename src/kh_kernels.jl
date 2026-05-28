# =============================================================================
# Kirchhoff–Helmholtz analytical kernels + direct-form K-H convolutions.
#
# Pure math + threading; no Domain / config / global dependencies. Used by
# diagnostics that need an analytical-truth K-H boundary extrapolation.
#
# Sign convention: kernels here are kept WITHOUT the De Hoop dipole minus
# (`p_v ≡ +∂_n G`, `v_v ≡ +∂_t ∂_n G / ρ`). The minus is applied at the K-H
# summation site via the explicit `scale_pv_f = -1 · dS · dt` factor in the
# caller, so the classical integrand structure stays visible at the sum site
# rather than being absorbed silently into the kernel definitions. (This is
# distinct from scripts/greens/analytical.jl, which DOES bake the dipole
# minus into its kernel — the two modules answer different questions.)
# =============================================================================

# ---- Math helpers: Gaussian wavelet, its derivative, and its integral. -------
# Internal helpers — used only by the kernel evaluators in this file.
# Custom _erf instead of SpecialFunctions to keep this module dependency-free.
@inline function _erf(x::Float64)
    s, ax = sign(x), abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
    return s * y
end
@inline _fg(τ, σ)  = (1 / (sqrt(2π) * σ)) * exp(-τ^2 / (2 * σ^2))
@inline _dfg(τ, σ) = -(τ / σ^2) * _fg(τ, σ)
@inline _Fg(τ, σ)  = 0.5 * (1 + _erf(τ / (σ * sqrt(2))))

# ---- Kernel evaluators ------------------------------------------------------
"""
    eval_analytical_kernels(src_pos, src_normals, rec_pos, rec_normals,
                            t_grid, σ_t, c0, ρ0; T = Float64)

Evaluate the four analytical K-H kernels at all (rec, src) point pairs on
`t_grid`. Output shapes `(nt, n_rec, n_src)`. Optional `T` keyword sets the
output element type (default Float64; Float32 halves memory at the cost of a
few digits of precision — useful at high N where the four kernels can be
GB-scale).
"""
function eval_analytical_kernels(src_pos, src_normals, rec_pos, rec_normals,
                                 t_grid, σ_t, c0, ρ0;
                                 T = Float64)
    n_src = size(src_pos, 1)
    n_rec = size(rec_pos, 1)
    nt    = length(t_grid)
    p_p   = zeros(T, nt, n_rec, n_src)
    p_v   = zeros(T, nt, n_rec, n_src)
    v_p   = zeros(T, nt, n_rec, n_src)
    v_v   = zeros(T, nt, n_rec, n_src)
    Threads.@threads for is in 1:n_src
        s_pt = @view src_pos[is, :]
        n_s  = @view src_normals[is, :]
        for ir in 1:n_rec
            r_pt = @view rec_pos[ir, :]
            n_r  = @view rec_normals[ir, :]
            dx, dy, dz = r_pt[1]-s_pt[1], r_pt[2]-s_pt[2], r_pt[3]-s_pt[3]
            r   = sqrt(dx^2 + dy^2 + dz^2)
            ihx, ihy, ihz = dx/r, dy/r, dz/r
            cos_θr = n_r[1]*ihx + n_r[2]*ihy + n_r[3]*ihz
            cos_θs = n_s[1]*ihx + n_s[2]*ihy + n_s[3]*ihz
            retard = r / c0
            inv_4πr  = 1 / (4π * r)
            inv_4πr2 = 1 / (4π * r^2)
            inv_4πr3 = 1 / (4π * r^3)
            @inbounds for it in 1:nt
                τ   = t_grid[it] - retard
                fτ  = _fg(τ,  σ_t)
                dfτ = _dfg(τ, σ_t)
                Fτ  = _Fg(τ,  σ_t)
                # No De Hoop dipole minus baked in; applied at the summation
                # site via `scale_pv_f = -1·dS·dt` in the caller.
                p_p[it, ir, is] = dfτ * inv_4πr
                v_p[it, ir, is] = (cos_θr / ρ0) *
                                  (fτ * inv_4πr2 + dfτ * inv_4πr / c0)
                p_v[it, ir, is] = cos_θs *
                                  (fτ * inv_4πr2 + dfτ * inv_4πr / c0)
                # v_v carries a 2·F(τ)/(4πr³) near-field term where F = ∫f.
                # For Gaussian f, F → 1 as τ → ∞ → v_v has a non-decaying DC
                # pedestal that breaks time-truncation of the kernel.
                v_v[it, ir, is] = (cos_θr * cos_θs / ρ0) *
                                  (2 * Fτ      * inv_4πr3 +
                                   2 * fτ      * inv_4πr2 / c0 +
                                       dfτ     * inv_4πr  / c0^2)
            end
        end
    end
    return p_p, p_v, v_p, v_v
end

"""
    eval_one_kernel(which, outer_pts, outer_nrm, inner_pts, inner_nrm,
                    t_grid, σ_t, c0, ρ0; T = Float32)

Evaluate ONE analytical kernel by name (`:p_p`, `:p_v`, `:v_p`, `:v_v`) in
the `(nt, n_outer, n_inner)` layout — the layout consumed by `direct_kh_*!`.

Compared to `eval_analytical_kernels`, this:
  • returns ONE kernel (caller can free between channels → peak memory = 1
    kernel rather than 4 + 4 prefactored + 2 pin-combined ≈ 10× growth),
  • writes directly into the K-H-consumer layout (no `permutedims`
    allocation on top of the eval result),
  • shares the per-pair scaffolding (r, cos_θ, retard, τ, fτ, dfτ, Fτ) with
    `eval_analytical_kernels`; only the inner write differs per kernel.
"""
function eval_one_kernel(which::Symbol,
                         outer_pts, outer_nrm, inner_pts, inner_nrm,
                         t_grid, σ_t, c0, ρ0;
                         T = Float32)
    which in (:p_p, :p_v, :v_p, :v_v) || error("which must be one of :p_p, :p_v, :v_p, :v_v")
    n_outer = size(outer_pts, 1)
    n_inner = size(inner_pts, 1)
    nt      = length(t_grid)
    K = zeros(T, nt, n_outer, n_inner)
    Threads.@threads for ir in 1:n_inner
        r_pt = @view inner_pts[ir, :]
        n_r  = @view inner_nrm[ir, :]
        for is in 1:n_outer
            s_pt = @view outer_pts[is, :]
            n_s  = @view outer_nrm[is, :]
            dx, dy, dz = r_pt[1]-s_pt[1], r_pt[2]-s_pt[2], r_pt[3]-s_pt[3]
            r   = sqrt(dx^2 + dy^2 + dz^2)
            ihx, ihy, ihz = dx/r, dy/r, dz/r
            cos_θr = n_r[1]*ihx + n_r[2]*ihy + n_r[3]*ihz
            cos_θs = n_s[1]*ihx + n_s[2]*ihy + n_s[3]*ihz
            retard = r / c0
            inv_4πr  = 1 / (4π * r)
            inv_4πr2 = 1 / (4π * r^2)
            inv_4πr3 = 1 / (4π * r^3)
            @inbounds for it in 1:nt
                τ   = t_grid[it] - retard
                if which === :p_p
                    dfτ = _dfg(τ, σ_t)
                    K[it, is, ir] = T(dfτ * inv_4πr)
                elseif which === :p_v
                    fτ  = _fg(τ,  σ_t)
                    dfτ = _dfg(τ, σ_t)
                    K[it, is, ir] = T(cos_θs * (fτ * inv_4πr2 + dfτ * inv_4πr / c0))
                elseif which === :v_p
                    fτ  = _fg(τ,  σ_t)
                    dfτ = _dfg(τ, σ_t)
                    K[it, is, ir] = T((cos_θr / ρ0) *
                                      (fτ * inv_4πr2 + dfτ * inv_4πr / c0))
                else  # :v_v
                    fτ  = _fg(τ,  σ_t)
                    dfτ = _dfg(τ, σ_t)
                    Fτ  = _Fg(τ,  σ_t)
                    K[it, is, ir] = T((cos_θr * cos_θs / ρ0) *
                                      (2 * Fτ      * inv_4πr3 +
                                       2 * fτ      * inv_4πr2 / c0 +
                                           dfτ     * inv_4πr  / c0^2))
                end
            end
        end
    end
    return K
end

"""
    eval_kernel_pedestal(which, outer_pts, outer_nrm, inner_pts, inner_nrm,
                         ρ0; T = Float32)

Closed-form τ → ∞ limit `K_dc[is, ir]` of the named kernel at every
(outer, inner) pair. Returns an `(n_outer, n_inner)` matrix.

For a Gaussian wavelet `f`, only `:v_v` has a non-vanishing pedestal
because it carries the integral `F(τ) → 1`. The other three kernels are
built from `f` and `f'` which both → 0, so their pedestal is exactly
zero. Returning the zeros matrix for those is still useful: it lets the
convolution caller use a single uniform code path
(`K_residual = K − K_dc` + cumsum-tail correction) without per-kernel
branching at the call site. `c0` does not appear — the limit only
involves the geometric `1/(4π r³)` and the angle cosines.

  :p_p, :p_v, :v_p  →  zeros (no pedestal)
  :v_v              →  (cos_θr · cos_θs / ρ0) · 2 / (4π r³)
"""
function eval_kernel_pedestal(which::Symbol,
                              outer_pts, outer_nrm, inner_pts, inner_nrm,
                              ρ0;
                              T = Float32)
    which in (:p_p, :p_v, :v_p, :v_v) || error("which must be one of :p_p, :p_v, :v_p, :v_v")
    n_outer = size(outer_pts, 1)
    n_inner = size(inner_pts, 1)
    K_dc    = zeros(T, n_outer, n_inner)
    which === :v_v || return K_dc
    Threads.@threads for ir in 1:n_inner
        r_pt = @view inner_pts[ir, :]
        n_r  = @view inner_nrm[ir, :]
        for is in 1:n_outer
            s_pt = @view outer_pts[is, :]
            n_s  = @view outer_nrm[is, :]
            dx, dy, dz = r_pt[1]-s_pt[1], r_pt[2]-s_pt[2], r_pt[3]-s_pt[3]
            r   = sqrt(dx^2 + dy^2 + dz^2)
            ihx, ihy, ihz = dx/r, dy/r, dz/r
            cos_θr = n_r[1]*ihx + n_r[2]*ihy + n_r[3]*ihz
            cos_θs = n_s[1]*ihx + n_s[2]*ihy + n_s[3]*ihz
            inv_4πr3 = 1 / (4π * r^3)
            K_dc[is, ir] = T((cos_θr * cos_θs / ρ0) * 2 * inv_4πr3)
        end
    end
    return K_dc
end

# ---- K-H convolution routines -----------------------------------------------
"""
    direct_kh_single!(out, K, in_arr)

Single-kernel K-H convolution. `out[t, s] = Σ_r Σ_tt K[tt, r, s] · in[t-tt+1, r]`.
Used by the `pv_from_pin` 1-way K-H formulation (one kernel × one input
rather than the 2-kernel two-way form). Generic in element type. Tolerates
K with fewer time samples than the output (truncation-friendly).
"""
function direct_kh_single!(out::AbstractMatrix,
                           K::AbstractArray{<:Real,3},
                           in_arr::AbstractMatrix)
    nt_K, n_outer, n_inner = size(K)
    nt = size(out, 1)
    @assert size(in_arr) == (nt, n_outer)              "input shape mismatch"
    @assert size(out, 2) == n_inner                    "output cols ≠ n_inner"
    @assert nt_K <= nt                                 "K longer than output"
    fill!(out, 0.0)
    Threads.@threads for s in 1:n_inner
        @inbounds for r in 1:n_outer
            for t in 1:nt
                acc = 0.0
                @simd for tt in 1:min(t, nt_K)
                    acc += K[tt, r, s] * in_arr[t-tt+1, r]
                end
                out[t, s] += acc
            end
        end
    end
    return out
end

"""
    direct_kh_accumulate!(out, K, in_arr; α = 1.0,
                          K_dc = nothing, cumsum_in = nothing)

Accumulating K-H convolution: `out[t, s] += α · Σ_r Σ_tt K[tt, r, s] · in[t-tt+1, r]`.
Like `direct_kh_single!` but does NOT zero `out` first — caller pre-allocates
the accumulator and calls this once per (kernel, signal) pair. The scalar
`α` folds the K-H prefactor (e.g. `-ρ·dS_out·dt` or `(1/c)·dS_out·dt`) into
the sum at integration time, so a single eval'd kernel can contribute to
both the `pv_from_pv` and `pv_from_pin` output channels with different
scales — no prefactored copies needed (keeps peak memory at ~1 kernel).

When `K_dc` and `cumsum_in` are both provided, `K` is treated as the
residual kernel `K_full − K_dc[r, s]` and the τ → ∞ pedestal contribution
is added analytically as `α · Σ_r K_dc[r, s] · cumsum_in[t, r]`, where
`cumsum_in[t, r] = Σ_{t'=1}^{t} in[t', r]`. Mathematically equivalent to
running the full (untruncated) kernel; lets the caller truncate `K` in
time without losing the late-time tail. See `eval_kernel_pedestal`.
"""
function direct_kh_accumulate!(out::AbstractMatrix,
                               K::AbstractArray{<:Real,3},
                               in_arr::AbstractMatrix;
                               α::Real = 1.0,
                               K_dc::Union{Nothing,AbstractMatrix}      = nothing,
                               cumsum_in::Union{Nothing,AbstractMatrix} = nothing)
    nt_K, n_outer, n_inner = size(K)
    nt = size(out, 1)
    @assert size(in_arr) == (nt, n_outer)              "input shape mismatch"
    @assert size(out, 2) == n_inner                    "output cols ≠ n_inner"
    @assert nt_K <= nt                                 "K longer than output"
    @assert (K_dc === nothing) == (cumsum_in === nothing) "K_dc and cumsum_in must be provided together"
    if K_dc !== nothing
        @assert size(K_dc)      == (n_outer, n_inner)  "K_dc shape mismatch"
        @assert size(cumsum_in) == (nt, n_outer)       "cumsum_in shape mismatch"
    end
    Threads.@threads for s in 1:n_inner
        @inbounds for r in 1:n_outer
            for t in 1:nt
                acc = 0.0
                @simd for tt in 1:min(t, nt_K)
                    acc += K[tt, r, s] * in_arr[t-tt+1, r]
                end
                out[t, s] += α * acc
            end
        end
    end
    if K_dc !== nothing
        Threads.@threads for s in 1:n_inner
            @inbounds for r in 1:n_outer
                k = K_dc[r, s]
                k == 0 && continue
                αk = α * k
                @simd for t in 1:nt
                    out[t, s] += αk * cumsum_in[t, r]
                end
            end
        end
    end
    return out
end

"""
    direct_kh!(out, K_a, K_b, in_a, in_b)

Two-kernel K-H convolution: `out[t, s] = Σ_r Σ_tt (K_a[tt, r, s] · in_a[t-tt+1, r]
                                                 + K_b[tt, r, s] · in_b[t-tt+1, r])`.
Generic in element type (kernels and signals can be Float32 or Float64).
Accepts kernels with FEWER time samples than the input/output — the
convolution sum stops once we've consumed all kernel samples (useful when
the kernel has finite support and is truncated for memory).
"""
function direct_kh!(out::AbstractMatrix,
                    K_a::AbstractArray{<:Real,3}, K_b::AbstractArray{<:Real,3},
                    in_a::AbstractMatrix, in_b::AbstractMatrix)
    nt_K, n_outer, n_inner = size(K_a)
    nt = size(out, 1)
    @assert size(K_b)  == (nt_K, n_outer, n_inner)        "K_b shape mismatch"
    @assert size(in_a) == (nt, n_outer)                   "in_a shape mismatch"
    @assert size(in_b) == (nt, n_outer)                   "in_b shape mismatch"
    @assert size(out, 2) == n_inner                       "out has wrong number of inner columns"
    @assert nt_K <= nt                                    "K longer in time than output — refusing"
    fill!(out, 0.0)
    Threads.@threads for s in 1:n_inner
        @inbounds for r in 1:n_outer
            for t in 1:nt
                acc = 0.0
                @simd for tt in 1:min(t, nt_K)
                    acc += K_a[tt, r, s] * in_a[t-tt+1, r] +
                           K_b[tt, r, s] * in_b[t-tt+1, r]
                end
                out[t, s] += acc
            end
        end
    end
    return out
end
