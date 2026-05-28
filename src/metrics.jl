# =============================================================================
# Inner-disk spatial metrics: weighted RMS, weighted inner product, tapered
# circular mask + sub-array extraction, and zero-padding between two domains.
# Used by diagnostics that score K-H injection cancellation inside the inner
# sphere's y=0 cross-section.
# =============================================================================

"""
    rms_in_mask(slab, w)

Weighted RMS of a 2D slab over a weight matrix `w[i,k] ≥ 0`. Reduces to the
boolean-mask version when `w` is 0/1. Supports Float weights so the
inner-disk taper near r = r_inner is handled (see `inner_disk_weight`).
"""
function rms_in_mask(slab::AbstractMatrix, w::AbstractMatrix)
    s, n = 0.0, 0.0
    @inbounds for k in axes(slab, 2), i in axes(slab, 1)
        wi = Float64(w[i, k])
        if wi > 0
            v = Float64(slab[i, k]); s += wi * v * v; n += wi
        end
    end
    return n == 0 ? NaN : sqrt(s / n)
end

"""
    inner_dot(a, b, w)

Weighted inner product of two 2D slabs: `⟨a, b⟩_w = Σ w[i,k] · a[i,k] · b[i,k]`
— the natural weighted L² inner product. With a tapered weight, the closed
form `s_opt = -⟨A,B⟩_w / ⟨B,B⟩_w` still gives the minimum of
`Σ w · (A + s·B)²`, just with sources near r = r_inner weighted less
heavily (which damps the injection-source near-field's contribution).
"""
function inner_dot(a::AbstractMatrix, b::AbstractMatrix, w::AbstractMatrix)
    s = 0.0
    @inbounds for k in axes(a, 2), i in axes(a, 1)
        wi = Float64(w[i, k])
        if wi > 0
            s += wi * Float64(a[i, k]) * Float64(b[i, k])
        end
    end
    return s
end

"""
    inner_disk_subarray(slab, dom, r_inner)

Square crop of a y=0 slab to the cells with `|x|, |z| ≤ r_inner`. Returns a
`view`. The crop bounds match the indices used by `inner_disk_weight(dom,
r_inner)` so the two outputs can be element-wise multiplied.
"""
function inner_disk_subarray(slab::AbstractMatrix, dom, r_inner::Real)
    xs = collect(range(-dom.xmax, dom.xmax, length = dom.nx))
    zs = collect(range(-dom.zmax, dom.zmax, length = dom.nz))
    ix = findall(x -> abs(x) <= r_inner, xs)
    iz = findall(z -> abs(z) <= r_inner, zs)
    return view(slab, ix, iz)
end

"""
    inner_disk_weight(dom, r_inner; taper_frac = 0.1)

Tapered inner-disk weight matrix over the inner-disk sub-array (Float64 in
[0, 1]). Smoothly drops from 1 (interior) to 0 (boundary) over a band of
width `taper_frac · r_inner` near r = r_inner:

  • r ≤ r_inner · (1 − taper_frac)            →  w = 1            (full weight)
  • r ∈ [r_inner · (1 − taper_frac), r_inner] →  cosine taper (1 → 0)
  • r > r_inner                                →  w = 0           (excluded)

`taper_frac = 0.0` recovers the hard mask; default 0.1 is wide enough to
suppress the ~λ/6 near-field of injection sources sitting on r = r_inner.
"""
function inner_disk_weight(dom, r_inner::Real; taper_frac::Float64 = 0.1)
    xs = collect(range(-dom.xmax, dom.xmax, length = dom.nx))
    zs = collect(range(-dom.zmax, dom.zmax, length = dom.nz))
    ix = findall(x -> abs(x) <= r_inner, xs)
    iz = findall(z -> abs(z) <= r_inner, zs)
    xs_sub = xs[ix]; zs_sub = zs[iz]
    R       = Float64(r_inner)
    R_flat  = R * (1 - taper_frac)
    w       = zeros(Float64, length(xs_sub), length(zs_sub))
    @inbounds for k in eachindex(zs_sub), i in eachindex(xs_sub)
        r = sqrt(xs_sub[i]^2 + zs_sub[k]^2)
        if r <= R_flat
            w[i, k] = 1.0
        elseif r < R
            t = (r - R_flat) / max(R - R_flat, eps())
            w[i, k] = 0.5 * (1 + cos(π * t))
        end
    end
    return w
end

"""
    pad_inj_slab_to_pw(slab_inj, dom_inj, dom_pw)

Zero-pad a slab recorded on `dom_inj` (smaller injection domain) into the
larger `dom_pw` plane-wave domain shape. Used to render the cancellation-
residual panel `slab_A + slab_B` on `dom_pw`'s grid.
"""
function pad_inj_slab_to_pw(slab_inj::AbstractMatrix, dom_inj, dom_pw)
    padded = zeros(Float32, dom_pw.nx, dom_pw.nz)
    xs_pw = collect(range(-dom_pw.xmax, dom_pw.xmax, length = dom_pw.nx))
    zs_pw = collect(range(-dom_pw.zmax, dom_pw.zmax, length = dom_pw.nz))
    ix_pw = findall(x -> abs(x) <= dom_inj.xmax + 0.5 * dom_pw.dx, xs_pw)
    iz_pw = findall(z -> abs(z) <= dom_inj.zmax + 0.5 * dom_pw.dz, zs_pw)
    @assert length(ix_pw) == size(slab_inj, 1) "ix_pw length ($(length(ix_pw))) != slab_inj rows ($(size(slab_inj, 1)))"
    @assert length(iz_pw) == size(slab_inj, 2) "iz_pw length ($(length(iz_pw))) != slab_inj cols ($(size(slab_inj, 2)))"
    padded[ix_pw, iz_pw] .= Float32.(slab_inj)
    return padded
end

"""
    rms_region_diff(slab, ref, dom, pred)

RMS of `slab .- ref` over the cells satisfying `pred(x, z)`.
"""
function rms_region_diff(slab::AbstractMatrix, ref::AbstractMatrix, dom, pred)
    xs = collect(range(-dom.xmax, dom.xmax, length = dom.nx))
    zs = collect(range(-dom.zmax, dom.zmax, length = dom.nz))
    s, n = 0.0, 0
    @inbounds for k in eachindex(zs), i in eachindex(xs)
        if pred(xs[i], zs[k])
            d = Float64(slab[i, k]) - Float64(ref[i, k])
            s += d * d; n += 1
        end
    end
    return n == 0 ? NaN : sqrt(s / n)
end
