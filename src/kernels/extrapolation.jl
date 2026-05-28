# -----------------------------------------------------------------------------
# Green's-function extrapolation kernels.
#
# Three flavours, dispatched by the concrete subtype of `GFMap` returned by
# `load_gf` (see docs/extrapolation_conventions.md):
#
#   PVFromPV   → two-way Kirchhoff representation Σ[G^{p|q}·v + G^{p|f}·p]
#                over the 4-component reference GFs (p_p, p_v, v_p, v_v).
#                Not the production path; reachable via `extrap_type` for
#                the one-way/two-way equivalence demo.
#   PVFromPin  → one-way incoming-pressure form: extrapolates
#                p_in = (p − z₀vₙ)/2 with the 2-component pin GFs
#                (p_pin, v_pin). The production path — half the
#                convolution work of PVFromPV. Used by load_gf_ref's
#                default and load_gf_mdd.
#   PFromPin   → split into one-way (pos / neg) components for boundary
#                driving with explicit one-way wavefields.
#
# All three replace dense `sum(... .* ...; dims=...)` broadcasts with explicit
# nested loops to avoid temporaries and let the time-marching dimension stay
# in the inner loop where the compiler can vectorise it. Threaded over the
# output (source) axis.
# -----------------------------------------------------------------------------

"""
    abstract type GFMap end

Tagged container of GF tensors + index vectors driving the extrapolation
inside `forward_onestep!`. Three concrete subtypes — `PVFromPV`, `PVFromPin`,
`PFromPin` — each carrying the arrays the corresponding kernel consumes.

`iRec` / `iSrc` are populated post-construction at the call site
(`build_hologram` in src/hologram.jl, or test code) by direct field assignment;
loaders initialise them to empty `Int[]`.
"""
abstract type GFMap end

"""
    PVFromPV(pp, pv, vp, vv)

Two-way Kirchhoff form, 4-component reference GFs. Each array is
`(nt, nsrc, nrec)`, eltype `<:AbstractFloat` (Float32 from the on-disk
reference path).
"""
mutable struct PVFromPV{A<:AbstractArray{<:AbstractFloat,3}} <: GFMap
    pp::A
    pv::A
    vp::A
    vv::A
    iRec::Vector{Int}
    iSrc::Vector{Int}
end
PVFromPV(pp::A, pv::A, vp::A, vv::A) where {A<:AbstractArray{<:AbstractFloat,3}} =
    PVFromPV{A}(pp, pv, vp, vv, Int[], Int[])

"""
    PVFromPin(p_pin, v_pin)

One-way incoming-pressure form, 2-component pin GFs. Production path
(load_gf_ref default + load_gf_mdd). Each array is `(nt, nsrc, nrec)`,
eltype `<:AbstractFloat` (Float32 from MDD; Float64 from load_gf_ref's
pin extraction, which widens via `dom.z0::Float64`).
"""
mutable struct PVFromPin{A<:AbstractArray{<:AbstractFloat,3}} <: GFMap
    p_pin::A
    v_pin::A
    iRec::Vector{Int}
    iSrc::Vector{Int}
end
PVFromPin(p_pin::A, v_pin::A) where {A<:AbstractArray{<:AbstractFloat,3}} =
    PVFromPin{A}(p_pin, v_pin, Int[], Int[])

"""
    PFromPin(pos, neg)

Split into one-way (pos / neg) components for boundary driving with explicit
one-way wavefields. Each array is `(nt, nrec, nsrc)`, eltype `<:AbstractFloat`
(Float64 from load_gf_ref's arithmetic involving `dom.z0::Float64`).
"""
mutable struct PFromPin{A<:AbstractArray{<:AbstractFloat,3}} <: GFMap
    pos::A
    neg::A
    iRec::Vector{Int}
    iSrc::Vector{Int}
end
PFromPin(pos::A, neg::A) where {A<:AbstractArray{<:AbstractFloat,3}} =
    PFromPin{A}(pos, neg, Int[], Int[])

"""
    gf_arrays(g::GFMap) -> Tuple

Live references to the GF tensors held by `g` (4-tuple for `PVFromPV`,
2-tuple for `PVFromPin` / `PFromPin`). Returned by reference — in-place
mutation propagates. Used by generic tests that scale or count GF memory
without branching on type.
"""
gf_arrays(g::PVFromPV)  = (g.pp, g.pv, g.vp, g.vv)
gf_arrays(g::PVFromPin) = (g.p_pin, g.v_pin)
gf_arrays(g::PFromPin)  = (g.pos, g.neg)

function extrapolate_pv_from_pv(
    p_rec,
    v_rec,
    pp,
    vp,
    pv,
    vv,
    it,
)
    # Original broadcast `pv(T,S,R) .* p(T,R,1)` requires S == R, and
    # treats the shared middle dim as the contraction axis. We preserve
    # that semantics with an explicit loop, no temp arrays.
    nt_gf, nm, nrec = size(pp)
    @assert size(p_rec, 2) == nm "extrapolate_pv_from_pv: dim mismatch — p_rec has $(size(p_rec,2)) columns, expected $nm"
    T = min(it, nt_gf)
    t_offset = it - T + 1                # field index for tt=1
    F = eltype(pp)
    pn_out = zeros(F, nrec)
    v_out  = zeros(F, nrec)

    Threads.@threads for r in 1:nrec
        psum = zero(F)
        vsum = zero(F)
        @inbounds for m in 1:nm
            for tt in 1:T
                gf_idx = T - tt + 1
                p_t  = p_rec[t_offset + tt - 1, m]
                vn_t = v_rec[t_offset + tt - 1, m]
                psum += pv[gf_idx, m, r] * p_t  + pp[gf_idx, m, r] * vn_t
                vsum += vv[gf_idx, m, r] * p_t  + vp[gf_idx, m, r] * vn_t
            end
        end
        @inbounds pn_out[r] = psum
        @inbounds v_out[r]  = vsum
    end
    return pn_out, v_out
end

function extrapolate_pv_from_pin!(
    pin,
    p_src,
    v_src,
    pp,
    vp,
    it,
)
    # Replaces `sum(pp[I,:,:] .* pin[I,:]; dims=(1,3))` with an explicit
    # triple loop. No temp arrays, threaded over the output (source) axis.
    nt_gf, nsrc, nrec = size(pp)
    T = min(it, nt_gf)
    t_offset = it - T + 1                # field index for tt=1
    F = eltype(p_src)

    Threads.@threads for s in 1:nsrc
        psum = zero(F)
        vsum = zero(F)
        @inbounds for r in 1:nrec
            for tt in 1:T
                gf_idx = T - tt + 1
                pin_t  = pin[t_offset + tt - 1, r]
                psum += pp[gf_idx, s, r] * pin_t
                vsum += vp[gf_idx, s, r] * pin_t
            end
        end
        # Original convention: p_src ← (vp-derived), v_src ← (pp-derived).
        @inbounds p_src[it + 1, s] = vsum
        @inbounds v_src[it + 1, s] = psum
    end
    return nothing
end

function extrapolate_p_from_pin(pin, g, it)
    # Replaces `sum(g[I,:,:] .* pin[I,:]; dims=(1,2))` with an explicit
    # triple loop. Output is shape (nsrc,), threaded over s.
    nt_gf, nrec, nsrc = size(g)
    T = min(it, nt_gf)
    t_offset = it - T + 1
    F = eltype(g)
    out = zeros(F, nsrc)

    Threads.@threads for s in 1:nsrc
        sum_s = zero(F)
        @inbounds for r in 1:nrec
            for tt in 1:T
                gf_idx = T - tt + 1
                sum_s += g[gf_idx, r, s] * pin[t_offset + tt - 1, r]
            end
        end
        @inbounds out[s] = sum_s
    end
    return out
end

# -----------------------------------------------------------------------------
# extrapolate! — dispatch entry consumed by forward_onestep!
#
# Shared signature: extrapolate!(g::<Concrete>, txs::TransceiverGrid, z0, it).
# `z0` is unused by PVFromPV (the two-way form is already in (p, vₙ) coords)
# but kept in the signature so all three methods share a call shape.
# -----------------------------------------------------------------------------

function extrapolate!(g::PVFromPV, txs::TransceiverGrid, z0::Real, it::Integer)
    iSrc = g.iSrc
    iRec = g.iRec

    pn_extrap, v_extrap = extrapolate_pv_from_pv(
        txs.p.rec[:, iRec],
        txs.vn_rec[:, iRec],
        g.pp,
        g.vp,
        g.pv,
        g.vv,
        it,
    )
    txs.p.src[:, iSrc][it+1, :] = v_extrap
    txs.vn_src[:, iSrc][it+1, :] = pn_extrap
    return nothing
end

function extrapolate!(g::PVFromPin, txs::TransceiverGrid, z0::Real, it::Integer)
    iSrc = g.iSrc
    iRec = g.iRec

    pin = (txs.p.rec[:, iRec] - txs.vn_rec[:, iRec] * z0) / 2

    extrapolate_pv_from_pin!(
        pin,
        txs.p.src[:, iSrc],
        txs.vn_src[:, iSrc],
        g.p_pin,
        g.v_pin,
        it,
    )
    return nothing
end

function extrapolate!(g::PFromPin, txs::TransceiverGrid, z0::Real, it::Integer)
    iSrc = g.iSrc
    iRec = g.iRec

    pin = (txs.p.rec[:, iRec] - txs.vn_rec[:, iRec] * z0) / 2

    pos = extrapolate_p_from_pin(pin, g.pos, it)
    neg = extrapolate_p_from_pin(pin, g.neg, it)

    txs.p.src[:, iSrc][it+1, :] = pos .- neg
    txs.vn_src[:, iSrc][it+1, :] = (pos .+ neg) / z0
    return nothing
end
