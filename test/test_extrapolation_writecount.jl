#!/usr/bin/env julia
# =============================================================================
# test_extrapolation_writecount.jl  —  gather vs. scatter boundary extrapolation
#
# Substantiates the manuscript claim about the discrete Kirchhoff boundary
# extrapolation, a convolution mapping recorded data on the outer control
# surface X^O onto the inner emitting surface X^I:
#
#     u[x,t_n] = Σ_{x'∈X^O} Σ_{k=0}^{K-1} G[x,k,x'] · s[x', t_{n-k}]
#
# for each inner point x, timestep n; s the recorded boundary field (here the
# one-way constituent p_in), N points per surface, K kernel taps, n_t steps.
#
#   • GATHER (the production scheme) — for each output sample (x, t_n) the
#     (x',k) double sum is accumulated in a register and committed with ONE
#     memory write.  O(1) writes per extrapolated sample.
#   • SCATTER (forward extrapolation, van Manen et al. 2007) — each recorded
#     input sample (x', t_m) is pushed into its K future output slots
#     (x, t_{m+k}); a read-modify-write per (x,k).  O(K) writes per extrapolated
#     sample — O(N·K) if the per-source contributions are not pre-summed.
#
# Both schemes compute the identical convolution.  This file:
#   1. confirms the production routine is the gather scheme (behaviourally:
#      one write per output, zero reads of its own output, K-independent);
#   2. implements minimal reference gather + scatter (two flavours) and asserts
#      all of them — and the production routine — agree to round-off;
#   3. routes every array through a counting wrapper and asserts the measured
#      write/load counts equal a closed-form analytic prediction;
#   4. sweeps K, N, n_t and shows the write count is flat in K for gather and
#      linear in K for scatter;
#   5. reports that LOAD counts are comparable — the saving is writes
#      eliminated, not writes traded for reads;
#   6. (standalone only) wall-clocks both at a realistic size, N=400, K=300,
#      n_t=2000, gather with no atomics vs. scatter with atomic adds.
#
# Production routine — confirmed below to be the gather scheme:
#     src/kernels/extrapolation.jl:65    extrapolate_pv_from_pin!  (definition)
#     src/kernels/extrapolation.jl:80    Threads.@threads for s in 1:nsrc
#                                        — threaded over the OUTPUT (source)
#                                          axis: each output sample is owned by
#                                          exactly one thread ⇒ no write
#                                          collisions, no atomics needed.
#     src/kernels/extrapolation.jl:81-90 register accumulation (psum, vsum) of
#                                        the Σ_{x',k} double sum.
#     src/kernels/extrapolation.jl:92-93 p_src[it+1,s]=… ; v_src[it+1,s]=…
#                                        — exactly ONE write per output sample
#                                          per field component.
#   extrapolate_pv_from_pv and extrapolate_p_from_pin (same file) share this
#   structure; pv_from_pin is the production path.
#
# Run:
#   • as a unit test — included by test/runtests.jl. Fast: correctness +
#     write/load counting + the K/N/n_t sweep. The wall-clock benchmark is
#     skipped (it only runs when this file is executed directly).
#   • standalone —
#       julia --project=. --threads=auto test/test_extrapolation_writecount.jl
#     additionally runs the wall-clock benchmark at N=400, K=300, n_t=2000.
# =============================================================================

using Test
using AcousticDisguising
using Printf
using Random

# The production extrapolation routine under test (not exported).
const extrapolate_pv_from_pin! = AcousticDisguising.extrapolate_pv_from_pin!

# -----------------------------------------------------------------------------
# Counting wrapper
# -----------------------------------------------------------------------------
# An AbstractArray that tallies every element load (getindex) and store
# (setindex!). Counters are atomic so the tally is exact even when the wrapped
# array is written by the production routine's `Threads.@threads` loop.

struct CountingArray{T,N} <: AbstractArray{T,N}
    data   :: Array{T,N}
    reads  :: Threads.Atomic{Int}
    writes :: Threads.Atomic{Int}
end
CountingArray(a::Array{T,N}) where {T,N} =
    CountingArray{T,N}(a, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

Base.size(a::CountingArray) = size(a.data)
Base.IndexStyle(::Type{<:CountingArray}) = IndexCartesian()

@inline function Base.getindex(a::CountingArray{T,N}, I::Vararg{Int,N}) where {T,N}
    Threads.atomic_add!(a.reads, 1)
    a.data[I...]
end
@inline function Base.setindex!(a::CountingArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    Threads.atomic_add!(a.writes, 1)
    a.data[I...] = v
    v
end

nreads(a::CountingArray)  = a.reads[]
nwrites(a::CountingArray) = a.writes[]

# -----------------------------------------------------------------------------
# Reference convolutions — gather and scatter, both computing
#     u[n,x] = Σ_{x'} Σ_{k=0}^{K-1} G[k+1,x,x'] · s[n-1-k, x']
# with s[m,·] ≡ 0 for m ∉ [1,n_t]  (causal, zero-padded).
#
# Layout matches the production routine (extrapolation.jl):
#     G    :: (K, N_out, N_in)    kernel taps × inner pts × outer pts   (= pp)
#     s_in :: (n_t, N_in)         recorded boundary field               (= pin)
#     u    :: (n_t, N_out)        extrapolated inner-surface field       (= v_src)
# Output time n is `it+1` in the production routine; with s zero-padded the
# partial-overlap start-up (T = min(it, nt_gf)) is handled automatically.
#
# Generic over the array type so the same code runs on plain Arrays (for the
# correctness checks) and on CountingArrays (for the write/load tally). The
# register accumulates in Float64 so a gather/scatter mismatch can only be a
# summation-order difference, not a precision difference.
# -----------------------------------------------------------------------------

"""
    gather_extrapolate!(u, G, s_in) -> u

GATHER reference: for each output sample (n,x) accumulate the Σ_{x',k} double
sum in a register, then commit with exactly ONE write to `u[n,x]`. `u` is never
read. Mirrors `extrapolate_pv_from_pin!`.
"""
function gather_extrapolate!(u, G, s_in)
    K, N_out, N_in = size(G)
    n_t = size(s_in, 1)
    for x in 1:N_out, n in 1:n_t
        acc = 0.0
        for xp in 1:N_in
            for k in 0:K-1
                m = n - 1 - k
                if 1 <= m <= n_t
                    acc += Float64(G[k+1, x, xp]) * Float64(s_in[m, xp])
                end
            end
        end
        u[n, x] = acc                       # the single write per output
    end
    return u
end

"""
    scatter_extrapolate_naive!(u, G, s_in) -> u

SCATTER reference, not pre-summed — the literal "forward-extrapolate every
recorded sample" of van Manen et al. (2007). For each input sample (m,x') and
each (x,k) it does a read-modify-write into the future slot u[m+1+k, x].
O(N·K) writes per output sample; O(n_t·N²·K) total.
"""
function scatter_extrapolate_naive!(u, G, s_in)
    K, N_out, N_in = size(G)
    n_t = size(s_in, 1)
    fill!(u, 0.0)                           # accumulator must be zero-initialised
    for m in 1:n_t, xp in 1:N_in, x in 1:N_out
        for k in 0:K-1
            n = m + 1 + k
            if n <= n_t
                u[n, x] += Float64(G[k+1, x, xp]) * Float64(s_in[m, xp])
            end
        end
    end
    return u
end

"""
    scatter_extrapolate_presummed!(u, P, G, s_in) -> u

SCATTER reference, pre-summed over sources. For each input time m it first
gathers the per-source contributions into an (K, N_out) panel `P`, then
scatters the panel into the future output slots. O(K) writes per output
sample; O(n_t·N·K) total — plus an equal number of panel writes (the panel
build is itself a gather; pre-summing relocates the per-source work, it does
not remove it).
"""
function scatter_extrapolate_presummed!(u, P, G, s_in)
    K, N_out, N_in = size(G)
    n_t = size(s_in, 1)
    fill!(u, 0.0)
    for m in 1:n_t
        keff = min(K, n_t - m)              # taps from input m landing in [1,n_t]
        # phase A — gather sources into the panel: P[k+1,x] = Σ_x' G·s_in
        for x in 1:N_out, k in 0:keff-1
            acc = 0.0
            for xp in 1:N_in
                acc += Float64(G[k+1, x, xp]) * Float64(s_in[m, xp])
            end
            P[k+1, x] = acc
        end
        # phase B — scatter the panel forward into u (read-modify-write)
        for x in 1:N_out, k in 0:keff-1
            u[m+1+k, x] += P[k+1, x]
        end
    end
    return u
end

# -----------------------------------------------------------------------------
# Analytic write/load counts
# -----------------------------------------------------------------------------

"""
    tap_pairs(K, n_t) -> S

S = Σ_{n=1}^{n_t} min(K, n-1), the number of (output-time, effective-tap)
pairs. Every count below is N-scaled multiple of S.
"""
tap_pairs(K::Integer, n_t::Integer) =
    K <= n_t ? (K * (K - 1)) ÷ 2 + (n_t - K) * K : (n_t * (n_t - 1)) ÷ 2

"""
    analytic_counts(scheme, N_in, N_out, K, n_t) -> NamedTuple

Closed-form memory-access counts (asserted equal to the instrumented counts).
`u` = output/accumulator, `P` = pre-sum panel.
"""
function analytic_counts(scheme::Symbol, N_in, N_out, K, n_t)
    S = tap_pairs(K, n_t)
    if scheme === :gather
        # one write per output; never reads its own output.
        return (u_writes = n_t * N_out,
                u_reads  = 0,
                G_reads  = N_out * N_in * S,
                s_reads  = N_out * N_in * S,
                P_writes = 0,
                P_reads  = 0)
    elseif scheme === :scatter_naive
        # n_t·N_out zero-init writes + one read-modify-write per (m,x',x,k).
        return (u_writes = n_t * N_out + N_in * N_out * S,
                u_reads  = N_in * N_out * S,
                G_reads  = N_in * N_out * S,
                s_reads  = N_in * N_out * S,
                P_writes = 0,
                P_reads  = 0)
    elseif scheme === :scatter_presummed
        # n_t·N_out zero-init writes + one read-modify-write per (m,x,k).
        return (u_writes = n_t * N_out + N_out * S,
                u_reads  = N_out * S,
                G_reads  = N_out * N_in * S,
                s_reads  = N_out * N_in * S,
                P_writes = N_out * S,
                P_reads  = N_out * S)
    else
        error("unknown scheme $scheme")
    end
end

"""
    measure_counts(scheme, G, s_in) -> NamedTuple

Run `scheme` with every array wrapped in a `CountingArray` and return the
instrumented access counts (same shape as `analytic_counts`).
"""
function measure_counts(scheme::Symbol, G::Array, s_in::Array)
    K, N_out, _ = size(G)
    n_t = size(s_in, 1)
    Gc = CountingArray(G)
    Sc = CountingArray(s_in)
    uc = CountingArray(zeros(Float64, n_t, N_out))
    Pw, Pr = 0, 0
    if scheme === :gather
        gather_extrapolate!(uc, Gc, Sc)
    elseif scheme === :scatter_naive
        scatter_extrapolate_naive!(uc, Gc, Sc)
    elseif scheme === :scatter_presummed
        Pc = CountingArray(zeros(Float64, K, N_out))
        scatter_extrapolate_presummed!(uc, Pc, Gc, Sc)
        Pw, Pr = nwrites(Pc), nreads(Pc)
    else
        error("unknown scheme $scheme")
    end
    return (u_writes = nwrites(uc), u_reads = nreads(uc),
            G_reads = nreads(Gc), s_reads = nreads(Sc),
            P_writes = Pw, P_reads = Pr)
end

total_loads(c) = c.u_reads + c.G_reads + c.s_reads + c.P_reads

"Random kernel G (K,N_out,N_in) and recorded field s_in (n_t,N_in)."
function random_problem(N_in, N_out, K, n_t; T = Float64, seed = 0xC0FFEE)
    rng = MersenneTwister(seed)
    return randn(rng, T, K, N_out, N_in), randn(rng, T, n_t, N_in)
end

# -----------------------------------------------------------------------------
# Wall-clock benchmark kernel — threaded scatter with atomic adds
# -----------------------------------------------------------------------------

"""
    bench_scatter_atomic!(acc, G, s_in)

Forward-extrapolation scatter, threaded over the input-time axis. Because
several input times feed the same future output slot, the accumulator is a
matrix of `Threads.Atomic` and every store is an atomic add — the CPU analogue
of the GPU write-collision problem. Computes the same convolution as the
gather, with kernel `G`.
"""
function bench_scatter_atomic!(acc::Matrix{Threads.Atomic{Float32}}, G, s_in)
    K, N_out, N_in = size(G)
    n_t = size(s_in, 1)
    Threads.@threads for m in 1:n_t
        keff = min(K, n_t - m)
        @inbounds for xp in 1:N_in
            sv = s_in[m, xp]
            for x in 1:N_out
                for k in 0:keff-1
                    Threads.atomic_add!(acc[m+1+k, x], G[k+1, x, xp] * sv)
                end
            end
        end
    end
    return acc
end

# =============================================================================
# Test suite
# =============================================================================

@testset "extrapolation: gather vs scatter write count" begin

    # ----- 1. the production routine IS the gather scheme --------------------
    # Behavioural proof: wrap the production routine's two output arrays in
    # CountingArrays and step it across a whole run. Gather ⇔ exactly one write
    # per output sample, zero reads of the output, and a write count that does
    # not depend on K.
    @testset "production routine is the gather scheme" begin
        function production_output_io(K)
            n_t, N_in, N_out = 20, 6, 5
            rng = MersenneTwister(3)
            pin = randn(rng, Float32, n_t, N_in)
            pp  = randn(rng, Float32, K, N_out, N_in)
            vp  = randn(rng, Float32, K, N_out, N_in)
            p_src = CountingArray(zeros(Float32, n_t, N_out))
            v_src = CountingArray(zeros(Float32, n_t, N_out))
            for it in 1:n_t-1
                extrapolate_pv_from_pin!(pin, p_src, v_src, pp, vp, it)
            end
            return (writes = nwrites(p_src) + nwrites(v_src),
                    reads  = nreads(p_src)  + nreads(v_src),
                    n_out  = 2 * (n_t - 1) * N_out)   # 2 field components
        end
        a = production_output_io(11)
        b = production_output_io(23)        # K more than doubled
        @test a.writes == a.n_out           # exactly one write per output sample
        @test b.writes == b.n_out
        @test a.writes == b.writes          # write count independent of K ⇒ O(1)
        @test a.reads  == 0                 # never reads its own output (gather)
        @test b.reads  == 0
        @printf("    extrapolate_pv_from_pin!  output writes = %d (K=11) = %d (K=23); reads = %d\n",
                a.writes, b.writes, a.reads)
        @printf("    ⇒ one write per output sample, K-independent — confirmed gather.\n")
    end

    # ----- 2/3. correctness anchor — every scheme computes the same thing ----
    @testset "all schemes agree to round-off" begin
        G, s = random_problem(9, 7, 13, 40)
        K, N_out, _ = size(G)
        n_t = size(s, 1)
        u_g  = zeros(Float64, n_t, N_out)
        u_sn = zeros(Float64, n_t, N_out)
        u_sp = zeros(Float64, n_t, N_out)
        P    = zeros(Float64, K, N_out)
        gather_extrapolate!(u_g, G, s)
        scatter_extrapolate_naive!(u_sn, G, s)
        scatter_extrapolate_presummed!(u_sp, P, G, s)
        relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(a)), eps())
        @test maximum(abs.(u_g)) > 0                 # non-trivial output
        @test relerr(u_g, u_sn) < 1e-9               # gather ≡ scatter (naive)
        @test relerr(u_g, u_sp) < 1e-9               # gather ≡ scatter (presummed)
        @printf("    gather vs scatter(naive)      max rel diff = %.3g\n", relerr(u_g, u_sn))
        @printf("    gather vs scatter(presummed)  max rel diff = %.3g\n", relerr(u_g, u_sp))

        # And the reference gather reproduces the production routine. The
        # production routine writes v_src from kernel pp, p_src from vp
        # (see forward.jl:138-139 / extrapolation.jl:91-93).
        n_t2, N_in2, N_out2, K2 = 28, 8, 6, 11
        rng = MersenneTwister(7)
        pin = randn(rng, Float32, n_t2, N_in2)
        pp  = randn(rng, Float32, K2, N_out2, N_in2)
        vp  = randn(rng, Float32, K2, N_out2, N_in2)
        p_src = zeros(Float32, n_t2, N_out2)
        v_src = zeros(Float32, n_t2, N_out2)
        for it in 1:n_t2-1
            extrapolate_pv_from_pin!(pin, p_src, v_src, pp, vp, it)
        end
        u_pp = zeros(Float64, n_t2, N_out2); gather_extrapolate!(u_pp, pp, pin)
        u_vp = zeros(Float64, n_t2, N_out2); gather_extrapolate!(u_vp, vp, pin)
        rel_v = maximum(abs.(v_src[2:end, :] .- u_pp[2:end, :])) / maximum(abs.(u_pp[2:end, :]))
        rel_p = maximum(abs.(p_src[2:end, :] .- u_vp[2:end, :])) / maximum(abs.(u_vp[2:end, :]))
        @test rel_v < 1e-3                           # Float32 production vs Float64 ref
        @test rel_p < 1e-3
        @printf("    production extrapolate_pv_from_pin! vs reference gather: max rel diff = %.3g\n",
                max(rel_v, rel_p))
    end

    # ----- 4. instrumented counts match the analytic formula -----------------
    @testset "instrumented counts match analytic formula" begin
        # asymmetric N_in ≠ N_out included, to exercise the formula fully.
        for (N_in, N_out, K, n_t) in [(5, 6, 9, 22), (7, 4, 5, 15), (3, 3, 12, 30)]
            G, s = random_problem(N_in, N_out, K, n_t)
            for scheme in (:gather, :scatter_naive, :scatter_presummed)
                @test measure_counts(scheme, G, s) ==
                      analytic_counts(scheme, N_in, N_out, K, n_t)
            end
        end
    end

    # ----- 5. sweep K, N, n_t ------------------------------------------------
    @testset "sweep: gather flat in K, scatter linear in K" begin
        # (group, N, K, n_t) with N_in = N_out = N.
        rows = [("K",   5,  8, 600), ("K",   5, 16,  600),
                ("K",   5, 32, 600), ("K",   5, 64,  600),
                ("n_t", 5, 16, 300), ("n_t", 5, 16, 1200),
                ("N",   3, 16, 600), ("N",   9, 16,  600)]

        println()
        println("    write counts  (u = output field; per-output = u_writes / (n_t·N))")
        println("    ───────────────────────────────────────────────────────────────────────────────")
        println("    grp    N    K   n_t │   gather  scat(presum)  scat(naive) │   per output: g / presum / naive")
        println("    ───────────────────────────────────────────────────────────────────────────────")

        gather_W = Dict{NTuple{3,Int},Int}()
        presum_W = Dict{NTuple{3,Int},Int}()

        for (grp, N, K, n_t) in rows
            G, s = random_problem(N, N, K, n_t)
            cg = measure_counts(:gather,             G, s)
            cp = measure_counts(:scatter_presummed,  G, s)
            cn = measure_counts(:scatter_naive,      G, s)
            # every measured count must equal the analytic prediction.
            @test cg == analytic_counts(:gather,            N, N, K, n_t)
            @test cp == analytic_counts(:scatter_presummed, N, N, K, n_t)
            @test cn == analytic_counts(:scatter_naive,     N, N, K, n_t)
            per = n_t * N
            @printf("    %-3s  %3d  %3d  %4d │ %8d  %11d  %11d │  %6.1f / %7.1f / %8.1f\n",
                    grp, N, K, n_t, cg.u_writes, cp.u_writes, cn.u_writes,
                    cg.u_writes / per, cp.u_writes / per, cn.u_writes / per)
            gather_W[(N, K, n_t)] = cg.u_writes
            presum_W[(N, K, n_t)] = cp.u_writes
        end
        println("    ───────────────────────────────────────────────────────────────────────────────")

        # gather: write count is exactly n_t·N — flat in K, independent of K.
        ks = (8, 16, 32, 64)
        @test all(gather_W[(5, K, 600)] == 600 * 5 for K in ks)
        # gather per-output write count is exactly 1 — the O(1) claim.
        @test all(gather_W[(5, K, 600)] == 5 * 600 for K in ks)

        # scatter: the writes in excess of the n_t·N baseline scale ~linearly
        # in K — each doubling of K multiplies them by ≈2 (exactly 2 in the
        # n_t ≫ K limit; slightly under here from the start-up taper).
        excess(K) = presum_W[(5, K, 600)] - 600 * 5
        for (K1, K2) in ((8, 16), (16, 32), (32, 64))
            r = excess(K2) / excess(K1)
            @test 1.6 <= r <= 2.05
        end
        # scatter per-output write count is ≈ K (the O(K) claim): with the
        # start-up taper it lands a little below K, never above.
        for K in ks
            per_out = presum_W[(5, K, 600)] / (600 * 5)
            @test 0.7 * K <= per_out <= K + 1
        end

        # gather scales as n_t·N (independent of K); a spot check.
        @test gather_W[(5, 16,  300)] == 300 * 5
        @test gather_W[(5, 16, 1200)] == 1200 * 5
        @test gather_W[(3, 16,  600)] == 600 * 3
        @test gather_W[(9, 16,  600)] == 600 * 9
    end

    # ----- 6. loads are comparable; only writes differ -----------------------
    @testset "loads comparable — the saving is writes, not loads" begin
        N_in = N_out = 8
        K, n_t = 50, 300
        G, s = random_problem(N_in, N_out, K, n_t)
        cg = measure_counts(:gather,            G, s)
        cp = measure_counts(:scatter_presummed, G, s)
        cn = measure_counts(:scatter_naive,     G, s)

        load_ratio_p  = total_loads(cp) / total_loads(cg)
        load_ratio_n  = total_loads(cn) / total_loads(cg)
        write_ratio_p = cp.u_writes / cg.u_writes
        write_ratio_n = cn.u_writes / cg.u_writes

        println()
        @printf("    size N=%d, K=%d, n_t=%d\n", N_in, K, n_t)
        @printf("    %-22s %12s %12s\n", "", "total loads", "output writes")
        @printf("    %-22s %12d %12d\n", "gather",            total_loads(cg), cg.u_writes)
        @printf("    %-22s %12d %12d\n", "scatter (presummed)", total_loads(cp), cp.u_writes)
        @printf("    %-22s %12d %12d\n", "scatter (naive)",    total_loads(cn), cn.u_writes)
        @printf("    scatter/gather  —  loads: %.2f× (presum), %.2f× (naive)   |   writes: %.1f× (presum), %.1f× (naive)\n",
                load_ratio_p, load_ratio_n, write_ratio_p, write_ratio_n)

        # loads are the same order of magnitude (within ~1+1/N_in for the
        # presummed scheme, ~1.5 for the naive scheme)…
        @test load_ratio_p < 1.5
        @test load_ratio_n < 2.0
        # …while the write counts differ by a factor that tracks K (≈K for the
        # presummed scheme, ≈N·K for the naive scheme).
        @test write_ratio_p > 0.6 * K
        @test write_ratio_n > 0.6 * N_in * K
    end

    # ----- headline ----------------------------------------------------------
    println()
    println("    ── result ──────────────────────────────────────────────────────────")
    println("    gather  : 1 write  per extrapolated sample  — O(1),  flat in K")
    println("    scatter : K writes per extrapolated sample  — O(K)  (O(N·K) un-pre-summed)")
    println("    loads are comparable; the production routine is the gather scheme.")
    println("    ─────────────────────────────────────────────────────────────────────")
end

# =============================================================================
# Wall-clock benchmark — runs only when this file is executed directly
# =============================================================================

function run_benchmark(; N = 400, K = 300, n_t = 2000)
    println()
    printstyled("══════ wall-clock benchmark — N=$N, K=$K, n_t=$n_t ══════\n"; bold = true)
    @printf("  Julia threads: %d   |   ACOUSTIC_DISGUISING_BACKEND: %s\n",
            Threads.nthreads(), AcousticDisguising.BACKEND)
    if Threads.nthreads() == 1
        @warn "single-threaded — rerun with `julia --threads=auto` for the parallel comparison"
    end
    if Sys.which("ncu") === nothing
        println("  NVIDIA Nsight Compute (ncu): not on this platform — wall-clock only,")
        println("  no hardware memory-write counters.")
    end

    rng = MersenneTwister(20240521)
    pp  = randn(rng, Float32, K, N, N)
    pin = randn(rng, Float32, n_t, N)

    # warm-up (JIT) on a tiny problem.
    let
        wp = randn(rng, Float32, 4, 3, 3)
        ws = randn(rng, Float32, 6, 3)
        ps = zeros(Float32, 6, 3); vs = zeros(Float32, 6, 3)
        for it in 1:5
            extrapolate_pv_from_pin!(ws, ps, vs, wp, wp, it)
        end
        bench_scatter_atomic!([Threads.Atomic{Float32}(0f0) for _ in 1:6, _ in 1:3], wp, ws)
        ug = zeros(Float64, 6, 3); gather_extrapolate!(ug, wp, ws)
        un = zeros(Float64, 6, 3); scatter_extrapolate_naive!(un, wp, ws)
    end

    # --- gather: the production routine, threaded over the output axis -------
    p_src = zeros(Float32, n_t, N)
    v_src = zeros(Float32, n_t, N)
    t_gather = @elapsed begin
        for it in 1:n_t-1
            extrapolate_pv_from_pin!(pin, p_src, v_src, pp, pp, it)
        end
    end

    # --- scatter: forward extrapolation, threaded, atomic adds ---------------
    acc = [Threads.Atomic{Float32}(0f0) for _ in 1:n_t, _ in 1:N]
    t_scatter = @elapsed bench_scatter_atomic!(acc, pp, pin)

    # cross-check: both schemes computed the same convolution (kernel pp).
    peak = maximum(abs, v_src)
    dmax = 0.0f0
    for n in 2:n_t, x in 1:N
        dmax = max(dmax, abs(acc[n, x][] - v_src[n, x]))
    end
    rel = dmax / peak

    # analytic memory-write counts at this size (per field component).
    S = tap_pairs(K, n_t)
    n_samples     = (n_t - 1) * N                  # extrapolated output samples
    gather_writes = n_samples                      # 1 write per sample
    presum_writes = n_t * N + N * S                # ≈ K writes per sample
    naive_writes  = N * N * S                      # ≈ N·K writes per sample (timed)
    mul_adds      = N * N * S                      # identical arithmetic both ways

    println()
    @printf("  arithmetic (both schemes)          : %.3g multiply-adds\n", Float64(mul_adds))
    println("  output / accumulator writes per field component:")
    @printf("    gather                           : %14d  (%.1f per sample)  — O(1)\n",
            gather_writes, gather_writes / n_samples)
    @printf("    scatter, pre-summed              : %14d  (%.0f per sample)  — O(K)\n",
            presum_writes, presum_writes / n_samples)
    @printf("    scatter, naive  (the timed one)  : %14.0f  (%.0f per sample)  — O(N·K)\n",
            Float64(naive_writes), naive_writes / n_samples)
    println()
    @printf("  gather  (production routine, threaded, no atomics)  : %.4g s\n", t_gather)
    @printf("  scatter (threaded, atomic adds — naive)             : %.4g s   [%.2f× gather]\n",
            t_scatter, t_scatter / t_gather)
    @printf("  cross-check  max|gather − scatter| / peak           : %.3g\n", rel)

    # --- single-core micro-comparison ---------------------------------------
    # Pure algorithm, no threading: gather (one store) vs naive scatter
    # (non-atomic read-modify-write). Same arithmetic; isolates the store cost.
    Nm, Km, ntm = 64, 300, 400
    Gm, sm = random_problem(Nm, Nm, Km, ntm)
    um = zeros(Float64, ntm, Nm)
    t_g1 = @elapsed gather_extrapolate!(um, Gm, sm)
    t_s1 = @elapsed scatter_extrapolate_naive!(um, Gm, sm)
    println()
    @printf("  single-core micro-comparison (N=%d, K=%d, n_t=%d):\n", Nm, Km, ntm)
    @printf("    gather  : %.4g s     scatter(naive) : %.4g s     [%.2f× gather]\n",
            t_g1, t_s1, t_s1 / t_g1)
    println()
    println("  Reading: the two schemes do the same arithmetic and stream the same")
    println("  loads, so on a single CPU core they are comparable. The gather win is")
    println("  the eliminated writes — and, when parallelised, the absence of the")
    println("  write collisions that force the scatter onto atomic adds.")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_benchmark()
end
