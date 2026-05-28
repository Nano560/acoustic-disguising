using Test
using AcousticDisguising

# -----------------------------------------------------------------------------
# Green's-function reciprocity: G(r_a, r_b, t) ≈ G(r_b, r_a, t) for an acoustic
# wave equation in a homogeneous medium. Two FDTD runs:
#
#   Run 1: pressure monopole at A, record pressure at B  →  trace_AB
#   Run 2: pressure monopole at B, record pressure at A  →  trace_BA
#
# These should agree within numerical tolerance (small differences come from
# the trilinear-interpolation weights at A vs B and from finite-grid
# anisotropy of the staggered FDTD stencil).
#
# Tiny 51³ grid, ~250 time steps, runs in well under a minute on CPU.
# -----------------------------------------------------------------------------

@testset "reciprocity (homogeneous medium)" begin
    # Tiny domain so the test is CI-friendly.
    dom = Domain(;
        tmax = 4.0e-4,
        xmax = 0.3,  ymax = 0.3,  zmax = 0.3,
        nx   = 51,   ny   = 51,   nz   = 51,
        cf   = 0.5,
    )

    # Two off-axis points, neither on a grid node, so the trilinear
    # interpolation weights are nontrivial in both directions.
    A = [ 0.10, -0.05,  0.02]
    B = [-0.04,  0.07,  0.06]

    # Same compact-Gaussian wavelet used by `impulsive_gfs`. Bandwidth
    # mirrors the paper config's [greens].f_3db_hz default.
    srctf, _  = impulsive_wavelet(dom; f_3db_hz = 7000.0)
    rectf     = zeros(Float64, dom.nt)

    # Pressure source: direction [1, 0, 0, 0]   (q=1, no v components).
    function record_at(src_pt::Vector{Float64}, rec_pt::Vector{Float64})
        src = Transceiver(dom; point = src_pt, direction = [1.0, 0.0, 0.0, 0.0], tf = srctf)
        rec = Transceiver(dom; point = rec_pt, direction = [1.0, 0.0, 0.0, 0.0], tf = rectf)
        txs = [src, rec]
        txs_on_grid = transceiversToGrid(dom, txs)

        cpml = Cpml(dom; npml = 0, rcoef = 1e-4, fc = 3e3)
        # `va = nothing` selects the unmasked (homogeneous) FDTD fast path.
        # Plain Float32 array, not ParallelStencil's `@zeros`: that macro
        # requires `@init_parallel_stencil` in the calling module (here
        # `Main`), and this CPU-only test runs on the Threads backend where
        # `@zeros` is just a Float32 `Array` anyway.
        field = zeros(Float32, dom.nx, dom.ny, dom.nz, 4)
        run_fdtd!(dom, field, txs_on_grid, cpml, nothing)

        # First channel of `p_rec` is the source's self-record; second is
        # the receiver. Drop the self-record.
        Array(p_rec(txs_on_grid)[:, 2])
    end

    trace_AB = record_at(A, B)        # source at A, receive at B
    trace_BA = record_at(B, A)        # source at B, receive at A

    # Both traces should be the same up to grid anisotropy. Tolerate 1% RMS
    # mismatch — staircase anisotropy + trilinear-weight asymmetry.
    rms = sqrt(sum(abs2, trace_AB .- trace_BA) / length(trace_AB))
    peak = maximum(abs, trace_AB)
    @test rms / peak < 0.01

    # Sanity: traces should not be identically zero.
    @test peak > 0
end
