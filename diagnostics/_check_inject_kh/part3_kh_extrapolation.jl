# ============================================================================
# Part 3 — K-H benchmark
# ============================================================================
# Two layers:
#
#   Layer (a) — K-H extrapolation check.
#       Convolve recorded outer fields with analytical Green's functions to
#       predict the inner field; compare to recorded inner. Validates the
#       analytical kernels + the K-H formula in physical units.
#
#   Layer (b) — K-H + injection check.
#       Use the K-H-extrapolated inner field (instead of recorded) as the Run-B
#       source. Compute s_opt. Should match Part 2's s_opt within K-H
#       extrapolation error — confirms that "extrapolate-then-inject" and
#       "directly-inject-recorded" pipelines produce the same field.
#
# K-H formula (interior pressure, classical Helmholtz integral with
# ∂_n p = -ρ ∂_t v_n substituted, NO integration by parts in t):
#     p(x_in, t)  =  ∮ [ -ρ·∂_t G_0 ⊛ vn  -  ∂_n G_0 ⊛ p ] dS
# In our kernel convention (p_p = ∂_t(G_0⋆f); p_v has -cos_θ_s baked in, so
# p_v = -∂_n(G_0⋆f) for the outward source normal):
#     p_inner_KH (t,s) = Σ_outer dS_out · dt · Σ_tt [-ρ · p_p · vn_outer  +  p_v · p_outer]
#     vn_inner_KH(t,s) = Σ_outer dS_out · dt · Σ_tt [-ρ · v_p · vn_outer  +  v_v · p_outer]
# Note the `+ p_v·p` (not `-`): p_v in our convention is -∂_n G, so
# `-∂_n G · p = +p_v · p`. The `· dt` is the time-integral discretization.
#
# Memory: kernels are (nt × N × N × 4 bytes) with Float32. For N=500,
# nt=1024 → ~1 GB / kernel × 4 kernels ≈ 4 GB. Tractable. At N=1000 → 16 GB.
# `--kernel-mode={full|trunc|trunc+ped}` controls the kernel time axis;
# `full` is the safe default, `trunc+ped` is mathematically equivalent at
# ~2.4× lower memory (see the v_v truncation block below). --ns-part3
# caps Part 3 to a subset of NS.
# ============================================================================

println()
printstyled("══════ Part 3 — K-H benchmark ══════\n"; bold = true)

# --ns-part3: subset of NS to run Part 3 on (memory cap). Default: all of NS.
# Pass an explicit subset (`--ns-part3=300,500`) to cap Part 3 memory at high N.
const NS_PART3_STR = _parse_flag(ARGS, "--ns-part3=", "")
const NS_PART3 = if isempty(NS_PART3_STR)
    sort(NS)
else
    parse.(Int, split(NS_PART3_STR, ","))
end
@info "Part 3 N values" NS_PART3 NS

function run_part3_for_N(N::Int)
    if !(N in keys(RECORDED_PER_N))
        @warn "[Part 3 N=$N] Skipped — N not in --ns so no Part 1 recording cached"
        return nothing
    end
    rec = RECORDED_PER_N[N]
    sp  = SPHERES_PER_N[N]
    Cs  = C_for_N(N)

    # =============================================================================
    # ⚠  v_v kernel truncation — beware the F-pedestal  ⚠
    # =============================================================================
    # The three kernels p_p, p_v, v_p contain only the wavelet f and its
    # derivative f' (which both decay as |τ| → ∞ for a Gaussian f). So their
    # time support is bounded by max_arrival + a few σ_t and truncation is
    # essentially exact.
    #
    # The v_v kernel, however, has a near-field term
    #     v_v ⊃ (cos_θ_r·cos_θ_s / ρ) · 2 F(τ)/(4πr³)
    # where F = ∫f dτ. For a Gaussian f (our default), F is the Gaussian CDF
    # → 1 as τ → ∞, so v_v carries a NON-DECAYING "DC pedestal" at every pair:
    #     v_v(τ → ∞) = (cos_θ_r·cos_θ_s / ρ) · 2/(4πr³)
    # The pedestal is largest for close (outer, inner) pairs (small r → 1/r³)
    # and grows with N (more close pairs are sampled).
    #
    # Empirical impact (N=300, this script's geometry):
    #     `trunc` (nt_kernel ≈ 433): vn_err_median = 0.47  ← bleeds 17 %
    #     `full`  (nt_kernel = 1024): vn_err_median = 0.30  ← K-H floor
    #     pressure errors are UNCHANGED at 0.024 either way.
    #
    # Three kernel modes (`--kernel-mode=...`):
    #   full       — no truncation (default). Correct, ~16 GB at N=1000.
    #   trunc      — naive truncation. Lossy (17 % vn bleed). Kept for the
    #                comparison plots; do NOT use for production.
    #   trunc+ped  — pedestal-subtract: K_residual = K − K_dc[r,s] is
    #                truncated (its tail → 0 cleanly), and the missing
    #                pedestal contribution is added analytically at
    #                convolution time via
    #                    out[t, s] += α · Σ_r K_dc[r, s] · cumsum_in[t, r].
    #                K_dc is the closed-form τ → ∞ limit (zero for the
    #                three decaying kernels; closed form above for v_v).
    #                Mathematically equivalent to `full` up to float
    #                roundoff, at ~2.4× lower kernel RAM. This is the
    #                preferred mode for N ≥ 500. See
    #                src/kh_kernels.jl::eval_kernel_pedestal and
    #                direct_kh_accumulate!(K_dc=, cumsum_in=).
    #
    # `trunc+ped` is "future cleanup option #1" (AC-subtract) implemented.
    # Two further wavelet- / formulation-level cleanups still possible:
    #   • Use a wavelet whose F decays (Ricker f'', or f'); then ALL
    #     kernels truncate cleanly without special-casing. Requires
    #     re-deriving _fg/_dfg/_Fg consistently.
    #   • Reformulate via one-way K-H (`p_from_pin` form): incoming-
    #     pressure decomposition avoids the v_v kernel entirely.
    # =============================================================================
    # KERNEL_MODE / _KH_MODE_TAG / _MODE_STR live at module scope (hoisted
    # near the other CLI flags so Part 4 can read them too).
    if KERNEL_MODE === :full
        nt_kernel    = dom_pw.nt
    else  # :trunc or :trunc_ped
        max_r        = R_OUTER + R_INNER     # diametrically opposite outer-inner pair
        buffer_t     = 5 * σ_T               # ~99 % of Gaussian f wavelet
        t_kernel_max = max_r / dom_pw.c0 + buffer_t
        nt_kernel    = min(dom_pw.nt, ceil(Int, t_kernel_max / dom_pw.dt))
    end
    T_GRID_KERNEL  = collect(range(0.0, (nt_kernel - 1) * dom_pw.dt; length = nt_kernel))
    @info "[Part 3 N=$N] kernel time grid" kernel_mode=KERNEL_MODE_STR nt_kernel dom_pw_nt=dom_pw.nt mem_GB=round(4*nt_kernel*N*N*4/1e9, digits=2)

    # ---- One-kernel-at-a-time eval + accumulating K-H convolution.
    # Memory profile: peak ~1 raw kernel alive (Float32, nt × N × N) — at N=900
    # that's ~3.3 GB, vs the old "4 raw + 4 prefactored + 2 combined" peak of
    # ~10× the same kernel size. The trade is ~4× scaffolding redundancy in
    # kernel eval (r, cos_θ, retard, τ, ... recomputed per kernel), which is
    # cheap compared to the K-H convolution.
    #
    # Two-way K-H summation (pv_from_pv form). Starting from the classical
    # interior Helmholtz integral with outward surface normal,
    #
    #     p_inner = ∮[ G · ∂_n p  −  ∂_n G · p ] dS                    (i)
    #
    # (i) carries TWO independent sign sources that the discrete sum needs
    # to preserve verbatim — they are *not* the same minus appearing twice:
    #
    #   (a) The linearised Euler momentum equation gives
    #           ∂_n p = −ρ ∂_t v_n
    #       which substituted into the first term of (i) yields
    #           G · ∂_n p = −ρ · ∂_t G · v_n = −ρ · p_p · v_n              (q-channel)
    #       → the `-ρ` factor in `scale_pv_q` below. This minus is the
    #       pressure-gradient / particle-acceleration sign relation; it
    #       has nothing to do with the dipole-source convention.
    #
    #   (b) The second term of (i) is `−∂_n G · p` — a structural minus
    #       between G·∂_n p and ∂_n G·p in the K-H representation theorem.
    #       This minus IS the De Hoop dipole/body-force convention
    #       (f = −p·n̂, positive surface pressure ↔ inward-pointing body
    #       force). With `p_v ≡ +∂_n G` (textbook definition, no baked-in
    #       sign — see `eval_one_kernel`), this minus is the `-1` factor
    #       in `scale_pv_f` below.
    #
    # After substituting (a) and identifying `p_v ≡ +∂_n G`, (i) reads
    #
    #     p_inner  = ∮[ −ρ · p_p · v_n   −   p_v · p ] dS · dt
    #                   ╰── (a) Euler ──╯   ╰── (b) De Hoop ─╯
    #     vn_inner = ∮[ −ρ · v_p · v_n   −   v_v · p ] dS · dt
    #                   ╰── (a) Euler ──╯   ╰── (b) De Hoop ─╯
    #
    # Both minus signs are physical; neither cancels the other.
    #
    # One-way (pv_from_pin) form: the bounded-domain assumption p_out → 0
    # collapses each (p_p, p_v) and (v_p, v_v) pair to a single combined
    # kernel acting on the incoming pressure p_in = (p − z₀·v_n)/2:
    #     p_inner_pin  = ∮[ (1/c)·p_p − p_v ] · p_in dS·dt
    #     vn_inner_pin = ∮[ (1/c)·v_p − v_v ] · p_in dS·dt
    # The "−p_v" / "−v_v" leg carries the same De Hoop dipole minus as
    # the two-way form (`scale_pin_f = -1 · dS · dt` below); the "+(1/c)·p_p"
    # / "+(1/c)·v_p" leg is the Euler-substitution rewrite of the
    # monopole channel for the one-way decomposition (no minus here).
    #
    # Kernel definitions (in `eval_one_kernel`): `p_v ≡ +∂_n G`,
    # `v_v ≡ +∂_t ∂_n G / ρ`. No sign baked into the kernel.
    dS_out      = 4π * R_OUTER^2 / N
    ρ           = dom_pw.r0
    dt_KH       = dom_pw.dt                  # time-integral discretization (Σ_tt ≈ ∫ dt)
    z0_val      = dom_pw.z0
    inv_c       = 1.0 / dom_pw.c0
    p_in_outer  = Float32.((rec.p_outer .- z0_val .* rec.vn_outer) ./ 2)
    # q-channel scale: -ρ from substituting ∂_n p = -ρ·∂_t v_n into the
    # K-H integrand (linearised Euler). The `-` is the pressure-gradient /
    # velocity-time-derivative sign coupling — NOT the De Hoop minus.
    scale_pv_q  = -ρ * dS_out * dt_KH        # K · vn_outer    (q-channel, EULER-substitution minus)
    # f-channel scale: -1 IS the De Hoop dipole/surface-pressure sign
    # equivalence (f = -p·n̂). Previously absorbed into `p_v ≡ -∂_n G`
    # via `-cos_θ_s` in eval_one_kernel; now lifted here so the classical
    # K-H integrand structure `-∂_n G · p` is visible at the sum site.
    scale_pv_f  = -1 * dS_out * dt_KH        # K · p_outer     (f-channel, DE-HOOP-dipole minus)
    # Pin (one-way) scales — same naming logic for the f-leg:
    scale_pin_q =  inv_c * dS_out * dt_KH    # K · p_in_outer  (q-leg /c, unchanged)
    scale_pin_f = -1 * dS_out * dt_KH        # K · p_in_outer  (f-leg, DE-HOOP-dipole minus)

    # Output accumulators (Float64 for numerical headroom; the convolution sum
    # over (n_outer × nt) accumulates millions of products).
    p_inner_KH      = zeros(Float64, dom_pw.nt, N)
    vn_inner_KH     = zeros(Float64, dom_pw.nt, N)
    p_inner_KH_pin  = zeros(Float64, dom_pw.nt, N)
    vn_inner_KH_pin = zeros(Float64, dom_pw.nt, N)

    # Pre-compute diagnostic indices needed to cache per-kernel slices for the
    # kernel-heatmap plot before each kernel is freed.
    ir_diag   = argmin([abs(sp.inner_pts[ir, 2]) + abs(sp.inner_pts[ir, 3])
                        for ir in 1:N])
    r_to_diag = [norm(sp.outer_pts[is, :] .- sp.inner_pts[ir_diag, :])
                 for is in 1:N]
    sort_idx  = sortperm(r_to_diag)
    r_sorted  = r_to_diag[sort_idx]
    diag_slices = Dict{Symbol, Matrix{Float32}}()

    # `trunc+ped` mode: precompute running-sum integrals of the three input
    # traces once per N. Used by direct_kh_accumulate!(K_dc=, cumsum_in=)
    # to add back the τ → ∞ pedestal contribution of the truncated kernel
    # as α · Σ_r K_dc[r,s] · cumsum_in[t,r]. Equivalent to convolving the
    # full kernel; lets us truncate the kernel without losing v_v's
    # non-decaying DC tail.
    cumsum_vn_outer = KERNEL_MODE === :trunc_ped ? cumsum(rec.vn_outer; dims = 1) : nothing
    cumsum_p_outer  = KERNEL_MODE === :trunc_ped ? cumsum(rec.p_outer;  dims = 1) : nothing
    cumsum_p_in     = KERNEL_MODE === :trunc_ped ? cumsum(p_in_outer;   dims = 1) : nothing

    # The per-kernel jobs. Each row: which kernel; which pv accumulator + input
    # + scale; which pin accumulator + pin scale (input is always p_in_outer).
    kernel_jobs = [
        (:p_p, p_inner_KH,  rec.vn_outer, scale_pv_q, p_inner_KH_pin,  scale_pin_q),
        (:p_v, p_inner_KH,  rec.p_outer,  scale_pv_f, p_inner_KH_pin,  scale_pin_f),
        (:v_p, vn_inner_KH, rec.vn_outer, scale_pv_q, vn_inner_KH_pin, scale_pin_q),
        (:v_v, vn_inner_KH, rec.p_outer,  scale_pv_f, vn_inner_KH_pin, scale_pin_f),
    ]
    @info "[Part 3 N=$N] eval + K-H convolve, one kernel at a time" peak_GB_per_kernel=round(nt_kernel*N*N*4/1e9, digits=2)
    t0_total = time()
    for (which, out_pv, in_pv, scale_pv, out_pin, scale_pin) in kernel_jobs
        t0 = time()
        K = eval_one_kernel(which,
                            sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm,
                            T_GRID_KERNEL, σ_T, dom_pw.c0, dom_pw.r0;
                            T = Float32)
        # K shape: (nt_kernel, n_outer, n_inner). Cache the diag-receiver slice
        # with outer sources sorted by r — this is what the heatmap plot wants.
        # Snapshot BEFORE the pedestal subtract so the heatmap shows the
        # physical kernel, not the residual.
        diag_slices[which] = K[:, sort_idx, ir_diag]
        # Pedestal subtract in-place for `trunc+ped` mode: K becomes the
        # residual `K_full − K_dc[r, s]`, whose tail → 0 cleanly so
        # truncation is exact. The pedestal contribution is added back via
        # the K_dc + cumsum_in kwargs to direct_kh_accumulate! below. Only
        # v_v has a non-vanishing pedestal (Gaussian F → 1); the other
        # three kernels' K_dc is the zero matrix, so we skip the subtract.
        K_dc = (KERNEL_MODE === :trunc_ped && which === :v_v) ?
               eval_kernel_pedestal(which,
                                    sp.outer_pts, sp.outer_nrm,
                                    sp.inner_pts, sp.inner_nrm,
                                    dom_pw.r0; T = Float32) :
               nothing
        if K_dc !== nothing
            @inbounds for s in 1:N, r in 1:N
                K_dc_rs = K_dc[r, s]
                K_dc_rs == 0 && continue
                @simd for tt in axes(K, 1)
                    K[tt, r, s] -= K_dc_rs
                end
            end
        end
        # Match the cumsum-of-input to the kernel's pv input. (pin input is
        # always p_in_outer, so cumsum_p_in is reused across all four jobs.)
        cs_pv = KERNEL_MODE === :trunc_ped ?
                (in_pv === rec.vn_outer ? cumsum_vn_outer : cumsum_p_outer) :
                nothing
        # Accumulate into BOTH formulations before freeing this kernel.
        direct_kh_accumulate!(out_pv,  K, in_pv;      α = scale_pv,
                              K_dc = K_dc, cumsum_in = cs_pv)
        direct_kh_accumulate!(out_pin, K, p_in_outer; α = scale_pin,
                              K_dc = K_dc, cumsum_in = cumsum_p_in)
        @printf("[Part 3 N=%d]   %s  wall: %.2f s   (eval+pv+pin)\n", N, string(which), time() - t0)
        K    = nothing
        K_dc = nothing
        GC.gc()
    end
    @printf("[Part 3 N=%d] kernel+convolution total wall: %.2f s\n", N, time() - t0_total)

    # ---- Layer (a) errors — computed BEFORE plotting so the plots can annotate
    # the formulation comparison directly. Two metrics per formulation:
    #
    #   • per-pt rel-RMS  norm(KH[:, ir] - rec[:, ir]) / norm(rec[:, ir])
    #     — used in the outlier-scatter plot. Beware: for vn at equatorial
    #       inner pts (`θ ≈ 90°`) the recorded vn ≈ 0 → division blows up.
    #       Per-pt metric is a SHAPE diagnostic, not a quality number.
    #
    #   • global rel-RMS  norm(KH - rec) / norm(rec)
    #     — single robust scalar per channel × formulation; immune to the
    #       equator artifact. THIS is the trustworthy convergence metric.
    p_err_per_pt      = [norm(p_inner_KH[:, ir]      .- rec.p_inner[:, ir]) /
                         max(norm(rec.p_inner[:, ir]), eps()) for ir in 1:N]
    vn_err_per_pt     = [norm(vn_inner_KH[:, ir]     .- rec.vn_inner[:, ir]) /
                         max(norm(rec.vn_inner[:, ir]), eps()) for ir in 1:N]
    p_err_per_pt_pin  = [norm(p_inner_KH_pin[:, ir]  .- rec.p_inner[:, ir]) /
                         max(norm(rec.p_inner[:, ir]), eps()) for ir in 1:N]
    vn_err_per_pt_pin = [norm(vn_inner_KH_pin[:, ir] .- rec.vn_inner[:, ir]) /
                         max(norm(rec.vn_inner[:, ir]), eps()) for ir in 1:N]
    p_err_global      = norm(p_inner_KH      .- rec.p_inner ) / norm(rec.p_inner)
    vn_err_global     = norm(vn_inner_KH     .- rec.vn_inner) / norm(rec.vn_inner)
    p_err_global_pin  = norm(p_inner_KH_pin  .- rec.p_inner ) / norm(rec.p_inner)
    vn_err_global_pin = norm(vn_inner_KH_pin .- rec.vn_inner) / norm(rec.vn_inner)
    # Absolute (un-normalised) time-RMS residuals in physical units (Pa for
    # p, m/s for vn). Immune to the equator denominator collapse that
    # inflates the per-point rel-RMS for vn near θ=90°.
    #   per-pt:  sqrt(mean_t (KH[:, ir] − rec[:, ir]).^2)
    #   global:  sqrt(mean_{t,ir} (KH .− rec).^2)
    _nT  = size(rec.p_inner, 1)
    _nTN = length(rec.p_inner)
    p_abs_per_pt      = [norm(p_inner_KH[:, ir]      .- rec.p_inner[:, ir])  / sqrt(_nT) for ir in 1:N]
    vn_abs_per_pt     = [norm(vn_inner_KH[:, ir]     .- rec.vn_inner[:, ir]) / sqrt(_nT) for ir in 1:N]
    p_abs_per_pt_pin  = [norm(p_inner_KH_pin[:, ir]  .- rec.p_inner[:, ir])  / sqrt(_nT) for ir in 1:N]
    vn_abs_per_pt_pin = [norm(vn_inner_KH_pin[:, ir] .- rec.vn_inner[:, ir]) / sqrt(_nT) for ir in 1:N]
    p_abs_global      = norm(p_inner_KH      .- rec.p_inner ) / sqrt(_nTN)
    vn_abs_global     = norm(vn_inner_KH     .- rec.vn_inner) / sqrt(_nTN)
    p_abs_global_pin  = norm(p_inner_KH_pin  .- rec.p_inner ) / sqrt(_nTN)
    vn_abs_global_pin = norm(vn_inner_KH_pin .- rec.vn_inner) / sqrt(_nTN)
    @printf("[Part 3a N=%d] pv_from_pv  GLOBAL rel-RMS  p=%.3g  vn=%.3g     per-pt vn median=%.3g  max=%.3g\n",
            N, p_err_global,  vn_err_global, median(vn_err_per_pt), maximum(vn_err_per_pt))
    @printf("[Part 3a N=%d] pv_from_pin GLOBAL rel-RMS  p=%.3g  vn=%.3g     (one-way form, ~half the cost)\n",
            N, p_err_global_pin, vn_err_global_pin)

    # ---- Per-time-step rel-RMS over inner-sphere receivers. Equivalence
    # between pin and pv must hold at every t independently, so the
    # time-integrated GLOBAL rel-RMS above is insufficient: a constant
    # scale bug, an early-time transient artifact, and a late-time
    # dispersion drift would all give different time-resolved signatures
    # while collapsing into similar GLOBAL numbers.
    let
        nt_p = dom_pw.nt
        # rel-RMS over the n_inner receivers, separately per t.
        function _rel_rms_t(KH, R)
            out = zeros(Float64, nt_p)
            @inbounds for ti in 1:nt_p
                num = 0.0
                den = 0.0
                @inbounds for ir in 1:N
                    d = KH[ti, ir] - R[ti, ir]
                    num += d * d
                    den += R[ti, ir] * R[ti, ir]
                end
                out[ti] = sqrt(num) / max(sqrt(den), eps())
            end
            return out
        end
        p_rel_t_pv   = _rel_rms_t(p_inner_KH,      rec.p_inner)
        p_rel_t_pin  = _rel_rms_t(p_inner_KH_pin,  rec.p_inner)
        vn_rel_t_pv  = _rel_rms_t(vn_inner_KH,     rec.vn_inner)
        vn_rel_t_pin = _rel_rms_t(vn_inner_KH_pin, rec.vn_inner)
        t_full_ms    = (0:(nt_p-1)) .* dom_pw.dt .* 1e3

        fig = Figure(size = (1500, 600))
        Label(fig[0, 1:2],
              "Part 3 — per-time-step rel-RMS over inner-sphere receivers (N=$N).   "*
              "If pv and pin curves overlay within ~1× at every t, the two forms are "*
              "equivalent; the integrated GLOBAL rel-RMS can hide shape mismatches.";
              fontsize = 12, font = :bold)
        function _t_panel!(col, y_pv, y_pin, channel, gerr_pv, gerr_pin)
            ax = Axis(fig[1, col];
                      title  = @sprintf("%s_inner   GLOBAL rel-RMS  pv=%.3g   pin=%.3g",
                                        channel, gerr_pv, gerr_pin),
                      xlabel = "t [ms]", ylabel = "rel-RMS over inner pts",
                      yscale = log10)
            lines!(ax, t_full_ms, max.(y_pv,  1e-6);
                   color = _TAB_RED,  linewidth = 1.5, linestyle = :dash,
                   label = "pv_from_pv  (2-way)")
            lines!(ax, t_full_ms, max.(y_pin, 1e-6);
                   color = _TAB_BLUE, linewidth = 1.5, linestyle = :dot,
                   label = "pv_from_pin (1-way)")
            axislegend(ax, position = :rt)
        end
        _t_panel!(1, p_rel_t_pv,  p_rel_t_pin,  "p",  p_err_global,  p_err_global_pin)
        _t_panel!(2, vn_rel_t_pv, vn_rel_t_pin, "vn", vn_err_global, vn_err_global_pin)
        p_trel = joinpath(DIAG_DIR,
            "part3_pin_vs_pv_time_resolved_n$(lpad(N, 4, '0'))_$(_MED_TAG).png")
        save(p_trel, fig)
        @info "[Part 3 N=$N] Saved per-time-step pin-vs-pv rel-RMS curves" p_trel
    end

    # ---- Diagnostic plot 1: 2×2 heatmap of the raw analytical kernels at the
    # diagnostic inner receiver (`ir_diag`), with outer sources sorted by
    # distance. The t = r/c arrival should appear as a clean diagonal.
    # Uses `diag_slices` cached during the per-kernel accumulation loop above
    # (the full kernels are freed before we get here).
    let
        t_axis_ms = (0:(nt_kernel-1)) .* dom_pw.dt .* 1e3
        fig = Figure(size = (1500, 1000))
        Label(fig[0, 1:2],
              "Part 3 — analytical kernels at inner pt $ir_diag  (xyz=$(round.(sp.inner_pts[ir_diag, :], digits=3)))   "*
              "N=$N, σ_t=$(round(σ_T*1e6, digits=2)) µs, nt_kernel=$nt_kernel of $(dom_pw.nt)";
              fontsize = 14, font = :bold)
        function _plot_kernel!(row, col, M, name)
            # M shape (nt_kernel, n_outer_sorted) — pre-sliced + r-sorted.
            v = quantile(abs.(vec(Float64.(M))), 0.999)
            v = v == 0 ? 1.0 : v
            ax = Axis(fig[row, col]; title = name,
                      xlabel = "outer-src distance r [m] (sorted)",
                      ylabel = "t [ms]", yreversed = true)
            heatmap!(ax, r_sorted, t_axis_ms, Float32.(transpose(M));
                     colormap = :balance, colorrange = (-v, v))
            # Overlay the analytical arrival t = r/c.
            lines!(ax, r_sorted, (r_sorted ./ dom_pw.c0) .* 1e3;
                   color = :black, linewidth = 1, linestyle = :dash)
        end
        _plot_kernel!(1, 1, diag_slices[:p_p], "p_p — pressure ← monopole  [1/(m·s²)]")
        _plot_kernel!(1, 2, diag_slices[:p_v], "p_v — pressure ← dipole  [1/(m²·s)]")
        _plot_kernel!(2, 1, diag_slices[:v_p], "v_p — velocity ← monopole  [m/(kg·s)]")
        _plot_kernel!(2, 2, diag_slices[:v_v], "v_v — velocity ← dipole  [1/kg]")
        p_kern = joinpath(DIAG_DIR, "part3_kernels_n$(lpad(N, 4, '0'))_$(_MED_TAG).png")
        save(p_kern, fig)
        @info "[Part 3 N=$N] Saved kernel heatmap" p_kern
    end

    # ---- Diagnostic plot 2: K-H extrapolation vs recorded inner field at the
    # diagnostic inner point. Three lines per panel — recorded (black solid),
    # pv_from_pv (red dashed, two-way 4-kernel form), pv_from_pin (blue dotted,
    # one-way 1-kernel form). Global rel-RMS errors annotated in the panel
    # titles so the plot is self-explanatory.
    let
        t_full_ms = (0:(dom_pw.nt-1)) .* dom_pw.dt .* 1e3
        fig2 = Figure(size = (1500, 600))
        Label(fig2[0, 1:2],
              "Part 3 — K-H extrapolation vs recorded inner field at pt $ir_diag (N=$N)   "*
              "[ black = recorded,   red dashed = pv_from_pv (2-way, 4 kernels),   "*
              "blue dotted = pv_from_pin (1-way, 1 combined kernel × p_in) ]";
              fontsize = 13, font = :bold)
        axp = Axis(fig2[1, 1];
                   title  = @sprintf("p_inner  [Pa]   GLOBAL rel-RMS:  pv=%.3g   pin=%.3g",
                                     p_err_global, p_err_global_pin),
                   xlabel = "t [ms]", ylabel = "p [Pa]")
        lines!(axp, t_full_ms, Float64.(rec.p_inner[:, ir_diag]);
               color = :black,    linewidth = 2,   label = "recorded")
        lines!(axp, t_full_ms, p_inner_KH[:, ir_diag];
               color = _TAB_RED,  linewidth = 1.5, linestyle = :dash, label = "pv_from_pv")
        lines!(axp, t_full_ms, p_inner_KH_pin[:, ir_diag];
               color = _TAB_BLUE, linewidth = 1.5, linestyle = :dot,  label = "pv_from_pin")
        axislegend(axp, position = :rt)
        axv = Axis(fig2[1, 2];
                   title  = @sprintf("vn_inner  [m/s]   GLOBAL rel-RMS:  pv=%.3g   pin=%.3g",
                                     vn_err_global, vn_err_global_pin),
                   xlabel = "t [ms]", ylabel = "vn [m/s]")
        lines!(axv, t_full_ms, Float64.(rec.vn_inner[:, ir_diag]);
               color = :black,    linewidth = 2,   label = "recorded")
        lines!(axv, t_full_ms, vn_inner_KH[:, ir_diag];
               color = _TAB_RED,  linewidth = 1.5, linestyle = :dash, label = "pv_from_pv")
        lines!(axv, t_full_ms, vn_inner_KH_pin[:, ir_diag];
               color = _TAB_BLUE, linewidth = 1.5, linestyle = :dot,  label = "pv_from_pin")
        axislegend(axv, position = :rt)
        p_KHcomp = joinpath(DIAG_DIR, "part3_khcomp_n$(lpad(N, 4, '0'))_$(_MED_TAG).png")
        save(p_KHcomp, fig2)
        @info "[Part 3 N=$N] Saved K-H comparison plot" p_KHcomp
    end

    # ---- Diagnostic plot 3: 2×5 wavefield heatmap on the inner sphere.
    # Rows = channel (p, vn);
    # columns = (recorded | pv_from_pv | pv − rec | pv_from_pin | pin − rec).
    # Inner points sorted by incident angle θ_inc = acos(-x/R_in) so θ=0° (left
    # edge) is the first inner pt reached by the +x-propagating plane wave and
    # θ=180° (right edge) is the last. The three FIELD columns (rec | pv | pin)
    # share a symmetric colour range per row so the formulations are directly
    # comparable; the two DIFF columns share a SEPARATE per-row symmetric range
    # (much smaller than the field), so error structure is visible without
    # being crushed by the field scale. Time top→bottom (seismic convention).
    let
        θ_inc      = acos.(clamp.(-sp.inner_pts[:, 1] ./ R_INNER, -1.0, 1.0))
        θ_sort     = sortperm(θ_inc)
        θ_deg      = θ_inc[θ_sort] .* 180/π
        t_full_ms  = (0:(dom_pw.nt-1)) .* dom_pw.dt .* 1e3
        p_rec_s    = rec.p_inner[:, θ_sort]
        p_pv_s     = p_inner_KH[:, θ_sort]
        p_pin_s    = p_inner_KH_pin[:, θ_sort]
        vn_rec_s   = rec.vn_inner[:, θ_sort]
        vn_pv_s    = vn_inner_KH[:, θ_sort]
        vn_pin_s   = vn_inner_KH_pin[:, θ_sort]
        # Diff fields (extrapolated − recorded) per method/channel.
        p_pv_d_s   = p_pv_s   .- Float64.(p_rec_s)
        p_pin_d_s  = p_pin_s  .- Float64.(p_rec_s)
        vn_pv_d_s  = vn_pv_s  .- Float64.(vn_rec_s)
        vn_pin_d_s = vn_pin_s .- Float64.(vn_rec_s)

        sym_q(M, q=0.99) = let v = quantile(abs.(vec(Float64.(M))), q); v == 0 ? 1.0 : v end
        v_p_max    = max(sym_q(p_rec_s),    sym_q(p_pv_s),    sym_q(p_pin_s))
        v_vn_max   = max(sym_q(vn_rec_s),   sym_q(vn_pv_s),   sym_q(vn_pin_s))
        v_p_diff   = max(sym_q(p_pv_d_s),   sym_q(p_pin_d_s))
        v_vn_diff  = max(sym_q(vn_pv_d_s),  sym_q(vn_pin_d_s))

        fig3 = Figure(size = (2200, 900))
        Label(fig3[0, 1:5],
              "Part 3 — inner-sphere wavefield heatmaps (N=$N).   "*
              "x = incident angle θ wrt +x  (0° = first hit by plane wave, 180° = exit pole).   "*
              "y = t [ms], top→bottom.   "*
              "Field cols share a per-row scale; diff cols share a separate (smaller) per-row scale.";
              fontsize = 13, font = :bold)

        # Column headers (one per data column).
        for (c, t) in enumerate(("recorded", "pv_from_pv (2-way)", "pv − rec",
                                 "pv_from_pin (1-way)", "pin − rec"))
            Label(fig3[1, c], t; fontsize = 14, font = :bold,
                  padding = (0, 0, 0, 4))
        end

        function _wf_panel!(row, col, M, subtitle, vmax; bottom, ylabel)
            ax = Axis(fig3[row, col];
                      title    = subtitle, titlesize = 12,
                      xlabel   = bottom ? "θ [°]" : "",
                      ylabel   = ylabel,                # "" disables it
                      yreversed = true,
                      xticklabelsvisible = bottom,
                      yticklabelsvisible = ylabel != "")
            heatmap!(ax, θ_deg, t_full_ms, Float32.(transpose(M));
                     colormap = :balance, colorrange = (-vmax, vmax))
        end
        # Channel name is encoded once per row, in the leftmost ylabel.
        # Row 2: p_inner — xlabel hidden (top row); ylabel labels the channel
        _wf_panel!(2, 1, p_rec_s,    "",                                            v_p_max;   bottom=false, ylabel="p [Pa]    |    t [ms]")
        _wf_panel!(2, 2, p_pv_s,     @sprintf("rel-RMS %.3g",  p_err_global),       v_p_max;   bottom=false, ylabel="")
        _wf_panel!(2, 3, p_pv_d_s,   @sprintf("|max| %.3g",    v_p_diff),           v_p_diff;  bottom=false, ylabel="")
        _wf_panel!(2, 4, p_pin_s,    @sprintf("rel-RMS %.3g",  p_err_global_pin),   v_p_max;   bottom=false, ylabel="")
        _wf_panel!(2, 5, p_pin_d_s,  @sprintf("|max| %.3g",    v_p_diff),           v_p_diff;  bottom=false, ylabel="")
        # Row 3: vn_inner — xlabel shown (bottom row)
        _wf_panel!(3, 1, vn_rec_s,   "",                                            v_vn_max;  bottom=true,  ylabel="vn [m/s]    |    t [ms]")
        _wf_panel!(3, 2, vn_pv_s,    @sprintf("rel-RMS %.3g",  vn_err_global),      v_vn_max;  bottom=true,  ylabel="")
        _wf_panel!(3, 3, vn_pv_d_s,  @sprintf("|max| %.3g",    v_vn_diff),          v_vn_diff; bottom=true,  ylabel="")
        _wf_panel!(3, 4, vn_pin_s,   @sprintf("rel-RMS %.3g",  vn_err_global_pin),  v_vn_max;  bottom=true,  ylabel="")
        _wf_panel!(3, 5, vn_pin_d_s, @sprintf("|max| %.3g",    v_vn_diff),          v_vn_diff; bottom=true,  ylabel="")

        # Force the 5 panel columns to equal widths so the figure canvas fills.
        # (Without this, Makie auto-sizes per content and the right side of the
        # canvas is left blank.)
        for c in 1:5
            colsize!(fig3.layout, c, Relative(0.2))
        end

        p_wf = joinpath(DIAG_DIR, "part3_wavefield_n$(lpad(N, 4, '0'))_$(_MED_TAG).png")
        save(p_wf, fig3)
        @info "[Part 3 N=$N] Saved inner-wavefield heatmap" p_wf
    end

    # ---- Diagnostic plot 3b: outer-sphere planar decomposition heatmap.
    # Shows what pin "sees" vs "discards" at the input of the K-H integral.
    # 1×4 layout: p_outer | z₀·vn_outer (scaled to Pa for comparison) | p_in
    # | p_out. All four share a single symmetric colour scale (Pa). For a
    # plane wave passing through:
    #   • Upstream pole (vn=−p/z₀): p_in ≈ p,    p_out ≈ 0
    #   • Equator      (vn≈ 0    ): p_in ≈ p/2,  p_out ≈ p/2  (planar
    #     decomposition has no physical meaning for grazing incidence)
    #   • Downstream pole (vn=+p/z₀): p_in ≈ 0,  p_out ≈ p
    # The pin K-H sum sees only p_in; the p_out column is the part of the
    # outer field that pin discards (and that pv keeps via the separate
    # p_outer / vn_outer kernels).
    let
        θ_inc_o   = acos.(clamp.(-sp.outer_pts[:, 1] ./ R_OUTER, -1.0, 1.0))
        θ_sort_o  = sortperm(θ_inc_o)
        θ_deg_o   = θ_inc_o[θ_sort_o] .* 180/π
        t_full_ms = (0:(dom_pw.nt-1)) .* dom_pw.dt .* 1e3

        p_o_s     = rec.p_outer[:, θ_sort_o]
        z0vn_o_s  = z0_val .* rec.vn_outer[:, θ_sort_o]
        p_in_o_s  = (Float64.(p_o_s) .- Float64.(z0vn_o_s)) ./ 2
        p_out_o_s = (Float64.(p_o_s) .+ Float64.(z0vn_o_s)) ./ 2

        sym_q(M, q=0.99) = let v = quantile(abs.(vec(Float64.(M))), q); v == 0 ? 1.0 : v end
        v_max = max(sym_q(p_o_s), sym_q(z0vn_o_s), sym_q(p_in_o_s), sym_q(p_out_o_s))

        fig4 = Figure(size = (1900, 600))
        Label(fig4[0, 1:4],
              "Part 3 — outer-sphere planar decomposition (N=$N).   "*
              "x = θ_outer wrt +x  (0° = first hit by plane wave, 180° = exit pole).   "*
              "y = t [ms], top→bottom.   "*
              "All panels in Pa, shared symmetric scale.   "*
              "pin's K-H sum sees only column 3 (p_in); column 4 (p_out) is what pin discards.";
              fontsize = 13, font = :bold)

        for (c, t) in enumerate(("p_outer", "z₀·vn_outer", "p_in = (p − z₀·vn)/2",
                                 "p_out = (p + z₀·vn)/2"))
            Label(fig4[1, c], t; fontsize = 14, font = :bold, padding = (0, 0, 0, 4))
        end

        function _outer_panel!(col, M, ylabel)
            ax = Axis(fig4[2, col];
                      xlabel = "θ_outer [°]",
                      ylabel = ylabel,
                      yreversed = true,
                      yticklabelsvisible = ylabel != "")
            heatmap!(ax, θ_deg_o, t_full_ms, Float32.(transpose(M));
                     colormap = :balance, colorrange = (-v_max, v_max))
        end
        _outer_panel!(1, p_o_s,     "t [ms]")
        _outer_panel!(2, z0vn_o_s,  "")
        _outer_panel!(3, p_in_o_s,  "")
        _outer_panel!(4, p_out_o_s, "")

        for c in 1:4
            colsize!(fig4.layout, c, Relative(0.25))
        end

        p_outer_wf = joinpath(DIAG_DIR,
            "part3_outer_decomp_n$(lpad(N, 4, '0'))_$(_MED_TAG).png")
        save(p_outer_wf, fig4)
        @info "[Part 3 N=$N] Saved outer-sphere planar decomposition" p_outer_wf
    end

    # ---- Per-inner-point K-H error vs incident angle θ_inc — separates the
    # two channels (p, vn) into their own panels so each gets an independent
    # log-y range (vn's equatorial pathology no longer crushes the p band),
    # and overlays the two K-H forms (pv_from_pv vs pv_from_pin) in each panel
    # so the pv-vs-pin comparison is co-located rather than spread across rows.
    # θ_inc is measured from +x (plane-wave arrival direction): 0° = upstream
    # pole hit first; 90° = equator (vn pathology peaks here — grazing
    # incidence, small denominator); 180° = downstream pole.
    let
        θ_inc           = acos.(clamp.(-sp.inner_pts[:, 1] ./ R_INNER, -1.0, 1.0))
        θ_deg_per_inner = θ_inc .* 180/π

        fig = Figure(size = (1600, 650))

        # One panel per channel; overlay the two K-H forms. GLOBAL rel-RMS for
        # each formulation goes into the panel title (no median lines/text —
        # the per-point scatter and the GLOBAL number cover the same ground).
        function _channel_panel!(col, err_pv, err_pin, gerr_pv, gerr_pin, channel)
            ax = Axis(fig[1, col];
                      title = @sprintf("%s_inner: K-H vs recorded   GLOBAL rel-RMS  pv=%.3g   pin=%.3g",
                                       channel, gerr_pv, gerr_pin) * _MODE_STR,
                      xlabel = "polar angle θ [°]   (0° = upstream pole, wave enters;   "*
                               "180° = downstream pole, wave exits)",
                      ylabel = "per-point rel-RMS error",
                      yscale = log10,
                      xticks = 0:30:180)
            xlims!(ax, -4, 184)
            vlines!(ax, [0.0];   color = :seagreen,   linestyle = :dash, linewidth = 2.5,
                    label = "wave enters  (θ=0°, −x pole)")
            vlines!(ax, [180.0]; color = :darkorange, linestyle = :dash, linewidth = 2.5,
                    label = "wave exits   (θ=180°, +x pole)")
            scatter!(ax, θ_deg_per_inner, max.(err_pv,  1e-6);
                     color = _TAB_RED,  markersize = 6, label = "pv_from_pv  (2-way)")
            scatter!(ax, θ_deg_per_inner, max.(err_pin, 1e-6);
                     color = _TAB_BLUE, markersize = 6, label = "pv_from_pin (1-way)")
            axislegend(ax, position = :rt, framevisible = false, labelsize = 10)
        end

        _channel_panel!(1, p_err_per_pt,  p_err_per_pt_pin,
                        p_err_global,  p_err_global_pin,  "p")
        _channel_panel!(2, vn_err_per_pt, vn_err_per_pt_pin,
                        vn_err_global, vn_err_global_pin, "vn")

        p_outl = joinpath(DIAG_DIR, "part3_per_point_err_n$(lpad(N, 4, '0'))_$(_MED_TAG)$(_KH_MODE_TAG).png")
        save(p_outl, fig)
        @info "[Part 3 N=$N] Saved per-inner-point error distribution" p_outl
    end

    # ---- Spatial-map view of the same per-point errors: scatter each inner
    # point on the (θ_inc, φ_inc) plane coloured by log₁₀(rel-RMS). This shows
    # *where on the inner surface* errors concentrate — the equatorial-band
    # vn pathology, polar caps, any φ-asymmetry — rather than just the
    # marginal distribution vs θ. (θ, φ) is the natural spherical chart with
    # +x as the polar axis (plane-wave arrival direction): θ = angle from +x
    # (0..180°), φ = azimuth in the (y, z) plane (-180..180°).
    # Per-channel shared colour scale (left col: p; right col: vn) so the
    # pv_from_pv (top) vs pv_from_pin (bottom) comparison is direct.
    let
        θ_inc           = acos.(clamp.(-sp.inner_pts[:, 1] ./ R_INNER, -1.0, 1.0))
        θ_deg_per_inner = θ_inc .* 180/π
        φ_deg_per_inner = atan.(sp.inner_pts[:, 3], sp.inner_pts[:, 2]) .* 180/π

        # Shared log-colour ranges (clamp floor at 1e-6 so log10 is finite).
        _logc(x) = log10.(max.(x, 1e-6))
        p_lo  = min(minimum(_logc(p_err_per_pt)),  minimum(_logc(p_err_per_pt_pin)))
        p_hi  = max(maximum(_logc(p_err_per_pt)),  maximum(_logc(p_err_per_pt_pin)))
        vn_lo = min(minimum(_logc(vn_err_per_pt)), minimum(_logc(vn_err_per_pt_pin)))
        vn_hi = max(maximum(_logc(vn_err_per_pt)), maximum(_logc(vn_err_per_pt_pin)))

        fig = Figure(size = (1700, 1050))
        Label(fig[0, 1:4],
              "Part 3 — per-inner-point K-H error  spatial map  (N=$N)$(_MODE_STR)";
              fontsize = 13, font = :bold)
        Label(fig[3, 1:4],
              "(θ, φ) chart with +x as polar axis (= plane-wave direction).   "*
              "Green dashed = θ=0° (−x pole, wave enters);   "*
              "Orange dashed = θ=180° (+x pole, wave exits);   "*
              "Black dotted = θ=90° (equator).   "*
              "Each pole is one 3D point spanning the full φ-line (chart degeneracy).";
              fontsize = 11, color = :gray30)

        # Marker size: shrink as N grows so the dots don't overlap into blobs.
        msize = clamp(round(Int, 500 / sqrt(N)), 6, 18)

        function _map_panel!(row, col, err, title_, crange)
            ax = Axis(fig[row, col]; title = title_,
                      xlabel = "polar angle θ [°]",
                      ylabel = "azimuth φ [°]",
                      xticks = 0:30:180, yticks = -180:90:180)
            # Pad xlims a touch so markers at θ=0/180 sit *inside* the frame
            # and aren't clipped by the axis spine.
            xlims!(ax, -4, 184); ylims!(ax, -180, 180)
            sc = scatter!(ax, θ_deg_per_inner, φ_deg_per_inner;
                          color = _logc(err),
                          colormap = :plasma, colorrange = crange,
                          markersize = msize)
            # Plane-wave entry / exit / equator markers (axially symmetric about +x).
            # In the (θ, φ) chart each pole is a single 3D point but spans the
            # full vertical line in φ (chart degeneracy at the poles).
            vlines!(ax, [0.0];   color = :seagreen,   linestyle = :dash, linewidth = 3.0)
            vlines!(ax, [180.0]; color = :darkorange, linestyle = :dash, linewidth = 3.0)
            # Equator: black-dotted reads against both ends of the plasma map
            # (purple lows AND yellow highs); white was invisible on yellow.
            vlines!(ax, [90.0];  color = :black,      linestyle = :dot,  linewidth = 2.0)
            return sc
        end

        sc_p_pv = _map_panel!(1, 1, p_err_per_pt,
                              @sprintf("pv_from_pv (2-way): p_inner err   (GLOBAL %.3g)",  p_err_global),
                              (p_lo, p_hi))
        sc_vn_pv = _map_panel!(1, 3, vn_err_per_pt,
                              @sprintf("pv_from_pv (2-way): vn_inner err  (GLOBAL %.3g)",  vn_err_global),
                              (vn_lo, vn_hi))
        _map_panel!(2, 1, p_err_per_pt_pin,
                    @sprintf("pv_from_pin (1-way): p_inner err  (GLOBAL %.3g)", p_err_global_pin),
                    (p_lo, p_hi))
        _map_panel!(2, 3, vn_err_per_pt_pin,
                    @sprintf("pv_from_pin (1-way): vn_inner err (GLOBAL %.3g)", vn_err_global_pin),
                    (vn_lo, vn_hi))

        Colorbar(fig[1:2, 2], sc_p_pv;  label = "log₁₀(p_inner rel-RMS)")
        Colorbar(fig[1:2, 4], sc_vn_pv; label = "log₁₀(vn_inner rel-RMS)")

        # Make the panels much wider than the colorbars.
        colsize!(fig.layout, 1, Relative(0.42))
        colsize!(fig.layout, 2, Relative(0.05))
        colsize!(fig.layout, 3, Relative(0.42))
        colsize!(fig.layout, 4, Relative(0.05))

        p_map = joinpath(DIAG_DIR, "part3_per_point_err_map_n$(lpad(N, 4, '0'))_$(_MED_TAG)$(_KH_MODE_TAG).png")
        save(p_map, fig)
        @info "[Part 3 N=$N] Saved per-inner-point error spatial map" p_map
    end

    # ---- Absolute (physical-unit) versions of the two per-point error plots.
    # Same layouts and markers as the relative versions, but the y-axis /
    # colorbar shows the time-RMS residual in Pa (p) or m/s (vn) — no
    # denominator collapse at the equator, so linear scale works fine.
    let
        θ_inc           = acos.(clamp.(-sp.inner_pts[:, 1] ./ R_INNER, -1.0, 1.0))
        θ_deg_per_inner = θ_inc .* 180/π

        # --- Marginal plot: per-point abs RMS vs θ_inc, linear scale ---
        fig_abs = Figure(size = (1600, 650))
        function _abs_channel_panel!(col, err_pv, err_pin, gerr_pv, gerr_pin, channel, unit)
            ax = Axis(fig_abs[1, col];
                      title = @sprintf("%s_inner: K-H vs recorded   GLOBAL abs RMS  pv=%.3g %s   pin=%.3g %s",
                                       channel, gerr_pv, unit, gerr_pin, unit) * _MODE_STR,
                      xlabel = "polar angle θ [°]   (0° = upstream pole, wave enters;   "*
                               "180° = downstream pole, wave exits)",
                      ylabel = "per-point abs RMS error [$unit]",
                      xticks = 0:30:180)
            xlims!(ax, -4, 184)
            vlines!(ax, [0.0];   color = :seagreen,   linestyle = :dash, linewidth = 2.5,
                    label = "wave enters  (θ=0°, −x pole)")
            vlines!(ax, [180.0]; color = :darkorange, linestyle = :dash, linewidth = 2.5,
                    label = "wave exits   (θ=180°, +x pole)")
            scatter!(ax, θ_deg_per_inner, err_pv;
                     color = _TAB_RED,  markersize = 6, label = "pv_from_pv  (2-way)")
            scatter!(ax, θ_deg_per_inner, err_pin;
                     color = _TAB_BLUE, markersize = 6, label = "pv_from_pin (1-way)")
            axislegend(ax, position = :rt, framevisible = false, labelsize = 10)
        end
        _abs_channel_panel!(1, p_abs_per_pt,  p_abs_per_pt_pin,
                            p_abs_global,  p_abs_global_pin,  "p",  "Pa")
        _abs_channel_panel!(2, vn_abs_per_pt, vn_abs_per_pt_pin,
                            vn_abs_global, vn_abs_global_pin, "vn", "m/s")

        p_abs_outl = joinpath(DIAG_DIR, "part3_per_point_err_abs_n$(lpad(N, 4, '0'))_$(_MED_TAG)$(_KH_MODE_TAG).png")
        save(p_abs_outl, fig_abs)
        @info "[Part 3 N=$N] Saved per-inner-point ABS error distribution" p_abs_outl

        # --- Spatial map: per-point abs RMS on (θ, φ), linear colour scale ---
        φ_deg_per_inner = atan.(sp.inner_pts[:, 3], sp.inner_pts[:, 2]) .* 180/π
        # Shared linear colour range per column (p left, vn right).
        p_hi_abs  = max(maximum(p_abs_per_pt),  maximum(p_abs_per_pt_pin))
        vn_hi_abs = max(maximum(vn_abs_per_pt), maximum(vn_abs_per_pt_pin))

        fig_map_abs = Figure(size = (1700, 1050))
        msize_abs = clamp(round(Int, 500 / sqrt(N)), 6, 18)

        function _abs_map_panel!(row, col, err, title_, chi)
            ax = Axis(fig_map_abs[row, col]; title = title_,
                      xlabel = "polar angle θ [°]",
                      ylabel = "azimuth φ [°]",
                      xticks = 0:30:180, yticks = -180:90:180)
            xlims!(ax, -4, 184); ylims!(ax, -180, 180)
            sc = scatter!(ax, θ_deg_per_inner, φ_deg_per_inner;
                          color = err,
                          colormap = :plasma, colorrange = (0.0, chi),
                          markersize = msize_abs)
            vlines!(ax, [0.0];   color = :seagreen,   linestyle = :dash, linewidth = 3.0)
            vlines!(ax, [180.0]; color = :darkorange, linestyle = :dash, linewidth = 3.0)
            vlines!(ax, [90.0];  color = :black,      linestyle = :dot,  linewidth = 2.0)
            return sc
        end

        sc_p_pv_abs = _abs_map_panel!(1, 1, p_abs_per_pt,
                                      @sprintf("pv_from_pv (2-way): p_inner abs RMS   (GLOBAL %.3g Pa)",
                                               p_abs_global),
                                      p_hi_abs)
        sc_vn_pv_abs = _abs_map_panel!(1, 3, vn_abs_per_pt,
                                       @sprintf("pv_from_pv (2-way): vn_inner abs RMS  (GLOBAL %.3g m/s)",
                                                vn_abs_global),
                                       vn_hi_abs)
        _abs_map_panel!(2, 1, p_abs_per_pt_pin,
                        @sprintf("pv_from_pin (1-way): p_inner abs RMS  (GLOBAL %.3g Pa)",
                                 p_abs_global_pin),
                        p_hi_abs)
        _abs_map_panel!(2, 3, vn_abs_per_pt_pin,
                        @sprintf("pv_from_pin (1-way): vn_inner abs RMS (GLOBAL %.3g m/s)",
                                 vn_abs_global_pin),
                        vn_hi_abs)

        Colorbar(fig_map_abs[1:2, 2], sc_p_pv_abs;  label = "p_inner abs RMS  [Pa]")
        Colorbar(fig_map_abs[1:2, 4], sc_vn_pv_abs; label = "vn_inner abs RMS  [m/s]")

        Label(fig_map_abs[3, 1:4],
              "(θ, φ) chart with +x as polar axis (= plane-wave direction).   "*
              "Green dashed = θ=0° (−x pole, wave enters);   "*
              "Orange dashed = θ=180° (+x pole, wave exits);   "*
              "Black dotted = θ=90° (equator).   "*
              "Each pole is one 3D point spanning the full φ-line (chart degeneracy).";
              fontsize = 11, color = :gray30)

        colsize!(fig_map_abs.layout, 1, Relative(0.42))
        colsize!(fig_map_abs.layout, 2, Relative(0.05))
        colsize!(fig_map_abs.layout, 3, Relative(0.42))
        colsize!(fig_map_abs.layout, 4, Relative(0.05))

        p_abs_map = joinpath(DIAG_DIR, "part3_per_point_err_abs_map_n$(lpad(N, 4, '0'))_$(_MED_TAG)$(_KH_MODE_TAG).png")
        save(p_abs_map, fig_map_abs)
        @info "[Part 3 N=$N] Saved per-inner-point ABS error spatial map" p_abs_map
    end

    # ---- Layer (b): inject the K-H-extrapolated inner field; compute s_opt.
    # Cache key embeds the kernel-mode tag so flipping --kernel-mode
    # invalidates only the affected cache entries.
    runb_kh_path = joinpath(DATA_DIR,
        "part3_runB_kh_inject_n$(N)_$(_MED_TAG)$(_KH_MODE_TAG)_dx$(round(DX, sigdigits=4))_L$(round(L, sigdigits=4))_dist$(round(Int, INIT_DIST*1000))mm_fc$(round(Int, FC_HZ))_nt$(dom_inj.nt).h5")
    local slab_B_kh::Matrix{Float64}, t_B_kh::Float64
    if isfile(runb_kh_path)
        @info "[Part 3b N=$N] Run-B-with-KH-source cache hit" runb_kh_path
        slab_B_kh, t_B_kh = h5open(runb_kh_path, "r") do f
            (Float64.(read(f["slab_B"])), Float64(HDF5.attrs(f)["t_used"]))
        end
    else
        @info "[Part 3b N=$N] Run B FDTD with K-H-extrapolated source" runb_kh_path
        tx     = build_txs(dom_inj, sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm)
        txs_on_grid = transceiversToGrid(dom_inj, tx.txs)
        cpml   = Cpml(dom_inj;
                      npml  = Int(cfg["pml"]["n"]),
                      rcoef = Float64(cfg["pml"]["rcoef"]),
                      fc    = Float64(cfg["pml"]["fc"]))
        # Same sign convention as Part 2's direct injection (see note there
        # and the sign-convention block above C_for_N): no explicit minus
        # on C_f; the De Hoop `f = -p·n̂` sign is implicit in the outward-
        # normal convention of `inner_nrm` + the FDTD vn-channel update.
        inject_inner_sources!(txs_on_grid, tx.inner_rng;
            vn_inject = Cs.C_f .* p_inner_KH,
            p_inject  = Cs.C_q .* vn_inner_KH,
        )
        field = zeros(Float32, dom_inj.nx, dom_inj.ny, dom_inj.nz, 4)
        t0 = time()
        slabs_B, snap_times_B = run_with_slab_capture!(dom_inj, field, txs_on_grid, cpml)
        @printf("[Part 3b N=%d] FDTD wall: %.2f s\n", N, time() - t0)
        slab_B_kh, t_B_kh = slab_at_time(slabs_B, snap_times_B, T_TGT)
        h5open(runb_kh_path, "w") do f
            f["slab_B"] = Float32.(slab_B_kh)
            HDF5.attrs(f)["t_used"]        = t_B_kh
            HDF5.attrs(f)["C_formula"] = C_FORMULA
            HDF5.attrs(f)["N"]             = N
            HDF5.attrs(f)["source_type"]   = "kh_extrapolated"
        end
        @info "[Part 3b N=$N] Saved cache" runb_kh_path
    end

    # ---- Layer (b): cancellation residual ‖slab_A + slab_B_KH‖ inside
    # the tapered inner disk (addition; same sign convention as Part 2).
    # With closed-form C_f/C_q the K-H pipeline should produce the same
    # residual as direct-recorded injection (Part 2's `rms_residual` at
    # the same N) up to the K-H extrapolation error.
    sub_A    = inner_disk_subarray(rec.slab_A, dom_pw,  R_INNER)
    sub_B_kh = inner_disk_subarray(slab_B_kh,  dom_inj, R_INNER)
    inner_mask_sub  = inner_disk_weight(dom_pw, R_INNER; taper_frac = MASK_TAPER_FRAC)
    rms_A           = rms_in_mask(sub_A,    inner_mask_sub)
    rms_B_kh        = rms_in_mask(sub_B_kh, inner_mask_sub)
    rms_residual_kh = rms_in_mask(sub_A .+ sub_B_kh, inner_mask_sub)
    rms_residual_p2 = haskey(_PART2_BY_N, N) ? _PART2_BY_N[N].rms_residual : NaN
    Δrms            = rms_residual_kh - rms_residual_p2

    @printf("[Part 3b N=%d] rms_residual(K-H-inject)  = %.4g\n", N, rms_residual_kh)
    @printf("[Part 3b N=%d] rms_residual(recorded)    = %.4g    (from Part 2)\n", N, rms_residual_p2)
    @printf("[Part 3b N=%d] Δrms_residual             = %+.4g  (K-H − recorded)\n", N, Δrms)
    @printf("[Part 3b N=%d] suppression rms_A/rms_residual_kh = %.4g×\n", N, rms_A/max(rms_residual_kh, eps()))

    # ---- 3-panel slab plot (Part 2-style analog): slab_A, slab_B_KH, residual.
    let
        xs_pw  = collect(range(-dom_pw.xmax,  dom_pw.xmax,  length = dom_pw.nx))
        zs_pw  = collect(range(-dom_pw.zmax,  dom_pw.zmax,  length = dom_pw.nz))
        xs_inj = collect(range(-dom_inj.xmax, dom_inj.xmax, length = dom_inj.nx))
        zs_inj = collect(range(-dom_inj.zmax, dom_inj.zmax, length = dom_inj.nz))

        # Pad slab_B_kh (on dom_inj grid) into dom_pw shape for the
        # residual sum (slab_B_kh sign-tuned to ≈ −slab_A inside R_inner).
        slab_B_kh_padded = pad_inj_slab_to_pw(slab_B_kh, dom_inj, dom_pw)
        slab_pred_pw     = Float64.(rec.slab_A) .+ Float64.(slab_B_kh_padded)

        fig = Figure(size = (1700, 600))
        Label(fig[0, 1:3],
              "Part 3 N=$N — K-H-injected slabs   "*
              "(rms_residual(KH)=$(round(rms_residual_kh, sigdigits=4))   "*
              "Δ vs Part 2 = $(round(Δrms, sigdigits=4)))";
              fontsize = 14, font = :bold)
        function _slab!(col, xs, zs, slab, title_)
            ax = Axis(fig[1, col]; title = title_,
                      xlabel = "x [m]", ylabel = "z [m]",
                      aspect = DataAspect(), yreversed = true)
            heatmap!(ax, xs, zs, Float32.(slab);
                     colormap = :balance, colorrange = (-1.3, 1.3))
            add_sphere_overlays!(ax)
        end
        _slab!(1, xs_pw,  zs_pw,  rec.slab_A,    "slab_A — plane wave only (RMS=$(round(rms_A, sigdigits=4)))")
        _slab!(2, xs_inj, zs_inj, slab_B_kh,     "slab_B_KH — injection-only with K-H source (RMS=$(round(rms_B_kh, sigdigits=4)))")
        _slab!(3, xs_pw,  zs_pw,  slab_pred_pw,  "slab_A + slab_B_KH — cancellation residual (inner-disk RMS=$(round(rms_residual_kh, sigdigits=4)))")

        p_slab3 = joinpath(DIAG_DIR, "part3_slabs_n$(lpad(N, 4, '0'))_$(_MED_TAG).png")
        save(p_slab3, fig)
        @info "[Part 3 N=$N] Saved 3-panel slab plot" p_slab3
    end

    return (; N,
              p_err_global, vn_err_global,                  # pv_from_pv
              p_err_global_pin, vn_err_global_pin,          # pv_from_pin (1-way)
              p_err_median  = median(p_err_per_pt),
              p_err_max     = maximum(p_err_per_pt),
              vn_err_median = median(vn_err_per_pt),
              vn_err_max    = maximum(vn_err_per_pt),
              rms_A, rms_B_kh,
              rms_residual_kh, rms_residual_p2,
              Δrms_residual  = Δrms,
              suppression_kh = rms_A / max(rms_residual_kh, eps()))
end

const _PART3_RESULTS = NamedTuple[]
for N in NS_PART3
    r = run_part3_for_N(N)
    r === nothing || push!(_PART3_RESULTS, r)
end

# ---- Cross-N consolidated convergence: pin vs pv rel-RMS as a function of
# surface-point count N. Diagnoses the pin-pv gap structure:
#   • constant offset → scale bug in src/io.jl scaling chain;
#   • gap closing as N↑ → sphere-aliasing artifact in pin's recombination
#     of high spherical-harmonic modes;
#   • shared floor at large N → genuine DoF deficit (pin is one-way).
if length(_PART3_RESULTS) >= 2
    Ns_arr     = [r.N                  for r in _PART3_RESULTS]
    p_pv_arr   = [r.p_err_global       for r in _PART3_RESULTS]
    p_pin_arr  = [r.p_err_global_pin   for r in _PART3_RESULTS]
    vn_pv_arr  = [r.vn_err_global      for r in _PART3_RESULTS]
    vn_pin_arr = [r.vn_err_global_pin  for r in _PART3_RESULTS]

    fig = Figure(size = (1500, 600))
    Label(fig[0, 1:2],
          "Part 3 — GLOBAL rel-RMS convergence vs N: pv_from_pv vs pv_from_pin.   "*
          "Constant offset → scale bug; gap closing as N↑ → aliasing; "*
          "shared floor → DoF deficit.";
          fontsize = 12, font = :bold)
    function _n_panel!(col, y_pv, y_pin, channel)
        ax = Axis(fig[1, col];
                  title  = "$(channel)_inner GLOBAL rel-RMS",
                  xlabel = "N (outer/inner surface point count)",
                  ylabel = "rel-RMS",
                  yscale = log10,
                  xscale = log10)
        scatterlines!(ax, Ns_arr, max.(y_pv,  1e-6);
                      color = _TAB_RED,  linewidth = 2, markersize = 10,
                      label = "pv_from_pv  (2-way)")
        scatterlines!(ax, Ns_arr, max.(y_pin, 1e-6);
                      color = _TAB_BLUE, linewidth = 2, markersize = 10,
                      label = "pv_from_pin (1-way)")
        axislegend(ax, position = :rt)
    end
    _n_panel!(1, p_pv_arr,  p_pin_arr,  "p")
    _n_panel!(2, vn_pv_arr, vn_pin_arr, "vn")
    p_conv = joinpath(DIAG_DIR, "part3_pin_vs_pv_convergence_$(_MED_TAG).png")
    save(p_conv, fig)
    @info "[Part 3] Saved cross-N pin-vs-pv convergence plot" p_conv
elseif length(_PART3_RESULTS) == 1
    @info "[Part 3] Skipping cross-N convergence plot — only 1 N value in --ns-part3"
end

