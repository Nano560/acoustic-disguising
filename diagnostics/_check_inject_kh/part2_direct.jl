# ============================================================================
# Part 2 — Empirical scaling test (per-N loop: Run B + s_opt + 3-panel plot)
# ============================================================================

println()
printstyled("══════ Part 2 — Empirical scaling test ══════\n"; bold = true)
@info "Closed-form analytical α (no fitted constant)" dt=dom_pw.dt dx=dom_pw.dx rho=dom_pw.r0 z0=dom_pw.z0

function _runB_cache_path(N)
    return joinpath(DATA_DIR,
        "part2_runB_inject_only_n$(N)_$(_MED_TAG)$(_FLIP_VN_TAG)_dx$(round(DX, sigdigits=4))_L$(round(L, sigdigits=4))_dist$(round(Int, INIT_DIST*1000))mm_fc$(round(Int, FC_HZ))_nt$(dom_inj.nt).h5")
end

function run_part2_for_N(N::Int)
    rec = RECORDED_PER_N[N]
    sp  = SPHERES_PER_N[N]
    Cs  = C_for_N(N)

    @info "[Part 2] N=$N — α coefficients" N C_f=Cs.C_f C_q=Cs.C_q dS_in=Cs.dS_in

    # ---- Run B: injection only, zero initial field, on dom_inj.
    runb_path = _runB_cache_path(N)
    local slab_B::Matrix{Float64}, t_B::Float64
    if isfile(runb_path)
        @info "[Part 2 N=$N] Run B cache hit" runb_path
        slab_B, t_B = h5open(runb_path, "r") do f
            (Float64.(read(f["slab_B"])), Float64(HDF5.attrs(f)["t_used"]))
        end
    else
        @info "[Part 2 N=$N] Run B cache miss — injection-only FDTD on dom_inj" runb_path
        tx     = build_txs(dom_inj, sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm)
        txs_on_grid = transceiversToGrid(dom_inj, tx.txs)
        cpml   = Cpml(dom_inj;
                      npml  = Int(cfg["pml"]["n"]),
                      rcoef = Float64(cfg["pml"]["rcoef"]),
                      fc    = Float64(cfg["pml"]["fc"]))
        # Sign on the f-channel injection: `VN_INJECT_SIGN · C_f · p_recorded`.
        # Default is `+C_f` (per the comment block above `C_for_N` in
        # `injection_coefficients.jl`: the De Hoop `f = -p·n̂` convention
        # would imply `-C_f`, but empirically that breaks cancellation
        # — re-verified 2026-05-27 at N=300, rms_B 0.853 → 0.500 and
        # rms_residual 0.044 → 1.121, see the dated block in
        # `injection_coefficients.jl`). Pass `--flip-vn-sign` to flip to
        # `-C_f`; the two runs land in distinct `_flipvn`-tagged cache
        # files so they do not collide.
        @info "[Part 2 N=$N] f-channel injection sign" VN_INJECT_SIGN flip=FLIP_VN_SIGN
        inject_inner_sources!(txs_on_grid, tx.inner_rng;
            vn_inject = (VN_INJECT_SIGN * Cs.C_f) .* rec.p_inner,
            p_inject  = Cs.C_q .* rec.vn_inner,
        )
        field = zeros(Float32, dom_inj.nx, dom_inj.ny, dom_inj.nz, 4)
        t0 = time()
        slabs_B, snap_times_B = run_with_slab_capture!(dom_inj, field, txs_on_grid, cpml)
        @printf("[Part 2 N=%d] Run B FDTD wall: %.2f s\n", N, time() - t0)
        slab_B, t_B = slab_at_time(slabs_B, snap_times_B, T_TGT)
        h5open(runb_path, "w") do f
            f["slab_B"] = Float32.(slab_B)
            HDF5.attrs(f)["t_used"]         = t_B
            HDF5.attrs(f)["C_formula"]  = C_FORMULA
            HDF5.attrs(f)["C_f"]            = Cs.C_f
            HDF5.attrs(f)["C_q"]            = Cs.C_q
            HDF5.attrs(f)["N"]              = N
        end
        @info "[Part 2 N=$N] Saved Run B cache" runb_path
    end

    # ---- Cancellation residual: `slab_B` is the field radiated by the
    # injection sources, which are tuned (via the closed-form C_f / C_q
    # plus the outward-normal `inner_nrm` convention) to produce the
    # NEGATIVE of the plane wave inside r < R_inner. The cancellation
    # diagnostic is therefore ‖slab_A + slab_B‖ inside the tapered
    # inner disk. See the sign-convention block above `C_for_N` for
    # where each sign in the chain lives.
    sub_A = inner_disk_subarray(rec.slab_A, dom_pw,  R_INNER)
    sub_B = inner_disk_subarray(slab_B,     dom_inj, R_INNER)
    @assert size(sub_A) == size(sub_B) "inner-disk sub-arrays misaligned: $(size(sub_A)) vs $(size(sub_B))"
    inner_mask_sub = inner_disk_weight(dom_pw, R_INNER; taper_frac = MASK_TAPER_FRAC)   # circular disk + cosine taper near R_inner
    rms_A        = rms_in_mask(sub_A, inner_mask_sub)
    rms_B        = rms_in_mask(sub_B, inner_mask_sub)
    sub_pred     = sub_A .+ sub_B
    rms_residual = rms_in_mask(sub_pred, inner_mask_sub)

    @printf("\n[Part 2 N=%d] rms_A = %.4g    rms_B = %.4g    rms_residual = %.4g    suppression = %.4g×\n",
            N, rms_A, rms_B, rms_residual, rms_A / max(rms_residual, eps(Float64)))

    # ---- 3-panel slab plot for this N.
    let
        xs_pw  = collect(range(-dom_pw.xmax,  dom_pw.xmax,  length = dom_pw.nx))
        zs_pw  = collect(range(-dom_pw.zmax,  dom_pw.zmax,  length = dom_pw.nz))
        xs_inj = collect(range(-dom_inj.xmax, dom_inj.xmax, length = dom_inj.nx))
        zs_inj = collect(range(-dom_inj.zmax, dom_inj.zmax, length = dom_inj.nz))

        # Linearity-predicted residual on dom_pw grid: pad slab_B to dom_pw
        # shape, then add (slab_B is sign-tuned to ≈ −slab_A inside the
        # inner sphere; cancellation diagnostic is `slab_A + slab_B`).
        slab_B_padded = pad_inj_slab_to_pw(slab_B, dom_inj, dom_pw)
        slab_pred_pw  = Float64.(rec.slab_A) .+ Float64.(slab_B_padded)

        fig = Figure(size = (1700, 600))
        Label(fig[0, 1:3],
              "Part 2 N=$N   "*
              "(t=$(round(rec.t_A*1e3, sigdigits=4)) ms; T_TGT=$(round(T_TGT*1e3, sigdigits=4)) ms)";
              fontsize = 14, font = :bold)
        function _panel!(col, xs, zs, slab, title_)
            ax = Axis(fig[1, col]; title = title_,
                      xlabel = "x [m]", ylabel = "z [m]",
                      aspect = DataAspect(), yreversed = true)
            heatmap!(ax, xs, zs, Float32.(slab);
                     colormap = :balance, colorrange = (-1.3, 1.3))
            add_sphere_overlays!(ax)
        end
        _panel!(1, xs_pw,  zs_pw,  rec.slab_A,    "slab_A — plane wave only (RMS=$(round(rms_A, sigdigits=4)))")
        _panel!(2, xs_inj, zs_inj, slab_B,        "slab_B — injection only (RMS=$(round(rms_B, sigdigits=4)))")
        _panel!(3, xs_pw,  zs_pw,  slab_pred_pw,  "slab_A + slab_B — cancellation residual (inner-disk RMS=$(round(rms_residual, sigdigits=4)))")

        p2 = joinpath(DIAG_DIR, "part2_3panel_n$(lpad(N, 4, '0'))_$(_MED_TAG)_L$(round(L, sigdigits=4)).png")
        save(p2, fig)
        @info "[Part 2 N=$N] Saved 3-panel plot" p2
    end

    # ---- Cancellation-residual + near-field visualisation:
    # plot   slab_A_inside_inner + slab_B
    # where slab_A_inside_inner = slab_A masked to ZERO outside r = R_inner.
    # slab_B is NOT masked. This separates the two interpretations cleanly:
    #
    #   • inside r < R_inner   → slab_A + slab_B = cancellation residual
    #                            (≈ 0 in the deep interior, with a near-field
    #                            ring right at r = R_inner where the
    #                            injection sources sit). Addition because
    #                            slab_B is tuned to ≈ −slab_A by the
    #                            injection scaling + outward-normal
    #                            convention (see sign-convention block).
    #   • outside r > R_inner  → slab_B alone — the OUTWARD radiation
    #                            from the injection sources, isolated
    #                            from the plane wave. Lets us see how
    #                            strongly the injection radiates beyond
    #                            the inner sphere.
    #
    # Cropped to a square enclosing R_outer (= 0.3 m) for context.
    # Auto-scaled symmetric colour range from the 99th-percentile of |.|.
    let
        slab_B_padded = pad_inj_slab_to_pw(slab_B, dom_inj, dom_pw)
        xs_pw = collect(range(-dom_pw.xmax, dom_pw.xmax, length = dom_pw.nx))
        zs_pw = collect(range(-dom_pw.zmax, dom_pw.zmax, length = dom_pw.nz))
        # Mask slab_A to zero outside r = R_inner — keep only the part of
        # slab_A that overlaps the injection support.
        slab_A_inner = [xs_pw[i]^2 + zs_pw[k]^2 <= R_INNER^2 ?
                        Float64(rec.slab_A[i, k]) : 0.0
                        for i in eachindex(xs_pw), k in eachindex(zs_pw)]
        slab_combined = slab_A_inner .+ Float64.(slab_B_padded)
        ix    = findall(x -> abs(x) <= R_OUTER, xs_pw)
        iz    = findall(z -> abs(z) <= R_OUTER, zs_pw)
        xs_sub = xs_pw[ix]; zs_sub = zs_pw[iz]
        sub   = slab_combined[ix, iz]
        v     = let q = quantile(abs.(vec(Float64.(sub))), 0.99)
            q == 0 ? 1.0 : q
        end
        fig = Figure(size = (760, 760))
        ax  = Axis(fig[1, 1];
                   title  = "Part 2 N=$N — slab_A (masked to r ≤ R_inner) + slab_B\n"*
                            "inside R_inner: cancellation residual (inner-disk RMS=$(round(rms_residual, sigdigits=4)))   "*
                            "outside: injection-source radiation in isolation   "*
                            "(colour ±$(round(v, sigdigits=3)))",
                   xlabel = "x [m]", ylabel = "z [m]",
                   aspect = DataAspect(), yreversed = true)
        heatmap!(ax, xs_sub, zs_sub, Float32.(sub);
                 colormap = :balance, colorrange = (-v, v))
        θc = range(0, 2π; length = 200)
        lines!(ax, R_OUTER .* cos.(θc), R_OUTER .* sin.(θc);
               color = "#1f77b4", linewidth = 1.5, linestyle = :dash,
               label = "r = R_outer")
        lines!(ax, R_INNER .* cos.(θc), R_INNER .* sin.(θc);
               color = "#d62728", linewidth = 1.5, linestyle = :dash,
               label = "r = R_inner (sources here)")
        lines!(ax, (1 - MASK_TAPER_FRAC)*R_INNER .* cos.(θc),
                   (1 - MASK_TAPER_FRAC)*R_INNER .* sin.(θc);
               color = :black, linewidth = 1.0, linestyle = :dot,
               label = "r = (1-taper)·R_inner")
        axislegend(ax, position = :rt, framevisible = false)
        p2nf = joinpath(DIAG_DIR, "part2_residual_n$(lpad(N, 4, '0'))_$(_MED_TAG)_L$(round(L, sigdigits=4)).png")
        save(p2nf, fig)
        @info "[Part 2 N=$N] Saved cancellation-residual + injection-radiation plot" p2nf
    end

    return (; N, rms_A, rms_B, rms_residual,
              suppression = rms_A / max(rms_residual, eps(Float64)))
end

const _PART2_RESULTS = NamedTuple[]
for N in NS
    push!(_PART2_RESULTS, run_part2_for_N(N))
end

# ---- Summary table.
println()
printstyled("══════ Part 2 summary ══════\n"; bold = true)
@printf("%-6s %-10s %-10s %-14s %-12s\n", "N", "rms_A", "rms_B", "rms_residual", "suppression")
println(repeat("─", 60))
for r in _PART2_RESULTS
    @printf("%-6d %-10.4g %-10.4g %-14.4g %-12.4g\n",
            r.N, r.rms_A, r.rms_B, r.rms_residual, r.suppression)
end
println()
println("Interpretation:")
println("  rms_residual / rms_A → 0 as N grows  ⇒  closed-form α is correct and cancellation converges.")
println("  K-H quadrature error: the leftover RMS at any N is the discrete-source residual; ")
println("  drop further as N → ∞ (or as cps = √(4πR²/N)/dx → 0).")

# Lookup of Part 2 results by N (moved here from Part 3 so Part 3 can just read it;
# consumed by Part 3 to report Δrms = rms_residual_kh − rms_residual_p2).
const _PART2_BY_N = Dict(r.N => r for r in _PART2_RESULTS)
