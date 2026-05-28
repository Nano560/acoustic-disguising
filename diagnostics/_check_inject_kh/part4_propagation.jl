# ============================================================================
# Part 4 — "Wave on the other side" propagation test (opt-in via --part4).
#
# Tests whether the K-H source-equivalence theorem's "zero outside the source
# surface" guarantee holds in the discrete FDTD regime — and whether pv_from_pin
# (which has 8× worse p-extrapolation than pv_from_pv) leaks more spurious
# outward radiation that distorts the plane wave PAST the spheres.
#
# Setup: dom_prop is dom_pw with tmax extended to ≈ 2·T_TGT, so the plane wave
# can propagate from x = -INIT_DIST through the spheres and out to x = +INIT_DIST.
# Four FDTDs per N (all cached):
#   (a) PW alone (reference for the bare wave at T_TGT and 2·T_TGT)
#   (b) PW + direct (recorded) inner injection
#   (c) PW + K-H inner injection using pv_from_pv extrapolation
#   (d) PW + K-H inner injection using pv_from_pin extrapolation
# y=0 slab snapshots at T_TGT and 2·T_TGT for each; errors computed in three
# regions: inside inner disk (cancellation), annulus R_INNER<r<R_OUTER
# (K-H exterior cleanliness in the gap), and downstream past the sphere
# (x > R_OUTER, the "wave on the other side" metric).
# ============================================================================

# Extended-time clone of dom_pw — only constructed when --part4 is on, since
# nothing else uses it. Sharing dx with dom_pw / dom_inj is required so that
# inner-disk masks line up cell-for-cell.
const dom_prop = if RUN_PART4
    Domain(;
        tmax = 2.5 * T_TGT,
        xmax = dom_pw.xmax, ymax = dom_pw.ymax, zmax = dom_pw.zmax,
        nx   = dom_pw.nx,   ny   = dom_pw.ny,   nz   = dom_pw.nz,
        cf   = dom_pw.cf,
        c0   = C0, r0 = RHO,
    )
else
    nothing
end
if RUN_PART4
    @info "dom_prop (Part 4)" xmax=dom_prop.xmax ymax=dom_prop.ymax zmax=dom_prop.zmax nx=dom_prop.nx cells=dom_prop.nx*dom_prop.ny*dom_prop.nz tmax=dom_prop.tmax nt=dom_prop.nt T_TGT_2=2*T_TGT
    @assert dom_prop.dt == dom_pw.dt "dom_prop must share dt with dom_pw"
end

function run_part4_for_N(N::Int)
    println()
    printstyled("══════ Part 4 (N=$N) — propagation-through test ══════\n"; bold = true)
    sp = SPHERES_PER_N[N]
    Cs = C_for_N(N)
    T_TGT_2 = 2 * T_TGT
    σ_T_p4  = sqrt(log(2)) / (2π * F_3DB_HZ)

    # ROI for outside-disk errors: skip the PML buffer.
    PML_N    = Int(cfg["pml"]["n"])
    pml_buf  = PML_N * dom_prop.dx
    roi_xmax = dom_prop.xmax - pml_buf
    roi_zmax = dom_prop.zmax - pml_buf

    # Predicates used by all error reductions below.
    pred_inside_disk(x, z) = (x*x + z*z) ≤ R_INNER * R_INNER
    pred_outside_roi(x, z) = (x*x + z*z) > R_INNER * R_INNER &&
                             abs(x) ≤ roi_xmax && abs(z) ≤ roi_zmax
    pred_downstream(x, z)  = x > R_OUTER && abs(x) ≤ roi_xmax && abs(z) ≤ roi_zmax

    @info "[Part 4 N=$N] geometry" T_TGT T_TGT_2 R_INNER R_OUTER roi_xmax roi_zmax pml_buf

    # ---- (a) Long-time PW-only recording on dom_prop -----------------------
    rec_path = joinpath(DATA_DIR,
        "part4_record_n$(N)_$(_MED_TAG)_dx$(round(DX, sigdigits=4))_nx$(dom_prop.nx)_nt$(dom_prop.nt).h5")
    rec4 = if isfile(rec_path)
        @info "[Part 4 N=$N] (a) PW-only recording — cache hit" rec_path
        h5open(rec_path, "r") do f
            (; p_outer  = Float64.(read(f["p_outer"])),
               vn_outer = Float64.(read(f["vn_outer"])),
               p_inner  = Float64.(read(f["p_inner"])),
               vn_inner = Float64.(read(f["vn_inner"])),
               slab_t1  = Float64.(read(f["slab_t1"])),
               slab_t2  = Float64.(read(f["slab_t2"])),
               t1_used  = Float64(HDF5.attrs(f)["t1_used"]),
               t2_used  = Float64(HDF5.attrs(f)["t2_used"]))
        end
    else
        @info "[Part 4 N=$N] (a) PW-only recording — running long FDTD on dom_prop"
        tx = build_txs(dom_prop, sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm)
        txs_on_grid = transceiversToGrid(dom_prop, tx.txs)
        cpml = Cpml(dom_prop;
                    npml  = PML_N,
                    rcoef = Float64(cfg["pml"]["rcoef"]),
                    fc    = Float64(cfg["pml"]["fc"]))
        field = Array(initialField(dom_prop, FC_HZ, INIT_DIST)) .* Float32(INIT_AMP)
        t0 = time()
        slabs_pw, snap_times_pw = run_with_slab_capture!(dom_prop, field, txs_on_grid, cpml)
        @printf("[Part 4 N=%d] (a) FDTD wall: %.2f s\n", N, time() - t0)
        slab_t1, t1_used = slab_at_time(slabs_pw, snap_times_pw, T_TGT)
        slab_t2, t2_used = slab_at_time(slabs_pw, snap_times_pw, T_TGT_2)
        p_outer  = Float64.(Array( p_rec(txs_on_grid))[:, tx.outer_rng])
        vn_outer = Float64.(Array(vn_rec(txs_on_grid))[:, tx.outer_rng])
        p_inner  = Float64.(Array( p_rec(txs_on_grid))[:, tx.inner_rng])
        vn_inner = Float64.(Array(vn_rec(txs_on_grid))[:, tx.inner_rng])
        h5open(rec_path, "w") do f
            f["p_outer"]  = Float32.(p_outer);  f["vn_outer"] = Float32.(vn_outer)
            f["p_inner"]  = Float32.(p_inner);  f["vn_inner"] = Float32.(vn_inner)
            f["slab_t1"]  = Float32.(slab_t1);  f["slab_t2"]  = Float32.(slab_t2)
            HDF5.attrs(f)["t1_used"] = t1_used
            HDF5.attrs(f)["t2_used"] = t2_used
            HDF5.attrs(f)["N"]       = N
        end
        @info "[Part 4 N=$N] (a) Saved cache" rec_path
        (; p_outer, vn_outer, p_inner, vn_inner, slab_t1, slab_t2, t1_used, t2_used)
    end
    @printf("[Part 4 N=%d] snapshot times: t1 = %.4g s  (T_TGT %.4g)   t2 = %.4g s  (2·T_TGT %.4g)\n",
            N, rec4.t1_used, T_TGT, rec4.t2_used, T_TGT_2)

    # ---- K-H extrapolation on the LONG recordings (pv_from_pv & pv_from_pin).
    # Same algebra as run_part3_for_N — repeated here on the dom_prop time axis.
    nt_k     = dom_prop.nt
    T_GRID_K = collect(range(0.0, (nt_k - 1) * dom_prop.dt; length = nt_k))
    dS_out   = 4π * R_OUTER^2 / N
    ρ        = dom_prop.r0
    dt_KH    = dom_prop.dt
    z0_val   = dom_prop.z0
    inv_c    = 1.0 / dom_prop.c0
    p_in_outer  = Float32.((rec4.p_outer .- z0_val .* rec4.vn_outer) ./ 2)
    scale_pv_q  = -ρ * dS_out * dt_KH
    scale_pv_f  = -1 * dS_out * dt_KH
    scale_pin_q =  inv_c * dS_out * dt_KH
    scale_pin_f = -1 * dS_out * dt_KH

    p_KH_pv  = zeros(Float64, nt_k, N); vn_KH_pv  = zeros(Float64, nt_k, N)
    p_KH_pin = zeros(Float64, nt_k, N); vn_KH_pin = zeros(Float64, nt_k, N)
    kernel_jobs = [
        (:p_p, p_KH_pv,  rec4.vn_outer, scale_pv_q, p_KH_pin,  scale_pin_q),
        (:p_v, p_KH_pv,  rec4.p_outer,  scale_pv_f, p_KH_pin,  scale_pin_f),
        (:v_p, vn_KH_pv, rec4.vn_outer, scale_pv_q, vn_KH_pin, scale_pin_q),
        (:v_v, vn_KH_pv, rec4.p_outer,  scale_pv_f, vn_KH_pin, scale_pin_f),
    ]
    @info "[Part 4 N=$N] K-H extrapolation on long-time axis — eval+convolve per kernel" peak_GB_per_kernel=round(nt_k*N*N*4/1e9, digits=2)
    t0 = time()
    for (which, out_pv, in_pv, scale_pv, out_pin, scale_pin) in kernel_jobs
        tk = time()
        K = eval_one_kernel(which,
                            sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm,
                            T_GRID_K, σ_T_p4, dom_prop.c0, dom_prop.r0;
                            T = Float32)
        direct_kh_accumulate!(out_pv,  K, in_pv;      α = scale_pv)
        direct_kh_accumulate!(out_pin, K, p_in_outer; α = scale_pin)
        @printf("[Part 4 N=%d]   %s  wall: %.2f s\n", N, string(which), time() - tk)
        K = nothing; GC.gc()
    end
    @printf("[Part 4 N=%d] K-H total wall: %.2f s\n", N, time() - t0)

    # ---- (b), (c), (d): PW + injection FDTDs. Each is dom_prop FDTD with both
    # initial-field plane wave AND continuous inner-sphere injection active.
    function _run_injected(label, vn_inj, p_inj, tag)
        cache_path = joinpath(DATA_DIR,
            "part4_runB_$(tag)_n$(N)_$(_MED_TAG)_dx$(round(DX, sigdigits=4))_nx$(dom_prop.nx)_nt$(dom_prop.nt)$(_KH_MODE_TAG).h5")
        if isfile(cache_path)
            @info "[Part 4 N=$N $label] Run cache hit" cache_path
            return h5open(cache_path, "r") do f
                (Float64.(read(f["slab_t1"])), Float64.(read(f["slab_t2"])))
            end
        end
        @info "[Part 4 N=$N $label] Running FDTD (PW IC + inner injection)"
        tx = build_txs(dom_prop, sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm)
        txs_on_grid = transceiversToGrid(dom_prop, tx.txs)
        cpml = Cpml(dom_prop;
                    npml  = PML_N,
                    rcoef = Float64(cfg["pml"]["rcoef"]),
                    fc    = Float64(cfg["pml"]["fc"]))
        # Positive sign — same convention as Part 2 / Part 3b. The outward inner_nrm
        # + FDTD vn-channel sign together already absorb the De Hoop minus, so
        # +C_f·p_recorded radiates a wave that EQUALS −A inside the inner sphere
        # (cancellation), and the K-H source-equivalence theorem keeps the
        # outside ≈ 0. See sign-convention block above C_for_N for the audit.
        inject_inner_sources!(txs_on_grid, tx.inner_rng;
            vn_inject = Cs.C_f .* p_inj,
            p_inject  = Cs.C_q .* vn_inj,
        )
        field = Array(initialField(dom_prop, FC_HZ, INIT_DIST)) .* Float32(INIT_AMP)
        t0 = time()
        slabs, snap_times = run_with_slab_capture!(dom_prop, field, txs_on_grid, cpml)
        @printf("[Part 4 N=%d %s] FDTD wall: %.2f s\n", N, label, time() - t0)
        slab_t1, _ = slab_at_time(slabs, snap_times, T_TGT)
        slab_t2, _ = slab_at_time(slabs, snap_times, T_TGT_2)
        h5open(cache_path, "w") do f
            f["slab_t1"] = Float32.(slab_t1); f["slab_t2"] = Float32.(slab_t2)
        end
        @info "[Part 4 N=$N $label] Saved cache" cache_path
        return (Float64.(slab_t1), Float64.(slab_t2))
    end

    slab_dir_t1, slab_dir_t2 = _run_injected("direct", rec4.vn_inner, rec4.p_inner, "direct")
    slab_pv_t1,  slab_pv_t2  = _run_injected("pv",     vn_KH_pv,      p_KH_pv,      "pv")
    slab_pin_t1, slab_pin_t2 = _run_injected("pin",    vn_KH_pin,     p_KH_pin,     "pin")

    # ---- Errors per (case × time × region). Cancellation uses |slab| inside
    # the inner disk; cloaking uses |slab − slab_PW| outside the disk (within
    # the PML-free ROI) and downstream-only (x > R_OUTER).
    cases = [("direct", slab_dir_t1, slab_dir_t2),
             ("pv",     slab_pv_t1,  slab_pv_t2),
             ("pin",    slab_pin_t1, slab_pin_t2)]
    err_rows = NamedTuple[]
    zeros_slab = zeros(size(rec4.slab_t1))
    for (lbl, st1, st2) in cases, (tstr, st, ref) in (("T_TGT", st1, rec4.slab_t1), ("2T_TGT", st2, rec4.slab_t2))
        rms_in   = rms_region_diff(st, zeros_slab, dom_prop, pred_inside_disk)
        rms_out  = rms_region_diff(st, ref,        dom_prop, pred_outside_roi)
        rms_down = rms_region_diff(st, ref,        dom_prop, pred_downstream)
        push!(err_rows, (; case = lbl, t = tstr, rms_in, rms_out, rms_down))
    end

    println()
    printstyled("Part 4 summary  (N=$N)\n"; bold = true)
    @printf("%-6s %-7s | %-12s  %-12s  %-12s\n",
            "case", "time", "rms_inside", "rms_outside", "rms_downstream")
    println(repeat("─", 60))
    for r in err_rows
        @printf("%-6s %-7s | %-12.4g  %-12.4g  %-12.4g\n",
                r.case, r.t, r.rms_in, r.rms_out, r.rms_down)
    end
    println()

    # ---- 4×2 heatmap plot: rows = cases (PW, direct, pv, pin), cols = (T_TGT,
    # 2·T_TGT). Each panel shows the y=0 slab with sphere overlays.
    xs = collect(range(-dom_prop.xmax, dom_prop.xmax, length = dom_prop.nx))
    zs = collect(range(-dom_prop.zmax, dom_prop.zmax, length = dom_prop.nz))
    cmax = 1.3
    case_panels = [
        ("PW alone",                  rec4.slab_t1,  rec4.slab_t2,  nothing),
        ("PW + direct injection",     slab_dir_t1,   slab_dir_t2,   "direct"),
        ("PW + K-H pv_from_pv inj.",  slab_pv_t1,    slab_pv_t2,    "pv"),
        ("PW + K-H pv_from_pin inj.", slab_pin_t1,   slab_pin_t2,   "pin"),
    ]
    fig4 = Figure(size = (1800, 1500))
    Label(fig4[0, 1:2],
          "Part 4 — wave-on-the-other-side propagation test  (N=$N)$(_MODE_STR)";
          fontsize = 13, font = :bold)
    # Key err_rows by (case_tag, time_label). Two rows per case (one per time).
    err_by_case = Dict((r.case, r.t) => r for r in err_rows)
    # Each column maps to a time label used both for display and as the key.
    col_specs = [("T_TGT", :t1), ("2·T_TGT", :t2)]
    err_keys  = ["T_TGT", "2T_TGT"]     # match the keys pushed into err_rows above
    for (row, (label, st1, st2, tag)) in enumerate(case_panels)
        for (col, (tstr, _)) in enumerate(col_specs)
            slab_disp = col == 1 ? st1 : st2
            title_ = if tag === nothing
                "$label  @  $tstr"
            else
                r = err_by_case[(tag, err_keys[col])]
                @sprintf("%s  @  %s   in=%.3g  out=%.3g  down=%.3g",
                         label, tstr, r.rms_in, r.rms_out, r.rms_down)
            end
            ax = Axis(fig4[row, col]; title = title_,
                      xlabel = "x [m]", ylabel = "z [m]",
                      aspect = DataAspect(), yreversed = true)
            heatmap!(ax, xs, zs, Float32.(slab_disp);
                     colormap = :balance, colorrange = (-cmax, cmax))
            add_sphere_overlays!(ax)
            # PML-free ROI box (where the error metrics are computed).
            lines!(ax, [-roi_xmax, roi_xmax, roi_xmax, -roi_xmax, -roi_xmax],
                       [-roi_zmax, -roi_zmax, roi_zmax,  roi_zmax, -roi_zmax];
                       color = (:black, 0.4), linewidth = 0.8, linestyle = :dot)
        end
    end
    p4 = joinpath(DIAG_DIR, "part4_propagation_n$(lpad(N, 4, '0'))_$(_MED_TAG)$(_KH_MODE_TAG).png")
    save(p4, fig4)
    @info "[Part 4 N=$N] Saved propagation plot" p4

    return (; N, err_rows)
end


# ---- Part 4 — long-propagation cloaking test (opt-in via --part4). Runs last
# so Part 3's summary + CSV append are already on screen / on disk before the
# expensive long FDTDs start.
if RUN_PART4
    for N in NS_PART3
        try
            run_part4_for_N(N)
        catch err
            @error "[Part 4 N=$N] failed" exception = (err, catch_backtrace())
        end
    end
end
