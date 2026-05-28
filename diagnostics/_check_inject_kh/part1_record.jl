# ============================================================================
# Part 1 — Record reference fields on dom_pw (combined over all N in --ns)
# ============================================================================

println()
printstyled("══════ Part 1 — Record reference fields (plane wave only) ══════\n";
            bold = true)

# Per-N cache file path.
function _record_cache_path(N)
    return joinpath(DATA_DIR,
        "part1_record_n$(N)_$(_MED_TAG)_dx$(round(DX, sigdigits=4))_L$(round(L, sigdigits=4))_dist$(round(Int, INIT_DIST*1000))mm_fc$(round(Int, FC_HZ))_nxpw$(dom_pw.nx)_nt$(dom_pw.nt).h5")
end

# Pre-generate Fibonacci spheres for each N. Stored so Part 2 can reuse the
# same sphere positions to build dom_inj transceivers.
const SPHERES_PER_N = Dict{Int, NamedTuple}()
for N in NS
    inner_pts_N, inner_nrm_N = fibonacci_sphere(N, R_INNER)
    outer_pts_N, outer_nrm_N = fibonacci_sphere(N, R_OUTER)
    SPHERES_PER_N[N] = (; outer_pts = outer_pts_N, outer_nrm = outer_nrm_N,
                          inner_pts = inner_pts_N, inner_nrm = inner_nrm_N)
end

# Records-per-N storage.
const RECORDED_PER_N = Dict{Int, NamedTuple}()

# Cache check.
const _MISSING_NS = filter(N -> !isfile(_record_cache_path(N)), NS)

if isempty(_MISSING_NS)
    @info "Part 1 — all N caches hit; loading" ns=NS
    for N in NS
        path = _record_cache_path(N)
        recs = h5open(path, "r") do f
            (; p_outer  = Float64.(read(f["p_outer"])),
               vn_outer = Float64.(read(f["vn_outer"])),
               p_inner  = Float64.(read(f["p_inner"])),
               vn_inner = Float64.(read(f["vn_inner"])),
               slab_A   = Float64.(read(f["slab_A"])),
               t_A      = Float64(HDF5.attrs(f)["t_used"]))
        end
        RECORDED_PER_N[N] = recs
    end
else
    @info "Part 1 cache miss — running combined FDTD on dom_pw" missing_ns=_MISSING_NS total_txs=2*sum(NS) total_ns=length(NS)
    # Concatenate per-N transceivers into one Vector. `build_txs` already
    # tells us the per-N (outer_rng, inner_rng) into its own returned
    # Vector; we shift them by the running `_offset` to get the ranges
    # into the combined `_txs_all`. No more magic `2N` indexing.
    # Wrapped in `let` to avoid a Julia soft-scope warning on top-level
    # globals.
    txs_all, ranges = let
        _txs_all = Transceiver[]
        _ranges  = Dict{Int, NamedTuple}()   # N → (outer_rng, inner_rng) into _txs_all
        _offset  = 0
        for N in NS
            sp = SPHERES_PER_N[N]
            tx_N = build_txs(dom_pw, sp.outer_pts, sp.outer_nrm, sp.inner_pts, sp.inner_nrm)
            _ranges[N] = (; outer_rng = tx_N.outer_rng .+ _offset,
                            inner_rng = tx_N.inner_rng .+ _offset)
            append!(_txs_all, tx_N.txs)
            _offset += length(tx_N.txs)
        end
        (_txs_all, _ranges)
    end
    txs_on_grid = transceiversToGrid(dom_pw, txs_all)
    cpml   = Cpml(dom_pw;
                  npml  = Int(cfg["pml"]["n"]),
                  rcoef = Float64(cfg["pml"]["rcoef"]),
                  fc    = Float64(cfg["pml"]["fc"]))
    field  = Array(initialField(dom_pw, FC_HZ, INIT_DIST)) .* Float32(INIT_AMP)
    t0 = time()
    slabs_A, snap_times_A = run_with_slab_capture!(dom_pw, field, txs_on_grid, cpml)
    @printf("[Part 1] FDTD wall: %.2f s\n", time() - t0)
    slab_A, t_A = slab_at_time(slabs_A, snap_times_A, T_TGT)
    # Slice the combined recording into per-N pieces and cache each separately.
    for N in NS
        rng = ranges[N]
        p_outer_N  = Float64.(Array( p_rec(txs_on_grid))[:, rng.outer_rng])
        vn_outer_N = Float64.(Array(vn_rec(txs_on_grid))[:, rng.outer_rng])
        p_inner_N  = Float64.(Array( p_rec(txs_on_grid))[:, rng.inner_rng])
        vn_inner_N = Float64.(Array(vn_rec(txs_on_grid))[:, rng.inner_rng])
        RECORDED_PER_N[N] = (; p_outer = p_outer_N, vn_outer = vn_outer_N,
                               p_inner = p_inner_N, vn_inner = vn_inner_N,
                               slab_A, t_A)
        path = _record_cache_path(N)
        h5open(path, "w") do f
            f["p_outer"]  = Float32.(p_outer_N)
            f["vn_outer"] = Float32.(vn_outer_N)
            f["p_inner"]  = Float32.(p_inner_N)
            f["vn_inner"] = Float32.(vn_inner_N)
            f["slab_A"]   = Float32.(slab_A)
            HDF5.attrs(f)["t_used"]    = t_A
            HDF5.attrs(f)["T_TGT"]     = T_TGT
            HDF5.attrs(f)["INIT_DIST"] = INIT_DIST
            HDF5.attrs(f)["FC_HZ"]     = FC_HZ
            HDF5.attrs(f)["N"]         = N
            HDF5.attrs(f)["DX"]        = DX
            HDF5.attrs(f)["L"]         = L
        end
        @info "Saved Part 1 cache" N path
    end
end

# Report (every N shares slab_A and t_A since they came from one FDTD run).
let slab_A = RECORDED_PER_N[NS[1]].slab_A,  t_A = RECORDED_PER_N[NS[1]].t_A
    rms_A_global = rms_in_mask(inner_disk_subarray(slab_A, dom_pw, R_INNER),
                               inner_disk_weight(dom_pw, R_INNER; taper_frac = MASK_TAPER_FRAC))
    @printf("[Part 1] snapshot t = %.4g s   (T_TGT = %.4g s; |Δ| = %.3g s)\n",
            t_A, T_TGT, abs(t_A - T_TGT))
    @printf("[Part 1] inner-disk RMS slab_A (plane wave at T_TGT) = %.4g\n", rms_A_global)
    for N in NS
        r = RECORDED_PER_N[N]
        @printf("[Part 1] N=%-4d peaks  |p_outer|=%.4g  |vn_outer|=%.4g  |p_inner|=%.4g  |vn_inner|=%.4g\n",
                N, maximum(abs, r.p_outer), maximum(abs, r.vn_outer),
                maximum(abs, r.p_inner),  maximum(abs, r.vn_inner))
    end

    # Save Part 1 slab plot (slab_A is identical for every N — plot once).
    fig = Figure(size = (820, 740))
    ax = Axis(fig[1, 1];
              title  = "Part 1 — slab_A (plane wave, no injection) at t = $(round(t_A*1e3, sigdigits=4)) ms,  RMS_inner = $(round(rms_A_global, sigdigits=4))",
              xlabel = "x [m]", ylabel = "z [m]",
              aspect = DataAspect(), yreversed = true)
    xs_pw = collect(range(-dom_pw.xmax, dom_pw.xmax, length = dom_pw.nx))
    zs_pw = collect(range(-dom_pw.zmax, dom_pw.zmax, length = dom_pw.nz))
    heatmap!(ax, xs_pw, zs_pw, Float32.(slab_A); colormap = :balance, colorrange = (-1.3, 1.3))
    add_sphere_overlays!(ax)
    p1 = joinpath(DIAG_DIR, "part1_slab_A_$(_MED_TAG)_L$(round(L, sigdigits=4)).png")
    save(p1, fig)
    @info "Saved Part 1 slab plot" p1
end

