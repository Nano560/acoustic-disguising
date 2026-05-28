# ---- 1D sanity check: initial wavefield along propagation direction --------
# Plots p(x, y=0, z=0) at t=0 on dom_pw with outer/inner sphere + box-wall
# overlays. Verifies the Ricker fits inside dom_pw (xmax may have been changed
# by --dx snapping).
let
    field0 = Array(initialField(dom_pw, FC_HZ, INIT_DIST)) .* Float32(INIT_AMP)
    cs = AcousticDisguising.coords(dom_pw)
    iy0 = argmin(abs.(cs[:y][:p] .- 0.0))
    iz0 = argmin(abs.(cs[:z][:p] .- 0.0))
    xs  = collect(cs[:x][:p])
    line = field0[:, iy0, iz0, 1]
    fig0 = Figure(size = (1200, 420))
    ax = Axis(fig0[1, 1];
              title  = "Initial Ricker plane wave at t=0 — p(x, y=0, z=0)   "*
                       "dom_pw: nx=$(dom_pw.nx), xmax=$(round(dom_pw.xmax, digits=3)), nt=$(dom_pw.nt), tmax=$(round(dom_pw.tmax*1e3, digits=2)) ms,  fc=$(round(FC_HZ, digits=0)) Hz",
              xlabel = "x [m]", ylabel = "p (init field)")
    lines!(ax, xs, line; color = :black, linewidth = 1.5, label = "p(x, y=0, z=0)")
    vlines!(ax, [-INIT_DIST];               color = :gray,     linestyle = :dash, label = "x = -INIT_DIST = -$(INIT_DIST) m")
    vlines!(ax, [-R_OUTER, +R_OUTER];       color = "#1f77b4", linestyle = :dash, label = "outer sphere (±$(R_OUTER) m)")    # tab:blue
    vlines!(ax, [-R_INNER, +R_INNER];       color = "#d62728", linestyle = :dash, label = "inner sphere (±$(R_INNER) m)")    # tab:red
    vlines!(ax, [-dom_pw.xmax, +dom_pw.xmax]; color = :black, linestyle = :solid, linewidth = 1.5, label = "box walls (±$(round(dom_pw.xmax, digits=3)) m)")
    axislegend(ax, position = :rt, framevisible = false)
    mkpath(DIAG_DIR)
    p0 = joinpath(DIAG_DIR, "init_wavefield_1d_$(_MED_TAG)_nxpw$(dom_pw.nx)_nt$(dom_pw.nt).png")
    save(p0, fig0)
    @info "Saved initial-wavefield 1D plot" p0
    peak = maximum(abs, line); edge = max(abs(line[1]), abs(line[end]))
    if edge > 0.01 * peak
        @warn "Initial wavelet may be clipped by box wall" peak_p=peak edge_p=edge ratio=edge/peak
    else
        @info "Initial wavelet fits in domain" peak_p=peak edge_p=edge ratio=edge/peak
    end
end


# ============================================================================
# Part 3 (DEFERRED) — analytical-kernel + K-H helpers.
# Currently unused. Kept here so the diff for the K-H follow-up is small.
# ============================================================================

# Analytical kernel evaluators (eval_analytical_kernels, eval_one_kernel)
# and K-H convolution routines (direct_kh_single!, direct_kh_accumulate!,
# direct_kh!) are exported from AcousticDisguising (src/kh_kernels.jl).


# ---- Diagnostic plot: the tapered inner-disk mask used by `s_opt`.
# Heatmap on the inner-disk sub-array (the y=0 plane, cropped to |x|, |z| ≤
# R_inner) showing where weight = 1 (inner solid disk), where it cosine-tapers
# to 0 (annulus of width `taper_frac · R_inner`), and where weight = 0 (outside).
# This is the mask consumed by `rms_in_mask`, `inner_dot`, and the s_opt linear
# fit. Plus a 1D radial slice (w vs r) so the taper profile is unambiguous.
let
    w = inner_disk_weight(dom_pw, R_INNER; taper_frac = MASK_TAPER_FRAC)
    xs = collect(range(-dom_pw.xmax, dom_pw.xmax, length = dom_pw.nx))
    zs = collect(range(-dom_pw.zmax, dom_pw.zmax, length = dom_pw.nz))
    ix = findall(x -> abs(x) <= R_INNER, xs)
    iz = findall(z -> abs(z) <= R_INNER, zs)
    xs_sub = xs[ix]; zs_sub = zs[iz]
    fig = Figure(size = (1300, 550))
    Label(fig[0, 1:2],
          "Inner-disk weight for s_opt linear fit  "*
          "(taper_frac = $(MASK_TAPER_FRAC), R_inner = $(R_INNER) m, "*
          "taper width ≈ $(round(MASK_TAPER_FRAC*R_INNER*1000, digits=1)) mm)";
          fontsize = 13, font = :bold)
    ax1 = Axis(fig[1, 1]; title  = "weight w(x, z)  [0=excluded, 1=full]",
               xlabel = "x [m]", ylabel = "z [m]", aspect = DataAspect())
    heatmap!(ax1, xs_sub, zs_sub, Float32.(w);
             colormap = :viridis, colorrange = (0.0, 1.0))
    # Boundary + flat-region overlays.
    θc = range(0, 2π; length = 200)
    lines!(ax1, R_INNER .* cos.(θc),                R_INNER .* sin.(θc);
           color = :white, linewidth = 1.5, linestyle = :dash, label = "r = R_inner")
    lines!(ax1, (1 - MASK_TAPER_FRAC)*R_INNER .* cos.(θc),
                (1 - MASK_TAPER_FRAC)*R_INNER .* sin.(θc);
           color = :black, linewidth = 1.0, linestyle = :dot,
           label = "r = (1-taper_frac)·R_inner")
    axislegend(ax1, position = :rb, framevisible = false)
    # Radial profile (along z=0).
    iz0   = argmin(abs.(zs_sub))
    w_rad = w[:, iz0]
    ax2 = Axis(fig[1, 2]; title = "radial profile w(r)  (along z = 0)",
               xlabel = "x [m]", ylabel = "weight")
    lines!(ax2, xs_sub, w_rad; color = "#1f77b4", linewidth = 2)
    vlines!(ax2, [+R_INNER, -R_INNER]; color = :black, linestyle = :dash,
            label = "r = ±R_inner")
    vlines!(ax2, [+(1 - MASK_TAPER_FRAC)*R_INNER, -(1 - MASK_TAPER_FRAC)*R_INNER];
            color = :gray, linestyle = :dot, label = "r = ±(1-taper_frac)·R_inner")
    axislegend(ax2, position = :lb, framevisible = false)
    p_mask = joinpath(DIAG_DIR, "inner_disk_mask_taper$(MASK_TAPER_FRAC).png")
    save(p_mask, fig)
    @info "Saved inner-disk mask plot" p_mask
end

