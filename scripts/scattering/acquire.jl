#!/usr/bin/env julia
# =============================================================================
# Scattering acquisition — sample p(t) and v_n(t) on a far-field Fibonacci
# sphere for one (scatterer, scatterer-mode, gf-method) combination. Drives an
# FDTD forward solve in an enlarged, PML-bounded box so the scattered pulse
# passes the receiver sphere cleanly before any wall-reflection contamination.
#
# The script appends `nPoints_farfield` Fibonacci-sphere passive receivers
# to `build_hologram` via its existing `txs_extra` argument; the kernel
# already records p / v at every transceiver, so no kernel changes are
# needed. Only the far-field columns of `txs_on_grid.p.rec` and
# `.vn_rec` are persisted, plus receiver positions and the time axis.
#
# Postprocess (figures/preprocess/preprocess_scattering.py) subtracts the
# matching `none` (homogeneous) baseline, integrates ∫p_scat²(t) dt per
# receiver, and exports polar-tikz + 3D-surface plot data.
#
# Reads:   <run_dir>/greens/{analytical,impulsive,mdd_extracted}_<scatterer>.h5
# Writes:  <run_dir>/scattering/scattering_<scatterer>_<gf_label>.h5
#          (where <gf_label> is `real` when mode=real, else the gf-method name)
# Usage:   julia --project=. --threads=10 scripts/scattering/acquire.jl \
#                     <run_dir> \
#                     [--scatterer-mode=real|hologram] \
#                     [--gf-method=analytical|impulsive|mdd] \
#                     [--scatterer=none|sphere|cube|cross|all]
# =============================================================================

using AcousticDisguising
using CairoMakie
using Dates
using Printf

set_theme!(theme_dark())

# GF_METHODS is exported from AcousticDisguising (src/AcousticDisguising.jl).
# Defaults when the CLI flags are omitted. CLI still wins.
const DEFAULT_SCATTERER_MODE = "real"
const DEFAULT_GF_METHOD      = "analytical"

# Initial Ricker peak position (matches scripts/hologram/synthesize.jl). With
# xmax = 1.5 and 10-cell PML, the Ricker (~0.3 m wide at fc = 4 kHz) clears
# the absorbing layer comfortably.
const INITIAL_DISTANCE = 0.65    # was 0.62 (used historically)

# -------------------------------------------------------------------------
# PML / wall-reflection diagnostic — multi-snapshot xz-plane slices
# -------------------------------------------------------------------------
"""
Plot a grid of pressure-on-(y=0) snapshots at evenly-spaced timesteps.
Box outline + PML edge are overlaid so PML failure is visually obvious:
working PML → wave enters the green strip and disappears; failing PML →
wave reflects back into the interior at later times.

`y_slices` is the inner Vector returned by `build_hologram` for the
single offset = 0 entry: each element is a `(nx, nz)` matrix on the y=0
plane at the corresponding `snapshot_steps` timestep.
"""
function _plot_pml_slices(y_slices, snapshot_steps, dom, cfg, r_ff,
                          scatterer, gf_label, out_path)
    nsnap = length(y_slices)
    isempty(y_slices) && return out_path

    # Pick up to 6 evenly-spaced snapshots for the 2x3 grid.
    npick = min(6, nsnap)
    pick  = round.(Int, range(1, nsnap; length = npick))

    # Symmetric color scale across all selected snapshots, with mild
    # saturation so the late-time tail isn't washed out.
    vmax = saturated_vmax(reduce(vcat, vec.(y_slices[pick])); q = 0.995)

    xs = range(-dom.xmax, dom.xmax; length = dom.nx)
    zs = range(-dom.zmax, dom.zmax; length = dom.nz)
    pml_n  = Int(cfg["pml"]["n"])
    pml_dx = pml_n * dom.dx                     # PML thickness in metres

    fig = Figure(size = (1500, 900))
    Label(fig[0, :], "PML check — $(scatterer) / $(gf_label) — y=0 plane\n" *
          "(green strip = PML; wave should enter and vanish)";
          fontsize = 14)

    for (k, idx) in enumerate(pick)
        row, col = fld1(k, 3), mod1(k, 3)
        t_ms = 1e3 * snapshot_steps[idx] * dom.dt
        ax = Axis(fig[row, col];
                  title = @sprintf("t = %.2g ms (it=%d)", t_ms, snapshot_steps[idx]),
                  xlabel = "x [m]", ylabel = "z [m]", aspect = DataAspect())
        heatmap!(ax, xs, zs, y_slices[idx];
                 colormap = :berlin, colorrange = (-vmax, vmax))

        # Far-field receiver ring (intersection with y=0 plane).
        θ = range(0, 2π; length = 200)
        lines!(ax, r_ff .* cos.(θ), r_ff .* sin.(θ);
               color = :white, linewidth = 1.0, linestyle = :dot)
        # PML inner edge as four green strips along the box walls.
        for v in (dom.xmax - pml_dx, -(dom.xmax - pml_dx))
            vlines!(ax, [v]; color = (:green, 0.6), linewidth = 1.0)
        end
        for v in (dom.zmax - pml_dx, -(dom.zmax - pml_dx))
            hlines!(ax, [v]; color = (:green, 0.6), linewidth = 1.0)
        end
    end
    Colorbar(fig[1:2, 4]; colormap = :berlin, colorrange = (-vmax, vmax),
             label = "p")
    try
        mkpath(dirname(out_path))
        save(out_path, fig)
    catch e
        @warn "PML-check plot save failed" exception=e out_path
    end
    return out_path
end

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
const RUN_DIR = parse_run_dir_arg(ARGS)
const CONFIG  = config_path(RUN_DIR)
cfg_root = load_config(CONFIG)
const DIRS = ensure_output_dirs(RUN_DIR)
scatterer_str = parse_flag(ARGS, "--scatterer="; default = get(cfg_root, "scatterer", "cross"))
scatterer_mode = Symbol(parse_flag(ARGS, "--scatterer-mode="; default = DEFAULT_SCATTERER_MODE))
gf_method      = Symbol(parse_flag(ARGS, "--gf-method=";      default = DEFAULT_GF_METHOD))
scatterer_mode in (:real, :hologram) ||
    error("--scatterer-mode must be :real or :hologram; got :$(scatterer_mode)")
if scatterer_mode === :hologram
    haskey(GF_METHODS, gf_method) ||
        error("--gf-method must be one of $(collect(keys(GF_METHODS))); got :$(gf_method)")
end
if scatterer_mode === :real && any(startswith(a, "--gf-method=") for a in ARGS)
    @warn "--gf-method=$(gf_method) ignored because --scatterer-mode=real (bare FDTD doesn't load a GF)."
end

const GF_LABEL = scatterer_mode === :real ? "real" : String(gf_method)
const IMPL = hologram_mode(scatterer_mode === :real ? :real : GF_METHODS[gf_method].impl)

# -------------------------------------------------------------------------
# Per-scatterer pipeline
# -------------------------------------------------------------------------
function run_for_scatterer(cfg_root, scatterer::Symbol)

    # Apply [scattering.grid] / [scattering.pml] overrides on top of the
    # shared [grid] / [pml] before constructing Domain / Cpml.
    cfg = apply_overrides(cfg_root, "scattering")
    g, pml = cfg["grid"], cfg["pml"]
    sct = cfg["scattering"]

    dom = Domain(;
        tmax = Float64(sct["duration"]),
        xmax = Float64(g["xmax"]),
        ymax = Float64(g["ymax"]),
        zmax = Float64(g["zmax"]),
        nx   = Int(g["n"]),  ny = Int(g["n"]),  nz = Int(g["n"]),
        cf   = Float64(g["cf"]),
    )

    r_ff = Float64(sct["radius_farfield"])
    n_ff = Int(sct["nPoints_farfield"])
    @info "Scattering" scatterer scatterer_mode gf_method duration=dom.tmax cf=dom.cf nx=dom.nx nt=dom.nt r_ff n_ff pml_n=pml["n"]

    dirs = DIRS

    # Load GFs (only on the hologram path). Build the staircase scatterer
    # mask only on the bare-FDTD path; the GF-driven paths use va=nothing.
    gf = !(IMPL isa RealMode) ?
         load_gf(find_gf_file(dirs.greens, GF_METHODS[gf_method].stem, scatterer),
                 dom, cfg; gf_method = gf_method) : nothing
    va = (IMPL isa RealMode && scatterer !== :none) ?
         build_update_mask(dom, scatterer, cfg) : nothing

    # Far-field passive receivers — q=0 (no pressure-source contribution),
    # outward radial normal so v2vn! produces the radial outgoing velocity.
    ff_pts, ff_normals = fibonacci_sphere(n_ff, r_ff)
    ff_directions = hcat(zeros(n_ff), ff_normals)
    ff_tf = zeros(dom.nt)
    ff_txs = Transceiver[]
    for (p, d) in zip(eachrow(ff_pts), eachrow(ff_directions))
        push!(ff_txs, Transceiver(dom; point=collect(p), direction=collect(d), tf=ff_tf))
    end

    # Index range of the far-field receivers inside the packed `txs` array
    # (build_hologram concatenates Outer, Inner, txs_extra in that order).
    n_outer = Int(cfg["surfaces"]["nPoints_outer"])
    n_inner = Int(cfg["surfaces"]["nPoints_inner"])
    ff_range = (n_outer + n_inner + 1):(n_outer + n_inner + n_ff)

    # Capture ~6 y=0 plane snapshots evenly across the run for the PML-
    # check plot. snapshot_every adds negligible overhead (~3.5 MB per
    # snapshot at n=545) and the slices stay on the host, not in the H5.
    snap_every = max(1, dom.nt ÷ 7)

    t0 = time()
    result = build_hologram(dom, cfg, gf;
                            scatterer        = scatterer,
                            implementation   = IMPL,
                            gf_of            = GFContent.heterogeneous,
                            va               = va,
                            txs_extra        = ff_txs,
                            snapshot_every   = snap_every,
                            slice_offsets_m  = [0.0],
                            initial_distance = INITIAL_DISTANCE)
    elapsed = time() - t0

    # Pull the far-field traces back to host. result.txs_on_grid.p.rec is
    # (nt, n_total_txs) on the device.
    p_ff  = Array(result.txs_on_grid.p.rec)[:, ff_range]
    vn_ff = Array(result.txs_on_grid.vn_rec)[:, ff_range]

    out_path = joinpath(dirs.scattering, "scattering_$(scatterer)_$(GF_LABEL).h5")
    save_h5(out_path,
        (; p_ff           = Float32.(p_ff),
           vn_ff          = Float32.(vn_ff),
           pos_ff         = Float32.(ff_pts),
           normals_ff     = Float32.(ff_normals),
           times          = Float32.(result.times));
        attrs = Dict(
            "created_at"      => string(now()),
            "scatterer"       => String(scatterer),
            "scatterer_mode"  => String(scatterer_mode),
            "gf_method"       => scatterer_mode === :real ? "" : String(gf_method),
            "implementation"  => string(IMPL),
            "backend"         => AcousticDisguising.BACKEND,
            "dom_dt"          => dom.dt,
            "dom_tmax"        => dom.tmax,
            "dom_dx"          => dom.dx,
            "dom_xmax"        => dom.xmax,
            "dom_n"           => dom.nx,
            "radius_farfield" => r_ff,
            "nPoints_farfield" => n_ff,
            "pml_n"           => Int(pml["n"]),
            "pml_fc"          => Float64(pml["fc"]),
            "pml_rcoef"       => Float64(pml["rcoef"]),
            "c0"              => dom.c0,
            "fc_incident"     => 4000.0,
            "initial_distance" => INITIAL_DISTANCE,
            "elapsed_sec"     => elapsed,
            "julia_version"   => string(VERSION),
            "config_file"     => abspath(CONFIG),
        ))
    @printf("[scat] %-7s %-10s done │ %5.2g min elapsed │ %s\n",
            String(scatterer), GF_LABEL, elapsed / 60, relpath(out_path))

    # PML-check slice grid. Single-offset snapshot setup → result.ySlices[1]
    # is the y=0 plane snapshot vector.
    if get(cfg, "plotting", true) === true && !isempty(result.ySlices) &&
       !isempty(result.ySlices[1])
        fig_dir = DIRS.figures
        fig_path = joinpath(fig_dir, "scattering_pml_check",
                            "$(scatterer)_$(GF_LABEL).png")
        _plot_pml_slices(result.ySlices[1], result.snapshot_steps, dom, cfg,
                         r_ff, scatterer, GF_LABEL, fig_path)
        @info "PML-check plot" fig_path
    end
    return out_path
end

# -------------------------------------------------------------------------
# Dispatch: single scatterer, or "all" to sweep every shape sequentially.
# -------------------------------------------------------------------------
scatterers_to_run = scatterer_str == "all" ? ALL_SCATTERERS : (Symbol(scatterer_str),)
for s in scatterers_to_run
    Base.invokelatest(run_for_scatterer, cfg_root, s)
end
