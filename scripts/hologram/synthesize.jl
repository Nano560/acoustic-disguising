#!/usr/bin/env julia
# =============================================================================
# Hologram synthesis from Green's functions.
#
# Two orthogonal CLI flags select how the scatterer's effect enters the FDTD:
#
#   --scatterer-mode=real      → bare FDTD with the staircase scatterer mask
#                                in the domain (reference / ground truth).
#   --scatterer-mode=hologram  → Kirchhoff-Helmholtz boundary integral on the
#                                inner / outer surfaces, driven by a Green's
#                                function. The GF is selected by --gf-method=.
#
# `--gf-method=analytical` reads the closed-form GFs from `greens/analytical.jl`
# via the same `load_gf_ref` path as `--gf-method=impulsive`. `load_gf_ref`
# auto-detects the `backend = "analytical"` H5 attr and applies the
# per-kernel α factors that bring physical-unit GFs onto the FDTD-recorded
# amplitude convention.
#
# Reads:   <run_dir>/greens/{analytical,impulsive,mdd_extracted}_<scatterer>.h5
# Writes:  <run_dir>/holograms/hologram_<scatterer>_<gf_label>.h5
#          (where <gf_label> is `real` when mode=real, else the gf-method name)
# Usage:   julia --project=. scripts/hologram/synthesize.jl \
#                     [config.toml] \
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
# Defaults when the CLI flags are omitted. Edit here to change the script's
# behaviour; CLI still wins.
const DEFAULT_SCATTERER_MODE = "hologram"
const DEFAULT_GF_METHOD      = "analytical"

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
const RUN_DIR = parse_run_dir_arg(ARGS)
const CONFIG  = config_path(RUN_DIR)
cfg = load_config(CONFIG)
scatterer_str = parse_flag(ARGS, "--scatterer="; default = get(cfg, "scatterer", "cross"))
scatterer_mode = Symbol(parse_flag(ARGS, "--scatterer-mode="; default = DEFAULT_SCATTERER_MODE))
gf_method      = Symbol(parse_flag(ARGS, "--gf-method=";      default = DEFAULT_GF_METHOD))
scatterer_mode in (:real, :hologram) ||
    error("--scatterer-mode must be :real or :hologram; got :$(scatterer_mode)")
if scatterer_mode === :hologram
    haskey(GF_METHODS, gf_method) ||
        error("--gf-method must be one of $(collect(keys(GF_METHODS))); got :$(gf_method)")
end
# `--gf-method=` is ignored when mode=real; warn if the user passed it.
if scatterer_mode === :real && any(startswith(a, "--gf-method=") for a in ARGS)
    @warn "--gf-method=$(gf_method) ignored because --scatterer-mode=real (bare FDTD doesn't load a GF)."
end

# Label that lands in filenames and HDF5 attrs identifying the run's
# scatterer realization. `real` for bare-FDTD; the gf-method name otherwise.
const GF_LABEL = scatterer_mode === :real ? "real" : String(gf_method)
# Internal `implementation` symbol consumed by `build_hologram` / `load_gf`.
const IMPL = hologram_mode(scatterer_mode === :real ? :real : GF_METHODS[gf_method].impl)

const DIRS = ensure_output_dirs(RUN_DIR)

# -------------------------------------------------------------------------
# Per-scatterer pipeline
# -------------------------------------------------------------------------
function run_for_scatterer(cfg, scatterer::Symbol)
    g = cfg["grid"]
    dom = Domain(;
        tmax = Float64(cfg["hologram"]["duration"]),
        xmax = Float64(g["xmax"]),
        ymax = Float64(g["ymax"]),
        zmax = Float64(g["zmax"]),
        nx   = Int(g["n"]),  ny = Int(g["n"]),  nz = Int(g["n"]),
        cf   = Float64(g["cf"]),
    )
    @info "Hologram" scatterer scatterer_mode gf_method cf=dom.cf nx=dom.nx nt=dom.nt

    dirs = DIRS

    # Load GFs (only on the hologram path). Build the staircase scatterer
    # mask only on the bare-FDTD path; the GF-driven paths use va=nothing.
    gf = !(IMPL isa RealMode) ?
         load_gf(find_gf_file(dirs.greens, GF_METHODS[gf_method].stem, scatterer),
                 dom, cfg; gf_method = gf_method) : nothing
    va = (IMPL isa RealMode && scatterer !== :none) ?
         build_update_mask(dom, scatterer, cfg) : nothing

    # Optional slice-snapshot capture: paper.toml [hologram].snapshot_every
    # = N → save the three orthogonal pressure cut planes every N FDTD steps.
    # Used for wavefield-slice movies. 0 (default) disables, no overhead.
    # The slice offset is hardcoded here (visualization-only, not in TOML).
    snap_every = Int(get(cfg["hologram"], "snapshot_every", 0))
    snapshot_every = snap_every > 0 ? snap_every : nothing
    # Configurable from [hologram].slice_offsets_m (vector). First entry is
    # the "primary" offset that the paper-figure renderer reads; the rest
    # are saved alongside for other figures / sanity checks. Default
    # `[-0.3, 0.0]` keeps the offset cuts (scatterer not occluded by the
    # +X+Y+Z Blender camera) plus the centreline cuts.
    SLICE_OFFSETS_M = collect(Float64,
        get(cfg["hologram"], "slice_offsets_m", [-0.3, 0.0]))

    t0 = time()
    result = build_hologram(dom, cfg, gf;
                            scatterer        = scatterer,
                            implementation   = IMPL,
                            gf_of            = GFContent.heterogeneous,
                            va               = va,
                            snapshot_every   = snapshot_every,
                            slice_offsets_m  = SLICE_OFFSETS_M,
                            # Ricker peak at world x = -initial_distance.
                            # With xmax = 1.1 and PML off, the Ricker
                            # (~0.3 m wide at fc = 4 kHz) fits cleanly
                            # between the left wall and the outer sphere
                            # left edge (−0.3 m). Was 0.62 historically
                            # (matches the old project's value).
                            initial_distance = 0.65)
    elapsed = time() - t0

    # Pack per-offset, per-snapshot slices into 4-D tensors with axes
    # (n_snap, n_offset, axis0, axis1). The offset axis is preserved so
    # consumers can pick the offset they need (paper figures use index 0,
    # the "primary" offset).
    snap_payload = if isempty(result.xSlices) || isempty(result.xSlices[1])
        NamedTuple()
    else
        n_off  = length(result.xSlices)
        ns     = length(result.xSlices[1])
        ny, nz = size(result.xSlices[1][1])
        nx, _  = size(result.ySlices[1][1])
        x_arr = Array{Float32,4}(undef, ns, n_off, ny, nz)
        y_arr = Array{Float32,4}(undef, ns, n_off, nx, nz)
        z_arr = Array{Float32,4}(undef, ns, n_off, nx, ny)
        for o in 1:n_off, k in 1:ns
            x_arr[k, o, :, :] = result.xSlices[o][k]
            y_arr[k, o, :, :] = result.ySlices[o][k]
            z_arr[k, o, :, :] = result.zSlices[o][k]
        end
        snap_times = Float32.(result.snapshot_steps .* dom.dt)
        @info "Hologram slice snapshots" n=ns every=snap_every offsets_m=result.slice_offsets_m t_first_ms=snap_times[1]*1e3 t_last_ms=snap_times[end]*1e3
        (; xSlice_snaps   = x_arr,
           ySlice_snaps   = y_arr,
           zSlice_snaps   = z_arr,
           snapshot_times = snap_times,
           slice_offsets_m = Float32.(result.slice_offsets_m))
    end

    out_path = joinpath(dirs.holograms, "hologram_$(scatterer)_$(GF_LABEL).h5")
    save_h5(out_path,
        merge((field_final = result.field_final, times = Float32.(result.times)),
              snap_payload);
        attrs = Dict(
            "created_at"     => string(now()),
            "scatterer"      => String(scatterer),
            "scatterer_mode" => String(scatterer_mode),
            "gf_method"      => scatterer_mode === :real ? "" : String(gf_method),
            "implementation" => string(IMPL),
            "backend"        => AcousticDisguising.BACKEND,
            "dom_dt"         => dom.dt,
            "dom_tmax"       => dom.tmax,
            "dom_dx"         => dom.dx,
            "dom_xmax"       => dom.xmax,
            "dom_ymax"       => dom.ymax,
            "dom_zmax"       => dom.zmax,
            "dom_n"          => dom.nx,
            "snapshot_every" => snap_every,
            "elapsed_sec"    => elapsed,
            "julia_version"  => string(VERSION),
            "config_file"    => abspath(CONFIG),
        ))
    @printf("[holo] %-7s done │ %5.2g min elapsed │ %s\n",
            String(scatterer), elapsed / 60, relpath(out_path))

    if get(cfg, "plotting", true) === true
        fig_path = joinpath(DIRS.figures, "hologram_cuts", "$(scatterer)_$(GF_LABEL).png")
        mkpath(dirname(fig_path))
        plot_hologram_cuts(@view(result.field_final[:, :, :, 1]), dom, fig_path;
            cfg = cfg, scatterer = scatterer)
    end
    return result
end

# -------------------------------------------------------------------------
# Dispatch: single scatterer, or "all" to sweep every shape sequentially.
# Sweep-level ETA mirrors `src/reverb.jl:247-391`: EMA-smoothed per-scatterer
# wall time (α=0.2 → half-life ≈ 3 scatterers) drives the ETA + finish
# wallclock columns.
# -------------------------------------------------------------------------
scatterers_to_run = scatterer_str == "all" ? ALL_SCATTERERS : (Symbol(scatterer_str),)
n_to_compute = length(scatterers_to_run)

@printf("\n[holo] sweep: %d scatterer(s) to synthesize (mode=%s, gf=%s)\n",
        n_to_compute, scatterer_mode, GF_LABEL)
@printf("        %9s  %9s  %9s  %9s  %16s\n",
        "scatterer", "elapsed", "min/sc", "ETA", "finish")
println("        ", "-"^60)

t_sweep_start = time()
n_computed    = 0
sec_per_sc    = 0.0

for (i, s) in enumerate(scatterers_to_run)
    global n_computed, sec_per_sc
    t_sc_start = time()
    Base.invokelatest(run_for_scatterer, cfg, s)

    n_computed += 1
    Δt = time() - t_sc_start
    sec_per_sc = n_computed == 1 ? Δt : 0.2 * Δt + 0.8 * sec_per_sc
    elapsed = time() - t_sweep_start
    n_left  = n_to_compute - n_computed
    eta_sec = n_left * sec_per_sc
    finish_at = now() + Second(round(Int, eta_sec))
    sc_label = @sprintf("%d/%d", i, n_to_compute)
    @printf("        %9s  %8.2fh  %8.2fm  %8.2fh  %16s\n",
            sc_label,
            elapsed / 3600, sec_per_sc / 60, eta_sec / 3600,
            Dates.format(finish_at, "mm-dd HH:MM"))
end
