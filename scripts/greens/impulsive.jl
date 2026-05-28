#!/usr/bin/env julia
# =============================================================================
# Green's-function retrieval — method 1: impulsive point source.
#
# FDTD simulation with an impulsive point source at every outer-sphere point;
# the resulting (p, vn) traces on the inner sphere are the reference GFs.
#
# Writes:  <run_dir>/greens/impulsive_<scatterer>.h5
# Usage:   julia --project=. scripts/greens/impulsive.jl \
#                     <run_dir> [--scatterer=none|sphere|cube|cross|all]
# =============================================================================

using AcousticDisguising
using CairoMakie
using Dates
using HDF5
using LinearAlgebra: norm
using Statistics: quantile

set_theme!(theme_dark())

const SAVE_INTERVAL_SEC = 30 * 60   # iterative checkpoint cadence

const RUN_DIR = parse_run_dir_arg(ARGS)
const CONFIG  = config_path(RUN_DIR)
cfg = load_config(CONFIG)
scatterer_str = parse_flag(ARGS, "--scatterer="; default = get(cfg, "scatterer", "sphere"))

const DIRS = ensure_output_dirs(RUN_DIR)

# Graceful-stop sentinel: `touch $STOP_FILE` from another terminal to
# finish the current source, save, and exit cleanly.
const STOP_FILE = joinpath(RUN_DIR, "STOP")
clear_stop_file!(STOP_FILE)
@info "graceful stop: `touch $STOP_FILE` to finish current source and exit"

# -------------------------------------------------------------------------
# Per-scatterer pipeline: build domain, run FDTD with live plots, save H5.
# -------------------------------------------------------------------------
function run_for_scatterer(cfg, scatterer::Symbol)
    # Apply [greens.grid] / [greens.pml] on top of the shared sections so
    # impulsive_gfs / build_update_mask see the GF-stage values (e.g.
    # [greens.pml].n = 0 → no absorption inside Cpml).
    cfg = apply_overrides(cfg, "greens")
    g, s = cfg["grid"], cfg["surfaces"]

    dom = Domain(;
        tmax = Float64(cfg["greens"]["duration"]),
        xmax = Float64(g["xmax"]),
        ymax = Float64(g["ymax"]),
        zmax = Float64(g["zmax"]),
        nx   = Int(g["n"]),  ny = Int(g["n"]),  nz = Int(g["n"]),
        cf   = Float64(g["cf"]),
    )

    n_po = Int(s["nPoints_outer"])
    n_pi = Int(s["nPoints_inner"])
    nt_gf = Int(cfg["greens"]["nt_save"])
    expected_shape = (nt_gf, n_pi, n_po)

    out_path = joinpath(DIRS.greens, "impulsive_$(scatterer).h5")

    # Resume: if a previous run wrote some sources, load them and skip those
    # iSrc indices in the upcoming sweep. The on-disk file is in physical
    # units; we multiply by α here to bring the in-memory state back to the
    # raw FDTD-recorded convention the simulator expects.
    init, n_done_init = _load_existing_gfs(out_path, expected_shape, n_po, dom)

    @info "GFs (impulsive)" scatterer cf=dom.cf nx=dom.nx nt=dom.nt resume_from=n_done_init out_path=relpath(out_path)

    # Wall-time estimate. The impulsive sweep runs 2·n_outer passes (one each
    # for :p and :v source types), with the inner sphere as receivers only.
    # log_eta("Impulsive GFs: rough ETA", dom, cfg, scatterer,
    #         2 * (n_po - n_done_init); include_outer = false)

    base_attrs = Dict{String,Any}(
        "scatterer"     => String(scatterer),
        "backend"       => AcousticDisguising.BACKEND,
        "dom_dt"        => dom.dt,
        "dom_dx"        => dom.dx,
        "dom_tmax"      => dom.tmax,
        "julia_version" => string(VERSION),
        "config_file"   => abspath(CONFIG),
        "n"             => Int(g["n"]),
        "nPoints_outer" => n_po,
        "nPoints_inner" => n_pi,
    )

    live_path   = joinpath(DIRS.figures, "gfs_live", "$(scatterer).png")
    mkpath(dirname(live_path))
    last_save_t = Ref(time())
    plot_live   = get(cfg, "plotting", true) === true

    function on_source_complete(iSrc, state)
        plot_live && _plot_source(state, iSrc, scatterer, live_path)
        if time() - last_save_t[] > SAVE_INTERVAL_SEC
            _flush_gfs(out_path, state, iSrc, base_attrs, dom)
            last_save_t[] = time()
            println("[gf] checkpoint at src $(iSrc) → $(relpath(out_path))")
        end
    end

    t0 = time()
    result = impulsive_gfs(dom, cfg;
                           scatterer = scatterer,
                           on_source_complete = on_source_complete,
                           init = init,
                           stopfile = STOP_FILE)
    elapsed = time() - t0

    C_p, C_v = gf_unit_factors(dom, Float64(result.t[2] - result.t[1]))
    # If the loop was cut short by a stop file, count actually-populated
    # iSrc slots instead of claiming all n_po are done. Resume logic in
    # `_load_existing_gfs` does its own non-zero scan and is the source of
    # truth, but the attr should still match.
    n_done = count(iSrc -> any(!iszero, @view result.p_p[:, :, iSrc]),
                   1:size(result.p_p, 3))
    final_attrs = merge(base_attrs, Dict{String,Any}(
        "created_at"       => string(now()),
        "elapsed_sec"      => elapsed,
        "n_done"           => n_done,
        "units_convention" => "physical",
    ))
    save_h5(out_path, gf_to_physical(result, C_p, C_v); attrs = final_attrs)
    return result
end

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

# FDTD-recorded ⇄ physical-units round-trip helpers live in
# src/greens_io.jl (exported from AcousticDisguising):
#   gf_unit_factors(dom, dt_save) -> (C_p, C_v)
#   gf_to_physical(state, C_p, C_v)
#   gf_to_raw_fdtd(loaded, C_p, C_v)

"Return `(init, n_done)` from an existing GF file or `(nothing, 0)` if absent / unusable."
function _load_existing_gfs(out_path, expected_shape, n_po, dom)
    isfile(out_path) || return (nothing, 0)
    try
        loaded = load_h5(out_path)
        if size(loaded.p_p) != expected_shape
            @warn "Existing GF file shape mismatch — starting fresh" expected=expected_shape got=size(loaded.p_p) out_path
            return (nothing, 0)
        end
        # No back-compat for pre-physical-units files: error out so the user
        # deletes them rather than silently double-scaling.
        units = _read_units_convention(out_path)
        units == "physical" || error(
            "load_existing_gfs: $out_path has units_convention=$(repr(units)); " *
            "expected \"physical\". Delete the file and re-run — back-compat is " *
            "intentionally not supported.")
        C_p, C_v = gf_unit_factors(dom, Float64(loaded.t[2] - loaded.t[1]))
        raw_init = gf_to_raw_fdtd(loaded, C_p, C_v)
        n_done = count(1:n_po) do iSrc
            any(!iszero, @view(raw_init.p_p[:, :, iSrc])) ||
            any(!iszero, @view(raw_init.p_v[:, :, iSrc])) ||
            any(!iszero, @view(raw_init.v_p[:, :, iSrc])) ||
            any(!iszero, @view(raw_init.v_v[:, :, iSrc]))
        end
        return (raw_init, n_done)
    catch e
        @warn "Failed to load existing GFs — starting fresh" exception=e out_path
        return (nothing, 0)
    end
end

"Read `units_convention` attr from `out_path`, or `nothing` if missing."
function _read_units_convention(out_path::AbstractString)
    HDF5.h5open(out_path, "r") do f
        haskey(HDF5.attrs(f), "units_convention") ?
            String(HDF5.attrs(f)["units_convention"]) : nothing
    end
end

"Write a partial GF state to `out_path`, with `n_done` recorded as an attr."
function _flush_gfs(out_path, state, n_done, base_attrs, dom)
    C_p, C_v = gf_unit_factors(dom, Float64(state.t[2] - state.t[1]))
    attrs = merge(base_attrs, Dict{String,Any}(
        "created_at"       => string(now()),
        "n_done"           => n_done,
        "units_convention" => "physical",
    ))
    physical = gf_to_physical(state, C_p, C_v)
    save_h5(out_path, (;
        physical.p_p, physical.p_v, physical.v_p, physical.v_v,
        t = Float32.(collect(physical.t)),
        src_positions = Float32.(physical.src_positions),
        rec_positions = Float32.(physical.rec_positions),
    ); attrs = attrs)
end

"Live diagnostic plot — four-panel heatmap of the four GF kernels for `iSrc`."
function _plot_source(state, iSrc::Int, scatterer::Symbol, out_path::AbstractString)
    src_pt = state.src_positions[iSrc, :]
    n_rec  = size(state.rec_positions, 1)
    dists  = [norm(state.rec_positions[j, :] .- src_pt) for j in 1:n_rec]
    order  = sortperm(dists)
    arrival_ms = 1e3 .* dists[order] ./ C0

    panels = (("p_p", state.p_p), ("p_v", state.p_v),
              ("v_p", state.v_p), ("v_v", state.v_v))
    fig = Figure(size = (1400, 600))
    Label(fig[0, :], "Live GFs — source $(iSrc) / $(state.n_src) (scatterer=$(scatterer))";
          fontsize = 18)
    axs = Axis[]
    for (k, (name, arr)) in enumerate(panels)
        slab = arr[:, order, iSrc]
        vmax = max(quantile(abs.(vec(slab)), 0.98), eps(eltype(slab)))
        ax = Axis(fig[fld1(k, 2), mod1(k, 2)];
                  title = name, xlabel = "receiver (sorted by distance)",
                  ylabel = "time [ms]", yreversed = true)
        heatmap!(ax, 1:n_rec, 1e3 .* state.t, permutedims(slab);
                 colormap = :berlin, colorrange = (-vmax, vmax))
        lines!(ax, 1:n_rec, arrival_ms;
               color = :white, linestyle = :dash, linewidth = 1.5)
        push!(axs, ax)
    end
    linkaxes!(axs...)
    # Diagnostic plot — never let a save failure (missing dir, full disk,
    # display backend hiccup) kill the surrounding multi-hour FDTD sweep.
    try
        mkpath(dirname(out_path))
        save(out_path, fig)
        display(fig)
    catch e
        @warn "live plot skipped" iSrc out_path exception=e
    end
end

# -------------------------------------------------------------------------
# Dispatch: single scatterer, or "all" to sweep every shape sequentially.
# -------------------------------------------------------------------------
scatterers_to_run = scatterer_str == "all" ? ALL_SCATTERERS : (Symbol(scatterer_str),)
for s in scatterers_to_run
    Base.invokelatest(run_for_scatterer, cfg, s)
    if stop_requested(STOP_FILE)
        @info "[gf] stop file present — skipping remaining scatterers" stopfile=STOP_FILE
        break
    end
end
# Intentionally leave STOP in place when triggered so run_all.sh / the user
# sees the signal and halts the surrounding pipeline. The next stage that
# honors STOP clears it at startup (see top of this script).
