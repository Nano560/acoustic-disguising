#!/usr/bin/env julia
# =============================================================================
# Green's-function retrieval — method 3 (part 1): reverberant FDTD data.
#
# Generates reverberating pressure + normal-velocity data on the inner /
# outer control surfaces; the Python MDD stage (`mdd_extract.py`) then
# deconvolves these into Green's functions. The box walls themselves act as
# the reverberant cavity (`pml_n = 0` keeps reflections in).
#
# Writes:  <run_dir>/reverb/reverb_<scatterer>.h5
#          <run_dir>/reverb/reverb_<scatterer>_state.h5  (if checkpoint)
#
# Note: n_ill is NOT in the folder name — it lives as an HDF5 attr (and as
# the iSrc-axis size). This lets a second run with a larger nPoints_ill
# extend the existing state file in place (ill positions use the r2
# sequence, so existing iSrc positions stay fixed when n_ill grows).
#
# Usage:   julia --project=. scripts/greens/reverb.jl \
#                     [config.toml] [--scatterer=none|sphere|cube|cross|all] \
#                     [--n-ill=N]
#
# The optional sidecar state file holds dom.dt-rate raw receiver traces and
# the final wavefield per iSrc, enabling resume-by-iSrc and tmax extension
# in place. It's disabled by default (storage cost can reach hundreds of
# GB for paper-scale runs) — set `[reverb].checkpoint_field = true` in the
# config to enable.
#
# Output HDF5 schema (see src/reverb.jl):
#   inner_p, inner_vnz, outer_p, outer_vnz : (nt_ill, n_rec, n_ill)  Float32
#   t               : (nt_ill,)             Float32
#   ill_positions   : (n_ill, 3)            Float32
#   inner_positions : (n_inner, 3)          Float32
#   outer_positions : (n_outer, 3)          Float32
# =============================================================================

using AcousticDisguising
using CairoMakie
using Dates
using HDF5
using LinearAlgebra: norm
using Printf
using Statistics: quantile

set_theme!(theme_dark())

const SAVE_INTERVAL_SEC = 30 * 60   # output-file resave cadence

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
const RUN_DIR = parse_run_dir_arg(ARGS)
const CONFIG  = config_path(RUN_DIR)
cfg = load_config(CONFIG)
scatterer_str = parse_flag(ARGS, "--scatterer="; default = get(cfg, "scatterer", "cross"))
n_ill_cli     = let s = parse_flag(ARGS, "--n-ill="); s === nothing ? nothing : parse(Int, s) end
plot_spc      = "--spatial-config" in ARGS    # opt-in; default off

const DIRS = ensure_output_dirs(RUN_DIR)

# Graceful-stop sentinel: `touch $STOP_FILE` from another terminal to
# finish the current source, save, and exit cleanly.
const STOP_FILE = joinpath(RUN_DIR, "STOP")
clear_stop_file!(STOP_FILE)
@info "graceful stop: `touch $STOP_FILE` to finish current source and exit"

# -------------------------------------------------------------------------
# Per-scatterer pipeline
# -------------------------------------------------------------------------
function run_for_scatterer(cfg, scatterer::Symbol; n_ill_cli = n_ill_cli)
    # Capture root-grid context before overrides rewrite cfg["grid"]; the
    # banner uses these to explain how the reverb-stage box was sized.
    root_n  = Int(cfg["grid"]["n"])
    root_dx = 2 * Float64(cfg["grid"]["xmax"]) / (root_n - 1)
    margin_cells = let mo = get(get(cfg, "reverb", Dict()), "grid", Dict())
        haskey(mo, "margin_cells") ? Int(mo["margin_cells"]) : nothing
    end

    cfg = _apply_reverb_overrides(cfg)
    g = cfg["grid"]

    dom = Domain(;
        tmax = Float64(cfg["reverb"]["duration"]),
        xmax = Float64(g["xmax"]),
        ymax = Float64(g["ymax"]),
        zmax = Float64(g["zmax"]),
        nx   = Int(g["n"]),  ny = Int(g["n"]),  nz = Int(g["n"]),
        cf   = Float64(g["cf"]),
    )

    rev_cfg = get(cfg, "reverb", Dict())
    # `[surfaces].nPoints_*` describes the production layout (consumed by MDD,
    # not by reverb). Reverb records at `[reverb].nPoints_*_record`, defaulting
    # to surfaces if absent. When the two differ we patch the in-memory cfg's
    # surfaces values so downstream Julia code (illumination_points, output
    # path tag, sanity-check plots) lays the recording receivers at the dense
    # count. The output H5 only stores what was actually recorded — the prod
    # count is a downstream MDD-side choice and is read from the cfg there.
    n_po_default = Int(cfg["surfaces"]["nPoints_outer"])
    n_pi_default = Int(cfg["surfaces"]["nPoints_inner"])
    n_po = Int(get(rev_cfg, "nPoints_outer_record", n_po_default))
    n_pi = Int(get(rev_cfg, "nPoints_inner_record", n_pi_default))
    if (n_po, n_pi) != (n_po_default, n_pi_default)
        cfg = nested_merge(cfg, Dict("surfaces" =>
              Dict("nPoints_outer" => n_po, "nPoints_inner" => n_pi)))
    end

    n_ill            = n_ill_cli === nothing ? Int(get(rev_cfg, "nPoints_ill", n_po)) : n_ill_cli
    checkpoint_field = Bool(get(rev_cfg, "checkpoint_field", false))

    # Output sample rate is derived, not set directly: the recording covers a
    # Ricker band of `bandwidth_factor · fc_source`, so its Nyquist rate is
    # `fs_out = 2 · bandwidth_factor · fc_source`. (The MDD stage caps its
    # inversion at the same `bandwidth_factor · fc_source` = fs_out/2.)
    bw_factor = get(rev_cfg, "bandwidth_factor", nothing)
    bw_factor === nothing && error(
        "[reverb].bandwidth_factor is required — it sets the recorded Ricker " *
        "band: fs_out = 2·bandwidth_factor·fc_source.")
    fs_out = 2.0 * Float64(bw_factor) * Float64(get(rev_cfg, "fc_source", 18_000.0))

    out_path   = joinpath(DIRS.reverb, "reverb_$(scatterer).h5")
    state_path = joinpath(DIRS.reverb, "reverb_$(scatterer)_state.h5")

    state_info, mode, n_saved = if checkpoint_field
        open_or_create_state(state_path; dom,
            n_outer = n_po, n_inner = n_pi, n_ill = n_ill, scatterer = scatterer)
    else
        (ReverbStateFile(state_path, dom.nt, n_po, n_pi, n_ill), :fresh, 0)
    end

    # Output-file resume (default, no flag). If `out_path` exists, validate
    # its attrs against the current config and use its already-computed iSrc
    # slots to pre-fill `build_reverb_data`'s output arrays. Validation
    # errors are fatal — the user is expected to delete the file and rerun
    # rather than have a config mismatch silently produce mixed-config data.
    # Only meaningful when checkpoint_field=false (else the sidecar covers
    # the same ground at FDTD rate and load_iSrc_saved takes precedence).
    prefill = checkpoint_field ?
              nothing :
              _load_resume_prefill(out_path; dom, n_po, n_pi, n_ill,
                                   scatterer, fs_out, rev_cfg, cfg)
    n_prefill_done = prefill === nothing ? 0 : _count_populated_iSrc(prefill.outer_p)
    n_done = checkpoint_field ? state_done_count(state_info) : n_prefill_done

    _print_run_summary(; scatterer, dom, cfg, rev_cfg,
                       n_po, n_pi, n_ill, fs_out,
                       checkpoint_field, mode, n_saved, n_done,
                       root_n, root_dx, margin_cells, out_path)

    # log_eta("Reverb: rough ETA", dom, cfg, scatterer,
    #         max(0, n_ill - n_done); include_outer = true)

    # Spatial-config sanity-check plot, before the FDTD work begins.
    # Opt-in via `--spatial-config`; not written by default because the
    # geometry rarely changes run-to-run.
    if plot_spc
        _spc = joinpath(DIRS.figures, "spatial_config", "$(scatterer).png")
        mkpath(dirname(_spc))
        plot_spatial_config(dom, cfg, n_ill, scatterer, _spc;
                            radius_ill = Float64(get(rev_cfg, "radius_ill", 0.6)))
    end

    # Output-file attrs (recording-side only). The MDD-side production
    # layout is read from the cfg by python/mdd/cli.py and the canonical
    # Fibonacci(N_prod) target sphere is regenerated there from the radius
    # in attrs — no reason to bake either into the H5.
    base_attrs = _build_reverb_attrs(cfg, dom, n_po, n_pi, n_ill,
                                     fs_out, scatterer, CONFIG)

    last_save_t = Ref(time())
    flush! = (state, iSrc; extra_attrs = Dict{String,Any}(), label = "checkpoint") -> begin
        _flush_reverb!(out_path, state, iSrc, base_attrs;
                       extra_attrs = extra_attrs, label = label)
        last_save_t[] = time()
    end

    # State-file callbacks (no-ops when checkpointing is disabled).
    load_iSrc = checkpoint_field && n_saved > 0 ?
                ((iSrc) -> read_iSrc_state(state_path, iSrc, n_saved)) :
                ((iSrc) -> nothing)
    save_iSrc = checkpoint_field ?
                ((iSrc, st) -> write_iSrc_state(state_path, iSrc, st)) :
                ((iSrc, st) -> nothing)

    # Live plot — captured once, on the first source we actually compute.
    live_path        = joinpath(DIRS.figures, "reverb_live", "$(scatterer).png")
    mkpath(dirname(live_path))
    plotting_enabled = get(cfg, "plotting", true) === true
    plotted          = Ref(false)
    snapshot_steps   = _snapshot_steps(dom, cfg)
    first_isrc       = _first_uncomputed_isrc(state_path, n_ill, dom.nt;
                                              checkpoint = checkpoint_field,
                                              n_saved, prefill)
    snapshot_iSrcs   = plotting_enabled ? Set([first_isrc]) : Set{Int}()

    function on_source_done(iSrc, state)
        if plotting_enabled && !plotted[]
            _plot_ill_source(state, iSrc, scatterer, cfg, rev_cfg, live_path)
            plotted[] = true
        end
        time() - last_save_t[] > SAVE_INTERVAL_SEC && flush!(state, iSrc)
        return nothing
    end

    t0 = time()
    result = build_reverb_data(dom, cfg;
        scatterer       = scatterer,
        n_ill           = n_ill_cli,
        n_saved         = n_saved,
        load_iSrc_saved = load_iSrc,
        save_iSrc_done  = save_iSrc,
        save_field      = checkpoint_field,
        on_source_complete = on_source_done,
        snapshot_steps  = snapshot_steps,
        snapshot_iSrc   = snapshot_iSrcs,
        prefill         = prefill,
        stopfile        = STOP_FILE,
    )
    elapsed = time() - t0

    # When the loop was cut short by a stop file, fewer than n_ill slots
    # are populated. Write the true count so resume logic / status banners
    # don't lie about completeness.
    n_done_final = _count_populated_iSrc(result.outer_p)
    flush!(result, n_done_final;
           extra_attrs = Dict{String,Any}("elapsed_sec" => elapsed),
           label = "final save")
    return result
end

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

"""
Apply `[reverb.grid]` / `[reverb.pml]` to `cfg`. Three convenience knobs in
`[reverb.grid]`:

* `ppw` — points per wavelength at the Ricker peak: derive reverb-stage
  `dx = c0 / (fc_source · ppw)` instead of inheriting root `dx`. Affects
  the reverb stage only; other stages keep using root `[grid]`.
* `margin_cells` — derive `xmax/ymax/zmax = radius_ill + margin_cells · dx`
  so the box just wraps the ill sphere with `margin_cells` of clearance.
  Uses the ppw-derived dx when `ppw` is set, otherwise root dx. Takes
  precedence over any explicit `xmax/ymax/zmax`.
* If `xmax` is overridden but not `n`, derive `n` from `dx` so the grid
  resolution stays the same (forced odd for symmetry about the origin).
"""
function _apply_reverb_overrides(cfg)
    rev               = get(cfg, "reverb", Dict{String,Any}())
    rev_grid_override = get(rev, "grid",   Dict{String,Any}())
    rev_pml_override  = get(rev, "pml",    Dict{String,Any}())
    root_g  = cfg["grid"]
    root_n  = Int(root_g["n"])
    root_dx = 2 * Float64(root_g["xmax"]) / (root_n - 1)

    # `ppw` overrides root dx with one tied to the Ricker peak frequency.
    # c0 is hardcoded to 1500.0 to match the Domain default (src/simulation.jl);
    # cfg doesn't expose c0.
    target_dx = if haskey(rev_grid_override, "ppw")
        ppw = Float64(rev_grid_override["ppw"])
        fc  = Float64(get(rev, "fc_source", 18_000.0))
        1500.0 / (fc * ppw)
    else
        root_dx
    end

    if haskey(rev_grid_override, "margin_cells")
        r_ill = Float64(get(rev, "radius_ill", 0.6))
        m     = Int(rev_grid_override["margin_cells"])
        xmax  = r_ill + m * target_dx
        rev_grid_override = merge(rev_grid_override,
            Dict{String,Any}("xmax" => xmax, "ymax" => xmax, "zmax" => xmax))
    end

    sub = Dict{String,Any}()
    isempty(rev_grid_override) || (sub["grid"] = rev_grid_override)
    isempty(rev_pml_override)  || (sub["pml"]  = rev_pml_override)
    cfg = nested_merge(cfg, sub)

    if haskey(rev_grid_override, "xmax") && !haskey(rev_grid_override, "n")
        n = round(Int, 2 * Float64(cfg["grid"]["xmax"]) / target_dx) + 1
        n = iseven(n) ? n + 1 : n
        cfg = nested_merge(cfg, Dict("grid" => Dict("n" => n)))
    end
    return cfg
end

"""
Compact, human-readable run banner. Replaces the four scattered `@info`
blocks (margin sizing, derived n, dense-recording mode, GF-run vitals) with
one structured summary so the operator can scan it at a glance.
"""
function _print_run_summary(; scatterer, dom, cfg, rev_cfg,
                            n_po, n_pi, n_ill, fs_out,
                            checkpoint_field, mode, n_saved, n_done,
                            root_n, root_dx, margin_cells, out_path)
    r_inner = Float64(cfg["surfaces"]["radius_inner"])
    r_outer = Float64(cfg["surfaces"]["radius_outer"])
    r_ill   = Float64(get(rev_cfg, "radius_ill", 0.6))
    fc      = Float64(get(rev_cfg, "fc_source",  18_000.0))

    box_origin = margin_cells === nothing ?
        @sprintf("explicit (root_dx=%.2g mm)", root_dx * 1e3) :
        @sprintf("r_ill=%.2g m + %d cells × root_dx=%.2g mm", r_ill, margin_cells, root_dx * 1e3)
    grid_origin = dom.nx == root_n ? "" : @sprintf("  (root n=%d)", root_n)

    receivers = @sprintf("%d outer / %d inner recorded (MDD-side production layout taken from cfg at extract time)",
                         n_po, n_pi)

    resume_str = if n_done == 0
        sidecar = checkpoint_field ? "sidecar=on" : "sidecar=off"
        "fresh ($sidecar)"
    elseif checkpoint_field
        @sprintf("%d/%d iSrc done from sidecar (mode=%s, n_saved=%d/%d)",
                 n_done, n_ill, mode, n_saved, dom.nt)
    else
        @sprintf("%d/%d iSrc done from prior output file (sidecar=off)", n_done, n_ill)
    end

    println()
    printstyled(@sprintf("─── reverb · scatterer=%s · cf=%.2g ───\n", scatterer, dom.cf);
                bold = true)
    @printf("  grid       n=%d%s,  dx=%.2g mm,  xmax=±%.3g m\n",
            dom.nx, grid_origin, dom.dx * 1e3, dom.xmax)
    @printf("             box from %s\n", box_origin)
    @printf("             ppw=%.1f at fc=%.1f kHz (c0=%.0f m/s)\n",
            C0 / (fc * dom.dx), fc / 1e3, C0)
    @printf("  surfaces   r_inner=%.2g m,  r_outer=%.2g m,  r_ill=%.2g m\n",
            r_inner, r_outer, r_ill)
    @printf("  receivers  %s\n", receivers)
    @printf("  sources    n_ill=%d  (Ricker fc=%.1g kHz)\n", n_ill, fc / 1e3)
    @printf("  time       tmax=%.2g ms,  dt=%.2g µs,  nt=%d,  fs_out=%.1g kHz\n",
            dom.tmax * 1e3, dom.dt * 1e6, dom.nt, fs_out / 1e3)
    @printf("  resume     %s\n", resume_str)
    @printf("  output     %s\n", relpath(out_path))
    println()
end

"Steps at which `build_reverb_data` should snapshot the pressure field."
function _snapshot_steps(dom, cfg)
    traversal = 2 * Float64(cfg["grid"]["xmax"]) / C0   # one full domain crossing
    targets = (traversal / 3, 2 * traversal / 3, traversal, dom.tmax)
    return unique(clamp.(round.(Int, targets ./ dom.dt), 1, dom.nt))
end

"""
Find the first `iSrc` that will actually be computed by `build_reverb_data`,
so the live-plot snapshot fires on the first fresh source. Sources behind
us in the loop (sidecar `iSrc_done == 1`, or prefill slot is populated) are
skipped, so we want the first iSrc past those.
"""
function _first_uncomputed_isrc(state_path, n_ill, dom_nt;
                                checkpoint, n_saved, prefill = nothing)
    if checkpoint && n_saved == dom_nt
        bitmap = h5open(state_path, "r") do f; read(f["iSrc_done"]); end
        idx = findfirst(==(UInt8(0)), bitmap)
        return idx === nothing ? n_ill + 1 : idx
    end
    if prefill !== nothing
        n_pre = size(prefill.outer_p, 3)
        for iSrc in 1:n_pre
            any(!iszero, @view prefill.outer_p[:, :, iSrc]) || return iSrc
        end
        return n_pre + 1   # all prefill slots populated; first fresh is past the file
    end
    return 1
end

"Count how many iSrc slots in `outer_p` (axis 3) have any nonzero data."
function _count_populated_iSrc(outer_p::AbstractArray)
    n = 0
    for iSrc in 1:size(outer_p, 3)
        any(!iszero, @view outer_p[:, :, iSrc]) && (n += 1)
    end
    return n
end

"""
Load already-computed iSrc data from `out_path` for resume. Returns
`nothing` if the file does not exist; errors out (instead of silently
overwriting) if the file's attrs don't match the current cfg.

Returns a NamedTuple `(; outer_p, outer_vnz, inner_p, inner_vnz)` of
`(nt_ill, n_rec, file_n_ill)` Float32 arrays — `file_n_ill` may be ≤
the requested `n_ill` (r2 sequence keeps existing iSrc positions stable
when n_ill grows).
"""
function _load_resume_prefill(out_path::AbstractString; dom, n_po, n_pi,
                              n_ill, scatterer, fs_out, rev_cfg, cfg)
    isfile(out_path) || return nothing

    h5open(out_path, "r") do f
        a = HDF5.attrs(f)

        required_attrs = ("scatterer", "n_ill", "nPoints_inner", "nPoints_outer",
                          "dom_dt", "dom_tmax", "fs_out", "fc_source",
                          "radius_ill", "radius_inner", "radius_outer")
        missing_a = String[k for k in required_attrs if !haskey(a, k)]
        isempty(missing_a) || error("""
            Existing reverb output file is missing attributes: $missing_a
            File: $out_path
            Either it was written by an older code version (pre-atomic save_h5,
            and a kill mid-flush truncated it) or by a script that doesn't
            tag attrs. Delete the file and rerun.""")

        for name in ("inner_p", "inner_vnz", "outer_p", "outer_vnz")
            haskey(f, name) || error("""
                Existing reverb output file lacks dataset '$name'.
                File: $out_path
                Likely a partial flush from a pre-atomic-write run. Delete
                the file and rerun.""")
        end

        check_eq = function (k, v_req)
            isapprox(Float64(a[k]), Float64(v_req); rtol = 1e-9) ||
                error("$k mismatch — file=$(a[k]) vs cfg=$v_req. " *
                      "Delete $out_path or restore the matching cfg.")
        end
        check_int = function (k, v_req)
            Int(a[k]) == Int(v_req) ||
                error("$k mismatch — file=$(a[k]) vs cfg=$v_req. " *
                      "Delete $out_path or restore the matching cfg.")
        end

        String(a["scatterer"]) == String(scatterer) ||
            error("scatterer mismatch — file=$(a["scatterer"]) vs requested=$scatterer. " *
                  "Delete $out_path or pick a different output dir.")

        # `dom_dt` is the FDTD timestep that produced the file. It only
        # matters when we're going to *append* more iSrc samples — those
        # new samples must run at the same FDTD step as the existing ones.
        # For a completed file (n_done == n_ill) there's no work to append
        # and the file becomes a closed dataset; the saved data lives on
        # the fs_out time grid and is independent of dom_dt. Allow a
        # different (or finer-resolution) historical run to satisfy the
        # check in that case.
        file_n_done = haskey(a, "n_done") ? Int(a["n_done"]) : 0
        is_complete = file_n_done >= Int(a["n_ill"])
        if !is_complete
            check_eq("dom_dt",   dom.dt)
        end
        check_eq("dom_tmax",     dom.tmax)
        check_int("nPoints_outer", n_po)
        check_int("nPoints_inner", n_pi)
        check_eq("fs_out",       fs_out)
        check_eq("fc_source",    Float64(get(rev_cfg, "fc_source", 18_000.0)))
        check_eq("radius_ill",   Float64(get(rev_cfg, "radius_ill", 0.6)))
        check_eq("radius_inner", Float64(cfg["surfaces"]["radius_inner"]))
        check_eq("radius_outer", Float64(cfg["surfaces"]["radius_outer"]))

        file_n_ill = Int(a["n_ill"])
        file_n_ill <= n_ill ||
            error("Existing file has n_ill=$file_n_ill > requested $n_ill. " *
                  "Refusing to shrink (would drop iSrc data). Delete $out_path " *
                  "or bump nPoints_ill ≥ $file_n_ill.")

        outer_p   = read(f["outer_p"])
        outer_vnz = read(f["outer_vnz"])
        inner_p   = read(f["inner_p"])
        inner_vnz = read(f["inner_vnz"])

        nt_ill_expected = length(range(0.0, dom.tmax; step = 1.0 / fs_out))
        size(outer_p, 1) == nt_ill_expected ||
            error("Existing file has nt_ill=$(size(outer_p, 1)) but cfg implies " *
                  "$nt_ill_expected. Delete $out_path.")

        return (; outer_p, outer_vnz, inner_p, inner_vnz)
    end
end

function _build_reverb_attrs(cfg, dom, n_po, n_pi, n_ill,
                             fs_out, scatterer, config_path)
    rev_cfg = get(cfg, "reverb", Dict())
    Dict{String,Any}(
        "scatterer"          => String(scatterer),
        "backend"            => AcousticDisguising.BACKEND,
        "dom_dt"             => dom.dt,
        "dom_dx"             => dom.dx,
        "dom_tmax"           => dom.tmax,
        "dom_c0"             => dom.c0,
        "dom_rho0"           => dom.r0,
        "n"                  => dom.nx,
        "radius_ill"         => Float64(get(rev_cfg, "radius_ill", 0.6)),
        "fc_source"          => Float64(get(rev_cfg, "fc_source", 18_000.0)),
        "fs_out"             => fs_out,
        "radius_inner"       => Float64(cfg["surfaces"]["radius_inner"]),
        "radius_outer"       => Float64(cfg["surfaces"]["radius_outer"]),
        "nPoints_inner"      => n_pi,         # actual recorded count
        "nPoints_outer"      => n_po,
        "n_ill"              => n_ill,
        "julia_version"      => string(VERSION),
        "config_file"        => abspath(config_path),
    )
end

function _flush_reverb!(out_path, state, iSrc, base_attrs;
                        extra_attrs = Dict{String,Any}(), label = "checkpoint")
    attrs = merge(base_attrs, Dict{String,Any}(
        "created_at" => string(now()),
        "n_done"     => iSrc,
    ), extra_attrs)
    t_save = time()
    save_h5(out_path, (;
        state.inner_p, state.inner_vnz, state.outer_p, state.outer_vnz,
        state.t, state.ill_positions, state.inner_positions, state.outer_positions,
    ); attrs = attrs)
    println("              ↳ $(label) at $(Dates.format(now(), "HH:MM:SS")) ",
            "→ $(relpath(out_path)) (in $(round(time() - t_save, digits=2))s)")
end

"Live diagnostic plot — z=0 pressure slices at four wave-travel time markers."
function _plot_ill_source(state, iSrc::Int, scatterer::Symbol, cfg, rev_cfg,
                          out_path::AbstractString)
    snaps = state.snapshots
    isempty(snaps) && return
    sorted_steps = sort(collect(keys(snaps)))

    d = state.dom
    xs = range(-d.xmax, d.xmax; length = d.nx)
    ys = range(-d.ymax, d.ymax; length = d.ny)
    k0 = (d.nz + 1) ÷ 2

    r_in   = Float64(cfg["surfaces"]["radius_inner"])
    r_out  = Float64(cfg["surfaces"]["radius_outer"])
    r_ill  = Float64(get(rev_cfg, "radius_ill", 0.6))
    ill_pts, _ = r2_sphere(state.n_ill, r_ill)
    active = ill_pts[iSrc, :]
    θs = range(0, 2π; length = 200)

    fig = Figure(size = (1700, 500))
    Label(fig[0, :], "Live reverb (xy at z=0) — ill source $(iSrc) / $(state.n_ill) (scatterer=$(scatterer))";
          fontsize = 18)
    for (col, it_step) in enumerate(sorted_steps)
        slab = @view snaps[it_step][:, :, k0, 1]
        vmax = max(quantile(abs.(vec(slab)), 0.99), eps(eltype(slab)))
        ax = Axis(fig[1, col];
            title = "t = $(round(it_step * d.dt * 1e3; sigdigits=2)) ms (step $(it_step)) · ill z=$(round(active[3]; sigdigits=2)) m",
            xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())
        heatmap!(ax, xs, ys, slab; colormap = :berlin, colorrange = (-vmax, vmax))
        for (R, colr) in ((r_in, :tomato), (r_out, :gold), (r_ill, :dodgerblue))
            lines!(ax, R .* cos.(θs), R .* sin.(θs);
                   color = (colr, 0.9), linewidth = 1.2, linestyle = :dash)
        end
        on_plane = abs(active[3]) < 0.5 * d.dz
        scatter!(ax, [active[1]], [active[2]];
                 color = on_plane ? :white : (:white, 0.4),
                 marker = :star5, markersize = on_plane ? 16 : 10,
                 strokewidth = 1, strokecolor = :black)
    end
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
# Guarded by `PROGRAM_FILE == @__FILE__` so `include`-ing the script (e.g.
# from tests that want the helper definitions) does not kick off a real run.
# -------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    scatterers_to_run = scatterer_str == "all" ? ALL_SCATTERERS : (Symbol(scatterer_str),)
    for s in scatterers_to_run
        Base.invokelatest(run_for_scatterer, cfg, s)
        if stop_requested(STOP_FILE)
            @info "[reverb] stop file present — skipping remaining scatterers" stopfile=STOP_FILE
            break
        end
    end
    # Intentionally leave STOP in place when triggered so run_all.sh /
    # run_reverb_mdd.sh and the user see the signal and halt the surrounding
    # pipeline. The next stage that honors STOP clears it at startup.
end
