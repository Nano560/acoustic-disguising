#!/usr/bin/env julia
# =============================================================================
# Acoustic disguising / cloaking hologram synthesis.
#
# Runs the same Kirchhoff-Helmholtz hologram FDTD as scripts/hologram/synthesize.jl,
# but with a twist: the outer-surface sources are driven by the Green's function
# of a TARGET scatterer B, while the FDTD interior contains the REAL scatterer
# A's staircase mask. The result reads, on the outer surface, as B's scattering
# signature even though A is the actual geometry inside. No optimization loop —
# pure superposition of the boundary GF drive with the interior mask.
#
# Disguising is hologram-only by construction (no `--scatterer-mode=real`);
# only `--gf-method=` is exposed.
#
# Reads:   <run_dir>/greens/{analytical,impulsive,mdd_extracted}_<target>.h5
# Writes:  <run_dir>/holograms/hologram_<real>_cloak_<target>_<gf_method>.h5
# Usage:   julia --project=. scripts/disguise/synthesize.jl <run_dir> \
#                     [--real=sphere|cube|cross|all] \
#                     [--target=none|sphere|cube|cross|all] \
#                     [--gf-method=analytical|impulsive|mdd]
#
# `real` selects the interior mask (the geometry actually sitting inside the
# domain). `target` selects which GF file to load for the outer-surface drive
# (the appearance the disguise should mimic). `--target=none` is the
# invisibility case: drive with the homogeneous-medium GF so the outer
# surface reads as undisturbed plane-wave propagation.
# =============================================================================

using AcousticDisguising
using CairoMakie
using Dates
using Printf

set_theme!(theme_dark())

# GF_METHODS is exported from AcousticDisguising (src/AcousticDisguising.jl).
# `:real` (bare FDTD) is intentionally NOT in the dict — disguising
# requires a GF for the target — so dispatch is keyed on `gf_method`.

const DEFAULT_GF_METHOD = "impulsive"

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
const RUN_DIR = parse_run_dir_arg(ARGS)
const CONFIG  = config_path(RUN_DIR)
cfg_root = apply_overrides(load_config(CONFIG), "disguise")

# `all` expands to ALL_SCATTERERS minus :none for `--real` (no geometry to
# disguise when the interior is homogeneous); kept whole for `--target`
# (invisibility is a primary use case).
const REAL_TARGETS = filter(!=(:none), collect(ALL_SCATTERERS))

real_str   = parse_flag(ARGS, "--real=";   default = "cross")
target_str = parse_flag(ARGS, "--target="; default = "sphere")
gf_method  = Symbol(parse_flag(ARGS, "--gf-method="; default = DEFAULT_GF_METHOD))
haskey(GF_METHODS, gf_method) ||
    error("--gf-method must be one of $(collect(keys(GF_METHODS))); got :$(gf_method)")

reals   = real_str   == "all" ? Tuple(REAL_TARGETS)    : (Symbol(real_str),)
targets = target_str == "all" ? Tuple(ALL_SCATTERERS)  : (Symbol(target_str),)

for r in reals
    r === :none &&
        error("--real=:none: nothing to disguise (homogeneous interior). " *
              "Use scripts/hologram/synthesize.jl --gf-method=$(gf_method) instead.")
end

const DIRS = ensure_output_dirs(RUN_DIR)

# -------------------------------------------------------------------------
# Per-(real, target) pipeline
# -------------------------------------------------------------------------
function run_for_pair(cfg, real_sym::Symbol, target_sym::Symbol)
    if real_sym === target_sym
        @info "Disguise: skip identity pair (use scripts/hologram/synthesize.jl)" real=real_sym target=target_sym
        return nothing
    end

    spec = GF_METHODS[gf_method]

    g = cfg["grid"]
    # Reuse [hologram].duration unless overridden by an optional [disguise].duration.
    duration = Float64(get(get(cfg, "disguise", Dict()),
                           "duration",
                           cfg["hologram"]["duration"]))
    dom = Domain(;
        tmax = duration,
        xmax = Float64(g["xmax"]),
        ymax = Float64(g["ymax"]),
        zmax = Float64(g["zmax"]),
        nx   = Int(g["n"]),  ny = Int(g["n"]),  nz = Int(g["n"]),
        cf   = Float64(g["cf"]),
    )
    @info "Disguise" real=real_sym target=target_sym gf_method cf=dom.cf nx=dom.nx nt=dom.nt

    # Outer-surface drive: GF of the TARGET. Interior physics: mask of the REAL.
    gf = load_gf(find_gf_file(DIRS.greens, spec.stem, target_sym),
                 dom, cfg; gf_method = gf_method)
    va = build_update_mask(dom, real_sym, cfg)

    snap_root = get(cfg, "disguise", Dict())
    snap_every = Int(get(snap_root, "snapshot_every",
                         get(cfg["hologram"], "snapshot_every", 0)))
    snapshot_every = snap_every > 0 ? snap_every : nothing
    SLICE_OFFSETS_M = collect(Float64,
        get(snap_root, "slice_offsets_m",
            get(cfg["hologram"], "slice_offsets_m", [-0.3, 0.0])))

    t0 = time()
    result = build_hologram(dom, cfg, gf;
                            scatterer        = real_sym,
                            implementation   = hologram_mode(spec.impl),
                            gf_of            = GFContent.heterogeneous,
                            va               = va,
                            snapshot_every   = snapshot_every,
                            slice_offsets_m  = SLICE_OFFSETS_M,
                            initial_distance = 0.65)    # was 0.62 (used historically)
    elapsed = time() - t0

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
        @info "Disguise slice snapshots" n=ns every=snap_every offsets_m=result.slice_offsets_m t_first_ms=snap_times[1]*1e3 t_last_ms=snap_times[end]*1e3
        (; xSlice_snaps   = x_arr,
           ySlice_snaps   = y_arr,
           zSlice_snaps   = z_arr,
           snapshot_times = snap_times,
           slice_offsets_m = Float32.(result.slice_offsets_m))
    end

    out_path = joinpath(DIRS.holograms,
                        "hologram_$(real_sym)_cloak_$(target_sym)_$(gf_method).h5")
    save_h5(out_path,
        merge((field_final = result.field_final, times = Float32.(result.times)),
              snap_payload);
        attrs = Dict(
            "created_at"     => string(now()),
            "real"           => String(real_sym),
            "target"         => String(target_sym),
            # `scatterer` mirrors `real` so downstream tools that key on
            # the standard hologram attr (plots, blender overlays) Just Work.
            "scatterer"      => String(real_sym),
            "scatterer_mode" => "hologram",
            "gf_method"      => String(gf_method),
            "implementation" => String(spec.impl),
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
    @printf("[disg] real=%-7s target=%-7s done │ %5.2g min elapsed │ %s\n",
            String(real_sym), String(target_sym), elapsed / 60, relpath(out_path))

    if get(cfg, "plotting", true) === true
        fig_path = joinpath(DIRS.figures, "hologram_cuts",
                            "$(real_sym)_cloak_$(target_sym)_$(gf_method).png")
        mkpath(dirname(fig_path))
        plot_hologram_cuts(@view(result.field_final[:, :, :, 1]), dom, fig_path;
            cfg = cfg, scatterer = real_sym)
    end
    return result
end

# -------------------------------------------------------------------------
# Dispatch matrix
# -------------------------------------------------------------------------
for r in reals, t in targets
    Base.invokelatest(run_for_pair, cfg_root, r, t)
end
