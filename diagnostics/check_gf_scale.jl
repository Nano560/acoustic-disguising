#!/usr/bin/env julia
# =============================================================================
# Linear-fit the optimal GF amplitude scale factor that makes the K-H
# extrapolation cancel the plane wave inside the inner sphere. Two FDTD
# runs total (linearity does the rest):
#
#   Run 1 (baseline):   build_hologram with the GF set to zero. The
#                       resulting y=0 slab is the bare plane wave
#                       propagating through homogeneous medium.
#   Run 2 (unit scale): build_hologram with the GF at its on-disk
#                       amplitude. The y=0 slab is the plane wave PLUS
#                       the K-H contribution.
#
# The K-H contribution is `delta = slab_1 − slab_0`. By linearity the
# hologram at any scale s is `slab(s) = slab_0 + s·delta`. The s that
# minimises the inner-disk RMS of slab(s) is
#
#   s_opt = − ⟨slab_0, delta⟩_inner / ‖delta‖²_inner          (linear fit)
#
# Interpretation:
#  * `hom + analytical`: by Huygens-Fresnel, the K-H integral on a
#    closed surface gives the negative of the field inside. We expect
#    s_opt ≈ −1 if the analytical GF is properly scaled.
#  * Other scatterer / GF combinations give different optima; the
#    suppression ratio rms_zero/rms_opt quantifies how well the K-H
#    extrapolation reproduces the right cancellation field inside the
#    inner sphere.
#
# Outputs (<run_dir>/figures/gf_scale_fit/<scatterer>_<gf_method>/):
#   slab_0.png            plane-wave baseline (no GF)
#   slab_delta.png        K-H contribution alone (slab_1 − slab_0)
#   slab_opt.png          slab at the closed-form s_opt
#   snapshots.h5          y=0 slab tensors over FDTD time (both runs)
#                         + coordinates + s_opt — input for downstream movies
#   stdout                s_opt, RMS baseline/optimum, suppression ratio
#
# The diagnostic also reports `cps = √(4π·R_outer²/N_outer)/dx`, the mean
# K-H source-spacing in FDTD cells. Per the (N, cf, dx) sweep in
# `diagnostics/check_inject_kh.jl` this single number predicts the K-H quadrature
# error floor — any `|s_opt − target|` beyond this is attributable to the GF
# (calibration / phase / extraction noise), not to the test geometry.
#
# Usage:
#   julia --threads=10 --project=. diagnostics/check_gf_scale.jl <run_dir>
#       [--scatterer=none|sphere|cube|cross|all] \
#       [--gf-method=analytical|impulsive|mdd]
#       [--mask-taper-frac=0.3] [--tmax-ms=0.5 (default; 0=cfg.hologram.duration)]
#       [--write-scale]
#
# `--scatterer=all` loops over every scatterer (`ALL_SCATTERERS`). GF files
# that aren't on disk are skipped with a `[skip]` line so the batch keeps
# moving — useful while data trickles in (e.g. mdd_extracted_cross.h5
# regenerating).
# =============================================================================

using AcousticDisguising
using HDF5
using CairoMakie
using Statistics: quantile
using Printf

# ----- Argparse -------------------------------------------------------------
# parse_flag is exported from AcousticDisguising (src/io.jl).
const RUN_DIR         = parse_run_dir_arg(ARGS)
const SCATTERER_STR   = parse_flag(ARGS, "--scatterer=";  default = "none")
const GF_METHOD       = parse_flag(ARGS, "--gf-method=";  default = "analytical")
const MASK_TAPER_FRAC = parse(Float64, parse_flag(ARGS, "--mask-taper-frac="; default = "0.3"))
const TMAX_OVERRIDE   = parse(Float64, parse_flag(ARGS, "--tmax-ms="; default = "0.5")) * 1e-3
const WRITE_SCALE     = "--write-scale" in ARGS

GF_METHOD in ("analytical", "impulsive", "mdd") ||
    error("--gf-method must be analytical|impulsive|mdd; got $(GF_METHOD)")

# Expand `--scatterer=all` to ALL_SCATTERERS in canonical order
# (none → sphere → cube → cross — increasing complexity).
const SCATTERERS = SCATTERER_STR == "all" ?
    String.(collect(ALL_SCATTERERS)) :
    [SCATTERER_STR]

# ----- Setup ----------------------------------------------------------------
const C0     = 1500.0
const INIT_D = 0.65          # Ricker peak initial offset (build_hologram default); was 0.62 (used historically)
const T_TGT  = INIT_D / C0   # time at which Ricker peak passes through origin

cfg = load_config(config_path(RUN_DIR))
const DIRS    = ensure_output_dirs(RUN_DIR)

# Default tmax (0.5 ms) captures the Ricker first-arrival at origin
# (T_TGT ≈ 0.433 ms for init_distance=0.65, c0=1500) with margin and no
# wasteful tail. Override with `--tmax-ms=0` to use cfg.hologram.duration,
# or any positive value for a custom window.
tmax_eff = TMAX_OVERRIDE > 0 ? TMAX_OVERRIDE :
                                Float64(cfg["hologram"]["duration"])
dom = Domain(;
    tmax = tmax_eff,
    xmax = Float64(cfg["grid"]["xmax"]),
    ymax = Float64(cfg["grid"]["ymax"]),
    zmax = Float64(cfg["grid"]["zmax"]),
    nx   = Int(cfg["grid"]["n"]),
    ny   = Int(cfg["grid"]["n"]),
    nz   = Int(cfg["grid"]["n"]),
    cf   = Float64(cfg["grid"]["cf"]),
)
tmax_eff < T_TGT && error("tmax_eff=$tmax_eff < T_TGT=$T_TGT — FDTD won't reach the comparison time")
@info "Domain" n=dom.nx nt=dom.nt dt=dom.dt tmax=dom.tmax

# Snapshot every N FDTD steps — fairly dense so the t_target index lands
# within ~0.5·snapshot_every·dt of the Ricker peak AND the saved tensors
# are usable for movies.
const SNAPSHOT_EVERY = max(1, dom.nt ÷ 200)

# Extrapolation implementation (depends only on GF_METHOD, not scatterer).
const IMPL = hologram_mode(GF_METHOD == "mdd" ? :mdd : :ref)

# GF filename pattern per scatterer.
function _gf_path_for(scatterer::AbstractString)
    stem = GF_METHOD == "mdd"       ? "mdd_extracted_$(scatterer)" :
           GF_METHOD == "impulsive" ? "impulsive_$(scatterer)" :
                                       "analytical_$(scatterer)"
    return joinpath(DIRS.greens, "$(stem).h5")
end

# Inner-disk tapered weight in the y=0 plane. Built on the cropped
# sub-array (cells with |x|, |z| ≤ R_INNER_FULL); slab consumers pair it
# with `inner_disk_subarray(slab, dom, R_INNER_FULL)`. `inner_disk_weight`
# is imported from AcousticDisguising (src/metrics.jl).
const R_INNER_FULL = Float64(cfg["surfaces"]["radius_inner"])
const R_FLAT       = (1 - MASK_TAPER_FRAC) * R_INNER_FULL
const XS = collect(range(-dom.xmax, dom.xmax, length=dom.nx))
const ZS = collect(range(-dom.zmax, dom.zmax, length=dom.nz))
const INNER_WEIGHT = inner_disk_weight(dom, R_INNER_FULL; taper_frac = MASK_TAPER_FRAC)
@info "Inner-disk tapered weight" mask_taper_frac=MASK_TAPER_FRAC r_flat=R_FLAT r_inner=R_INNER_FULL nonzero_pixels=count(>(0), INNER_WEIGHT) effective_area_m2=sum(INNER_WEIGHT)*dom.dx^2

# Quadrature-error budget. K-H surface sources sit on the outer sphere
# (src/greens.jl:10); their mean inter-source spacing in FDTD cells is
#     cps = √(4π · R_outer² / N_outer) / dx
# Empirical (N, cf, dx) sweep in diagnostics/check_inject_kh.jl:267-294 collapsed
# the K-H quadrature error onto cps alone. Rule of thumb derived from that
# sweep (analytical α → s_opt = 1 in the cps → 0 limit):
#   cps ≲ 5  → |s_opt − target| ≲ 2 %  (well-resolved)
#   cps ~ 10 → ~4 %
#   cps ~ 15 → ~10 %
#   cps ≳ 20 → >20 % (under-sampled — increase N_outer or coarsen dx)
const N_OUTER = Int(cfg["surfaces"]["nPoints_outer"])
const R_OUTER = Float64(cfg["surfaces"]["radius_outer"])
const CPS     = sqrt(4π * R_OUTER^2 / N_OUTER) / dom.dx
function _cps_floor_pct(cps)
    cps <= 5.0  && return 2.0
    cps <= 10.0 && return 4.0
    cps <= 15.0 && return 10.0
    cps <= 20.0 && return 20.0
    return 40.0
end
@info "K-H quadrature resolution" n_outer=N_OUTER r_outer=R_OUTER dx=dom.dx cps=CPS expected_floor_pct=_cps_floor_pct(CPS)

# ----- Helpers --------------------------------------------------------------
# rms_in_mask, inner_dot, inner_disk_subarray are imported from
# AcousticDisguising (src/metrics.jl). `_inner` crops a slab to the inner-disk
# sub-array shape that pairs with INNER_WEIGHT.
_inner(slab) = inner_disk_subarray(slab, dom, R_INNER_FULL)

function load_scaled_gf(scale::Float64, gf_path::AbstractString)
    """Load the GF and multiply every GF-tensor by `scale`. `gf_arrays`
    returns the underlying arrays by reference, so the in-place broadcast
    propagates into the returned `GFMap`."""
    gf_map = if IMPL isa MDDMode
        load_gf_mdd(gf_path, dom, cfg; gf_of=GFContent.heterogeneous)
    else
        load_gf_ref(gf_path, dom, cfg; gf_of=GFContent.heterogeneous)
    end
    for arr in gf_arrays(gf_map)
        arr .*= Float32(scale)
    end
    return gf_map
end

function plot_slab(slab::AbstractMatrix, label::AbstractString, rms::Float64,
                   t_actual::Float64, out_path::AbstractString)
    vmax = max(quantile(abs.(vec(slab)), 0.99), 1e-30)
    fig = Figure(size=(760, 700))
    ax = Axis(fig[1, 1];
              title="$(label)   RMS_inner=$(round(rms; sigdigits=4))   "
                  * "t=$(round(t_actual * 1e3; sigdigits=2)) ms",
              xlabel="x [m]", ylabel="z [m]",
              aspect=DataAspect(),
              yreversed=true)
    hm = heatmap!(ax, XS, ZS, slab; colormap=:balance,
                  colorrange=(-vmax, vmax))
    θ = range(0, 2π; length=200)
    lines!(ax, R_INNER_FULL .* cos.(θ), R_INNER_FULL .* sin.(θ);
           color=:white, linestyle=:dash, linewidth=1)
    lines!(ax, R_FLAT .* cos.(θ), R_FLAT .* sin.(θ);
           color=:gold,  linestyle=:dash, linewidth=1.5)
    Colorbar(fig[1, 2], hm; label="p [Pa]")
    save(out_path, fig)
end

function run_and_extract(scale::Float64, scatterer::AbstractString,
                         gf_path::AbstractString)
    """Run build_hologram with the GF scaled to `scale`. Return the
    full y=0 slice tensor (n_snap, nx, nz), the snapshot times, the
    slab at the snapshot closest to `T_TGT`, and the actual time of
    that snapshot."""
    gf_map = load_scaled_gf(scale, gf_path)
    result = build_hologram(dom, cfg, gf_map;
                            scatterer        = Symbol(scatterer),
                            implementation   = IMPL,
                            va               = nothing,
                            initial_distance = INIT_D,
                            snapshot_every   = SNAPSHOT_EVERY)
    snap_times = Float64.(collect(result.snapshot_steps .* dom.dt))
    i_t   = argmin(abs.(snap_times .- T_TGT))
    i_off = argmin(abs.(result.slice_offsets_m))
    n_snap = length(result.ySlices[i_off])
    nx, nz = size(result.ySlices[i_off][1])
    yslab_t = Array{Float32, 3}(undef, n_snap, nx, nz)
    for k in 1:n_snap
        yslab_t[k, :, :] = result.ySlices[i_off][k]
    end
    return yslab_t, snap_times, yslab_t[i_t, :, :], Float64(snap_times[i_t])
end

# ----- Baseline cache attrs (cfg signature) --------------------------------
# The scale=0 run is just bare-FDTD plane-wave propagation through the
# homogeneous medium. It does NOT depend on the GF — only on cfg (grid,
# tmax, snapshot rate) and on the scatterer mask (which is `nothing` for
# every case here because `va = nothing` in run_and_extract). So we cache
# it per scatterer × cfg signature. Delete the cache file to force a redo.
function _expected_attrs()
    Dict("dom_dt"        => dom.dt,
         "dom_tmax"      => dom.tmax,
         "dom_nx"        => dom.nx,
         "snapshot_every"=> SNAPSHOT_EVERY,
         "init_distance" => INIT_D)
end

function _check_baseline_attrs(f, baseline_path)
    expected = _expected_attrs()
    for (k, v) in expected
        haskey(HDF5.attrs(f), k) || error("baseline cache missing attr `$k`; delete $(baseline_path) and rerun")
        got = HDF5.attrs(f)[k]
        if !(got ≈ v)
            error("baseline cache attr `$k` = $got, expected $v; cfg changed — delete $(baseline_path) and rerun")
        end
    end
end

# ----- Per-scatterer calibration -------------------------------------------
function calibrate_one(scatterer::AbstractString)
    gf_path = _gf_path_for(scatterer)
    if !isfile(gf_path)
        @warn "[skip] GF file missing — calibration skipped" scatterer gf_path
        return
    end
    fig_dir = joinpath(DIRS.figures, "gf_scale_fit", "$(scatterer)_$(GF_METHOD)")
    mkpath(fig_dir)
    baseline_path = joinpath(DIRS.figures, "gf_scale_fit",
                             "baseline_$(scatterer).h5")

    @info "=== Calibrating ===" scatterer gf_method=GF_METHOD gf_path

    # FDTD 1/2 — baseline (cached per scatterer × cfg signature)
    if isfile(baseline_path)
        @info "=== FDTD 1/2: cache hit, loading baseline ===" baseline_path
        yslab_0_t, snap_times, slab_0, t_actual = h5open(baseline_path, "r") do f
            _check_baseline_attrs(f, baseline_path)
            ys = Array{Float32, 3}(read(f["yslab_0_t"]))
            ts = Array{Float64,1}(read(f["snapshot_times"]))
            i_t = argmin(abs.(ts .- T_TGT))
            ys, ts, ys[i_t, :, :], ts[i_t]
        end
    else
        @info "=== FDTD 1/2: GF scale = 0 (plane-wave baseline) ==="
        yslab_0_t, snap_times, slab_0, t_actual =
            run_and_extract(0.0, scatterer, gf_path)
        mkpath(dirname(baseline_path))
        h5open(baseline_path, "w") do f
            write(f, "yslab_0_t", yslab_0_t)
            write(f, "snapshot_times", Float32.(snap_times))
            for (k, v) in _expected_attrs()
                HDF5.attrs(f)[k] = v
            end
        end
        @info "Baseline cached" baseline_path
    end

    @info "=== FDTD 2/2: GF scale = 1 ==="
    yslab_1_t, _, slab_1, _ = run_and_extract(1.0, scatterer, gf_path)

    delta_slab = slab_1 .- slab_0          # K-H at single snapshot

    # ----- Closed-form optimal scale ----------------------------------------
    sub_0    = _inner(slab_0);      sub_d = _inner(delta_slab)
    dot_0d    = inner_dot(sub_0, sub_d, INNER_WEIGHT)
    dot_dd    = inner_dot(sub_d, sub_d, INNER_WEIGHT)
    s_opt     = -dot_0d / dot_dd
    slab_opt  = slab_0 .+ Float32(s_opt) .* delta_slab
    rms_opt   = rms_in_mask(_inner(slab_opt),   INNER_WEIGHT)
    rms_zero  = rms_in_mask(sub_0,              INNER_WEIGHT)
    rms_delta = rms_in_mask(sub_d,              INNER_WEIGHT)

    # ----- Reference plots --------------------------------------------------
    plot_slab(slab_0,     "slab_0 (s=0, plane-wave only)",
              rms_zero,  t_actual, joinpath(fig_dir, "slab_0.png"))
    plot_slab(delta_slab, "delta (slab_1 − slab_0, K-H at s=1)",
              rms_delta, t_actual, joinpath(fig_dir, "slab_delta.png"))
    plot_slab(slab_opt,   "slab_opt (s_opt=$(round(s_opt; sigdigits=4)))",
              rms_opt,   t_actual, joinpath(fig_dir, "slab_opt.png"))

    # ----- Snapshot save (for downstream movie making) ----------------------
    snapshot_h5 = joinpath(fig_dir, "snapshots.h5")
    h5open(snapshot_h5, "w") do f
        write(f, "yslab_0_t", yslab_0_t)             # plane wave only
        write(f, "yslab_1_t", yslab_1_t)             # plane wave + K-H at s=1
        write(f, "snapshot_times", Float32.(snap_times))
        write(f, "xs", Float32.(XS))
        write(f, "zs", Float32.(ZS))
        attrs(f)["scatterer"]      = scatterer
        attrs(f)["gf_method"]      = GF_METHOD
        attrs(f)["s_opt"]          = s_opt
        attrs(f)["radius_inner"]    = R_INNER_FULL
        attrs(f)["r_flat"]          = R_FLAT
        attrs(f)["mask_taper_frac"] = MASK_TAPER_FRAC
        attrs(f)["cps"]             = CPS
        attrs(f)["n_outer"]         = N_OUTER
        attrs(f)["r_outer"]         = R_OUTER
        attrs(f)["dom_dt"]          = dom.dt
        attrs(f)["snapshot_every"]  = SNAPSHOT_EVERY
    end
    @info "Snapshot tensors saved" snapshot_h5

    # ----- Opt-in: write s_opt back to the GF's empirical_scale attr --------
    # `load_gf_mdd` (and load_gf_ref) read this on every load and multiply
    # the tensor by it. Multiplicative update so repeated calibration
    # converges: new_scale = old_scale × s_opt.
    if WRITE_SCALE
        if IMPL isa MDDMode
            h5open(gf_path, "r+") do f
                prev = haskey(HDF5.attrs(f), "empirical_scale") ?
                       Float64(HDF5.attrs(f)["empirical_scale"]) : 1.0
                new_scale = prev * s_opt
                HDF5.attrs(f)["empirical_scale"] = new_scale
                @info "Wrote empirical_scale" path=relpath(gf_path) prev s_opt new_scale
            end
        else
            @info "--write-scale ignored: only mdd_extracted_*.h5 carries an empirical_scale attr" gf_method=GF_METHOD
        end
    end

    # ----- stdout summary ---------------------------------------------------
    println()
    @printf "[%s / %s]\n" scatterer GF_METHOD
    @printf "closed-form s_opt = %.4g\n" s_opt
    @printf "RMS_inner @ s=0   = %.4g   (plane-wave only)\n"     rms_zero
    @printf "RMS_inner @ s_opt = %.4g\n"                          rms_opt
    @printf "suppression       = %.4g×  (rms_zero / rms_opt)\n"  (rms_zero / max(rms_opt, 1e-300))
    @printf "K-H cps           = %.3g   (mean source-spacing / dx, N_outer=%d, R_outer=%.3g)\n" CPS N_OUTER R_OUTER
    @printf "expected |Δs| floor ≈ %.2g %%  (from check_inject_kh cps sweep — error beyond this is GF-attributable)\n" _cps_floor_pct(CPS)
    println()
    @info "Figures saved to" fig_dir
end

# ----- Main loop -----------------------------------------------------------
for sc in SCATTERERS
    calibrate_one(sc)
end
