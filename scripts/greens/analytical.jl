#!/usr/bin/env julia
# =============================================================================
# Green's-function retrieval — method 2: analytical (closed form, hom only).
#
# Closed-form free-space monopole/dipole evaluations on the same Fibonacci-
# sphere geometry as `scripts/greens/impulsive.jl`. Schema matches
# `impulsive_hom.h5`, so the file drops straight into
# `scripts/hologram/synthesize.jl` without any other changes.
#
# The four free-space kernels — convention "first letter = receiver,
# second = source" matching src/greens.jl:
#   p_p  =  f'(τ) / (4π r)
#   v_p  =  (cos θ_r / ρ) · [ f(τ)/(4π r²)            + f'(τ)/(4π r c)  ]
#   p_v  = -cos θ_s        · [ f(τ)/(4π r²)            + f'(τ)/(4π r c)  ]
#   v_v  = -(cos θ_r · cos θ_s / ρ)
#                          · [ 2F(τ)/(4π r³) + 2f(τ)/(4π r² c) + f'(τ)/(4π r c²) ]
# where τ = t − r/c, F(τ) = ∫_{-∞}^τ f dt'. The pressure-source kernels carry
# `f'` (one extra time derivative) because the FDTD pressure source acts
# directly on ∂p/∂t; the velocity source enters as a body force, so its
# kernels carry `f`. No FDTD source-injection prefactor is applied here —
# `load_gf_ref` does that at load time when the H5 has `backend="analytical"`.
#
# Wavelet: a Gaussian δ-approximation parameterised by the spectrum's −3 dB
# cutoff `[greens].f_3db_hz`, with σ_t = √(ln 2)/(2π·f_3db_hz). Centred at
# t=0, normalised so ∫f·dt = 1. Same shape as `impulsive_wavelet` (the FDTD
# GFs are time-shifted to the same convention) so any cfg change propagates
# to both stages by construction.
#
# Writes:  <run_dir>/greens/analytical_hom.h5
# Usage:   julia --threads=10 --project=. scripts/greens/analytical.jl <run_dir>
# =============================================================================

using AcousticDisguising
using Dates
using Printf

const R0 = 1000.0                       # background density (water) [kg/m³]

# Abramowitz-Stegun 7.1.26 erf approximation (max error 1.5e-7) — keeps the
# script self-contained without a SpecialFunctions dependency.
@inline function _erf(x::Float64)
    s, ax = sign(x), abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
    return s * y
end

# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
const RUN_DIR = parse_run_dir_arg(ARGS)
const CONFIG  = config_path(RUN_DIR)
cfg = load_config(CONFIG)
const DIRS = ensure_output_dirs(RUN_DIR)
const OUT_PATH = joinpath(DIRS.greens, "analytical_none.h5")

# Skip when the GF file is already on disk. The closed-form GFs are written in
# a single atomic save_h5, so an existing file is always complete — there is no
# partial state to resume (unlike the impulsive / reverb stages). Delete the
# file to force a recompute; the run-dir convention is delete-and-rerun rather
# than a staleness / back-compat check.
if isfile(OUT_PATH)
    @info "GFs (analytical) — already on disk, skipping" out_path=relpath(OUT_PATH)
    exit(0)
end

# Apply [greens.grid] / [greens.pml] so dx, dt, dom_dt, etc. line up with
# the FDTD GF stage. We don't run any FDTD here — the Domain is just a
# convenient bag for dx/dt.
cfg_gf = apply_overrides(cfg, "greens")
g_gf   = cfg_gf["grid"]
dom = Domain(;
    tmax = Float64(cfg_gf["greens"]["duration"]),
    xmax = Float64(g_gf["xmax"]),
    ymax = Float64(g_gf["ymax"]),
    zmax = Float64(g_gf["zmax"]),
    nx   = Int(g_gf["n"]),  ny = Int(g_gf["n"]),  nz = Int(g_gf["n"]),
    cf   = Float64(g_gf["cf"]),
)
# Bandwidth: prefer the analytical-stage override if set, else fall back
# to the shared [greens].f_3db_hz (matches the FDTD-impulsive wavelet by
# construction). Setting [greens.analytical].f_3db_hz to a higher value
# gives a broader-band reference than the impulsive FDTD can produce —
# useful as ground truth for cross-bandwidth checks, but breaks direct
# apples-to-apples NCC against impulsive_*.h5.
greens_cfg = cfg_gf["greens"]
ana_cfg = get(greens_cfg, "analytical", Dict())
f_3db_hz = Float64(get(ana_cfg, "f_3db_hz", greens_cfg["f_3db_hz"]))
f_3db_src = haskey(ana_cfg, "f_3db_hz") ? "[greens.analytical]" : "[greens]"
σ_t = sqrt(log(2)) / (2π * f_3db_hz)
@info "GFs (analytical, hom)" nx=dom.nx dt=dom.dt dom.dx f_3db_hz f_3db_src σ_t C0 R0

# -------------------------------------------------------------------------
# Geometry: same Fibonacci spheres as impulsive.jl
# -------------------------------------------------------------------------
pts = illumination_points(cfg_gf)
src_pos, rec_pos = pts.outer, pts.inner
src_normals, rec_normals = pts.outer_normals, pts.inner_normals
n_src, n_rec = size(src_pos, 1), size(rec_pos, 1)

nt_gf = Int(cfg_gf["greens"]["nt_save"])
t_gf  = collect(range(0.0, dom.tmax; length = nt_gf))

# -------------------------------------------------------------------------
# Kernel evaluation. Four (nt_gf, n_inner, n_outer) tensors, evaluated in
# parallel over the source axis.
# -------------------------------------------------------------------------
@inline _f(τ, σ)  = (1 / (sqrt(2π) * σ)) * exp(-τ^2 / (2 * σ^2))
@inline _df(τ, σ) = -(τ / σ^2) * _f(τ, σ)
@inline _F(τ, σ)  = 0.5 * (1 + _erf(τ / (σ * sqrt(2))))

function analytical_kernels!(p_p, p_v, v_p, v_v,
                             src_pos, src_normals, rec_pos, rec_normals,
                             t_gf, σ_t)
    n_src, n_rec, nt = size(src_pos, 1), size(rec_pos, 1), length(t_gf)
    Threads.@threads for is in 1:n_src
        s_pt = @view src_pos[is, :]
        n_s  = @view src_normals[is, :]
        for ir in 1:n_rec
            r_pt = @view rec_pos[ir, :]
            n_r  = @view rec_normals[ir, :]

            dx, dy, dz = r_pt[1] - s_pt[1], r_pt[2] - s_pt[2], r_pt[3] - s_pt[3]
            r  = sqrt(dx^2 + dy^2 + dz^2)
            ihx, ihy, ihz = dx / r, dy / r, dz / r           # r̂ from src to rec
            cos_θr = n_r[1]*ihx + n_r[2]*ihy + n_r[3]*ihz
            cos_θs = n_s[1]*ihx + n_s[2]*ihy + n_s[3]*ihz
            retard = r / C0

            inv_4πr  = 1 / (4π * r)
            inv_4πr2 = 1 / (4π * r^2)
            inv_4πr3 = 1 / (4π * r^3)

            @inbounds for it in 1:nt
                τ   = t_gf[it] - retard
                fτ  = _f(τ, σ_t)
                dfτ = _df(τ, σ_t)
                Fτ  = _F(τ, σ_t)

                p_p[it, ir, is] = dfτ * inv_4πr
                v_p[it, ir, is] = (cos_θr / R0) *
                                  (fτ * inv_4πr2 + dfτ * inv_4πr / C0)
                p_v[it, ir, is] = -cos_θs *
                                  (fτ * inv_4πr2 + dfτ * inv_4πr / C0)
                v_v[it, ir, is] = -(cos_θr * cos_θs / R0) *
                                  (2 * Fτ * inv_4πr3 +
                                   2 * fτ * inv_4πr2 / C0 +
                                       dfτ * inv_4πr / C0^2)
            end
        end
    end
end

p_p = zeros(Float32, nt_gf, n_rec, n_src)
p_v = zeros(Float32, nt_gf, n_rec, n_src)
v_p = zeros(Float32, nt_gf, n_rec, n_src)
v_v = zeros(Float32, nt_gf, n_rec, n_src)

@info "Sweeping (src, rec) pairs" n_src n_rec nt_gf
t0 = time()
analytical_kernels!(p_p, p_v, v_p, v_v,
                    src_pos, src_normals, rec_pos, rec_normals,
                    t_gf, σ_t)
elapsed = time() - t0
@printf("[analytical] swept %d × %d pairs in %.1g s\n", n_src, n_rec, elapsed)

# -------------------------------------------------------------------------
# Save with the same H5 schema as impulsive_none.h5.
# -------------------------------------------------------------------------
save_h5(OUT_PATH, (;
    p_p, p_v, v_p, v_v,
    t = Float32.(t_gf),
    src_positions = Float32.(src_pos),
    rec_positions = Float32.(rec_pos),
); attrs = Dict{String,Any}(
    "scatterer"            => "none",
    "backend"              => "analytical",
    "dom_dt"               => dom.dt,
    "dom_dx"               => dom.dx,
    "dom_tmax"             => dom.tmax,
    "julia_version"        => string(VERSION),
    "config_file"          => abspath(CONFIG),
    "n"                    => Int(g_gf["n"]),
    "nPoints_outer"        => n_src,
    "nPoints_inner"        => n_rec,
    "n_done"               => n_src,
    "elapsed_sec"          => elapsed,
    "created_at"           => string(now()),
    "c0"                   => C0,
    "rho0"                 => R0,
    "wavelet_f_3db_hz"     => f_3db_hz,
    "wavelet_sigma_t"      => σ_t,
    # Canonical on-disk amplitude convention. Values are the closed-form
    # kernels with units (p_p, v_p, p_v, v_v) = (1/(m·s²), m/(kg·s),
    # 1/(m²·s), 1/kg). `load_gf_ref` reads this attr and errors out on
    # files that pre-date it (no back-compat path).
    "units_convention"     => "physical",
))

@info "Saved analytical GFs" out_path=relpath(OUT_PATH)
