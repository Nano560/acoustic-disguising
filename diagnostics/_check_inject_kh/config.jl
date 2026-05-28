# ---- CLI ----------------------------------------------------------------
function _parse_flag(args, prefix, default)
    for a in args
        startswith(a, prefix) && return String(split(a, "=", limit = 2)[2])
    end
    return default
end

const RUN_DIR  = parse_run_dir_arg(ARGS)
const CFG_PATH = config_path(RUN_DIR)
cfg = load_config(CFG_PATH)
const DIRS = ensure_output_dirs(RUN_DIR)

# Hologram-stage constants (need INIT_DIST early for --tmax-ms default).
const FC_HZ            = Float64(get(cfg["hologram"], "fc",                4.0e3))
const INIT_DIST        = Float64(get(cfg["hologram"], "initial_distance",  0.65))    # was 0.62 (used historically)
const INIT_AMP         = Float64(get(cfg["hologram"], "initial_amplitude", 1.5))

# --ns: comma-separated list of surface-point counts.
const NS = parse.(Int, split(_parse_flag(ARGS, "--ns=", "300,500,700"), ","))

# --part4: opt-in long-propagation test that runs PW + injection on a dom_pw-
# sized domain extended to ≈ 2·T_TGT. Snapshots y=0 slabs at T_TGT and 2·T_TGT
# for three injection sources (direct, K-H pv_from_pv, K-H pv_from_pin) plus
# the bare-PW reference, to probe the "wave on the other side" cloaking
# behaviour. Cost is large (4 long FDTDs per N on dom_pw + 2 K-H convolutions);
# off by default.
const RUN_PART4 = ("--part4" in ARGS)

# --flip-vn-sign: diagnostic flag to flip the sign of the f-channel injection
# (`vn_inject = ±C_f · p_recorded`). The comment block above `C_for_N` claims
# the De Hoop minus (`-C_f`) empirically breaks cancellation; this flag lets us
# re-verify that claim by running both signs side-by-side. Default `+C_f`.
# Tagged into the Part 2 run-B cache filename so the two runs do not collide.
const FLIP_VN_SIGN  = ("--flip-vn-sign" in ARGS)
const VN_INJECT_SIGN = FLIP_VN_SIGN ? -1.0 : +1.0
const _FLIP_VN_TAG  = FLIP_VN_SIGN ? "_flipvn" : ""

# Part 3 / Part 4 kernel-mode flag (hoisted to module scope so Part 4 can read
# it too — used to be a local inside `run_part3_for_N`). Three modes:
#   full       — kernel runs to dom_pw.nt (default; correct, heavy).
#   trunc      — kernel truncated to max-arrival + 5σ_t (lossy: ~17 % vn
#                rel-RMS bleed from the v_v F-pedestal being cut off; kept
#                for side-by-side comparison plots).
#   trunc+ped  — same truncation but with the closed-form v_v pedestal
#                subtracted from the kernel and added back analytically at
#                convolution time via a cumsum tail. Mathematically
#                equivalent to `full` up to float roundoff, at ~2.4× lower
#                kernel RAM. See src/kh_kernels.jl::eval_kernel_pedestal.
# _KH_MODE_TAG is the filename suffix consumed by Part 3 caches, Part 3/4
# plot filenames, and Part 4 caches — single source of truth for the
# `_trunc` / `_truncped` cache-key surfix. _MODE_STR is the matching plot-
# title display suffix; empty in the `full` case so titles stay clean.
const KERNEL_MODE_STR = _parse_flag(ARGS, "--kernel-mode=", "full")
const KERNEL_MODE     = Symbol(replace(KERNEL_MODE_STR, "+" => "_"))
KERNEL_MODE in (:full, :trunc, :trunc_ped) ||
    error("--kernel-mode must be one of full|trunc|trunc+ped, got $KERNEL_MODE_STR")
const _KH_MODE_TAG = KERNEL_MODE === :full      ? ""         :
                     KERNEL_MODE === :trunc     ? "_trunc"   :
                                                  "_truncped"
const _MODE_STR    = KERNEL_MODE === :full ? "" : "  kernel_mode=$(KERNEL_MODE_STR)"

# --f-3db-hz: bandwidth of the analytical-kernel Gaussian source. Also the
# reference frequency for --ppw below (since the K-H kernels need to resolve
# this bandwidth, not just the lower Ricker source fc).
const F_3DB_HZ = parse(Float64, _parse_flag(ARGS, "--f-3db-hz=", "20000.0"))

# --medium / --rho / --c: select acoustic medium for the entire run.
# `--medium=water|air|seawater` is a shortcut for canonical (ρ, c) pairs;
# `--rho=` / `--c=` override either default or shortcut to arbitrary values.
# Each invocation simulates ONE medium; for a multi-medium sweep, run the
# script multiple times — the CSV at <DIAG_DIR>/sweep_results.csv
# accumulates rows from every invocation, tagged with rho / c / medium.
const _MEDIUM_PRESETS = Dict(
    "water"    => (rho = 1000.0, c = 1500.0),    # canonical water at 20°C
    "air"      => (rho =    1.2, c =  343.0),    # canonical air at sea level
    "seawater" => (rho = 1025.0, c = 1480.0),    # canonical seawater
)
const MEDIUM = lowercase(_parse_flag(ARGS, "--medium=", "water"))
const _MED_PRESET = get(_MEDIUM_PRESETS, MEDIUM, nothing)
_MED_PRESET === nothing && error("Unknown --medium=$MEDIUM; choose from $(keys(_MEDIUM_PRESETS)) or pass --rho/--c explicitly")
const RHO = parse(Float64, _parse_flag(ARGS, "--rho=", string(_MED_PRESET.rho)))
const C0  = parse(Float64, _parse_flag(ARGS, "--c=",   string(_MED_PRESET.c)))

# Cache-key tag for the medium. `rho<R>_c<C>` (rounded), so water →
# "rho1000_c1500", air → "rho1_c343", seawater → "rho1025_c1480".
# Embedded into every Part 1 / Part 2 Run B / Part 3 Run B HDF5 cache
# filename so per-medium runs don't clobber each other's data.
const _MED_TAG = "rho$(round(Int, RHO))_c$(round(Int, C0))"

# --ppw: FDTD points-per-wavelength at F_3DB_HZ in the chosen medium.
#   dx = C0 / (F_3DB_HZ · ppw)
# So ppw=30 means 30 cells per λ_3dB IN THIS MEDIUM (medium-aware): air at
# c=343 gets a finer dx than water at c=1500 to keep the same resolution.
# Default 30 (oversamples vs 10-15 typical FDTD; needed for clean K-H
# extrapolation kernels at 3·σ of the Gaussian bandwidth).
# --dx (legacy): explicit dx override (takes precedence over --ppw).
const PPW = parse(Float64, _parse_flag(ARGS, "--ppw=", "30"))
const _DX_FROM_CFG = 2 * Float64(cfg["grid"]["xmax"]) / (Int(cfg["grid"]["n"]) - 1)
const _DX_FROM_PPW = C0 / (F_3DB_HZ * PPW)
const _DX_DEFAULT  = _DX_FROM_PPW   # ppw-derived is the new default; pass --dx=X to override
const DX = parse(Float64, _parse_flag(ARGS, "--dx=", string(_DX_DEFAULT)))

# --L: perpendicular half-extent for dom_pw (y, z) and all dims for dom_inj.
const L_USER = parse(Float64, _parse_flag(ARGS, "--L=", "0.4"))

# --cf: Courant factor (default = cfg's grid.cf, typically 0.5). Smaller cf
# → smaller dt (dt = cf·dx/(c·√3)) → more FDTD steps per simulation. Used to
# verify the closed-form α's dt-independence in the (N, cf, dx) parameter sweep.
const _CF_FROM_CFG = Float64(cfg["grid"]["cf"])
const CF_USER      = parse(Float64, _parse_flag(ARGS, "--cf=", string(_CF_FROM_CFG)))

# --tmax-ms: FDTD propagation length. Default 2·INIT_DIST/c (medium-aware) —
# Ricker traverses both spheres. The Domain constructor rounds up to
# nt = ceil(tmax/dt).
const _DEFAULT_TMAX_MS = 2 * INIT_DIST / C0 * 1e3        # uses --c / --medium
const TMAX_MS  = parse(Float64, _parse_flag(ARGS, "--tmax-ms=", string(_DEFAULT_TMAX_MS)))

# --mask-taper-frac: width (fraction of R_inner) over which the inner-disk
# weight smoothly drops from 1 → 0 near r = R_inner. Default 0.2 (40 mm at
# R_inner=0.2 m, ≈ λ_3db/2 at f_3db_hz=20 kHz). Avoids the injection-source
# near-field at r=R_inner contaminating `rms_residual` (computed as a
# weighted L² norm over the inner disk).
const MASK_TAPER_FRAC = parse(Float64, _parse_flag(ARGS, "--mask-taper-frac=", "0.2"))

# ---- Apply --tmax-ms to cfg["hologram"]["duration"] (used by Domain below) ----
if TMAX_MS > 0
    new_tmax = TMAX_MS * 1e-3
    @info "Setting dom.tmax (--tmax-ms; default = 2·INIT_DIST/c so Ricker travels -INIT_DIST → +INIT_DIST)" old_tmax_ms=cfg["hologram"]["duration"]*1e3 new_tmax_ms=TMAX_MS
    cfg["hologram"]["duration"] = new_tmax
end

# ---- Snap extents to integer-multiples of DX so both Domain objects share dx exactly.
# Domain constructs dx as 2·xmax/(nx-1). With both domains sharing this dx, the
# inner-disk sub-array extracts from slab_A and slab_B will have identical shape.
function _snap_extent(extent::Real, dx::Real)
    n_raw     = round(Int, 2 * extent / dx) + 1
    # Force odd n → x=0 lies exactly on a grid cell. Required so the inner-disk
    # sub-array extraction (cells with |x| ≤ R_INNER) is symmetric and produces
    # the same count in dom_pw and dom_inj — otherwise dom_inj's even-nx skews
    # the count by 1 and the s_opt linear fit asserts a shape mismatch.
    n         = isodd(n_raw) ? n_raw : n_raw + 1
    snapped   = (n - 1) * dx / 2
    return snapped, n
end
const XMAX_PW_USER = Float64(cfg["grid"]["xmax"])
const XMAX_PW, NX_PW  = _snap_extent(XMAX_PW_USER, DX)
const L,        N_L    = _snap_extent(L_USER,        DX)

@info "Grid configuration" dx=DX cf=CF_USER ns=NS L_user=L_USER L_snapped=L n_L=N_L xmax_pw=XMAX_PW nx_pw=NX_PW

# ---- Domain construction.
# dom_pw  : Part 1 / Run-A — plane wave only. xmax full (need the Ricker to fit),
#           ymax = zmax = L (plane wave is uniform in y/z so y/z truncation is
#           lossless — no radiation hits the y/z walls).
# dom_inj : Run B — injection only, zero initial field. All three dims = L.
#           Wall reflections from inner-sphere sources arrive at the central
#           inner disk after T_TGT (homogeneous medium → no PML needed).
dom_pw = Domain(;
    tmax = Float64(cfg["hologram"]["duration"]),
    xmax = XMAX_PW, ymax = L, zmax = L,
    nx = NX_PW,    ny = N_L, nz = N_L,
    cf = CF_USER,
    c0 = C0, r0 = RHO,
)
dom_inj = Domain(;
    tmax = Float64(cfg["hologram"]["duration"]),
    xmax = L, ymax = L, zmax = L,
    c0 = C0, r0 = RHO,
    nx = N_L, ny = N_L, nz = N_L,
    cf = CF_USER,
)
@assert dom_pw.dt == dom_inj.dt   "dom_pw and dom_inj must share dt (same dx/cf)"
@assert dom_pw.nt == dom_inj.nt   "dom_pw and dom_inj must share nt (same tmax/dt)"

const R_OUTER = Float64(cfg["surfaces"]["radius_outer"])
const R_INNER = Float64(cfg["surfaces"]["radius_inner"])
const T_TGT   = INIT_DIST / dom_pw.c0           # Ricker peak passes the origin
const σ_T     = sqrt(log(2)) / (2π * F_3DB_HZ)  # Part 3 (deferred)
const T_GRID  = collect(range(0.0, dom_pw.tmax; length = dom_pw.nt))   # Part 3

@info "Setup" run_dir=RUN_DIR ns=NS r_outer=R_OUTER r_inner=R_INNER medium=MEDIUM rho=RHO c0=C0 z0=dom_pw.z0 dx=dom_pw.dx cf=dom_pw.cf dt=dom_pw.dt nt=dom_pw.nt T_TGT
@info "dom_pw" xmax=dom_pw.xmax ymax=dom_pw.ymax zmax=dom_pw.zmax nx=dom_pw.nx ny=dom_pw.ny nz=dom_pw.nz cells=dom_pw.nx*dom_pw.ny*dom_pw.nz
@info "dom_inj" xmax=dom_inj.xmax ymax=dom_inj.ymax zmax=dom_inj.zmax nx=dom_inj.nx ny=dom_inj.ny nz=dom_inj.nz cells=dom_inj.nx*dom_inj.ny*dom_inj.nz

const DIAG_DIR = joinpath(DIRS.figures, "check_inject_kh")
const DATA_DIR = joinpath(DIAG_DIR, "data")
mkpath(DATA_DIR)
