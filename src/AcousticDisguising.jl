"""
    AcousticDisguising

Companion-code Julia package for the paper. The pipeline has two stages:

  1. Green's-function retrieval — three interchangeable methods:
       * `impulsive_gfs`        FDTD with an impulsive point source
       * analytical (closed form, `scripts/greens/analytical.jl`)
       * reverberant FDTD (`build_reverb_data`) → MDD (Python side)
  2. Hologram synthesis — `build_hologram` consumes any of the above.

Function groups (see exports below):

  Simulation     — `Domain`, `run_fdtd!`, `forward_onestep!`, `Cpml`,
                   `Transceiver`.
  Geometry       — `build_scatterer`, `build_update_mask`,
                   `volume_areas`, `getPolygons`.
  Illumination   — `fibonacci_sphere`, `r2_sphere`, `illumination_points`,
                   `impulsive_wavelet`, `ricker_wavelet`.
  Green's fns    — `impulsive_gfs`, `load_gf`, `load_gf_ref`, `load_gf_mdd`.
  Reverb         — `build_reverb_data`, `ReverbStateFile`,
                   `open_or_create_state`, …
  Hologram       — `build_hologram`.
  I/O            — `load_config`, `save_h5`, `load_h5`,
                   `ensure_output_dirs`.
  Plots / ETA    — `plot_hologram_cuts`, `plot_spatial_config`,
                   `benchmark_step`, `log_eta`.

The Python MDD side (`python/mdd/`) consumes the same HDF5 schema; see
`python/mdd/io.py` for the documented dataset names and dtypes.
"""
module AcousticDisguising

# ------------------------------------------------------------------
# Parallel / GPU backend
# ------------------------------------------------------------------
# The backend is selected via the ACOUSTIC_DISGUISING_BACKEND environment
# variable. If unset, the default is "Threads" everywhere — safe on
# every platform, no GPU driver required. Set ACOUSTIC_DISGUISING_BACKEND=CUDA
# explicitly to run on a Linux box with a working NVIDIA driver.
#
# Supported values: "CUDA", "Threads" (CPU — the backend used for the
# paper-scale runs).
# ------------------------------------------------------------------

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using EnumX

const BACKEND = get(ENV, "ACOUSTIC_DISGUISING_BACKEND", "Threads")

@static if BACKEND == "CUDA"
    using CUDA
    @init_parallel_stencil(CUDA, Float32, 2, inbounds = false)
    const DeviceArray = CUDA.CuArray
elseif BACKEND == "Threads"
    @init_parallel_stencil(Threads, Float32, 2, inbounds = false)
    const DeviceArray = Array
else
    error("Unknown ACOUSTIC_DISGUISING_BACKEND=\"$BACKEND\". Use \"CUDA\" or \"Threads\".")
end

# ------------------------------------------------------------------
# Public constants
# ------------------------------------------------------------------

"""
    C0 :: Float64

Default sound speed in the medium [m/s]. Water at 20 °C is 1500 m/s; this
matches the `Domain` default and is used by every pipeline stage that needs
a reference value (direct-arrival overlays, ETA estimates, etc.). Override
per-Domain via the `c0` keyword argument.
"""
const C0 = 1500.0

"""
    ALL_SCATTERERS :: NTuple{4,Symbol}

The four scatterer values the paper sweeps over: `:none` (no scatterer —
homogeneous-medium baseline run), `:sphere`, `:cube`, `:cross`. Pipeline
scripts iterate over this tuple when invoked with `--scatterer=all`.
"""
const ALL_SCATTERERS = (:none, :sphere, :cube, :cross)

"""
    GF_METHODS :: Dict{Symbol,@NamedTuple{impl::Symbol, stem::Symbol}}

Pipeline dispatch table for the three Green's-function methods (in
increasing computational cost). `impl` is the implementation symbol
consumed by `hologram_mode` / `build_hologram`; `stem` is the on-disk file
stem expected by `find_gf_file`. `:analytical` and `:impulsive` both go
through the `:ref` `build_hologram` path; `load_gf` disambiguates via the
H5 `backend` attribute.
"""
const GF_METHODS = Dict(
    :analytical => (impl = :ref, stem = "analytical"),
    :impulsive  => (impl = :ref, stem = "impulsive"),
    :mdd        => (impl = :mdd, stem = "mdd_extracted"),
)

"""
    GFContent

Selects which Green's-function content `load_gf` / `load_gf_ref` /
`load_gf_mdd` return. Two values:

  - `GFContent.heterogeneous` — the full GF as recorded with the
    scatterer present. The default everywhere.
  - `GFContent.scattered`     — the scatterer's contribution only;
    requires `homogeneous_path` to the matching homogeneous-medium GF
    file so the loader can subtract it.

Implemented via `EnumX.@enumx` for namespaced access; the underlying
type is `GFContent.T`.
"""
@enumx GFContent heterogeneous scattered

# ------------------------------------------------------------------
# Submodules
# ------------------------------------------------------------------
# Load order matters: cpmlcoeffs defines CPMLCoefficients (used by simulation's
# Cpml); simulation defines Domain + Cpml (used by geometry + everything below).
include("kernels/cpmlcoeffs.jl")
include("simulation.jl")              # defines TransceiverGrid used in forward.jl's signature
include("kernels/derivatives.jl")
include("kernels/updates.jl")
include("kernels/extrapolation.jl")
include("kernels/forward.jl")
include("resample.jl")
include("geometry.jl")
include("illumination.jl")
include("greens.jl")
include("reverb.jl")
include("hologram.jl")
include("io.jl")
include("greens_io.jl")
include("state.jl")
include("eta.jl")
include("plots.jl")
include("kh_kernels.jl")
include("metrics.jl")

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
# Simulation primitives
export Domain, Cpml, Transceiver, InterpolationPoint
export InterpolationIndices, InterpolationWeights
export ChannelBuffers, TransceiverGrid
export resetCPMLmemory!, transceiversToGrid, initialField
export p_src, p_rec, vn_src, vn_rec
export run_fdtd!, forward_onestep!
export GFMap, gf_arrays

# Geometry
export build_scatterer, build_update_mask, volume_areas, getPolygons

# Top-level pipeline stages (stubs; wired up as logic is ported)
export impulsive_gfs, build_reverb_data, build_hologram
export HologramMode, RealMode, RefMode, MDDMode, hologram_mode

# Illumination helpers
export illumination_points, fibonacci_sphere, r2_sphere, impulsive_wavelet, ricker_wavelet

# I/O
export load_config, save_h5, load_h5, ensure_output_dirs, config_path, find_gf_file
export parse_flag, parse_run_dir_arg
export stop_requested, clear_stop_file!
export apply_overrides, nested_merge, set_in!
export load_gf, load_gf_ref, load_gf_mdd
export gf_unit_factors, gf_to_physical, gf_to_raw_fdtd

# Resampling helper
export resample_time

# Pipeline-wide constants
export C0, ALL_SCATTERERS, GFContent, GF_METHODS

# ETA helpers
export benchmark_step, log_eta

# Reverb-stage state file (sidecar HDF5)
export ReverbStateFile, open_or_create_state, create_fresh_state
export extend_state_n_ill!, extend_state_time_dim!
export read_iSrc_state, write_iSrc_state, state_done_count

# Plots (CairoMakie)
export saturated_vmax, plot_hologram_cuts, plot_spatial_config

# K-H analytical kernels + direct-form convolutions (diagnostics use these)
export eval_analytical_kernels, eval_one_kernel
export direct_kh!, direct_kh_single!, direct_kh_accumulate!

# Inner-disk spatial metrics (used by K-H cancellation diagnostics)
export rms_in_mask, inner_dot, inner_disk_subarray, inner_disk_weight
export pad_inj_slab_to_pw, rms_region_diff

# Short grid-coordinate helpers (t/x/y/z/xv/yv/zv/coords) are deliberately
# NOT exported — their one-letter names collide with common local variables.
# Access as `AcousticDisguising.x(d)` if needed.

end # module
