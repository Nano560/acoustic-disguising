# -----------------------------------------------------------------------------
# I/O: config loading, output-directory setup, HDF5 read/write, and
# Green's-function loaders.
#
# All generated data flows through HDF5 so the Python MDD stage can read
# the Julia output (and vice versa) without a custom format. JLD2 is kept
# for Julia-native intermediate state that does not need to cross the
# language boundary (e.g. wavefield snapshots).
# -----------------------------------------------------------------------------

using HDF5, TOML, JLD2

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

"""
    load_config(path::AbstractString) -> Dict{String,Any}

Parse a TOML config file (e.g. configs/paper.toml) and return a dictionary.
"""
load_config(path::AbstractString) = TOML.parsefile(path)

const _REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

"""
    nested_merge(d, overrides) -> Dict

Recursively merge `overrides` into `d`. Sub-dicts are merged element-wise;
scalars in `overrides` replace their counterparts in `d`. The inputs are
not mutated. Used to apply `[greens.*]` / `[reverb.*]` cleanly
on top of the nested base config.
"""
function nested_merge(d::AbstractDict, overrides::AbstractDict)
    out = deepcopy(d)
    for (k, v) in overrides
        if v isa AbstractDict && haskey(out, k) && out[k] isa AbstractDict
            out[k] = nested_merge(out[k], v)
        else
            out[k] = deepcopy(v)
        end
    end
    return out
end

"""
    apply_overrides(cfg, stage) -> Dict

For a stage table (`"greens"`, `"reverb"`, ...), merge its `.grid`,
`.pml`, and `.surfaces` sub-tables into the matching shared top-level
sections. Stage-level scalars (`duration`, `nt_save`, ...) are left in
place — callers read them as `cfg[stage][...]`. No-op if the stage is
absent or carries no override sub-tables.
"""
function apply_overrides(cfg::AbstractDict, stage::AbstractString)
    stage_cfg = get(cfg, stage, Dict{String,Any}())
    isempty(stage_cfg) && return cfg
    sub = Dict{String,Any}()
    for k in ("grid", "pml", "surfaces")
        v = get(stage_cfg, k, nothing)
        v isa AbstractDict && !isempty(v) && (sub[k] = v)
    end
    isempty(sub) && return cfg
    return nested_merge(cfg, sub)
end

"""
    set_in!(cfg, section => Dict("k" => v, ...))

Mutate `cfg[section]` to merge in the given key/value pairs. Returns `cfg`.
"""
function set_in!(cfg::AbstractDict, section::AbstractString, overrides::AbstractDict)
    inner = get!(cfg, section, Dict{String,Any}())
    merge!(inner, overrides)
    return cfg
end

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

"""
    parse_flag(args, prefix; default = nothing)

Return the value of the first `--key=value` entry in `args` whose key matches
`prefix` (e.g. `"--scatterer="`), or `default` if no such entry is present.
"""
function parse_flag(args, prefix::AbstractString; default = nothing)
    for a in args
        startswith(a, prefix) && return String(split(a, "=", limit = 2)[2])
    end
    return default
end

"""
    parse_run_dir_arg(args) -> String

Return the first non-flag positional argument in `args` interpreted as a
run directory (the absolute path is returned). Errors if none is given —
every pipeline script requires an explicit run dir.
"""
function parse_run_dir_arg(args)
    for a in args
        !startswith(a, "--") && return abspath(String(a))
    end
    error("missing positional <run_dir> argument; usage: <script> <run_dir>")
end

# -----------------------------------------------------------------------------
# Graceful-interrupt sentinel
# -----------------------------------------------------------------------------

"""
    stop_requested(stopfile::AbstractString) -> Bool

Return `true` if `stopfile` exists. Long-running loops poll this at the top
of each iteration; `true` is the user's signal to finish the current item
and break out, letting surrounding save/checkpoint code run naturally.

The user triggers it from another terminal with `touch <stopfile>`. The
file itself has no contents — its existence is the signal.
"""
stop_requested(stopfile::AbstractString) = isfile(stopfile)

"""
    fib_sphere_area(radius::Real, n_points::Integer) -> Float64

Per-point Fibonacci-sphere surface-area element `4π·r²/N`. Used as the
Kirchhoff–Helmholtz quadrature weight on source/receiver surfaces.
"""
fib_sphere_area(radius::Real, n_points::Integer) = 4π * radius^2 / n_points

"""
    clear_stop_file!(stopfile::AbstractString) -> Nothing

Remove `stopfile` if it exists; no-op otherwise. Called once at script
startup (to drop leftovers from a previous run) and once at clean exit
(after the loops have observed and acted on it).
"""
function clear_stop_file!(stopfile::AbstractString)
    if isfile(stopfile)
        try
            rm(stopfile; force = true)
        catch e
            @warn "could not remove stop file" stopfile exception=e
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------

"""
    ensure_output_dirs(run_dir::AbstractString) -> NamedTuple

The `run_dir` is the single root for everything a pipeline run produces:
its `config.toml`, all stage outputs, all figures. Returns the canonical
subfolder paths as a NamedTuple but does NOT pre-create them — each
stage's save site (`save_h5`, `state.jl::open_or_create_state`, every
plot helper) already `mkpath`s its own parent. The result: a run dir
contains *only* the subfolders for stages that actually wrote output,
no empty stubs.

Run dir itself must already exist.
"""
function ensure_output_dirs(run_dir::AbstractString)
    base = abspath(run_dir)
    isdir(base) || error("run_dir does not exist: $base")
    return (
        run_dir    = base,
        greens     = joinpath(base, "greens"),
        reverb     = joinpath(base, "reverb"),
        # Plain holograms AND disguise H5s share `holograms/` — their filename
        # patterns (`hologram_<scat>_<gf>.h5` vs
        # `hologram_<real>_cloak_<target>_<gf>.h5`) don't collide.
        holograms  = joinpath(base, "holograms"),
        scattering = joinpath(base, "scattering"),
        figures    = joinpath(base, "figures"),
        renders    = joinpath(base, "renders"),
    )
end

"""
    config_path(run_dir::AbstractString) -> String

Path to the cfg file inside a run dir. Convention: `<run_dir>/config.toml`.
"""
config_path(run_dir::AbstractString) = joinpath(abspath(run_dir), "config.toml")

"""
    find_gf_file(greens_dir, stem, scatterer) -> String

Locate `<stem>_<scatterer>.h5` directly under `<greens_dir>/`. Errors
(with a hint) when not present.
"""
function find_gf_file(greens_dir::AbstractString, stem::AbstractString,
                      scatterer::Symbol)
    path = joinpath(greens_dir, "$(stem)_$(scatterer).h5")
    isfile(path) || error("GF file not found: $path. Did you run the upstream stage?")
    return path
end

# -----------------------------------------------------------------------------
# HDF5 round-trip for simple NamedTuple payloads
# -----------------------------------------------------------------------------
#
# HDF5 schema — keep this in sync with python/mdd/io.py
#   /gf              (nrec, nsrc, nt)  Float32   Green's functions
#   /times           (nt,)             Float32   time axis [s]
#   /src_positions   (nsrc, 3)         Float32   xyz [m]
#   /rec_positions   (nrec, 3)         Float32   xyz [m]
# attrs (root):
#   config_hash, git_sha, created_at, julia_version, cuda_version
# -----------------------------------------------------------------------------

"""
    save_h5(path, data::NamedTuple; attrs::AbstractDict = Dict())

Write each field of `data` as an HDF5 dataset under `path`, plus the given
root-level attributes. Parent directory is created if missing.

The write is atomic: data goes to `<path>.tmp` first, then `mv` replaces
`path` once the HDF5 close has flushed everything. A kill mid-write leaves
the previous good `path` untouched (instead of truncating it to a
half-written state, which is what plain `h5open(path, "w")` does).
"""
function save_h5(path::AbstractString, data::NamedTuple; attrs::AbstractDict = Dict())
    mkpath(dirname(path))
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        for (k, v) in pairs(data)
            f[string(k)] = v
        end
        for (k, v) in pairs(attrs)
            HDF5.attrs(f)[string(k)] = v
        end
    end
    mv(tmp, path; force = true)
    return path
end

"""
    load_h5(path) -> NamedTuple

Read every root-level dataset from `path` into a NamedTuple.
"""
function load_h5(path::AbstractString)
    pairs = Pair{Symbol,Any}[]
    h5open(path, "r") do f
        for name in keys(f)
            push!(pairs, Symbol(name) => read(f[name]))
        end
    end
    return (; pairs...)
end

