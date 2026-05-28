# -----------------------------------------------------------------------------
# Sidecar HDF5 state file used by `build_reverb_data` (`scripts/greens/reverb.jl`).
#
# Holds dom.dt-rate raw receiver traces and the final wavefield per ill
# source so that a reverb run can:
#
#   - resume by iSrc (restart the script, only un-computed sources are run);
#   - extend tmax in place (bump `[tmax].reverb`, the saved field is the
#     initial condition for the continuation);
#   - grow nPoints_ill (always safe — ill positions use the r2 sequence,
#     where position i depends only on i).
#
# File layout:
#
#   /raw_outer_p, /raw_outer_vnz   Float32 (dom.nt, n_outer, n_ill)  unlimited (t, n_ill)
#   /raw_inner_p, /raw_inner_vnz   Float32 (dom.nt, n_inner, n_ill)  unlimited (t, n_ill)
#   /field_chk                     Float32 (nx, ny, nz, 4, n_ill)    unlimited (n_ill)
#   /iSrc_done                     UInt8   (n_ill,)                   unlimited
#
#   attrs (root):
#     dom_dt, dom_tmax, dom_nt, nx, ny, nz,
#     n_outer, n_inner, n_ill, scatterer, created_at
# -----------------------------------------------------------------------------

using HDF5
using Dates: now

const STATE_CHUNK_T     = 4096
const STATE_FIELD_DTYPE = Float32

"""
    ReverbStateFile(path; n_outer, n_inner, n_ill, dom_nt)

Lightweight handle to the sidecar state file. Holds only the on-disk path
and the dataset shape — every read/write reopens the file.
"""
struct ReverbStateFile
    path    ::String
    dom_nt  ::Int
    n_outer ::Int
    n_inner ::Int
    n_ill   ::Int
end

# Read a root-level scalar attribute. Returns `default` if missing.
_attr_or(f, name, default) = haskey(HDF5.attrs(f), name) ? HDF5.attrs(f)[name] : default

# Pre-compute time-dim chunk shape for raw_* datasets given dom.nt.
_raw_chunk_t(dom_nt::Int) = min(STATE_CHUNK_T, max(dom_nt, 1))

"""
    open_or_create_state(path; dom, n_outer, n_inner, n_ill, scatterer)
        -> (state::ReverbStateFile, mode::Symbol, n_saved::Int)

`mode` ∈ (`:fresh`, `:resume`, `:extend`):

  - `:fresh`  — file did not exist or was incompatible; everything is empty.
  - `:resume` — saved tmax matches dom.tmax; un-done iSrc need running.
  - `:extend` — saved tmax < dom.tmax; saved iSrc resume from the saved
                 field, fresh iSrc start from scratch.

`n_saved` is the number of FDTD time steps already stored (`0` if `:fresh`).

Errors out (instead of silently truncating) when an existing file would have
to shrink along any axis.
"""
function open_or_create_state(state_path::AbstractString;
                              dom::Domain,
                              n_outer::Integer, n_inner::Integer, n_ill::Integer,
                              scatterer::Symbol)

    if isfile(state_path)
        # `static_ok` covers everything that MUST match exactly to keep the
        # existing data valid (grid, dt, receiver counts).
        static_ok, saved_dom_nt, saved_dom_tmax, saved_n_ill =
            h5open(state_path, "r") do f
                ok = isapprox(Float64(_attr_or(f, "dom_dt", -1.0)), dom.dt; rtol = 1e-9) &&
                     _attr_or(f, "nx", -1) == dom.nx &&
                     _attr_or(f, "ny", -1) == dom.ny &&
                     _attr_or(f, "nz", -1) == dom.nz &&
                     _attr_or(f, "n_outer", -1) == n_outer &&
                     _attr_or(f, "n_inner", -1) == n_inner
                (ok,
                 Int(_attr_or(f, "dom_nt", 0)),
                 Float64(_attr_or(f, "dom_tmax", 0.0)),
                 Int(_attr_or(f, "n_ill", -1)))
            end

        if !static_ok
            error("Existing state file is incompatible with the current cfg " *
                  "(grid, dt, n_outer, or n_inner differs). " *
                  "Delete $state_path and rerun.")
        end

        if saved_dom_tmax > dom.tmax + 1e-12
            error("Existing state has dom_tmax = $saved_dom_tmax > configured " *
                  "dom.tmax = $(dom.tmax). Refusing to shrink. Bump tmax in " *
                  "the config or delete $state_path.")
        end

        if saved_n_ill > n_ill
            error("Existing state has n_ill = $saved_n_ill > configured " *
                  "n_ill = $n_ill. Refusing to shrink. Bump nPoints_ill or " *
                  "delete $state_path.")
        elseif saved_n_ill < n_ill
            extend_state_n_ill!(state_path; new_n_ill = n_ill)
            @info "[reverb] state file extended along iSrc axis" path=basename(state_path) saved_n_ill new_n_ill=n_ill
        end

        if saved_dom_nt < dom.nt
            extend_state_time_dim!(state_path; new_dom_nt = dom.nt, new_dom_tmax = dom.tmax)
            @info "[reverb] state file extended" path=basename(state_path) saved_dom_nt new_dom_nt=dom.nt saved_dom_tmax new_dom_tmax=dom.tmax
            return (ReverbStateFile(state_path, saved_dom_nt, n_outer, n_inner, n_ill),
                    :extend, saved_dom_nt)
        else
            return (ReverbStateFile(state_path, saved_dom_nt, n_outer, n_inner, n_ill),
                    :resume, saved_dom_nt)
        end
    end

    return create_fresh_state(state_path; dom = dom,
                              n_outer = n_outer, n_inner = n_inner, n_ill = n_ill,
                              scatterer = scatterer)
end

"""
    create_fresh_state(path; dom, n_outer, n_inner, n_ill, scatterer)
        -> (state::ReverbStateFile, mode::Symbol = :fresh, n_saved::Int = 0)
"""
function create_fresh_state(state_path::AbstractString;
                            dom::Domain,
                            n_outer::Integer, n_inner::Integer, n_ill::Integer,
                            scatterer::Symbol)
    mkpath(dirname(state_path))
    chunk_t    = _raw_chunk_t(dom.nt)
    chunk_isrc = max(1, min(n_ill, 64))

    h5open(state_path, "w") do f
        # Raw FDTD-rate receiver traces. Time AND iSrc dims are unlimited so
        # later runs with larger tmax (extension) or larger n_ill can grow
        # them in place. ill positions use the r2 sequence, so growing n_ill
        # keeps existing iSrc positions fixed.
        space_outer = dataspace((dom.nt, n_outer, n_ill); max_dims = (-1, n_outer, -1))
        create_dataset(f, "raw_outer_p",   datatype(Float32), space_outer;
                       chunk = (chunk_t, n_outer, 1))
        space_outer = dataspace((dom.nt, n_outer, n_ill); max_dims = (-1, n_outer, -1))
        create_dataset(f, "raw_outer_vnz", datatype(Float32), space_outer;
                       chunk = (chunk_t, n_outer, 1))

        space_inner = dataspace((dom.nt, n_inner, n_ill); max_dims = (-1, n_inner, -1))
        create_dataset(f, "raw_inner_p",   datatype(Float32), space_inner;
                       chunk = (chunk_t, n_inner, 1))
        space_inner = dataspace((dom.nt, n_inner, n_ill); max_dims = (-1, n_inner, -1))
        create_dataset(f, "raw_inner_vnz", datatype(Float32), space_inner;
                       chunk = (chunk_t, n_inner, 1))

        # Final wavefield per iSrc — one chunk per iSrc, iSrc dim unlimited.
        space_field = dataspace((dom.nx, dom.ny, dom.nz, 4, n_ill);
                                max_dims = (dom.nx, dom.ny, dom.nz, 4, -1))
        create_dataset(f, "field_chk", datatype(Float32), space_field;
                       chunk = (dom.nx, dom.ny, dom.nz, 4, 1))

        # Per-iSrc completion bitmap.
        create_dataset(f, "iSrc_done", datatype(UInt8),
                       dataspace((n_ill,); max_dims = (-1,)); chunk = (chunk_isrc,))

        attrs = HDF5.attrs(f)
        attrs["dom_dt"]       = dom.dt
        attrs["dom_tmax"]     = dom.tmax
        attrs["dom_nt"]       = dom.nt
        attrs["nx"]           = dom.nx
        attrs["ny"]           = dom.ny
        attrs["nz"]           = dom.nz
        attrs["n_outer"]      = Int(n_outer)
        attrs["n_inner"]      = Int(n_inner)
        attrs["n_ill"]        = Int(n_ill)
        attrs["scatterer"]    = String(scatterer)
        attrs["created_at"]   = string(now())
    end

    return (ReverbStateFile(state_path, dom.nt, Int(n_outer), Int(n_inner), Int(n_ill)),
            :fresh, 0)
end

"""
    extend_state_n_ill!(path; new_n_ill)

Grow the iSrc axis to `new_n_ill` across `raw_*`, `field_chk`, `iSrc_done`.
Existing entries are preserved; the new range is zero-filled. Always safe
because ill positions use the r2 sequence (position i depends only on i).
"""
function extend_state_n_ill!(state_path::AbstractString; new_n_ill::Integer)
    h5open(state_path, "r+") do f
        for name in ("raw_outer_p", "raw_outer_vnz", "raw_inner_p", "raw_inner_vnz")
            d = f[name]
            sz = size(d)               # (nt, n_rec, n_ill_old)
            HDF5.set_extent_dims(d, (sz[1], sz[2], Int(new_n_ill)))
        end
        d = f["field_chk"]
        sz = size(d)                   # (nx, ny, nz, 4, n_ill_old)
        HDF5.set_extent_dims(d, (sz[1], sz[2], sz[3], sz[4], Int(new_n_ill)))
        d = f["iSrc_done"]
        HDF5.set_extent_dims(d, (Int(new_n_ill),))

        attrs = HDF5.attrs(f)
        haskey(attrs, "n_ill") && delete_attribute(f, "n_ill")
        attrs["n_ill"] = Int(new_n_ill)
    end
    return nothing
end

"""
    extend_state_time_dim!(path; new_dom_nt, new_dom_tmax)

Grow the time dimension of the `raw_*` datasets to `new_dom_nt`. Existing
entries `[1:old_nt, :, :]` are preserved by HDF5; the new region is
zero-filled. Updates the `dom_nt` / `dom_tmax` attrs.
"""
function extend_state_time_dim!(state_path::AbstractString;
                                new_dom_nt::Integer, new_dom_tmax::Real)
    h5open(state_path, "r+") do f
        for name in ("raw_outer_p", "raw_outer_vnz", "raw_inner_p", "raw_inner_vnz")
            d = f[name]
            sz = size(d)
            HDF5.set_extent_dims(d, (Int(new_dom_nt), sz[2], sz[3]))
        end
        attrs = HDF5.attrs(f)
        # HDF5.jl can't overwrite an existing attribute via assignment, so
        # delete + recreate.
        for name in ("dom_nt", "dom_tmax")
            haskey(attrs, name) && delete_attribute(f, name)
        end
        attrs["dom_nt"]   = Int(new_dom_nt)
        attrs["dom_tmax"] = Float64(new_dom_tmax)
    end
    return nothing
end

"""
    read_iSrc_state(state, iSrc, n_saved) -> NamedTuple | Nothing

Return the saved state for one iSrc, or `nothing` if the iSrc is not
marked done in the bitmap. `state` may be either a `ReverbStateFile` or
the bare path string. `n_saved` is the number of FDTD steps the caller
intends to read (it must be ≤ the saved time-axis length).
"""
function read_iSrc_state(state, iSrc::Integer, n_saved::Integer)
    state_path = state isa ReverbStateFile ? state.path : String(state)
    h5open(state_path, "r") do f
        done = read(f["iSrc_done"])[iSrc] != UInt8(0)
        done || return nothing

        outer_p   = f["raw_outer_p"][1:n_saved,   :, iSrc]
        outer_vnz = f["raw_outer_vnz"][1:n_saved, :, iSrc]
        inner_p   = f["raw_inner_p"][1:n_saved,   :, iSrc]
        inner_vnz = f["raw_inner_vnz"][1:n_saved, :, iSrc]
        field     = f["field_chk"][:, :, :, :, iSrc]
        return (; field, outer_p, outer_vnz, inner_p, inner_vnz)
    end
end

"""
    write_iSrc_state(state, iSrc, st)

Persist one iSrc's full-length raw traces and final wavefield. `st` is the
NamedTuple `(; field, outer_p, outer_vnz, inner_p, inner_vnz)` produced by
`build_reverb_data`'s `save_iSrc_done` callback. Marks the iSrc as done
in the bitmap.
"""
function write_iSrc_state(state, iSrc::Integer, st)
    state_path = state isa ReverbStateFile ? state.path : String(state)
    h5open(state_path, "r+") do f
        f["raw_outer_p"][:,   :, iSrc] = st.outer_p
        f["raw_outer_vnz"][:, :, iSrc] = st.outer_vnz
        f["raw_inner_p"][:,   :, iSrc] = st.inner_p
        f["raw_inner_vnz"][:, :, iSrc] = st.inner_vnz
        f["field_chk"][:, :, :, :, iSrc] = st.field
        done = read(f["iSrc_done"])
        done[iSrc] = UInt8(1)
        write(f["iSrc_done"], done)
    end
    return nothing
end

"""
    state_done_count(state) -> Int

Number of iSrc entries marked done in the bitmap. Returns 0 if the file or
the dataset doesn't exist.
"""
function state_done_count(state)
    state_path = state isa ReverbStateFile ? state.path : String(state)
    isfile(state_path) || return 0
    try
        h5open(state_path, "r") do f
            haskey(f, "iSrc_done") ? Int(sum(read(f["iSrc_done"]))) : 0
        end
    catch
        0
    end
end
