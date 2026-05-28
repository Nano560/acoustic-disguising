# -----------------------------------------------------------------------------
# Impulsive-source Green's function generation.
#
# Naming convention (first letter = receiver, second letter = source):
#   p_p  = pressure-receiver ← pressure-source
#   p_v  = pressure-receiver ← velocity-source
#   v_p  = vn-receiver       ← pressure-source
#   v_v  = vn-receiver       ← velocity-source
#
# All four have shape `(nt_gf, n_inner, n_outer)`. Sources sit on the OUTER
# Fibonacci sphere; receivers on the INNER. Two FDTD runs per source point,
# one per source type:
#   srcType = :p  → pressure monopole, direction [1, 0, 0, 0]
#   srcType = :v  → velocity monopole, direction [0, nx, ny, nz]
#                                               (outward normal of the source).
#
# By reciprocity these are also "inner-source → outer-receiver"; the hologram
# synthesis path (`load_gf_ref` / `load_gf_mdd`) consumes them under that
# interpretation.
# -----------------------------------------------------------------------------

using Dates
using Printf

"""
    impulsive_gfs(dom::Domain, cfg; scatterer::Symbol = :cross) -> NamedTuple

Run the FDTD simulation with an impulsive point source at each point of the
outer Fibonacci sphere, twice per position (pressure monopole, then velocity
monopole along the outward normal), recording pressure and normal-velocity at
every inner-sphere receiver.

Returns
    (; p_p, p_v, v_p, v_v, t, src_positions, rec_positions, dom_dt)

where each of `p_p, p_v, v_p, v_v::Array{Float32,3}` has shape
`(nt_gf, n_inner, n_outer)`. `src_positions` are the outer points (one row per
FDTD sweep); `rec_positions` are the inner points.

`t` is `0:dom.dt:dom.tmax` — the FDTD time step, not a resampled axis.
Downstream resampling (to the hologram-stage `dom.dt`, which may differ
because of CFL) happens inside `load_gf_ref`. `dom_dt` carries the same
value separately because `load_gf_ref`'s C_p / C_v amplitude-scaling
chain needs it as a scalar.

`scatterer` ∈ (`:none`, `:sphere`, `:cube`, `:cross`).
"""
function impulsive_gfs(dom::Domain, cfg; scatterer::Symbol = :cross,
                       on_source_complete = (iSrc, state) -> nothing,
                       init = nothing,
                       stopfile::Union{Nothing,AbstractString} = nothing)

    # -------------------------------------------------------------------------
    # Staircase scatterer mask (`nothing` for :none — no mask).
    # -------------------------------------------------------------------------
    va = build_update_mask(dom, scatterer, cfg)

    # -------------------------------------------------------------------------
    # Compact-Gaussian source-time function.
    # -------------------------------------------------------------------------
    f_3db_hz = Float64(cfg["greens"]["f_3db_hz"])
    srctf, timeshift = impulsive_wavelet(dom; f_3db_hz = f_3db_hz)
    rectf = zeros(Float64, dom.nt)

    # -------------------------------------------------------------------------
    # Fibonacci sources (outer) + receivers (inner).
    # -------------------------------------------------------------------------
    pts = illumination_points(cfg)
    src_positions = pts.outer        # n_outer × 3
    rec_positions = pts.inner        # n_inner × 3
    src_normals   = pts.outer_normals
    rec_normals   = pts.inner_normals

    n_src = size(src_positions, 1)   # = n_outer
    n_rec = size(rec_positions, 1)   # = n_inner

    # -------------------------------------------------------------------------
    # Receiver transceivers — same list is reused for every source FDTD run.
    # `direction = [q=0, nx, ny, nz]` makes transceiversToGrid project the
    # velocity components onto the receiver's outward normal, giving `vn`.
    # -------------------------------------------------------------------------
    receivers = Vector{Transceiver}(undef, n_rec)
    for i in 1:n_rec
        direction = vcat(0.0, rec_normals[i, :])
        receivers[i] = Transceiver(dom;
            point     = rec_positions[i, :],
            direction = direction,
            tf        = rectf,
        )
    end

    # -------------------------------------------------------------------------
    # CPML + preallocated field.
    # -------------------------------------------------------------------------
    cpml = Cpml(dom;
        npml  = cfg["pml"]["n"],
        rcoef = Float64(cfg["pml"]["rcoef"]),
        fc    = Float64(cfg["pml"]["fc"]),
    )
    field = @zeros(dom.nx, dom.ny, dom.nz, 4)

    # -------------------------------------------------------------------------
    # Output tensors: (nt_gf, n_rec, n_src) — matches the original gfs.jl
    # convention and what _read_gf_ref_h5 expects (axis 1 = time).
    # -------------------------------------------------------------------------
    nt_gf = Int(cfg["greens"]["nt_save"])
    t_gf  = range(0.0, dom.tmax; length = nt_gf)

    if init === nothing
        p_p = zeros(Float32, nt_gf, n_rec, n_src)
        p_v = zeros(Float32, nt_gf, n_rec, n_src)
        v_p = zeros(Float32, nt_gf, n_rec, n_src)
        v_v = zeros(Float32, nt_gf, n_rec, n_src)
    else
        expected = (nt_gf, n_rec, n_src)
        size(init.p_p) == expected || error("impulsive_gfs: init.p_p shape $(size(init.p_p)) does not match expected $expected (nt_gf, n_rec, n_src). Was it built with a different config?")
        p_p = Float32.(copy(init.p_p))
        p_v = Float32.(copy(init.p_v))
        v_p = Float32.(copy(init.v_p))
        v_v = Float32.(copy(init.v_v))
    end

    t_shifted = range(-timeshift, dom.tmax - timeshift; length = dom.nt)

    # -------------------------------------------------------------------------
    # Sweep over outer source positions × source types. Skip iSrc whose slice
    # already has any non-zero value — that signals a previously-computed
    # source loaded via `init`.
    # -------------------------------------------------------------------------
    source_done(iSrc) = any(!iszero, @view(p_p[:, :, iSrc])) ||
                        any(!iszero, @view(p_v[:, :, iSrc])) ||
                        any(!iszero, @view(v_p[:, :, iSrc])) ||
                        any(!iszero, @view(v_v[:, :, iSrc]))

    n_already_done = count(source_done, 1:n_src)
    n_to_compute   = n_src - n_already_done
    @printf("[impulsive_gfs] starting sweep: n_src=%d, already_done=%d, to_compute=%d\n",
            n_src, n_already_done, n_to_compute)
    t_sweep_start = time()
    n_computed    = 0
    sec_per_src   = 0.0    # EMA-smoothed per-iSrc wall time; α=0.2 below
                           # → half-life ≈ 3 iSrc, tracks drift without jitter
    for iSrc in 1:n_src
        if source_done(iSrc)
            continue
        end
        if stopfile !== nothing && stop_requested(stopfile)
            @info "[gf] stop file detected — exiting loop after iSrc $(iSrc-1) of $n_src" stopfile
            break
        end
        t_src_start = time()
        for srcType in (:p, :v)
            point = src_positions[iSrc, :]
            direction = srcType === :p ? [1.0, 0.0, 0.0, 0.0] :
                                         vcat(0.0, src_normals[iSrc, :])

            source = Transceiver(dom; point = point, direction = direction, tf = srctf)
            txs       = [source, receivers...]
            txs_on_grid = transceiversToGrid(dom, txs)

            field .= 0
            resetCPMLmemory!(cpml)

            run_fdtd!(dom, field, txs_on_grid, cpml, va)

            # Extract receiver traces. The first channel is the source's own
            # self-record; skip it with `[:, 2:end]`. Stack pressure and vn
            # into a single (nt, n_rec, 2) tensor for joint interpolation.
            A = cat(
                Array( p_rec(txs_on_grid)[:, 2:end]),
                Array(vn_rec(txs_on_grid)[:, 2:end]),
                dims = 3,
            )

            # Denoise tiny numerical residue (1e-4 × per-channel max).
            max_vals = maximum(A, dims = 1)
            A .= ifelse.(abs.(A) .< 1e-4 .* max_vals, zero(eltype(A)), A)

            # Resample onto the GF time grid, shifted so the source-pulse
            # centre sits at t=0. Scale by dt/dt_gf (handled inside
            # `resample_time` via `scale_amplitude=true`) to preserve the
            # impulse-response energy density when the sample rate changes.
            A_resampled = Float32.(resample_time(A, t_shifted, t_gf; kind = :cubic))

            if any(isinf, A_resampled) || any(isnan, A_resampled)
                @warn "impulsive_gfs: non-finite values in resampled GF" iSrc srcType
            end

            # Store into the four tensors. Convention:
            #   first letter  = receiver channel (p or v)
            #   second letter = source type      (p or v)
            if srcType === :p
                p_p[:, :, iSrc] .= @view A_resampled[:, :, 1]
                v_p[:, :, iSrc] .= @view A_resampled[:, :, 2]
            else  # srcType === :v
                p_v[:, :, iSrc] .= @view A_resampled[:, :, 1]
                v_v[:, :, iSrc] .= @view A_resampled[:, :, 2]
            end
        end
        on_source_complete(iSrc, (;
            p_p, p_v, v_p, v_v,
            t = t_gf,
            src_positions,
            rec_positions,
            n_src,
        ))

        n_computed += 1
        Δt = time() - t_src_start
        sec_per_src = n_computed == 1 ? Δt : 0.2 * Δt + 0.8 * sec_per_src
        elapsed = time() - t_sweep_start
        n_left = n_to_compute - n_computed
        eta_sec = n_left * sec_per_src
        finish_at = now() + Second(round(Int, eta_sec))
        @printf("[gf]    src %4d/%-4d │ %4d done │ elapsed %5.2fh │ %5.2g min/src │ ETA %5.2fh (%s)\n",
                iSrc, n_src, n_computed,
                elapsed / 3600, sec_per_src / 60, eta_sec / 3600,
                Dates.format(finish_at, "yyyy-mm-dd HH:MM:SS"))
    end

    # -------------------------------------------------------------------------
    # Return (; ...). `save_h5` will use these field names as HDF5 dataset
    # names, which is exactly what `_read_gf_ref_h5` looks for.
    # -------------------------------------------------------------------------
    return (;
        p_p,
        p_v,
        v_p,
        v_v,
        t             = Float32.(collect(t_gf)),
        src_positions = Float32.(src_positions),
        rec_positions = Float32.(rec_positions),
    )
end
