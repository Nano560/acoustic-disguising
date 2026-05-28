# -----------------------------------------------------------------------------
# Reverberating-source data generation (input for MDD).
#
# `build_reverb_data` is storage-agnostic: it pulls saved per-source state via
# the `load_iSrc_saved` callback and persists completed sources via
# `save_iSrc_done`. The HDF5 ORM behind those callbacks lives in `src/state.jl`.
#
# Returned NamedTuple (fields become HDF5 datasets via `save_h5`):
#
#   inner_p          (nt_ill, n_inner, n_ill)  Float32
#   inner_vnz        (nt_ill, n_inner, n_ill)  Float32   (multiplied by z0)
#   outer_p          (nt_ill, n_outer, n_ill)  Float32
#   outer_vnz        (nt_ill, n_outer, n_ill)  Float32   (multiplied by z0)
#   t                (nt_ill,)                 Float32
#   ill_positions    (n_ill, 3)                Float32
#   inner_positions  (n_inner, 3)              Float32
#   outer_positions  (n_outer, 3)              Float32
# -----------------------------------------------------------------------------

using Dates
using Printf

"""
    build_reverb_data(dom::Domain, cfg; scatterer::Symbol = :cross,
                      n_ill::Union{Int,Nothing} = nothing,
                      n_saved = 0,
                      load_iSrc_saved = (iSrc -> nothing),
                      save_iSrc_done = (iSrc, state) -> nothing,
                      on_source_complete = (iSrc, state) -> nothing,
                      snapshot_steps = Int[],
                      snapshot_iSrc = nothing) -> NamedTuple

Generate reverberant pressure + normal-velocity data on the inner and outer
control surfaces by firing impulses from an illumination sphere outside
both. This is the FDTD output that Multi-Dimensional Deconvolution
(python/mdd/) consumes to recover the Green's functions.

`n_ill` defaults to `cfg["reverb"]["nPoints_ill"]` if set, else
`cfg["surfaces"]["nPoints_outer"]`.

Resume / extend semantics. `n_saved` is the number of FDTD time steps
contained in any pre-existing state (0 if there is no saved state). For
each `iSrc` the function calls `load_iSrc_saved(iSrc)`:

  * `nothing`             — iSrc has never been simulated, run from scratch.
  * NamedTuple{:field, :outer_p, :outer_vnz, :inner_p, :inner_vnz} —
    iSrc was simulated up to `n_saved`. If `n_saved == dom.nt` the source
    is fully done (skip). If `n_saved < dom.nt` the saved field is loaded
    as initial condition and the FDTD run continues from step
    `n_saved + 1` with a zero source-time-function.

`prefill` is an alternative resume path that does NOT require the FDTD-rate
sidecar. Pass a NamedTuple `(; outer_p, outer_vnz, inner_p, inner_vnz)` of
arrays at the output `fs_out` rate (shape `(nt_ill, n_rec, n_prefill_ill)`,
with `n_prefill_ill ≤ n_ill`). Already-populated iSrc slots (any nonzero
data) are copied into the output arrays and skipped in the main loop. Only
supports iSrc-skip — cannot extend tmax (use `load_iSrc_saved` for that).
`prefill` and `load_iSrc_saved` can coexist; `load_iSrc_saved` wins for any
iSrc it returns non-`nothing` for.

After each iSrc finishes, `save_iSrc_done(iSrc, state)` is called with
the full-length raw traces and the final wavefield so the caller can
persist them. `on_source_complete` runs after that and is intended for
non-persistence side-effects (live-plot snapshots, ETA logging).

The wavelet is contained in `[0, 5/fc]` (≪ any reasonable saved tmax),
so zero-source continuation does not lose physical injection energy.
This also means CPML memory does NOT need to be checkpointed when
`pml_n == 0` (true for the reverb stage via `[reverb.pml]`); the
function will error if called with `pml_n > 0` and a non-zero `n_saved`.
"""
function build_reverb_data(dom::Domain, cfg; scatterer::Symbol = :cross,
                           n_ill::Union{Int,Nothing} = nothing,
                           on_source_complete = (iSrc, state) -> nothing,
                           n_saved::Int = 0,
                           load_iSrc_saved = (iSrc -> nothing),
                           save_iSrc_done = (iSrc, state) -> nothing,
                           save_field::Bool = true,
                           snapshot_steps::AbstractVector{<:Integer} = Int[],
                           snapshot_iSrc::Union{Nothing,AbstractSet{<:Integer}} = nothing,
                           prefill::Union{Nothing,NamedTuple} = nothing,
                           stopfile::Union{Nothing,AbstractString} = nothing)
    rev  = get(cfg, "reverb", Dict{String,Any}())
    r_ill  = Float64(get(rev, "radius_ill", 0.6))
    fc     = Float64(get(rev, "fc_source",  18_000.0))
    # fs_out is derived, not set: the recording covers a Ricker band of
    # `bandwidth_factor · fc_source`, so its Nyquist rate is
    # `fs_out = 2 · bandwidth_factor · fc_source`. Kept in lockstep with the
    # identical derivation in scripts/greens/reverb.jl::run_for_scatterer.
    bw_factor = get(rev, "bandwidth_factor", nothing)
    bw_factor === nothing && error(
        "[reverb].bandwidth_factor is required — it sets the recorded Ricker " *
        "band: fs_out = 2·bandwidth_factor·fc_source.")
    fs_out = 2.0 * Float64(bw_factor) * fc
    n_ill_ = n_ill === nothing ? Int(get(rev, "nPoints_ill", cfg["surfaces"]["nPoints_outer"])) : n_ill

    # Refuse to resume with active CPML — the memory state isn't checkpointed.
    if n_saved > 0 && Int(cfg["pml"]["n"]) > 0
        error("build_reverb_data: resume / extend is only supported with `[pml].n = 0` " *
              "(CPML memory is not checkpointed). Got pml.n = $(cfg["pml"]["n"]).")
    end
    if n_saved > dom.nt
        error("build_reverb_data: n_saved=$n_saved exceeds dom.nt=$(dom.nt). Refusing to shrink tmax.")
    end

    # -------------------------------------------------------------------------
    # Staircase scatterer mask (empty for :none — no mask).
    # -------------------------------------------------------------------------
    va = build_update_mask(dom, scatterer, cfg)

    # -------------------------------------------------------------------------
    # Ricker source wavelet with the injection scaling (cumsum × dt/dx),
    # matching the original illumination.jl. The zero variant is reused for
    # resumed iSrc (no re-firing — the wavelet has long decayed by tmax_old).
    # -------------------------------------------------------------------------
    srctf_full = ricker_wavelet(dom; fc = fc, injection = true)
    srctf_zero = zeros(Float64, dom.nt)
    rectf      = zeros(Float64, dom.nt)

    # -------------------------------------------------------------------------
    # Three spheres: inner (receivers), outer (receivers), ill (srcs).
    #   - inner / outer: Fibonacci (uniformity matters for K-H quadrature).
    #   - ill: r2 (Roberts' low-discrepancy sequence). Position i depends only
    #     on i, so growing nPoints_ill keeps existing iSrc positions fixed and
    #     just appends new ones — needed for incremental refinement of the
    #     reverb sweep without invalidating the saved state file.
    # -------------------------------------------------------------------------
    ill_points = r2_sphere(n_ill_, r_ill)[1]
    pts = illumination_points(cfg)
    inner_positions = pts.inner;  inner_normals = pts.inner_normals
    outer_positions = pts.outer;  outer_normals = pts.outer_normals

    n_inner = size(inner_positions, 1)
    n_outer = size(outer_positions, 1)

    # -------------------------------------------------------------------------
    # Build receiver transceivers: outer first, then inner. `direction =
    # [q=0, nx, ny, nz]` makes transceiversToGrid deposit the normal-velocity
    # projection into `txs_on_grid.vn_rec`.
    # -------------------------------------------------------------------------
    outer_txs = Vector{Transceiver}(undef, n_outer)
    for i in 1:n_outer
        outer_txs[i] = Transceiver(dom;
            point     = outer_positions[i, :],
            direction = vcat(0.0, outer_normals[i, :]),
            tf        = rectf)
    end
    inner_txs = Vector{Transceiver}(undef, n_inner)
    for i in 1:n_inner
        inner_txs[i] = Transceiver(dom;
            point     = inner_positions[i, :],
            direction = vcat(0.0, inner_normals[i, :]),
            tf        = rectf)
    end

    # Receiver index ranges inside the packed txs_on_grid.
    outer_range = 1:n_outer
    inner_range = (1:n_inner) .+ n_outer

    # -------------------------------------------------------------------------
    # CPML + preallocated wavefield.
    # -------------------------------------------------------------------------
    cpml = Cpml(dom;
        npml  = cfg["pml"]["n"],
        rcoef = Float64(cfg["pml"]["rcoef"]),
        fc    = fc,
    )
    field = @zeros(dom.nx, dom.ny, dom.nz, 4)

    # -------------------------------------------------------------------------
    # Output: resample FDTD traces (length dom.nt at dom.dt) onto a fs_out
    # grid. Matches the original `t_ill = 0:1/fs:dom.tmax`.
    # -------------------------------------------------------------------------
    t_ill = range(0.0, dom.tmax; step = 1.0 / fs_out)
    nt_ill = length(t_ill)

    outer_p   = zeros(Float32, nt_ill, n_outer, n_ill_)
    outer_vnz = zeros(Float32, nt_ill, n_outer, n_ill_)
    inner_p   = zeros(Float32, nt_ill, n_inner, n_ill_)
    inner_vnz = zeros(Float32, nt_ill, n_inner, n_ill_)

    # Output-file prefill: copy already-computed iSrc slots into the front of
    # the output arrays. The first-pass loop below marks the populated slots
    # as `fully_done` (any nonzero in the outer_p slice) and the main loop
    # skips them. Validation (shape, attrs match cfg) is the caller's job.
    n_prefill = 0
    if prefill !== nothing
        n_prefill = size(prefill.outer_p, 3)
        n_prefill <= n_ill_ ||
            error("build_reverb_data: prefill has $n_prefill iSrc but n_ill=$n_ill_")
        size(prefill.outer_p)   == (nt_ill, n_outer, n_prefill) ||
            error("build_reverb_data: prefill.outer_p shape $(size(prefill.outer_p)) " *
                  "does not match (nt_ill=$nt_ill, n_outer=$n_outer, n_prefill=$n_prefill)")
        size(prefill.outer_vnz) == (nt_ill, n_outer, n_prefill) ||
            error("build_reverb_data: prefill.outer_vnz shape mismatch")
        size(prefill.inner_p)   == (nt_ill, n_inner, n_prefill) ||
            error("build_reverb_data: prefill.inner_p shape mismatch")
        size(prefill.inner_vnz) == (nt_ill, n_inner, n_prefill) ||
            error("build_reverb_data: prefill.inner_vnz shape mismatch")
        @views begin
            outer_p[:,   :, 1:n_prefill] .= prefill.outer_p
            outer_vnz[:, :, 1:n_prefill] .= prefill.outer_vnz
            inner_p[:,   :, 1:n_prefill] .= prefill.inner_p
            inner_vnz[:, :, 1:n_prefill] .= prefill.inner_vnz
        end
    end

    z0 = dom.z0

    t_ill_arr     = Float32.(collect(t_ill))
    ill_pos_f32   = Float32.(ill_points)
    inner_pos_f32 = Float32.(inner_positions)
    outer_pos_f32 = Float32.(outer_positions)

    snapshot_set    = Set{Int}(snapshot_steps)
    snapshot_buffer = Dict{Int, Array{Float32, 4}}()

    # Pre-allocated per-iSrc raw-trace buffers — shape is constant across
    # iterations, so reuse them instead of `zeros(...)` allocating fresh
    # 4 × dom.nt × n_rec Float32 arrays each iteration (GC pressure was
    # showing up as an inter-iteration CPU dip).
    full_outer_p   = zeros(Float32, dom.nt, n_outer)
    full_outer_vnz = zeros(Float32, dom.nt, n_outer)
    full_inner_p   = zeros(Float32, dom.nt, n_inner)
    full_inner_vnz = zeros(Float32, dom.nt, n_inner)

    # First pass: probe load_iSrc_saved (FDTD-rate sidecar) and prefill
    # (fs_out-rate output file) to count fully-done vs to-compute without
    # driving the GPU. Cheap when load_iSrc_saved only reads metadata; the
    # prefill check is a memory scan over the already-loaded slice.
    fully_done = falses(n_ill_)
    has_saved  = falses(n_ill_)
    for iSrc in 1:n_ill_
        saved = load_iSrc_saved(iSrc)
        if saved !== nothing
            has_saved[iSrc] = true
            fully_done[iSrc] = (n_saved >= dom.nt)
            # Also pre-resample the saved iSrc into the output arrays now;
            # in mode A we never re-run them so the resample wouldn't happen
            # otherwise. In mode B we'll overwrite below.
            _resample_iSrc!(outer_p, outer_vnz, inner_p, inner_vnz, iSrc,
                            saved.outer_p, saved.outer_vnz,
                            saved.inner_p, saved.inner_vnz,
                            n_outer, n_inner, t(dom)[1:size(saved.outer_p, 1)],
                            t_ill_arr)
        elseif iSrc <= n_prefill && any(!iszero, @view outer_p[:, :, iSrc])
            # Prefill slot is populated → already done from a prior run.
            fully_done[iSrc] = true
        end
    end

    n_already_done = count(fully_done)
    n_to_compute   = n_ill_ - n_already_done
    @printf("\n[reverb] sweep: %d to compute (%d/%d already done, n_saved=%d, dom.nt=%d)\n",
            n_to_compute, n_already_done, n_ill_, n_saved, dom.nt)
    @printf("        %9s  %9s  %9s  %9s  %16s\n",
            "src", "elapsed", "min/src", "ETA", "finish")
    println("        ", "-"^60)

    t_sweep_start = time()
    n_computed    = 0
    sec_per_src   = 0.0    # EMA-smoothed per-iSrc wall time; α=0.2 below
                           # → half-life ≈ 3 iSrc, tracks drift without jitter

    for iSrc in 1:n_ill_
        if fully_done[iSrc]
            continue
        end
        if stopfile !== nothing && stop_requested(stopfile)
            @info "[reverb] stop file detected — exiting loop after iSrc $(iSrc-1) of $n_ill_" stopfile
            break
        end

        t_src_start = time()
        empty!(snapshot_buffer)

        # ---------------------------------------------------------------------
        # Decide init field, source TF, and which step to start the loop at.
        # ---------------------------------------------------------------------
        saved = has_saved[iSrc] ? load_iSrc_saved(iSrc) : nothing
        if saved !== nothing
            # Resume from a saved field; do not re-fire the source (Ricker
            # has long decayed at tmax_saved).
            field      .= saved.field
            srctf_for_this = srctf_zero
            start_step  = n_saved + 1
        else
            field      .= 0
            srctf_for_this = srctf_full
            start_step  = 1
        end

        # Pressure monopole at the ill point. Direction [z0, 0, 0, 0] matches
        # the original amplitude convention; transceiversToGrid scales the
        # source time function by the first component before injection.
        source = Transceiver(dom;
            point     = ill_points[iSrc, :],
            direction = [z0, 0.0, 0.0, 0.0],
            tf        = srctf_for_this,
        )
        txs       = vcat(outer_txs, inner_txs, [source])
        txs_on_grid = transceiversToGrid(dom, txs)

        # CPML memory must be reset per iSrc — but only the live fields. With
        # pml_n = 0 (enforced above for resume) the memory tensors are empty.
        resetCPMLmemory!(cpml)

        capture_this_iSrc = (snapshot_iSrc === nothing) || (iSrc in snapshot_iSrc)
        on_step = !capture_this_iSrc ? nothing :
            (it, f) -> (it in snapshot_set && (snapshot_buffer[it] = Array(f)))
        run_fdtd!(dom, field, txs_on_grid, cpml, va;
                  start_step = start_step, on_step = on_step)

        # ---------------------------------------------------------------------
        # Pull the receiver traces produced by this run. For resumed iSrc the
        # entries before start_step are zeros (we didn't iterate them); we
        # splice the saved old traces back in.
        # ---------------------------------------------------------------------
        txsOuterInner = 1:(n_outer + n_inner)
        p_rec_dev  =  p_rec(txs_on_grid)[:, txsOuterInner]
        vn_rec_dev = vn_rec(txs_on_grid)[:, txsOuterInner]
        p_host  = Array(p_rec_dev)    # (dom.nt, n_outer + n_inner) Float64
        vn_host = Array(vn_rec_dev)

        # `full_*` buffers are pre-allocated above the loop; we overwrite
        # rows [1:n_saved] from the saved checkpoint and [start_step:end]
        # from this iteration's fresh traces. Together that covers all
        # rows, so no zero-init is needed.
        if saved !== nothing
            @views full_outer_p[1:n_saved,   :] .= saved.outer_p
            @views full_outer_vnz[1:n_saved, :] .= saved.outer_vnz
            @views full_inner_p[1:n_saved,   :] .= saved.inner_p
            @views full_inner_vnz[1:n_saved, :] .= saved.inner_vnz
        end

        # Threaded Float32 cast + z0 scaling per receiver column.
        Threads.@threads for r in 1:n_outer
            rec = outer_range[r]
            @inbounds for ti in start_step:dom.nt
                pv = Float32(p_host[ti, rec])
                vv = Float32(vn_host[ti, rec]) * z0
                full_outer_p[ti, r]   = pv
                full_outer_vnz[ti, r] = vv
            end
        end
        Threads.@threads for r in 1:n_inner
            rec = inner_range[r]
            @inbounds for ti in start_step:dom.nt
                pv = Float32(p_host[ti, rec])
                vv = Float32(vn_host[ti, rec]) * z0
                full_inner_p[ti, r]   = pv
                full_inner_vnz[ti, r] = vv
            end
        end

        if any(!isfinite, full_outer_p) || any(!isfinite, full_inner_p)
            @warn "build_reverb_data: non-finite trace values" iSrc
        end

        # ---------------------------------------------------------------------
        # Persist iSrc state (raw traces + final field) before resampling so
        # a crash here loses at most the resample, not the simulation. The
        # ~437 MB device→host field copy is skipped when `save_field=false`
        # (set by callers that disable checkpointing), since the default
        # `save_iSrc_done` callback throws the field away anyway.
        # ---------------------------------------------------------------------
        final_field = save_field ? Float32.(Array(field)) : nothing
        save_iSrc_done(iSrc, (
            field     = final_field,
            outer_p   = full_outer_p,
            outer_vnz = full_outer_vnz,
            inner_p   = full_inner_p,
            inner_vnz = full_inner_vnz,
        ))

        # Resample iSrc's full-length raw traces onto the fs_out grid.
        _resample_iSrc!(outer_p, outer_vnz, inner_p, inner_vnz, iSrc,
                        full_outer_p, full_outer_vnz,
                        full_inner_p, full_inner_vnz,
                        n_outer, n_inner, t(dom), t_ill_arr)

        on_source_complete(iSrc, (;
            inner_p, inner_vnz, outer_p, outer_vnz,
            t               = t_ill_arr,
            ill_positions   = ill_pos_f32,
            inner_positions = inner_pos_f32,
            outer_positions = outer_pos_f32,
            n_ill           = n_ill_,
            snapshots       = snapshot_buffer,
            dom             = dom,
        ))

        n_computed += 1
        Δt = time() - t_src_start
        sec_per_src = n_computed == 1 ? Δt : 0.2 * Δt + 0.8 * sec_per_src
        elapsed = time() - t_sweep_start
        n_left = n_to_compute - n_computed
        eta_sec = n_left * sec_per_src
        finish_at = now() + Second(round(Int, eta_sec))
        src_label = @sprintf("%03d/%03d", iSrc, n_ill_)
        @printf("        %9s  %8.2fh  %8.2fm  %8.2fh  %16s\n",
                src_label,
                elapsed / 3600, sec_per_src / 60, eta_sec / 3600,
                Dates.format(finish_at, "mm-dd HH:MM"))
    end

    return (;
        inner_p,
        inner_vnz,
        outer_p,
        outer_vnz,
        t               = t_ill_arr,
        ill_positions   = ill_pos_f32,
        inner_positions = inner_pos_f32,
        outer_positions = outer_pos_f32,
    )
end

# Resample one iSrc's raw traces (dom.dt rate) onto fs_out and write into the
# output arrays at slice `iSrc`. `tin` is the time axis of the raw input;
# `tout` is the fs_out grid as a Float32 vector. The input arrays may be
# shorter than dom.nt (used during the pre-pass for fully-saved iSrc whose
# raw length is n_saved < dom.nt — not currently exercised, but defensive).
function _resample_iSrc!(outer_p, outer_vnz, inner_p, inner_vnz, iSrc::Int,
                          full_outer_p, full_outer_vnz,
                          full_inner_p, full_inner_vnz,
                          n_outer::Int, n_inner::Int,
                          tin::AbstractVector, tout::AbstractVector)
    nt_in = size(full_outer_p, 1)
    A_p  = hcat(full_outer_p,  full_inner_p)        # (nt_in, n_outer+n_inner)
    A_vn = hcat(full_outer_vnz, full_inner_vnz)
    A    = cat(A_p, A_vn; dims = 3)                 # (nt_in, n_total, 2)

    tin_range  = range(Float64(tin[1]), Float64(tin[nt_in]); length = nt_in)
    tout_range = range(Float64(tout[1]), Float64(tout[end]); length = length(tout))

    n_total = n_outer + n_inner
    # NOTE: scale_amplitude=false here — _resample_iSrc! preserves raw
    # amplitude on the resample, matching the original code that did NOT
    # apply the dt_out/dt_in factor at this site (the FDTD-rate traces are
    # already in the saved-amplitude convention).
    A_rs = Float32.(resample_time(A, tin_range, tout_range;
                                  kind = :cubic, scale_amplitude = false))

    @views begin
        outer_p[:,   :, iSrc] .= A_rs[:, 1:n_outer,            1]
        outer_vnz[:, :, iSrc] .= A_rs[:, 1:n_outer,            2]
        inner_p[:,   :, iSrc] .= A_rs[:, n_outer+1:n_total,    1]
        inner_vnz[:, :, iSrc] .= A_rs[:, n_outer+1:n_total,    2]
    end
    return nothing
end
