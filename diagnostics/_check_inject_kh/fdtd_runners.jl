function build_txs(dom, outer_pts, outer_nrm, inner_pts, inner_nrm)
    rectf   = zeros(Float64, dom.nt)
    n_outer = size(outer_pts, 1)
    n_inner = size(inner_pts, 1)
    txs     = Vector{Transceiver}(undef, n_outer + n_inner)
    for i in axes(outer_pts, 1)
        txs[i] = Transceiver(dom;
            point     = collect(outer_pts[i, :]),
            direction = collect(vcat(0.0, outer_nrm[i, :])),
            tf        = rectf)
    end
    for i in axes(inner_pts, 1)
        txs[n_outer + i] = Transceiver(dom;
            point     = collect(inner_pts[i, :]),
            direction = collect(vcat(0.0, inner_nrm[i, :])),
            tf        = rectf)
    end
    return (; txs, outer_rng = 1:n_outer, inner_rng = (n_outer + 1):(n_outer + n_inner))
end

"""
    inject_inner_sources!(txs_on_grid, inner_rng; vn_inject, p_inject) -> txs_on_grid

Write the q-channel (monopole, fed into the `p` source-time-function
register via `p_src(txs_on_grid)`) and f-channel (dipole, fed into the
`vn` register via `vn_src(txs_on_grid)`) source signals for the inner-
sphere transceivers in one call.

Caller controls the channel signs. In this script all three callers
(Part 2, Part 3b Run B, Part 4) use `vn_inject = +C_f · p_recorded`
and `p_inject = +C_q · vn_recorded` — see the sign-convention block
above `C_for_N` in `injection_coefficients.jl` for why the f-channel
is positive (i.e. opposite of the De Hoop `f = -p · n̂` body-force
convention, which would imply `-C_f`).

`inner_rng` is the range returned by `build_txs`; pass exactly that
(don't recompute as `(N+1):(2N)`).
"""
function inject_inner_sources!(txs_on_grid, inner_rng;
                               vn_inject::AbstractMatrix,
                               p_inject::AbstractMatrix)
    vn_src(txs_on_grid)[:, inner_rng] .= Float32.(vn_inject)
    p_src( txs_on_grid)[:, inner_rng] .= Float32.(p_inject)
    return txs_on_grid
end

"Run FDTD with y=0 slab capture via on_step callback. Returns (slabs, snap_times)."
function run_with_slab_capture!(dom, field, txs_on_grid, cpml; snap_every::Int = max(1, dom.nt ÷ 200))
    cs = AcousticDisguising.coords(dom)
    iy0 = argmin(abs.(cs[:y][:p] .- 0.0))
    slabs = Matrix{Float32}[]
    steps = Int[]
    cb = function (it, f)
        it % snap_every == 0 || return
        push!(slabs, Array(f[:, iy0, :, 1]))
        push!(steps, it)
    end
    run_fdtd!(dom, field, txs_on_grid, cpml, nothing; on_step = cb)
    snap_times = Float64.(steps .* dom.dt)
    return slabs, snap_times
end

"Slab + actual time of the snapshot closest to t_target."
function slab_at_time(slabs, snap_times, t_target)
    i = argmin(abs.(snap_times .- t_target))
    return Float64.(slabs[i]), snap_times[i]
end

"Inner-disk sub-array of a y=0 slab — square crop of cells with |x|, |z| ≤ R_INNER.
Both dom_pw and dom_inj share dx (with odd nx forced via `_snap_extent`), so
the returned sub-arrays for slab_A (on dom_pw) and slab_B (on dom_inj) have
the same shape and align cell-by-cell. NOTE: this is a SQUARE crop; pair with
`inner_disk_weight` for the actual circular-disk RMS / inner-dot."
