# -----------------------------------------------------------------------------
# ETA benchmark + log helper.
#
# Both stages 1 and 2 estimate wall-clock by running a handful of FDTD steps
# on the actual Domain / cfg / scatterer (so the kernel branch and threading
# match the production run) and extrapolating. This file is the single source
# of truth for that estimate.
# -----------------------------------------------------------------------------

using Dates, Printf

"""
    benchmark_step(dom::Domain, cfg, scatterer::Symbol;
                   include_outer::Bool = false,
                   n_warmup::Int = 2, n_timed::Int = 20) -> Float64

Run `forward_onestep!` `n_timed` times on a benchmark Domain after `n_warmup`
discarded steps. Returns mean seconds per step.

The transceiver list mirrors what the production stage allocates so the
kernel takes the same branch (`update_p_mask!` vs `update_p!`, etc.):

- Stage 1 (impulsive GFs) records on the inner sphere only.  → `include_outer = false`
- Stage 2 (reverb) records on inner + outer.                 → `include_outer = true`

Falls back to a fixed 12 ns/cell-step if anything throws.
"""
function benchmark_step(dom::Domain, cfg, scatterer::Symbol;
                        include_outer::Bool = false,
                        n_warmup::Int = 2, n_timed::Int = 20)
    cells = dom.nx * dom.ny * dom.nz
    try
        field = @zeros(dom.nx, dom.ny, dom.nz, 4)

        pts   = illumination_points(cfg)
        rectf = zeros(Float64, dom.nt)
        txs   = Transceiver[]
        for i in 1:size(pts.inner, 1)
            push!(txs, Transceiver(dom;
                point     = pts.inner[i, :],
                direction = vcat(0.0, pts.inner_normals[i, :]),
                tf        = rectf))
        end
        if include_outer
            for i in 1:size(pts.outer, 1)
                push!(txs, Transceiver(dom;
                    point     = pts.outer[i, :],
                    direction = vcat(0.0, pts.outer_normals[i, :]),
                    tf        = rectf))
            end
        end
        txs_on_grid = transceiversToGrid(dom, txs)

        cpml = Cpml(dom;
            npml  = Int(cfg["pml"]["n"]),
            rcoef = Float64(cfg["pml"]["rcoef"]),
            fc    = Float64(cfg["pml"]["fc"]))
        va = build_update_mask(dom, scatterer, cfg)

        for _ in 1:n_warmup
            forward_onestep!(dom, field, txs_on_grid, cpml, va, 1)
        end
        t0 = time()
        for it in 1:n_timed
            forward_onestep!(dom, field, txs_on_grid, cpml, va, it)
        end
        return (time() - t0) / n_timed
    catch e
        @warn "benchmark_step: failed; using fallback 12 ns/cell-step" exception=e
        return cells * 12e-9
    end
end

"""
    log_eta(label, dom, cfg, scatterer, n_passes; include_outer = false)

Print the standard "rough ETA" log line used by stages 1 and 2:

    [label] remaining_passes=… time_per_pass="…" total_wall_time="≈ Xh Ym"
            eta_clock="yyyy-mm-dd HH:MM" rate_ns_per_cell_step=…
"""
function log_eta(label::AbstractString, dom::Domain, cfg, scatterer::Symbol,
                 n_passes::Integer; include_outer::Bool = false)
    sec_per_step = benchmark_step(dom, cfg, scatterer; include_outer = include_outer)
    sec_per_pass = sec_per_step * dom.nt
    est_sec      = sec_per_pass * n_passes
    h, rem       = divrem(est_sec, 3600)
    m, _         = divrem(rem, 60)
    eta          = now() + Dates.Second(round(Int, est_sec))

    per_pass_str = sec_per_pass < 60   ? @sprintf("%ds", round(Int, sec_per_pass))     :
                   sec_per_pass < 3600 ? @sprintf("%dm%02ds", Int(sec_per_pass ÷ 60),
                                                            round(Int, sec_per_pass % 60)) :
                                         @sprintf("%dh%02dm", Int(sec_per_pass ÷ 3600),
                                                            Int((sec_per_pass % 3600) ÷ 60))
    cells = dom.nx * dom.ny * dom.nz
    @info label remaining_passes=Int(n_passes) time_per_pass="≈ $(per_pass_str)" total_wall_time="≈ $(Int(h))h $(Int(m))m" eta_clock=Dates.format(eta, "yyyy-mm-dd HH:MM") rate_ns_per_cell_step=round(sec_per_step / cells * 1e9; sigdigits=2)
    return sec_per_step
end
