# -----------------------------------------------------------------------------
# Shared CairoMakie plot helpers used by the pipeline scripts and figures.
# Helpers tightly coupled to a single script's local state (live per-source
# GF panels, live reverb snapshots) stay in their respective scripts; only
# routines reused by multiple call sites are lifted here.
# -----------------------------------------------------------------------------

using CairoMakie
using Statistics: quantile

"""
    saturated_vmax(slab; q = 0.98) -> Float64

98th-percentile (default) of `|slab|`, robust to sparse data: takes the
percentile only over non-zero entries (otherwise mostly-zero arrays give
`vmax = 0` and the heatmap renders all black). Returns `1.0` if every
entry is zero, so callers don't need to guard against that.
"""
function saturated_vmax(slab::AbstractArray; q::Real = 0.98)
    absv = abs.(vec(slab))
    nz   = filter(>(0), absv)
    v = if !isempty(nz)
        quantile(nz, q)
    elseif !isempty(absv)
        maximum(absv)
    else
        0.0
    end
    return v > 0 ? Float64(v) : 1.0
end

"""
    plot_hologram_cuts(p::AbstractArray{<:Real,3}, dom, out_path::AbstractString;
                       vmax::Real = 0.8,
                       cfg = nothing,
                       scatterer = nothing) -> Nothing

xy pressure-field slice at z = 0, saved as an image-only PNG (no axes,
title, or colorbar — the figure canvas is exactly the heatmap, sized to
match the data aspect).

`vmax` is the colour-axis cap (symmetric around zero). Defaults to 0.8.

When `cfg` is provided, the inner / outer recording-surface circles
(radii from `cfg["surfaces"]`) are overlaid as dashed lines (red / blue).
When `scatterer` is also provided (`:none`, `:sphere`, `:cube`, or
`:cross`), the scatterer cross-section is overlaid as a semi-transparent
white fill with a solid outline — convention shared with `getPolygons`.
"""
function plot_hologram_cuts(p::AbstractArray{<:Real,3}, dom, out_path::AbstractString;
                            vmax::Real = 0.8,
                            cfg = nothing, scatterer = nothing)
    nx, ny, nz = size(p)
    k0 = (nz + 1) ÷ 2
    xs = range(-dom.xmax, dom.xmax; length = nx)
    ys = range(-dom.ymax, dom.ymax; length = ny)

    # Figure size tracks the data aspect so the heatmap fills the canvas
    # without letterboxing.
    npix = 800
    aspect = (2 * dom.xmax) / (2 * dom.ymax)
    fig_size = aspect >= 1 ? (npix, round(Int, npix / aspect)) :
                             (round(Int, npix * aspect), npix)
    fig = Figure(size = fig_size, figure_padding = 0)
    ax = Axis(fig[1, 1]; aspect = DataAspect(),
              xautolimitmargin = (0.0, 0.0),
              yautolimitmargin = (0.0, 0.0))
    hidedecorations!(ax)
    hidespines!(ax)

    heatmap!(ax, xs, ys, p[:, :, k0]; colormap = :berlin,
             colorrange = (-vmax, vmax))

    if cfg !== nothing
        θ = range(0, 2π; length = 200)
        cθ, sθ = cos.(θ), sin.(θ)
        r_in  = Float64(cfg["surfaces"]["radius_inner"])
        r_out = Float64(cfg["surfaces"]["radius_outer"])
        lines!(ax, r_in  .* cθ, r_in  .* sθ; color = :red,
               linewidth = 1.5, linestyle = :dash)
        lines!(ax, r_out .* cθ, r_out .* sθ; color = :blue,
               linewidth = 1.5, linestyle = :dash)

        if scatterer !== nothing && Symbol(scatterer) !== :none
            cpml = Cpml(dom;
                        npml  = Int(cfg["pml"]["n"]),
                        rcoef = Float64(cfg["pml"]["rcoef"]),
                        fc    = Float64(cfg["pml"]["fc"]))
            for (name, sx, sy) in getPolygons(dom, cfg, cpml, String(scatterer))
                if name == "scatterer"
                    poly!(ax, Point2f.(sx, sy);
                          color = (:white, 0.35),
                          strokecolor = :white, strokewidth = 1.5)
                end
            end
        end
    end

    save(out_path, fig)
    display(fig)
    return nothing
end

"""
    plot_spatial_config(dom, cfg, n_ill::Integer, scatterer::Symbol, out_path;
                        radius_ill::Real = 0.6) -> Nothing

3-D scatter of the three sample spheres (inner / outer Fibonacci, ill r2)
inside the FDTD box wireframe. Saved to `out_path` (and displayed) so the
user can sanity-check radii / point counts / box margins before stage 2
starts the expensive FDTD work.
"""
function plot_spatial_config(dom, cfg, n_ill::Integer, scatterer::Symbol,
                             out_path::AbstractString;
                             radius_ill::Real = 0.6)
    pts_in_out = illumination_points(cfg)
    ill_pts, _ = r2_sphere(Int(n_ill), Float64(radius_ill))

    bx, by, bz = dom.xmax, dom.ymax, dom.zmax
    corners = [(s1*bx, s2*by, s3*bz) for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)] |> vec
    edges = [
        (1,2),(3,4),(5,6),(7,8),
        (1,3),(2,4),(5,7),(6,8),
        (1,5),(2,6),(3,7),(4,8),
    ]

    fig = Figure(size = (900, 800))
    Label(fig[0, :],
          "Spatial configuration · scatterer=$(scatterer) · " *
          "box ±$(round(bx; sigdigits=2)) m, n=$(dom.nx)";
          fontsize = 16)
    ax = Axis3(fig[1, 1]; aspect = (1, 1, 1),
               xlabel = "x [m]", ylabel = "y [m]", zlabel = "z [m]")

    for (i, j) in edges
        xs = [corners[i][1], corners[j][1]]
        ys = [corners[i][2], corners[j][2]]
        zs = [corners[i][3], corners[j][3]]
        lines!(ax, xs, ys, zs; color = (:gray, 0.5), linewidth = 1)
    end

    scatter!(ax, pts_in_out.inner[:, 1], pts_in_out.inner[:, 2], pts_in_out.inner[:, 3];
             color = :tomato, markersize = 6,
             label = "inner (r=$(cfg["surfaces"]["radius_inner"]), N=$(size(pts_in_out.inner,1)))")
    scatter!(ax, pts_in_out.outer[:, 1], pts_in_out.outer[:, 2], pts_in_out.outer[:, 3];
             color = :gold, markersize = 6,
             label = "outer (r=$(cfg["surfaces"]["radius_outer"]), N=$(size(pts_in_out.outer,1)))")
    scatter!(ax, ill_pts[:, 1], ill_pts[:, 2], ill_pts[:, 3];
             color = :dodgerblue, markersize = 8,
             label = "ill (r=$(radius_ill), N=$(size(ill_pts,1)))")
    axislegend(ax; position = :rt, framevisible = false)

    save(out_path, fig)
    display(fig)
    return nothing
end
