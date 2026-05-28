# -----------------------------------------------------------------------------
# Single-step FDTD driver. Composes the pressure / velocity update kernels
# from `updates.jl` with the source/receiver interpolation from
# `derivatives.jl` and (optionally) the GF extrapolation from
# `extrapolation.jl`, applying CPML and the staircase scatterer mask
# according to the call-site arguments.
#
# Used by `run_fdtd!` (in src/simulation.jl), which iterates over
# `1:dom.nt`. Pipeline scripts and the test suite call `run_fdtd!` rather
# than this function directly.
# -----------------------------------------------------------------------------

@views function forward_onestep!(
    d,
    field,
    txs::TransceiverGrid,
    cpml,
    va,
    it;
    gf_map::Union{Nothing,GFMap} = nothing,
)

    dx = d.dx
    dy = d.dy
    dz = d.dz
    nx = d.nx
    ny = d.ny
    nz = d.nz
    nt = d.nt
    z0 = d.z0

    fq = d.fact_m0
    fv = d.fact_m1

    P = selectdim(field, 4, 1)
    Vx = selectdim(field, 4, 2)
    Vy = selectdim(field, 4, 3)
    Vz = selectdim(field, 4, 4)

    p = txs.p
    vx = txs.vx
    vy = txs.vy
    vz = txs.vz
    normals = txs.direction

    extrap = gf_map !== nothing

    mv_x = cpml.v[1]
    mv_y = cpml.v[2]
    mv_z = cpml.v[3]

    mp_x = cpml.p[1]
    mp_y = cpml.p[2]
    mp_z = cpml.p[3]

    a = cat(cpml.c.a_l, cpml.c.a_r, dims=2)
    b = cat(cpml.c.b_l, cpml.c.b_r, dims=2)
    ah = cat(cpml.c.a_hl, cpml.c.a_hr, dims=2)
    bh = cat(cpml.c.b_hl, cpml.c.b_hr, dims=2)
    npml = size(a, 1)

    nIntPoints = size(p.i, 2)

    rigidBoundaryYZ = true

    if rigidBoundaryYZ
        @parallel (1:nx-1, ny:ny, 1:nz-1) update_p!(field, fq, dx, dy, dz)
        @parallel (1:nx-1, 1:ny-1, nz:nz) update_p!(field, fq, dx, dy, dz)

    end

    # =========== update pressure ===========
    # Two modes:
    #   va === nothing → plain FDTD (no scatterer mask)
    #   else           → mask-aware staircase (skip cells where va[...,channel] is false)
    I = (1:nx-1, 1:ny-1, 1:nz-1)
    if va === nothing
        if npml == 0
            @parallel I update_p!(field, fq, dx, dy, dz)
        else
            @parallel I update_p_cpml!(field, fq, dx, dy, dz, mv_x, mv_y, mv_z, a, b)
        end
    else
        if npml == 0
            @parallel I update_p_mask!(field, fq, dx, dy, dz, va)
        else
            @parallel I update_p_cpml_mask!(field, fq, dx, dy, dz, va, mv_x, mv_y, mv_z, a, b)
        end
    end

    # inject sources q
    @parallel_async (1:nIntPoints) interp_trilinear!(P, it, p.i, p.w, p.src, p.rec)

    @synchronize

    # =========== update velocities ===========

    # update velocities
    if va === nothing
        @parallel_async (2:nx, 1:ny, 1:nz) update_vx!(P, Vx, fv, dx, mp_x, ah, bh)
        @parallel_async (1:nx, 2:ny, 1:nz) update_vy!(P, Vy, fv, dy, mp_y, ah, bh)
        @parallel_async (1:nx, 1:ny, 2:nz) update_vz!(P, Vz, fv, dz, mp_z, ah, bh)
    else
        @parallel_async (2:nx, 1:ny, 1:nz) update_vx_mask!(field, dx, va, fv, mp_x, ah, bh)
        @parallel_async (1:nx, 2:ny, 1:nz) update_vy_mask!(field, dy, va, fv, mp_y, ah, bh)
        @parallel_async (1:nx, 1:ny, 2:nz) update_vz_mask!(field, dz, va, fv, mp_z, ah, bh)
    end
    @synchronize

    # inject sources f
    if nIntPoints > 0
        @parallel (1:nIntPoints) interp_trilinear!(Vx, it, vx.i, vx.w, vx.src, vx.rec)
        @parallel (1:nIntPoints) interp_trilinear!(Vy, it, vy.i, vy.w, vy.src, vy.rec)
        @parallel (1:nIntPoints) interp_trilinear!(Vz, it, vz.i, vz.w, vz.src, vz.rec)
    end

    @synchronize

    # calculate normal velocities
    @parallel (1:nIntPoints) v2vn!(vx.rec, vy.rec, vz.rec, txs.vn_rec, normals, it)


    # =========== extrapolate ===========
    # Per-flavour dispatch lives next to the kernels it wraps —
    # see `extrapolate!` methods in src/kernels/extrapolation.jl.
    if extrap && it < nt
        extrapolate!(gf_map, txs, z0, it)
    end

    if it < nt
        @parallel (1:nIntPoints) vn2v!(vx.src, vy.src, vz.src, txs.vn_src, normals, it + 1)
    end

    if rigidBoundaryYZ
        field[:, 1, :, 3] .= -field[:, 2, :, 3]
        field[:, :, 1, 4] .= -field[:, :, 2, 4]
    end

end
