# -----------------------------------------------------------------------------
# FDTD pressure / velocity update kernels (plain, CPML, and staircase-mask
# variants). Each kernel writes into the staggered-grid `field` tensor in
# place; CPML memory variables are updated alongside.
#
# Layout convention:
#   field[:, :, :, 1]  → P    (cell-centred pressure)
#   field[:, :, :, 2]  → Vx
#   field[:, :, :, 3]  → Vy
#   field[:, :, :, 4]  → Vz
#
# `update[i,j,k,channel]` is the staircase scatterer mask: channel 1 is
# pressure, 2/3/4 are vx/vy/vz. A face channel is open iff both adjacent
# pressure cells are fluid.
# -----------------------------------------------------------------------------

# ============================================================================
# ==== unmasked pressure updates ====
# ============================================================================

@parallel_indices (i, j, k) function update_p_cpml!(
    field,
    fq,
    dx,
    dy,
    dz,
    mv_x,
    mv_y,
    mv_z,
    a,
    b,
)

    P = @view field[:, :, :, 1]
    Vx = @view field[:, :, :, 2]
    Vy = @view field[:, :, :, 3]
    Vz = @view field[:, :, :, 4]

    # Compute velocity derivatives
    dv_dx = @d_dx_2nd(Vx, i, j, k) / dx
    dv_dy = @d_dy_2nd(Vy, i, j, k) / dy
    dv_dz = @d_dz_2nd(Vz, i, j, k) / dz

    # Update CPML memory arrays if on the boundary
    # x: left and right
    npml = size(mv_x, 1)
    if npml > 1
        ndim = size(P, 1)
        mv = @view mv_x[:, j, k, :]

        l, iBoundary = getBoundaryIndex(i, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view a[:, iBoundary]
            b_ = @view b[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dv_dx
            dv_dx += m[l]
        end
    end

    # y: left and right
    npml = size(mv_y, 2)
    if npml > 1
        ndim = size(P, 2)
        mv = @view mv_y[i, :, k, :]

        l, iBoundary = getBoundaryIndex(j, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view a[:, iBoundary]
            b_ = @view b[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dv_dy
            dv_dy += m[l]
        end
    end

    # z: left and right
    npml = size(mv_z, 3)
    if npml > 1
        ndim = size(P, 3)
        mv = @view mv_z[i, j, :, :]

        l, iBoundary = getBoundaryIndex(k, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view a[:, iBoundary]
            b_ = @view b[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dv_dz
            dv_dz += m[l]
        end
    end

    # Update pressure
    P[i, j, k] -= fq * (dv_dx + dv_dy + dv_dz)

    return nothing
end

@parallel_indices (i, j, k) function update_p!(
    field,
    fq,
    dx,
    dy,
    dz,
)
    P = @view field[:, :, :, 1]
    Vx = @view field[:, :, :, 2]
    Vy = @view field[:, :, :, 3]
    Vz = @view field[:, :, :, 4]

    nx = size(field, 1)
    ny = size(field, 2)
    nz = size(field, 3)

    # Compute velocity derivatives
    if i == nx
        dv_dx = -2 .* Vx[i, j, k] / dx # rigid boundary
    else
        dv_dx = @d_dx_2nd(Vx, i, j, k) / dx
    end
    if j == ny
        dv_dy = -2 .* Vy[i, j, k] / dy    # rigid boundary
    else
        dv_dy = @d_dy_2nd(Vy, i, j, k) / dy
    end
    if k == nz
        dv_dz = -2 .* Vz[i, j, k] / dz    # rigid boundary
    else
        dv_dz = @d_dz_2nd(Vz, i, j, k) / dz
    end

    # Update pressure
    P[i, j, k] -= fq * (dv_dx + dv_dy + dv_dz)

    return nothing
end


# ============================================================================
# ==== single-axis CPML pressure helpers ====
# ============================================================================

@parallel_indices (i, j, k) function pml_px!(P, Vx, fq, dx, mv, a, b)
    # Compute velocity derivatives
    dv_dx = @d_dx_2nd(Vx, i, j, k) / dx

    # Update CPML memory arrays if on the boundary
    mv[i, j, k] = b[i] * mv[i, j, k] + a[i] * dv_dx

    # Update pressure
    P[i, j, k] -= fq * mv[i, j, k]

    return nothing
end

@parallel_indices (i, j, k) function pml_py!(P, Vy, fq, dy, mv, a, b)
    # Compute velocity derivatives
    dv_dy = @d_dy_2nd(Vy, i, j, k) / dy

    # Update CPML memory arrays if on the boundary
    mv[i, j, k] = b[j] * mv[i, j, k] + a[j] * dv_dy

    # Update pressure
    P[i, j, k] -= fq * mv[i, j, k]

    return nothing
end

@parallel_indices (i, j, k) function pml_pz!(P, Vz, fq, dz, mv, a, b)
    # Compute velocity derivatives
    dv_dz = @d_dz_2nd(Vz, i, j, k) / dz

    # Update CPML memory arrays if on the boundary
    mv[i, j, k] = b[k] * mv[i, j, k] + a[k] * dv_dz

    # Update pressure
    P[i, j, k] -= fq * mv[i, j, k]

    return nothing
end

# ============================================================================
# ==== unmasked velocity updates (with optional CPML) ====
# ============================================================================

@parallel_indices (i, j, k) function update_vx!(P, Vx, fv, dx, mp_x, ah, bh)
    # Compute pressure derivative in x direction
    dp_dx = @d_dx_2nd(P, i - 1, j, k) / dx

    # Update CPML memory arrays if on the boundary
    npml = size(mp_x, 1)
    if npml > 1
        ndim = size(P, 1)
        mv = @view mp_x[:, j, k, :]

        l, iBoundary = getBoundaryIndex(i, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view ah[:, iBoundary]
            b_ = @view bh[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dp_dx
            dp_dx += m[l]
        end
    end

    # Update velocity
    Vx[i, j, k] -= fv * dp_dx

    return nothing
end

@parallel_indices (i, j, k) function update_vy!(P, Vy, fv, dy, mp_y, ah, bh)
    # Compute pressure derivative in y direction
    dp_dy = @d_dy_2nd(P, i, j - 1, k) / dy

    # Update CPML memory arrays if on the boundary
    npml = size(mp_y, 2)
    if npml > 1
        ndim = size(P, 2)
        mv = @view mp_y[i, :, k, :]

        l, iBoundary = getBoundaryIndex(j, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view ah[:, iBoundary]
            b_ = @view bh[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dp_dy
            dp_dy += m[l]
        end
    end

    # Update velocity
    Vy[i, j, k] -= fv * dp_dy

    return nothing
end

@parallel_indices (i, j, k) function update_vz!(P, Vz, fv, dz, mp_z, ah, bh)
    # Compute pressure derivative in y direction
    dp_dz = @d_dz_2nd(P, i, j, k - 1) / dz

    # Update CPML memory arrays if on the boundary
    npml = size(mp_z, 3)
    if npml > 1
        ndim = size(P, 3)
        mv = @view mp_z[i, j, :, :]

        l, iBoundary = getBoundaryIndex(k, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view ah[:, iBoundary]
            b_ = @view bh[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dp_dz
            dp_dz += m[l]
        end
    end

    # Update velocity
    Vz[i, j, k] -= fv * dp_dz

    return nothing
end

# ============================================================================
# ==== Mask-aware pressure / velocity updates (staircase scatterer) ====
# Same as update_p! / update_p_cpml! / update_v*!, but a solid cell gets a
# zero update coefficient instead of being skipped. `update[i,j,k,channel]`
# is the scatterer mask — a 0/1 array (`channel = 1` pressure, 2/3/4 for
# vx/vy/vz; `build_update_mask` builds it as `Bool` then `Data.Array`-casts
# to the Float32 device type). The kernels fold it in branch-free as
# `coeff = f * mask`, keeping the loop body uniform across SIMD lanes — a
# data-dependent `if` here would not vectorise. `mask = 0` → `coeff = 0` →
# `field -= 0`: the cell is frozen, exactly what the old `if mask` skip did.
# ============================================================================

@parallel_indices (i, j, k) function update_p_mask!(field, fq, dx, dy, dz, update)
    P  = @view field[:, :, :, 1]
    Vx = @view field[:, :, :, 2]
    Vy = @view field[:, :, :, 3]
    Vz = @view field[:, :, :, 4]

    nx_ = size(field, 1)
    ny_ = size(field, 2)
    nz_ = size(field, 3)

    if i == nx_
        dv_dx = -2 .* Vx[i, j, k] / dx
    else
        dv_dx = @d_dx_2nd(Vx, i, j, k) / dx
    end
    if j == ny_
        dv_dy = -2 .* Vy[i, j, k] / dy
    else
        dv_dy = @d_dy_2nd(Vy, i, j, k) / dy
    end
    if k == nz_
        dv_dz = -2 .* Vz[i, j, k] / dz
    else
        dv_dz = @d_dz_2nd(Vz, i, j, k) / dz
    end

    # Branch-free staircase mask (0/1) → coeff 0 inside the scatterer.
    coeff = fq * update[i, j, k, 1]
    P[i, j, k] -= coeff * (dv_dx + dv_dy + dv_dz)
    return nothing
end

@parallel_indices (i, j, k) function update_p_cpml_mask!(
    field, fq, dx, dy, dz, update,
    mv_x, mv_y, mv_z, a, b,
)
    P  = @view field[:, :, :, 1]
    Vx = @view field[:, :, :, 2]
    Vy = @view field[:, :, :, 3]
    Vz = @view field[:, :, :, 4]

    dv_dx = @d_dx_2nd(Vx, i, j, k) / dx
    dv_dy = @d_dy_2nd(Vy, i, j, k) / dy
    dv_dz = @d_dz_2nd(Vz, i, j, k) / dz

    # x: left and right
    npml = size(mv_x, 1)
    ndim = size(P, 1)
    mv = @view mv_x[:, j, k, :]
    l, iBoundary = getBoundaryIndex(i, ndim, npml)
    if iBoundary > 0
        m  = @view mv[:, iBoundary]
        a_ = @view a[:, iBoundary]
        b_ = @view b[:, iBoundary]
        m[l] = b_[l] * m[l] + a_[l] * dv_dx
        dv_dx += m[l]
    end

    # y: left and right
    npml = size(mv_y, 2)
    ndim = size(P, 2)
    mv = @view mv_y[i, :, k, :]
    l, iBoundary = getBoundaryIndex(j, ndim, npml)
    if iBoundary > 0
        m  = @view mv[:, iBoundary]
        a_ = @view a[:, iBoundary]
        b_ = @view b[:, iBoundary]
        m[l] = b_[l] * m[l] + a_[l] * dv_dy
        dv_dy += m[l]
    end

    # z: left and right
    npml = size(mv_z, 3)
    ndim = size(P, 3)
    mv = @view mv_z[i, j, :, :]
    l, iBoundary = getBoundaryIndex(k, ndim, npml)
    if iBoundary > 0
        m  = @view mv[:, iBoundary]
        a_ = @view a[:, iBoundary]
        b_ = @view b[:, iBoundary]
        m[l] = b_[l] * m[l] + a_[l] * dv_dz
        dv_dz += m[l]
    end

    # CPML memory above only fires where `iBoundary > 0` (the PML layers);
    # the scatterer sits in the interior, so a solid cell never touches it.
    # Branch-free staircase mask (0/1) → coeff 0 inside the scatterer.
    coeff = fq * update[i, j, k, 1]
    P[i, j, k] -= coeff * (dv_dx + dv_dy + dv_dz)
    return nothing
end

@parallel_indices (i, j, k) function update_vx_mask!(
    field,
    dx,
    update,
    fv,
    mp_x,
    ah,
    bh,
)
    P = @view field[:, :, :, 1]
    Vx = @view field[:, :, :, 2]

    # Compute pressure derivative in x direction
    dp_dx = @d_dx_2nd(P, i - 1, j, k) / dx

    # Update CPML memory arrays if on the boundary (matches update_vx!'s
    # `npml > 1` guard — skips the CPML scaffolding entirely when npml = 0).
    npml = size(mp_x, 1)
    if npml > 1
        ndim = size(P, 1)
        mv = @view mp_x[:, j, k, :]

        l, iBoundary = getBoundaryIndex(i, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view ah[:, iBoundary]
            b_ = @view bh[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dp_dx
            dp_dx += m[l]
        end
    end

    # Branch-free mask (0/1): closed face → coeff 0 → velocity frozen at 0.
    coeff = fv * update[i, j, k, 2]
    Vx[i, j, k] -= coeff * dp_dx

    return nothing
end

@parallel_indices (i, j, k) function update_vy_mask!(
    field,
    dy,
    update,
    fv,
    mp_y,
    ah,
    bh,
)
    P = @view field[:, :, :, 1]
    Vy = @view field[:, :, :, 3]

    # Compute pressure derivative in y direction
    dp_dy = @d_dy_2nd(P, i, j - 1, k) / dy

    # Update CPML memory arrays if on the boundary (matches update_vy!'s
    # `npml > 1` guard — skips the CPML scaffolding entirely when npml = 0).
    npml = size(mp_y, 2)
    if npml > 1
        ndim = size(P, 2)
        mv = @view mp_y[i, :, k, :]

        l, iBoundary = getBoundaryIndex(j, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view ah[:, iBoundary]
            b_ = @view bh[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dp_dy
            dp_dy += m[l]
        end
    end

    # Branch-free mask (0/1): closed face → coeff 0 → velocity frozen at 0.
    coeff = fv * update[i, j, k, 3]
    Vy[i, j, k] -= coeff * dp_dy

    return nothing
end

@parallel_indices (i, j, k) function update_vz_mask!(
    field,
    dz,
    update,
    fv,
    mp_z,
    ah,
    bh,
)
    P = @view field[:, :, :, 1]
    Vz = @view field[:, :, :, 4]

    # Compute pressure derivative in z direction
    dp_dz = @d_dz_2nd(P, i, j, k - 1) / dz

    # Update CPML memory arrays if on the boundary (matches update_vz!'s
    # `npml > 1` guard — skips the CPML scaffolding entirely when npml = 0).
    npml = size(mp_z, 3)
    if npml > 1
        ndim = size(P, 3)
        mv = @view mp_z[i, j, :, :]

        l, iBoundary = getBoundaryIndex(k, ndim, npml)

        if iBoundary > 0
            m = @view mv[:, iBoundary]
            a_ = @view ah[:, iBoundary]
            b_ = @view bh[:, iBoundary]

            m[l] = b_[l] * m[l] + a_[l] * dp_dz
            dp_dz += m[l]
        end
    end

    # Branch-free mask (0/1): closed face → coeff 0 → velocity frozen at 0.
    coeff = fv * update[i, j, k, 4]
    Vz[i, j, k] -= coeff * dp_dz

    return nothing
end
