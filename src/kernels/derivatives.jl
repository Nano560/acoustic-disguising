# -----------------------------------------------------------------------------
# Derivative macros, off-grid trilinear interpolation, vector ↔ vn projection
# helpers, and the staggered CPML boundary index lookup.
#
# Building blocks for the FDTD pressure/velocity kernels in `updates.jl`,
# `extrapolation.jl`, and `forward.jl`. Included into the AcousticDisguising
# module — ParallelStencil imports live there.
# -----------------------------------------------------------------------------

# ============================================================================
# ==== derivative macros ====
# ============================================================================

macro d_dx_2nd(a, i, j, k)
    return esc(:(($a[$i+1, $j, $k] - $a[$i, $j, $k])))
end

macro d_dy_2nd(a, i, j, k)
    return esc(:(($a[$i, $j+1, $k] - $a[$i, $j, $k])))
end

macro d_dz_2nd(a, i, j, k)
    return esc(:(($a[$i, $j, $k+1] - $a[$i, $j, $k])))
end

# ============================================================================
# ==== off-grid source/receiver interpolation ====
# ============================================================================

@parallel_indices (is) function interp_trilinear!(
    field, # field
    it, # time index
    indices, # indices
    weights, # weights
    src, # time function
    rec, # time function
)
    I = Int(indices[1, is])
    J = Int(indices[2, is])
    K = Int(indices[3, is])

    # check if indices are within bounds (not on boundary)
    inBounds = I >= 1 && I < size(field, 1) &&
               J >= 1 && J < size(field, 2) &&
               K >= 1 && K < size(field, 3)

    if inBounds
        w = @view weights[:, :, :, is]
        f = @view field[I:I+1, J:J+1, K:K+1]

        for (i, j, k) in Iterators.product(1:2, 1:2, 1:2)

            rec[it, is] += w[i, j, k] * f[i, j, k]

            f[i, j, k] += w[i, j, k] * src[it, is]
        end
    end

    return nothing
end

# ============================================================================
# ==== CPML boundary index helper ====
# ============================================================================

function getBoundaryIndex(i, ndim, npml)
    iBoundary = 0
    ii = 0
    if i <= npml
        iBoundary = 1
        ii = i
    else
        ii = i - (ndim - npml)
        if ii > 0
            iBoundary = 2
        end
    end
    return ii, iBoundary
end

# ============================================================================
# ==== vector ↔ normal-velocity projection ====
# ============================================================================

@parallel_indices (i) function v2vn!(vx, vy, vz, vn, normals, it)
    vn[it, i] = vx[it, i] * normals[2, i] +
                vy[it, i] * normals[3, i] +
                vz[it, i] * normals[4, i]
    return nothing
end

@parallel_indices (i) function vn2v!(vx, vy, vz, vn, normals, it)
    vx[it, i] += vn[it, i] * normals[2, i]
    vy[it, i] += vn[it, i] * normals[3, i]
    vz[it, i] += vn[it, i] * normals[4, i]
    return nothing
end
