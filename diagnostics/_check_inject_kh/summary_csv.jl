println()
printstyled("══════ Part 3 summary ══════\n"; bold = true)
@printf("%-6s | %-9s %-9s | %-9s %-9s | %-13s %-13s\n",
        "N", "pv:p_err", "pv:vn_err", "pin:p_err", "pin:vn_err", "rms_resid_KH", "Δrms (KH-rec)")
println(repeat("─", 94))
for r in _PART3_RESULTS
    @printf("%-6d | %-9.4g %-9.4g | %-9.4g %-9.4g | %-13.4g %-+13.4g\n",
            r.N, r.p_err_global, r.vn_err_global,
            r.p_err_global_pin, r.vn_err_global_pin,
            r.rms_residual_kh, r.Δrms_residual)
end
# ---- Append all Part 2 + Part 3 rows to a CSV in DIAG_DIR so multi-run sweeps
# accumulate in one file (re-readable by Python/Julia/pandas for plotting
# convergence). One row per N. Part 3 columns are NaN if N ∉ NS_PART3.
let
    csv_path = joinpath(DIAG_DIR, "sweep_results.csv")
    header = "timestamp,C_formula,medium,rho,c,z0,L,dx,cf,dt,nt,fc_hz,init_dist_m,mask_taper_frac,kernel_mode," *
             "N,cps,rms_A,rms_B,rms_residual,suppression," *
             "pv_p_err,pv_vn_err,pin_p_err,pin_vn_err,rms_residual_kh,delta_rms_residual,suppression_kh\n"
    is_new = !isfile(csv_path)
    ts = string(now())
    part3_by_N = Dict(r.N => r for r in _PART3_RESULTS)
    open(csv_path, "a") do io
        is_new && write(io, header)
        for r2 in _PART2_RESULTS
            N = r2.N
            cps = sqrt(4π * R_INNER^2 / N) / dom_pw.dx
            r3 = get(part3_by_N, N, nothing)
            pv_p   = r3 === nothing ? NaN : r3.p_err_global
            pv_vn  = r3 === nothing ? NaN : r3.vn_err_global
            pin_p  = r3 === nothing ? NaN : r3.p_err_global_pin
            pin_vn = r3 === nothing ? NaN : r3.vn_err_global_pin
            rms_kh   = r3 === nothing ? NaN : r3.rms_residual_kh
            Δrms_    = r3 === nothing ? NaN : r3.Δrms_residual
            supp_kh  = r3 === nothing ? NaN : r3.suppression_kh
            @printf(io, "%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g,%s,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                    ts, C_FORMULA,
                    MEDIUM, RHO, C0, dom_pw.z0,
                    L, dom_pw.dx, dom_pw.cf, dom_pw.dt, dom_pw.nt, FC_HZ, INIT_DIST,
                    MASK_TAPER_FRAC,
                    KERNEL_MODE_STR,
                    N, cps,
                    r2.rms_A, r2.rms_B, r2.rms_residual, r2.suppression,
                    pv_p, pv_vn, pin_p, pin_vn, rms_kh, Δrms_, supp_kh)
        end
    end
    @info "Appended sweep results to CSV" csv_path is_new
end

println()
println("Interpretation (Part 3):")
println("  Layer (a): pv:p_err, pv:vn_err, pin:p_err, pin:vn_err → GLOBAL rel-RMS")
println("             errors (‖KH-rec‖/‖rec‖) for the analytical K-H extrapolation;")
println("             ≪ 1 means the kernels + K-H formula reproduce the recorded inner field.")
println("  Layer (b): rms_resid_KH = ‖slab_A + slab_B_KH‖_inner_disk after running")
println("             FDTD with the K-H-extrapolated inner field as injection source.")
println("             Δrms = rms_resid_KH − rms_residual(Part 2) measures how much")
println("             the K-H extrapolation step shifts the cancellation residual.")
