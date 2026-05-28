"""Dimensional consistency: Barton -> Eq.(1) -> eq:kh -> eq:C-coef.

Programmatic assertion of the unit identities derived in the Barton-equivalence
section of diagnostics/_check_inject_kh/injection_coefficients.jl. Resolves the
SI TODO at lib/sup.tex:~267 of the Overleaf project (companion paper
"Acoustic disguising: a unified framework for cloaking and holography").

Checks (each `test_*` function below):
  1. Kernel identifications G^{p|q} = -rho * partial_t G_bare,
     G^{p|f} = -partial_n G_bare have the correct units.
  2. Eq.(1) integrand pieces, after integration over dt' and dS',
     balance to pressure (Pa) on the LHS.
  3. Discrete Yee prefactors K_p = rho c^2 dt and K_v = dt/rho realise
     the same dimensional structure as the continuous kernels; the
     boxed C_q, C_f reproduce per-step pressure/velocity increments,
     and C_q/C_f has the impedance-squared dimensions z_0^2 = (rho c)^2.

If any test fails, a factor is silently missing or double-counted somewhere
in the chain Barton -> Eq.(1) -> eq:kh -> eq:C-coef. As of 2026-05-27 all
three tests pass and the SI TODO is RESOLVED.
"""
import pint

ureg = pint.UnitRegistry()
m, s, kg = ureg.meter, ureg.second, ureg.kilogram
Pa = kg / (m * s ** 2)


def test_kernel_identifications_have_correct_units():
    """G^{p|q} = -rho * partial_t G_bare  ->  units kg / (m^4 s^2)
    G^{p|f} = -partial_n G_bare           ->  units 1 / (s m^2)
    (signs are dimensionally irrelevant; only magnitudes are checked.)
    """
    dt_G = 1 / (s ** 2 * m)               # partial_t of the bare 3D
    dn_G = 1 / (s * m ** 2)               # retarded Helmholtz GF
    rho = kg / m ** 3

    Gpq = rho * dt_G                      # = -rho * partial_t G_bare
    Gpf = dn_G                            # = -partial_n G_bare

    assert Gpq.dimensionality == (kg / (m ** 4 * s ** 2)).dimensionality
    assert Gpf.dimensionality == (1 / (s * m ** 2)).dimensionality


def test_eq1_integrand_balances_to_pressure():
    """Eq.(1) of the Letter:
       p(x,t) = integral dt' dS' [G^{p|q} v + G^{p|f} p].
    After integrating each integrand over dt' (units s) and dS' (units m^2),
    both terms must give Pa on the RHS (matching LHS [p] = Pa).
    """
    Gpq = (kg / m ** 3) * (1 / (s ** 2 * m))    # kg / (m^4 s^2)
    Gpf = 1 / (s * m ** 2)
    v = m / s
    p = Pa
    dt = s
    dS = m ** 2

    assert (Gpq * v * dt * dS).dimensionality == p.dimensionality
    assert (Gpf * p * dt * dS).dimensionality == p.dimensionality


def test_discrete_Yee_prefactors_realise_same_structure():
    """The discrete C_q = rho c^2 dt * dS_in / dx^3 and
                C_f = dt/rho   * dS_in / dx^3
    must:
      (a) produce per-step pressure increment when multiplied by v;
      (b) produce per-step velocity increment when multiplied by p;
      (c) satisfy the impedance-squared identity C_q / C_f = z_0^2 = (rho c)^2.
    """
    rho = kg / m ** 3
    c = m / s
    dt = s
    dx = m
    dS_in = m ** 2

    K_p = rho * c ** 2 * dt               # pressure-channel Yee prefactor
    K_v = dt / rho                        # velocity-channel Yee prefactor

    C_q = K_p * dS_in / dx ** 3
    C_f = K_v * dS_in / dx ** 3

    v = m / s
    p = Pa

    # (a) and (b): discrete injections give the right channel increment
    assert (C_q * v).dimensionality == p.dimensionality   # Pa per step
    assert (C_f * p).dimensionality == v.dimensionality   # m/s per step

    # (c) impedance-squared identity
    assert (C_q / C_f).dimensionality == ((rho * c) ** 2).dimensionality