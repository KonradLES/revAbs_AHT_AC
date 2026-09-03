"""LiBr/H2O property functions for the AWT simulation.

Conventions
-----------
- x : LiBr mole fraction in the solution [-]
- w : LiBr mass fraction in the solution [-]
- T : temperature [K]
- p : pressure [Pa]
- h : specific enthalpy of the solution [kJ/kg]
- s : specific entropy of the solution [kJ/kg/K]
- rho : density of the solution [kg/m^3]

The implementation uses the Patek correlations. Only mass-based quantities
with consistent units are exposed externally, for use by the simulation
model.
"""

from __future__ import annotations

import math
import CoolProp.CoolProp as CP
from dataclasses import dataclass
import numpy

from scipy.optimize import newton, root_scalar

M_LIBR = 0.08685  # kg/mol
M_H2O = 0.018015268  # kg/mol

T_MIN_PAT = 273.15
T_MAX_PAT = 500.0
X_MIN_PAT = 1.0e-9
X_MAX_PAT = 0.399999  # Patek-Formulierungen enthalten den Faktor (0.4 - x)


class PropertyError(RuntimeError):
    """Error during property or inversion calculations."""


@dataclass(frozen=True)
class StateValidity:
    in_patek_range: bool
    crystallization_checked: bool
    crystallization_safe: bool
    message: str


def _validate_x_patek_range(x_libr_mol: float, *, function_name: str) -> float:
    """Checks the LiBr mole fraction against the validity range of the Patek correlations."""
    x = float(x_libr_mol)
    if not (X_MIN_PAT <= x <= X_MAX_PAT):
        raise PropertyError(
            f"{function_name}: LiBr mole fraction x={x:.9f} is outside the implemented "
            f"Patek validity range [{X_MIN_PAT:.9f}, {X_MAX_PAT:.6f}] [-]. "
            f"The correlations used contain the term (0.4 - x) and are only evaluated "
            f"for x < 0.4 in this implementation."
        )
    return x



def _validate_T_patek_range(T: float, *, function_name: str) -> float:
    """Checks the temperature against the validity range of the Patek correlations."""
    T_val = float(T)
    if not (T_MIN_PAT <= T_val <= T_MAX_PAT):
        raise PropertyError(
            f"{function_name}: temperature T={T_val:.6f} K is outside the implemented "
            f"Patek validity range [{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K."
        )
    return T_val


# ---------------------------------------------------------------------------
# Concentration and molar mass relationships
# ---------------------------------------------------------------------------

def mixture_molar_mass(x_libr_mol: float) -> float:
    x = _validate_x_patek_range(x_libr_mol, function_name="mixture_molar_mass")
    return x * M_LIBR + (1.0 - x) * M_H2O



def w_libr_from_x(x_libr_mol: float) -> float:
    """LiBr mass fraction w from LiBr mole fraction x."""
    x = _validate_x_patek_range(x_libr_mol, function_name="w_libr_from_x")
    return x * M_LIBR / mixture_molar_mass(x)



def x_from_w_libr(w_libr: float) -> float:
    """LiBr mole fraction x from LiBr mass fraction w."""
    w = float(w_libr)
    denominator = M_LIBR - w * (M_LIBR - M_H2O)
    return (w * M_H2O) / denominator


# ---------------------------------------------------------------------------
# Direct Patek functions (molar basis)
# ---------------------------------------------------------------------------

def calc_cp_molar_patek(T: float, x_libr_mol: float) -> float:
    """Molar heat capacity of the LiBr/H2O solution [J/mol/K]."""
    T = _validate_T_patek_range(T, function_name="calc_cp_molar_patek")
    cp_t = 76.0226
    T_c = 647.096
    T_t = 273.16

    koef_a = [-1.42094e1, 4.04943e1, 1.11135e2, 2.29980e2, 1.34526e3, -1.41010e-2, 1.24977e-2, -6.83209e-4]
    koef_t = [0, 0, 0, 0, 0, 2, 3, 4]
    koef_n = [0, 0, 1, 2, 3, 0, 3, 2]
    koef_m = [2, 3, 3, 3, 3, 2, 1, 1]

    koef_beta = [0, 2, 3, 6, 34]
    koef_gamma = [0, 2, 3, 5, 0]
    koef_alpha = [1.38801, -2.95318, 3.18721, -0.645473, 9.18946e5]

    x = _validate_x_patek_range(x_libr_mol, function_name="calc_cp_molar_patek")

    cp_sat = cp_t * sum(
        koef_alpha[i] * (1.0 - T / T_c) ** koef_beta[i] * (T / T_t) ** koef_gamma[i]
        for i in range(5)
    )

    correction = 0.0
    for a, t, n, m in zip(koef_a, koef_t, koef_n, koef_m):
        correction += a * x**m * (0.4 - x) ** n * (T_c / (T - 221.0)) ** t

    return (1.0 - x) * cp_sat + cp_t * correction



def calc_h_molar_patek(T: float, x_libr_mol: float) -> float:
    """Molar enthalpy of the LiBr/H2O solution [J/mol]."""
    T = _validate_T_patek_range(T, function_name="calc_h_molar_patek")
    T_c = 647.096
    h_c = 37548.5

    koef_a = [
        2.27431e0, -7.99511e0, 3.85239e2, -1.63940e4, -4.22562e2,
        1.13314e-1, -8.33474e0, -1.73833e4, 6.49763e0, 3.24552e3,
        -1.34643e4, 3.99322e4, -2.58877e5, -1.93046e-3, 2.80616e0,
        -4.04479e1, 1.45342e2, -2.74873e0, -4.49743e2, -1.21794e1,
        -5.83739e-3, 2.33910e-1, 3.41888e-1, 8.85259e0, -1.78731e1,
        7.35179e-2, -1.79430e-4, 1.84261e-3, -6.24282e-3, 6.84765e-3,
    ]
    koef_t = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5]
    koef_n = [0, 1, 6, 6, 2, 0, 0, 4, 0, 4, 5, 5, 6, 0, 3, 5, 7, 0, 3, 1, 0, 4, 2, 6, 7, 0, 0, 1, 2, 3]
    koef_m = [1, 1, 2, 3, 6, 1, 3, 5, 4, 5, 5, 6, 6, 1, 2, 2, 2, 5, 6, 7, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1]

    koef_beta = [1.0 / 3.0, 2.0 / 3.0, 5.0 / 6.0, 21.0 / 6.0]
    koef_alpha = [-4.37196e-1, 3.03440e-1, -1.29582e0, -1.76410e-1]

    x = _validate_x_patek_range(x_libr_mol, function_name="calc_h_molar_patek")

    h_sat = h_c * (1.0 + sum(koef_alpha[i] * (1.0 - T / T_c) ** koef_beta[i] for i in range(4)))

    grouped: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    for a, t, n, m in zip(koef_a, koef_t, koef_n, koef_m):
        grouped[t] += a * x**m * (0.4 - x) ** n

    poly = sum(grouped[t] * (T_c / (T - 221.0)) ** t for t in range(6))
    return (1.0 - x) * h_sat + h_c * poly



def calc_p_sat_patek(T: float, x_libr_mol: float) -> float:
    """Saturation pressure of the LiBr/H2O solution [Pa]."""
    T = _validate_T_patek_range(T, function_name="calc_p_sat_patek")
    T_c = 647.096
    p_c = 22.064e6

    koef_a = [-2.41303e2, 1.91750e7, -1.75521e8, 3.25430e7, 3.92571e2, -2.12626e3, 1.85127e8, 1.91216e3]
    koef_t = [0, 0, 0, 0, 1, 1, 1, 1]
    koef_n = [0, 5, 6, 3, 0, 2, 6, 0]
    koef_m = [3, 4, 4, 8, 1, 1, 4, 6]

    koef_beta = [1.0, 1.5, 3.0, 3.5, 4.0, 7.5]
    koef_alpha = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502]

    x = _validate_x_patek_range(x_libr_mol, function_name="calc_p_sat_patek")

    theta = T - sum(
        a * x**m * (0.4 - x) ** n * (T / T_c) ** t
        for a, t, n, m in zip(koef_a, koef_t, koef_n, koef_m)
    )
    exponent = sum(koef_alpha[i] * (1.0 - theta / T_c) ** koef_beta[i] for i in range(6))
    return p_c * math.exp((T_c / theta) * exponent)



def calc_rho_molar_patek(T: float, x_libr_mol: float) -> float:
    """Molar density of the LiBr/H2O solution [mol/m^3]."""
    T = _validate_T_patek_range(T, function_name="calc_rho_molar_patek")
    T_c = 647.096
    rho_c = 17.873

    koef_a = [1.746, 4.709]
    koef_t = [0, 6]
    koef_m = [1, 1]

    koef_beta = [1.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0, 16.0 / 3.0, 43.0 / 3.0, 110.0 / 3.0]
    koef_alpha = [1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -6.7469445e5]

    x = _validate_x_patek_range(x_libr_mol, function_name="calc_rho_molar_patek")

    rho_sat = rho_c * (1.0 + sum(koef_alpha[i] * (1.0 - T / T_c) ** koef_beta[i] for i in range(6)))
    rho = (1.0 - x) * rho_sat + rho_c * sum(
        koef_a[i] * x**koef_m[i] * (T / T_c) ** koef_t[i]
        for i in range(2)
    )
    return rho * 1000.0

def calc_s_molar_patek(T: float, x_libr_mol: float) -> float:
    """Molar entropy of the LiBr/H2O solution [J/mol/K]."""
    T = _validate_T_patek_range(T, function_name="calc_s_molar_patek")
    x = _validate_x_patek_range(x_libr_mol, function_name="calc_s_molar_patek")
    T_c = 647.096              #[K]
    s_c = 79.3933              #[J/molK]
    T_0 = 221                  #[K]

    # Table 8
    koef_a = [  1.53091     *   10**0,
                -4.52564    *   10**0,
                6.98302     *   10**2,
                -2.1666     *   10**4,
                -1.47533    *   10**3,
                8.47012     *   10**-2,
                -6.59523    *   10**0,
                -2.95331    *   10**4,
                9.56314     *   10**-3,
                -1.88679    *   10**-1,
                9.31752     *   10**0,
                5.78104     *   10**0,
                1.38931     *   10**4,
                -1.71762    *   10**4,
                4.15108     *   10**2,
                -5.55647    *   10**4,
                -4.23409    *   10**-3,
                3.05242     *   10**1,
                -1.67620    *   10**0,
                1.48283     *   10**1,
                3.03055     *   10**-3,
                -4.01810    *   10**-2,
                1.49252     *   10**-1,
                2.59240     *   10**0,
                -1.77421    *   10**-1,
                -6.99650    *   10**-5,
                6.05007     *   10**-4,
                -1.65228    *   10**-3,
                1.22966     *   10**-3]
    koef_t = [0,0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5]
    koef_n = [0,1,6,6,2,0,0,4,0,0,4,0,4,5,2,5,0,4,0,1,0,2,4,7,1,0,1,2,3]
    koef_m = [1,1,2,3,6,1,3,5,1,2,2,4,5,5,6,6,1,3,5,7,1,1,1,2,3,1,1,1,1]
    # Table 15
    koef_beta = [1/3,1,8/3,8]
    koef_alpha = [  -3.34112    *   10**-1,
                    -8.47987    *   10**-1,
                    -9.11980    *   10**-1,
                    -1.64046    *   10**0]

    # Calculation of s_sat
    s_sat = s_c * (1.0 + sum(koef_alpha[i] * (1.0 - T / T_c) ** koef_beta[i] for i in range(4)))

    # Calculation of s
    factors = numpy.zeros((29,))
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    for i in range(29):
        factors[i] =  koef_a[i]*x**koef_m[i]*(0.4-x)**koef_n[i]
        if koef_t[i] == 0:
            f = f + factors[i]
        elif koef_t[i] == 1:
            e = e + factors[i]
        elif koef_t[i] == 2:
            d = d + factors[i]
        elif koef_t[i] == 3:
            c = c + factors[i]
        elif koef_t[i] == 4:
            b = b + factors[i]
        elif koef_t[i] == 5:
            a = a + factors[i]
    s = (1-x)*s_sat + s_c*(a*(T_c/(T-T_0))**5 + b*(T_c/(T-T_0))**4 + c*(T_c/(T-T_0))**3 + d*(T_c/(T-T_0))**2 + e*(T_c/(T-T_0))**1 + f)
    return s

# ---------------------------------------------------------------------------
# Mass-based wrappers
# ---------------------------------------------------------------------------

def h_solution_mass_kjkg(T: float, x_libr_mol: float) -> float:
    """Specific enthalpy of the solution [kJ/kg]."""
    T = _validate_T_patek_range(T, function_name="h_solution_mass_kjkg")
    x = _validate_x_patek_range(x_libr_mol, function_name="h_solution_mass_kjkg")
    return calc_h_molar_patek(T, x) / mixture_molar_mass(x) / 1000.0



def s_solution_mass_kjkgK(T: float, x_libr_mol: float) -> float:
    """Specific entropy of the LiBr/H2O solution [kJ/kg/K]."""
    T = _validate_T_patek_range(T, function_name="s_solution_mass_kjkgK")
    x = _validate_x_patek_range(x_libr_mol, function_name="s_solution_mass_kjkgK")
    return calc_s_molar_patek(T, x) / mixture_molar_mass(x) / 1000.0



def cp_solution_mass_kjkgk(T: float, x_libr_mol: float) -> float:
    """Specific heat capacity of the solution [kJ/kg/K]."""
    T = _validate_T_patek_range(T, function_name="cp_solution_mass_kjkgk")
    x = _validate_x_patek_range(x_libr_mol, function_name="cp_solution_mass_kjkgk")
    return calc_cp_molar_patek(T, x) / mixture_molar_mass(x) / 1000.0



def rho_solution_mass(T: float, x_libr_mol: float) -> float:
    """Mass density of the solution [kg/m^3]."""
    T = _validate_T_patek_range(T, function_name="rho_solution_mass")
    x = _validate_x_patek_range(x_libr_mol, function_name="rho_solution_mass")
    return calc_rho_molar_patek(T, x) * mixture_molar_mass(x)



def T_sat_solution_from_p_x(p_pa: float, x_libr_mol: float) -> float:
    """Saturation temperature of the solution from pressure and LiBr mole fraction [K]."""
    x = _validate_x_patek_range(x_libr_mol, function_name="T_sat_solution_from_p_x")

    def fun(T: float) -> float:
        return calc_p_sat_patek(T, x) - p_pa

    try:
        sol = root_scalar(fun, bracket=[T_MIN_PAT + 1e-6, T_MAX_PAT - 1e-6], method="brentq")
    except ValueError as exc:
        raise PropertyError(
            f"T_sat_solution_from_p_x: no saturation temperature found in the implemented Patek "
            f"temperature range [{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K for p={p_pa:.3e} Pa and x={x:.9f}."
        ) from exc
    return _validate_T_patek_range(float(sol.root), function_name="T_sat_solution_from_p_x")



def T_from_h_x_mass(h_kjkg: float, x_libr_mol: float) -> float:
    """Temperature of the solution from specific enthalpy and LiBr mole fraction [K]."""
    x = _validate_x_patek_range(x_libr_mol, function_name="T_from_h_x_mass")

    def fun(T: float) -> float:
        return h_solution_mass_kjkg(T, x) - h_kjkg

    try:
        sol = root_scalar(fun, bracket=[T_MIN_PAT + 1e-6, T_MAX_PAT - 1e-6], method="brentq")
    except ValueError as exc:
        raise PropertyError(
            f"T_from_h_x_mass: no temperature found in the implemented Patek temperature range "
            f"[{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K for h={h_kjkg:.6f} kJ/kg and x={x:.9f}."
        ) from exc
    return _validate_T_patek_range(float(sol.root), function_name="T_from_h_x_mass")


def _flash_valve_state(
    p_out_pa: float,
    h_in_kJkg: float,
    m_in_kg_s: float,
    x_in_libr_mol: float,
) -> dict:
    """
    Isenthalpic flash throttle for LiBr/H2O (shared implementation).

    Assumptions
    -----------
    - The throttle is isenthalpic.
    - After the throttle, equilibrium at p_out is established.
    - The liquid phase is LiBr/H2O solution.
    - The vapor phase is pure water.
    - LiBr remains entirely in the liquid phase.

    Energy basis
    ------------
    h_in_kJkg is the specific enthalpy of the inlet stream ahead of the
    throttle, on a mass basis [kJ/kg]. Since the inlet stream consists
    entirely of solution, this is consistent with the mass-averaged outlet
    enthalpy h_mix_kJ_kg on a total-mass basis.

    Returns
    -------
    Dictionary with:
    - T_K
    - x_LiBr_mol
    - w_LiBr
    - m_sol_kg_s
    - m_flash_kg_s
    - h_sol_kJ_kg
    - h_flash_kJ_kg
    - h_mix_kJ_kg
    - flash_fraction
    """
    if p_out_pa <= 0.0:
        raise PropertyError(
            f"Flash throttle: p_out must be positive, got p_out={p_out_pa:.6e} Pa."
        )

    if m_in_kg_s <= 0.0:
        raise PropertyError(
            f"Flash throttle: m_in must be positive, got m_in={m_in_kg_s:.6f} kg/s."
        )

    x_in = float(x_in_libr_mol)
    if not (X_MIN_PAT <= x_in <= X_MAX_PAT):
        raise PropertyError(
            f"Flash throttle: x_in={x_in:.6f} is outside the allowed range "
            f"[{X_MIN_PAT:.6f}, {X_MAX_PAT:.6f}]."
        )

    w_in = w_libr_from_x(x_in)

    # Lower temperature bound: no flash, original composition on the saturation line
    T_lo = T_sat_solution_from_p_x(p_out_pa, x_in)

    # Upper temperature bound: strongly concentrated solution at the same pressure
    x_hi = X_MAX_PAT - 1.0e-4
    if x_hi <= x_in:
        x_hi = X_MAX_PAT - 1.0e-9

    T_hi = T_sat_solution_from_p_x(p_out_pa, x_hi)

    if T_hi <= T_lo:
        raise PropertyError(
            "Flash throttle: invalid temperature interval for the flash search. "
            f"T_lo={T_lo:.6f} K, T_hi={T_hi:.6f} K."
        )

    # Tolerances for numerical edge cases
    T_tol = 1.0e-8
    p_tol = 1.0e-6
    w_tol = 1.0e-10
    m_tol = 1.0e-12

    def x_eq_from_T(T_K: float) -> float:
        """
        Equilibrium liquid-phase composition x_out at a given temperature T_K
        and pressure p_out_pa.
        """
        if not (T_MIN_PAT <= T_K <= T_MAX_PAT):
            raise PropertyError(
                f"Flash throttle: T={T_K:.6f} K is outside the allowed range "
                f"[{T_MIN_PAT:.6f}, {T_MAX_PAT:.6f}] K."
            )

        # Explicit handling of the no-flash edge case
        if T_K <= T_lo + T_tol:
            return x_in

        def f_x(x_val: float) -> float:
            return calc_p_sat_patek(T_K, x_val) - p_out_pa

        f_left = f_x(x_in)
        if abs(f_left) <= p_tol:
            return x_in

        x_right = X_MAX_PAT - 1.0e-9
        f_right = f_x(x_right)

        if f_left * f_right > 0.0:
            raise PropertyError(
                "Flash throttle: no equilibrium composition x found in the interval "
                f"[x_in, X_MAX_PAT] at T={T_K:.6f} K and p={p_out_pa:.3e} Pa. "
                f"f(x_in)={f_left:.6e}, f(x_max)={f_right:.6e}."
            )

        try:
            sol_x = root_scalar(
                f_x,
                bracket=[x_in, x_right],
                method="brentq",
            )
        except ValueError as exc:
            raise PropertyError(
                f"Flash throttle: no equilibrium composition x found at "
                f"T={T_K:.6f} K and p={p_out_pa:.3e} Pa."
            ) from exc

        return float(sol_x.root)

    def build_state(T_K: float, strict: bool = False) -> dict:
        # Explicit no-flash edge case
        if T_K <= T_lo + T_tol:
            x_out = x_in
            w_out = w_in
            m_out_sol = m_in_kg_s
            m_out_flash = 0.0
            h_out_sol = h_solution_mass_kjkg(T_K, x_out)
            h_out_flash = CP.PropsSI("H", "T", T_K, "P", p_out_pa, "Water") / 1000.0
            h_out_mix = h_out_sol

            return {
                "T_K": T_K,
                "x_LiBr_mol": x_out,
                "w_LiBr": w_out,
                "m_sol_kg_s": m_out_sol,
                "m_flash_kg_s": m_out_flash,
                "h_sol_kJ_kg": h_out_sol,
                "h_flash_kJ_kg": h_out_flash,
                "h_mix_kJ_kg": h_out_mix,
                "flash_fraction": 0.0,
            }

        x_out = x_eq_from_T(T_K)
        w_out = w_libr_from_x(x_out)

        # Near the flash boundary, numerical noise can make w_out minimally smaller than w_in.
        if w_out < w_in:
            if (w_in - w_out) <= w_tol:
                x_out = x_in
                w_out = w_in
            elif strict:
                raise PropertyError(
                    "Flash throttle: the liquid solution after the throttle would be more "
                    f"dilute than the inlet state. w_out={w_out:.12f}, w_in={w_in:.12f}."
                )
            else:
                x_out = x_in
                w_out = w_in

        # LiBr balance: m_in * w_in = m_out_sol * w_out
        m_out_sol = m_in_kg_s * w_in / w_out
        m_out_flash = m_in_kg_s - m_out_sol

        if m_out_flash < 0.0:
            if abs(m_out_flash) <= m_tol:
                m_out_flash = 0.0
            elif strict:
                raise PropertyError(
                    f"Flash throttle: computed negative flash mass flow: "
                    f"m_flash={m_out_flash:.12e} kg/s."
                )
            else:
                m_out_flash = 0.0

        h_out_sol = h_solution_mass_kjkg(T_K, x_out)

        # Pure water vapor at (T_K, p_out_pa)
        h_out_flash = CP.PropsSI("H", "T", T_K, "P", p_out_pa, "Water") / 1000.0

        h_out_mix = (m_out_sol * h_out_sol + m_out_flash * h_out_flash) / m_in_kg_s

        return {
            "T_K": T_K,
            "x_LiBr_mol": x_out,
            "w_LiBr": w_out,
            "m_sol_kg_s": m_out_sol,
            "m_flash_kg_s": m_out_flash,
            "h_sol_kJ_kg": h_out_sol,
            "h_flash_kJ_kg": h_out_flash,
            "h_mix_kJ_kg": h_out_mix,
            "flash_fraction": m_out_flash / m_in_kg_s,
        }

    def residual(T_K: float) -> float:
        state = build_state(T_K, strict=False)
        return state["h_mix_kJ_kg"] - h_in_kJkg

    r_lo = residual(T_lo)
    h_tol = 1.0e-9

    if abs(r_lo) <= h_tol:
        return build_state(T_lo, strict=True)

    # -------------------------------------------------------------------------
    # True no-flash case, equivalent to the Modelica implementation
    # -------------------------------------------------------------------------
    #
    # Previously the root search started at T_lo, i.e. at the boiling
    # temperature of the inlet solution x_in at p_out.
    #
    # If h_in_kJkg is smaller than the liquid enthalpy at that point, the
    # solution cannot flash after the throttle. It stays subcooled:
    #
    #   Q = 0
    #   x_out = x_in
    #   w_out = w_in
    #   m_flash = 0
    #
    # Then T_out must be determined from the isenthalpic liquid condition:
    #
    #   h_solution_mass_kjkg(T_out, x_in) = h_in
    #
    # This corresponds exactly to the Modelica branch:
    #
    #   if Q_intern <= 0 then
    #       Q = 0;
    #       X_LiBr_out = X_LiBr_in;
    #       h_out = h_solution(T_out, X_LiBr_out, p_out);
    #   end if;
    #
    # In residual form:
    #
    #   r_lo = h_solution(T_lo, x_in) - h_in
    #
    # If r_lo > 0, h_in is too low for a saturated/flashing state.
    # We then search for T_out below T_lo.
    # -------------------------------------------------------------------------
    if r_lo > 0.0:

        def residual_no_flash(T_K: float) -> float:
            return h_solution_mass_kjkg(T_K, x_in) - h_in_kJkg

        T_no_flash_lo = T_MIN_PAT
        r_no_flash_lo = residual_no_flash(T_no_flash_lo)

        if abs(r_no_flash_lo) <= h_tol:
            return build_state(T_no_flash_lo, strict=True)

        if r_no_flash_lo * r_lo > 0.0:
            raise PropertyError(
                "Flash throttle: no no-flash state found in the temperature interval "
                f"[{T_no_flash_lo:.6f}, {T_lo:.6f}] K. "
                "The inlet enthalpy is presumably outside the valid enthalpy range "
                "of the LiBr/H2O correlation. "
                f"Residual at T_min = {r_no_flash_lo:.6e} kJ/kg, "
                f"residual at T_lo = {r_lo:.6e} kJ/kg."
            )

        try:
            sol_T_no_flash = root_scalar(
                residual_no_flash,
                bracket=[T_no_flash_lo, T_lo],
                method="brentq",
            )
        except ValueError as exc:
            raise PropertyError(
                "Flash throttle: root search for the no-flash case failed. "
                f"Search interval: T_min={T_no_flash_lo:.6f} K, "
                f"T_lo={T_lo:.6f} K."
            ) from exc

        return build_state(float(sol_T_no_flash.root), strict=True)

    # From here on r_lo < 0:
    # The inlet enthalpy exceeds the enthalpy of the saturated inlet solution
    # at p_out, so a flash state must be found.
    r_hi = residual(T_hi)
    if abs(r_hi) <= h_tol:
        return build_state(T_hi, strict=True)

    if r_lo * r_hi > 0.0:
        raise PropertyError(
            "Flash throttle: no sign change of the enthalpy residual in the "
            f"temperature interval [{T_lo:.6f}, {T_hi:.6f}] K. "
            f"Residual at lower bound = {r_lo:.6e} kJ/kg, "
            f"residual at upper bound = {r_hi:.6e} kJ/kg."
        )

    try:
        sol_T = root_scalar(
            residual,
            bracket=[T_lo, T_hi],
            method="brentq",
        )
    except ValueError as exc:
        raise PropertyError(
            "Flash throttle: root search for the isenthalpic flash condition failed. "
            f"Search interval: T_lo={T_lo:.6f} K, T_hi={T_hi:.6f} K."
        ) from exc

    return build_state(float(sol_T.root), strict=True)


def flash_valve_state_2_to_1(
    p_out_pa: float,
    h2_kJkg: float,
    m2_kg_s: float,
    x2_libr_mol: float,
    ) -> dict:
    """Isenthalpic flash throttle 2 -> 1 for LiBr/H2O.

    See _flash_valve_state() for the model assumptions and energy basis.

    Returns
    -------
    Dictionary with:
    - T1_K
    - x1_LiBr_mol
    - w1_LiBr
    - m1_sol_kg_s
    - m1_flash_kg_s
    - h1_sol_kJ_kg
    - h1_flash_kJ_kg
    - h1_mix_kJ_kg
    - flash_fraction
    """
    state = _flash_valve_state(p_out_pa, h2_kJkg, m2_kg_s, x2_libr_mol)
    return {
        "T1_K": state["T_K"],
        "x1_LiBr_mol": state["x_LiBr_mol"],
        "w1_LiBr": state["w_LiBr"],
        "m1_sol_kg_s": state["m_sol_kg_s"],
        "m1_flash_kg_s": state["m_flash_kg_s"],
        "h1_sol_kJ_kg": state["h_sol_kJ_kg"],
        "h1_flash_kJ_kg": state["h_flash_kJ_kg"],
        "h1_mix_kJ_kg": state["h_mix_kJ_kg"],
        "flash_fraction": state["flash_fraction"],
    }


def flash_valve_state_5_to_6(
    p_out_pa: float,
    h5_kJkg: float,
    m5_kg_s: float,
    x5_libr_mol: float,
    ) -> dict:
    """Isenthalpic flash throttle 5 -> 6 for LiBr/H2O.

    See _flash_valve_state() for the model assumptions and energy basis.

    Returns
    -------
    Dictionary with:
    - T6_K
    - x6_LiBr_mol
    - w6_LiBr
    - m6_sol_kg_s
    - m6_flash_kg_s
    - h6_sol_kJ_kg
    - h6_flash_kJ_kg
    - h6_mix_kJ_kg
    - flash_fraction
    """
    state = _flash_valve_state(p_out_pa, h5_kJkg, m5_kg_s, x5_libr_mol)
    return {
        "T6_K": state["T_K"],
        "x6_LiBr_mol": state["x_LiBr_mol"],
        "w6_LiBr": state["w_LiBr"],
        "m6_sol_kg_s": state["m_sol_kg_s"],
        "m6_flash_kg_s": state["m_flash_kg_s"],
        "h6_sol_kJ_kg": state["h_sol_kJ_kg"],
        "h6_flash_kJ_kg": state["h_flash_kJ_kg"],
        "h6_mix_kJ_kg": state["h_mix_kJ_kg"],
        "flash_fraction": state["flash_fraction"],
    }

# ---------------------------------------------------------------------------
# Crystallization correlation and state validation
# ---------------------------------------------------------------------------

def crystallization_limit(decider: str, value: float) -> float:
    """Critical temperature or critical LiBr mass fraction per Boryta/Albers.

    Parameters
    ----------
    decider:
        "w" -> critical temperature [degC] from LiBr mass fraction [-]
        "T" -> critical LiBr mass fraction [-] from temperature [K]
    """
    if decider == "w":
        parameter_T = [
            42.90198341384762, 34.67510890651030, 31.30778644395644, 2.99859601946791,
            -19.36781324384540, -4.88856108511827, 4.61433775768846, 1.80636830673333,
        ]
        w_r = (value - 0.64794) / 0.044858
        return sum(parameter_T[i] * w_r**i for i in range(8))

    if decider == "T":
        parameter_w = [
            0.66136507494441, 0.02262634534253, -0.02216522722755, 0.05134156572205,
            0.00034455919818, -0.03628931060739, 0.00252166562759, 0.00796985214167,
        ]
        T_c = value - 273.15
        T_r = (T_c - 54.793) / 33.111
        return sum(parameter_w[i] * T_r**i for i in range(8))

    raise ValueError("decider must be 'T' or 'w'")



def validate_solution_state(T: float, w_libr: float, label: str = "") -> StateValidity:
    if not (T_MIN_PAT <= T <= T_MAX_PAT):
        return StateValidity(
            in_patek_range=False,
            crystallization_checked=False,
            crystallization_safe=False,
            message=f"{label}: temperature {T:.3f} K is outside the Patek range [{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K.",
        )

    if not (273.15 <= T <= 374.0):
        return StateValidity(
            in_patek_range=True,
            crystallization_checked=False,
            crystallization_safe=True,
            message=f"{label}: temperature {T:.3f} K is outside the Boryta check range; crystallization was not evaluated.",
        )

    if w_libr <= 0.57:
        return StateValidity(
            in_patek_range=True,
            crystallization_checked=True,
            crystallization_safe=True,
            message=f"{label}: below the usual crystallization limit of the correlation used.",
        )

    w_crit = crystallization_limit("T", T)
    T_crit = crystallization_limit("w", w_libr) + 273.15
    safe = (w_libr <= w_crit) and (T >= T_crit)

    if safe:
        message = (
            f"{label}: crystallization check passed "
            f"(w_LiBr={w_libr:.4f} <= w_crit={w_crit:.4f}, T={T:.2f} K >= T_crit={T_crit:.2f} K)."
        )
    else:
        message = (
            f"{label}: crystallization risk "
            f"(w_LiBr={w_libr:.4f}, w_crit={w_crit:.4f}, T={T:.2f} K, T_crit={T_crit:.2f} K)."
        )

    return StateValidity(
        in_patek_range=True,
        crystallization_checked=True,
        crystallization_safe=safe,
        message=message,
    )


__all__ = [
    "PropertyError",
    "StateValidity",
    "w_libr_from_x",
    "x_from_w_libr",
    "h_solution_mass_kjkg",
    "s_solution_mass_kjkgK",
    "cp_solution_mass_kjkgk",
    "rho_solution_mass",
    "T_sat_solution_from_p_x",
    "T_from_h_x_mass",
    "validate_solution_state",
    "crystallization_limit",
    "flash_valve_state_2_to_1",
    "flash_valve_state_5_to_6",
]
