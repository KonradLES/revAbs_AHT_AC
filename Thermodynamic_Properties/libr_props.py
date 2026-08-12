"""LiBr/H2O-Stofffunktionen für die AWT-Simulation.

Konventionen
------------
- x : LiBr-Molanteil in der Lösung [-]
- w : LiBr-Massenanteil in der Lösung [-]
- T : Temperatur [K]
- p : Druck [Pa]
- h : spezifische Enthalpie der Lösung [kJ/kg]
- rho : Dichte der Lösung [kg/m^3]

Die Implementierung verwendet die Patek-Korrelationen. Nach außen werden nur
massenbasierte Größen mit konsistenten Einheiten für das Simulationsmodell
bereitgestellt.
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
    """Fehler bei Stoffwert- oder Inversionsberechnungen."""


@dataclass(frozen=True)
class StateValidity:
    in_patek_range: bool
    crystallization_checked: bool
    crystallization_safe: bool
    message: str


def _validate_x_patek_range(x_libr_mol: float, *, function_name: str) -> float:
    """Prüft den LiBr-Molanteil gegen den Gültigkeitsbereich der Patek-Korrelationen."""
    x = float(x_libr_mol)
    if not (X_MIN_PAT <= x <= X_MAX_PAT):
        raise PropertyError(
            f"{function_name}: LiBr-Molanteil x={x:.9f} liegt außerhalb des implementierten "
            f"Patek-Gültigkeitsbereichs [{X_MIN_PAT:.9f}, {X_MAX_PAT:.6f}] [-]. "
            f"Die verwendeten Korrelationen enthalten den Term (0.4 - x) und werden in dieser "
            f"Implementierung nur für x < 0.4 ausgewertet."
        )
    return x



def _validate_T_patek_range(T: float, *, function_name: str) -> float:
    """Prüft die Temperatur gegen den Gültigkeitsbereich der Patek-Korrelationen."""
    T_val = float(T)
    if not (T_MIN_PAT <= T_val <= T_MAX_PAT):
        raise PropertyError(
            f"{function_name}: Temperatur T={T_val:.6f} K liegt außerhalb des implementierten "
            f"Patek-Gültigkeitsbereichs [{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K."
        )
    return T_val


# ---------------------------------------------------------------------------
# Konzentrations- und Molmassenbeziehungen
# ---------------------------------------------------------------------------

def mixture_molar_mass(x_libr_mol: float) -> float:
    x = _validate_x_patek_range(x_libr_mol, function_name="mixture_molar_mass")
    return x * M_LIBR + (1.0 - x) * M_H2O



def w_libr_from_x(x_libr_mol: float) -> float:
    """LiBr-Massenanteil w aus LiBr-Molanteil x."""
    x = _validate_x_patek_range(x_libr_mol, function_name="w_libr_from_x")
    return x * M_LIBR / mixture_molar_mass(x)



def x_from_w_libr(w_libr: float) -> float:
    """LiBr-Molanteil x aus LiBr-Massenanteil w."""
    w = float(w_libr)
    denominator = M_LIBR - w * (M_LIBR - M_H2O)
    return (w * M_H2O) / denominator


# ---------------------------------------------------------------------------
# Direkte Patek-Funktionen (molare Basis)
# ---------------------------------------------------------------------------

def calc_cp_molar_patek(T: float, x_libr_mol: float) -> float:
    """Molare Wärmekapazität der LiBr/H2O-Lösung [J/mol/K]."""
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
    """Molare Enthalpie der LiBr/H2O-Lösung [J/mol]."""
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
    """Sättigungsdruck der LiBr/H2O-Lösung [Pa]."""
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
    """Molare Dichte der LiBr/H2O-Lösung [mol/m^3]."""
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
    """Molare Entropie der LiBr/H2O-Lösung [J/mol/K]."""
    T = _validate_T_patek_range(T, function_name="calc_s_molar_patek")
    x = _validate_x_patek_range(x_libr_mol, function_name="calc_s_molar_patek")
    T_c = 647.096              #[K]
    s_c = 79.3933              #[J/molK]
    T_0 = 221                  #[K]

    # Table 8
    Koef_a = [  1.53091     *   10**0,
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
    Koef_t = [0,0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5]
    Koef_n = [0,1,6,6,2,0,0,4,0,0,4,0,4,5,2,5,0,4,0,1,0,2,4,7,1,0,1,2,3]
    Koef_m = [1,1,2,3,6,1,3,5,1,2,2,4,5,5,6,6,1,3,5,7,1,1,1,2,3,1,1,1,1]
    # Table 15
    Koef_beta = [1/3,1,8/3,8]
    Koef_alpha = [  -3.34112    *   10**-1,
                    -8.47987    *   10**-1,
                    -9.11980    *   10**-1,
                    -1.64046    *   10**0]

    # Calculation of s_sat
    sum = 0
    for i in range(4):
        sum = sum + Koef_alpha[i]*(1-(T/T_c))**Koef_beta[i]
    s_sat = s_c * (1 + sum)
    # Calculation of s
    factors = numpy.zeros((29,))
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    for i in range(29):
        factors[i] =  Koef_a[i]*x**Koef_m[i]*(0.4-x)**Koef_n[i]
        if Koef_t[i] == 0:
            f = f + factors[i]
        elif Koef_t[i] == 1:
            e = e + factors[i]
        elif Koef_t[i] == 2:
            d = d + factors[i]
        elif Koef_t[i] == 3:
            c = c + factors[i]
        elif Koef_t[i] == 4:
            b = b + factors[i]
        elif Koef_t[i] == 5:
            a = a + factors[i]
    s = (1-x)*s_sat + s_c*(a*(T_c/(T-T_0))**5 + b*(T_c/(T-T_0))**4 + c*(T_c/(T-T_0))**3 + d*(T_c/(T-T_0))**2 + e*(T_c/(T-T_0))**1 + f)
    return s

# ---------------------------------------------------------------------------
# Massenbasierte Wrapper
# ---------------------------------------------------------------------------

def h_solution_mass_kjkg(T: float, x_libr_mol: float) -> float:
    """Spezifische Enthalpie der Lösung [kJ/kg]."""
    T = _validate_T_patek_range(T, function_name="h_solution_mass_kjkg")
    x = _validate_x_patek_range(x_libr_mol, function_name="h_solution_mass_kjkg")
    return calc_h_molar_patek(T, x) / mixture_molar_mass(x) / 1000.0



def s_solution_mass_kjkgK(T: float, x_libr_mol: float) -> float:
    """"Spezifische Entropie der LiBr/H2O-Lösung [kJ/kg/K]."""
    T = _validate_T_patek_range(T, function_name="s_solution_mass_kjkgK")
    x = _validate_x_patek_range(x_libr_mol, function_name="s_solution_mass_kjkgK")
    return calc_s_molar_patek(T, x) / mixture_molar_mass(x) / 1000.0



def cp_solution_mass_kjkgk(T: float, x_libr_mol: float) -> float:
    """Spezifische Wärmekapazität der Lösung [kJ/kg/K]."""
    T = _validate_T_patek_range(T, function_name="cp_solution_mass_kjkgk")
    x = _validate_x_patek_range(x_libr_mol, function_name="cp_solution_mass_kjkgk")
    return calc_cp_molar_patek(T, x) / mixture_molar_mass(x) / 1000.0



def rho_solution_mass(T: float, x_libr_mol: float) -> float:
    """Massendichte der Lösung [kg/m^3]."""
    T = _validate_T_patek_range(T, function_name="rho_solution_mass")
    x = _validate_x_patek_range(x_libr_mol, function_name="rho_solution_mass")
    return calc_rho_molar_patek(T, x) * mixture_molar_mass(x)



def T_sat_solution_from_p_x(p_pa: float, x_libr_mol: float) -> float:
    """Sättigungstemperatur der Lösung aus Druck und LiBr-Molanteil [K]."""
    x = _validate_x_patek_range(x_libr_mol, function_name="T_sat_solution_from_p_x")

    def fun(T: float) -> float:
        return calc_p_sat_patek(T, x) - p_pa

    try:
        sol = root_scalar(fun, bracket=[T_MIN_PAT + 1e-6, T_MAX_PAT - 1e-6], method="brentq")
    except ValueError as exc:
        raise PropertyError(
            f"T_sat_solution_from_p_x: Keine Sättigungstemperatur im implementierten Patek-Temperaturbereich "
            f"[{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K für p={p_pa:.3e} Pa und x={x:.9f} gefunden."
        ) from exc
    return _validate_T_patek_range(float(sol.root), function_name="T_sat_solution_from_p_x")



def T_from_h_x_mass(h_kjkg: float, x_libr_mol: float) -> float:
    """Temperatur der Lösung aus spezifischer Enthalpie und LiBr-Molanteil [K]."""
    x = _validate_x_patek_range(x_libr_mol, function_name="T_from_h_x_mass")

    def fun(T: float) -> float:
        return h_solution_mass_kjkg(T, x) - h_kjkg

    try:
        sol = root_scalar(fun, bracket=[T_MIN_PAT + 1e-6, T_MAX_PAT - 1e-6], method="brentq")
    except ValueError as exc:
        raise PropertyError(
            f"T_from_h_x_mass: Keine Temperatur im implementierten Patek-Temperaturbereich "
            f"[{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K für h={h_kjkg:.6f} kJ/kg und x={x:.9f} gefunden."
        ) from exc
    return _validate_T_patek_range(float(sol.root), function_name="T_from_h_x_mass")


def flash_valve_state_2_to_1(
    p_out_pa: float,
    h2_kJkg: float,
    m2_kg_s: float,
    x2_libr_mol: float,
    ) -> dict:
    """
    Einfache isenthalpe Flash-Drossel 2 -> 1 für LiBr/H2O.

    Annahmen
    --------
    - Die Drossel ist isenthalp.
    - Nach der Drossel liegt Gleichgewicht bei p_out vor.
    - Die Flüssigphase ist LiBr/H2O-Lösung.
    - Die Dampfphase ist reines Wasser.
    - LiBr verbleibt vollständig in der Flüssigphase.

    Energiebasis
    ------------
    h2_kJkg ist als spezifische Enthalpie des Eintrittsstroms vor der Drossel
    auf Massenbasis [kJ/kg] zu verstehen. Da der Eintrittsstrom vollständig
    aus Lösung besteht, ist dies mit der gemittelten Austrittsenthalpie
    h1_mix_kJ_kg auf Gesamtmassenbasis konsistent.

    Rückgabe
    --------
    Dictionary mit:
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
    if p_out_pa <= 0.0:
        raise PropertyError(
            f"Flash-Drossel: p_out muss positiv sein, erhalten p_out={p_out_pa:.6e} Pa."
        )

    if m2_kg_s <= 0.0:
        raise PropertyError(
            f"Flash-Drossel: m2 muss positiv sein, erhalten m2={m2_kg_s:.6f} kg/s."
        )

    x2 = float(x2_libr_mol)
    if not (X_MIN_PAT <= x2 <= X_MAX_PAT):
        raise PropertyError(
            f"Flash-Drossel: x2={x2:.6f} liegt außerhalb des zulässigen Bereichs "
            f"[{X_MIN_PAT:.6f}, {X_MAX_PAT:.6f}]."
        )

    w2 = w_libr_from_x(x2)

    # Untere Temperaturgrenze: kein Flash, ursprüngliche Zusammensetzung auf Sättigungslinie
    T_lo = T_sat_solution_from_p_x(p_out_pa, x2)

    # Obere Temperaturgrenze: stark konzentrierte Lösung bei gleichem Druck
    x_hi = X_MAX_PAT - 1.0e-4
    if x_hi <= x2:
        x_hi = X_MAX_PAT - 1.0e-9

    T_hi = T_sat_solution_from_p_x(p_out_pa, x_hi)

    if T_hi <= T_lo:
        raise PropertyError(
            "Flash-Drossel: Ungültiges Temperaturintervall für die Flash-Suche. "
            f"T_lo={T_lo:.6f} K, T_hi={T_hi:.6f} K."
        )

    # Toleranzen für numerische Grenzfälle
    T_tol = 1.0e-8
    p_tol = 1.0e-6
    w_tol = 1.0e-10
    m_tol = 1.0e-12

    def x_eq_from_T(T_K: float) -> float:
        """
        Gleichgewichtszusammensetzung x1 der Flüssigphase bei gegebener Temperatur T_K
        und Druck p_out_pa.
        """
        if not (T_MIN_PAT <= T_K <= T_MAX_PAT):
            raise PropertyError(
                f"Flash-Drossel: T={T_K:.6f} K liegt außerhalb des zulässigen Bereichs "
                f"[{T_MIN_PAT:.6f}, {T_MAX_PAT:.6f}] K."
            )

        # Explizite Behandlung des No-Flash-Grenzfalls
        if T_K <= T_lo + T_tol:
            return x2

        def f_x(x_val: float) -> float:
            return calc_p_sat_patek(T_K, x_val) - p_out_pa

        f_left = f_x(x2)
        if abs(f_left) <= p_tol:
            return x2

        x_right = X_MAX_PAT - 1.0e-9
        f_right = f_x(x_right)

        if f_left * f_right > 0.0:
            raise PropertyError(
                "Flash-Drossel: Keine Gleichgewichtszusammensetzung x im Intervall "
                f"[x2, X_MAX_PAT] bei T={T_K:.6f} K und p={p_out_pa:.3e} Pa gefunden. "
                f"f(x2)={f_left:.6e}, f(x_max)={f_right:.6e}."
            )

        try:
            sol_x = root_scalar(
                f_x,
                bracket=[x2, x_right],
                method="brentq",
            )
        except ValueError as exc:
            raise PropertyError(
                f"Flash-Drossel: Keine Gleichgewichtszusammensetzung x bei "
                f"T={T_K:.6f} K und p={p_out_pa:.3e} Pa gefunden."
            ) from exc

        return float(sol_x.root)

    def build_state(T_K: float, strict: bool = False) -> dict:
        # Expliziter No-Flash-Grenzfall
        if T_K <= T_lo + T_tol:
            x1 = x2
            w1 = w2
            m1_sol = m2_kg_s
            m1_flash = 0.0
            h1_sol = h_solution_mass_kjkg(T_K, x1)
            h1_flash = CP.PropsSI("H", "T", T_K, "P", p_out_pa, "Water") / 1000.0
            h1_mix = h1_sol

            return {
                "T1_K": T_K,
                "x1_LiBr_mol": x1,
                "w1_LiBr": w1,
                "m1_sol_kg_s": m1_sol,
                "m1_flash_kg_s": m1_flash,
                "h1_sol_kJ_kg": h1_sol,
                "h1_flash_kJ_kg": h1_flash,
                "h1_mix_kJ_kg": h1_mix,
                "flash_fraction": 0.0,
            }

        x1 = x_eq_from_T(T_K)
        w1 = w_libr_from_x(x1)

        # Nahe der Flash-Grenze kann durch numerisches Rauschen w1 minimal kleiner als w2 werden.
        if w1 < w2:
            if (w2 - w1) <= w_tol:
                x1 = x2
                w1 = w2
            elif strict:
                raise PropertyError(
                    "Flash-Drossel: Die Flüssiglösung nach der Drossel wäre verdünnter "
                    f"als der Eintrittszustand. w1={w1:.12f}, w2={w2:.12f}."
                )
            else:
                x1 = x2
                w1 = w2

        # LiBr-Bilanz: m2 * w2 = m1_sol * w1
        m1_sol = m2_kg_s * w2 / w1
        m1_flash = m2_kg_s - m1_sol

        if m1_flash < 0.0:
            if abs(m1_flash) <= m_tol:
                m1_flash = 0.0
            elif strict:
                raise PropertyError(
                    f"Flash-Drossel: negativer Flashmassenstrom berechnet: "
                    f"m1_flash={m1_flash:.12e} kg/s."
                )
            else:
                m1_flash = 0.0

        h1_sol = h_solution_mass_kjkg(T_K, x1)

        # Reiner Wasserdampf bei (T_K, p_out_pa)
        h1_flash = CP.PropsSI("H", "T", T_K, "P", p_out_pa, "Water") / 1000.0

        h1_mix = (m1_sol * h1_sol + m1_flash * h1_flash) / m2_kg_s

        return {
            "T1_K": T_K,
            "x1_LiBr_mol": x1,
            "w1_LiBr": w1,
            "m1_sol_kg_s": m1_sol,
            "m1_flash_kg_s": m1_flash,
            "h1_sol_kJ_kg": h1_sol,
            "h1_flash_kJ_kg": h1_flash,
            "h1_mix_kJ_kg": h1_mix,
            "flash_fraction": m1_flash / m2_kg_s,
        }

    def residual(T_K: float) -> float:
        state = build_state(T_K, strict=False)
        return state["h1_mix_kJ_kg"] - h2_kJkg

    r_lo = residual(T_lo)
    h_tol = 1.0e-9

    if abs(r_lo) <= h_tol:
        return build_state(T_lo, strict=True)

    # -------------------------------------------------------------------------
    # NEU: echter No-Flash-Fall äquivalent zur Modelica-Implementierung
    # -------------------------------------------------------------------------
    #
    # Bisher begann die Nullstellensuche bei T_lo, also bei der Siedetemperatur
    # der Eintrittslösung x2 bei p_out.
    #
    # Falls h2_kJkg kleiner ist als die Flüssigkeitsenthalpie an diesem Punkt,
    # kann die Lösung nach der Drossel nicht flashen. Sie bleibt unterkühlt:
    #
    #   Q = 0
    #   x1 = x2
    #   w1 = w2
    #   m1_flash = 0
    #
    # Dann muss T1 aus der isenthalpen Flüssigkeitsbedingung bestimmt werden:
    #
    #   h_solution_mass_kjkg(T1, x2) = h2_kJkg
    #
    # Genau das entspricht dem Modelica-Zweig:
    #
    #   if Q_intern <= 0 then
    #       Q = 0;
    #       X_LiBr_out = X_LiBr_in;
    #       h_out = h_solution(T_out, X_LiBr_out, p_out);
    #   end if;
    #
    # In Residual-Schreibweise ist:
    #
    #   r_lo = h_solution(T_lo, x2) - h2
    #
    # Wenn r_lo > 0, ist h2 zu niedrig für einen gesättigten/flashenden Zustand.
    # Dann suchen wir T1 unterhalb von T_lo.
    # -------------------------------------------------------------------------
    if r_lo > 0.0:

        def residual_no_flash(T_K: float) -> float:
            return h_solution_mass_kjkg(T_K, x2) - h2_kJkg

        T_no_flash_lo = T_MIN_PAT
        r_no_flash_lo = residual_no_flash(T_no_flash_lo)

        if abs(r_no_flash_lo) <= h_tol:
            return build_state(T_no_flash_lo, strict=True)

        if r_no_flash_lo * r_lo > 0.0:
            raise PropertyError(
                "Flash-Drossel: Kein No-Flash-Zustand im Temperaturintervall "
                f"[{T_no_flash_lo:.6f}, {T_lo:.6f}] K gefunden. "
                "Die Eintrittsenthalpie liegt vermutlich außerhalb des zulässigen "
                "Enthalpiebereichs der LiBr/H2O-Korrelation. "
                f"Residual bei T_min = {r_no_flash_lo:.6e} kJ/kg, "
                f"Residual bei T_lo = {r_lo:.6e} kJ/kg."
            )

        try:
            sol_T_no_flash = root_scalar(
                residual_no_flash,
                bracket=[T_no_flash_lo, T_lo],
                method="brentq",
            )
        except ValueError as exc:
            raise PropertyError(
                "Flash-Drossel: Nullstellensuche für den No-Flash-Fall fehlgeschlagen. "
                f"Prüfintervall: T_min={T_no_flash_lo:.6f} K, "
                f"T_lo={T_lo:.6f} K."
            ) from exc

        return build_state(float(sol_T_no_flash.root), strict=True)

    # Ab hier gilt r_lo < 0:
    # Die Eintrittsenthalpie ist größer als die Enthalpie der gesättigten
    # Eintrittslösung bei p_out. Es muss daher ein Flash-Zustand gesucht werden.
    r_hi = residual(T_hi)
    if abs(r_hi) <= h_tol:
        return build_state(T_hi, strict=True)

    if r_lo * r_hi > 0.0:
        raise PropertyError(
            "Flash-Drossel: Kein Vorzeichenwechsel des Enthalpie-Residuals im "
            f"Temperaturintervall [{T_lo:.6f}, {T_hi:.6f}] K. "
            f"Residual unten = {r_lo:.6e} kJ/kg, "
            f"Residual oben  = {r_hi:.6e} kJ/kg."
        )

    try:
        sol_T = root_scalar(
            residual,
            bracket=[T_lo, T_hi],
            method="brentq",
        )
    except ValueError as exc:
        raise PropertyError(
            "Flash-Drossel: Nullstellensuche für die isenthalpe Flash-Bedingung fehlgeschlagen. "
            f"Prüfintervall: T_lo={T_lo:.6f} K, T_hi={T_hi:.6f} K."
        ) from exc

    return build_state(float(sol_T.root), strict=True)

def flash_valve_state_5_to_6(
    p_out_pa: float,
    h5_kJkg: float,
    m5_kg_s: float,
    x5_libr_mol: float,
    ) -> dict:
    """
    Einfache isenthalpe Flash-Drossel 5 -> 6 für LiBr/H2O.

    Annahmen
    --------
    - Die Drossel ist isenthalp.
    - Nach der Drossel liegt Gleichgewicht bei p_out vor.
    - Die Flüssigphase ist LiBr/H2O-Lösung.
    - Die Dampfphase ist reines Wasser.
    - LiBr verbleibt vollständig in der Flüssigphase.

    Energiebasis
    ------------
    h5_kJkg ist als spezifische Enthalpie des Eintrittsstroms vor der Drossel
    auf Massenbasis [kJ/kg] zu verstehen. Da der Eintrittsstrom vollständig
    aus Lösung besteht, ist dies mit der gemittelten Austrittsenthalpie
    h6_mix_kJ_kg auf Gesamtmassenbasis konsistent.

    Rückgabe
    --------
    Dictionary mit:
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
    if p_out_pa <= 0.0:
        raise PropertyError(
            f"Flash-Drossel: p_out muss positiv sein, erhalten p_out={p_out_pa:.6e} Pa."
        )

    if m5_kg_s <= 0.0:
        raise PropertyError(
            f"Flash-Drossel: m5 muss positiv sein, erhalten m5={m5_kg_s:.6f} kg/s."
        )

    x5 = float(x5_libr_mol)
    if not (X_MIN_PAT <= x5 <= X_MAX_PAT):
        raise PropertyError(
            f"Flash-Drossel: x5={x5:.6f} liegt außerhalb des zulässigen Bereichs "
            f"[{X_MIN_PAT:.6f}, {X_MAX_PAT:.6f}]."
        )

    w5 = w_libr_from_x(x5)

    # Untere Temperaturgrenze: kein Flash, ursprüngliche Zusammensetzung auf Sättigungslinie
    T_lo = T_sat_solution_from_p_x(p_out_pa, x5)

    # Obere Temperaturgrenze: stark konzentrierte Lösung bei gleichem Druck
    x_hi = X_MAX_PAT - 1.0e-4
    if x_hi <= x5:
        x_hi = X_MAX_PAT - 1.0e-9

    T_hi = T_sat_solution_from_p_x(p_out_pa, x_hi)

    if T_hi <= T_lo:
        raise PropertyError(
            "Flash-Drossel: Ungültiges Temperaturintervall für die Flash-Suche. "
            f"T_lo={T_lo:.6f} K, T_hi={T_hi:.6f} K."
        )

    # Toleranzen für numerische Grenzfälle
    T_tol = 1.0e-8
    p_tol = 1.0e-6
    w_tol = 1.0e-10
    m_tol = 1.0e-12

    def x_eq_from_T(T_K: float) -> float:
        """
        Gleichgewichtszusammensetzung x6 der Flüssigphase bei gegebener Temperatur T_K
        und Druck p_out_pa.
        """
        if not (T_MIN_PAT <= T_K <= T_MAX_PAT):
            raise PropertyError(
                f"Flash-Drossel: T={T_K:.6f} K liegt außerhalb des zulässigen Bereichs "
                f"[{T_MIN_PAT:.6f}, {T_MAX_PAT:.6f}] K."
            )

        # Explizite Behandlung des No-Flash-Grenzfalls
        if T_K <= T_lo + T_tol:
            return x5

        def f_x(x_val: float) -> float:
            return calc_p_sat_patek(T_K, x_val) - p_out_pa

        f_left = f_x(x5)
        if abs(f_left) <= p_tol:
            return x5

        x_right = X_MAX_PAT - 1.0e-9
        f_right = f_x(x_right)

        if f_left * f_right > 0.0:
            raise PropertyError(
                "Flash-Drossel: Keine Gleichgewichtszusammensetzung x im Intervall "
                f"[x5, X_MAX_PAT] bei T={T_K:.6f} K und p={p_out_pa:.3e} Pa gefunden. "
                f"f(x5)={f_left:.6e}, f(x_max)={f_right:.6e}."
            )

        try:
            sol_x = root_scalar(
                f_x,
                bracket=[x5, x_right],
                method="brentq",
            )
        except ValueError as exc:
            raise PropertyError(
                f"Flash-Drossel: Keine Gleichgewichtszusammensetzung x bei "
                f"T={T_K:.6f} K und p={p_out_pa:.3e} Pa gefunden."
            ) from exc

        return float(sol_x.root)

    def build_state(T_K: float, strict: bool = False) -> dict:
        # Expliziter No-Flash-Grenzfall
        if T_K <= T_lo + T_tol:
            x6 = x5
            w6 = w5
            m6_sol = m5_kg_s
            m6_flash = 0.0
            h6_sol = h_solution_mass_kjkg(T_K, x6)
            h6_flash = CP.PropsSI("H", "T", T_K, "P", p_out_pa, "Water") / 1000.0
            h6_mix = h6_sol

            return {
                "T6_K": T_K,
                "x6_LiBr_mol": x6,
                "w6_LiBr": w6,
                "m6_sol_kg_s": m6_sol,
                "m6_flash_kg_s": m6_flash,
                "h6_sol_kJ_kg": h6_sol,
                "h6_flash_kJ_kg": h6_flash,
                "h6_mix_kJ_kg": h6_mix,
                "flash_fraction": 0.0,
            }

        x6 = x_eq_from_T(T_K)
        w6 = w_libr_from_x(x6)

        # Nahe der Flash-Grenze kann durch numerisches Rauschen w6 minimal kleiner als w5 werden.
        if w6 < w5:
            if (w5 - w6) <= w_tol:
                x6 = x5
                w6 = w5
            elif strict:
                raise PropertyError(
                    "Flash-Drossel: Die Flüssiglösung nach der Drossel wäre verdünnter "
                    f"als der Eintrittszustand. w6={w6:.12f}, w5={w5:.12f}."
                )
            else:
                x6 = x5
                w6 = w5

        # LiBr-Bilanz: m5 * w5 = m6_sol * w6
        m6_sol = m5_kg_s * w5 / w6
        m6_flash = m5_kg_s - m6_sol

        if m6_flash < 0.0:
            if abs(m6_flash) <= m_tol:
                m6_flash = 0.0
            elif strict:
                raise PropertyError(
                    f"Flash-Drossel: negativer Flashmassenstrom berechnet: "
                    f"m6_flash={m6_flash:.12e} kg/s."
                )
            else:
                m6_flash = 0.0

        h6_sol = h_solution_mass_kjkg(T_K, x6)

        # Reiner Wasserdampf bei (T_K, p_out_pa)
        h6_flash = CP.PropsSI("H", "T", T_K, "P", p_out_pa, "Water") / 1000.0

        h6_mix = (m6_sol * h6_sol + m6_flash * h6_flash) / m5_kg_s

        return {
            "T6_K": T_K,
            "x6_LiBr_mol": x6,
            "w6_LiBr": w6,
            "m6_sol_kg_s": m6_sol,
            "m6_flash_kg_s": m6_flash,
            "h6_sol_kJ_kg": h6_sol,
            "h6_flash_kJ_kg": h6_flash,
            "h6_mix_kJ_kg": h6_mix,
            "flash_fraction": m6_flash / m5_kg_s,
        }

    def residual(T_K: float) -> float:
        state = build_state(T_K, strict=False)
        return state["h6_mix_kJ_kg"] - h5_kJkg

    r_lo = residual(T_lo)
    h_tol = 1.0e-9

    if abs(r_lo) <= h_tol:
        return build_state(T_lo, strict=True)

    # -------------------------------------------------------------------------
    # NEU: echter No-Flash-Fall äquivalent zur Modelica-Implementierung
    # -------------------------------------------------------------------------
    #
    # Bisher begann die Nullstellensuche bei T_lo, also bei der Siedetemperatur
    # der Eintrittslösung x5 bei p_out.
    #
    # Falls h5_kJkg kleiner ist als die Flüssigkeitsenthalpie an diesem Punkt,
    # kann die Lösung nach der Drossel nicht flashen. Sie bleibt unterkühlt:
    #
    #   Q = 0
    #   x6 = x5
    #   w6 = w5
    #   m6_flash = 0
    #
    # Dann muss T6 aus der isenthalpen Flüssigkeitsbedingung bestimmt werden:
    #
    #   h_solution_mass_kjkg(T6, x5) = h5_kJkg
    #
    # Genau das entspricht dem Modelica-Zweig:
    #
    #   if Q_intern <= 0 then
    #       Q = 0;
    #       X_LiBr_out = X_LiBr_in;
    #       h_out = h_solution(T_out, X_LiBr_out, p_out);
    #   end if;
    #
    # In Residual-Schreibweise ist:
    #
    #   r_lo = h_solution(T_lo, x5) - h5
    #
    # Wenn r_lo > 0, ist h5 zu niedrig für einen gesättigten/flashenden Zustand.
    # Dann suchen wir T6 unterhalb von T_lo.
    # -------------------------------------------------------------------------
    if r_lo > 0.0:

        def residual_no_flash(T_K: float) -> float:
            return h_solution_mass_kjkg(T_K, x5) - h5_kJkg

        T_no_flash_lo = T_MIN_PAT
        r_no_flash_lo = residual_no_flash(T_no_flash_lo)

        if abs(r_no_flash_lo) <= h_tol:
            return build_state(T_no_flash_lo, strict=True)

        if r_no_flash_lo * r_lo > 0.0:
            raise PropertyError(
                "Flash-Drossel: Kein No-Flash-Zustand im Temperaturintervall "
                f"[{T_no_flash_lo:.6f}, {T_lo:.6f}] K gefunden. "
                "Die Eintrittsenthalpie liegt vermutlich außerhalb des zulässigen "
                "Enthalpiebereichs der LiBr/H2O-Korrelation. "
                f"Residual bei T_min = {r_no_flash_lo:.6e} kJ/kg, "
                f"Residual bei T_lo = {r_lo:.6e} kJ/kg."
            )

        try:
            sol_T_no_flash = root_scalar(
                residual_no_flash,
                bracket=[T_no_flash_lo, T_lo],
                method="brentq",
            )
        except ValueError as exc:
            raise PropertyError(
                "Flash-Drossel: Nullstellensuche für den No-Flash-Fall fehlgeschlagen. "
                f"Prüfintervall: T_min={T_no_flash_lo:.6f} K, "
                f"T_lo={T_lo:.6f} K."
            ) from exc

        return build_state(float(sol_T_no_flash.root), strict=True)

    # Ab hier gilt r_lo < 0:
    # Die Eintrittsenthalpie ist größer als die Enthalpie der gesättigten
    # Eintrittslösung bei p_out. Es muss daher ein Flash-Zustand gesucht werden.
    r_hi = residual(T_hi)
    if abs(r_hi) <= h_tol:
        return build_state(T_hi, strict=True)

    if r_lo * r_hi > 0.0:
        raise PropertyError(
            "Flash-Drossel: Kein Vorzeichenwechsel des Enthalpie-Residuals im "
            f"Temperaturintervall [{T_lo:.6f}, {T_hi:.6f}] K. "
            f"Residual unten = {r_lo:.6e} kJ/kg, "
            f"Residual oben  = {r_hi:.6e} kJ/kg."
        )

    try:
        sol_T = root_scalar(
            residual,
            bracket=[T_lo, T_hi],
            method="brentq",
        )
    except ValueError as exc:
        raise PropertyError(
            "Flash-Drossel: Nullstellensuche für die isenthalpe Flash-Bedingung fehlgeschlagen. "
            f"Prüfintervall: T_lo={T_lo:.6f} K, T_hi={T_hi:.6f} K."
        ) from exc

    return build_state(float(sol_T.root), strict=True)

# ---------------------------------------------------------------------------
# Kristallisationskorrelation und Zustandsvalidierung
# ---------------------------------------------------------------------------

def crystallization_limit(decider: str, value: float) -> float:
    """Kritische Temperatur oder kritischer LiBr-Massenanteil nach Boryta/Albers.

    Parameters
    ----------
    decider:
        "w" -> kritische Temperatur [°C] aus LiBr-Massenanteil [-]
        "T" -> kritischer LiBr-Massenanteil [-] aus Temperatur [K]
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

    raise ValueError("decider muss 'T' oder 'w' sein")



def validate_solution_state(T: float, w_libr: float, label: str = "") -> StateValidity:
    if not (T_MIN_PAT <= T <= T_MAX_PAT):
        return StateValidity(
            in_patek_range=False,
            crystallization_checked=False,
            crystallization_safe=False,
            message=f"{label}: Temperatur {T:.3f} K liegt außerhalb des Patek-Bereichs [{T_MIN_PAT:.2f}, {T_MAX_PAT:.2f}] K.",
        )

    if not (273.15 <= T <= 374.0):
        return StateValidity(
            in_patek_range=True,
            crystallization_checked=False,
            crystallization_safe=True,
            message=f"{label}: Temperatur {T:.3f} K liegt außerhalb des Boryta-Prüfbereichs; Kristallisation wurde nicht bewertet.",
        )

    if w_libr <= 0.57:
        return StateValidity(
            in_patek_range=True,
            crystallization_checked=True,
            crystallization_safe=True,
            message=f"{label}: Unterhalb der üblichen Kristallisationsgrenze der verwendeten Korrelation.",
        )

    w_crit = crystallization_limit("T", T)
    T_crit = crystallization_limit("w", w_libr) + 273.15
    safe = (w_libr <= w_crit) and (T >= T_crit)

    if safe:
        message = (
            f"{label}: Kristallisationsprüfung bestanden "
            f"(w_LiBr={w_libr:.4f} <= w_crit={w_crit:.4f}, T={T:.2f} K >= T_crit={T_crit:.2f} K)."
        )
    else:
        message = (
            f"{label}: Kristallisationsrisiko "
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
