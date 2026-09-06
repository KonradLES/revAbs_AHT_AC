from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

"""Startwertsuche und m6-Optimierung für einen neuen Betriebspunkt.

Workflow
--------
1. Ziel-Betriebspunkt vorgeben (externe Temperaturen unten).
2. Multi-Start-Suche: Der Solver wird mit einem Gitter aus Startvektoren
   gestartet, um für das neue Temperaturniveau eine konvergierte Lösung
   zu finden. Die beste konvergierte Lösung (kleinste Residuennorm) wird
   als Referenzpunkt übernommen.
3. m6-Optimierung: Ausgehend vom gefundenen Referenzpunkt wird m6 in einem
   1D-Scan variiert. Der Wert, der die höchste Absorberleistung Q_abs liefert,
   wird als optimales m6 ausgegeben.
4. Abschließende Ausgabe: Startvektoren und optimale Konfiguration werden
   so ausgegeben, dass sie direkt ins main-Skript übernommen werden können.
"""

import itertools
import numpy as np

from Models.AHT_UA_LMTD import (
    AWTInputs,
    AWTResult,
    primary_temperatures_C_to_K,
    primary_temperatures_K_to_C,
    solve_awt,
    print_summary,
    PRIMARY_VARIABLE_NAMES,
)

# ---------------------------------------------------------------------------
# >>>  Ziel-Betriebspunkt hier eintragen  <
# ---------------------------------------------------------------------------

TARGET_T_EVP_DES_C = 65.0   # T_13_C = T_15_C  Verdampfer-/Desorbereinlass
TARGET_T_ABS_IN_C  = 80.0   # T_11_C           Absorbereinlass
TARGET_T_COND_IN_C = 20.0   # T_17_C           Kondensatoreinlass

# ---------------------------------------------------------------------------
# >>>  m6-Optimierungsbereich  <
# ---------------------------------------------------------------------------
M6_MIN   = 0.05
M6_MAX   = 3.0
M6_STEPS = 60     # Anzahl Stützstellen im 1D-Scan

# ---------------------------------------------------------------------------
# Feste Anlagenparameter (UA-Werte, Massenströme, Spezifikationsmodi)
# ---------------------------------------------------------------------------
FIXED_PARAMS = dict(
    m_13=4.0,
    m_15=4.0,
    m_17=4.0,
    UA_cond=10.0,
    UA_evap=15.0,
    UA_abs=10.0,
    UA_des=25.0,
    UA_shex=70.8 / 6.43,
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="m11",
    cycle_scale_spec_mode="m6",
    desorber_evaporator_routing_mode="parallel",
    m11_spec=4.0,
)

# ---------------------------------------------------------------------------
# Gitter für die Multi-Start-Suche
# Jede Liste enthält Kandidatenwerte für die jeweilige Komponente des x0-Vektors.
# [T8, T10, x3, x6, x20, T2, T4, beta]  (Temperaturen in °C)
# ---------------------------------------------------------------------------
MULTISTART_GRID = dict(
    T8_C  = [TARGET_T_COND_IN_C + 5,  TARGET_T_COND_IN_C + 15],
    T10_C = [TARGET_T_EVP_DES_C - 15, TARGET_T_EVP_DES_C - 5],
    x3    = [0.15, 0.23],
    x6    = [0.18, 0.27],
    x20   = [0.17, 0.26],
    T2_C  = [TARGET_T_ABS_IN_C + 10,  TARGET_T_ABS_IN_C + 25],
    T4_C  = [TARGET_T_EVP_DES_C - 5,  TARGET_T_EVP_DES_C + 10],
    beta  = [0.10, 0.20],
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def build_inputs(m6: float) -> AWTInputs:
    return AWTInputs(
        T_11_C=TARGET_T_ABS_IN_C,
        T_13_C=TARGET_T_EVP_DES_C,
        T_15_C=TARGET_T_EVP_DES_C,
        T_17_C=TARGET_T_COND_IN_C,
        m6_spec=m6,
        **FIXED_PARAMS,
    )


def is_converged(result: AWTResult, tol: float = 1e-4) -> bool:
    return (
        result.solve_info.final_point_evaluable
        and result.solve_info.scaled_residual_norm < tol
    )


def result_to_x0_K(result: AWTResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def format_x0_for_main(z_K: np.ndarray) -> str:
    """Gibt den Startvektor als kopierbaren Block für das main-Skript aus."""
    z_C = primary_temperatures_K_to_C(z_K)
    names = PRIMARY_VARIABLE_NAMES
    units = ["°C", "°C", "-", "-", "-", "°C", "°C", "-"]
    lines = ["x0 = primary_temperatures_C_to_K(", "    np.array(["]
    for name, val, unit in zip(names, z_C, units):
        lines.append(f"        {val:10.4f},  # {name}  [{unit}]")
    lines.append("    ], dtype=float)")
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: Multi-Start-Suche
# ---------------------------------------------------------------------------

def multistart_search(m6_initial: float = 1.0) -> AWTResult | None:
    """Durchsucht das Startwertgitter und gibt das beste konvergierte Ergebnis zurück."""
    inputs = build_inputs(m6_initial)

    keys   = list(MULTISTART_GRID.keys())
    values = list(MULTISTART_GRID.values())
    candidates = list(itertools.product(*values))
    n = len(candidates)
    print(f"Multi-Start-Suche: {n} Kandidaten bei m6 = {m6_initial:.3f} kg/s ...")

    best_result: AWTResult | None = None
    best_norm   = float("inf")
    n_converged = 0

    for i, cand in enumerate(candidates):
        x0_C = np.array(cand, dtype=float)
        x0_K = primary_temperatures_C_to_K(x0_C)
        result = solve_awt(inputs, x0=x0_K)

        if is_converged(result):
            n_converged += 1
            norm = result.solve_info.scaled_residual_norm
            if norm < best_norm:
                best_norm   = norm
                best_result = result

    print(f"  {n_converged}/{n} Kandidaten konvergiert.")
    if best_result is not None:
        print(f"  Beste Residuennorm: {best_norm:.3e}")
    else:
        print("  Kein Kandidat konvergiert – Startwertgitter ggf. anpassen.")
    return best_result


# ---------------------------------------------------------------------------
# Phase 2: m6-Optimierung (1D-Scan, Warmstart)
# ---------------------------------------------------------------------------

def optimize_m6(seed_result: AWTResult) -> tuple[float, AWTResult]:
    """Scannt m6 im vorgegebenen Bereich; gibt (m6_opt, result_opt) zurück."""
    m6_values = np.linspace(M6_MIN, M6_MAX, M6_STEPS)

    best_m6     = float("nan")
    best_q_abs  = float("-inf")
    best_result : AWTResult | None = None

    # Startwert aus Seed-Ergebnis
    x0_K = result_to_x0_K(seed_result)

    print(f"\nm6-Scan von {M6_MIN:.3f} bis {M6_MAX:.3f} kg/s ({M6_STEPS} Punkte) ...")
    print(f"  {'m6 [kg/s]':>12}  {'Q_abs [kW]':>12}  {'ECOP':>8}  {'COP':>8}  konvergiert")

    for m6 in m6_values:
        inputs = build_inputs(m6)
        result = solve_awt(inputs, x0=x0_K)

        if is_converged(result):
            q_abs = result.heat_flows_kW.get("Q_abs", float("nan"))
            ecop  = result.exergy_kW.get("Exergy_efficiency", float("nan"))
            cop   = result.kpis.get("COP", float("nan"))
            print(f"  {m6:12.4f}  {q_abs:12.4f}  {ecop:8.4f}  {cop:8.4f}  ✓")

            # Warmstart für nächsten Schritt
            x0_K = result_to_x0_K(result)

            if q_abs > best_q_abs:
                best_q_abs  = q_abs
                best_m6     = m6
                best_result = result
        else:
            print(f"  {m6:12.4f}  {'–':>12}  {'–':>8}  {'–':>8}  ✗")

    if best_result is None:
        raise RuntimeError("m6-Scan: Kein einziger Punkt konvergiert.")

    return best_m6, best_result


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AWT – Startwertsuche und m6-Optimierung")
    print("=" * 70)
    print(f"  Ziel-Betriebspunkt:")
    print(f"    T_evap/des  = {TARGET_T_EVP_DES_C:.1f} °C")
    print(f"    T_abs_in    = {TARGET_T_ABS_IN_C:.1f} °C")
    print(f"    T_cond_in   = {TARGET_T_COND_IN_C:.1f} °C")
    print()

    # Phase 1
    seed = multistart_search(m6_initial=1.0)
    if seed is None:
        print("\nAbbruch: Keine konvergierte Startlösung gefunden.")
        print("Tipp: MULTISTART_GRID anpassen oder Temperaturen prüfen.")
        sys.exit(1)

    # Phase 2
    m6_opt, result_opt = optimize_m6(seed)

    # Ergebnisausgabe
    print("\n" + "=" * 70)
    print(f"Optimales m6 = {m6_opt:.4f} kg/s  →  Q_abs = "
          f"{result_opt.heat_flows_kW.get('Q_abs', float('nan')):.4f} kW")
    print("=" * 70)

    print("\n--- Vollständige Ergebnisübersicht ---")
    print_summary(result_opt)

    print("\n--- Startvektor für main-Skript (direkt kopierbar) ---")
    z_opt_K = result_to_x0_K(result_opt)
    print(format_x0_for_main(z_opt_K))
    print(f"\nm6_spec = {m6_opt:.4f}  # kg/s  (optimiert auf max. Q_abs)")