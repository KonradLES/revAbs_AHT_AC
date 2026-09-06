from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

"""Parameterstudie für das Pinch-Point-AHT-Modell bei konstanter Q_abs.

Q_abs_spec_kW bleibt fix; m6 wird vom Solver für jeden Betriebspunkt
automatisch bestimmt. Es kann eine von drei Temperaturen variiert werden:

  "T_abs_in"   -> T_11_C   (Absorbereinlasstemperatur)
  "T_evap_des" -> T_13_C = T_15_C  (Verdampfer-/Desorbereinlasstemperatur)
  "T_cond_in"  -> T_17_C   (Kondensatoreinlasstemperatur)

Jeder Betriebspunkt nutzt das Ergebnis des vorherigen Punkts als Startwert
(Warmstart), um die Konvergenz zu beschleunigen und zur korrekten Lösung
zu führen.
"""

import numpy as np
import matplotlib.pyplot as plt

from Models.AHT_UA_LMTD import (
    AWTInputs,
    AWTResult,
    primary_temperatures_C_to_K,
    solve_awt,
    PRIMARY_VARIABLE_NAMES,
)

# ---------------------------------------------------------------------------
# >>>  Scanvariable auswählen  <
# ---------------------------------------------------------------------------
SCAN_VARIABLE = "T_abs_in"   # "T_abs_in" | "T_evap_des" | "T_cond_in"

# ---------------------------------------------------------------------------
# Scan-Bereiche je Variable [Min, Max, Startpunkt] in °C
# ---------------------------------------------------------------------------
SCAN_CONFIG = {
    "T_abs_in": dict(
        min_C=70.0, max_C=95.0, start_C=85.0,
        xlabel="Absorbereinlasstemperatur $T_{11}$ [°C]",
        title_suffix="Absorbereinlasstemperatur",
    ),
    "T_evap_des": dict(
        min_C=60.0, max_C=78.0, start_C=65.0,
        xlabel="Verdampfer-/Desorbereinlasstemperatur $T_{13} = T_{15}$ [°C]",
        title_suffix="Verdampfer-/Desorbereinlasstemperatur",
    ),
    "T_cond_in": dict(
        min_C=10.0, max_C=35.0, start_C=20.0,
        xlabel="Kondensatoreinlasstemperatur $T_{17}$ [°C]",
        title_suffix="Kondensatoreinlasstemperatur",
    ),
}

T_SCAN_STEP_K = 1.0

# ---------------------------------------------------------------------------
# Feste Basiskonfiguration – Q_abs bleibt konstant, m6 wird vom Solver bestimmt
# ---------------------------------------------------------------------------
BASE_KWARGS_FIXED = dict(
    T_11_C=85.0,
    T_13_C=65.0,
    T_15_C=65.0,
    T_17_C=20.0,
    m_13=20.0,
    m_15=20.0,
    m_17=20.0,
    UA_shex=11,  # 10, 1.0025, 25.2578
    UA_cond=10,  # 10, 1.0025, 25.2578
    UA_evap=15,  # 15, 1.5079, 11.3518
    UA_abs=10,      # 10, 1.5, 8.1355
    UA_des=25,   # 25, 2.4895, 10.4058s
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="m11",
    m11_spec=20.0,
    cycle_scale_spec_mode="m6",   # "m6", "Qabs"
    # Qabs_spec_kW=60.0,          # ← konstant über den gesamten Sweep
    m6_spec=0.45,
    desorber_evaporator_routing_mode="parallel",
)

X0_CENTER_C = np.array(
    [30.1458, 57.2958, 0.1787, 0.2003, 0.1981, 66.1183, 85.6828, 0.0905],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def build_inputs(scan_value_C: float) -> AWTInputs:
    """Erstellt AWTInputs mit dem aktuellen Scanwert; alle anderen Werte fix."""
    kwargs = BASE_KWARGS_FIXED.copy()
    if SCAN_VARIABLE == "T_abs_in":
        kwargs["T_11_C"] = scan_value_C
    elif SCAN_VARIABLE == "T_evap_des":
        kwargs["T_13_C"] = scan_value_C
        kwargs["T_15_C"] = scan_value_C
    elif SCAN_VARIABLE == "T_cond_in":
        kwargs["T_17_C"] = scan_value_C
    else:
        raise ValueError(f"Unbekannte SCAN_VARIABLE: {SCAN_VARIABLE!r}")
    return AWTInputs(**kwargs)


def is_converged(result: AWTResult, tol: float = 1e-4) -> bool:
    return (
        result.solve_info.final_point_evaluable
        and result.solve_info.scaled_residual_norm < tol
    )


def result_to_x0(result: AWTResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def extract_results(result: AWTResult) -> dict | None:
    if not result.solve_info.final_point_evaluable:
        return None

    cop  = result.kpis.get("COP", float("nan"))
    ecop = result.exergy_kW.get("Exergy_efficiency", float("nan"))
    if np.isnan(cop) or np.isnan(ecop):
        return None

    # Absorberheizfläche: aus UA_abs = Q_abs / LMTD_abs, falls vorhanden;
    # im Pinch-Modell gibt es kein UA, daher wird hier die "wirksame"
    # Absorberfläche über UA_abs_equiv = Q_abs / LMTD_abs_pinch angenähert,
    # sofern eine Wärmeübergangszahl k_abs bekannt ist. Ohne k_abs wird
    # stattdessen UA_abs_equiv [kW/K] ausgegeben (Fläche ~ UA_abs_equiv / k_abs).
    # diag = result.diagnostics
    # dT_abs_1 = diag.get("deltaT_abs_hot_end_K", diag.get("dT_abs_hot_end_K", float("nan")))
    # dT_abs_2 = diag.get("deltaT_abs_cold_end_K", diag.get("dT_abs_cold_end_K", float("nan")))
    # # LMTD aus den beiden Pinch-Enden (Gegenstrom-Näherung)
    # if np.isfinite(dT_abs_1) and np.isfinite(dT_abs_2) and dT_abs_1 > 0 and dT_abs_2 > 0:
    #     if abs(dT_abs_1 - dT_abs_2) < 1e-9:
    #         lmtd_abs = dT_abs_1
    #     else:
    #         lmtd_abs = (dT_abs_1 - dT_abs_2) / np.log(dT_abs_1 / dT_abs_2)
    # else:
    #     lmtd_abs = float("nan")

    q_abs = result.heat_flows_kW.get("Q_abs", float("nan"))
    # ua_abs_equiv = q_abs / lmtd_abs if (np.isfinite(lmtd_abs) and lmtd_abs > 0) else float("nan")

    return {
        "COP":          cop,
        "ECOP":         ecop,
        "Q_abs":        q_abs,
        "m6":           result.diagnostics.get("m6_kg_s", float("nan")),
        # "UA_abs_equiv": ua_abs_equiv,   # kW/K – proportional zur Heizfläche
    }


# ---------------------------------------------------------------------------
# Parameterstudie mit Warmstart
# ---------------------------------------------------------------------------

def run_sweep() -> dict[float, dict]:
    cfg = SCAN_CONFIG[SCAN_VARIABLE]
    temperatures = np.arange(
        cfg["min_C"], cfg["max_C"] + 0.5 * T_SCAN_STEP_K, T_SCAN_STEP_K
    )
    idx_start = int(np.argmin(np.abs(temperatures - cfg["start_C"])))
    results: dict[float, dict] = {}

    def solve_at(T_C: float, x0_K: np.ndarray):
        inp = build_inputs(T_C)
        res = solve_awt(inp, x0=x0_K)
        next_x0 = result_to_x0(res) if is_converged(res) else x0_K
        return res, next_x0

    x0_K = primary_temperatures_C_to_K(X0_CENTER_C)

    # --- Startpunkt ---
    res_center, _ = solve_at(float(temperatures[idx_start]), x0_K)
    kpi = extract_results(res_center)
    T_c = float(temperatures[idx_start])
    if kpi:
        results[T_c] = kpi
        print(f"  T={T_c:6.1f} °C  m6={kpi['m6']:.4f}  COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
    else:
        print(f"  T={T_c:6.1f} °C  nicht konvergiert (Startpunkt)")

    # --- Aufwärts ---
    x0_up = result_to_x0(res_center) if is_converged(res_center) else x0_K
    for idx in range(idx_start + 1, len(temperatures)):
        T_C = float(temperatures[idx])
        res, x0_up = solve_at(T_C, x0_up)
        kpi = extract_results(res)
        if kpi:
            results[T_C] = kpi
            print(f"  T={T_C:6.1f} °C  m6={kpi['m6']:.4f}  COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
        else:
            print(f"  T={T_C:6.1f} °C  nicht konvergiert")

    # --- Abwärts ---
    x0_down = result_to_x0(res_center) if is_converged(res_center) else x0_K
    for idx in range(idx_start - 1, -1, -1):
        T_C = float(temperatures[idx])
        res, x0_down = solve_at(T_C, x0_down)
        kpi = extract_results(res)
        if kpi:
            results[T_C] = kpi
            print(f"  T={T_C:6.1f} °C  m6={kpi['m6']:.4f}  COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
        else:
            print(f"  T={T_C:6.1f} °C  nicht konvergiert")

    return results



# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results: dict[float, dict]) -> None:
    if not results:
        print("Keine konvergierten Ergebnisse zum Plotten.")
        return

    cfg      = SCAN_CONFIG[SCAN_VARIABLE]
    T_ref    = cfg["start_C"]
    xlabel   = cfg["xlabel"]
    t_suffix = cfg["title_suffix"]

    T_vals      = np.array(sorted(results.keys()))
    cop_vals    = np.array([results[T]["COP"]          for T in T_vals])
    ecop_vals   = np.array([results[T]["ECOP"]         for T in T_vals])
    q_abs_vals = np.array([results[T]["Q_abs"]         for T in T_vals])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.78)

    color_cop  = "#1f77b4"
    color_ecop = "#d62728"
    color_ua   = "#2ca02c"

    ax1.plot(T_vals, cop_vals,  color=color_cop,  lw=2, marker="o", ms=4, label="COP")
    ax1.plot(T_vals, ecop_vals, color=color_ecop, lw=2, marker="s", ms=4, label="ECOP")
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel("COP / ECOP [-]", fontsize=11)
    ax1.axvline(T_ref, color="gray", ls="--", lw=1, alpha=0.6,
                label=f"Referenz {T_ref:.0f} °C")

    ax2 = ax1.twinx()
    ax2.plot(T_vals, q_abs_vals, color=color_ua, lw=2, marker="^", ms=4,
             label="$Q_\\mathrm{Abs}$")
    ax2.set_ylabel("Absorberleistung $Q_\\mathrm{Abs}$ [kW]",
                   color=color_ua, fontsize=11)
    # ax2.set_ylim(59.5, 60.5)
    ax2.set_ylim(5, 250)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

    # ax1.set_title(
    #     f"AHT (Pinch-Modell) – COP, ECOP und Absorber-Heizflächenkennwert\n"
    #     f"über {t_suffix}  ($Q_\\mathrm{{Abs}}$ = {BASE_KWARGS_FIXED['Qabs_spec_kW']:.0f} kW konstant)",
    #     fontsize=11,
    # )
    ax1.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    print("Q_abs min =", np.min(q_abs_vals))
    print("Q_abs max =", np.max(q_abs_vals))
    print("Q_abs span =", np.max(q_abs_vals)-np.min(q_abs_vals))
    plt.show()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SCAN_CONFIG[SCAN_VARIABLE]
    print(f"Scanvariable : {SCAN_VARIABLE}")
    print(f"Bereich      : {cfg['min_C']:.0f} – {cfg['max_C']:.0f} °C")
    print(f"Startpunkt   : {cfg['start_C']:.0f} °C, Schrittweite: {T_SCAN_STEP_K:.0f} K")
    # print(f"Q_abs (fix)  : {BASE_KWARGS_FIXED['Qabs_spec_kW']:.1f} kW\n")
    results = run_sweep()
    n_total = round((cfg["max_C"] - cfg["min_C"]) / T_SCAN_STEP_K) + 1
    print(f"\n{len(results)} von {n_total} Punkten konvergiert.")
    plot_results(results)