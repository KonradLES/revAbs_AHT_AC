from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

"""Parameterstudie: COP, ECOP und Exergieverlustanteile.

Auswahl der zu variierenden Temperatur über SCAN_VARIABLE:
  "T_evap_des"  -> T_13_C = T_15_C  (Verdampfer-/Desorbereinlass)
  "T_abs_in"    -> T_11_C            (Absorbereinlass)
  "T_cond_in"   -> T_17_C            (Kondensatoreinlass)
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
# >>>  Hier die gewünschte Scanvariable auswählen  <
# ---------------------------------------------------------------------------
SCAN_VARIABLE = "T_evap_des"   # "T_evap_des" | "T_abs_in" | "T_cond_in"

# ---------------------------------------------------------------------------
# Scan-Bereiche je Variable  [Min, Max, Startpunkt]  alle in °C
# ---------------------------------------------------------------------------
SCAN_CONFIG = {
    "T_evap_des": dict(
        min_C=60.0, max_C=70.0, start_C=65.0,
        xlabel="Verdampfer-/Desorbereinlasstemperatur $T_{13} = T_{15}$ [°C]",
        title_suffix="Verdampfer-/Desorbereinlasstemperatur",
    ),
    "T_abs_in": dict(
        min_C=100.0, max_C=160.0, start_C=135.0,
        xlabel="Absorbereinlasstemperatur $T_{11}$ [°C]",
        title_suffix="Absorbereinlasstemperatur $T_{11}$",
    ),
    "T_cond_in": dict(
        min_C=10.0, max_C=50.0, start_C=30.0,
        xlabel="Kondensatoreinlasstemperatur $T_{17}$ [°C]",
        title_suffix="Kondensatoreinlasstemperatur $T_{17}$",
    ),
}

T_SCAN_STEP_K = 1.0

# ---------------------------------------------------------------------------
# Feste Basiskonfiguration (Werte, die nie variiert werden)
# ---------------------------------------------------------------------------
BASE_KWARGS_FIXED = dict(
    T_11_C=85.0,
    T_13_C=65.0,
    T_15_C=65.0,
    T_17_C=20.0,
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
    m6_spec=0.45,
)

X0_CENTER_C = np.array([30.1458, 57.2958, 0.1787, 0.2003, 0.1981, 66.1183, 85.6828, 0.0905], dtype=float)

# Komponenten-Reihenfolge im gestapelten Plot
LOSS_COMPONENTS = ["E_abs", "E_des", "E_evap", "E_cond", "E_SHEX", "E_throttle"]
LOSS_LABELS     = ["Absorber", "Desorber", "Verdampfer", "Kondensator", "SHEX", "Drossel"]
LOSS_COLORS     = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def build_inputs(scan_value_C: float) -> AWTInputs:
    """Erstellt AWTInputs mit dem aktuellen Scanwert; alle anderen Werte aus BASE_KWARGS_FIXED."""
    kwargs = BASE_KWARGS_FIXED.copy()
    if SCAN_VARIABLE == "T_evap_des":
        kwargs["T_13_C"] = scan_value_C
        kwargs["T_15_C"] = scan_value_C
    elif SCAN_VARIABLE == "T_abs_in":
        kwargs["T_11_C"] = scan_value_C
    elif SCAN_VARIABLE == "T_cond_in":
        kwargs["T_17_C"] = scan_value_C
    else:
        raise ValueError(f"Unbekannte SCAN_VARIABLE: {SCAN_VARIABLE!r}")
    return AWTInputs(**kwargs)


def extract_results(result: AWTResult) -> dict | None:
    if not result.solve_info.final_point_evaluable:
        return None

    cop  = result.kpis.get("COP", float("nan"))
    ecop = result.exergy_kW.get("Exergy_efficiency", float("nan"))
    if np.isnan(cop) or np.isnan(ecop):
        return None

    loss_fracs: dict[str, float] = {}
    for key in LOSS_COMPONENTS:
        raw = result.exergy_kW.get(key, "")
        try:
            pct = float(str(raw).split("(")[1].replace("%", "").replace(")", "").strip())
        except Exception:
            pct = float("nan")
        loss_fracs[key] = pct / 100.0

    total = sum(v for v in loss_fracs.values() if not np.isnan(v))
    if total <= 0.0:
        return None
    loss_fracs_norm = {k: v / total for k, v in loss_fracs.items()}

    return {
            "COP":        cop,
            "ECOP":       ecop,
            "Q_abs":      result.heat_flows_kW.get("Q_abs", float("nan")),
            "loss_fracs": loss_fracs_norm,
        }

def result_to_x0(result: AWTResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def is_converged(result: AWTResult, tol: float = 1e-4) -> bool:
    return (
        result.solve_info.final_point_evaluable
        and result.solve_info.scaled_residual_norm < tol
    )


# ---------------------------------------------------------------------------
# Parameterstudie
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

    # Startpunkt
    res_center, _ = solve_at(float(temperatures[idx_start]), x0_K)
    kpi = extract_results(res_center)
    T_c = float(temperatures[idx_start])
    if kpi:
        results[T_c] = kpi
        print(f"  T={T_c:6.1f} °C  COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
    else:
        print(f"  T={T_c:6.1f} °C  nicht konvergiert (Startpunkt)")

    # Aufwärts
    x0_up = result_to_x0(res_center) if is_converged(res_center) else x0_K
    for idx in range(idx_start + 1, len(temperatures)):
        T_C = float(temperatures[idx])
        res, x0_up = solve_at(T_C, x0_up)
        kpi = extract_results(res)
        if kpi:
            results[T_C] = kpi
            print(f"  T={T_C:6.1f} °C  COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
        else:
            print(f"  T={T_C:6.1f} °C  nicht konvergiert")

    # Abwärts
    x0_down = result_to_x0(res_center) if is_converged(res_center) else x0_K
    for idx in range(idx_start - 1, -1, -1):
        T_C = float(temperatures[idx])
        res, x0_down = solve_at(T_C, x0_down)
        kpi = extract_results(res)
        if kpi:
            results[T_C] = kpi
            print(f"  T={T_C:6.1f} °C  COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
        else:
            print(f"  T={T_C:6.1f} °C  nicht konvergiert")

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(results: dict[float, dict]) -> None:
    if not results:
        print("Keine konvergierten Ergebnisse zum Plotten.")
        return

    cfg       = SCAN_CONFIG[SCAN_VARIABLE]
    T_ref     = cfg["start_C"]
    xlabel    = cfg["xlabel"]
    t_suffix  = cfg["title_suffix"]

    T_vals    = np.array(sorted(results.keys()))
    cop_vals  = np.array([results[T]["COP"]   for T in T_vals])
    ecop_vals = np.array([results[T]["ECOP"]  for T in T_vals])
    q_abs_vals= np.array([results[T]["Q_abs"] for T in T_vals])

# -----------------------------------------------------------------------
    # Plot 1: COP + ECOP (links) | Q_abs (rechts innen) | T12 (rechts außen)
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.subplots_adjust(right=0.78)   # Platz für zwei rechte Achsen

    color_cop  = "#1f77b4"
    color_ecop = "#d62728"
    color_qabs = "#2ca02c"
    color_t12  = "#9467bd"

    ax1.plot(T_vals, cop_vals,  color=color_cop,  lw=2, marker="o", ms=4, label="COP")
    ax1.plot(T_vals, ecop_vals, color=color_ecop, lw=2, marker="s", ms=4, label="ECOP")
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel("COP / ECOP [-]", fontsize=11)
    ax1.axvline(T_ref, color="gray", ls="--", lw=1, alpha=0.6,
                label=f"Referenz {T_ref:.0f} °C")

    # Rechte Achse 1: Q_abs
    ax2 = ax1.twinx()
    q_abs_vals = np.array([results[T]["Q_abs"] for T in T_vals])
    ax2.plot(T_vals, q_abs_vals, color=color_qabs, lw=2, marker="^", ms=4,
             label="$Q_\\mathrm{Abs}$")
    ax2.set_ylabel("Absorberleistung $Q_\\mathrm{Abs}$ [kW]", color=color_qabs, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_qabs)

    # Gemeinsame Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="best", fontsize=10)

    ax1.set_title(f"AWT – COP, ECOP, Absorberleistung und $T_{{12}}$ über {t_suffix}",
                  fontsize=11)
    ax1.grid(True, ls="--", alpha=0.4)
    fig1.tight_layout()

    # -----------------------------------------------------------------------
    # Plot 2: ECOP + gestapelte Exergieverlustanteile
    # -----------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(9, 5))

    gap = 1.0 - ecop_vals
    loss_abs = {}
    for key in LOSS_COMPONENTS:
        fracs = np.array([results[T]["loss_fracs"].get(key, 0.0) for T in T_vals])
        loss_abs[key] = fracs * gap

    bottom = ecop_vals.copy()
    for key, label, color in zip(LOSS_COMPONENTS, LOSS_LABELS, LOSS_COLORS):
        top = bottom + loss_abs[key]
        ax.fill_between(T_vals, bottom, top, color=color, alpha=0.82, label=label)
        bottom = top

    ax.plot(T_vals, ecop_vals, color="black", lw=2.0, label="ECOP", zorder=5)
    ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.5, label="Ideale Effizienz (1)")
    ax.axvline(T_ref, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"Referenz {T_ref:.0f} °C")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Exergetische Effizienz / Verlustanteile [-]", fontsize=11)
    ax.set_title(f"AWT – Exergieverlustanteile über {t_suffix}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, ls="--", alpha=0.3)
    fig2.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SCAN_CONFIG[SCAN_VARIABLE]
    print(f"Scanvariable  : {SCAN_VARIABLE}")
    print(f"Bereich       : {cfg['min_C']:.0f} – {cfg['max_C']:.0f} °C")
    print(f"Startpunkt    : {cfg['start_C']:.0f} °C,  Schrittweite: {T_SCAN_STEP_K:.0f} K\n")
    results = run_sweep()
    n_total = round((cfg["max_C"] - cfg["min_C"]) / T_SCAN_STEP_K) + 1
    print(f"\n{len(results)} von {n_total} Punkten konvergiert.")
    plot_results(results)