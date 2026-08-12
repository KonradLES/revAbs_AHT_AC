from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

"""Parameterstudie mit m6-Optimierung je Betriebspunkt.

Für jeden Scan-Punkt wird m6 so optimiert, dass Q_abs maximal wird.
Die Optimierung startet beim Optimum des Vorschritts (Warmstart),
sucht in einem Fenster [m6_opt * (1 - WINDOW), m6_opt * (1 + WINDOW)]
mit dem Goldenen-Schnitt-Verfahren.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from Models.AHT_Pinch_Point import (
    AWTInputs,
    AWTResult,
    primary_temperatures_C_to_K,
    solve_awt,
    PRIMARY_VARIABLE_NAMES,
)

# ---------------------------------------------------------------------------
# >>>  Scanvariable auswählen  <
# ---------------------------------------------------------------------------
SCAN_VARIABLE = "T_cond_in"   # "T_evap_des" | "T_abs_in" | "T_cond_in"

# ---------------------------------------------------------------------------
# Scan-Bereiche
# ---------------------------------------------------------------------------
SCAN_CONFIG = {
    "T_evap_des": dict(
        min_C=107.0, max_C=130.0, start_C=120.0,
        xlabel="Verdampfer-/Desorbereinlasstemperatur $T_{13} = T_{15}$ [°C]",
        title_suffix="Verdampfer-/Desorbereinlasstemperatur",
    ),
    "T_abs_in": dict(
        min_C=100.0, max_C=160.0, start_C=135.0,
        xlabel="Absorbereinlasstemperatur $T_{11}$ [°C]",
        title_suffix="Absorbereinlasstemperatur $T_{11}$",
    ),
    "T_cond_in": dict(
        min_C=10.0, max_C=43.0, start_C=30.0,
        xlabel="Kondensatoreinlasstemperatur $T_{17}$ [°C]",
        title_suffix="Kondensatoreinlasstemperatur $T_{17}$",
    ),
}

T_SCAN_STEP_K = 1.0

# ---------------------------------------------------------------------------
# m6-Optimierungsparameter
# ---------------------------------------------------------------------------
M6_INITIAL   = 1.0    # Startwert für den ersten Betriebspunkt
M6_ABS_MIN   = 0.05    # absolutes Untergrenze für m6 [kg/s]
M6_ABS_MAX   = 3.0     # absolutes Obergrenze für m6 [kg/s]
M6_WINDOW    = 0.40    # relatives Suchfenster um letztes Optimum (± 40 %)
M6_OPT_XTOL  = 1e-3    # Abbruchtoleranz des Goldenen Schnitts [kg/s]

# ---------------------------------------------------------------------------
# Feste Basiskonfiguration
# ---------------------------------------------------------------------------
BASE_KWARGS_FIXED = dict(
    T_11_C=135.0,
    T_13_C=120.0,
    T_15_C=120.0,
    T_17_C=30.0,
    m_13=4.0,
    m_15=4.0,
    m_17=4.0,
    dT_min_shex=4.3,
    dT_min_des=6.3,
    dT_min_cond=13.8,
    dT_min_evap=19.0,
    dT_min_abs=17.8,
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="m11",
    cycle_scale_spec_mode="m6",
    desorber_evaporator_routing_mode="parallel",
    m11_spec=4.0,
)

X0_CENTER_C = np.array(
            [   55,   # T8  [°C] 55, 29.98, 30
                101,   # T10 [°C] 101, 50.02, 55
                0.23,    # x3  [-] 0.23, 0.15, 0.23
                0.27,    # x6  [-] 0.27, 0.18, 0.27
                0.26,   # x20 [-] 0.26, 0.17, 0.26
                121,   # T2  [°C] 121, 59.50, 70
                150,   # T4  [°C] 150, 68.98, 80
                0.15,     # beta [-] 0.2, 0.1, 0.1
                ],
    dtype=float,
)

LOSS_COMPONENTS = ["E_abs", "E_des", "E_evap", "E_cond", "E_SHEX", "E_throttle"]
LOSS_LABELS     = ["Absorber", "Desorber", "Verdampfer", "Kondensator", "SHEX", "Drossel"]
LOSS_COLORS     = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def build_inputs(scan_value_C: float, m6: float) -> AWTInputs:
    kwargs = BASE_KWARGS_FIXED.copy()
    kwargs["m6_spec"] = m6
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


def is_converged(result: AWTResult, tol: float = 1e-4) -> bool:
    return (
        result.solve_info.final_point_evaluable
        and result.solve_info.scaled_residual_norm < tol
    )


def result_to_x0(result: AWTResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def extract_results(result: AWTResult, m6_opt: float) -> dict | None:
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
        "m6_opt":     m6_opt,
        "loss_fracs": loss_fracs_norm,
    }


# ---------------------------------------------------------------------------
# m6-Optimierung für einen einzelnen Betriebspunkt
# ---------------------------------------------------------------------------

def optimize_m6_at(
    scan_T_C: float,
    m6_seed: float,
    x0_seed_K: np.ndarray,
) -> tuple[float, AWTResult | None, np.ndarray]:
    """Optimiert m6 für einen Betriebspunkt via Goldener Schnitt.

    Gibt (m6_opt, best_result, best_x0_K) zurück.
    best_result ist None, wenn kein einziger Punkt konvergiert.
    """
    # Suchfenster um letztes Optimum, begrenzt durch absolute Schranken
    lo = max(M6_ABS_MIN, m6_seed * (1.0 - M6_WINDOW))
    hi = min(M6_ABS_MAX, m6_seed * (1.0 + M6_WINDOW))

    # Cache: x0 für jeden m6-Aufruf warm halten
    # Der Solver startet immer beim zuletzt konvergierten Zustand.
    warm_x0: dict[str, np.ndarray] = {"current": x0_seed_K.copy()}
    best_store: dict[str, object] = {"result": None, "x0": x0_seed_K.copy(), "m6": m6_seed}

    def neg_q_abs(m6: float) -> float:
        """Zielfunktion für minimize_scalar (minimiert → maximiert Q_abs)."""
        inp = build_inputs(scan_T_C, m6)
        res = solve_awt(inp, x0=warm_x0["current"])
        if is_converged(res):
            warm_x0["current"] = result_to_x0(res)
            q = res.heat_flows_kW.get("Q_abs", float("nan"))
            # bestes Ergebnis merken
            best = best_store["result"]
            if best is None or q > best.heat_flows_kW.get("Q_abs", float("-inf")):  # type: ignore[union-attr]
                best_store["result"] = res
                best_store["x0"]     = result_to_x0(res)
                best_store["m6"]     = m6
            return -q if not np.isnan(q) else 0.0
        return 0.0   # nicht konvergiert → schlechter Wert

    minimize_scalar(
        neg_q_abs,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": M6_OPT_XTOL},
    )

    m6_best   = float(best_store["m6"])
    res_best  = best_store["result"]        # type: ignore[assignment]
    x0_best   = best_store["x0"]            # type: ignore[assignment]
    return m6_best, res_best, x0_best       # type: ignore[return-value]


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

    def solve_and_store(T_C: float, m6_seed: float, x0_K: np.ndarray):
        """Optimiert m6, speichert Ergebnis; gibt (m6_opt, next_x0) zurück."""
        m6_opt, res, x0_opt = optimize_m6_at(T_C, m6_seed, x0_K)
        if res is not None and is_converged(res):
            kpi = extract_results(res, m6_opt)
            if kpi is not None:
                results[T_C] = kpi
                print(
                    f"  T={T_C:6.1f} °C  m6={m6_opt:.4f}  "
                    f"Q_abs={kpi['Q_abs']:.3f} kW  "
                    f"COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}"
                )
                return m6_opt, x0_opt
        print(f"  T={T_C:6.1f} °C  nicht konvergiert")
        return m6_seed, x0_K   # Fallback: Seed unverändert weitergeben

    x0_K = primary_temperatures_C_to_K(X0_CENTER_C)

    # --- Startpunkt ---
    m6_center, x0_center = solve_and_store(float(temperatures[idx_start]), M6_INITIAL, x0_K)

    # --- Aufwärts ---
    m6_up, x0_up = m6_center, x0_center
    for idx in range(idx_start + 1, len(temperatures)):
        m6_up, x0_up = solve_and_store(float(temperatures[idx]), m6_up, x0_up)

    # --- Abwärts ---
    m6_down, x0_down = m6_center, x0_center
    for idx in range(idx_start - 1, -1, -1):
        m6_down, x0_down = solve_and_store(float(temperatures[idx]), m6_down, x0_down)

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

    T_vals     = np.array(sorted(results.keys()))
    cop_vals   = np.array([results[T]["COP"]    for T in T_vals])
    ecop_vals  = np.array([results[T]["ECOP"]   for T in T_vals])
    q_abs_vals = np.array([results[T]["Q_abs"]  for T in T_vals])
    m6_vals    = np.array([results[T]["m6_opt"] for T in T_vals])

    # -----------------------------------------------------------------------
    # Plot 1: COP + ECOP (links) | Q_abs (rechts innen) | T12 (rechts außen)
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.subplots_adjust(right=0.78)

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

    ax2 = ax1.twinx()
    ax2.plot(T_vals, q_abs_vals, color=color_qabs, lw=2, marker="^", ms=4,
             label="$Q_\\mathrm{Abs}$")
    ax2.set_ylabel("Absorberleistung $Q_\\mathrm{Abs}$ [kW]", color=color_qabs, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_qabs)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="best", fontsize=10)
    ax1.set_title(f"AWT – COP, ECOP, $Q_\\mathrm{{Abs}}$ und $T_{{12}}$ über {t_suffix}"
                  f"\n(m6 je Punkt auf max. $Q_\\mathrm{{Abs}}$ optimiert)", fontsize=11)
    ax1.grid(True, ls="--", alpha=0.4)
    fig1.tight_layout()
    plt.savefig("Auswertung_KO/Results_Plots/AHT_Pinch_Point_Performance_ref.png", dpi=150, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Plot 2: Exergieverlustanteile
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
    ax.set_title(f"AWT – Exergieverlustanteile über {t_suffix}"
                 f"\n(m6 je Punkt auf max. $Q_\\mathrm{{Abs}}$ optimiert)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, ls="--", alpha=0.3)
    fig2.tight_layout()
    plt.savefig("Auswertung_KO/Results_Plots/AHT_Pinch_Point_Exergy_ref.png", dpi=150, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Plot 3: optimales m6 über Scantemperatur (Kontrollplot)
    # -----------------------------------------------------------------------
    fig3, ax_m6 = plt.subplots(figsize=(8, 4))
    ax_m6.plot(T_vals, m6_vals, color="#8c564b", lw=2, marker="o", ms=4)
    ax_m6.set_xlabel(xlabel, fontsize=11)
    ax_m6.set_ylabel("Optimales $m_6$ [kg/s]", fontsize=11)
    ax_m6.set_title(f"AWT – Optimales $m_6$ über {t_suffix}", fontsize=11)
    ax_m6.axvline(T_ref, color="gray", ls="--", lw=1, alpha=0.6)
    ax_m6.grid(True, ls="--", alpha=0.4)
    fig3.tight_layout()
    plt.savefig("Auswertung_KO/Results_Plots/AHT_Pinch_Point_mass_flow_opt_ref.png", dpi=150, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SCAN_CONFIG[SCAN_VARIABLE]
    print(f"Scanvariable  : {SCAN_VARIABLE}")
    print(f"Bereich       : {cfg['min_C']:.0f} – {cfg['max_C']:.0f} °C")
    print(f"Startpunkt    : {cfg['start_C']:.0f} °C,  Schrittweite: {T_SCAN_STEP_K:.0f} K")
    print(f"m6-Fenster    : ±{M6_WINDOW*100:.0f} % um letztes Optimum  "
          f"[abs. {M6_ABS_MIN}–{M6_ABS_MAX} kg/s]\n")
    results = run_sweep()
    n_total = round((cfg["max_C"] - cfg["min_C"]) / T_SCAN_STEP_K) + 1
    print(f"\n{len(results)} von {n_total} Punkten konvergiert.")
    plot_results(results)