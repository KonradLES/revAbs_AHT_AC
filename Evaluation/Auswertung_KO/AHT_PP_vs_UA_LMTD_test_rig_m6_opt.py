from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

"""Vergleichsstudie: UA/LMTD-Modell vs. Pinch-Point-Modell.

Beide Modelle werden über denselben Temperaturbereich gescannt,
m6 wird je Betriebspunkt auf maximale Q_abs optimiert.
Die Ergebnisse werden direkt gegenübergestellt.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from Models.AHT_UA_LMTD import (
    AWTInputs as AWTInputs_UA,
    AWTResult as AWTResult_UA,
    primary_temperatures_C_to_K,
    solve_awt as solve_awt_UA,
    PRIMARY_VARIABLE_NAMES,
)
from Models.AHT_Pinch_Point import (
    AWTInputs as AWTInputs_PP,
    AWTResult as AWTResult_PP,
    solve_awt as solve_awt_PP,
)
# ---------------------------------------------------------------------------
# >>>  Name des Vergleichs  <
# ---------------------------------------------------------------------------

filename =  "PP_at_NTU_Absorber_Temp"

# ---------------------------------------------------------------------------
# >>>  Scanvariable auswählen  <
# ---------------------------------------------------------------------------
SCAN_VARIABLE = "T_abs_in"   # "T_evap_des" | "T_abs_in" | "T_cond_in"

SCAN_CONFIG = {
    "T_evap_des": dict(
        min_C=61.0, max_C=75.0, start_C=65.0,
        xlabel="Verdampfer-/Desorbereinlasstemperatur $T_{13} = T_{15}$ [°C]",
        title_suffix="Verdampfer-/Desorbereinlasstemperatur",
    ),
    "T_abs_in": dict(
        min_C=70.0, max_C=100.0, start_C=85.0,
        xlabel="Absorbereinlasstemperatur $T_{11}$ [°C]",
        title_suffix="Absorbereinlasstemperatur $T_{11}$",
    ),
    "T_cond_in": dict(
        min_C=10.0, max_C=30.0, start_C=20.0,
        xlabel="Kondensatoreinlasstemperatur $T_{17}$ [°C]",
        title_suffix="Kondensatoreinlasstemperatur $T_{17}$",
    ),
}

T_SCAN_STEP_K = 1.0

# ---------------------------------------------------------------------------
# m6-Optimierungsparameter (gemeinsam für beide Modelle)
# ---------------------------------------------------------------------------
M6_INITIAL  = 0.45
M6_ABS_MIN  = 0.05
M6_ABS_MAX  = 3.0
M6_WINDOW   = 0.40
M6_OPT_XTOL = 1e-3

# ---------------------------------------------------------------------------
# Gemeinsame externe Randbedingungen
# ---------------------------------------------------------------------------
COMMON_BOUNDARY = dict(
    T_11_C=85.0,
    T_13_C=65.0,
    T_15_C=65.0,
    T_17_C=20.0,
    m_13=4.0,
    m_15=4.0,
    m_17=4.0,
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="m11",
    cycle_scale_spec_mode="m6",
    desorber_evaporator_routing_mode="parallel",
    m11_spec=4.0,
)

# Modellspezifische Parameter
UA_PARAMS = dict(
    shex_model="NTU",
    UA_cond=0.5*10.0,
    UA_evap=0.5*15.0,
    UA_abs=0.5*10.0,
    UA_des=0.5*25.0,
    # UA_shex=0.5*70.8 / 6.43,
    Effectiveness_shex=0.9,
)

PP_PARAMS = dict(
    dT_min_shex=3,
    dT_min_des=5,
    dT_min_cond=5,
    dT_min_evap=5,
    dT_min_abs=5,
)

X0_CENTER_C = np.array(
    [30.1458, 57.2958, 0.1787, 0.2003, 0.1981, 66.1183, 85.6828, 0.0905],
    dtype=float,
)

LOSS_COMPONENTS = ["E_abs", "E_des", "E_evap", "E_cond", "E_SHEX", "E_throttle"]
LOSS_LABELS     = ["Absorber", "Desorber", "Verdampfer", "Kondensator", "SHEX", "Drossel"]
LOSS_COLORS     = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]


# ---------------------------------------------------------------------------
# Eingaben bauen
# ---------------------------------------------------------------------------

def _apply_scan(kwargs: dict, scan_value_C: float) -> dict:
    if SCAN_VARIABLE == "T_evap_des":
        kwargs["T_13_C"] = scan_value_C
        kwargs["T_15_C"] = scan_value_C
    elif SCAN_VARIABLE == "T_abs_in":
        kwargs["T_11_C"] = scan_value_C
    elif SCAN_VARIABLE == "T_cond_in":
        kwargs["T_17_C"] = scan_value_C
    return kwargs

def build_inputs_UA(scan_value_C: float, m6: float) -> AWTInputs_UA:
    kw = {**COMMON_BOUNDARY, **UA_PARAMS}
    kw["m6_spec"] = m6
    return AWTInputs_UA(**_apply_scan(kw, scan_value_C))

def build_inputs_PP(scan_value_C: float, m6: float) -> AWTInputs_PP:
    kw = {**COMMON_BOUNDARY, **PP_PARAMS}
    kw["m6_spec"] = m6
    return AWTInputs_PP(**_apply_scan(kw, scan_value_C))


# ---------------------------------------------------------------------------
# Konvergenz und Extraktion
# ---------------------------------------------------------------------------

def is_converged(result, tol: float = 1e-4) -> bool:
    return (
        result.solve_info.final_point_evaluable
        and result.solve_info.scaled_residual_norm < tol
    )

def result_to_x0(result) -> np.ndarray:
    return np.array(
        [result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES], dtype=float
    )

def extract_results(result, m6_opt: float, model: str = "UA") -> dict | None:
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

    # Minimaler Pinch aus diagnostics (nur UA-Modell)
    min_pinch_val = float("nan")
    min_pinch_loc = ""
    if model == "UA":
        diag = result.diagnostics
        pinch_candidates = {
            "SHEX (1)":       diag.get("deltaT_shex_1_K", float("inf")),
            "SHEX (2)":       diag.get("deltaT_shex_2_K", float("inf")),
            "Desorber (1)":   diag.get("deltaT_des_1_K",  float("inf")),
            "Desorber (2)":   diag.get("deltaT_des_2_K",  float("inf")),
            "Kondensator (1)":diag.get("deltaT_cond_1_K", float("inf")),
            "Kondensator (2)":diag.get("deltaT_cond_2_K", float("inf")),
            "Verdampfer (1)": diag.get("deltaT_evap_1_K", float("inf")),
            "Verdampfer (2)": diag.get("deltaT_evap_2_K", float("inf")),
            "Absorber (1)":   diag.get("deltaT_abs_1_K",  float("inf")),
            "Absorber (2)":   diag.get("deltaT_abs_2_K",  float("inf")),
        }
        min_pinch_loc = min(pinch_candidates, key=pinch_candidates.get)
        min_pinch_val = pinch_candidates[min_pinch_loc]

    return {
        "COP":           cop,
        "ECOP":          ecop,
        "Q_abs":         result.heat_flows_kW.get("Q_abs", float("nan")),
        "m6_opt":        m6_opt,
        "loss_fracs":    {k: v / total for k, v in loss_fracs.items()},
        "min_pinch_K":   min_pinch_val,
        "min_pinch_loc": min_pinch_loc,
    }

# ---------------------------------------------------------------------------
# m6-Optimierung (modellunabhängig über Callback)
# ---------------------------------------------------------------------------

def optimize_m6_at(scan_T_C, m6_seed, x0_seed_K, build_fn, solve_fn, label):
    lo = max(M6_ABS_MIN, m6_seed * (1.0 - M6_WINDOW))
    hi = min(M6_ABS_MAX, m6_seed * (1.0 + M6_WINDOW))

    warm_x0   = {"current": x0_seed_K.copy()}
    best_store = {"result": None, "x0": x0_seed_K.copy(), "m6": m6_seed}

    def neg_q_abs(m6):
        res = solve_fn(build_fn(scan_T_C, m6), x0=warm_x0["current"])
        if is_converged(res):
            warm_x0["current"] = result_to_x0(res)
            q = res.heat_flows_kW.get("Q_abs", float("nan"))
            best = best_store["result"]
            if best is None or q > best.heat_flows_kW.get("Q_abs", float("-inf")):
                best_store.update(result=res, x0=result_to_x0(res), m6=m6)
            return -q if not np.isnan(q) else 0.0
        return 0.0

    minimize_scalar(neg_q_abs, bounds=(lo, hi), method="bounded",
                    options={"xatol": M6_OPT_XTOL})

    return float(best_store["m6"]), best_store["result"], best_store["x0"]


# ---------------------------------------------------------------------------
# Sweep für ein Modell
# ---------------------------------------------------------------------------

def run_sweep(build_fn, solve_fn, label: str, model: str = "UA") -> dict[float, dict]:
    cfg = SCAN_CONFIG[SCAN_VARIABLE]
    temperatures = np.arange(
        cfg["min_C"], cfg["max_C"] + 0.5 * T_SCAN_STEP_K, T_SCAN_STEP_K
    )
    idx_start = int(np.argmin(np.abs(temperatures - cfg["start_C"])))
    results: dict[float, dict] = {}

    def solve_and_store(T_C, m6_seed, x0_K):
        m6_opt, res, x0_opt = optimize_m6_at(T_C, m6_seed, x0_K, build_fn, solve_fn, label)
        if res is None or not is_converged(res):          # ← None-Guard hier
            print(f"  [{label}] T={T_C:5.1f}°C  nicht konvergiert")
            return m6_seed, x0_K
        kpi = extract_results(res, m6_opt, model=model)  # ← model weitergeben
        if kpi is not None:
            results[T_C] = kpi
            print(f"  [{label}] T={T_C:5.1f}°C  m6={m6_opt:.3f}  "
                  f"Q_abs={kpi['Q_abs']:.2f} kW  "
                  f"COP={kpi['COP']:.4f}  ECOP={kpi['ECOP']:.4f}")
            return m6_opt, x0_opt
        print(f"  [{label}] T={T_C:5.1f}°C  nicht konvergiert")
        return m6_seed, x0_K

    x0_K = primary_temperatures_C_to_K(X0_CENTER_C)
    m6_c, x0_c = solve_and_store(float(temperatures[idx_start]), M6_INITIAL, x0_K)

    m6_up, x0_up = m6_c, x0_c
    for idx in range(idx_start + 1, len(temperatures)):
        m6_up, x0_up = solve_and_store(float(temperatures[idx]), m6_up, x0_up)

    m6_dn, x0_dn = m6_c, x0_c
    for idx in range(idx_start - 1, -1, -1):
        m6_dn, x0_dn = solve_and_store(float(temperatures[idx]), m6_dn, x0_dn)

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(res_UA: dict, res_PP: dict) -> None:
    cfg      = SCAN_CONFIG[SCAN_VARIABLE]
    T_ref    = cfg["start_C"]
    xlabel   = cfg["xlabel"]
    t_suffix = cfg["title_suffix"]

    def arrays(res):
        T = np.array(sorted(res.keys()))
        return (T,
                np.array([res[t]["COP"]          for t in T]),
                np.array([res[t]["ECOP"]         for t in T]),
                np.array([res[t]["Q_abs"]        for t in T]),
                np.array([res[t]["m6_opt"]       for t in T]),
                [res[t]["min_pinch_K"]   for t in T],
                [res[t]["min_pinch_loc"] for t in T])

    T_ua, cop_ua, ecop_ua, qabs_ua, m6_ua, pinch_ua, ploc_ua = arrays(res_UA)
    T_pp, cop_pp, ecop_pp, qabs_pp, m6_pp, _,        _       = arrays(res_PP)

    c_ua    = "#1f77b4"
    c_pp    = "#d62728"
    c_q     = "#2ca02c"
    c_pinch = "#8c564b"
    ls_ua   = "-"
    ls_pp   = "--"

    # -----------------------------------------------------------------------
    # Plot 1: COP + ECOP (links) | Q_abs (rechts innen) | Pinch_min UA (rechts außen)
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    fig1.subplots_adjust(right=0.75)

    ax1.plot(T_ua, cop_ua,  color=c_ua, ls=ls_ua, lw=2, marker="o", ms=4, label="COP  (UA)")
    ax1.plot(T_pp, cop_pp,  color=c_ua, ls=ls_pp, lw=2, marker="o", ms=4, label="COP  (PP)")
    ax1.plot(T_ua, ecop_ua, color=c_pp, ls=ls_ua, lw=2, marker="s", ms=4, label="ECOP (UA)")
    ax1.plot(T_pp, ecop_pp, color=c_pp, ls=ls_pp, lw=2, marker="s", ms=4, label="ECOP (PP)")
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel("COP / ECOP [-]", fontsize=11)
    ax1.axvline(T_ref, color="gray", ls=":", lw=1, alpha=0.6,
                label=f"Referenz {T_ref:.0f} °C")

    # Rechte Achse 1: Q_abs
    ax2 = ax1.twinx()
    ax2.plot(T_ua, qabs_ua, color=c_q, ls=ls_ua, lw=2, marker="^", ms=4,
            label="$Q_\\mathrm{Abs}$ (UA)")
    ax2.plot(T_pp, qabs_pp, color=c_q, ls=ls_pp, lw=2, marker="^", ms=4,
            label="$Q_\\mathrm{Abs}$ (PP)")
    ax2.set_ylabel("Absorberleistung $Q_\\mathrm{Abs}$ [kW]", color=c_q, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=c_q)

    # Rechte Achse 2: minimaler Pinch UA – nach außen versetzt
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.18))
    pinch_vals = np.array([v if not np.isnan(v) else np.nan for v in pinch_ua],
                        dtype=float)
    ax3.plot(T_ua, pinch_vals, color=c_pinch, ls=ls_ua, lw=2,
            marker="D", ms=5, label="$\\Delta T_{\\min}$ UA [K]")
    ax3.set_ylabel("Minimaler Pinch $\\Delta T_{\\min}$ [K]", color=c_pinch, fontsize=11)
    ax3.tick_params(axis="y", labelcolor=c_pinch)

    # Pinch-Ort als Text über jeden Datenpunkt (nur jeden 2. um Überlappung zu vermeiden)
    for i, (T_i, pv, loc) in enumerate(zip(T_ua, pinch_vals, ploc_ua)):
        if i % 2 == 0 and not np.isnan(pv):
            ax3.annotate(
                loc,
                xy=(T_i, pv),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6.5,
                color=c_pinch,
                rotation=45,
            )

    # Gemeinsame Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
            loc="best", fontsize=8, ncol=2)

    ax1.set_title(
        f"Modellvergleich UA vs. PP – COP, ECOP, $Q_\\mathrm{{Abs}}$, "
        f"$\\Delta T_{{\\min}}$ (UA)\nüber {t_suffix}",
        fontsize=11,
    )
    ax1.grid(True, ls="--", alpha=0.4)
    fig1.tight_layout()
    plt.savefig(
        f"Auswertung_KO/Results_Plots/Vergleich_Performance_{filename}.png",
        dpi=150, bbox_inches="tight",
    )

    # -----------------------------------------------------------------------
    # Plot 2: Exergieverluste nebeneinander (2 Subplots)
    # -----------------------------------------------------------------------
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, res, T_vals, ecop_vals, title in [
        (axL, res_UA, T_ua, ecop_ua, "UA/LMTD-Modell"),
        (axR, res_PP, T_pp, ecop_pp, "Pinch-Point-Modell"),
    ]:
        gap = 1.0 - ecop_vals
        bottom = ecop_vals.copy()
        for key, lbl, color in zip(LOSS_COMPONENTS, LOSS_LABELS, LOSS_COLORS):
            fracs = np.array([res[T]["loss_fracs"].get(key, 0.0) for T in T_vals])
            top = bottom + fracs * gap
            ax.fill_between(T_vals, bottom, top, color=color, alpha=0.82, label=lbl)
            bottom = top
        ax.plot(T_vals, ecop_vals, color="black", lw=2.0, label="ECOP", zorder=5)
        ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.5, label="Ideal (1)")
        ax.axvline(T_ref, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls="--", alpha=0.3)

    axL.set_ylabel("Exergetische Effizienz / Verlustanteile [-]", fontsize=11)
    axL.legend(loc="lower left", fontsize=8, ncol=2)
    fig2.suptitle(f"Exergieverlustanteile – Modellvergleich\nüber {t_suffix}", fontsize=12)
    fig2.tight_layout()
    plt.savefig(f"Auswertung_KO/Results_Plots/Vergleich_Exergy_{filename}.png", dpi=150, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Plot 3: optimales m6
    # -----------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(T_ua, m6_ua, color=c_ua, ls=ls_ua, lw=2, marker="o", ms=4, label="UA/LMTD")
    ax3.plot(T_pp, m6_pp, color=c_pp, ls=ls_pp, lw=2, marker="o", ms=4, label="Pinch-Point")
    ax3.set_xlabel(xlabel, fontsize=11)
    ax3.set_ylabel("Optimales $m_6$ [kg/s]", fontsize=11)
    ax3.set_title(f"Optimales $m_6$ – Modellvergleich\nüber {t_suffix}", fontsize=11)
    ax3.axvline(T_ref, color="gray", ls="--", lw=1, alpha=0.6)
    ax3.legend(fontsize=10)
    ax3.grid(True, ls="--", alpha=0.4)
    fig3.tight_layout()
    plt.savefig(f"Auswertung_KO/Results_Plots/Vergleich_m6_{filename}.png", dpi=150, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SCAN_CONFIG[SCAN_VARIABLE]
    print(f"Scanvariable : {SCAN_VARIABLE}")
    print(f"Bereich      : {cfg['min_C']:.0f} – {cfg['max_C']:.0f} °C")
    print(f"Startpunkt   : {cfg['start_C']:.0f} °C,  Schrittweite: {T_SCAN_STEP_K:.0f} K\n")

    print("=== UA/LMTD-Modell ===")
    results_UA = run_sweep(build_inputs_UA, solve_awt_UA, "UA")

    print("\n=== Pinch-Point-Modell ===")
    results_PP = run_sweep(build_inputs_PP, solve_awt_PP, "PP")

    n_total = round((cfg["max_C"] - cfg["min_C"]) / T_SCAN_STEP_K) + 1
    print(f"\nUA: {len(results_UA)}/{n_total} konvergiert")
    print(f"PP: {len(results_PP)}/{n_total} konvergiert")

    plot_comparison(results_UA, results_PP)