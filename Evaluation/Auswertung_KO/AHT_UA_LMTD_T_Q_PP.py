from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

"""Pinch-Diagramme für alle fünf Wärmeübertrager des AWT.

Für jeden Apparat wird ein Q-T-Diagramm (Gegenstrom) erstellt:
  - x-Achse: übertragene Wärmeleistung Q [kW], von 0 bis Q_total
  - y-Achse: Temperatur T [°C]
  - heiße Seite läuft von links (Eintritt) nach rechts (Austritt)
  - kalte Seite läuft von rechts (Eintritt) nach links (Austritt)
    → im Gegenstrom zeigen sich beide Kurven gegenläufig

Apparate und Zustandspunkte
---------------------------
  SHEX      : heiß 3→2,  kalt 5→4
  Desorber  : heiß 13→14, kalt 1→6
  Kondensator: heiß 7→8,  kalt 17→18
  Verdampfer : heiß 15→16, kalt 9→10  (Verdampfung bei konst. T)
  Absorber  : heiß 20→3, kalt 11→12
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Models.AHT_UA_LMTD import (
    AWTInputs,
    AWTResult,
    primary_temperatures_C_to_K,
    solve_awt,
    print_summary,
)

# ---------------------------------------------------------------------------
# >>>  Betriebspunkt hier einstellen  <
# ---------------------------------------------------------------------------
INPUTS = AWTInputs(
    T_11_C=135.0,   # 135, 60, 80
    T_13_C=120.0,   # 120, 60
    T_15_C=120.0,   # 120, 60
    T_17_C=30.0,   # 30, 20, 20
    m_13=4,      # 4, 0.2, 4
    m_15=4,      # 4, 0.2, 4
    m_17=4,      # 4, 0.2, 4
    UA_cond=10,  # 10, 1.0025, 25.2578
    UA_evap=15,  # 15, 1.5079, 11.3518
    UA_abs=10,      # 10, 1.5, 8.1355
    UA_des=25,   # 25, 2.4895, 10.4058s
    UA_shex=70.8/6.43,  # 11.0109, 0.7584, 1.6796
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="m11",
    cycle_scale_spec_mode="m6",
    desorber_evaporator_routing_mode="parallel",
    m11_spec=4.0,
    m6_spec=1.0,
)

X0_C =  np.array(
            [
                55,   # T8  [°C] 55, 29.98, 30
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
# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def k(T_C: float) -> float:
    """Celsius → Kelvin, nur zur Lesbarkeit."""
    return T_C + 273.15


def c(T_K: float) -> float:
    """Kelvin → Celsius."""
    return T_K - 273.15


def state_T(result: AWTResult, sid: str) -> float:
    """Temperatur eines Zustands in °C."""
    return c(result.states[sid]["T_K"])


def state_m(result: AWTResult, sid: str) -> float:
    return result.states[sid]["m_kg_s"]


def state_h(result: AWTResult, sid: str) -> float:
    return result.states[sid]["h_kJ_kg"]


# ---------------------------------------------------------------------------
# Q-T-Kurven für jeden Apparat
# ---------------------------------------------------------------------------

def qt_shex(result: AWTResult) -> dict:
    """SHEX: heiß 3→2, kalt 5→4 (Gegenstrom, einphasig beidseitig)."""
    Q_total = result.heat_flows_kW["Q_shex"]

    T3 = state_T(result, "3")
    T2 = state_T(result, "2")
    T4 = state_T(result, "4")
    T5 = state_T(result, "5")

    # Heiße Seite: Eintritt bei Q=0 (Zustand 3), Austritt bei Q=Q_total (Zustand 2)
    hot_Q = [0.0, Q_total]
    hot_T = [T3, T2]

    # Kalte Seite im Gegenstrom: Eintritt bei Q=Q_total (Zustand 5), Austritt bei Q=0 (Zustand 4)
    cold_Q = [0.0, Q_total]
    cold_T = [T4, T5]

    return dict(
        title="Lösungswärmeübertrager (SHEX)",
        Q_total=Q_total,
        hot_Q=hot_Q, hot_T=hot_T, hot_label="Lösung reich (3→2)",
        cold_Q=cold_Q, cold_T=cold_T, cold_label="Lösung arm (5→4)",
        pinch_hot=min(T3 - T4, T2 - T5),
    )


def qt_desorber(result: AWTResult) -> dict:
    """Desorber: heiß 13→14 (extern, einphasig), kalt 1→6 (Lösung, einphasig)."""
    Q_total = result.heat_flows_kW["Q_des"]

    T13 = c(result.diagnostics["T13_K"])
    T14 = c(result.diagnostics["T14_K"])
    T1  = state_T(result, "1")
    T6  = state_T(result, "6")

    hot_Q = [0.0, Q_total]
    hot_T = [T13, T14]

    cold_Q = [0.0, Q_total]
    cold_T = [T6, T1]   # Gegenstrom: Austritt (6) bei Q=0, Eintritt (1) bei Q=Q_total

    return dict(
        title="Desorber",
        Q_total=Q_total,
        hot_Q=hot_Q, hot_T=hot_T, hot_label="Heizwasser (13→14)",
        cold_Q=cold_Q, cold_T=cold_T, cold_label="LiBr-Lösung (1→6)",
        pinch_hot=min(T13 - T6, T14 - T1),
    )


def qt_condenser(result: AWTResult) -> dict:
    """Kondensator: heiß 7→8 (Dampf→Kondensat, Phasenwechsel), kalt 17→18."""
    Q_total = result.heat_flows_kW["Q_cond"]

    T7  = state_T(result, "7")
    T8  = state_T(result, "8")   # = T_sat bei p_low → i.d.R. = T7 wenn kein Superheat
    T17 = c(result.states["17"]["T_K"])
    T18 = c(result.diagnostics["T18_K"])

    # Kondensation: Dampf kühlt von T7 auf T8 (Sättigungstemperatur),
    # dann isotherme Kondensation. Wir bilden drei Punkte:
    #   Q=0        : T_hot = T7  (Dampfeintritt)
    #   Q=Q_superheat : T_hot = T8  (Beginn Kondensation)
    #   Q=Q_total  : T_hot = T8  (Ende Kondensation = Kondensat)
    # Bei desorber_vapor_superheat_K=0 fallen die ersten zwei Punkte zusammen.
    m7   = state_m(result, "7")
    h7   = state_h(result, "7")
    h8   = state_h(result, "8")

    # Anteil Überhitzung an Q_total
    try:
        from Models.AHT_UA_LMTD import water_h_kjkg_PQ
        p_low = result.diagnostics["p_low_Pa"]
        h_sat_vapor = water_h_kjkg_PQ(p_low, 1.0)
        Q_superheat = m7 * (h7 - h_sat_vapor)
    except Exception:
        Q_superheat = 0.0
    Q_superheat = max(Q_superheat, 0.0)

    if Q_superheat > 1e-6:
        hot_Q = [0.0,         Q_superheat, Q_total]
        hot_T = [T7,          T8,          T8     ]
    else:
        hot_Q = [0.0, Q_total]
        hot_T = [T8,  T8    ]

    cold_Q = [0.0, Q_total]
    cold_T = [T18, T17]   # Gegenstrom

    return dict(
        title="Kondensator",
        Q_total=Q_total,
        hot_Q=hot_Q, hot_T=hot_T, hot_label="Kältemittel (7→8)",
        cold_Q=cold_Q, cold_T=cold_T, cold_label="Kühlwasser (17→18)",
        pinch_hot=min(T8 - T17, T8 - T18),
    )


def qt_evaporator(result: AWTResult) -> dict:
    """Verdampfer: heiß 15→16 (extern), kalt 9→10 (Verdampfung isotherm)."""
    Q_total = result.heat_flows_kW["Q_evap"]

    T15 = c(result.diagnostics["T15_K"])
    T16 = c(result.diagnostics["T16_K"])
    T9  = state_T(result, "9")
    T10 = state_T(result, "10")   # Sättigungsdampf, i.d.R. ≈ T9

    hot_Q = [0.0, Q_total]
    hot_T = [T15, T16]

    # Verdampfung weitgehend isotherm (subgekühlte Flüssigkeit → Sattdampf)
    cold_Q = [0.0, Q_total]
    cold_T = [T10, T9]   # Gegenstrom: T10 bei Q=0, T9 bei Q=Q_total

    return dict(
        title="Verdampfer",
        Q_total=Q_total,
        hot_Q=hot_Q, hot_T=hot_T, hot_label="Heizwasser (15→16)",
        cold_Q=cold_Q, cold_T=cold_T, cold_label="Kältemittel (9→10)",
        pinch_hot=min(T15 - T10, T16 - T9),
    )


def qt_absorber(result: AWTResult) -> dict:
    """Absorber: heiß 20→3 (Lösung, einphasig), kalt 11→12."""
    Q_total = result.heat_flows_kW["Q_abs"]

    T20 = state_T(result, "20")
    T3  = state_T(result, "3")
    T11 = c(result.states["11"]["T_K"])
    T12 = c(result.diagnostics["T12_K"])

    hot_Q = [0.0, Q_total]
    hot_T = [T20, T3]

    cold_Q = [0.0, Q_total]
    cold_T = [T12, T11]   # Gegenstrom

    return dict(
        title="Absorber",
        Q_total=Q_total,
        hot_Q=hot_Q, hot_T=hot_T, hot_label="LiBr-Lösung (20→3)",
        cold_Q=cold_Q, cold_T=cold_T, cold_label="Nutzwasser (11→12)",
        pinch_hot=min(T20 - T12, T3 - T11),
    )


# ---------------------------------------------------------------------------
# Plot-Funktion
# ---------------------------------------------------------------------------

def plot_qt(ax: plt.Axes, data: dict, show_pinch: bool = True) -> None:
    """Zeichnet ein Q-T-Diagramm in die übergebene Axes."""
    color_hot  = "#d62728"
    color_cold = "#1f77b4"

    ax.plot(data["hot_Q"],  data["hot_T"],  color=color_hot,  lw=2.5,
            marker="o", ms=6, label=data["hot_label"])
    ax.plot(data["cold_Q"], data["cold_T"], color=color_cold, lw=2.5,
            marker="s", ms=6, label=data["cold_label"])

    # Pinch-Temperatur annotieren
    if show_pinch:
        pinch = data["pinch_hot"]
        ax.text(
            0.97, 0.05,
            f"$\\Delta T_{{\\mathrm{{min}}}}$ = {pinch:.1f} K",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        )

    ax.set_title(data["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Wärmeleistung $\\dot{Q}$ [kW]", fontsize=10)
    ax.set_ylabel("Temperatur $T$ [°C]", fontsize=10)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(left=0.0)


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulation lösen
    x0_K = primary_temperatures_C_to_K(X0_C)
    result = solve_awt(INPUTS, x0=x0_K)

    if not result.solve_info.final_point_evaluable:
        print("Simulation nicht konvergiert – bitte Startwerte prüfen.")
        print(f"  Status   : {result.solve_info.status}")
        print(f"  Nachricht: {result.solve_info.message}")
        sys.exit(1)

    print_summary(result)

    # Q-T-Daten für alle fünf Apparate
    apparatus = [
        qt_shex(result),
        qt_desorber(result),
        qt_condenser(result),
        qt_evaporator(result),
        qt_absorber(result),
    ]

    # 5 Plots in einem 2×3-Raster (letztes Feld leer)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for pos, data in zip(axes_positions, apparatus):
        ax = fig.add_subplot(gs[pos])
        plot_qt(ax, data)

    # Letztes Feld: Betriebspunkt-Übersicht als Text
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")
    cop  = result.kpis.get("COP",  float("nan"))
    ecop = result.exergy_kW.get("Exergy_efficiency", float("nan"))
    q_abs = result.heat_flows_kW["Q_abs"]
    q_des = result.heat_flows_kW["Q_des"]
    q_evap= result.heat_flows_kW["Q_evap"]
    q_cond= result.heat_flows_kW["Q_cond"]
    q_shex= result.heat_flows_kW["Q_shex"]
    T12   = c(result.diagnostics["T12_K"])

    info_text = (
        "Betriebspunkt\n"
        "─────────────────────\n"
        f"$T_{{11}}$ = {INPUTS.T_11_C:.1f} °C\n"
        f"$T_{{13}}$ = $T_{{15}}$ = {INPUTS.T_13_C:.1f} °C\n"
        f"$T_{{17}}$ = {INPUTS.T_17_C:.1f} °C\n"
        f"$T_{{12}}$ = {T12:.2f} °C\n"
        "─────────────────────\n"
        f"COP  = {cop:.4f}\n"
        f"ECOP = {ecop:.4f}\n"
        "─────────────────────\n"
        f"$Q_{{\\mathrm{{Abs}}}}$  = {q_abs:.2f} kW\n"
        f"$Q_{{\\mathrm{{Des}}}}$  = {q_des:.2f} kW\n"
        f"$Q_{{\\mathrm{{Eva}}}}$  = {q_evap:.2f} kW\n"
        f"$Q_{{\\mathrm{{Kon}}}}$  = {q_cond:.2f} kW\n"
        f"$Q_{{\\mathrm{{SHEX}}}}$ = {q_shex:.2f} kW\n"
    )
    ax_info.text(
        0.05, 0.95, info_text,
        transform=ax_info.transAxes,
        va="top", ha="left", fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="gray"),
    )

    fig.suptitle(
        "AWT – Q-T-Diagramme (Pinch-Analyse)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.savefig("Auswertung_KO/Results_Plots/AWT_UA_LMTD_Pinch_QT.png", dpi=150, bbox_inches="tight")
    plt.show()