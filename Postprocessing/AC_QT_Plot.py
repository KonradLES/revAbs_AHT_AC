"""Q-T-Diagramme (Pinch-Analyse) für die AC-Simulation.

Analogon zu Postprocessing/AHT_QT_Plot.py, aber für die
Absorptionskältemaschine. Erzeugt eine 2x3-Rasterdarstellung mit den
Temperaturverläufen der fünf Wärmeübertrager (SHEX, Desorber, Kondensator,
Verdampfer, Absorber) über der kumulierten Wärmeleistung Q, sowie einer
Betriebspunkt-Infobox.

Alle benötigten Größen (Zustände, Wärmeströme, Pinch-Temperaturen,
Diagnostik) werden ausschließlich aus dem `AHTResult`-Objekt gelesen, das
Models.AC_Pinch_Point.solve_aht() liefert. Das Modell selbst wird nicht
verändert.

Zuordnung Q-Achsen-Position <-> Zustand (Gegenstrom-Wärmeübertrager)
----------------------------------------------------------------------
Bei einem Gegenstromwärmeübertrager liegt der Heißstrom-Eintritt am selben
Ende wie der Kaltstrom-Austritt (und umgekehrt) -- siehe die
dT_..._hot_end/dT_..._cold_end-Definitionen in Models/AC_Pinch_Point.py, die
hier 1:1 für die Q-Achsen-Zuordnung übernommen werden:
    SHEX      : Q=0 (4, 3)         Q=Q_shex (5, 2)
    Desorber  : Q=0 (T11, 4)       Q=Q_des  (T12, 7)
    Kondensator: Q=0 (7*, 16)       Q=Q_cond (8, T15)     (*siehe Enthitzung)
    Verdampfer: Q=0 (T17, 10)      Q=Q_evap (T18, 9)
    Absorber  : Q=0 (6, 14)        Q=Q_abs  (1, T13)

Sonderfall Kondensator
-----------------------
Wie beim AHT verlässt der Kältemitteldampf (Zustand 7) den Desorber
überhitzt gegenüber der reinen Wassersättigungstemperatur bei p_high
(Zustand 8) -- Siedepunktserhöhung durch die LiBr-Konzentration der
Restlösung, aus der er entstanden ist. Im Kondensator wird er daher
zunächst enthitzt (kleiner Q-Anteil) und anschließend isotherm bei T8
kondensiert. Der Verdampfer braucht dagegen KEINEN analogen Knick: Zustand 9
(nach der Drossel) liegt bei der AC numerisch praktisch exakt auf der
Sättigungstemperatur von Zustand 10 (beide bei p_low) -- die Drossel liefert
hier direkt ein Zweiphasengemisch ohne nennenswerte Unterkühlung.

Nutzung im Hauptskript
-----------------------
    from Postprocessing.AC_QT_Plot import plot_qt_diagrams

    if ENABLE_QT_PLOT:
        plot_qt_diagrams(result, save_path="AC_QT_Diagramme.png")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from Models.AC_Pinch_Point import AHTResult, kelvin_to_celsius, water_h_kjkg_PQ


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _condenser_hot_side_points(result: AHTResult) -> Tuple[List[float], List[float]]:
    """Liefert (Q_kumuliert, T) für die heiße (Kältemittel-)Seite des Kondensators.

    Bildet den Knick durch Enthitzung (überhitzter, durch Siedepunktserhöhung
    "heißer" Kältemitteldampf -> Sättigungstemperatur bei p_high) gefolgt von
    isothermer Kondensation ab. Q=0 entspricht dem Dampfeintritt (Zustand 7,
    heißestes Ende), analog zur Desorber-/Absorber-Konvention hier.
    """
    state7 = result.states["7"]
    state8 = result.states["8"]
    p_high = result.diagnostics["p_high_Pa"]

    m7 = state7["m_kg_s"]
    T7_C = kelvin_to_celsius(state7["T_K"])
    T8_C = kelvin_to_celsius(state8["T_K"])

    h7 = state7["h_kJ_kg"]
    h_g_high = water_h_kjkg_PQ(p_high, 1.0)  # gesättigter Dampf bei p_high
    Q_total = result.heat_flows_kW["Q_cond"]

    Q_desuperheat = m7 * (h7 - h_g_high)
    # Numerisch robust auf [0, Q_total] begrenzen (z. B. superheat_K = 0)
    Q_desuperheat = max(0.0, min(Q_desuperheat, Q_total))

    if Q_desuperheat < 1.0e-6:
        # Praktisch kein Enthitzungsanteil -> einfache Gerade wie bei den
        # übrigen Wärmeübertragern
        return [0.0, Q_total], [T7_C, T8_C]

    return [0.0, Q_desuperheat, Q_total], [T7_C, T8_C, T8_C]


def _annotate_state_points(ax, Q, T, labels, color):
    for q, t, lbl in zip(Q, T, labels):
        ax.annotate(
            lbl,
            (q, t),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec=color,
                alpha=0.8,
            ),
        )


def _hx_panel(
    ax,
    *,
    title,
    hot_label,
    cold_label,
    hot_Q,
    hot_T,
    cold_Q,
    cold_T,
    dT_min_K,
    hot_states=None,
    cold_states=None,
):
    ax.plot(hot_Q, hot_T, "o-", color="tab:red", linewidth=2, label=hot_label)
    ax.plot(cold_Q, cold_T, "o-", color="tab:blue", linewidth=2, label=cold_label)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Wärmeleistung $\dot Q$ [kW]")
    ax.set_ylabel(r"Temperatur $T$ [°C]")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.07,
        0.05,
        rf"$\Delta T_{{\min}}$ = {dT_min_K:.1f} K",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
    )
    if hot_states is not None:
        _annotate_state_points(ax, hot_Q, hot_T, hot_states, "tab:red")

    if cold_states is not None:
        _annotate_state_points(ax, cold_Q, cold_T, cold_states, "tab:blue")


def _info_panel(ax, result: AHTResult) -> None:
    ax.axis("off")
    inputs = result.inputs
    kpis = result.kpis
    Q = result.heat_flows_kW

    if inputs.absorber_condenser_routing_mode == "parallel":
        T13_T15_line = f"T13 = T15 = {inputs.T_13_C:.1f} °C"
    else:
        T13_T15_line = "T13/T15: seriell gekoppelt"

    lines = [
        "Betriebspunkt",
        "",
        f"T11 = {inputs.T_11_C:.1f} °C",
        T13_T15_line,
        f"T17 = {inputs.T_17_C:.1f} °C",
        f"T12 = {kelvin_to_celsius(result.diagnostics['T12_K']):.2f} °C",
        "",
        f"COP  = {kpis['COP']:.4f}",
        "",
        f"Q_Abs  = {Q['Q_abs']:.2f} kW",
        f"Q_Des  = {Q['Q_des']:.2f} kW",
        f"Q_Eva  = {Q['Q_evap']:.2f} kW",
        f"Q_Kon  = {Q['Q_cond']:.2f} kW",
        f"Q_SHEX = {Q['Q_shex']:.2f} kW",
    ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
    )


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def plot_qt_diagrams(
    result: AHTResult,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Erzeugt die Q-T-Diagramme (Pinch-Analyse) für einen gelösten
    AC-Betriebspunkt.

    Parameters
    ----------
    result:
        Ergebnisobjekt von Models.AC_Pinch_Point.solve_aht(). Muss
        physikalisch auswertbar sein (`result.solve_info.final_point_evaluable`).
    show:
        Öffnet ein interaktives Fenster (`plt.show()`), falls True.
    save_path:
        Optionaler Dateipfad, unter dem die Grafik gespeichert wird.
    dpi:
        Auflösung beim Speichern.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not result.solve_info.final_point_evaluable:
        raise ValueError(
            "Q-T-Diagramme können nicht erzeugt werden: Endpunkt ist nicht "
            "physikalisch auswertbar (result.solve_info.final_point_evaluable=False)."
        )

    s = result.states
    hf = result.heat_flows_kW
    pinch = result.pinch_temperatures_K

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("AC – Q-T-Diagramme (Pinch-Analyse)", fontsize=14, fontweight="bold")

    # Lösungswärmeübertrager (SHEX): heiß = starke Lösung (4->5), kalt = schwache Lösung (2->3)
    _hx_panel(
        axes[0, 0],
        title="Lösungswärmeübertrager (SHEX)",
        hot_label="Lösung stark (4→5)",
        cold_label="Lösung schwach (2→3)",
        hot_Q=[0.0, hf["Q_shex"]],
        hot_T=[kelvin_to_celsius(s["4"]["T_K"]), kelvin_to_celsius(s["5"]["T_K"])],
        cold_Q=[0.0, hf["Q_shex"]],
        cold_T=[kelvin_to_celsius(s["3"]["T_K"]), kelvin_to_celsius(s["2"]["T_K"])],
        dT_min_K=pinch["pinch_shex_K"],
        hot_states=["4", "5"],
        cold_states=["3", "2"],
    )

    # Desorber: heiß = externes Antriebswasser (11->12), kalt = Lösung/Dampf (7->4)
    _hx_panel(
        axes[0, 1],
        title="Desorber",
        hot_label="Antriebswasser (11→12)",
        cold_label="Lösung/Dampf (7→4)",
        hot_Q=[0.0, hf["Q_des"]],
        hot_T=[kelvin_to_celsius(s["11"]["T_K"]), kelvin_to_celsius(s["12"]["T_K"])],
        cold_Q=[0.0, hf["Q_des"]],
        cold_T=[kelvin_to_celsius(s["4"]["T_K"]), kelvin_to_celsius(s["7"]["T_K"])],
        dT_min_K=pinch["pinch_des_K"],
        hot_states=["11", "12"],
        cold_states=["4", "7"],
    )

    # Kondensator (mit Enthitzungsknick): heiß = Kältemittel (7->8), kalt = Rückkühlwasser (15->16)
    cond_hot_Q, cond_hot_T = _condenser_hot_side_points(result)
    _hx_panel(
        axes[0, 2],
        title="Kondensator",
        hot_label="Kältemittel (7→8)",
        cold_label="Rückkühlwasser (15→16)",
        hot_Q=cond_hot_Q,
        hot_T=cond_hot_T,
        cold_Q=[0.0, hf["Q_cond"]],
        cold_T=[kelvin_to_celsius(s["16"]["T_K"]), kelvin_to_celsius(s["15"]["T_K"])],
        dT_min_K=pinch["pinch_cond_K"],
        hot_states=["7", "Sat.", "8"],
        cold_states=["16", "15"],
    )

    # Verdampfer: heiß = externe Nutzkälte (17->18), kalt = Kältemittel (9->10), isotherm
    _hx_panel(
        axes[1, 0],
        title="Verdampfer",
        hot_label="Nutzkälte-Rücklauf (17→18)",
        cold_label="Kältemittel (9→10)",
        hot_Q=[0.0, hf["Q_evap"]],
        hot_T=[kelvin_to_celsius(s["17"]["T_K"]), kelvin_to_celsius(s["18"]["T_K"])],
        cold_Q=[0.0, hf["Q_evap"]],
        cold_T=[kelvin_to_celsius(s["10"]["T_K"]), kelvin_to_celsius(s["9"]["T_K"])],
        dT_min_K=pinch["pinch_evap_K"],
        hot_states=["17", "18"],
        cold_states=["10", "9"],
    )

    # Absorber: heiß = Lösung (6->1), kalt = Rückkühlwasser (13->14)
    _hx_panel(
        axes[1, 1],
        title="Absorber",
        hot_label="Lösung (6→1)",
        cold_label="Rückkühlwasser (13→14)",
        hot_Q=[0.0, hf["Q_abs"]],
        hot_T=[kelvin_to_celsius(s["6"]["T_K"]), kelvin_to_celsius(s["1"]["T_K"])],
        cold_Q=[0.0, hf["Q_abs"]],
        cold_T=[kelvin_to_celsius(s["14"]["T_K"]), kelvin_to_celsius(s["13"]["T_K"])],
        dT_min_K=pinch["pinch_abs_K"],
        hot_states=["6", "1"],
        cold_states=["14", "13"],
    )

    # Betriebspunkt-Infobox
    _info_panel(axes[1, 2], result)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


__all__ = ["plot_qt_diagrams"]
