"""Q-T diagrams (pinch analysis) for the AHT simulation.

Produces a 2x3 grid of temperature-vs-cumulative-heat-flow curves for the
five heat exchangers (SHEX, desorber, condenser, evaporator, absorber),
plus an operating-point info box.

All quantities (states, heat flows, pinch temperatures, diagnostics) are
read exclusively from the `AHTResult` object returned by `solve_aht()`.
The model itself is not modified.

Condenser special case
------------------------
Because of the LiBr solution's boiling-point elevation, the refrigerant
vapor (state 7) leaves the desorber superheated relative to the pure-water
saturation temperature at p_low (state 8). In the condenser it is
therefore first desuperheated (small Q share) and then condensed
isothermally at T8. This kink is computed explicitly via the saturated
vapor enthalpy at p_low; the other four heat exchangers are drawn -- as in
the original template -- as simple straight lines between inlet and
outlet state.

Usage from the main script
-----------------------------
    from Postprocessing.AHT_QT_Plot import plot_qt_diagrams

    if ENABLE_QT_PLOT:
        plot_qt_diagrams(result, save_path="AHT_QT_Diagramme.png")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from Models.AHT_Pinch_Point import AHTResult, kelvin_to_celsius, water_h_kjkg_PQ


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _condenser_hot_side_points(result: AHTResult) -> Tuple[List[float], List[float]]:
    """Returns (Q_cumulative, T) for the hot (refrigerant) side of the
    condenser: desuperheating kink (superheated vapor -> saturation
    temperature at p_low) followed by isothermal condensation."""
    state7 = result.states["7"]
    state8 = result.states["8"]
    p_low = result.diagnostics["p_low_Pa"]

    m7 = state7["m_kg_s"]
    T7_C = kelvin_to_celsius(state7["T_K"])
    T8_C = kelvin_to_celsius(state8["T_K"])

    h7 = state7["h_kJ_kg"]
    h_g_low = water_h_kjkg_PQ(p_low, 1.0)  # saturated vapor at p_low
    Q_total = result.heat_flows_kW["Q_cond"]

    Q_desuperheat = m7 * (h7 - h_g_low)
    # Clamp to [0, Q_total] for numerical robustness (e.g. superheat_K = 0)
    Q_desuperheat = max(0.0, min(Q_desuperheat, Q_total))

    if Q_desuperheat < 1.0e-6:
        # Practically no desuperheating share -- simple line like the
        # other heat exchangers
        return [0.0, Q_total], [T7_C, T8_C]

    return [0.0, Q_desuperheat, Q_total], [T7_C, T8_C, T8_C]


def _evaporator_cold_side_points(result: AHTResult) -> Tuple[List[float], List[float]]:
    """Returns (Q_cumulative, T) for the cold refrigerant side of the
    evaporator: subcooled liquid (9) -> saturated liquid -> vaporization
    -> saturated vapor (10)."""
    state9 = result.states["9"]
    state10 = result.states["10"]

    p_high = result.diagnostics["p_high_Pa"]

    m9 = state9["m_kg_s"]

    T9_C = kelvin_to_celsius(state9["T_K"])
    T10_C = kelvin_to_celsius(state10["T_K"])
    T_sat_C = kelvin_to_celsius(state10["T_K"])

    h9 = state9["h_kJ_kg"]

    h_f_low = water_h_kjkg_PQ(p_high, 0.0)  # saturated liquid at p_high

    Q_total = result.heat_flows_kW["Q_evap"]

    Q_subcool = m9 * (h_f_low - h9)
    Q_subcool = max(0.0, min(Q_subcool, Q_total))

    if Q_subcool < 1e-6:
        return [0.0, Q_total], [T10_C, T9_C]

    return (
        [0.0, Q_total - Q_subcool, Q_total],
        [T10_C, T_sat_C, T9_C],
    )


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
    ax.set_xlabel(r"Heat flow $\dot Q$ [kW]")
    ax.set_ylabel(r"Temperature $T$ [°C]")
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

    if inputs.desorber_evaporator_routing_mode == "parallel":
        T13_T15_line = f"T13 = T15 = {inputs.T_13_C:.1f} °C"
    else:
        T13_T15_line = "T13/T15: serially coupled"

    lines = [
        "Operating point",
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
# Public API
# ---------------------------------------------------------------------------

def plot_qt_diagrams(
    result: AHTResult,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Produces the Q-T diagrams (pinch analysis) for a solved AHT
    operating point.

    Parameters
    ----------
    result:
        Result object from `solve_aht()`. Must be physically evaluable
        (`result.solve_info.final_point_evaluable`).
    show:
        Opens an interactive window (`plt.show()`) if True.
    save_path:
        Optional file path to save the figure under.
    dpi:
        Resolution when saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not result.solve_info.final_point_evaluable:
        raise ValueError(
            "Cannot produce Q-T diagrams: the end point is not physically "
            "evaluable (result.solve_info.final_point_evaluable=False)."
        )

    s = result.states
    hf = result.heat_flows_kW
    pinch = result.pinch_temperatures_K

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Solution heat exchanger (SHEX)
    _hx_panel(
        axes[0, 0],
        title="Solution heat exchanger (SHEX)",
        hot_label="Rich solution (3→2)",
        cold_label="Weak solution (5→4)",
        hot_Q=[0.0, hf["Q_shex"]],
        hot_T=[kelvin_to_celsius(s["3"]["T_K"]), kelvin_to_celsius(s["2"]["T_K"])],
        cold_Q=[0.0, hf["Q_shex"]],
        cold_T=[kelvin_to_celsius(s["4"]["T_K"]), kelvin_to_celsius(s["5"]["T_K"])],
        dT_min_K=pinch["pinch_shex_K"],
        hot_states=["3", "2"],
        cold_states=["4", "5"],
    )

    # Desorber
    _hx_panel(
        axes[0, 1],
        title="Desorber",
        hot_label="Heating water (13→14)",
        cold_label="LiBr solution (1→6)",
        hot_Q=[0.0, hf["Q_des"]],
        hot_T=[kelvin_to_celsius(s["13"]["T_K"]), kelvin_to_celsius(s["14"]["T_K"])],
        cold_Q=[0.0, hf["Q_des"]],
        cold_T=[kelvin_to_celsius(s["6"]["T_K"]), kelvin_to_celsius(s["1"]["T_K"])],
        dT_min_K=pinch["pinch_des_K"],
        hot_states=["13", "14"],
        cold_states=["6", "1"],
    )

    # Condenser (with desuperheating kink)
    cond_hot_Q, cond_hot_T = _condenser_hot_side_points(result)
    _hx_panel(
        axes[0, 2],
        title="Condenser",
        hot_label="Refrigerant (7→8)",
        cold_label="Cooling water (17→18)",
        hot_Q=cond_hot_Q,
        hot_T=cond_hot_T,
        cold_Q=[0.0, hf["Q_cond"]],
        cold_T=[kelvin_to_celsius(s["18"]["T_K"]), kelvin_to_celsius(s["17"]["T_K"])],
        dT_min_K=pinch["pinch_cond_K"],
        hot_states=["7", "Sat.", "8"],
        cold_states=["18", "17"],
    )

    # Evaporator
    eva_cold_Q, eva_cold_T = _evaporator_cold_side_points(result)
    _hx_panel(
        axes[1, 0],
        title="Evaporator",
        hot_label="Heating water (15→16)",
        cold_label="Refrigerant (9→10)",
        hot_Q=[0.0, hf["Q_evap"]],
        hot_T=[kelvin_to_celsius(s["15"]["T_K"]), kelvin_to_celsius(s["16"]["T_K"])],
        cold_Q=eva_cold_Q,
        cold_T=eva_cold_T,
        dT_min_K=pinch["pinch_evap_K"],
        hot_states=["15", "16"],
        cold_states=["10", "Sat.", "9"],
    )

    # Absorber
    _hx_panel(
        axes[1, 1],
        title="Absorber",
        hot_label="LiBr solution (20→3)",
        cold_label="Useful-heat water (11→12)",
        hot_Q=[0.0, hf["Q_abs"]],
        hot_T=[kelvin_to_celsius(s["20"]["T_K"]), kelvin_to_celsius(s["3"]["T_K"])],
        cold_Q=[0.0, hf["Q_abs"]],
        cold_T=[kelvin_to_celsius(s["12"]["T_K"]), kelvin_to_celsius(s["11"]["T_K"])],
        dT_min_K=pinch["pinch_abs_K"],
        hot_states=["20", "3"],
        cold_states=["12", "11"],
    )

    # Operating-point info box (no ECOP)
    _info_panel(axes[1, 2], result)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


__all__ = ["plot_qt_diagrams"]
