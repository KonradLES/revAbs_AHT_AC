"""Feasibility / COP map for the water/water compression heat pump.

Companion to revAbs_AHT_AC/Design_Point/AHT_feasibility_sweep.py, but for
the vapor-compression heat pump in Compression_Models/HeatPump_Pinch_Point.py.

A compression heat pump does not have the same narrow, internally-limited
feasibility window as the AHT (there is no absorption/desorption
concentration hierarchy to violate) -- for any source/sink temperature
pair you can, in principle, just pick a compressor pressure ratio that
gets you there. The real, practically-relevant limit is the refrigerant's
critical temperature: as the condensing temperature approaches T_crit,
COP collapses and eventually no valid subcritical cycle exists at all.

This script therefore sweeps a grid of
    x-axis: T_source_in    (heat source inlet, e.g. waste heat / ambient water)
    y-axis: T_lift = T_sink_out - T_source_in  (same definition as the AHT's
            GTL, so this plot is directly comparable to
            AHT_feasibility_sweep's GTL-vs-waste-heat plot)
and for each point:
    - solves the design point via HeatPump_Pinch_Point.solve_heat_pump
    - marks the point infeasible if the network fails to converge, or if
      the resulting condensing temperature comes within
      `condensing_margin_K` of the refrigerant's critical temperature
    - records COP and the external temperature lift (T_sink_out -
      T_source_in), the same lift definition as the AHT's GTL

Output: one figure with COP as a filled contour, lift as overlaid contour
lines, and infeasible points masked out.

Standalone usage
-----------------
    python Compression_Models/HeatPump_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from CoolProp.CoolProp import PropsSI

from Compression_Models.HeatPump_Pinch_Point import (
    HeatPumpInputs, HeatPumpModel, HeatPumpEvaluationError,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    # R134a's critical temperature (~101 degC) is too low to reach typical
    # district-heating supply levels; R1233zd(E) (T_crit ~166 degC) is a
    # common low-GWP choice for high-temperature industrial heat pumps in
    # this range (e.g. waste-heat-to-district-heating applications)
    refrigerant: str = "R1233zd(E)"

    # Pinch values (design assumption, not an optimization target)
    dT_min_evap: float = 5.0
    dT_min_cond: float = 5.0

    # Cooldown of the source stream / heat-up spread of the sink stream
    dT_source_spread: float = 5.0
    dT_sink_spread: float = 20.0

    # Isentropic efficiency. If use_pr_dependent_eta_s is True, eta_s below
    # is only the fallback/reference value and the actual value used per
    # grid point comes from eta_s_from_pr() instead.
    eta_s: float = 0.75
    superheat_K: float = 3.0
    subcooling_K: float = 2.0

    # Compressor isentropic efficiency as a function of pressure ratio.
    # Real compressor efficiency maps are peaked, not monotonic: they fall
    # off both towards very low PR (clearance-volume re-expansion losses
    # dominate a small compression job) and towards high PR (leakage,
    # valve losses, volumetric efficiency loss). This is a generic,
    # illustrative parabola calibrated to a plausible peak (eta_max at
    # PR_opt) -- NOT digitized from a specific manufacturer curve. For a
    # rigorous version, see the compressor efficiency/working-domain
    # models in Jensen et al. (2015), "Technical and economic working
    # domains of industrial heat pumps: Part 1 - single stage vapor
    # compression heat pumps", Int. J. Refrigeration, or the semi-empirical
    # compressor model in Winandy et al. (2002).
    use_pr_dependent_eta_s: bool = True
    eta_s_max: float = 0.78
    PR_opt: float = 4.0
    eta_s_curvature: float = 0.015
    eta_s_min: float = 0.35

    # Mechanical + motor efficiency: converts the compressor's isentropic
    # shaft-power COP into an electrical-input COP. Applied as a constant
    # multiplicative factor on the design-point COP (does not affect the
    # thermodynamic cycle state, i.e. discharge temperature etc.).
    eta_mech_motor: float = 0.94

    Q_cond_kW: float = 500.0

    # Safety margin to the refrigerant's critical temperature: a point is
    # marked infeasible if T_cond >= T_crit - condensing_margin_K
    condensing_margin_K: float = 10.0

    # Practical compressor limits (beyond the refrigerant's critical point):
    # - discharge temperature: oil/material limit, often more restrictive
    #   than the critical-temperature margin above
    # - pressure ratio: single-stage reciprocating/screw machines are
    #   practically limited to about this range; higher lifts would
    #   require two-stage compression with intercooling, which this
    #   single-stage model does not represent
    T_discharge_max_C: float = 150.0
    PR_max: float = 6.0

    # Source water must not be cooled anywhere near freezing
    min_source_out_C: float = 2.0


# T_source_in: data center waste heat available at the evaporator inlet
# T_lift_K: T_sink_out - T_source_in, same axis definition as the AHT's GTL
T_SOURCE_IN_C = np.arange(40.0, 81.0, 5.0)
T_LIFT_K = np.arange(5.0, 71.0, 5.0)

plot_name = "Compression_Models/Plots/heat_pump_feasibility_sweep_075"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _estimate_pressure_ratio(
    T_source_in_C: float, T_sink_out_C: float, config: SweepConfig
) -> float:
    """Analytic PR estimate from the pinch-fixed saturation temperatures,
    without solving the network. Valid because the refrigerant pressure
    levels are set entirely by the ttd_l equations (temperature-driven);
    eta_s never enters that part of the model, so PR can be computed
    upfront and used to pick the eta_s for the actual solve."""
    T_source_out_C = T_source_in_C - config.dT_source_spread
    T_sink_in_C = T_sink_out_C - config.dT_sink_spread

    T_evap_sat_K = (T_source_out_C - config.dT_min_evap) + 273.15
    T_cond_sat_K = (T_sink_in_C + config.dT_min_cond + config.subcooling_K) + 273.15

    p_evap = PropsSI("P", "T", T_evap_sat_K, "Q", 1, config.refrigerant)
    p_cond = PropsSI("P", "T", T_cond_sat_K, "Q", 0, config.refrigerant)
    return p_cond / p_evap


def eta_s_from_pr(PR: float, config: SweepConfig) -> float:
    """Parabolic isentropic efficiency vs. pressure ratio, see SweepConfig
    docstring comment for the caveat on where this comes from."""
    eta = config.eta_s_max - config.eta_s_curvature * (PR - config.PR_opt) ** 2
    return float(np.clip(eta, config.eta_s_min, config.eta_s_max))


def _build_inputs(
    T_source_in_C: float, T_sink_out_C: float, config: SweepConfig, eta_s: float
) -> HeatPumpInputs:
    T_sink_in_C = T_sink_out_C - config.dT_sink_spread
    return HeatPumpInputs(
        T_source_in_C=T_source_in_C,
        T_sink_in_C=T_sink_in_C,
        dT_min_evap=config.dT_min_evap,
        dT_min_cond=config.dT_min_cond,
        refrigerant=config.refrigerant,
        eta_s=eta_s,
        superheat_K=config.superheat_K,
        subcooling_K=config.subcooling_K,
        scale_spec_mode="Q_cond",
        Q_cond_kW=config.Q_cond_kW,
        source_spec_mode="T",
        T_source_out_C=T_source_in_C - config.dT_source_spread,
        sink_spec_mode="T",
        T_sink_out_C=T_sink_out_C,
    )


def _is_physically_valid(result) -> bool:
    """Reject spurious Newton roots: the solver can converge to a
    mathematically consistent but unphysical state (e.g. compressor
    pressure ratio < 1, or reversed heat flow) especially very close to
    zero external lift, where the design point becomes ill-conditioned."""
    if result.kpis["pressure_ratio"] <= 1.0 + 1.0e-6:
        return False
    if result.kpis["T_lift_external_K"] <= 0.0:
        return False
    if result.kpis["COP"] <= 0.0:
        return False
    for label in ("0", "11", "21"):
        if result.states[label]["m_kg_s"] <= 0.0:
            return False
    return True


INFEASIBLE_REASONS = {
    0: "feasible",
    1: "source near freezing",
    2: "condensing temperature too close to T_crit",
    3: "solver did not converge / spurious root",
    4: "discharge temperature exceeds limit",
    5: "pressure ratio exceeds single-stage limit",
}


def run_sweep(config: SweepConfig):
    model = HeatPumpModel(refrigerant=config.refrigerant)
    model.nw.iterinfo = False
    T_crit_C = PropsSI("Tcrit", config.refrigerant) - 273.15
    T_cond_max_C = T_crit_C - config.condensing_margin_K

    n_rows, n_cols = len(T_LIFT_K), len(T_SOURCE_IN_C)
    cop = np.full((n_rows, n_cols), np.nan)
    T_sink_out = np.full((n_rows, n_cols), np.nan)
    feasible = np.zeros((n_rows, n_cols), dtype=bool)
    reason = np.zeros((n_rows, n_cols), dtype=int)

    for i, lift_K in enumerate(T_LIFT_K):
        for j, T_source_in_C in enumerate(T_SOURCE_IN_C):
            T_sink_out_C = T_source_in_C + lift_K

            # source water must not approach freezing
            if T_source_in_C - config.dT_source_spread < config.min_source_out_C:
                reason[i, j] = 1
                continue
            # sink stream must be able to be heated at all, and the
            # nominal condensing level (T_sink_out + pinch + subcooling)
            # must stay clear of the refrigerant's critical point
            T_cond_est_C = T_sink_out_C + config.dT_min_cond + config.subcooling_K
            if T_cond_est_C >= T_cond_max_C:
                reason[i, j] = 2
                continue

            if config.use_pr_dependent_eta_s:
                PR_est = _estimate_pressure_ratio(T_source_in_C, T_sink_out_C, config)
                eta_s = eta_s_from_pr(PR_est, config)
            else:
                eta_s = config.eta_s

            inputs = _build_inputs(T_source_in_C, T_sink_out_C, config, eta_s)
            try:
                result = model.solve(inputs)
            except Exception:
                # Broad on purpose: TESPy's property backend can raise a
                # raw ValueError/RuntimeError directly out of nw.solve()
                # on bad points (e.g. a property flash failing near the
                # refrigerant's critical point), not just the
                # HeatPumpEvaluationError this module raises itself on
                # non-convergence -- either way, mark the grid point
                # infeasible rather than crashing the whole sweep.
                reason[i, j] = 3
                continue

            if not _is_physically_valid(result):
                reason[i, j] = 3
                continue

            if result.states["2"]["T_C"] > config.T_discharge_max_C:
                reason[i, j] = 4
                continue
            if result.kpis["pressure_ratio"] > config.PR_max:
                reason[i, j] = 5
                continue

            # eta_s already captures the internal (shaft-power) cycle loss;
            # eta_mech_motor converts that to an electrical-input COP
            cop[i, j] = result.kpis["COP"] * config.eta_mech_motor
            T_sink_out[i, j] = T_sink_out_C
            feasible[i, j] = True

    return cop, T_sink_out, feasible, reason, T_crit_C


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(cop, T_sink_out, feasible, reason, T_crit_C, config: SweepConfig):
    fig, ax = plt.subplots(figsize=(9, 7.0))

    # near-zero lift corners give mathematically valid but practically
    # irrelevant COPs in the hundreds; clip the color scale so the
    # interesting range (a few up to ~15) stays legible
    cop_masked = np.ma.masked_invalid(cop)
    levels = np.linspace(0, 15, 21)
    cf = ax.contourf(
        T_SOURCE_IN_C, T_LIFT_K, cop_masked, levels=levels,
        cmap="viridis", extend="max",
    )
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("COP")

    # overlay the resulting absolute sink outlet temperature as a secondary
    # readout -- the lift alone doesn't tell you the useful temperature level
    T_sink_out_masked = np.ma.masked_invalid(T_sink_out)
    cs = ax.contour(
        T_SOURCE_IN_C, T_LIFT_K, T_sink_out_masked, colors="white",
        linewidths=1.0, levels=8,
    )
    ax.clabel(cs, inline=True, fmt="%.0f °C", fontsize=9)

    # hatch infeasible cells, distinguishing *why* they are infeasible:
    # thermodynamic/critical-point limits vs. practical compressor limits
    thermo_infeasible = np.isin(reason, [1, 2, 3])
    discharge_infeasible = reason == 4
    pr_infeasible = reason == 5

    legend_handles = []
    for mask, color, hatch, label in (
        (thermo_infeasible, "0.85", "///", "T_crit margin / source freezing"),
        (discharge_infeasible, "peachpuff", "xxx", f"discharge temp. > {config.T_discharge_max_C:.0f} °C"),
        (pr_infeasible, "lightblue", "...", f"pressure ratio > {config.PR_max:.0f} (single stage)"),
    ):
        if mask.any():
            ax.contourf(
                T_SOURCE_IN_C, T_LIFT_K, mask.astype(float),
                levels=[0.5, 1.5], colors=[color], hatches=[hatch],
            )
            legend_handles.append(mpatches.Patch(facecolor=color, hatch=hatch, label=label, edgecolor="0.3"))

    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=9, frameon=False,
        )

    ax.set_xlabel("Source inlet temperature T_source,in [°C]")
    ax.set_ylabel("Temperature lift T_sink,out - T_source,in [K]")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{plot_name}.svg")
    fig.savefig(f"{plot_name}.png", dpi=150)
    print(f"Saved plot to {plot_name}.svg / .png")


if __name__ == "__main__":
    config = SweepConfig()
    cop, T_sink_out, feasible, reason, T_crit_C = run_sweep(config)
    n_feasible = int(feasible.sum())
    n_total = feasible.size
    print(f"Feasible points: {n_feasible} / {n_total}")
    if n_feasible:
        print(f"COP range: {np.nanmin(cop):.2f} - {np.nanmax(cop):.2f}")
    print("Infeasibility breakdown:")
    for code, label in INFEASIBLE_REASONS.items():
        if code == 0:
            continue
        count = int((reason == code).sum())
        if count:
            print(f"  {label}: {count}")
    plot_sweep(cop, T_sink_out, feasible, reason, T_crit_C, config)
