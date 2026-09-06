"""Pure Duehring/equilibrium screening for the DOUBLE-LIFT AHT -- NO solver.

Direct analog of AHT_duehring_screening.py (single-lift), extended by a
second evaporator-absorber stage. Answers the same question as the
single-lift script ("what GTL is even thermodynamically achievable at most
at a given waste-heat temperature?"), this time for the two-stage
("double-lift") AHT design, before any pinch model or optimizer runs. Uses
only:

  - the LiBr/H2O equilibrium relation (Patek correlations in
    Thermodynamic_Properties.libr_props),
  - the Albers/Boryta crystallization limit,
  - the water saturation functions from Models.AHT_Pinch_Point (same
    CoolProp source as the actual solver model, so consistent).

No mass/energy balance solve, no cycle scaling, no circulation ratio --
exactly the same simplifications as in the single-lift script, see its
docstring. Here too the result is deliberately an OPTIMISTIC estimate
(pinch only at the binding end, not over the full counterflow temperature
profile).

Physical picture (double-lift AHT, "serial flow" design per Saito et al.
2015 / Lubis et al. 2017; see also the review Cudok et al. 2021,
"Absorption heat transformer - state-of-the-art of industrial applications",
Renew. Sustain. Energy Rev. 141, 110757, Fig. 2 right side)
-------------------------------------------------------------------------------
Unlike the single-lift AHT with 2 pressure levels, a double-lift AHT has
THREE pressure levels, but still only ONE desorber and ONE condenser:

  - Desorber (G) + condenser (C) at the LOWEST pressure level p_low: waste
    heat at T13 drives solution out in the desorber (as in the single-lift
    case), the vapor condenses at T17 (reject cooling). The desorber
    delivers the strongly concentrated solution x_strong -- identical to
    the single-lift derivation (steps 1-3 below are literally the same
    equations).

  - Low-stage evaporator/absorber pair (EL/AL) at the intermediate pressure
    level p_mid: EL is fed -- like the single evaporator in the single-lift
    case -- by the external waste heat at T15. AL absorbs this vapor with
    the (in this approximation undiluted) strong solution x_strong,
    delivering a first raised temperature T_AL -- exactly the "GTL" of the
    single-lift case, but here only the FIRST of two stages.

  - High-stage evaporator/absorber pair (EH/AH) at the highest pressure
    level p_high: the key idea of the double-lift cycle is that AL does
    NOT reject its heat externally as useful heat, but instead internally
    drives EH ("the low pressure absorber AL drives the higher pressure
    evaporator EH by internal heat exchange", Cudok et al. 2021). AH
    absorbs this second vapor stream -- again with x_strong -- and only
    here, at T_AH, delivers the actually usable heat. The total GTL
    relative to the waste heat T15 is thus the sum of two "lifts":
    (T_AL - T15) + (T_AH - T_AL).

Simplification vs. the real "serial flow" circuit (IMPORTANT, please read)
-------------------------------------------------------------------------------
In the real serial-flow circuit, ONE solution stream first passes through AH
(still undiluted there, x_strong) and only afterward -- throttled to p_mid
and already partially diluted -- through AL. So in reality AL would have a
WEAKER concentration available than x_strong, which would reduce the GTL1
achievable in AL compared to the bound computed here.

This coupling could only be resolved with a mass balance (circulation
ratio) -- exactly what the single-lift script also deliberately omits. To
keep both scripts structurally comparable and at the same level of
simplification, this script instead makes the (parallel) approximation
that both AL and AH are fed with the full, undiluted concentration
x_strong from the desorber (corresponding, in the literature, to the
"parallel feed" variant of the double/dual absorption heat transformer).
This makes the GTL_total bound computed here EVEN MORE optimistic than a
real serial-flow design would achieve -- so still a valid, just somewhat
more generous, upper bound. The full pinch model with a mass balance then
gives the actually achievable, tighter limit (analogous to the role of
AHT_feasibility_sweep.py / AHT_design_point_optimizer.py for the
single-lift case).

As in the single-lift script, crystallization is checked only at the
desorber outlet (T_gen, w_strong): that's the coldest point in the whole
cycle where the solution carries the concentration x_strong, and hence the
binding crystallization check (AL and AH sit at the same concentration but
higher temperature, so less critical).

What this deliberately does NOT capture (for speed, see also the
single-lift script):
  - mass-flow split / circulation ratio (FR) per stage
  - the real dilution of the solution passing through AH -> AL (see above)
  - SHEX heat recovery, pre-absorption
  - the actual temperature profile across the heat exchangers (only the
    binding pinch point is considered, not LMTD/counterflow)
  - UA values / equipment size

-> The result is an UPPER bound, more optimistic than the real serial-flow
   design. A full pinch model with a mass balance then gives the actually
   achievable, tighter limit.

Standalone usage
-----------------
    python Design_Point/AHT_DL_duehring_screening.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

import Thermodynamic_Properties.libr_props as lp
from Models.AHT_Pinch_Point import (
    celsius_to_kelvin,
    kelvin_to_celsius,
    water_p_sat_from_T,
)

X_LO = 1.0e-6
X_HI = lp.X_MAX_PAT - 1.0e-6


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def concentration_for_boiling_point(p_pa: float, T_target_K: float) -> float:
    """Inverts T_sat_solution_from_p_x: returns x such that the solution
    boils at exactly T_target_K at p_pa. Raises ValueError if T_target_K is
    outside the range reachable at this pressure.

    Identical to the same-named function in AHT_duehring_screening.py
    (single-lift) -- duplicated here so this script stays runnable
    standalone, without needing an import from the single-lift module."""

    def f(x: float) -> float:
        return lp.T_sat_solution_from_p_x(p_pa, x) - T_target_K

    f_lo = f(X_LO)
    f_hi = f(X_HI)
    if f_lo > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K is below the boiling point of pure "
            f"water at p={p_pa:.1f} Pa -- no concentrating possible "
            "(driving temperature too low relative to this pressure level)."
        )
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K is not reachable at p={p_pa:.1f} Pa "
            f"with any LiBr concentration in the valid range [{X_LO:.2e}, {X_HI:.6f}] "
            "(driving temperature too high / outside the Patek range)."
        )
    return float(brentq(f, X_LO, X_HI))


@dataclass(frozen=True)
class DuehringScreeningResultDoubleLift:
    T13_C: float
    T15_C: float
    T17_C: float
    dT_min_des: float
    dT_min_evap: float
    dT_min_cond: float
    dT_min_abs: float
    dT_min_evap2: float
    dT_min_abs2: float

    feasible: bool
    message: str

    p_low_Pa: float = float("nan")
    p_mid_Pa: float = float("nan")
    p_high_Pa: float = float("nan")

    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_gen_C: float = float("nan")

    # Stage 1 (EL/AL, intermediate pressure level)
    T10L_C: float = float("nan")
    T_AL_max_C: float = float("nan")
    GTL1_max_K: float = float("nan")

    # Stage 2 (EH/AH, highest pressure level) -- internally driven by AL
    T10H_C: float = float("nan")
    T_AH_max_C: float = float("nan")
    GTL2_max_K: float = float("nan")

    # Total lift relative to the waste heat T15 (= GTL1_max_K + GTL2_max_K)
    GTL_total_max_K: float = float("nan")

    crystallization_safe: bool = True
    crystallization_message: str = ""
    # True if x_strong was clamped to the solubility limit -- see the
    # comment in estimate_max_gtl_double_lift() and in the single-lift script.
    crystallization_limited: bool = False


def estimate_max_gtl_double_lift(
    T13_C: float,
    T15_C: float,
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    dT_min_evap2: float = 5.0,
    dT_min_abs2: float = 5.0,
) -> DuehringScreeningResultDoubleLift:
    """Optimistic upper bound on the achievable total GTL of a double-lift
    AHT, see module docstring.

    dT_min_des/_evap/_cond/_abs refer -- as in the single-lift script -- to
    the desorber, stage-1 evaporator (EL), condenser, and stage-1 absorber
    (AL). dT_min_evap2/_abs2 are the analogous pinch assumptions for the
    internally driven stage-2 evaporator (EH) and final absorber (AH).
    """

    common = dict(
        T13_C=T13_C, T15_C=T15_C, T17_C=T17_C,
        dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
        dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
        dT_min_evap2=dT_min_evap2, dT_min_abs2=dT_min_abs2,
    )

    # 1) Condenser pressure from T17 (pinch at the cold end) -- identical to single-lift
    T8_K = celsius_to_kelvin(T17_C) + dT_min_cond
    p_low = water_p_sat_from_T(T8_K, Q=0.0)

    # 2) Stage-1 evaporator pressure from T15 (pinch at the hot end, optimistic)
    T10L_K = celsius_to_kelvin(T15_C) - dT_min_evap
    try:
        p_mid = water_p_sat_from_T(T10L_K, Q=1.0)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False,
            message=f"Stage-1 (EL) evaporator pressure not computable: {exc}",
            p_low_Pa=p_low,
        )

    if p_mid <= p_low:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False,
            message=(
                f"p_mid ({p_mid:.0f} Pa) <= p_low ({p_low:.0f} Pa): "
                "waste-heat temperature T15 too low relative to reject "
                "cooling T17 -- stage 1 (EL/AL) cannot be driven this way."
            ),
            p_low_Pa=p_low, p_mid_Pa=p_mid,
        )

    # 3) Desorber equilibrium: x_strong from T13 and p_low -- identical to single-lift
    T_gen_K = celsius_to_kelvin(T13_C) - dT_min_des
    try:
        x_strong = concentration_for_boiling_point(p_low, T_gen_K)
    except ValueError as exc:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False, message=str(exc),
            p_low_Pa=p_low, p_mid_Pa=p_mid,
            T_gen_C=kelvin_to_celsius(T_gen_K),
            T10L_C=kelvin_to_celsius(T10L_K),
        )

    w_strong = lp.w_libr_from_x(x_strong)

    # 4) Crystallization check at the desorber outlet (coldest point at x_strong)
    validity = lp.validate_solution_state(
        T_gen_K, w_strong, label="Desorber outlet (double-lift Duehring screening)"
    )

    # 4b) Clamp to the solubility limit instead of discarding the point --
    # identical logic as in the single-lift script (commented in detail there).
    crystallization_limited = False
    if validity.crystallization_checked and not validity.crystallization_safe:
        w_lo, w_hi = 0.57, w_strong
        for _ in range(60):
            w_mid = 0.5 * (w_lo + w_hi)
            if lp.validate_solution_state(T_gen_K, w_mid).crystallization_safe:
                w_lo = w_mid
            else:
                w_hi = w_mid

        x_strong = lp.x_from_w_libr(w_lo)
        w_strong = lp.w_libr_from_x(x_strong)
        validity = lp.validate_solution_state(
            T_gen_K, w_strong,
            label="Desorber outlet (double-lift Duehring screening, clamped to solubility limit)",
        )
        crystallization_limited = True

    # 5) Stage 1: AL with x_strong at p_mid -> T_AL_max (= "GTL1", analogous
    #    to T12_max in the single-lift script)
    try:
        T_AL_solution_K = lp.T_sat_solution_from_p_x(p_mid, x_strong)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False,
            message=f"Stage-1 (AL) absorber equilibrium temperature not computable: {exc}",
            p_low_Pa=p_low, p_mid_Pa=p_mid,
            x_strong=x_strong, w_strong=w_strong,
            T_gen_C=kelvin_to_celsius(T_gen_K), T10L_C=kelvin_to_celsius(T10L_K),
            crystallization_safe=validity.crystallization_safe,
            crystallization_message=validity.message,
            crystallization_limited=crystallization_limited,
        )

    T_AL_max_K = T_AL_solution_K - dT_min_abs
    T_AL_max_C = kelvin_to_celsius(T_AL_max_K)
    GTL1_max_K = T_AL_max_C - T15_C

    partial_common = dict(
        **common,
        p_low_Pa=p_low, p_mid_Pa=p_mid,
        x_strong=x_strong, w_strong=w_strong,
        T_gen_C=kelvin_to_celsius(T_gen_K), T10L_C=kelvin_to_celsius(T10L_K),
        T_AL_max_C=T_AL_max_C, GTL1_max_K=GTL1_max_K,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
        crystallization_limited=crystallization_limited,
    )

    if not validity.crystallization_safe:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False,
            message=f"Crystallization risk: {validity.message}",
        )
    if GTL1_max_K <= 0.0:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False,
            message=(
                f"T_AL_max ({T_AL_max_C:.2f} °C) is not above T15 "
                f"({T15_C:.2f} °C) -- stage 1 delivers no positive lift, "
                "stage 2 cannot be driven this way."
            ),
        )

    # 6) Stage-2 (EH) evaporator pressure: driven by AL, pinch at the hot end
    T10H_K = T_AL_max_K - dT_min_evap2
    try:
        p_high = water_p_sat_from_T(T10H_K, Q=1.0)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False,
            message=f"Stage-2 (EH) evaporator pressure not computable: {exc}",
        )

    if p_high <= p_mid:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=(
                f"p_high ({p_high:.0f} Pa) <= p_mid ({p_mid:.0f} Pa): "
                "T_AL_max too low relative to p_mid to drive EH (stage 2) "
                "over the pinch dT_min_evap2 assumed here."
            ),
        )

    # 7) Stage 2: AH with x_strong at p_high -> T_AH_max (final useful heat)
    try:
        T_AH_solution_K = lp.T_sat_solution_from_p_x(p_high, x_strong)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=f"Stage-2 (AH) absorber equilibrium temperature not computable: {exc}",
        )

    T_AH_max_K = T_AH_solution_K - dT_min_abs2
    T_AH_max_C = kelvin_to_celsius(T_AH_max_K)
    GTL2_max_K = T_AH_max_C - T_AL_max_C
    GTL_total_max_K = T_AH_max_C - T15_C

    feasible = validity.crystallization_safe and GTL1_max_K > 0.0 and GTL2_max_K > 0.0
    if GTL2_max_K <= 0.0:
        message = (
            f"T_AH_max ({T_AH_max_C:.2f} °C) is not above T_AL_max "
            f"({T_AL_max_C:.2f} °C) -- stage 2 delivers no positive "
            "additional lift."
        )
    elif crystallization_limited:
        message = "OK (clamped to solubility limit, full desorber pinch not used)."
    else:
        message = "OK (optimistic upper bound, parallel-feed approximation)."

    return DuehringScreeningResultDoubleLift(
        **partial_common, feasible=feasible, message=message,
        p_high_Pa=p_high, T10H_C=kelvin_to_celsius(T10H_K),
        T_AH_max_C=T_AH_max_C, GTL2_max_K=GTL2_max_K,
        GTL_total_max_K=GTL_total_max_K,
    )


# ---------------------------------------------------------------------------
# Sweep over waste-heat temperature (T13 = T15, "parallel" case, as in single-lift)
# ---------------------------------------------------------------------------

def sweep_waste_heat_temperature_double_lift(
    T_waste_values_C: Sequence[float],
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    dT_min_evap2: float = 5.0,
    dT_min_abs2: float = 5.0,
) -> List[DuehringScreeningResultDoubleLift]:
    """Sets T13 = T15 = T_waste (parallel routing of desorber and stage-1
    evaporator) and evaluates estimate_max_gtl_double_lift() for each value
    in T_waste_values_C."""
    return [
        estimate_max_gtl_double_lift(
            T13_C=t, T15_C=t, T17_C=T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
            dT_min_evap2=dT_min_evap2, dT_min_abs2=dT_min_abs2,
        )
        for t in T_waste_values_C
    ]


def print_results_table(results: Sequence[DuehringScreeningResultDoubleLift]) -> None:
    print("=" * 140)
    print(
        f"{'T_waste[C]':>10} {'T17[C]':>7} {'x_strong':>9} "
        f"{'T_AL_max[C]':>12} {'GTL1[K]':>8} "
        f"{'T_AH_max[C]':>12} {'GTL2[K]':>8} {'GTL_tot[K]':>11} "
        f"{'Cryst.':>10}  Note"
    )
    print("-" * 140)
    for r in results:
        if not r.crystallization_safe:
            krist = "RISK"
        elif r.crystallization_limited:
            krist = "clamped"
        else:
            krist = "safe"
        print(
            f"{r.T13_C:10.2f} {r.T17_C:7.2f} {r.x_strong:9.4f} "
            f"{r.T_AL_max_C:12.2f} {r.GTL1_max_K:8.2f} "
            f"{r.T_AH_max_C:12.2f} {r.GTL2_max_K:8.2f} {r.GTL_total_max_K:11.2f} "
            f"{krist:>10}  {r.message}"
        )
    print("=" * 140)


def plot_gtl_vs_waste_heat_double_lift(
    T_waste_values_C: Sequence[float],
    T17_values_C: Sequence[float],
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    dT_min_evap2: float = 5.0,
    dT_min_abs2: float = 5.0,
    save_path: Optional[str] = "Design_Point/Plots/duehring_screening_GTL_double_lift_3K.png",
    show: bool = True,
):
    """One curve of GTL_total_max vs. waste-heat temperature per T17 value.
    Infeasible/crystallization-risk points are drawn as hollow markers --
    layout identical to plot_gtl_vs_waste_heat() in the single-lift script."""
    import matplotlib.pyplot as plt

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_prop_cycle(color=palette)

    all_results = {}
    for T17_C in T17_values_C:
        results = sweep_waste_heat_temperature_double_lift(
            T_waste_values_C, T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
            dT_min_evap2=dT_min_evap2, dT_min_abs2=dT_min_abs2,
        )
        all_results[T17_C] = results

        x = np.array([r.T13_C for r in results])
        y = np.array([r.GTL_total_max_K for r in results])
        ok = np.array([r.feasible for r in results])

        (line,) = ax.plot(x[ok], y[ok], "o-", linewidth=1.8, markersize=5, label=f"T17 = {T17_C:.0f} °C")
        if np.any(~ok):
            ax.plot(x[~ok], y[~ok], "x", color=line.get_color(), markersize=7, markeredgewidth=2)

    ax.set_xlabel("Waste-heat temperature T13 = T15 [°C]")
    ax.set_ylabel("Max. total GTL [K]")
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.legend(title="× not feasible", frameon=False, fontsize=9, title_fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if show:
        plt.show()

    return fig, ax, all_results


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # ADJUST HERE: pinch assumptions for the optimistic estimate
    # ------------------------------------------------------------------
    DT_MIN_DES = 5.0
    DT_MIN_EVAP = 5.0
    DT_MIN_COND = 5.0
    DT_MIN_ABS = 5.0
    DT_MIN_EVAP2 = 5.0
    DT_MIN_ABS2 = 5.0

    T_WASTE_RANGE_C = list(np.arange(40.0, 85.0, 5.0))
    T17_CURVES_C = [15.0, 20.0, 25.0]

    print("Double-lift Duehring screening for T17 = 20 °C:")
    results = sweep_waste_heat_temperature_double_lift(
        T_WASTE_RANGE_C, T17_C=20.0,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
        dT_min_evap2=DT_MIN_EVAP2, dT_min_abs2=DT_MIN_ABS2,
    )
    print_results_table(results)

    plot_gtl_vs_waste_heat_double_lift(
        T_WASTE_RANGE_C, T17_CURVES_C,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
        dT_min_evap2=DT_MIN_EVAP2, dT_min_abs2=DT_MIN_ABS2,
    )