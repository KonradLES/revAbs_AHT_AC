"""Pure Duehring/equilibrium screening for the AHT -- NO solver.

Answers "what GTL is even *thermodynamically achievable at most* at a given
waste-heat temperature", before any pinch model or optimizer runs. Uses only:

  - the LiBr/H2O equilibrium relation (Patek correlations in
    Thermodynamic_Properties.libr_props),
  - the Albers/Boryta crystallization limit,
  - the water saturation functions from Models.AHT_Pinch_Point (same
    CoolProp source as the actual solver model, so consistent).

No mass/energy balance solve, no cycle scaling, no circulation ratio. The
result is deliberately an OPTIMISTIC estimate (pinch only at the binding
end, not over the full counterflow temperature profile):

Physical picture (AHT, as implemented in Models.AHT_Pinch_Point)
-------------------------------------------------------------------
- Desorber + condenser sit on the LOW-pressure side (p_low): waste heat at
  T13 drives solution out in the desorber, the vapor condenses at T17
  (reject cooling).
- Evaporator + absorber sit on the HIGH-pressure side (p_high): waste heat
  at T15 evaporates the refrigerant at p_high (hence the higher pressure
  level than the condenser!); the vapor is absorbed in the absorber by the
  solution concentrated ("strong") in the desorber, delivering useful heat
  at T12 > T15 -- this is the actual "transformer" lift.

Core idea of the estimate
--------------------------
1. p_low from T17 (condenser pinch at the cold end: T8 = T17 + dT_min_cond).
2. p_high from T15 (evaporator pinch at the hot end, optimistic:
   T10 = T15 - dT_min_evap).
3. The maximum concentration x_strong reachable in the desorber follows
   directly from the Duehring equilibrium condition
       T_sat_solution(p_low, x_strong) = T13 - dT_min_des
   (the solution boils up to the concentration whose boiling point at
   p_low equals the driving temperature minus the pinch).
4. Crystallization check at (T13 - dT_min_des, w_strong).
5. The maximum deliverable useful-heat temperature follows from the
   boiling point of the SAME (strong) concentration at p_high, minus the
   absorber pinch:
       T12_max = T_sat_solution(p_high, x_strong) - dT_min_abs
       GTL_max = T12_max - T15

This chain needs no circulation ratio (m_strong/m_weak): the *hottest*-
entering (= strongest) solution is what sets the maximum absorber
temperature -- exactly the state step 3 produces. This matches the classic
graphical Duehring construction (two isotherms + two isosteres), also drawn
as a hexagon in Postprocessing/AHT_Duehring_Plot.py.

What this deliberately does NOT capture (for speed):
  - mass-flow split / circulation ratio (FR)
  - SHEX heat recovery, pre-absorption
  - the actual temperature profile across the heat exchangers (only the
    binding pinch point is considered, not LMTD/counterflow)
  - UA values / equipment size

-> The result is an UPPER bound. The full pinch model
   (AHT_feasibility_sweep.py, AHT_design_point_optimizer.py) then gives the
   actually achievable, tighter limit.

Standalone usage
-----------------
    python Design_Point/AHT_duehring_screening.py
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
    water_T_sat_from_p,
)

X_LO = 1.0e-6
X_HI = lp.X_MAX_PAT - 1.0e-6


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def concentration_for_boiling_point(p_pa: float, T_target_K: float) -> float:
    """Inverts T_sat_solution_from_p_x: returns x such that the solution boils
    at exactly T_target_K at p_pa. Raises ValueError if T_target_K is outside
    the range reachable at this pressure."""

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
class DuehringScreeningResult:
    T13_C: float
    T15_C: float
    T17_C: float
    dT_min_des: float
    dT_min_evap: float
    dT_min_cond: float
    dT_min_abs: float

    feasible: bool
    message: str

    p_low_Pa: float = float("nan")
    p_high_Pa: float = float("nan")
    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_gen_C: float = float("nan")
    T10_C: float = float("nan")
    T12_max_C: float = float("nan")
    GTL_max_K: float = float("nan")
    crystallization_safe: bool = True
    crystallization_message: str = ""
    # True if x_strong was clamped to the solubility limit (see
    # estimate_max_gtl()): T12_max_K/GTL_max_K are then limited NOT by the
    # desorber pinch (full driving temperature used) but by the
    # crystallization limit -- still a valid upper bound, just solubility-
    # rather than temperature-limited.
    crystallization_limited: bool = False


def estimate_max_gtl(
    T13_C: float,
    T15_C: float,
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> DuehringScreeningResult:
    """Optimistic upper bound on the achievable GTL, see module docstring."""

    common = dict(
        T13_C=T13_C, T15_C=T15_C, T17_C=T17_C,
        dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
        dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
    )

    # 1) Condenser pressure from T17 (pinch at the cold end)
    T8_K = celsius_to_kelvin(T17_C) + dT_min_cond
    p_low = water_p_sat_from_T(T8_K, Q=0.0)

    # 2) Evaporator pressure from T15 (pinch at the hot end, optimistic)
    T10_K = celsius_to_kelvin(T15_C) - dT_min_evap
    try:
        p_high = water_p_sat_from_T(T10_K, Q=1.0)
    except Exception as exc:
        return DuehringScreeningResult(
            **common, feasible=False,
            message=f"Evaporator pressure not computable: {exc}",
            p_low_Pa=p_low,
        )

    if p_high <= p_low:
        return DuehringScreeningResult(
            **common, feasible=False,
            message=(
                f"p_high ({p_high:.0f} Pa) <= p_low ({p_low:.0f} Pa): "
                "waste-heat temperature T15 too low relative to reject "
                "cooling T17 -- this pressure ratio cannot drive an AHT."
            ),
            p_low_Pa=p_low, p_high_Pa=p_high,
        )

    # 3) Desorber equilibrium: x_strong from T13 and p_low
    T_gen_K = celsius_to_kelvin(T13_C) - dT_min_des
    try:
        x_strong = concentration_for_boiling_point(p_low, T_gen_K)
    except ValueError as exc:
        return DuehringScreeningResult(
            **common, feasible=False, message=str(exc),
            p_low_Pa=p_low, p_high_Pa=p_high,
            T_gen_C=kelvin_to_celsius(T_gen_K),
            T10_C=kelvin_to_celsius(T10_K),
        )

    w_strong = lp.w_libr_from_x(x_strong)

    # 4) Crystallization check at the desorber outlet
    validity = lp.validate_solution_state(
        T_gen_K, w_strong, label="Desorber outlet (Duehring screening)"
    )

    # 4b) Clamp to the solubility limit instead of discarding the point.
    #
    # x_strong above is the concentration whose boiling point at p_low
    # equals EXACTLY the full driving temperature (T13 - dT_min_des). If
    # this concentration isn't soluble, the real solution can't actually
    # concentrate up that far -- it crystallizes first and stays at the
    # solubility limit. The desorber is then no longer pinch-limited but
    # solubility-limited (more driving temperature is available than can be
    # used). GTL_max is therefore recomputed from the clamped concentration
    # -- still a valid upper bound, just with a different binding constraint.
    crystallization_limited = False
    if validity.crystallization_checked and not validity.crystallization_safe:
        # Bisect DIRECTLY on the real safety check (not just on the T->w
        # correlation): the two crystallization correlations (T_cr(w) and
        # w_cr(T)) are independent fits, not exact inverses of each other --
        # at higher temperatures they diverge by more than 1 K. w=0.57 is
        # always safe by definition (see validate_solution_state), and
        # w_strong is unsafe here by precondition -- plain bisection
        # converges to the actual limit regardless of which correlation binds.
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
            label="Desorber outlet (Duehring screening, clamped to solubility limit)",
        )
        crystallization_limited = True

    # 5) Absorber: same (strong, possibly clamped) concentration at p_high -> T12_max
    try:
        T_abs_solution_K = lp.T_sat_solution_from_p_x(p_high, x_strong)
    except Exception as exc:
        return DuehringScreeningResult(
            **common, feasible=False,
            message=f"Absorber equilibrium temperature not computable: {exc}",
            p_low_Pa=p_low, p_high_Pa=p_high,
            x_strong=x_strong, w_strong=w_strong,
            T_gen_C=kelvin_to_celsius(T_gen_K), T10_C=kelvin_to_celsius(T10_K),
            crystallization_safe=validity.crystallization_safe,
            crystallization_message=validity.message,
            crystallization_limited=crystallization_limited,
        )

    T12_max_K = T_abs_solution_K - dT_min_abs
    T12_max_C = kelvin_to_celsius(T12_max_K)
    GTL_max_K = T12_max_C - T15_C

    feasible = validity.crystallization_safe and GTL_max_K > 0.0
    if not validity.crystallization_safe:
        message = f"Crystallization risk: {validity.message}"
    elif GTL_max_K <= 0.0:
        message = (
            f"T12_max ({T12_max_C:.2f} °C) is not above T15 ({T15_C:.2f} °C) "
            "-- no positive GTL achievable."
        )
    elif crystallization_limited:
        message = "OK (clamped to solubility limit, full desorber pinch not used)."
    else:
        message = "OK (optimistic upper bound)."

    return DuehringScreeningResult(
        **common, feasible=feasible, message=message,
        p_low_Pa=p_low, p_high_Pa=p_high,
        x_strong=x_strong, w_strong=w_strong,
        T_gen_C=kelvin_to_celsius(T_gen_K), T10_C=kelvin_to_celsius(T10_K),
        T12_max_C=T12_max_C, GTL_max_K=GTL_max_K,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
        crystallization_limited=crystallization_limited,
    )


# ---------------------------------------------------------------------------
# Sweep over waste-heat temperature (T13 = T15, "parallel" case)
# ---------------------------------------------------------------------------

def sweep_waste_heat_temperature(
    T_waste_values_C: Sequence[float],
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> List[DuehringScreeningResult]:
    """Sets T13 = T15 = T_waste (parallel routing) and evaluates
    estimate_max_gtl() for each value in T_waste_values_C."""
    return [
        estimate_max_gtl(
            T13_C=t, T15_C=t, T17_C=T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
        )
        for t in T_waste_values_C
    ]


def print_results_table(results: Sequence[DuehringScreeningResult]) -> None:
    print("=" * 100)
    print(
        f"{'T_waste[C]':>10} {'T17[C]':>7} {'x_strong':>9} {'w_strong':>9} "
        f"{'T12_max[C]':>11} {'GTL_max[K]':>11} {'Cryst.':>10}  Note"
    )
    print("-" * 100)
    for r in results:
        if not r.crystallization_safe:
            krist = "RISK"
        elif r.crystallization_limited:
            krist = "clamped"
        else:
            krist = "safe"
        print(
            f"{r.T13_C:10.2f} {r.T17_C:7.2f} {r.x_strong:9.4f} {r.w_strong:9.4f} "
            f"{r.T12_max_C:11.2f} {r.GTL_max_K:11.2f} {krist:>10}  {r.message}"
        )
    print("=" * 100)


def plot_gtl_vs_waste_heat(
    T_waste_values_C: Sequence[float],
    T17_values_C: Sequence[float],
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    save_path: Optional[str] = "Design_Point/Plots/duehring_screening_GTL_3K.png",
    show: bool = True,
):
    """One curve of GTL_max vs. waste-heat temperature per T17 value.
    Infeasible/crystallization-risk points are drawn as hollow markers."""
    import matplotlib.pyplot as plt

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_prop_cycle(color=palette)

    all_results = {}
    for T17_C in T17_values_C:
        results = sweep_waste_heat_temperature(
            T_waste_values_C, T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
        )
        all_results[T17_C] = results

        x = np.array([r.T13_C for r in results])
        y = np.array([r.GTL_max_K for r in results])
        ok = np.array([r.feasible for r in results])

        (line,) = ax.plot(x[ok], y[ok], "o-", linewidth=1.8, markersize=5, label=f"T17 = {T17_C:.0f} °C")
        if np.any(~ok):
            ax.plot(x[~ok], y[~ok], "x", color=line.get_color(), markersize=7, markeredgewidth=2)

    ax.set_xlabel("Waste-heat temperature T13 = T15 [°C]")
    ax.set_ylabel("Max. GTL [K]")
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
    DT_MIN_DES = 3.0
    DT_MIN_EVAP = 3.0
    DT_MIN_COND = 3.0
    DT_MIN_ABS = 3.0

    T_WASTE_RANGE_C = list(np.arange(40.0, 100.0, 5.0))
    T17_CURVES_C = [15.0, 20.0, 25.0]

    print("Duehring screening for T17 = 20 °C:")
    results = sweep_waste_heat_temperature(
        T_WASTE_RANGE_C, T17_C=20.0,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )
    print_results_table(results)

    plot_gtl_vs_waste_heat(
        T_WASTE_RANGE_C, T17_CURVES_C,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )
