"""Pure Duehring/equilibrium screening for the AC (single-effect absorption
chiller) -- NO solver.

Third script in the Duehring-screening family, after AHT_duehring_screening.py
(single-lift AHT) and AHT_DL_duehring_screening.py (double-lift AHT). Same
philosophy, same building blocks (Patek correlations from
Thermodynamic_Properties.libr_props, water saturation from
Models.AHT_Pinch_Point, Albers/Boryta crystallization limit), but different
routing: in the AC, desorber AND condenser sit on the HIGH-pressure side
(p_high), absorber AND evaporator on the LOW-pressure side (p_low) -- exactly
reversed from the AHT. The "free", resulting quantity here is no longer the
absorber temperature (as in the AHT) but the evaporator temperature T_evap,
i.e. the actual cooling-duty temperature.

No mass/energy balance solve, no cycle scaling, no circulation ratio, no
SHEX -- exactly the same simplifications as in the two AHT scripts. The
result is again a deliberately OPTIMISTIC upper/lower bound.

Physical picture
-----------------
- Desorber (generator) + condenser at p_high: driving heat at T_gen expels
  solution in the desorber (concentration x_strong); the vapor condenses in
  the condenser against reject-cooling water at T_reject.
- Evaporator + absorber at p_low: the refrigerant (water) evaporates at
  T_evap, delivering the cooling duty. The (undiluted) strong solution
  x_strong goes -- with no SHEX -- straight from the desorber to the
  absorber, is cooled there by the reject-cooling water (also at T_reject),
  and absorbs the vapor.

Two calculation directions are offered because they answer different
questions and handle the crystallization limit differently (see below):

1) FORWARD -- estimate_min_evap_temperature(T_gen, T_reject):
   "What's the lowest possible evaporator temperature at a given driving and
   reject-cooling temperature?" Direct analog of the question asked by the
   two AHT scripts. NOTE: for most realistic (T_gen, T_reject) combinations
   the plain equilibrium bound drops quickly far below 0 °C and so isn't very
   informative as a plot -- there's deliberately no standard plot for it
   here anymore. The function remains available, e.g. for one-off
   evaluations or to check the crystallization limit at a specific point.

2) BACKWARD -- estimate_min_generator_temperature(T_evap_target, T_reject):
   "What's the minimum driving temperature I need, at a given reject-cooling
   temperature, to still reach a given target evaporator temperature?" This
   is usually the more practically relevant question (screening: is my
   waste-heat source enough for the required cooling temperature, as a
   function of ambient/reject-cooling temperature over the year?), and it
   typically shows much more structure in a plot than variant 1.

Crystallization: two structurally different cases
----------------------------------------------------
Without a SHEX, the coldest point in the cycle where the solution still
carries the full concentration x_strong is NOT (as in the AHT) the desorber
outlet but the ABSORBER: the strong solution coming from the hot desorber is
cooled there directly by the (cold) reject-cooling water. That's exactly the
crystallization case known from practice (cold reject-cooling water in
winter). The two calculation directions therefore handle a violation of this
limit differently:

- FORWARD: x_strong is determined from the desorber (T_gen, p_high)
  independently of T_reject. If it turns out too concentrated at the
  absorber (which depends on T_reject), x_strong is clamped to the
  solubility limit -- as in the AHT scripts -- and the calculation continues
  with the clamped (weaker) solution (crystallization_limited=True). This
  makes sense because the desorber simply COULD deliver more concentration
  than may safely arrive at the absorber.

- BACKWARD: here x_strong is determined directly FROM the absorber boundary
  condition (T_evap_target, T_reject), independent of T_gen. If the
  concentration needed for that is already insoluble at T_reject, there is
  NO T_gen that changes anything about it -- the combination
  (T_evap_target, T_reject) is then structurally infeasible, not just
  "clamped".

What this deliberately does NOT capture (for speed, see also the two AHT
scripts): mass-flow split/circulation ratio, SHEX heat recovery, the real
counterflow temperature profile across the heat exchangers (only the
binding pinch point), UA values/equipment size. In particular, the missing
SHEX pre-cooling of the strong solution makes the crystallization risk
computed here at the absorber MORE CONSERVATIVE (more pessimistic) than a
real plant with a SHEX -- the opposite of this script family's usual
"optimistic upper bound" nature. Worth keeping in mind when weighing the two
effects against each other.

Naming convention: T_gen/T_cond/T_abs/T_evap instead of fixed state numbers,
since the state-point numbering of your AC solver model isn't assumed here --
feel free to rename 1:1 if you have a fixed convention.

Standalone usage
-----------------
    python Design_Point/AC_duehring_screening.py
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
# Core functions (inverting the Duehring relation, in both directions)
# ---------------------------------------------------------------------------

def concentration_for_boiling_point(p_pa: float, T_target_K: float) -> float:
    """Inverts T_sat_solution_from_p_x for x: returns x such that the solution
    boils at exactly T_target_K at p_pa. Identical to the same-named
    functions in the two AHT scripts."""

    def f(x: float) -> float:
        return lp.T_sat_solution_from_p_x(p_pa, x) - T_target_K

    f_lo = f(X_LO)
    f_hi = f(X_HI)
    if f_lo > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K is below the boiling point of pure "
            f"water at p={p_pa:.1f} Pa -- no concentrating possible."
        )
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K is not reachable at p={p_pa:.1f} Pa "
            f"with any LiBr concentration in the valid range [{X_LO:.2e}, {X_HI:.6f}]."
        )
    return float(brentq(f, X_LO, X_HI))


def pressure_for_boiling_point(
    x: float, T_target_K: float, p_floor: float = 1.0e-3
) -> float:
    """Inverts T_sat_solution_from_p_x for p (at fixed concentration x):
    returns p such that the solution at concentration x is in equilibrium
    exactly at T_target_K. Needed for the AC to get the evaporator pressure
    p_low from the absorber boundary condition (T_abs_max, x_strong) -- the
    inverse of what concentration_for_boiling_point does.

    Upper bracket: the pressure at which PURE water boils at exactly
    T_target_K (Q=0.0) -- because LiBr raises the boiling point, the
    solution at THIS pressure only boils at a higher temperature, i.e.
    f(p_hi) = T_sat_solution(p_hi, x) - T_target_K > 0 for x > 0.

    Lower bracket: NOT fixed (e.g. 1 Pa), but searched adaptively. For weak
    concentrations (x near 0, e.g. a dilute solution at low T_gen), the
    sought pressure is often already several thousand Pa -- a fixed, very
    low lower bound would then easily fall below the valid Patek
    temperature range [T_MIN_PAT, T_MAX_PAT], and T_sat_solution_from_p_x()
    raises PropertyError there instead of returning a (very negative)
    value. The search below exploits exactly this: it bisects
    geometrically (pressures span multiple orders of magnitude) until it
    finds a pressure where the evaluation is still valid AND returns a
    negative value -- that point is then a safe, evaluable lower bracket
    for brentq()."""

    p_hi = water_p_sat_from_T(T_target_K, Q=0.0)

    def f(p: float) -> float:
        return lp.T_sat_solution_from_p_x(p, x) - T_target_K

    try:
        f_hi = f(p_hi)
    except lp.PropertyError as exc:
        raise ValueError(
            f"T_target={T_target_K:.3f} K at x={x:.4f} not evaluable "
            f"(upper bracket p_hi={p_hi:.1f} Pa): {exc}"
        ) from exc
    if f_hi <= 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K for x={x:.4f} does not exceed the "
            f"boiling point of pure water (p_hi={p_hi:.1f} Pa) -- "
            "unexpected/inconsistent state."
        )

    low, high = p_floor, p_hi
    p_valid_negative = None
    for _ in range(80):
        mid = (low * high) ** 0.5  # geometric mean (pressures span multiple orders of magnitude)
        try:
            f_mid = f(mid)
        except lp.PropertyError:
            # mid is below the valid Patek temperature range for this x --
            # raise the lower bound.
            low = mid
            continue
        if f_mid >= 0.0:
            high = mid
        else:
            p_valid_negative = mid
            break
    else:
        p_valid_negative = None

    if p_valid_negative is None:
        raise ValueError(
            f"T_target={T_target_K:.3f} K for x={x:.4f} is not reachable with "
            "any pressure in the valid Patek temperature range (the adaptive "
            "search for a lower bracket did not converge either)."
        )

    return float(brentq(f, p_valid_negative, p_hi))


# ---------------------------------------------------------------------------
# 1) FORWARD: lowest possible evaporator temperature at (T_gen, T_reject)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinEvapResult:
    T_gen_C: float
    T_recool_C: float
    dT_min_des: float
    dT_min_cond: float
    dT_min_abs: float

    feasible: bool
    message: str

    p_high_Pa: float = float("nan")
    p_low_Pa: float = float("nan")
    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_abs_max_C: float = float("nan")
    T_evap_min_C: float = float("nan")

    crystallization_safe: bool = True
    crystallization_message: str = ""
    crystallization_limited: bool = False


def estimate_min_evap_temperature(
    T_gen_C: float,
    T_recool_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> MinEvapResult:
    """Optimistic LOWER bound on the achievable evaporator temperature at a
    given driving temperature T_gen and reject-cooling temperature T_recool
    (applies equally to absorber and condenser). See module docstring for
    the crystallization handling (clamped, not infeasible)."""

    common = dict(
        T_gen_C=T_gen_C, T_recool_C=T_recool_C,
        dT_min_des=dT_min_des, dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
    )

    # 1) Condenser pressure from T_recool (pinch: condenser must be warmer
    #    than the cooling water)
    T_cond_K = celsius_to_kelvin(T_recool_C) + dT_min_cond
    p_high = water_p_sat_from_T(T_cond_K, Q=0.0)

    # 2) Desorber equilibrium: x_strong from T_gen and p_high
    T_gen_eff_K = celsius_to_kelvin(T_gen_C) - dT_min_des
    try:
        x_strong = concentration_for_boiling_point(p_high, T_gen_eff_K)
    except ValueError as exc:
        return MinEvapResult(**common, feasible=False, message=str(exc), p_high_Pa=p_high)

    w_strong = lp.w_libr_from_x(x_strong)

    # 3) Absorber boundary condition: best possible (coldest) absorber
    #    temperature via the reject-cooling water
    T_abs_max_K = celsius_to_kelvin(T_recool_C) + dT_min_abs
    T_abs_max_C = kelvin_to_celsius(T_abs_max_K)

    # 4) Crystallization check at EXACTLY this point: undiluted x_strong,
    #    cooled to T_abs_max -- the coldest point in the cycle at this
    #    concentration (no SHEX modeled).
    validity = lp.validate_solution_state(
        T_abs_max_K, w_strong, label="Absorber inlet (AC Duehring screening)"
    )

    crystallization_limited = False
    if validity.crystallization_checked and not validity.crystallization_safe:
        # Clamp to the solubility limit at T_abs_max -- identical bisection
        # logic as in the two AHT scripts, just evaluated at a different
        # temperature.
        w_lo, w_hi = 0.57, w_strong
        for _ in range(60):
            w_mid = 0.5 * (w_lo + w_hi)
            if lp.validate_solution_state(T_abs_max_K, w_mid).crystallization_safe:
                w_lo = w_mid
            else:
                w_hi = w_mid

        x_strong = lp.x_from_w_libr(w_lo)
        w_strong = lp.w_libr_from_x(x_strong)
        validity = lp.validate_solution_state(
            T_abs_max_K, w_strong,
            label="Absorber inlet (AC Duehring screening, clamped to solubility limit)",
        )
        crystallization_limited = True

    partial_common = dict(
        **common, p_high_Pa=p_high,
        x_strong=x_strong, w_strong=w_strong, T_abs_max_C=T_abs_max_C,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
        crystallization_limited=crystallization_limited,
    )

    if not validity.crystallization_safe:
        # Should practically no longer occur after clamping, but kept as a
        # safeguard (e.g. in case even w=0.57 itself were unsafe).
        return MinEvapResult(
            **partial_common, feasible=False,
            message=f"Crystallization risk (even after clamping): {validity.message}",
        )

    # 5) Evaporator pressure p_low from (x_strong, T_abs_max) -- inverting
    #    the Duehring relation for p.
    try:
        p_low = pressure_for_boiling_point(x_strong, T_abs_max_K)
    except ValueError as exc:
        return MinEvapResult(**partial_common, feasible=False, message=str(exc))

    if p_low >= p_high:
        return MinEvapResult(
            **partial_common, feasible=False, p_low_Pa=p_low,
            message=(
                f"p_low ({p_low:.0f} Pa) >= p_high ({p_high:.0f} Pa): no "
                "pressure drop -- T_gen too low relative to T_recool."
            ),
        )

    # 6) Evaporation temperature of the pure refrigerant at p_low
    T_evap_min_K = water_T_sat_from_p(p_low, Q=1.0)
    T_evap_min_C = kelvin_to_celsius(T_evap_min_K)

    message = (
        "OK (clamped to solubility limit)." if crystallization_limited
        else "OK (optimistic lower bound)."
    )
    return MinEvapResult(
        **partial_common, feasible=True, message=message,
        p_low_Pa=p_low, T_evap_min_C=T_evap_min_C,
    )


# ---------------------------------------------------------------------------
# 2) BACKWARD: minimum driving temperature for a target evaporator temperature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinGenResult:
    T_evap_target_C: float
    T_recool_C: float
    dT_min_des: float
    dT_min_cond: float
    dT_min_abs: float

    feasible: bool
    message: str

    p_high_Pa: float = float("nan")
    p_low_Pa: float = float("nan")
    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_abs_max_C: float = float("nan")
    T_gen_min_C: float = float("nan")

    crystallization_safe: bool = True
    crystallization_message: str = ""


def estimate_min_generator_temperature(
    T_evap_target_C: float,
    T_recool_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> MinGenResult:
    """Minimum driving temperature T_gen that -- purely from equilibrium --
    is needed to still reach the target evaporator temperature
    T_evap_target at a given reject-cooling temperature T_recool. Unlike
    estimate_min_evap_temperature(), a crystallization violation here leads
    to a genuine infeasible (see module docstring) instead of clamping,
    because x_strong follows directly from the absorber boundary condition
    here, independent of T_gen."""

    common = dict(
        T_evap_target_C=T_evap_target_C, T_recool_C=T_recool_C,
        dT_min_des=dT_min_des, dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
    )

    # 1) Evaporator pressure directly from the target evaporator temperature
    p_low = water_p_sat_from_T(celsius_to_kelvin(T_evap_target_C), Q=1.0)

    # 2) Absorber boundary condition via the reject-cooling water
    T_abs_max_K = celsius_to_kelvin(T_recool_C) + dT_min_abs
    T_abs_max_C = kelvin_to_celsius(T_abs_max_K)

    # 3) Which concentration is at least needed at the absorber to still
    #    absorb at p_low, given the absorber can't get colder than T_abs_max?
    try:
        x_strong = concentration_for_boiling_point(p_low, T_abs_max_K)
    except ValueError as exc:
        return MinGenResult(
            **common, feasible=False, message=str(exc),
            p_low_Pa=p_low, T_abs_max_C=T_abs_max_C,
        )

    w_strong = lp.w_libr_from_x(x_strong)

    # 4) Crystallization check at exactly this (T_abs_max, w_strong) -- NO
    #    clamping here: if this is already unsafe, no T_gen in the world
    #    helps (x_strong is already the fixed boundary condition sought).
    validity = lp.validate_solution_state(
        T_abs_max_K, w_strong,
        label="Absorber inlet (AC Duehring screening, backward)",
    )

    partial_common = dict(
        **common, p_low_Pa=p_low, x_strong=x_strong, w_strong=w_strong,
        T_abs_max_C=T_abs_max_C,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
    )

    if not validity.crystallization_safe:
        return MinGenResult(
            **partial_common, feasible=False,
            message=(
                f"T_evap_target={T_evap_target_C:.2f} °C is not reachable at "
                f"T_recool={T_recool_C:.2f} °C with ANY T_gen -- "
                f"crystallization risk at the absorber: {validity.message}"
            ),
        )

    # 5) Condenser pressure (independent of T_evap_target, only of T_recool)
    T_cond_K = celsius_to_kelvin(T_recool_C) + dT_min_cond
    p_high = water_p_sat_from_T(T_cond_K, Q=0.0)

    if p_high <= p_low:
        return MinGenResult(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=(
                f"p_high ({p_high:.0f} Pa) <= p_low ({p_low:.0f} Pa): "
                "T_evap_target is not below the condenser side -- "
                "implausible combination."
            ),
        )

    # 6) Desorber temperature needed to produce x_strong at p_high
    try:
        T_gen_min_K = lp.T_sat_solution_from_p_x(p_high, x_strong) + dT_min_des
    except Exception as exc:
        return MinGenResult(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=f"Desorber equilibrium temperature not computable: {exc}",
        )

    T_gen_min_C = kelvin_to_celsius(T_gen_min_K)
    return MinGenResult(
        **partial_common, feasible=True, message="OK (optimistic lower bound).",
        p_high_Pa=p_high, T_gen_min_C=T_gen_min_C,
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_generator_temperature(
    T_gen_values_C: Sequence[float],
    T_recool_C: float,
    **kwargs,
) -> List[MinEvapResult]:
    """Forward: T_evap_min over T_gen, at fixed reject-cooling temperature."""
    return [estimate_min_evap_temperature(t, T_recool_C, **kwargs) for t in T_gen_values_C]


def sweep_recool_temperature(
    T_recool_values_C: Sequence[float],
    T_evap_target_C: float,
    **kwargs,
) -> List[MinGenResult]:
    """Backward: T_gen_min over T_recool, at fixed target evaporator temperature."""
    return [estimate_min_generator_temperature(T_evap_target_C, t, **kwargs) for t in T_recool_values_C]


def print_min_evap_table(results: Sequence[MinEvapResult]) -> None:
    print("=" * 110)
    print(
        f"{'T_gen[C]':>9} {'T_recool[C]':>10} {'x_strong':>9} "
        f"{'T_evap_min[C]':>14} {'Cryst.':>10}  Note"
    )
    print("-" * 110)
    for r in results:
        if not r.crystallization_safe:
            krist = "RISK"
        elif r.crystallization_limited:
            krist = "clamped"
        else:
            krist = "safe"
        print(
            f"{r.T_gen_C:9.2f} {r.T_recool_C:10.2f} {r.x_strong:9.4f} "
            f"{r.T_evap_min_C:14.2f} {krist:>10}  {r.message}"
        )
    print("=" * 110)


def print_min_gen_table(results: Sequence[MinGenResult]) -> None:
    print("=" * 110)
    print(
        f"{'T_evap_target[C]':>15} {'T_recool[C]':>10} {'x_strong':>9} "
        f"{'T_gen_min[C]':>13} {'Cryst.':>10}  Note"
    )
    print("-" * 110)
    for r in results:
        krist = "safe" if r.crystallization_safe else "RISK"
        print(
            f"{r.T_evap_target_C:15.2f} {r.T_recool_C:10.2f} {r.x_strong:9.4f} "
            f"{r.T_gen_min_C:13.2f} {krist:>10}  {r.message}"
        )
    print("=" * 110)


def plot_min_gen_vs_recool_temperature(
    T_recool_values_C: Sequence[float],
    T_evap_target_values_C: Sequence[float],
    *,
    save_path: Optional[str] = "Design_Point/Plots/duehring_screening_AKM_Tgen_vs_Trecool.png",
    show: bool = True,
    **kwargs,
):
    """BACKWARD plot (recommended as the main diagram): T_gen_min vs.
    T_recool, one curve per target evaporator temperature T_evap_target."""
    import matplotlib.pyplot as plt

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_prop_cycle(color=palette)
    all_results = {}
    for T_evap_target_C in T_evap_target_values_C:
        results = sweep_recool_temperature(T_recool_values_C, T_evap_target_C, **kwargs)
        all_results[T_evap_target_C] = results

        x = np.array([r.T_recool_C for r in results])
        y = np.array([r.T_gen_min_C for r in results])
        ok = np.array([r.feasible for r in results])

        (line,) = ax.plot(
            x[ok], y[ok], "o-", linewidth=1.8, markersize=5,
            label=f"T_evap = {T_evap_target_C:.0f} °C",
        )
        if np.any(~ok):
            ax.plot(x[~ok], y[~ok], "x", color=line.get_color(), markersize=7, markeredgewidth=2)

    ax.set_xlabel("Reject-cooling temperature T_recool [°C]")
    ax.set_ylabel("Min. driving temperature T_gen,min [°C]")
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
    DT_MIN_DES = 5.0
    DT_MIN_COND = 5.0
    DT_MIN_ABS = 5.0

    # Backward: the more practically relevant view (see module docstring)
    T_RECOOL_RANGE_C = list(np.arange(15.0, 40.0, 2.5))
    T_EVAP_TARGET_CURVES_C = [3.0, 5.0, 7.0, 10.0]

    print("AC Duehring screening (backward) for T_evap_target = 5 °C:")
    results_bwd = sweep_recool_temperature(
        T_RECOOL_RANGE_C, T_evap_target_C=5.0,
        dT_min_des=DT_MIN_DES, dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )
    print_min_gen_table(results_bwd)

    plot_min_gen_vs_recool_temperature(
        T_RECOOL_RANGE_C, T_EVAP_TARGET_CURVES_C,
        dT_min_des=DT_MIN_DES, dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )