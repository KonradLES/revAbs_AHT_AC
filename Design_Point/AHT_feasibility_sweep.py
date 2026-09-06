"""Fast pinch feasibility map for the AHT -- simulation ONLY, NO optimization.

Difference from AHT_design_point_optimizer.py
---------------------------------------------
The bilevel optimizer there minimizes Sum(UA) over 5 dT_min values per
operating point (DE + Nelder-Mead) -- powerful, but expensive (minutes per
point). For the question "how far can I even shift the external
temperatures and still have the plant run", that's overkill.

This script instead holds dT_min FIXED at your realistically
assumed/built pinch values (not an optimization target!) and, for a grid of
waste-heat temperatures (T13 = T15, parallel routing), finds the ENTIRE
feasible T12 window [T12_min, T12_max] -- not just the maximum. Each point
costs only a handful of solve_aht() calls instead of a full DE search.

Core concept: minimum lift instead of a fixed target temperature
--------------------------------------------------------------------
Instead of a globally fixed useful-heat sink T_11_C, T11 is set PER POINT as
`T_waste_C + min_lift_offset_C` -- a design requirement ("I want at least X
Kelvin of lift above the respective waste heat") that scales along with
T_waste, instead of enforcing a fixed absolute target (which would demand
an unrealistically large lift at low T_waste). The resulting window
[T12_min, T12_max] can be read directly as "what useful temperature can I
reasonably pick at this waste-heat temperature".

Important for interpretation
------------------------------
- GTL (Gross Temperature Lift) = T12 - T_waste is the physically meaningful
  figure of merit of a heat transformer (T12 must be > T_waste, otherwise
  the plant "transforms" nothing). T11 itself has practically NO effect on
  internal feasibility -- it only determines the external absorber mass
  flow m11 = Q_abs/(cp*(T12-T11)), see _resolve_absorber_external_stream in
  Models.AHT_Pinch_Point.
- The solver warm start (x0) has only a narrow basin of attraction (often
  just ~2-4 K in T12, sometimes also in T_waste itself). Too large a jump
  can land the solver at a spurious convergence point (scipy reports
  "success" even though the pinch residuals deviate clearly from 0). That's
  why every search function here uses a small-step continuation walk or
  adaptive homotopy (halve the step on failure, grow it on success) --
  analogous to AHT_stable_design_point.py, just over T12/T_waste instead of
  dT_min.
- Some windows are very narrow (<2 K) just before an operating point hits
  its real feasibility limit (the window width there reproducibly goes to
  0 -- a genuine turning point, not a search-grid artifact). Too coarse a
  search grid can skip such windows and wrongly report "infeasible".

Recommendation
---------------
Use this script first to map the rough trend of the achievable window over
T_waste (moderate pinch/approach values, see the configuration below). Only
for the 3-5 operating points actually of interest picked from that is the
full UA optimizer (AHT_design_point_optimizer.py) worth running.

Standalone usage
-----------------
    python Design_Point/AHT_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from Models.AHT_Pinch_Point import (
    AHTInputs,
    AHTResult,
    PRIMARY_VARIABLE_NAMES,
    initial_guess,
    solve_aht,
)

RESIDUAL_TOL = 1.0e-6

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeasibilitySweepConfig:
    # --- Operating-point boundary conditions -------------------------------
    T_11_C: float = 70.0     # useful-heat sink, cold inlet (only used by
                              # sweep_feasibility())
    T_17_C: float = 20.0     # 15.0  reject cooling, cold inlet
    Qabs_spec_kW: float = 500.0

    # --- Minimum-lift requirement (T11 = T_waste + min_lift_offset_C) ------
    min_lift_offset_C: float = 2.0

    # --- Pinch values (design assumption, not an optimization target) ------
    dT_min_shex: float = 5.0    # 3.0
    dT_min_des: float = 5.0     # 3.0
    dT_min_cond: float = 5.0    # 3.0
    dT_min_evap: float = 5.0    # 3.0
    dT_min_abs: float = 5.0     # 3.0

    # --- External approach values (design assumption) -----------------------
    dT_approach_des_C: float = 4.0      # 4.0
    dT_approach_evap_C: float = 4.0     # 4.0
    dT_approach_cond_C: float = 3.0     # 3.0

    desorber_evaporator_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # -------------------------------------------------------------------
    # Search/solver parameters -- usually NOT to be touched
    # -------------------------------------------------------------------
    T12_search_margin_C: float = 1.0   # starting offset above T_11_C (only for the 1st point)
    T12_step_C: float = 2.0            # expansion step for the window search
    T12_bisect_tol_C: float = 0.2      # bisection stopping width
    max_expand_steps: int = 40         # at T12_step_C=2.0 -> up to 80 K reach
    max_bisect_steps: int = 25

    anchor_search_span_C: float = 60.0
    anchor_search_step_C: float = 1.0

    # --- Relaxed solver tolerances for the probe solves --------------------
    probe_solver_tol: float = 1.0e-6
    probe_max_nfev: int = 300
# ---------------------------------------------------------------------------
# Search grid for the sweep
# ---------------------------------------------------------------------------
T_WASTE_START_C = 85.0   # highest examined waste-heat temperature [°C]
T_WASTE_END_C = 40.0     # lowest DESIRED waste-heat temperature [°C]
T_WASTE_STEP_C = 5.0     # grid spacing [K]

plot_name = "feasibility_sweep_10_PP_5"

ENABLE_DUEHRING_MULTI_PLOT = True
duehring_plot_name = "duehring_multi_process_10_PP_5"

ENABLE_QT_MULTI_PDF = True
qt_pdf_name = "qt_multi_process_10_PP_5"

MULTI_PLOT_EVERY_NTH = 2


@dataclass(frozen=True)
class FeasibilityPoint:
    T_waste_C: float
    T12_min_C: float
    T12_max_C: float
    GTL_min_K: float
    GTL_max_K: float
    feasible: bool
    message: str
    result: Optional[AHTResult] = None


# ---------------------------------------------------------------------------
# Solve helper functions
# ---------------------------------------------------------------------------

def _build_inputs(
    T_waste_C: float, T12_spec_C: float, config: FeasibilitySweepConfig, *,
    fast: bool = True, T11_C: Optional[float] = None,
) -> AHTInputs:
    """T11_C overrides config.T_11_C for a single call -- used by the
    relative-lift functions, where T11 is derived per point from T_waste_C
    instead of being fixed."""
    kwargs = dict(
        T_11_C=T11_C if T11_C is not None else config.T_11_C,
        T_13_C=T_waste_C,
        T_15_C=T_waste_C,
        T_17_C=config.T_17_C,
        dT_min_shex=config.dT_min_shex,
        dT_min_des=config.dT_min_des,
        dT_min_cond=config.dT_min_cond,
        dT_min_evap=config.dT_min_evap,
        dT_min_abs=config.dT_min_abs,
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="Qabs",
        Qabs_spec_kW=config.Qabs_spec_kW,
        absorber_spec_mode="T12",
        T12_spec_C=T12_spec_C,
        desorber_spec_mode="T14",
        T14_spec_C=T_waste_C - config.dT_approach_des_C,
        evaporator_spec_mode="T16",
        T16_spec_C=T_waste_C - config.dT_approach_evap_C,
        condenser_spec_mode="T18",
        T18_spec_C=config.T_17_C + config.dT_approach_cond_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.probe_solver_tol
        kwargs["max_nfev"] = config.probe_max_nfev
    return AHTInputs(**kwargs)


def _is_valid_solution(result: AHTResult) -> bool:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _x0_from_result(result: AHTResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def _try_solve(
    T_waste_C: float, T12_spec_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True, T11_C: Optional[float] = None,
) -> Tuple[bool, Optional[AHTResult]]:
    try:
        inputs = _build_inputs(T_waste_C, T12_spec_C, config, fast=fast, T11_C=T11_C)
    except ValueError:
        return False, None
    try:
        result = solve_aht(inputs, x0=x0)
    except Exception:
        return False, None
    if not _is_valid_solution(result):
        return False, None
    return True, result


def _solve_raw(
    T_waste_C: float, T12_spec_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True, T11_C: Optional[float] = None,
) -> Optional[AHTResult]:
    """Like _try_solve(), but always returns the result (even if not
    "valid") -- only None on a genuine error. Used for warm-start chains,
    where even a non-converged solve's vector beats a generic guess."""
    try:
        inputs = _build_inputs(T_waste_C, T12_spec_C, config, fast=fast, T11_C=T11_C)
    except ValueError:
        return None
    try:
        return solve_aht(inputs, x0=x0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Window determination: [T12_min, T12_max] for a given waste-heat temperature
# ---------------------------------------------------------------------------

def _locate_anchor(
    T_waste_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, guess_C: float,
) -> Tuple[Optional[float], Optional[AHTResult]]:
    """Searches for ONE feasible T12 value via a continuation WALK in small
    steps from guess_C (both directions) -- NOT independent jumps from the
    same starting vector. The basin of attraction is often only ~2-4 K
    wide, so the walk passes on each attempt's solution vector even if not
    (yet) "valid" (see _solve_raw) -- same principle as
    AHT_stable_design_point.py, just over T12 instead of dT_min.
    """
    lo_bound = config.T_11_C + 1.0e-3
    step = config.anchor_search_step_C
    n_steps = max(1, int(round(config.anchor_search_span_C / step)))

    if guess_C > lo_bound:
        result = _solve_raw(T_waste_C, guess_C, x0_seed, config)
        if result is not None and _is_valid_solution(result):
            return guess_C, result

    for direction in (+1, -1):
        x0_walk = x0_seed
        for k in range(1, n_steps + 1):
            candidate = guess_C + direction * k * step
            if candidate <= lo_bound:
                break

            result = _solve_raw(T_waste_C, candidate, x0_walk, config)
            if result is None:
                try:
                    fresh_inputs = _build_inputs(T_waste_C, candidate, config)
                    x0_fresh = initial_guess(fresh_inputs)
                except ValueError:
                    continue
                result = _solve_raw(T_waste_C, candidate, x0_fresh, config)
                if result is None:
                    continue

            x0_walk = _x0_from_result(result)  # pass on the chain, even if not "valid"
            if _is_valid_solution(result):
                return candidate, result

    return None, None


def _bisect_boundary(
    feasible_T: float, x0_feasible: np.ndarray, infeasible_T: float,
    T_waste_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Bisects between a known-feasible and a known-infeasible T12 value
    (order/direction arbitrary) and returns the last feasible value plus
    its associated warm-start vector."""
    lo_feasible, x0_lo = feasible_T, x0_feasible
    hi_infeasible = infeasible_T
    for _ in range(config.max_bisect_steps):
        if abs(hi_infeasible - lo_feasible) < config.T12_bisect_tol_C:
            break
        mid = 0.5 * (lo_feasible + hi_infeasible)
        ok, result = _try_solve(T_waste_C, mid, x0_lo, config)
        if ok:
            lo_feasible = mid
            x0_lo = _x0_from_result(result)
        else:
            hi_infeasible = mid
    return lo_feasible, x0_lo


def _expand_and_bisect(
    anchor_T: float, x0_anchor: np.ndarray, direction: int,
    T_waste_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Expands from anchor_T toward `direction` (+1=max, -1=min) until
    infeasible, then bisects to the boundary. Stops at the hard limit
    T_11_C. Returns (boundary value, associated warm-start vector)."""
    lo_bound = config.T_11_C + 1.0e-3
    feasible_T = anchor_T
    x0_feasible = x0_anchor
    for _ in range(config.max_expand_steps):
        candidate = feasible_T + direction * config.T12_step_C
        if direction < 0 and candidate <= lo_bound:
            ok, result = _try_solve(T_waste_C, lo_bound, x0_feasible, config)
            return (lo_bound, _x0_from_result(result)) if ok else (feasible_T, x0_feasible)
        ok, result = _try_solve(T_waste_C, candidate, x0_feasible, config)
        if ok:
            feasible_T = candidate
            x0_feasible = _x0_from_result(result)
        else:
            return _bisect_boundary(feasible_T, x0_feasible, candidate, T_waste_C, config)
    return feasible_T, x0_feasible  # max_expand_steps reached, see caller's warning


def _refine_boundary(
    T_waste_C: float, T12_C: float, x0_seed: np.ndarray, config: FeasibilitySweepConfig,
) -> Optional[AHTResult]:
    """Final solve with strict tolerances at a boundary found via fast
    probing, for reliable KPIs/UA values. Falls back to the relaxed
    solve on failure."""
    ok, result = _try_solve(T_waste_C, T12_C, x0_seed, config, fast=False)
    if ok:
        return result
    ok, result = _try_solve(T_waste_C, T12_C, x0_seed, config, fast=True)
    return result if ok else None


def find_feasible_window(
    T_waste_C: float,
    config: FeasibilitySweepConfig,
    x0_seed: np.ndarray,
    T12_anchor_guess_C: Optional[float] = None,
) -> FeasibilityPoint:
    """Locates an anchor, then determines the full feasible T12 window
    [T12_min, T12_max] using relaxed tolerances; each boundary found is
    then re-solved strictly once."""

    guess = (
        T12_anchor_guess_C if T12_anchor_guess_C is not None
        else config.T_11_C + config.T12_search_margin_C
    )

    anchor_T, anchor_result = _locate_anchor(T_waste_C, config, x0_seed, guess)
    if anchor_T is None:
        return FeasibilityPoint(
            T_waste_C=T_waste_C, T12_min_C=float("nan"), T12_max_C=float("nan"),
            GTL_min_K=float("nan"), GTL_max_K=float("nan"), feasible=False,
            message=(
                f"No feasible solution found at T_waste={T_waste_C:.2f} °C "
                f"(anchor search around {guess:.2f} °C ± {config.anchor_search_span_C:.0f} K) "
                "-- see AHT_duehring_screening.py for a pre-check."
            ),
        )

    x0_anchor = _x0_from_result(anchor_result)
    T12_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_waste_C, config)
    T12_min, _x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_waste_C, config)

    result_max = _refine_boundary(T_waste_C, T12_max, x0_max, config)

    return FeasibilityPoint(
        T_waste_C=T_waste_C, T12_min_C=T12_min, T12_max_C=T12_max,
        GTL_min_K=T12_min - T_waste_C, GTL_max_K=T12_max - T_waste_C,
        feasible=True, message="OK", result=result_max if result_max is not None else anchor_result,
    )


def _duehring_initial_guess_C(
    T_waste_C: float, config: FeasibilitySweepConfig, *, fraction: float = 0.6
) -> Optional[float]:
    """Returns T_waste_C + fraction * GTL_max(Duehring) as a rough T12
    estimate for the first search point (real pinch window is smaller,
    hence fraction<1)."""
    try:
        from AHT_duehring_screening import estimate_max_gtl
    except ImportError:
        try:
            from Design_Point.AHT_duehring_screening import estimate_max_gtl
        except ImportError:
            return None

    duehring = estimate_max_gtl(
        T13_C=T_waste_C, T15_C=T_waste_C, T17_C=config.T_17_C,
        dT_min_des=config.dT_min_des, dT_min_evap=config.dT_min_evap,
        dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
    )
    if duehring.feasible and duehring.GTL_max_K > 0:
        return T_waste_C + fraction * duehring.GTL_max_K
    return None


# ---------------------------------------------------------------------------
# Sweep A: fixed, absolute sink temperature T_11_C for all T_waste values
# ---------------------------------------------------------------------------

def sweep_feasibility(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    initial_anchor_guess_C: Optional[float] = None,
) -> List[FeasibilityPoint]:
    """Warm-start-chained window search with a sink temperature
    config.T_11_C FIXED for ALL points (e.g. "I have a real application
    that needs exactly 70°C"). At low T_waste this can simply yield "no
    window" because T_11_C is too high, not because the plant can't lift
    at all -- for that question, use sweep_relative_lift_window[_homotopy]()
    instead.

    T_waste_values_C should be monotonic so the warm start carries over
    point to point. initial_anchor_guess_C seeds only the first point
    (e.g. from _duehring_initial_guess_C()); without it,
    config.T_11_C + T12_search_margin_C is used, which can be far off at
    low T_waste_C.
    """
    points: List[FeasibilityPoint] = []
    x0_carry: Optional[np.ndarray] = None
    anchor_guess_carry: Optional[float] = initial_anchor_guess_C

    for i, T_waste_C in enumerate(T_waste_values_C):
        if x0_carry is None:
            seed_guess_C = (
                anchor_guess_carry if anchor_guess_carry is not None
                else config.T_11_C + config.T12_search_margin_C
            )
            seed_inputs = _build_inputs(T_waste_C, seed_guess_C, config)
            x0_carry = initial_guess(seed_inputs)

        point = find_feasible_window(
            T_waste_C, config, x0_carry, T12_anchor_guess_C=anchor_guess_carry
        )
        points.append(point)

        status = "OK" if point.feasible else "FAILED"
        print(
            f"[{i+1}/{len(T_waste_values_C)}] T_waste={T_waste_C:6.2f} °C -> "
            f"T12 in [{point.T12_min_C:6.2f}, {point.T12_max_C:6.2f}] °C, "
            f"GTL in [{point.GTL_min_K:6.2f}, {point.GTL_max_K:6.2f}] K "
            f"[{status}] {point.message if not point.feasible else ''}"
        )

        if point.feasible and point.result is not None:
            x0_carry = _x0_from_result(point.result)
            anchor_guess_carry = point.T12_max_C

    return points


# ---------------------------------------------------------------------------
# Sweep B: co-scaling minimum useful temperature T11 = T_waste + lift
# ---------------------------------------------------------------------------

def sweep_relative_lift_window(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    min_lift_offset_C: Optional[float] = None,
) -> List[FeasibilityPoint]:
    """Like sweep_feasibility(), but T_11_C = T_waste_C + min_lift_offset_C
    per point instead of being globally fixed (see module docstring "Core
    concept"). Makes one jump per point (continuation walk only in T12);
    for larger T_waste jumps (e.g. 10 K+) use
    sweep_relative_lift_window_homotopy() instead.
    """
    offset = min_lift_offset_C if min_lift_offset_C is not None else config.min_lift_offset_C

    points: List[FeasibilityPoint] = []
    x0_carry: Optional[np.ndarray] = None
    anchor_guess_carry: Optional[float] = None

    for i, T_waste_C in enumerate(T_waste_values_C):
        T11_this = T_waste_C + offset
        config_point = replace(config, T_11_C=T11_this)

        if x0_carry is None:
            seed_guess_C = (
                anchor_guess_carry if anchor_guess_carry is not None
                else _duehring_initial_guess_C(T_waste_C, config)
                or T11_this + config.T12_search_margin_C
            )
            seed_inputs = _build_inputs(T_waste_C, seed_guess_C, config_point)
            x0_carry = initial_guess(seed_inputs)

        point = find_feasible_window(
            T_waste_C, config_point, x0_carry, T12_anchor_guess_C=anchor_guess_carry
        )
        points.append(point)

        status = "OK" if point.feasible else "FAILED"
        print(
            f"[{i+1}/{len(T_waste_values_C)}] T_waste={T_waste_C:6.2f} °C "
            f"(T11={T11_this:6.2f} °C) -> "
            f"T12 in [{point.T12_min_C:6.2f}, {point.T12_max_C:6.2f}] °C, "
            f"GTL in [{point.GTL_min_K:6.2f}, {point.GTL_max_K:6.2f}] K "
            f"[{status}] {point.message if not point.feasible else ''}"
        )

        if point.feasible and point.result is not None:
            x0_carry = _x0_from_result(point.result)
            anchor_guess_carry = point.T12_max_C

    return points


# ---------------------------------------------------------------------------
# Sweep C: like B, but with adaptive homotopy BETWEEN the grid points
# ---------------------------------------------------------------------------
#
# (T_waste, T12) move together in small adaptive steps (holding
# GTL = T12 - T_waste ~constant), step halved on failure / grown on
# success -- needed because even ~10 K T_waste jumps can break the
# warm-start chain. Recommended, most robust variant -- see __main__.

def _window_at_point(
    T_waste_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, anchor_guess_C: float,
) -> Optional[Tuple[FeasibilityPoint, np.ndarray]]:
    """Like find_feasible_window(), but additionally returns the warm-start
    vector at T12_min (for continuing the homotopy to the next T_waste
    target). None if the anchor search fails."""
    anchor_T, anchor_result = _locate_anchor(T_waste_C, config, x0_seed, anchor_guess_C)
    if anchor_T is None:
        return None

    x0_anchor = _x0_from_result(anchor_result)
    T12_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_waste_C, config)
    T12_min, x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_waste_C, config)
    result_max = _refine_boundary(T_waste_C, T12_max, x0_max, config)

    point = FeasibilityPoint(
        T_waste_C=T_waste_C, T12_min_C=T12_min, T12_max_C=T12_max,
        GTL_min_K=T12_min - T_waste_C, GTL_max_K=T12_max - T_waste_C,
        feasible=True, message="OK", result=result_max if result_max is not None else anchor_result,
    )
    return point, x0_min


def _homotopy_walk_T_waste(
    T_waste_from_C: float, T12_from_C: float, x0_from: np.ndarray, T_waste_to_C: float,
    config: FeasibilitySweepConfig, min_lift_offset_C: float,
    *, step_initial_C: float = 3.0, step_min_C: float = 0.25, max_steps: int = 200,
) -> Tuple[float, float, np.ndarray, bool]:
    """Moves (T_waste, T12) together toward T_waste_to_C, holding
    GTL = T12 - T_waste constant (at least min_lift_offset_C). Step halved
    on failure (below step_min_C returns the furthest point reached),
    grown on success.

    Returns: (T_waste_reached, T12_reached, x0_reached, target_fully_reached)
    """
    GTL_hold = max(T12_from_C - T_waste_from_C, min_lift_offset_C)
    direction = 1.0 if T_waste_to_C > T_waste_from_C else -1.0

    T_waste_cur, T12_cur, x0_cur = T_waste_from_C, T12_from_C, x0_from
    step = step_initial_C

    for _ in range(max_steps):
        remaining = direction * (T_waste_to_C - T_waste_cur)
        if remaining <= 1.0e-9:
            return T_waste_cur, T12_cur, x0_cur, True

        step_trial = min(step, remaining)
        T_waste_trial = T_waste_cur + direction * step_trial
        T12_trial = T_waste_trial + GTL_hold
        T11_trial = T_waste_trial + min_lift_offset_C
        config_trial = replace(config, T_11_C=T11_trial)

        result = _solve_raw(T_waste_trial, T12_trial, x0_cur, config_trial)
        if result is not None and _is_valid_solution(result):
            T_waste_cur, T12_cur = T_waste_trial, T12_trial
            x0_cur = _x0_from_result(result)
            step = min(step_trial * 1.5, step_initial_C)
        else:
            step = step_trial / 2.0
            if step < step_min_C:
                return T_waste_cur, T12_cur, x0_cur, False

    return T_waste_cur, T12_cur, x0_cur, False


def sweep_relative_lift_window_homotopy(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    min_lift_offset_C: Optional[float] = None,
    homotopy_step_initial_C: float = 3.0,
    homotopy_step_min_C: float = 0.25,
) -> List[FeasibilityPoint]:
    """Like sweep_relative_lift_window(), but with adaptive homotopy
    BETWEEN grid points (see section comment) instead of a single jump --
    the recommended, most robust variant. Give T_waste_values_C in walk
    order (e.g. descending from a high, unproblematic starting value).
    """
    offset = min_lift_offset_C if min_lift_offset_C is not None else config.min_lift_offset_C

    points: List[FeasibilityPoint] = []
    T_waste_cur: Optional[float] = None
    T12_cur: Optional[float] = None
    x0_cur: Optional[np.ndarray] = None

    for i, T_waste_target in enumerate(T_waste_values_C):
        if x0_cur is None:
            T11_first = T_waste_target + offset
            config_first = replace(config, T_11_C=T11_first)
            guess_C = (
                _duehring_initial_guess_C(T_waste_target, config)
                or T11_first + config.T12_search_margin_C
            )
            seed_inputs = _build_inputs(T_waste_target, guess_C, config_first)
            x0_seed = initial_guess(seed_inputs)
            outcome = _window_at_point(T_waste_target, config_first, x0_seed, guess_C)
            reported_T_waste = T_waste_target
        else:
            reached_T_waste, reached_T12, reached_x0, fully_reached = _homotopy_walk_T_waste(
                T_waste_cur, T12_cur, x0_cur, T_waste_target, config, offset,
                step_initial_C=homotopy_step_initial_C, step_min_C=homotopy_step_min_C,
            )
            if not fully_reached:
                print(
                    f"  [Homotopy] Target {T_waste_target:.2f} °C not fully reached, "
                    f"stopped at {reached_T_waste:.2f} °C "
                    f"(step size fell below {homotopy_step_min_C:.2f} K)."
                )
            T11_target = reached_T_waste + offset
            config_target = replace(config, T_11_C=T11_target)
            # Anchor right on T11_target can miss narrow windows -- push
            # it slightly above.
            anchor_guess_C = max(reached_T12, T11_target + 1.0)
            outcome = _window_at_point(reached_T_waste, config_target, reached_x0, anchor_guess_C)
            reported_T_waste = reached_T_waste

        if outcome is None:
            points.append(FeasibilityPoint(
                T_waste_C=reported_T_waste, T12_min_C=float("nan"), T12_max_C=float("nan"),
                GTL_min_K=float("nan"), GTL_max_K=float("nan"), feasible=False,
                message=f"Even the anchor search at {reported_T_waste:.2f} °C found no solution.",
            ))
            print(f"[{i+1}/{len(T_waste_values_C)}] T_waste={reported_T_waste:6.2f} °C -> FAILED")
            continue

        point, x0_min = outcome
        points.append(point)
        print(
            f"[{i+1}/{len(T_waste_values_C)}] T_waste={reported_T_waste:6.2f} °C -> "
            f"T12 in [{point.T12_min_C:6.2f}, {point.T12_max_C:6.2f}] °C, "
            f"GTL in [{point.GTL_min_K:6.2f}, {point.GTL_max_K:6.2f}] K [OK]"
        )

        T_waste_cur = reported_T_waste
        T12_cur = point.T12_min_C
        x0_cur = x0_min

    return points


# ---------------------------------------------------------------------------
# Output: table + plot
# ---------------------------------------------------------------------------

def print_sweep_table(points: Sequence[FeasibilityPoint]) -> None:
    print("=" * 100)
    print(
        f"{'T_waste[C]':>10} {'T12_min[C]':>11} {'T12_max[C]':>11} "
        f"{'GTL_min[K]':>11} {'GTL_max[K]':>11} {'Status':>8}"
    )
    print("-" * 100)
    for p in points:
        status = "OK" if p.feasible else "FAIL"
        print(
            f"{p.T_waste_C:10.2f} {p.T12_min_C:11.2f} {p.T12_max_C:11.2f} "
            f"{p.GTL_min_K:11.2f} {p.GTL_max_K:11.2f} {status:>8}"
        )
    print("=" * 100)


def plot_feasibility_sweep(
    points: Sequence[FeasibilityPoint],
    *,
    duehring_reference: Optional[Sequence] = None,
    save_path: Optional[str] = f"Design_Point/Plots/{plot_name}.png",
    show: bool = True,
):
    """Plots the feasible GTL window vs. waste-heat temperature; optionally
    compares against the optimistic Duehring upper bound (list of
    DuehringScreeningResult from
    AHT_duehring_screening.sweep_waste_heat_temperature())."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ok_points = [p for p in points if p.feasible]
    x = np.array([p.T_waste_C for p in ok_points])
    y_min = np.array([p.GTL_min_K for p in ok_points])
    y_max = np.array([p.GTL_max_K for p in ok_points])

    ax.fill_between(x, y_min, y_max, color="tab:blue", alpha=0.18, label="feasible window")
    ax.plot(x, y_max, "o-", color="tab:blue", label="GTL_max")
    ax.plot(x, y_min, "o--", color="tab:blue", linewidth=1.2, label="GTL_min")

    fail_points = [p for p in points if not p.feasible]
    if fail_points:
        xf = np.array([p.T_waste_C for p in fail_points])
        ax.plot(
            xf, np.zeros_like(xf), "x", color="tab:red", markersize=8,
            markeredgewidth=2, label="not solvable",
        )

    if duehring_reference is not None:
        xr = np.array([r.T13_C for r in duehring_reference])
        yr = np.array([r.GTL_max_K for r in duehring_reference])
        okr = np.array([r.feasible for r in duehring_reference])
        ax.plot(xr[okr], yr[okr], "--", color="0.4", label="Duehring bound (optimistic)")

    ax.set_xlabel("Waste-heat temperature T13 = T15 [°C]")
    ax.set_ylabel("GTL [K]")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    config = FeasibilitySweepConfig()

    # All physical assumptions (pinch values, approach values, minimum-lift
    # requirement min_lift_offset_C, T_17_C) live centrally in
    # FeasibilitySweepConfig above -- adjust there, not here.
    T_WASTE_RANGE_C = list(
        np.arange(T_WASTE_START_C, T_WASTE_END_C - 0.5 * T_WASTE_STEP_C, -T_WASTE_STEP_C)
    )

    print(f"Achievable useful-temperature window, T11 = T_waste + {config.min_lift_offset_C:.0f} K")
    print(f"(T_17_C = {config.T_17_C:.1f} °C constant)")
    points = sweep_relative_lift_window_homotopy(T_WASTE_RANGE_C, config)
    print_sweep_table(points)

    duehring_reference = None
    try:
        from AHT_duehring_screening import sweep_waste_heat_temperature

        duehring_reference = sweep_waste_heat_temperature(
            T_WASTE_RANGE_C, T17_C=config.T_17_C,
            dT_min_des=config.dT_min_des, dT_min_evap=config.dT_min_evap,
            dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
        )
    except ImportError:
        pass

    plot_feasibility_sweep(points, duehring_reference=duehring_reference)

    # Additional evaluations: reuse the `points` already computed above (no
    # re-running the sweep). Lazy import so the two scripts themselves stay
    # independently importable/runnable.
    if ENABLE_DUEHRING_MULTI_PLOT:
        from Design_Point.Visualization_Scripts.AHT_duehring_multi_process_plot import (
            select_and_plot_duehring,
        )
        select_and_plot_duehring(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{duehring_plot_name}.png",
        )

    if ENABLE_QT_MULTI_PDF:
        from Design_Point.Visualization_Scripts.AHT_qt_multi_process_plot import (
            select_and_plot_qt_pdf,
        )
        select_and_plot_qt_pdf(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{qt_pdf_name}.pdf",
        )
