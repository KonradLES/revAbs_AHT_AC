"""Fast pinch feasibility map for the AC -- simulation ONLY, NO optimization.

Analog of AHT_feasibility_sweep.py, but for the absorption chiller.

Calibration status (IMPORTANT, read before your first run)
------------------------------------------------------------------
Two real root causes were found and fixed:

1) Models.AC_Pinch_Point.initial_guess() used to start at x4=0.22,
   x1=0.243 -- backwards relative to the required x4 > x1 concentration
   hierarchy, which reliably stuck the solver on the wrong branch. FIXED
   in Models/AC_Pinch_Point.py itself; this script calls initial_guess()
   directly with no extra correction. Most points now converge to
   residual ~1e-10 with no further effort.
2) The evaporator has a hard model limit T10 >= 1 °C (no ice modeled). At
   T18_spec_C=5.0 and dT_min_evap=5.0 the internal evaporation temperature
   is arithmetically forced to ~0 °C -- structurally unsolvable. Keep
   dT_min_evap noticeably below (T18_spec_C - 1 °C).
Also: dT_approach_evap_C sets T17 = T18_spec_C + dT_approach_evap_C, and
initial_guess() estimates T10 ~= T17 - 8 K from that -- below T17 ~9 °C
this guess alone already violates the 1 °C limit. Don't pick
dT_approach_evap_C too small at low T18_spec_C (>= 5.0 recommended).

Difference from AC_design_point_optimizer.py
--------------------------------------------
The bilevel optimizer there minimizes Sum(UA) over 5 dT_min values per
point (DE + Nelder-Mead) -- powerful but expensive (minutes per point).
For "how hot must my driving water be at minimum, at a given
reject-cooling temperature", that's overkill.

This script instead holds dT_min FIXED at realistic pinch values and, for
a grid of reject-cooling temperatures (T13 = T15, parallel routing),
finds the ENTIRE feasible T11 window [T11_min, T11_max] -- not just the
minimum -- at a cost of a handful of solve_ac() calls per point.

Role swap vs. the AHT (important for understanding)
--------------------------------------------------------------
Pressure levels are EXACTLY REVERSED vs. the AC: desorber and condenser
on the HIGH-pressure side, absorber and evaporator on the LOW-pressure
side -- which also swaps which components share an external temperature
("pair") vs. are independently specified ("single"):

                    AHT (heat transformer)     AC (chiller)
    Pair (shared external temperature):
        Desorber + evaporator <-> T_waste       Absorber + condenser <-> T_reject
    Single (independently given):
        Absorber (product)  : T11 -> T12         Desorber (drive): T11 -> T12
        Condenser (reject)  : T17 -> T18          Evaporator (product): T17 -> T18

Hence this script sweeps the reject-cooling temperature T_reject (=T13=T15,
the AC's analog of T_waste), holds T18 (useful cooling) fixed, and
searches for the generator inlet window T11 (desorber drive) -- the
analog of the AC's T12 window, except T11 itself is the free search
variable here (T12 follows via a fixed approach, dT_approach_des_C; see
_build_inputs).

Important for interpretation
------------------------------
- Unlike the AC (T11 barely affects internal feasibility), T11 here
  DIRECTLY sets the desorber equilibrium concentration x4, and hence how
  close the plant runs to its crystallization limit (state "6", just
  before the absorber). Too LOW a T11 makes the plant pinch-infeasible
  (T11_min, the value of interest here); too HIGH a T11 can trigger
  crystallization at the SHEX outlet (T11_max) -- so, unlike the AC, the
  window [T11_min, T11_max] can close from ABOVE at low reject-cooling
  temperatures.
- The solver warm start has only a narrow basin of attraction (often just
  ~2-4 K), so every search here uses a small-step continuation walk or
  adaptive homotopy, exactly as in AHT_feasibility_sweep.py (over
  T11/T_reject instead of T12/T_waste).
- Windows can be very narrow (<2 K) just before a real feasibility limit
  -- too coarse a search grid can skip them and wrongly report
  "infeasible".

Recommendation
---------------
Use this script first to map the rough trend of required generator inlet
temperature over T_reject. Only run the full UA optimizer
(AC_design_point_optimizer.py) for the 3-5 points actually of interest.

Standalone usage
-----------------
    python Design_Point/AC_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from Models.AC_Pinch_Point import (
    ACInputs,
    ACResult,
    PRIMARY_VARIABLE_NAMES,
    initial_guess,
    solve_ac,
)

RESIDUAL_TOL = 1.0e-6

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeasibilitySweepConfig:
    # --- Operating-point boundary conditions -------------------------------
    T18_spec_C: float = 5.0     # 5.0 fixed evaporator outlet temperature (useful cooling)
    Qevap_spec_kW: float = 40.9

    # --- Pinch values (design assumption, not an optimization target) ------
    dT_min_shex: float = 5.0
    dT_min_des: float = 5.0
    dT_min_cond: float = 5.0
    dT_min_evap: float = 3.0    # T18 - dT_min_evap >= 0 °C, otherwise a cold-start failure
    dT_min_abs: float = 5.0

    # --- External approach values (design assumption) -----------------------
    dT_approach_des_C: float = 4.0 # 18.0
    dT_approach_abs_C: float = 3.0  # 7.0
    dT_approach_cond_C: float = 3.0 # 7.0
    dT_approach_evap_C: float = 4.0 # 6.0 if too small, T10 can go below 0 °C (hard evaporator model limit)

    absorber_condenser_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # -------------------------------------------------------------------
    # Search/solver parameters
    # -------------------------------------------------------------------
    T11_search_margin_C: float = 40.0   # starting offset above T_reject (only for the 1st point, if no Duehring estimate)
    T11_step_C: float = 2.0            # expansion step for the window search
    T11_bisect_tol_C: float = 0.2      # bisection stopping width
    max_expand_steps: int = 40         # at T11_step_C=2.0 -> up to 80 K reach
    max_bisect_steps: int = 25

    anchor_search_span_C: float = 60.0
    anchor_search_step_C: float = 1.0

    probe_solver_tol: float = 1.0e-6
    probe_max_nfev: int = 300

# ---------------------------------------------------------------------------
# Search grid for the sweep
# ---------------------------------------------------------------------------
T_REJECT_START_C = 15.0   # numerically unproblematic starting value [°C]
T_REJECT_END_C = 35.0     # highest DESIRED reject-cooling temperature [°C]
T_REJECT_STEP_C = 2.5     # grid spacing [K]

plot_name = "AC_feasibility_sweep_ex_8_5_5_5"

ENABLE_DUEHRING_MULTI_PLOT = True
duehring_plot_name = "AC_duehring_multi_process_ex_8_5_5_5"

ENABLE_QT_MULTI_PDF = True
qt_pdf_name = "AC_qt_multi_process_ex_8_5_5_5"

MULTI_PLOT_EVERY_NTH = 2

@dataclass(frozen=True)
class FeasibilityPoint:
    T_reject_C: float
    T11_min_C: float
    T11_max_C: float
    dT_drive_min_K: float
    dT_drive_max_K: float
    feasible: bool
    message: str
    result: Optional[ACResult] = None


# ---------------------------------------------------------------------------
# Solve helper functions
# ---------------------------------------------------------------------------

def _build_inputs(
    T_reject_C: float, T11_C: float, config: FeasibilitySweepConfig, *, fast: bool = True,
) -> ACInputs:
    kwargs = dict(
        T_11_C=T11_C,
        T_13_C=T_reject_C,
        T_15_C=T_reject_C,
        T_17_C=config.T18_spec_C + config.dT_approach_evap_C,
        dT_min_shex=config.dT_min_shex,
        dT_min_des=config.dT_min_des,
        dT_min_cond=config.dT_min_cond,
        dT_min_evap=config.dT_min_evap,
        dT_min_abs=config.dT_min_abs,
        absorber_condenser_routing_mode=config.absorber_condenser_routing_mode,
        cycle_scale_spec_mode="Qeva",
        Qevap_spec_kW=config.Qevap_spec_kW,
        desorber_spec_mode="T12",
        T12_spec_C=T11_C - config.dT_approach_des_C,
        absorber_spec_mode="T14",
        T14_spec_C=T_reject_C + config.dT_approach_abs_C,
        condenser_spec_mode="T16",
        T16_spec_C=T_reject_C + config.dT_approach_cond_C,
        evaporator_spec_mode="T18",
        T18_spec_C=config.T18_spec_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.probe_solver_tol
        kwargs["max_nfev"] = config.probe_max_nfev
    return ACInputs(**kwargs)


def _is_valid_solution(result: ACResult) -> bool:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _x0_from_result(result: ACResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def _try_solve(
    T_reject_C: float, T11_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True,
) -> Tuple[bool, Optional[ACResult]]:
    try:
        inputs = _build_inputs(T_reject_C, T11_C, config, fast=fast)
    except ValueError:
        return False, None
    try:
        result = solve_ac(inputs, x0=x0)
    except Exception:
        return False, None
    if not _is_valid_solution(result):
        return False, None
    return True, result


def _solve_raw(
    T_reject_C: float, T11_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True,
) -> Optional[ACResult]:
    """Like _try_solve(), but always returns the result (even if not
    "valid") -- only None on a genuine error. Used for warm-start chains,
    where even a non-converged solve's vector beats a generic guess."""
    try:
        inputs = _build_inputs(T_reject_C, T11_C, config, fast=fast)
    except ValueError:
        return None
    try:
        return solve_ac(inputs, x0=x0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Window determination: [T11_min, T11_max] for a given reject-cooling temperature
# ---------------------------------------------------------------------------

def _locate_anchor(
    T_reject_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, guess_C: float,
) -> Tuple[Optional[float], Optional[ACResult]]:
    """Searches for ONE feasible T11 value, as a continuation WALK in small
    steps from guess_C (both directions) -- NOT as independent jumps with
    the same starting vector. See AC_feasibility_sweep.py for the detailed
    rationale of the continuation principle."""
    lo_bound = T_reject_C + 1.0e-3  # T11 must be above the reject-cooling temperature (pressure drop)
    step = config.anchor_search_step_C
    n_steps = max(1, int(round(config.anchor_search_span_C / step)))

    if guess_C > lo_bound:
        result = _solve_raw(T_reject_C, guess_C, x0_seed, config)
        if result is not None and _is_valid_solution(result):
            return guess_C, result

    for direction in (+1, -1):
        x0_walk = x0_seed
        for k in range(1, n_steps + 1):
            candidate = guess_C + direction * k * step
            if candidate <= lo_bound:
                break

            result = _solve_raw(T_reject_C, candidate, x0_walk, config)
            if result is None or not _is_valid_solution(result):
                # Also try a fresh cold start here -- the chained warm
                # start can get stuck at an unfavorable point that
                # initial_guess() escapes more easily.
                try:
                    fresh_inputs = _build_inputs(T_reject_C, candidate, config)
                    x0_fresh = initial_guess(fresh_inputs)
                except ValueError:
                    x0_fresh = None
                if x0_fresh is not None:
                    fresh_result = _solve_raw(T_reject_C, candidate, x0_fresh, config)
                    if fresh_result is not None and (
                        result is None or _is_valid_solution(fresh_result)
                    ):
                        result = fresh_result

            if result is None:
                continue

            x0_walk = _x0_from_result(result)  # keep chaining even if not "valid"
            if _is_valid_solution(result):
                return candidate, result

    return None, None


def _bisect_boundary(
    feasible_T: float, x0_feasible: np.ndarray, infeasible_T: float,
    T_reject_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Bisects between a known-feasible and a known-infeasible T11 value
    (order/direction doesn't matter) and returns the last feasible value +
    its associated warm-start vector."""
    lo_feasible, x0_lo = feasible_T, x0_feasible
    hi_infeasible = infeasible_T
    for _ in range(config.max_bisect_steps):
        if abs(hi_infeasible - lo_feasible) < config.T11_bisect_tol_C:
            break
        mid = 0.5 * (lo_feasible + hi_infeasible)
        ok, result = _try_solve(T_reject_C, mid, x0_lo, config)
        if ok:
            lo_feasible = mid
            x0_lo = _x0_from_result(result)
        else:
            hi_infeasible = mid
    return lo_feasible, x0_lo


def _expand_and_bisect(
    anchor_T: float, x0_anchor: np.ndarray, direction: int,
    T_reject_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Expands from anchor_T toward `direction` (+1=max, -1=min) until
    infeasible, then bisects to the boundary. Stops at the hard edge
    T_reject_C. Returns (boundary value, associated warm-start vector)."""
    lo_bound = T_reject_C + 1.0e-3
    feasible_T = anchor_T
    x0_feasible = x0_anchor
    for _ in range(config.max_expand_steps):
        candidate = feasible_T + direction * config.T11_step_C
        if direction < 0 and candidate <= lo_bound:
            ok, result = _try_solve(T_reject_C, lo_bound, x0_feasible, config)
            return (lo_bound, _x0_from_result(result)) if ok else (feasible_T, x0_feasible)
        ok, result = _try_solve(T_reject_C, candidate, x0_feasible, config)
        if ok:
            feasible_T = candidate
            x0_feasible = _x0_from_result(result)
        else:
            return _bisect_boundary(feasible_T, x0_feasible, candidate, T_reject_C, config)
    return feasible_T, x0_feasible  # max_expand_steps reached, see the caller's warning


def _refine_boundary(
    T_reject_C: float, T11_C: float, x0_seed: np.ndarray, config: FeasibilitySweepConfig,
) -> Optional[ACResult]:
    """Final solve with strict tolerances at a boundary found via fast
    probing, for reliable KPIs/UA values. Falls back to the relaxed
    solve on failure."""
    ok, result = _try_solve(T_reject_C, T11_C, x0_seed, config, fast=False)
    if ok:
        return result
    ok, result = _try_solve(T_reject_C, T11_C, x0_seed, config, fast=True)
    return result if ok else None


def find_feasible_window(
    T_reject_C: float,
    config: FeasibilitySweepConfig,
    x0_seed: np.ndarray,
    T11_anchor_guess_C: Optional[float] = None,
) -> FeasibilityPoint:
    """Locates an anchor, then determines the full feasible T11 window
    [T11_min, T11_max] using relaxed tolerances; the minimum boundary is
    then re-solved strictly."""

    guess = (
        T11_anchor_guess_C if T11_anchor_guess_C is not None
        else T_reject_C + config.T11_search_margin_C
    )

    anchor_T, anchor_result = _locate_anchor(T_reject_C, config, x0_seed, guess)
    if anchor_T is None:
        return FeasibilityPoint(
            T_reject_C=T_reject_C, T11_min_C=float("nan"), T11_max_C=float("nan"),
            dT_drive_min_K=float("nan"), dT_drive_max_K=float("nan"), feasible=False,
            message=(
                f"No feasible solution found at T_reject={T_reject_C:.2f} °C "
                f"(anchor search around {guess:.2f} °C +/- {config.anchor_search_span_C:.0f} K) "
                "-- see AC_duehring_screening.py for a pre-check."
            ),
        )

    x0_anchor = _x0_from_result(anchor_result)
    T11_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_reject_C, config)
    T11_min, x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_reject_C, config)

    result_min = _refine_boundary(T_reject_C, T11_min, x0_min, config)

    return FeasibilityPoint(
        T_reject_C=T_reject_C, T11_min_C=T11_min, T11_max_C=T11_max,
        dT_drive_min_K=T11_min - T_reject_C, dT_drive_max_K=T11_max - T_reject_C,
        feasible=True, message="OK", result=result_min if result_min is not None else anchor_result,
    )


def _duehring_initial_guess_C(
    T_reject_C: float, config: FeasibilitySweepConfig, *, margin_C: float = 25.0
) -> Optional[float]:
    """Returns T_gen_min(Duehring) + margin_C as a rough T11 estimate for
    the first search point. margin_C=25 is not arbitrary: at
    T_reject=25°C/T18=5°C, the real pinch-model minimum was empirically
    about 27 K above the optimistic Duehring estimate (63°C vs. ~90°C)."""
    try:
        from AC_duehring_screening import estimate_min_generator_temperature
    except ImportError:
        try:
            from Design_Point.AC_duehring_screening import estimate_min_generator_temperature
        except ImportError:
            return None

    T_evap_target_C = config.T18_spec_C - config.dT_min_evap
    try:
        duehring = estimate_min_generator_temperature(
            T_evap_target_C=T_evap_target_C, T_recool_C=T_reject_C,
            dT_min_des=config.dT_min_des, dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
        )
    except Exception:
        # Patek correlations can raise at edge values -- skip the estimate,
        # caller falls back to T_reject_C+margin.
        return None
    if duehring.feasible:
        return duehring.T_gen_min_C + margin_C
    return None


# ---------------------------------------------------------------------------
# Sweep: adaptive homotopy over T_reject (recommended, most robust variant)
# ---------------------------------------------------------------------------
#
# (T_reject, T11) move together in small adaptive steps (holding
# dT_drive = T11 - T_reject ~constant), step halved on failure / grown on
# success. See AC_feasibility_sweep.py for the rationale.

def _window_at_point(
    T_reject_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, anchor_guess_C: float,
) -> Optional[Tuple[FeasibilityPoint, np.ndarray]]:
    """Like find_feasible_window(), but also returns the warm-start vector at
    T11_min (to continue the homotopy toward the next T_reject target). None
    if the anchor search fails."""
    anchor_T, anchor_result = _locate_anchor(T_reject_C, config, x0_seed, anchor_guess_C)
    if anchor_T is None:
        return None

    x0_anchor = _x0_from_result(anchor_result)
    T11_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_reject_C, config)
    T11_min, x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_reject_C, config)
    result_min = _refine_boundary(T_reject_C, T11_min, x0_min, config)

    point = FeasibilityPoint(
        T_reject_C=T_reject_C, T11_min_C=T11_min, T11_max_C=T11_max,
        dT_drive_min_K=T11_min - T_reject_C, dT_drive_max_K=T11_max - T_reject_C,
        feasible=True, message="OK", result=result_min if result_min is not None else anchor_result,
    )
    return point, x0_min


def _homotopy_walk_T_reject(
    T_reject_from_C: float, T11_from_C: float, x0_from: np.ndarray, T_reject_to_C: float,
    config: FeasibilitySweepConfig,
    *, step_initial_C: float = 3.0, step_min_C: float = 0.25, max_steps: int = 200,
) -> Tuple[float, float, np.ndarray, bool]:
    """Moves (T_reject, T11) together toward T_reject_to_C, holding
    dT_drive = T11 - T_reject constant. Step halved on failure (below
    step_min_C returns the furthest point reached), grown on success.

    Returns: (T_reject_reached, T11_reached, x0_reached, target_fully_reached)
    """
    dT_drive_hold = T11_from_C - T_reject_from_C
    direction = 1.0 if T_reject_to_C > T_reject_from_C else -1.0

    T_reject_cur, T11_cur, x0_cur = T_reject_from_C, T11_from_C, x0_from
    step = step_initial_C

    for _ in range(max_steps):
        remaining = direction * (T_reject_to_C - T_reject_cur)
        if remaining <= 1.0e-9:
            return T_reject_cur, T11_cur, x0_cur, True

        step_trial = min(step, remaining)
        T_reject_trial = T_reject_cur + direction * step_trial
        T11_trial = T_reject_trial + dT_drive_hold

        result = _solve_raw(T_reject_trial, T11_trial, x0_cur, config)
        if result is not None and _is_valid_solution(result):
            T_reject_cur, T11_cur = T_reject_trial, T11_trial
            x0_cur = _x0_from_result(result)
            step = min(step_trial * 1.5, step_initial_C)
        else:
            step = step_trial / 2.0
            if step < step_min_C:
                return T_reject_cur, T11_cur, x0_cur, False

    return T_reject_cur, T11_cur, x0_cur, False


def sweep_min_generator_temperature_homotopy(
    T_reject_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    homotopy_step_initial_C: float = 3.0,
    homotopy_step_min_C: float = 0.25,
) -> List[FeasibilityPoint]:
    """Determines the feasible T11 window for each reject-cooling temperature
    in T_reject_values_C, using adaptive homotopy BETWEEN grid points (see
    the section docstring) -- the recommended, most robust variant. Give
    T_reject_values_C in traversal order (e.g. descending from a warm,
    unproblematic starting value, see the module docstring)."""

    points: List[FeasibilityPoint] = []
    T_reject_cur: Optional[float] = None
    T11_cur: Optional[float] = None
    x0_cur: Optional[np.ndarray] = None

    for i, T_reject_target in enumerate(T_reject_values_C):
        outcome = None
        reported_T_reject = T_reject_target

        if x0_cur is not None:
            reached_T_reject, reached_T11, reached_x0, fully_reached = _homotopy_walk_T_reject(
                T_reject_cur, T11_cur, x0_cur, T_reject_target, config,
                step_initial_C=homotopy_step_initial_C, step_min_C=homotopy_step_min_C,
            )
            if fully_reached:
                # Anchor right on the boundary can miss narrow windows --
                # push it slightly above.
                anchor_guess_C = max(reached_T11, reached_T_reject + 1.0)
                outcome = _window_at_point(reached_T_reject, config, reached_x0, anchor_guess_C)
                reported_T_reject = reached_T_reject
            else:
                # A stuck continuation walk isn't necessarily real
                # infeasibility at the target (see module docstring) --
                # fall back to a fresh, Duehring-guided anchor instead.
                print(
                    f"  [homotopy] target {T_reject_target:.2f} °C not reached via continuation "
                    f"walk (stopped at {reached_T_reject:.2f} °C) -- trying a fresh anchor "
                    "directly at the target."
                )

        if outcome is None:
            guess_C = (
                _duehring_initial_guess_C(T_reject_target, config)
                or T_reject_target + config.T11_search_margin_C
            )
            seed_inputs = _build_inputs(T_reject_target, guess_C, config)
            x0_seed = initial_guess(seed_inputs)
            outcome = _window_at_point(T_reject_target, config, x0_seed, guess_C)
            reported_T_reject = T_reject_target

        if outcome is None:
            points.append(FeasibilityPoint(
                T_reject_C=reported_T_reject, T11_min_C=float("nan"), T11_max_C=float("nan"),
                dT_drive_min_K=float("nan"), dT_drive_max_K=float("nan"), feasible=False,
                message=f"The anchor search at {reported_T_reject:.2f} °C also found no solution.",
            ))
            print(f"[{i+1}/{len(T_reject_values_C)}] T_reject={reported_T_reject:6.2f} °C -> FAILED")
            continue

        point, x0_min = outcome
        points.append(point)
        print(
            f"[{i+1}/{len(T_reject_values_C)}] T_reject={reported_T_reject:6.2f} °C -> "
            f"T11 in [{point.T11_min_C:6.2f}, {point.T11_max_C:6.2f}] °C, "
            f"dT_drive in [{point.dT_drive_min_K:6.2f}, {point.dT_drive_max_K:6.2f}] K [OK]"
        )

        T_reject_cur = reported_T_reject
        T11_cur = point.T11_min_C
        x0_cur = x0_min

    return points


# ---------------------------------------------------------------------------
# Output: table + plot
# ---------------------------------------------------------------------------

def print_sweep_table(points: Sequence[FeasibilityPoint]) -> None:
    print("=" * 100)
    print(
        f"{'T_reject[C]':>10} {'T11_min[C]':>11} {'T11_max[C]':>11} "
        f"{'dT_drv_min[K]':>13} {'dT_drv_max[K]':>13} {'Status':>8}"
    )
    print("-" * 100)
    for p in points:
        status = "OK" if p.feasible else "FAIL"
        print(
            f"{p.T_reject_C:10.2f} {p.T11_min_C:11.2f} {p.T11_max_C:11.2f} "
            f"{p.dT_drive_min_K:13.2f} {p.dT_drive_max_K:13.2f} {status:>8}"
        )
    print("=" * 100)


def plot_feasibility_sweep(
    points: Sequence[FeasibilityPoint],
    *,
    duehring_reference: Optional[Sequence] = None,
    save_path: Optional[str] = f"Design_Point/Plots/{plot_name}.png",
    show: bool = True,
):
    """Plots the feasible T11 window vs. reject-cooling temperature;
    optionally compares against the optimistic Duehring lower bound (a list
    of MinGenResult from AC_duehring_screening.sweep_recool_temperature())."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ok_points = [p for p in points if p.feasible]
    x = np.array([p.T_reject_C for p in ok_points])
    y_min = np.array([p.T11_min_C for p in ok_points])
    y_max = np.array([p.T11_max_C for p in ok_points])

    ax.fill_between(x, y_min, y_max, color="tab:orange", alpha=0.18, label="feasible window")
    ax.plot(x, y_min, "o-", color="tab:orange", label="T11_min")
    ax.plot(x, y_max, "o--", color="tab:orange", linewidth=1.2, label="T11_max")

    fail_points = [p for p in points if not p.feasible]
    if fail_points:
        xf = np.array([p.T_reject_C for p in fail_points])
        ax.plot(
            xf, np.zeros_like(xf), "x", color="tab:red", markersize=8,
            markeredgewidth=2, label="infeasible",
        )

    if duehring_reference is not None:
        xr = np.array([r.T_recool_C for r in duehring_reference])
        yr = np.array([r.T_gen_min_C for r in duehring_reference])
        okr = np.array([r.feasible for r in duehring_reference])
        ax.plot(xr[okr], yr[okr], "--", color="0.4", label="Duehring bound (optimistic)")

    ax.set_xlabel("Reject-cooling temperature T13 = T15 [°C]")
    ax.set_ylabel("Generator inlet temperature T11 [°C]")
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

    # All physical assumptions (pinch values, approach values, T18_spec_C)
    # live centrally in FeasibilitySweepConfig above -- adjust them there,
    # not here.
    T_REJECT_RANGE_C = list(
        np.arange(T_REJECT_START_C, T_REJECT_END_C - 0.5 * T_REJECT_STEP_C, -abs(T_REJECT_STEP_C))
        if T_REJECT_END_C < T_REJECT_START_C else
        np.arange(T_REJECT_START_C, T_REJECT_END_C + 0.5 * T_REJECT_STEP_C, abs(T_REJECT_STEP_C))
    )

    print(f"Minimum required generator inlet temperature T11 vs. reject-cooling temperature")
    print(f"(T18_spec_C = {config.T18_spec_C:.1f} °C constant, fixed evaporator outlet temperature)")
    points = sweep_min_generator_temperature_homotopy(T_REJECT_RANGE_C, config)
    print_sweep_table(points)

    duehring_reference = None
    try:
        from AC_duehring_screening import sweep_recool_temperature
    except ImportError:
        try:
            from Design_Point.AC_duehring_screening import sweep_recool_temperature
        except ImportError:
            sweep_recool_temperature = None
    if sweep_recool_temperature is not None:
        duehring_reference = sweep_recool_temperature(
            T_REJECT_RANGE_C, T_evap_target_C=config.T18_spec_C - config.dT_min_evap,
            dT_min_des=config.dT_min_des, dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
        )

    plot_feasibility_sweep(points, duehring_reference=duehring_reference)

    # Additional evaluations: reuse the `points` already computed above (no
    # re-running the sweep). Lazy imports so the two scripts themselves stay
    # independently importable/runnable.
    if ENABLE_DUEHRING_MULTI_PLOT:
        from Design_Point.Visualization_Scripts.AC_duehring_multi_process_plot import (
            select_and_plot_duehring,
        )
        select_and_plot_duehring(
            points, everys_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{duehring_plot_name}.png",
        )

    if ENABLE_QT_MULTI_PDF:
        from Design_Point.Visualization_Scripts.AC_qt_multi_process_plot import (
            select_and_plot_qt_pdf,
        )
        select_and_plot_qt_pdf(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{qt_pdf_name}.pdf",
        )
