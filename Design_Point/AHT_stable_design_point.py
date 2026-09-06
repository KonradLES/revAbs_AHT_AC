"""
For a fixed operating point (external temperatures + duty), finds via a
continuation/homotopy strategy:

  (a) a stable initial guess x0 for the 7 primary unknowns, and
  (b) the smallest simultaneously achievable set of minimum pinch
      temperature differences (dT_min) for the 5 heat exchangers (SHEX,
      desorber, condenser, evaporator, absorber),

so that operating point + x0 + dT_min can then be passed as the starting
point to the bilevel optimizer.

Approach (homotopy):
  - Start at very loose (large) dT_min values -> the equation system is
    "soft" and converges essentially regardless of the initial guess.
  - Homotopy parameter t in [0, 1] linearly interpolates between the loose
    starting values (t=0) and the desired "floor" target values (t=1).
  - t is increased stepwise; each converged solution is used as a warm
    start (x0) for the next, slightly harder step.
  - If a step fails, the step size is halved (bisection). Once the step
    size gets too small, the last converged point is taken as the
    (practical) pinch limit for this operating point.

This version uses the real interfaces from AHT_Pinch_Point.py directly:
  - initial_guess(inputs)  -> the model's generic initial-guess heuristic
  - solve_aht(...) -> AHTResult with .primary_variables (dict!), .solve_info
    (success, final_point_evaluable, scaled_residual_norm), and .checks
  - A solution point only counts as "converged" here once ALL of the
    following hold (plain scipy success isn't enough, since trf can
    terminate "successfully" at an unphysical/unstable point if the soft
    residuals happen to be small there):
      1. solve_info.success               == True
      2. solve_info.final_point_evaluable == True  (strict final evaluation ok)
      3. solve_info.scaled_residual_norm  <= RESIDUAL_TOL
      4. all result.checks values         == True  (concentration ordering,
         crystallization safety, mass balances, etc.)
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import dataclasses

import numpy as np

from Models.AHT_Pinch_Point import (
    AHTInputs,
    AHTResult,
    PRIMARY_VARIABLE_NAMES,
    initial_guess,
    primary_temperatures_K_to_C,
    print_summary,
    solve_aht,
)

# Threshold on the scaled residual norm above which a point counts as
# "converged" (not just "terminated by scipy").
RESIDUAL_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# 1) Fix the operating point (stays CONSTANT for the whole run)
#    -> enter your external temperatures and required duty here
# ---------------------------------------------------------------------------
OPERATING_POINT = dict(
    T_11_C=75.0,
    T_13_C=60.0,
    T_15_C=60.0,
    T_17_C=20.0,
    Qabs_spec_kW=500.4,
    T12_spec_C=80.02,
    T14_spec_C=53.92,
    T16_spec_C=53.80,
    T18_spec_C=26.26,
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="T12",
    desorber_spec_mode="T14",
    evaporator_spec_mode="T16",
    condenser_spec_mode="T18",
    cycle_scale_spec_mode="Qabs",
    desorber_evaporator_routing_mode="parallel",
)

# ---------------------------------------------------------------------------
# 2) Homotopy settings for the pinch temperature differences
# ---------------------------------------------------------------------------
# "Loose" = guaranteed unproblematic starting point (don't change unless
# even this doesn't converge for you -> then raise it further)
DT_MIN_LOOSE = dict(
    dT_min_shex=65.0,
    dT_min_des=65.0,
    dT_min_cond=65.0,
    dT_min_evap=65.0,
    dT_min_abs=65.0,
)

# "Floor" = what you actually want to reach (feel free to pick this
# aggressively small -- the script automatically finds the real limit if
# it's not fully reachable)
DT_MIN_FLOOR = dict(
    dT_min_shex=1.0,
    dT_min_des=1.0,
    dT_min_cond=1.0,
    dT_min_evap=1.0,
    dT_min_abs=1.0,
)

T_STEP_INITIAL = 0.25       # initial homotopy step (fraction of [0,1])
T_STEP_MIN = 1e-3           # abort once the step size drops below this
MAX_ITERATIONS = 500


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _x0_from_result(result: AHTResult) -> np.ndarray:
    """Extracts the solution vector from an AHTResult in PRIMARY_VARIABLE_NAMES
    order, so it can be used directly as x0 (warm start) for the next
    solve_aht call.
    """
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES],
        dtype=float,
    )


def _is_valid_solution(result: AHTResult) -> bool:
    """Strict convergence check, see module docstring above."""
    info = result.solve_info
    if not info.success:
        return False
    if not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _with_dT_min(inputs: AHTInputs, dT_min: dict) -> AHTInputs:
    """Returns a copy of `inputs` with updated dT_min values."""
    try:
        return dataclasses.replace(inputs, **dT_min)
    except TypeError:
        # Fallback in case AHTInputs is not a dataclass
        data = vars(inputs).copy()
        data.update(dT_min)
        return AHTInputs(**data)


def _interp_dT_min(t: float) -> dict:
    """Linear interpolation between the loose and target dT_min values."""
    return {
        key: (1.0 - t) * DT_MIN_LOOSE[key] + t * DT_MIN_FLOOR[key]
        for key in DT_MIN_LOOSE
    }


def _try_solve(inputs: AHTInputs, x0: np.ndarray):
    """Calls solve_aht and strictly checks the result (see _is_valid_solution).
    solve_aht does NOT raise on non-convergence; it returns an AHTResult with
    solve_info/checks fields set accordingly -- the try/except here is
    defensive nonetheless (e.g. in case AHTInputs itself raises a ValueError
    at construction).
    """
    try:
        result = solve_aht(inputs, x0=x0)
    except Exception as exc:
        print(f"    -> exception in solve_aht: {exc}")
        return False, None

    if not _is_valid_solution(result):
        info = result.solve_info
        print(
            f"    -> not converged (success={info.success}, "
            f"final_point_evaluable={info.final_point_evaluable}, "
            f"residual_norm={info.scaled_residual_norm:.3e})"
        )
        if result.checks and not all(result.checks.values()):
            failed = [k for k, v in result.checks.items() if not v]
            print(f"       failed plausibility checks: {failed}")
        return False, None

    return True, result


def find_stable_operating_point(base_inputs: AHTInputs):
    """Runs the homotopy.

    Returns:
        best_inputs: AHTInputs with the smallest dT_min reached
        best_x0: corresponding converged solution vector (warm-start ready)
        best_dT: dict of the dT_min values reached
        best_result: last successful solve_aht result
    """
    x0 = initial_guess(base_inputs)

    t = 0.0
    t_step = T_STEP_INITIAL

    best_inputs = _with_dT_min(base_inputs, _interp_dT_min(t))
    ok, result = _try_solve(best_inputs, x0)
    if not ok:
        raise RuntimeError(
            "Even the loose dT_min starting values (DT_MIN_LOOSE) don't converge. "
            "Please raise DT_MIN_LOOSE further (e.g. to 35-40 K) or check the "
            "operating point in OPERATING_POINT."
        )
    best_x0 = _x0_from_result(result)
    best_dT = _interp_dT_min(t)
    best_result = result

    print(f"[t=0.00] OK   dT_min={best_dT}")

    iteration = 0
    while t < 1.0 and iteration < MAX_ITERATIONS:
        iteration += 1
        t_trial = min(1.0, t + t_step)
        dT_trial = _interp_dT_min(t_trial)
        trial_inputs = _with_dT_min(base_inputs, dT_trial)

        print(f"[t={t_trial:.4f}] trying dT_min={dT_trial}")
        ok, result = _try_solve(trial_inputs, best_x0)

        if ok:
            t = t_trial
            best_inputs = trial_inputs
            best_x0 = _x0_from_result(result)
            best_dT = dT_trial
            best_result = result
            # after success, cautiously grow the step size again
            t_step = min(T_STEP_INITIAL, t_step * 1.5)
            print("    -> OK")
        else:
            t_step /= 2.0
            if t_step < T_STEP_MIN:
                print(
                    f"\nAborting: step size fell below the minimum. "
                    f"Last stable point at t={t:.4f}."
                )
                break

    if t < 1.0 - 1e-9:
        print(
            "\nNOTE: the desired floor values (DT_MIN_FLOOR) are NOT fully "
            "reachable for this operating point.\n"
            "The dT_min printed below are the smallest simultaneously "
            "reachable values (the practical pinch limit of this operating point)."
        )
    else:
        print("\nTarget dT_min (DT_MIN_FLOOR) fully reached.")

    return best_inputs, best_x0, best_dT, best_result


if __name__ == "__main__":
    base_inputs = AHTInputs(**OPERATING_POINT, **DT_MIN_LOOSE)

    stable_inputs, stable_x0, stable_dT_min, result = find_stable_operating_point(
        base_inputs
    )

    print("\n" + "=" * 70)
    print("RESULT - stable operating point found")
    print("=" * 70)

    print("\nMinimum simultaneously reachable pinch temperature differences:")
    for k, v in stable_dT_min.items():
        print(f"  {k:15s} = {v:.3f} K")

    stable_x0_C = primary_temperatures_K_to_C(stable_x0)
    print("\nStable initial vector x0:")
    print(f"  internal (K or -)   : {stable_x0}")
    print(f"  readable (degC or -): {dict(zip(PRIMARY_VARIABLE_NAMES, stable_x0_C))}")

    print("\nFull result summary:")
    print_summary(result)