"""Normalized AC (summer chiller) characteristic map at FIXED UA geometry.

2D raster over (T_reject_C, load_fraction):

    FIXED  : T_11_C (desorber drive -- deep geothermal supply, assumed
             constant year-round)
             T_17_C / T18_spec_C (chilled-water loop, constant supply
             setpoint at constant Netzspreizung)
    SWEPT  : T_reject_C (cooling-tower / ambient reject temperature --
             absorber+condenser pair)
             load_fraction = Q_evap / Q_evap_design, in (0, 1]

Routing of the absorber+condenser reject pair follows whatever
absorber_condenser_routing_mode the design point itself was exported with
(design["boundary_conditions"]["absorber_condenser_routing_mode"]) -- in
particular "series_absorber_to_condenser" (one shared reject stream,
absorber first then condenser, T15=T14 internally -- the common industrial
arrangement) works directly: T_reject_C feeds the single external inlet of
whichever component comes first in the chain (see
_routing_temperature_kwargs).

CONTROL STRATEGY: CONSTANT EXTERNAL TEMPERATURE DIFFERENCES, NOT A COP
SEARCH (this replaced two earlier versions of this file -- read this
before trusting the numbers from either of them):

  v1 fixed m_11 = load_fraction * m11_design everywhere (the ratio the
     design-point optimizer implicitly settled on, but only AT its own
     design T_reject). Away from that temperature it was badly
     suboptimal: at T_reject=20 degC, load=0.95, COP=0.49, while a
     different m_11 gave COP=0.80 -- BETTER than the T_reject=30 degC
     design point itself.
  v2 tried to fix that by SEARCHING for the m_11 (and bounding m_13/m_15)
     that maximizes COP at every grid point. This worked, but exposed a
     genuinely disconnected feasible region in m_11 -- finding the true
     optimum reliably needed an expensive fine-grained scan, >1200 s for
     a single hard raster cell (>80 h for the whole 2D raster -- not
     tractable). A cheaper, coarser compromise existed but still left
     real doubt about whether any given cell was close to optimal.

v3 (this version) sidesteps the search entirely: instead of asking "which
flows maximize COP here", it asks "which flows reproduce the DESIGN
POINT's own external approach temperatures here" -- i.e. holds the
external temperature differences across all four heat exchangers at their
design-point values, and solves for the flows (m_11, m_13=m_15) that
reproduce them via a small FIXED-POINT ITERATION (same technique as
aht_reference_point_export.calibrate_reference_flows), not an
optimization search. This is both much cheaper (3-6 solves per grid point
instead of 10-40+) and arguably more realistic: it is what you would
actually expect a plant controller to do (hold design approach
temperatures), not an idealized "always instantaneously COP-optimal"
assumption no real controller achieves. It also directly addresses why v1
collapsed at low T_reject: that collapse came from holding the flow/load
RATIO fixed, not the approach temperature -- holding the temperature
difference fixed instead lets m_11 adapt the way a real controller would.

Design-point delta-Ts used as the fixed targets (read directly from
design["boundary_conditions"], which already carries T12/T14/T16_spec_C
from the pinch-model design-point run):
    dT_des     = T_11_C - T12_spec_C          (desorber, single)
    dT_evap    = T_17_C - T18_spec_C          (evaporator, single --
                 already enforced for free: T_17_C/T18_spec_C are both
                 fixed constants and evaporator_spec_mode="T18", so m_17
                 comes out of the model already respecting this dT)
    dT_reject  = T16_spec_C - T_13_C(design)  (absorber+condenser TOGETHER)

IMPORTANT about dT_reject: m_13 and m_15 are the SAME physical flow
(series routing), so you cannot independently pin dT_abs AND dT_cond --
doing so is only consistent if Q_abs/Q_cond happens to keep exactly the
dT_abs/dT_cond ratio, which the cycle does not guarantee. This version
therefore pins the ONE meaningful external quantity for a shared-flow
series loop: the TOTAL temperature rise across the whole chain
(T16-T13_design), via m_13 = m_15 = (Q_abs+Q_cond) / (cp * dT_reject).
How that total split internally between absorber and condenser (T14) is
left to the cycle, exactly as it would be on a real shared reject loop.

Output fields (normalized by the design duty unless noted):
    Q_evap_frac : achieved Q_evap / Q_evap_design (convergence sanity
                  check -- should equal load_fraction almost exactly)
    COP         : resulting COP under the constant-dT control strategy
                  (not a COP-maximizing search result)
    m11_frac    : desorber (geothermal) mass flow / m11_design that
                  reproduces dT_des at this point -- a genuine map output,
                  no longer == load_fraction by construction
    m13_frac    : reject-loop mass flow / m13_design that reproduces
                  dT_reject at this point -- likewise a genuine output
    T12_C       : geothermal return temperature leaving the desorber

Standalone usage
-----------------
    python TEA/ac_characteristic_map.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AC_UA_LMTD import (
    ACInputs,
    PRIMARY_VARIABLE_NAMES,
    bounds as ac_bounds,
    initial_guess as ac_initial_guess,
    solve_ac,
)
from TEA.characteristic_map import CharacteristicMap
from TEA.design_point_export import DEFAULT_EXPORT_PATH, load_design_point

RESIDUAL_TOL = 1.0e-6
DEFAULT_MAP_PATH = Path(__file__).resolve().parent / "maps" / "ac_characteristic_map.npz"


@dataclass
class ACMapConfig:
    T_reject_values_C: Tuple[float, ...] = tuple(np.arange(10.0, 36.0, 2.0))
    load_fraction_values: Tuple[float, ...] = tuple(np.round(np.arange(0.15, 1.0001, 0.05), 3))
    min_load_fraction: float = 0.15  # below this, treat the machine as OFF in the dispatch
    solver_tol: float = 1.0e-7
    max_nfev: int = 400
    max_nfev_retry: int = 1500  # fallback budget when the fast attempt fails to converge (cost note below)

    # --- constant-dT fixed-point iteration (cheap starting estimate) ---
    deltaT_max_iter: int = 15
    deltaT_tol_K: float = 0.02
    deltaT_relaxation: float = 0.5  # 1.0 = undamped substitution (overshoots, see _solve_with_constant_deltaT)

    # --- directional refinement toward smaller flow, from that estimate ---
    # multipliers applied to the fixed point's (m_11, m_13), tried in
    # order; the walk stops at the first step that fails to converge
    # (single retry only, see _solve_with_constant_deltaT's cost note).
    refine_k_values: Tuple[float, ...] = (0.6, 0.35, 0.2, 0.12, 0.07)


def _clip_to_bounds(z: np.ndarray, inputs: ACInputs) -> np.ndarray:
    lower, upper = ac_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


def _routing_temperature_kwargs(routing_mode: str, T_reject_C: float) -> dict:
    """T_13_C/T_15_C for the given routing_mode, with T_reject_C as the
    ONE externally given reject temperature (the Kennfeld's sweep axis):
    for "parallel" it feeds both independent reject streams; for a series
    mode it is the inlet of whichever component comes FIRST in that
    series chain, the other temperature is None (computed internally,
    T15=T14 or T13=T16). Mirrors
    Design_Point/AC_design_point_optimizer.py's _routing_temperature_kwargs.
    """
    if routing_mode == "series_absorber_to_condenser":
        return dict(T_13_C=T_reject_C, T_15_C=None)
    if routing_mode == "series_condenser_to_absorber":
        return dict(T_13_C=None, T_15_C=T_reject_C)
    return dict(T_13_C=T_reject_C, T_15_C=T_reject_C)


def _build_inputs(
    T_reject_C: float,
    load_fraction: float,
    m_11: float,
    m_13: float,
    design: dict,
    config: ACMapConfig,
) -> ACInputs:
    bc = design["boundary_conditions"]
    ua = design["UA_kW_K"]
    routing_mode = bc["absorber_condenser_routing_mode"]

    return ACInputs(
        T_11_C=bc["T_11_C"],
        **_routing_temperature_kwargs(routing_mode, T_reject_C),
        T_17_C=bc["T_17_C"],
        m_11=m_11,
        m_13=m_13,
        m_15=m_13,  # series routing: same physical flow through absorber and condenser
        UA_cond=ua["UA_cond"],
        UA_evap=ua["UA_evap"],
        UA_abs=ua["UA_abs"],
        UA_des=ua["UA_des"],
        UA_shex=ua["UA_shex"],
        absorber_condenser_routing_mode=routing_mode,
        cycle_scale_spec_mode="Qeva",
        Qevap_spec_kW=load_fraction * design["Q_evap_design_kW"],
        evaporator_spec_mode="T18",
        T18_spec_C=bc["T18_spec_C"],
        cp_w_kJkgK=bc["cp_w_kJkgK"],
        solver_tol=config.solver_tol,
        max_nfev=config.max_nfev,
    )


def _is_converged(result) -> bool:
    """Numerically converged to a real physical state -- NOT the same as
    safe/acceptable (see _is_valid). A converged-but-unsafe point (e.g.
    crystallization risk) still has a trustworthy energy balance and is
    fine to drive the next fixed-point iteration from; only a FINAL answer
    needs the full _is_valid check.
    """
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    return info.scaled_residual_norm <= RESIDUAL_TOL


def _is_valid(result) -> bool:
    if not _is_converged(result):
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _solve_with_constant_deltaT(
    T_reject_C: float,
    load_fraction: float,
    design: dict,
    config: ACMapConfig,
    x0_hint: np.ndarray,
    *,
    m_11_guess: Optional[float] = None,
    m_13_guess: Optional[float] = None,
):
    """Finds m_11 and m_13(=m_15) via fixed-point iteration so that the
    desorber's and the reject-loop's external temperature differences
    match their design-point values -- see the module docstring for the
    full rationale and the dT_reject caveat (one shared dT for the
    absorber+condenser series pair, not two independent ones).

    Each round: solve the cycle at the current (m_11, m_13) guess, read
    back the resulting Q_des and Q_abs+Q_cond, and recompute
        m_11 = Q_des / (cp * dT_des_target)
        m_13 = (Q_abs + Q_cond) / (cp * dT_reject_target)
    repeating until both the desorber and reject-loop dT are within
    config.deltaT_tol_K of target (or config.deltaT_max_iter is reached,
    in which case the LAST solved point is returned anyway).

    m_11_guess/m_13_guess seed the FIRST round (default: load_fraction *
    the design-point flow). IMPORTANT: the constant-dT fixed point is not
    always unique -- confirmed directly, restarting from the naive
    load-scaled guess at EVERY grid cell independently can converge to a
    markedly worse fixed point away from the design T_reject (one
    measured case: COP=0.23 from the naive guess vs. COP=0.86 reachable
    from a better start at the same (T_reject, load) point). Callers
    sweeping a grid should therefore pass the already-solved NEIGHBOR
    cell's (m11_frac, m13_frac) here, the same continuation-from-a-known-
    good-point principle used throughout this codebase -- see build_map(),
    which tracks this across the raster. A cold restart at every cell is
    what produced the bad fixed point in testing.

    Returns (ACResult, m11_frac, m13_frac, x_solution) on success, None if
    no round produced a valid solve at all.
    """
    bc = design["boundary_conditions"]
    flows = design["design_flows_kg_s"]
    cp = bc["cp_w_kJkgK"]

    dT_des_target = bc["T_11_C"] - bc["T12_spec_C"]
    dT_reject_target = bc["T16_spec_C"] - bc["T_13_C"]  # design-point T13, not the swept T_reject_C

    # Initial guesses: caller-supplied (ideally a solved neighbor's
    # result, see the docstring) or, failing that, design-point flows
    # scaled by load_fraction -- refined below from the actual solved Q's.
    m_11 = m_11_guess if m_11_guess is not None else load_fraction * flows["m11"]
    m_13 = m_13_guess if m_13_guess is not None else load_fraction * flows["m13"]

    x0 = x0_hint
    # Tracks the best VALID (result, m_11, m_13, x) seen so far, matched
    # to the flows that actually produced it -- a later iteration's solve
    # failing must not discard an earlier, perfectly good point (this was
    # a real bug: T_reject=30 deg C, load=0.95, essentially the design
    # point itself, was mis-reported "infeasible" because iteration 2
    # failed and the whole attempt was abandoned instead of falling back
    # to iteration 1's valid solve).
    last_good: Optional[tuple] = None

    for _ in range(config.deltaT_max_iter):
        inputs = _build_inputs(T_reject_C, load_fraction, m_11, m_13, design, config)
        x0c = _clip_to_bounds(x0, inputs)
        try:
            result = solve_ac(inputs, x0=x0c)
        except Exception:
            result = None
        if result is None or not _is_converged(result):
            # Fast attempt failed to even CONVERGE (this check does not
            # care about crystallization/other safety checks yet -- see
            # _is_converged) -- retry once at a much larger iteration
            # budget before falling back to a cold start.
            inputs_retry = replace(inputs, max_nfev=config.max_nfev_retry)
            try:
                result_retry = solve_ac(inputs_retry, x0=x0c)
            except Exception:
                result_retry = None
            if result_retry is not None and _is_converged(result_retry):
                result = result_retry
            else:
                cold_x0 = ac_initial_guess(inputs)
                try:
                    result_cold = solve_ac(inputs_retry, x0=cold_x0)
                except Exception:
                    result_cold = None
                result = result_cold if result_cold is not None and _is_converged(result_cold) else None

        if result is None:
            # This round's (m_11, m_13) guess didn't even converge
            # numerically -- stop and fall back to the last known-good
            # point rather than discarding it (see the note above).
            break

        x0 = np.array([result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES])
        # Only record as a candidate FINAL answer if it also passes the
        # full safety checks (crystallization etc.) -- but keep iterating
        # from it regardless, since the energy balance (Q_des, Q_abs,
        # Q_cond) of a merely-converged-but-unsafe point is still
        # trustworthy for computing the next (m_11, m_13) guess. A point
        # that is converged but, say, crystallization-unsafe at one
        # (m_11, m_13) can easily become safe at the NEXT iteration's
        # guess (confirmed directly: T_reject=30 deg C, load=0.95 --
        # essentially the design point itself -- converged immediately
        # with COP=0.79 but failed the crystallization check; aborting
        # right there, as an earlier version of this function did, mis-
        # reported a near-design-point as fully infeasible).
        if _is_valid(result):
            last_good = (result, m_11, m_13, x0)

        T12_now = result.diagnostics["T12_K"] - 273.15
        T16_now = result.diagnostics["T16_K"] - 273.15
        dT_des_now = bc["T_11_C"] - T12_now
        dT_reject_now = T16_now - T_reject_C

        if (
            _is_valid(result)
            and abs(dT_des_now - dT_des_target) < config.deltaT_tol_K
            and abs(dT_reject_now - dT_reject_target) < config.deltaT_tol_K
        ):
            break

        # Damped update: a direct (undamped) fixed-point substitution
        # overshot badly in testing (e.g. m11_frac landing at 3.25, COP
        # collapsing to 0.23, at a point where a bounded search had
        # previously found COP=0.86) -- this system's Q(m) response is
        # steep enough that the naive update is not contractive on its
        # own. Blending the fixed-point target with the previous guess
        # damps that oscillation while still converging to the same
        # fixed point once close to it.
        Q_des = result.heat_flows_kW["Q_des"]
        Q_reject_total = result.heat_flows_kW["Q_abs"] + result.heat_flows_kW["Q_cond"]
        m_11_target = Q_des / (cp * dT_des_target)
        m_13_target = Q_reject_total / (cp * dT_reject_target)
        m_11 = (1.0 - config.deltaT_relaxation) * m_11 + config.deltaT_relaxation * m_11_target
        m_13 = (1.0 - config.deltaT_relaxation) * m_13 + config.deltaT_relaxation * m_13_target

    if last_good is None:
        return None
    result, m_11_used, m_13_used, x = last_good

    # --- Directional refinement toward smaller flow ---
    # The constant-dT fixed point is a cheap, thermodynamically CONSISTENT
    # starting estimate, not a claim of optimality (see the module
    # docstring and the chat discussion): away from the design T_reject it
    # can itself be the worse of (at least) two branches, because "hold dT
    # constant via more flow when duty grows" is self-reinforcing in the
    # wrong direction at fixed UA. Direct testing found the better branch
    # consistently at SMALLER flow than the fixed point, not larger -- so
    # walk a geometric sequence of shrinking multipliers from here and
    # keep whichever point has the lowest Q_des, stopping the walk (but
    # keeping the best point found so far) at the first step that fails
    # even the single retry below.
    #
    # COST NOTE: an earlier version also tried a cold-start rescue (like
    # the fixed-point phase above) on every failing refinement step. That
    # made a 39-cell test take ~100 minutes (>80 h extrapolated to the
    # full 234-cell raster) -- most of it spent on expensive (max_nfev_
    # retry-budget) solves for steps that were going to be abandoned
    # anyway once the walk stops. Dropping the cold-start rescue here (one
    # retry only, then stop) is a deliberate speed/thoroughness trade-off:
    # a step right at the edge of the feasible region may be given up on
    # slightly early, keeping whatever the walk already found instead of
    # possibly finding one more step's improvement.
    best_result, best_m11, best_m13, best_x = result, m_11_used, m_13_used, x
    cur_x = x
    for k in config.refine_k_values:
        m11_trial = k * m_11_used
        m13_trial = k * m_13_used
        inputs = _build_inputs(T_reject_C, load_fraction, m11_trial, m13_trial, design, config)
        x0c = _clip_to_bounds(cur_x, inputs)
        try:
            r = solve_ac(inputs, x0=x0c)
        except Exception:
            r = None
        if r is None or not _is_converged(r):
            inputs_retry = replace(inputs, max_nfev=config.max_nfev_retry)
            try:
                r_retry = solve_ac(inputs_retry, x0=x0c)
            except Exception:
                r_retry = None
            r = r_retry if r_retry is not None and _is_converged(r_retry) else None

        if r is None:
            break  # this direction is exhausted -- keep the best found so far

        cur_x = np.array([r.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES])
        if _is_valid(r) and r.heat_flows_kW["Q_des"] < best_result.heat_flows_kW["Q_des"]:
            best_result, best_m11, best_m13, best_x = r, m11_trial, m13_trial, cur_x

    return best_result, best_m11 / flows["m11"], best_m13 / flows["m13"], best_x


def _anchor_expand_order(n: int, anchor_index: int) -> list:
    """Row order: the anchor first, then alternating outward
    (anchor-1, anchor+1, anchor-2, anchor+2, ...), each paired with its
    nearer already-visited neighbor ("parent") to seed from. The anchor
    itself has no parent (None).

    NOT the same "anchor and expand" idea tried earlier for warm-starting
    the raw solver state x (and rejected there, see characteristic_map.py
    -- it measurably hurt coverage for that purpose). Here it solves a
    different problem: which (m11_frac, m13_frac) FIXED POINT the
    constant-dT iteration converges to is path-dependent (see
    _solve_with_constant_deltaT), and marching T_reject monotonically from
    one edge let a bad fixed point found far from the design temperature
    drag the whole raster onto it -- confirmed directly: the KNOWN-GOOD
    T_reject=30 deg C point (COP=0.79 solved in isolation) came out
    COP=0.66 when reached by ascending continuation from T_reject=20 deg C
    (COP=0.23) instead. Anchoring at the design T_reject and expanding
    outward keeps every row seeded from the direction of the one point
    guaranteed to sit on the good branch: the design point itself.
    """
    order = [(anchor_index, None)]
    lo, hi = anchor_index - 1, anchor_index + 1
    while lo >= 0 or hi < n:
        if lo >= 0:
            order.append((lo, lo + 1))
            lo -= 1
        if hi < n:
            order.append((hi, hi - 1))
            hi += 1
    return order


def build_map(design: dict, config: ACMapConfig = ACMapConfig(), *, verbose: bool = True) -> CharacteristicMap:
    Q_design = design["Q_evap_design_kW"]
    flows = design["design_flows_kg_s"]
    bc = design["boundary_conditions"]

    T_values = np.asarray(config.T_reject_values_C, dtype=float)
    L_values = np.asarray(config.load_fraction_values, dtype=float)
    n_T, n_L = len(T_values), len(L_values)

    field_names = ["Q_evap_frac", "COP", "m11_frac", "m13_frac", "T12_C"]
    grids = {name: np.full((n_T, n_L), np.nan) for name in field_names}
    x_solutions: list = [[None] * n_L for _ in range(n_T)]
    # (m11_frac, m13_frac) per solved cell -- see _anchor_expand_order and
    # _solve_with_constant_deltaT for why seeding this from a neighbor
    # (not restarting from the naive load_fraction*design guess) matters.
    flow_frac_grid: dict = {}

    anchor_index = int(np.argmin(np.abs(T_values - bc["T_13_C"])))  # design-point T_reject

    for i, parent_i in _anchor_expand_order(n_T, anchor_index):
        T_reject_C = float(T_values[i])
        prev_j = None
        for j in range(n_L - 1, -1, -1):
            load_fraction = float(L_values[j])

            x0_candidates = []
            if prev_j is not None and x_solutions[i][prev_j] is not None:
                x0_candidates.append(x_solutions[i][prev_j])
            if parent_i is not None and x_solutions[parent_i][j] is not None:
                x0_candidates.append(x_solutions[parent_i][j])

            flow_guess = None
            if prev_j is not None and (i, prev_j) in flow_frac_grid:
                flow_guess = flow_frac_grid[(i, prev_j)]
            elif parent_i is not None and (parent_i, j) in flow_frac_grid:
                flow_guess = flow_frac_grid[(parent_i, j)]

            if flow_guess is not None:
                m_11_guess = flow_guess[0] * flows["m11"]
                m_13_guess = flow_guess[1] * flows["m13"]
            else:
                m_11_guess = load_fraction * flows["m11"]
                m_13_guess = load_fraction * flows["m13"]
            x0_candidates.append(
                ac_initial_guess(_build_inputs(T_reject_C, load_fraction, m_11_guess, m_13_guess, design, config))
            )

            outcome = None
            for x0 in x0_candidates:
                outcome = _solve_with_constant_deltaT(
                    T_reject_C, load_fraction, design, config, x0,
                    m_11_guess=m_11_guess if flow_guess is not None else None,
                    m_13_guess=m_13_guess if flow_guess is not None else None,
                )
                if outcome is not None:
                    break

            prev_j = j
            if outcome is None:
                if verbose:
                    print(f"  [T={T_reject_C:6.2f} C, load={load_fraction:5.2f}] -> INFEASIBLE")
                continue

            result, m11_frac, m13_frac, x_sol = outcome
            x_solutions[i][j] = x_sol
            flow_frac_grid[(i, j)] = (m11_frac, m13_frac)
            grids["Q_evap_frac"][i, j] = result.heat_flows_kW["Q_evap"] / Q_design
            grids["COP"][i, j] = result.kpis["COP"]
            grids["m11_frac"][i, j] = m11_frac
            grids["m13_frac"][i, j] = m13_frac
            grids["T12_C"][i, j] = result.diagnostics["T12_K"] - 273.15
            if verbose:
                print(
                    f"  [T={T_reject_C:6.2f} C, load={load_fraction:5.2f}] -> OK  "
                    f"COP={result.kpis['COP']:.4f}, m11_frac={m11_frac:.4f}, m13_frac={m13_frac:.4f}"
                )

    return CharacteristicMap(
        T_ambient_C=np.asarray(config.T_reject_values_C, dtype=float),
        load_fraction=np.asarray(config.load_fraction_values, dtype=float),
        fields=grids,
        meta={
            "mode": "AC",
            "design_point_source": str(DEFAULT_EXPORT_PATH),
            "Q_evap_design_kW": Q_design,
            "T_11_C": design["boundary_conditions"]["T_11_C"],
            "T_17_C": design["boundary_conditions"]["T_17_C"],
            "T18_spec_C": design["boundary_conditions"]["T18_spec_C"],
            "min_load_fraction": config.min_load_fraction,
            "control_strategy": "constant_external_deltaT",
        },
    )


if __name__ == "__main__":
    design = load_design_point()
    cmap = build_map(design)
    cmap.save(DEFAULT_MAP_PATH)
    print(f"AC characteristic map saved: {DEFAULT_MAP_PATH}")
