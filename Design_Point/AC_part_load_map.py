"""Part-load map of the AC (UA/LMTD model), return-flow admixture.

Goal
----
For a FIXED machine (UA values from one design point) and a geothermal drive
source with constant supply temperature T_geo, determine the COP-optimal
drive-side operation for every operating point and collect the results in a
map that a later techno-economic analysis (TEA) can use:

    axes:  PLR = Q_evap / Q_nom            cooling demand
           T_reject                        cooling-water inlet T13 (coldest
                                           temperature the cooling tower can
                                           deliver -- always used, since a
                                           colder reject temperature never
                                           hurts the COP)
           m_cw / m_cw_design              cooling-water flow (m13 = m15 in
                                           series)

The cooling-water flow is an AXIS, not a decision variable: without pump and
fan power it would always run to its upper bound. The TEA adds pump/fan
power per hour and picks the cheapest m_cw level for its prices, so the map
itself stays independent of prices and of the cooling-tower model.

Decision variables per map point
--------------------------------
- T11 <= T_geo     return-flow admixture (3-way mixing valve)
- m11              desorber flow, kept near its design value by a small
                   penalty (real mixing circuits run a constant-flow pump;
                   the COP is nearly flat along (T11, m11) anyway)

Chilled water T17 -> T18 is fixed (m17 follows), the cycle is closed via
Qevap_spec_kW (Qeva mode), so the solution flow m1 follows from the solver,
i.e. it acts like an ideally controlled (variable-speed) solution pump.

Cooling-water routing (cfg.routing)
-----------------------------------
"series_absorber_to_condenser" (default): T13 -> absorber -> T14 = T15 ->
condenser -> T16, one stream m13 = m15. "parallel": T13 = T15, m13 and m15
both scaled with the same m_cw ratio.

Numerics
--------
1) Start point: 1-D continuation scan over T11 at design m11 (+ bounded
   refinement). The scan starts a little above the T11 optimum of an
   already computed neighbor point; if that finds nothing, it falls back to
   the full scan from T_geo downward.
2) Bounded Nelder-Mead over (T11, m11) in normalized variables.
Warm starts come from the already computed neighbors along all three axes.
Infeasible points are expensive (every scan probe fails), so the map uses
monotonicity of the two operating limits:
- warm side (T_reject >= design): maximum load (capacity). If a point is
  infeasible, so is every point with higher PLR, warmer T_reject or less
  cooling water; PLR is walked upward.
- cold side (T_reject < design): minimum load. With cold reject cooling the
  feasible T11 window narrows with decreasing load and closes; if a point is
  infeasible, so is every point with lower PLR or colder T_reject; PLR is
  walked downward.
Such points are recorded as "skipped" without solving. An infeasible point
only counts as one of these limits if a point before it in walk direction was
feasible (column "limit"); otherwise it is "unknown" and skips nothing.
Before a point is declared infeasible, a finer T11 scan around the neighbors'
optima is tried (narrow feasible windows at cold reject cooling).
Solutions on the high-circulation branch of the Qeva mode (FR > FR_max) are
rejected.

Resume: the CSV also stores the internal solver vector (x_* columns). If the
CSV exists with the same columns, a restart takes over all points already in
it and only computes the missing ones.
Every solve_ac call runs in a worker process with a wall-clock limit.
Results are appended to the CSV after every point, so an interrupted run
keeps everything computed so far.

Known limitations
-----------------
- UA values are constant (no UA(m) dependence).
- Crystallization is only RECORDED (margin dw = w_crit(T) - w, minimum over
  the checked states); negative = unrealistic point. Enforcing it is opt-in
  via dw_cryst_min.
- No minimum pinch constraint; the smallest pinch and its component are
  recorded (at fixed UA, small pinches at part load are consistent).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

import Thermodynamic_Properties.libr_props as lp
from Models.AC_UA_LMTD import (
    ACInputs,
    ACResult,
    PRIMARY_VARIABLE_NAMES,
    primary_temperatures_C_to_K,
    solve_ac,
)

RESIDUAL_TOL = 1.0e-6

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PartLoadMapConfig:
    # --- Design point (defines the machine: UA values + design flows) ------
    T_geo_C: float = 100.0          # geothermal supply = upper bound for T11
    T12_design_C: float = 75.0
    T_reject_design_C: float = 30.0
    T14_design_C: float = 33.0      # only used for parallel routing
    T16_design_C: float = 36.0      # parallel: condenser outlet; series: final cooling-water outlet
    routing: str = "series_absorber_to_condenser"   # or "parallel"
    T17_C: float = 12.0
    T18_C: float = 6.0
    Q_nom_kW: float = 500.0

    UA_des: float = 150.28
    UA_abs: float = 103.76
    UA_cond: float = 116.93
    UA_evap: float = 123.37
    UA_shex: float = 7.648
    cp_w_kJkgK: float = 4.2

    # Initial vector for the design solve [T8, T10, x4, x1, T3, T5] (°C / -)
    x0_design: Tuple[float, ...] = (40.06, 1.39, 0.262, 0.243, 63.61, 53.11)

    # --- Map grid ------------------------------------------------------------
    plr_values: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
    T_reject_values_C: Tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
    m_cw_ratio_values: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.25, 1.5)

    # --- Decision-variable bounds ---------------------------------------------
    T11_min_C: float = 20.0
    m11_ratio_min: float = 0.3      # m11 relative to its design value
    m11_ratio_max: float = 1.5

    # --- Start-point scan (T11 at design m11) ---------------------------------
    T11_step_C: float = 2.5
    scan_margin_C: float = 15.0     # neighbor start: begin this far above the neighbors' highest T11 optimum
    max_consecutive_failures: int = 2
    refine_maxiter: int = 6
    # Cold side only: before a point is declared infeasible, a finer scan around
    # the neighbors' T11 optima. With cold reject cooling the feasible T11
    # window is only a few K wide at low load and the coarse step can jump over it.
    fine_step_C: float = 0.75
    fine_span_C: float = 12.0       # fine scan covers [min - span, max + span/2] of the neighbors' T11

    # --- Optimization over (T11, m11) -------------------------------------------
    nm_maxfev: int = 60
    nm_initial_step: float = 0.1    # initial simplex size in normalized variables
    nm_xatol: float = 5.0e-3
    nm_fatol: float = 1.0e-4
    # Tie-breaker: COP is nearly flat along (T11, m11) combinations that give
    # the desorber the same heat, so m11 would wander randomly. The objective
    # is -COP + m11_penalty * (m11 / m11_design - 1)^2; the reported COP is
    # unpenalized. 0.0 switches it off.
    m11_penalty: float = 2.0e-3
    # Polish step: if the optimum moved m11 away from its design value, T11 is
    # re-optimized once more at design m11. If that COP is at most this much
    # below, the design-m11 solution is kept (m11 only deviates where it truly
    # helps, e.g. at T11 = T_geo). None switches the step off.
    m11_design_tolerance: Optional[float] = 1.0e-3
    polish_span_C: float = 8.0
    polish_maxiter: int = 12

    # --- Validity ---------------------------------------------------------------
    dw_cryst_min: Optional[float] = None   # e.g. 0.005 to enforce a crystallization margin
    # Qeva mode has a second, high-circulation solution branch (same Q_evap with
    # a much larger solution flow and a clearly lower COP); a real controller
    # never operates there. Solutions with FR = m1 / m7 above this are rejected.
    # Physical points of the default machine stay at FR <= ~10.5 even at PLR 0.2.
    FR_max: Optional[float] = 12.0
    probe_max_nfev: int = 80   # converging warm starts need ~10; failures are what costs time
    solve_timeout_s: Optional[float] = 60.0   # per solve_ac call (separate process); None = in-process

    # --- Output -----------------------------------------------------------------
    # Base name of CSV, config file and plots. Use a NEW name for every changed
    # configuration (machine, design point, bounds, solver settings); resuming
    # into a file computed with different settings is refused. Only the grid
    # (GRID_FIELDS) may change between runs with the same name.
    output_name: str = "AC_part_load_map"


# Fields that may change between runs writing into the same CSV: every map
# point depends only on its own coordinates, so extending/refining the grid
# keeps all existing points valid. Everything else changes the results.
GRID_FIELDS = ("plr_values", "T_reject_values_C", "m_cw_ratio_values", "output_name")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NominalDesign:
    m11_kg_s: float
    m13_kg_s: float
    m15_kg_s: float
    COP: float
    x: np.ndarray


@dataclass(frozen=True)
class Setpoints:
    T11_C: float
    m11_kg_s: float
    T_reject_C: float
    m13_kg_s: float
    m15_kg_s: float


@dataclass(frozen=True)
class OperatingPoint:
    setpoints: Setpoints
    COP: float
    Q_des_kW: float
    Q_reject_kW: float     # Q_abs + Q_cond, heat to the cooling tower
    T12_C: float
    T14_C: float
    T16_C: float
    m1_kg_s: float
    FR: float
    m_geo_kg_s: float      # geothermal flow drawn from the source (mixing balance)
    dw_cryst: float        # crystallization margin, > 0 is safe
    pinch_min_K: float
    pinch_min_at: str      # component with the smallest pinch
    x: np.ndarray = field(repr=False)


@dataclass
class MapPoint:
    plr: float
    T_reject_C: float
    m_cw_ratio: float
    cop_start: Optional[float]          # T11-only optimum at design m11
    optimum: Optional[OperatingPoint]   # (T11, m11) optimized
    n_solves: int
    runtime_s: float
    status: str = "ok"                  # "ok", "infeasible" or "skipped" (an easier neighbor was infeasible)
    limit: str = ""                     # for infeasible/skipped: "capacity" (max load) or "min_load"


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def _x_from_result(result: ACResult) -> np.ndarray:
    return np.array([result.primary_variables[k] for k in PRIMARY_VARIABLE_NAMES], dtype=float)


def _crystallization_margin(result: ACResult) -> float:
    """Minimum of w_crit(T) - w over the states the model checks itself."""
    candidates = [(result.flash_outputs["T6_K"], result.flash_outputs["w6_LiBr"])]
    for label in ("1", "2", "3", "4"):
        s = result.states[label]
        candidates.append((s["T_K"], s["w_LiBr"]))
    margin = math.inf
    for T_K, w in candidates:
        if w <= 0.57:          # below the range of the crystallization correlation
            continue
        try:
            margin = min(margin, lp.crystallization_limit("T", T_K) - w)
        except Exception:
            return -math.inf
    return margin


def _is_series(cfg: PartLoadMapConfig) -> bool:
    if cfg.routing not in ("series_absorber_to_condenser", "parallel"):
        raise ValueError("routing must be 'series_absorber_to_condenser' or 'parallel'.")
    return cfg.routing == "series_absorber_to_condenser"


def solve_design(cfg: PartLoadMapConfig) -> NominalDesign:
    common = dict(
        T_11_C=cfg.T_geo_C, T_13_C=cfg.T_reject_design_C, T_17_C=cfg.T17_C,
        UA_des=cfg.UA_des, UA_abs=cfg.UA_abs, UA_cond=cfg.UA_cond, UA_evap=cfg.UA_evap,
        UA_shex=cfg.UA_shex, shex_model="UA", cp_w_kJkgK=cfg.cp_w_kJkgK,
        absorber_condenser_routing_mode=cfg.routing,
        cycle_scale_spec_mode="Qeva", Qevap_spec_kW=cfg.Q_nom_kW,
        desorber_spec_mode="T12", T12_spec_C=cfg.T12_design_C,
        evaporator_spec_mode="T18", T18_spec_C=cfg.T18_C,
    )
    warm = {"x": primary_temperatures_C_to_K(np.array(cfg.x0_design, dtype=float))}

    def solve(inputs: ACInputs) -> ACResult:
        result = solve_ac(inputs, x0=warm["x"])
        if not result.solve_info.success or result.solve_info.scaled_residual_norm > RESIDUAL_TOL:
            raise RuntimeError(f"Design point did not converge: {result.solve_info.message}")
        warm["x"] = _x_from_result(result)
        return result

    if _is_series(cfg):
        # One cooling-water stream (m13 = m15): choose it so that the final
        # outlet after the condenser reaches T16_design_C.
        def solve_series(m_cw: float) -> ACResult:
            return solve(ACInputs(**common, T_15_C=None,
                                  absorber_spec_mode="m13", m13_spec=m_cw,
                                  condenser_spec_mode="m15", m15_spec=m_cw))

        def T16_error(m_cw: float) -> float:
            return solve_series(m_cw).states["16"]["T_K"] - 273.15 - cfg.T16_design_C

        # Energy balance estimate (Q_abs + Q_cond ~ Q_evap (1 + 1/COP)); walk
        # down from a large flow with warm starts -- too small flows have no
        # solution, so a fixed lower bracket end is not safe.
        m_est = cfg.Q_nom_kW * (1.0 + 1.0 / 0.75) / (
            cfg.cp_w_kJkgK * (cfg.T16_design_C - cfg.T_reject_design_C))
        m_hi, err_hi = 2.0 * m_est, T16_error(2.0 * m_est)
        while err_hi < 0.0:
            m_lo = 0.9 * m_hi
            err_lo = T16_error(m_lo)
            if err_lo >= 0.0:
                break
            m_hi, err_hi = m_lo, err_lo
        else:
            raise RuntimeError("Series design: T16_design_C already exceeded at the largest flow.")
        m_cw = brentq(T16_error, m_lo, m_hi, xtol=1.0e-4)
        result = solve_series(m_cw)
    else:
        result = solve(ACInputs(**common, T_15_C=cfg.T_reject_design_C,
                                absorber_spec_mode="T14", T14_spec_C=cfg.T14_design_C,
                                condenser_spec_mode="T16", T16_spec_C=cfg.T16_design_C))
    d = result.diagnostics
    return NominalDesign(
        m11_kg_s=d["m11_kg_s"], m13_kg_s=d["m13_kg_s"], m15_kg_s=d["m15_kg_s"],
        COP=result.kpis["COP"], x=_x_from_result(result),
    )


def _build_operating_inputs(cfg: PartLoadMapConfig, plr: float, sp: Setpoints) -> ACInputs:
    return ACInputs(
        T_11_C=sp.T11_C, T_13_C=sp.T_reject_C,
        T_15_C=None if _is_series(cfg) else sp.T_reject_C, T_17_C=cfg.T17_C,
        UA_des=cfg.UA_des, UA_abs=cfg.UA_abs, UA_cond=cfg.UA_cond, UA_evap=cfg.UA_evap,
        UA_shex=cfg.UA_shex, shex_model="UA", cp_w_kJkgK=cfg.cp_w_kJkgK,
        absorber_condenser_routing_mode=cfg.routing,
        cycle_scale_spec_mode="Qeva", Qevap_spec_kW=plr * cfg.Q_nom_kW,
        desorber_spec_mode="m11", m11_spec=sp.m11_kg_s,
        absorber_spec_mode="m13", m13_spec=sp.m13_kg_s,
        condenser_spec_mode="m15", m15_spec=sp.m15_kg_s,
        evaporator_spec_mode="T18", T18_spec_C=cfg.T18_C,
        max_nfev=cfg.probe_max_nfev,
    )


def _solver_worker(conn) -> None:
    """Runs in a separate process: receives (inputs, x0), returns the result."""
    while True:
        message = conn.recv()
        if message is None:
            return
        inputs, x0 = message
        try:
            conn.send(("ok", solve_ac(inputs, x0=x0)))
        except Exception as exc:
            conn.send(("error", repr(exc)))


class TimedSolver:
    """solve_ac with a wall-clock limit.

    solve_ac cannot be interrupted from outside, so it runs in a worker
    process; on timeout the worker is killed and restarted, and the call
    raises TimeoutError. With timeout_s=None it calls solve_ac directly.
    """

    def __init__(self, timeout_s: Optional[float]):
        self.timeout_s = timeout_s
        self.n_timeouts = 0
        self.n_worker_restarts = 0
        self._proc = None
        if timeout_s is not None:
            self._start()

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=_solver_worker, args=(child_conn,), daemon=True)
        self._proc.start()

    def _restart(self) -> None:
        if self._proc.is_alive():
            self._proc.kill()
        self._proc.join()
        self._start()

    def __call__(self, inputs: ACInputs, x0: np.ndarray) -> ACResult:
        if self.timeout_s is None:
            return solve_ac(inputs, x0=x0)
        if not self._proc.is_alive():
            # Worker died (crash or killed from outside): without a restart
            # every following call would fail instantly and all remaining
            # points would be written as infeasible.
            self.n_worker_restarts += 1
            self._restart()
        try:
            self._conn.send((inputs, x0))
            if not self._conn.poll(self.timeout_s):
                self.n_timeouts += 1
                self._restart()
                raise TimeoutError(f"solve_ac exceeded {self.timeout_s:.0f} s")
            status, payload = self._conn.recv()
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            self.n_worker_restarts += 1
            self._restart()
            # One retry on the fresh worker; if that fails too, the error propagates.
            self._conn.send((inputs, x0))
            if not self._conn.poll(self.timeout_s):
                self.n_timeouts += 1
                self._restart()
                raise TimeoutError(f"solve_ac exceeded {self.timeout_s:.0f} s")
            status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(payload)
        return payload

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._conn.send(None)
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.kill()


class Evaluator:
    """Solves one operating point, trying several warm starts; counts solves."""

    def __init__(self, cfg: PartLoadMapConfig):
        self.cfg = cfg
        self.n_solves = 0
        self.solver = TimedSolver(cfg.solve_timeout_s)
        # Solver vector of the last call's converged solution that was rejected
        # only for lying on the high-circulation branch. Not a result, but a
        # usable warm start: a scan starting at T11 = T_geo at low load first
        # meets that branch and continues from it into the physical one.
        self.hint_x: Optional[np.ndarray] = None

    def _is_valid(self, result: ACResult, dw: float) -> bool:
        info = result.solve_info
        if not info.success or not info.final_point_evaluable:
            return False
        if info.scaled_residual_norm > RESIDUAL_TOL:
            return False
        hard_checks = {k: v for k, v in result.checks.items()
                       if k != "crystallization_safe_all_checked_states"}
        if not all(hard_checks.values()):
            return False
        if self.cfg.dw_cryst_min is not None and dw < self.cfg.dw_cryst_min:
            return False
        return True

    def _on_high_fr_branch(self, result: ACResult) -> bool:
        return self.cfg.FR_max is not None and not result.kpis["FR"] <= self.cfg.FR_max

    def __call__(
        self, plr: float, sp: Setpoints, x0_candidates: Sequence[np.ndarray],
    ) -> Optional[OperatingPoint]:
        self.hint_x = None
        try:
            inputs = _build_operating_inputs(self.cfg, plr, sp)
        except ValueError:
            return None
        for x0 in x0_candidates:
            self.n_solves += 1
            try:
                result = self.solver(inputs, x0)
            except Exception:
                continue
            if not result.flash_outputs:     # evaluation failed at the final point
                continue
            dw = _crystallization_margin(result)
            if not self._is_valid(result, dw):
                continue
            if self._on_high_fr_branch(result):
                self.hint_x = _x_from_result(result)
                continue
            T12_C = result.states["12"]["T_K"] - 273.15
            Q_des = result.heat_flows_kW["Q_des"]
            pinches = {k[len("Pinch_"):]: v for k, v in result.lmtd_K.items() if k.startswith("Pinch_")}
            pinch_at = min(pinches, key=pinches.get) if pinches else ""
            return OperatingPoint(
                setpoints=sp, COP=result.kpis["COP"], Q_des_kW=Q_des,
                Q_reject_kW=result.heat_flows_kW["Q_abs"] + result.heat_flows_kW["Q_cond"],
                T12_C=T12_C,
                T14_C=result.states["14"]["T_K"] - 273.15,
                T16_C=result.states["16"]["T_K"] - 273.15,
                m1_kg_s=result.states["1"]["m_kg_s"], FR=result.kpis["FR"],
                m_geo_kg_s=Q_des / (self.cfg.cp_w_kJkgK * (self.cfg.T_geo_C - T12_C)),
                dw_cryst=dw, pinch_min_K=pinches[pinch_at] if pinches else float("nan"),
                pinch_min_at=pinch_at, x=_x_from_result(result),
            )
        return None


# ---------------------------------------------------------------------------
# Step 1: start point -- 1-D T11 scan at design m11
# ---------------------------------------------------------------------------


def _scan_and_refine(
    solve_at: Callable[[float, Sequence[np.ndarray]], Optional[OperatingPoint]],
    grid: Sequence[float],
    x0_candidates: Sequence[np.ndarray],
    max_consecutive_failures: int,
    refine_maxiter: int,
    hint: Optional[Callable[[], Optional[np.ndarray]]] = None,
) -> Optional[OperatingPoint]:
    """`hint` returns the solver vector of a converged but rejected solution
    of the last probe (or None); before the first success it is used as warm
    start for the next probe, so the scan can walk along a rejected branch."""
    scan: List[Tuple[int, OperatingPoint]] = []
    x_chain = list(x0_candidates)
    failures = 0
    for i, p in enumerate(grid):
        op = solve_at(p, x_chain)
        if op is None:
            failures += 1
            if scan and failures >= max_consecutive_failures:
                break
            x_hint = hint() if (hint is not None and not scan) else None
            if x_hint is not None:
                x_chain = [x_hint] + list(x0_candidates)[:1]
            continue
        failures = 0
        scan.append((i, op))
        # Once on the branch, only continue from the neighbor: retrying all
        # candidates on every failing probe is what makes scans expensive.
        x_chain = [op.x]
    if not scan:
        return None

    i_best, best = max(scan, key=lambda item: item[1].COP)
    lo = grid[max(i_best - 1, 0)]
    hi = grid[min(i_best + 1, len(grid) - 1)]
    lo, hi = min(lo, hi), max(lo, hi)
    if refine_maxiter > 0 and hi > lo:
        cache: List[OperatingPoint] = []

        def objective(p: float) -> float:
            op = solve_at(p, [best.x])
            if op is None:
                return 1.0          # COP is always > 0 -> any feasible point wins
            cache.append(op)
            return -op.COP

        minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                        options={"maxiter": refine_maxiter, "xatol": 1.0e-2 * (hi - lo)})
        if cache:
            refined = max(cache, key=lambda op: op.COP)
            if refined.COP > best.COP:
                best = refined
    return best


# ---------------------------------------------------------------------------
# Step 2: optimization over (T11, m11)
# ---------------------------------------------------------------------------


def optimize_point(
    cfg: PartLoadMapConfig, nom: NominalDesign, evaluate: Evaluator,
    plr: float, start: OperatingPoint,
) -> OperatingPoint:
    lo = np.array([cfg.T11_min_C, cfg.m11_ratio_min * nom.m11_kg_s])
    hi = np.array([cfg.T_geo_C, cfg.m11_ratio_max * nom.m11_kg_s])
    base = start.setpoints

    def to_setpoints(u: np.ndarray) -> Setpoints:
        T11, m11 = lo + np.clip(u, 0.0, 1.0) * (hi - lo)
        return Setpoints(float(T11), float(m11), base.T_reject_C, base.m13_kg_s, base.m15_kg_s)

    def score(op: OperatingPoint) -> float:
        return -op.COP + cfg.m11_penalty * (op.setpoints.m11_kg_s / nom.m11_kg_s - 1.0) ** 2

    state = {"best": start, "x": start.x}

    def objective(u: np.ndarray) -> float:
        op = evaluate(plr, to_setpoints(u), [state["x"], state["best"].x])
        if op is None:
            return 1.0
        state["x"] = op.x
        if score(op) < score(state["best"]):
            state["best"] = op
        return score(op)

    u0 = np.clip((np.array([base.T11_C, base.m11_kg_s]) - lo) / (hi - lo), 0.0, 1.0)
    # Initial simplex: step towards the interior so no vertex leaves [0, 1].
    simplex = [u0]
    for k in range(len(u0)):
        u = u0.copy()
        u[k] += cfg.nm_initial_step if u[k] + cfg.nm_initial_step <= 1.0 else -cfg.nm_initial_step
        simplex.append(u)
    minimize(
        objective, u0, method="Nelder-Mead", bounds=[(0.0, 1.0)] * len(u0),
        options={"maxfev": cfg.nm_maxfev, "xatol": cfg.nm_xatol, "fatol": cfg.nm_fatol,
                 "initial_simplex": np.array(simplex)},
    )
    best = state["best"]

    # Polish: a poor start can make Nelder-Mead reach the (flat) optimum via a
    # detour in m11, leaving m11 at an arbitrary value. Re-optimize T11 at
    # design m11 and prefer that solution if it is (almost) as good.
    if cfg.m11_design_tolerance is None or abs(best.setpoints.m11_kg_s / nom.m11_kg_s - 1.0) < 0.01:
        return best
    T_lo = max(cfg.T11_min_C, best.setpoints.T11_C - cfg.polish_span_C)
    T_hi = min(cfg.T_geo_C, best.setpoints.T11_C + cfg.polish_span_C)
    polished: List[OperatingPoint] = []

    def at_design_m11(T11: float) -> float:
        op = evaluate(plr, Setpoints(float(T11), nom.m11_kg_s, base.T_reject_C,
                                     base.m13_kg_s, base.m15_kg_s),
                      [polished[-1].x if polished else best.x, best.x, start.x])
        if op is None:
            return 1.0
        polished.append(op)
        return -op.COP

    minimize_scalar(at_design_m11, bounds=(T_lo, T_hi), method="bounded",
                    options={"maxiter": cfg.polish_maxiter, "xatol": 0.05})
    if polished:
        candidate = max(polished, key=lambda op: op.COP)
        if candidate.COP >= best.COP - cfg.m11_design_tolerance:
            return candidate
    return best


# ---------------------------------------------------------------------------
# Map computation
# ---------------------------------------------------------------------------


def _outward_order(values: Sequence[float], center: float) -> List[int]:
    """Indices sorted by distance from `center`, so every point (except the
    first) has an already computed neighbor closer to the center."""
    return sorted(range(len(values)), key=lambda i: (abs(values[i] - center), values[i]))


def _adjacent(values: Sequence[float], idx: int, direction: int) -> Optional[int]:
    """Index of the next larger (direction=+1) or smaller (-1) value, or None."""
    others = [i for i in range(len(values)) if direction * (values[i] - values[idx]) > 0]
    return min(others, key=lambda i: abs(values[i] - values[idx])) if others else None


def compute_part_load_map(
    cfg: PartLoadMapConfig, *, csv_path: Optional[Path] = None, resume: bool = True,
) -> Tuple[NominalDesign, List[MapPoint]]:
    """Computes the map. With resume=True and an existing CSV (same columns),
    all points already in the CSV are taken over and only missing ones are
    computed -- an interrupted run can simply be restarted."""
    nom = solve_design(cfg)
    print(
        f"Design: Q_nom = {cfg.Q_nom_kW:.0f} kW, T_reject = {cfg.T_reject_design_C:.1f} °C, "
        f"T11 = {cfg.T_geo_C:.1f} °C -> COP = {nom.COP:.4f}, m11 = {nom.m11_kg_s:.3f}, "
        f"m13 = {nom.m13_kg_s:.3f}, m15 = {nom.m15_kg_s:.3f} kg/s"
    )
    evaluate = Evaluator(cfg)

    T_vals, C_vals, P_vals = cfg.T_reject_values_C, cfg.m_cw_ratio_values, cfg.plr_values
    done: Dict[Tuple[int, int, int], OperatingPoint] = {}   # (i_T, i_cw, i_P) -> optimum
    # Infeasible/skipped points per limit type -- the two limits propagate in
    # opposite directions and must not be mixed up.
    failed: Dict[str, set] = {"capacity": set(), "min_load": set()}
    points: List[MapPoint] = []

    previous = _read_csv(csv_path) if (resume and csv_path is not None) else None
    if previous is None:
        if csv_path is not None:
            _write_csv_header(csv_path)
            _write_config(cfg, _config_path(csv_path))
    else:
        _check_config(cfg, _config_path(csv_path))
        print(f"Resuming from {csv_path}: {len(previous)} points already computed")
    n_total = len(T_vals) * len(C_vals) * len(P_vals)
    t_run = time.time()

    T_order = _outward_order(T_vals, cfg.T_reject_design_C)
    C_order = _outward_order(C_vals, 1.0)

    # Resume: make ALL points taken over from the CSV known up front, so they
    # serve as neighbors/warm starts regardless of the walk order.
    if previous is not None:
        for (plr_p, T_p, r_p), point in previous.items():
            if plr_p in P_vals and T_p in T_vals and r_p in C_vals:
                key = (T_vals.index(T_p), C_vals.index(r_p), P_vals.index(plr_p))
                if point.optimum is not None:
                    done[key] = point.optimum
                elif point.limit in failed:
                    failed[point.limit].add(key)

    def seed_by_load_path(plr_target: float, T_rej: float, r_cw: float) -> Optional[np.ndarray]:
        """Warm start for a point without computed neighbors: lower the load
        step by step from full load (design start, T11 = T_geo) to the target,
        re-scanning T11 at every step. A direct jump from full load to e.g.
        PLR 0.2 does not converge onto the physical branch."""
        x_seed, T_prev = nom.x, cfg.T_geo_C
        for plr_step in np.arange(1.0, plr_target - 1.0e-9, -0.2)[1:].tolist() + [plr_target]:
            grid = [T for T in np.arange(cfg.T_geo_C, cfg.T11_min_C - 1.0e-9, -cfg.T11_step_C)
                    if T <= min(cfg.T_geo_C, T_prev + cfg.scan_margin_C)]
            op = _scan_and_refine(
                lambda T11, xs: evaluate(plr_step, Setpoints(T11, nom.m11_kg_s, T_rej, r_cw * nom.m13_kg_s,
                                                             r_cw * nom.m15_kg_s), xs),
                grid, [x_seed, nom.x], cfg.max_consecutive_failures, 0, hint=lambda: evaluate.hint_x)
            if op is None:
                return None
            x_seed, T_prev = op.x, op.setpoints.T11_C
        return x_seed

    def report(point: MapPoint, msg: str) -> None:
        print(f"[{len(points):3d}/{n_total}, {(time.time() - t_run) / 60:5.0f} min] "
              f"T_rej={point.T_reject_C:4.1f} °C m_cw={point.m_cw_ratio:4.2f}x PLR={point.plr:4.2f} "
              f"({point.runtime_s:5.1f} s)  {msg}", flush=True)

    try:
        for i in T_order:
            T_rej = T_vals[i]
            # Two limits with opposite direction:
            # - warm side (T_reject >= design): MAXIMUM load (capacity). Walk the
            #   load upward; a point is skipped if the lower load, the cooler
            #   T_reject or the larger cooling-water flow hit the capacity limit.
            # - cold side (T_reject < design): MINIMUM load -- the feasible T11
            #   window closes at low load (drive too strong even at the lowest
            #   feasible T11). Walk the load downward; a point is skipped if
            #   the higher load or the warmer T_reject hit the minimum-load limit.
            # An infeasible point only counts as such a limit if a point before
            # it in walk direction (same T_reject, m_cw) was feasible; otherwise
            # it is "unknown" and nothing is skipped because of it.
            capacity_side = T_rej >= cfg.T_reject_design_C
            limit_type = "capacity" if capacity_side else "min_load"
            P_order = sorted(range(len(P_vals)), key=lambda j: P_vals[j] if capacity_side else -P_vals[j])
            for k in C_order:
                for n_p, j in enumerate(P_order):
                    t0 = time.time()
                    n0 = evaluate.n_solves
                    timeouts0 = evaluate.solver.n_timeouts
                    plr, r_cw = P_vals[j], C_vals[k]

                    if previous is not None and (plr, T_rej, r_cw) in previous:
                        point = previous[(plr, T_rej, r_cw)]
                        points.append(point)
                        if point.optimum is not None:
                            done[(i, k, j)] = point.optimum
                        elif point.limit in failed:
                            failed[point.limit].add((i, k, j))
                        continue

                    if capacity_side:
                        easier = [(i, k, _adjacent(P_vals, j, -1))]
                        if T_rej > cfg.T_reject_design_C:
                            easier.append((_adjacent(T_vals, i, -1), k, j))
                        if r_cw < 1.0:
                            easier.append((i, _adjacent(C_vals, k, +1), j))
                    else:
                        easier = [(i, k, _adjacent(P_vals, j, +1)), (_adjacent(T_vals, i, +1), k, j)]
                    if any(key in failed[limit_type] for key in easier):
                        failed[limit_type].add((i, k, j))
                        point = MapPoint(plr, T_rej, r_cw, None, None, 0, 0.0,
                                         status="skipped", limit=limit_type)
                        points.append(point)
                        if csv_path is not None:
                            _append_csv_row(csv_path, point)
                        report(point, f"skipped ({limit_type.replace('_', ' ')} limit of an easier neighbor)")
                        continue

                    neighbor_keys = [(i, k, _adjacent(P_vals, j, -1)), (i, k, _adjacent(P_vals, j, +1)),
                                     (i, _adjacent(C_vals, k, -1), j), (i, _adjacent(C_vals, k, +1), j),
                                     (_adjacent(T_vals, i, -1), k, j), (_adjacent(T_vals, i, +1), k, j)]
                    neighbors = [done[key] for key in neighbor_keys if key in done]
                    cands = [op.x for op in neighbors][:3] + [nom.x]
                    if not neighbors and plr < 1.0:
                        x_seed = seed_by_load_path(plr, T_rej, r_cw)
                        if x_seed is not None:
                            cands = [x_seed] + cands

                    def at_T11(T11: float, xs: Sequence[np.ndarray]) -> Optional[OperatingPoint]:
                        return evaluate(plr, Setpoints(T11, nom.m11_kg_s, T_rej,
                                                       r_cw * nom.m13_kg_s, r_cw * nom.m15_kg_s), xs)

                    def scan(grid: Sequence[float]) -> Optional[OperatingPoint]:
                        return _scan_and_refine(at_T11, grid, cands, cfg.max_consecutive_failures,
                                                cfg.refine_maxiter, hint=lambda: evaluate.hint_x)

                    full_grid = list(np.arange(cfg.T_geo_C, cfg.T11_min_C - 1.0e-9, -cfg.T11_step_C))
                    start = None
                    if neighbors:
                        T_nb = [op.setpoints.T11_C for op in neighbors]
                        T_hi = min(cfg.T_geo_C, max(T_nb) + cfg.scan_margin_C)
                        start = scan([T for T in full_grid if T <= T_hi] or full_grid)
                    if start is None:
                        start = scan(full_grid)
                    # Fine scan only on the cold side: narrow T11 windows occur
                    # near the minimum-load limit; at the capacity limit it only
                    # costs time (a failing fine scan took ~30 min there).
                    if start is None and neighbors and not capacity_side:
                        fine_hi = min(cfg.T_geo_C, max(T_nb) + 0.5 * cfg.fine_span_C)
                        fine_lo = max(cfg.T11_min_C, min(T_nb) - cfg.fine_span_C)
                        start = scan(list(np.arange(fine_hi, fine_lo - 1.0e-9, -cfg.fine_step_C)))
                    optimum = optimize_point(cfg, nom, evaluate, plr, start) if start else None

                    limit = ""
                    if optimum is not None:
                        done[(i, k, j)] = optimum
                    else:
                        walked_before = [(i, k, jj) for jj in P_order[:n_p]]
                        if any(key in done for key in walked_before):
                            limit = limit_type
                            failed[limit_type].add((i, k, j))
                        else:
                            limit = "unknown"

                    point = MapPoint(plr, T_rej, r_cw, start.COP if start else None, optimum,
                                     evaluate.n_solves - n0, time.time() - t0,
                                     status="ok" if optimum else "infeasible", limit=limit)
                    points.append(point)
                    if csv_path is not None:
                        _append_csv_row(csv_path, point)

                    if optimum is None:
                        msg = f"infeasible ({limit.replace('_', ' ')} limit)"
                    else:
                        s = optimum.setpoints
                        msg = (f"COP={optimum.COP:.3f} | T11={s.T11_C:5.1f} T12={optimum.T12_C:5.1f} "
                               f"m11={s.m11_kg_s / nom.m11_kg_s:.2f}x T16={optimum.T16_C:5.1f} "
                               f"| pinch {optimum.pinch_min_K:.2f} K ({optimum.pinch_min_at}) "
                               f"dw={optimum.dw_cryst:.3f}")
                    n_timeouts = evaluate.solver.n_timeouts - timeouts0
                    if n_timeouts:
                        msg += f" | {n_timeouts} timeout(s)"
                    report(point, msg)
    finally:
        evaluate.solver.close()
    print(f"Solver calls: {evaluate.n_solves}, timeouts: {evaluate.solver.n_timeouts}, "
          f"worker restarts: {evaluate.solver.n_worker_restarts}")
    return nom, points


# ---------------------------------------------------------------------------
# Output: table, CSV, plots
# ---------------------------------------------------------------------------

# x_* = internal solver vector (needed as warm start when a run is resumed)
X_FIELDS = tuple(f"x_{name}" for name in PRIMARY_VARIABLE_NAMES)
CSV_FIELDS = ("plr", "T_reject_C", "m_cw_ratio", "status", "limit", "feasible", "COP_start", "COP",
              "T11_C", "T12_C", "m11_kg_s", "m13_kg_s", "m15_kg_s", "T_reject_in_C", "T14_C", "T16_C",
              "Q_evap_kW", "Q_des_kW", "Q_reject_kW", "m1_kg_s", "FR", "m_geo_kg_s", "dw_cryst",
              "pinch_min_K", "pinch_min_at", "n_solves", "runtime_s") + X_FIELDS


def _csv_row(p: MapPoint) -> Dict[str, object]:
    row: Dict[str, object] = {k: float("nan") for k in CSV_FIELDS}
    row.update({"plr": p.plr, "T_reject_C": p.T_reject_C, "m_cw_ratio": p.m_cw_ratio,
                "status": p.status, "limit": p.limit, "feasible": p.optimum is not None,
                "n_solves": p.n_solves, "runtime_s": round(p.runtime_s, 1),
                "COP_start": p.cop_start if p.cop_start is not None else float("nan")})
    o = p.optimum
    if o is not None:
        s = o.setpoints
        row.update({"COP": o.COP, "T11_C": s.T11_C, "T12_C": o.T12_C, "m11_kg_s": s.m11_kg_s,
                    "m13_kg_s": s.m13_kg_s, "m15_kg_s": s.m15_kg_s, "T_reject_in_C": s.T_reject_C,
                    "T14_C": o.T14_C, "T16_C": o.T16_C, "Q_evap_kW": o.COP * o.Q_des_kW,
                    "Q_des_kW": o.Q_des_kW, "Q_reject_kW": o.Q_reject_kW, "m1_kg_s": o.m1_kg_s,
                    "FR": o.FR, "m_geo_kg_s": o.m_geo_kg_s, "dw_cryst": o.dw_cryst,
                    "pinch_min_K": o.pinch_min_K, "pinch_min_at": o.pinch_min_at})
        row.update(dict(zip(X_FIELDS, o.x.tolist())))
    return row


def _config_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_config.json")


def _config_signature(cfg: PartLoadMapConfig) -> Dict[str, object]:
    """All settings that influence the results (everything but the grid)."""
    signature = {k: v for k, v in asdict(cfg).items() if k not in GRID_FIELDS}
    return json.loads(json.dumps(signature))     # tuples -> lists, as stored in the file


def _write_config(cfg: PartLoadMapConfig, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_config_signature(cfg), f, indent=2)


def _check_config(cfg: PartLoadMapConfig, path: Path) -> None:
    """Refuses to resume into a CSV computed with different settings."""
    if not path.exists():
        # CSV written before the config file existed: adopt the current settings.
        print(f"Warning: {path.name} not found -- assuming the existing CSV was computed "
              f"with the current settings and writing them now.")
        _write_config(cfg, path)
        return
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    current = _config_signature(cfg)
    changed = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
    if changed:
        details = "\n".join(f"  {k}: stored {stored.get(k)!r} -> now {current.get(k)!r}" for k in changed)
        raise RuntimeError(
            f"The existing map '{path.stem.removesuffix('_config')}' was computed with different "
            f"settings:\n{details}\nSet a new output_name in PartLoadMapConfig to compute this "
            f"configuration as a separate map (or delete the old CSV to overwrite it)."
        )


def _read_csv(path: Path) -> Optional[Dict[Tuple[float, float, float], MapPoint]]:
    """Reads a map CSV written by this script, keyed by (plr, T_reject, m_cw_ratio).
    Returns None if the file does not exist or has different columns (e.g.
    written by an older version) -- then the run starts from scratch."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            print(f"{path} has different columns -- not resumable, starting from scratch.")
            return None
        rows = list(reader)

    points: Dict[Tuple[float, float, float], MapPoint] = {}
    for r in rows:
        key = (float(r["plr"]), float(r["T_reject_C"]), float(r["m_cw_ratio"]))
        optimum = None
        if r["status"] == "ok":
            num = lambda name: float(r[name])
            optimum = OperatingPoint(
                setpoints=Setpoints(num("T11_C"), num("m11_kg_s"), num("T_reject_in_C"),
                                    num("m13_kg_s"), num("m15_kg_s")),
                COP=num("COP"), Q_des_kW=num("Q_des_kW"), Q_reject_kW=num("Q_reject_kW"),
                T12_C=num("T12_C"), T14_C=num("T14_C"), T16_C=num("T16_C"),
                m1_kg_s=num("m1_kg_s"), FR=num("FR"), m_geo_kg_s=num("m_geo_kg_s"),
                dw_cryst=num("dw_cryst"), pinch_min_K=num("pinch_min_K"),
                pinch_min_at=r["pinch_min_at"], x=np.array([num(name) for name in X_FIELDS]),
            )
        cop_start = float(r["COP_start"]) if r["COP_start"] not in ("", "nan") else None
        points[key] = MapPoint(key[0], key[1], key[2], cop_start, optimum, int(r["n_solves"]),
                               float(r["runtime_s"]), status=r["status"], limit=r["limit"])
    return points


def _write_csv_header(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_csv_row(path: Path, point: MapPoint) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(_csv_row(point))


def save_csv(points: Sequence[MapPoint], path: Path) -> None:
    _write_csv_header(path)
    for p in points:
        _append_csv_row(path, p)
    print(f"CSV saved: {path}")


def print_map_table(points: Sequence[MapPoint]) -> None:
    print("=" * 132)
    print(f"{'T_rej':>6} {'m_cw':>5} {'PLR':>5} {'COP':>6} {'T11':>6} {'T12':>6} {'m11':>6} "
          f"{'T14':>6} {'T16':>6} {'Q_des':>7} {'Q_rej':>7} {'m1':>6} {'FR':>5} {'m_geo':>6} "
          f"{'dw_cr':>7} {'pinch':>6} {'at':>6} {'time':>6}")
    print("-" * 132)
    for p in sorted(points, key=lambda q: (q.T_reject_C, q.m_cw_ratio, -q.plr)):
        o = p.optimum
        head = f"{p.T_reject_C:6.1f} {p.m_cw_ratio:5.2f} {p.plr:5.2f}"
        if o is None:
            print(f"{head} {p.status:>40} {p.runtime_s:>79.0f}")
            continue
        s = o.setpoints
        print(f"{head} {o.COP:6.3f} {s.T11_C:6.1f} {o.T12_C:6.1f} {s.m11_kg_s:6.2f} {o.T14_C:6.1f} "
              f"{o.T16_C:6.1f} {o.Q_des_kW:7.1f} {o.Q_reject_kW:7.1f} {o.m1_kg_s:6.2f} {o.FR:5.1f} "
              f"{o.m_geo_kg_s:6.2f} {o.dw_cryst:7.4f} {o.pinch_min_K:6.2f} {o.pinch_min_at:>6} "
              f"{p.runtime_s:6.0f}")
    print("=" * 132)


def _grid_values(points: Sequence[MapPoint], m_cw_ratio: float,
                 value: Callable[[OperatingPoint], float],
                 T_vals: Sequence[float], P_vals: Sequence[float]) -> np.ndarray:
    Z = np.full((len(P_vals), len(T_vals)), np.nan)
    for p in points:
        if p.optimum is not None and math.isclose(p.m_cw_ratio, m_cw_ratio):
            Z[P_vals.index(p.plr), T_vals.index(p.T_reject_C)] = value(p.optimum)
    return Z


def _annotated_map(ax, Z, T_vals, P_vals, *, cmap, vmin, vmax, fmt, title, fontsize=7.5):
    mesh = ax.pcolormesh(T_vals, P_vals, np.ma.masked_invalid(Z), shading="nearest",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    for a, plr in enumerate(P_vals):
        for b, T in enumerate(T_vals):
            if np.isnan(Z[a, b]):
                ax.text(T, plr, "x", ha="center", va="center", color="tab:red", fontsize=11)
            else:
                ax.text(T, plr, fmt.format(Z[a, b]), ha="center", va="center", fontsize=fontsize)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("T_reject [°C]")
    ax.set_ylabel("PLR [-]")
    return mesh


def _finite_range(arrays: Sequence[np.ndarray]) -> Tuple[float, float]:
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays]) if arrays else np.array([])
    if not finite.size:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1.0e-6 * max(1.0, abs(hi)):    # constant panel
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def plot_part_load_map(cfg: PartLoadMapConfig, nom: NominalDesign, points: Sequence[MapPoint], *,
                       save_dir: Optional[Path], show: bool = True):
    import matplotlib.pyplot as plt

    T_vals = sorted(set(cfg.T_reject_values_C))
    P_vals = sorted(set(cfg.plr_values))
    C_vals = sorted(set(cfg.m_cw_ratio_values))
    figs = []

    # 1) + 2) COP and T11 maps, one panel per cooling-water level (shared color scale)
    for name, value, cmap, fmt in (
        ("cop", lambda o: o.COP, "viridis", "{:.3f}"),
        ("T11", lambda o: o.setpoints.T11_C, "magma", "{:.0f}"),
    ):
        Zs = [_grid_values(points, r, value, T_vals, P_vals) for r in C_vals]
        lo, hi = _finite_range(Zs)
        fig, axes = plt.subplots(1, len(C_vals), figsize=(4.2 * len(C_vals), 4.6),
                                 sharey=True, squeeze=False)
        for ax, r, Z in zip(axes[0], C_vals, Zs):
            mesh = _annotated_map(ax, Z, T_vals, P_vals, cmap=cmap, vmin=lo, vmax=hi, fmt=fmt,
                                  title=f"m_cw = {r:.2f} x design")
        fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.9,
                     label="COP [-]" if name == "cop" else "T11 setpoint [°C]")
        fig.suptitle(("COP-optimal COP" if name == "cop" else "COP-optimal T11 setpoint")
                     + f" (return-flow admixture, T11 <= {cfg.T_geo_C:.0f} °C, {cfg.routing})")
        figs.append((name, fig))

    # 3) Detail at the design cooling-water flow
    r_ref = min(C_vals, key=lambda r: abs(r - 1.0))
    panels = [
        ("COP", lambda o: o.COP, "viridis", "{:.3f}"),
        ("T11 setpoint [°C]", lambda o: o.setpoints.T11_C, "magma", "{:.1f}"),
        ("T12 desorber outlet [°C]", lambda o: o.T12_C, "magma", "{:.1f}"),
        ("m11 / m11_design [-]", lambda o: o.setpoints.m11_kg_s / nom.m11_kg_s, "cividis", "{:.2f}"),
        ("T16 cooling-water outlet [°C]", lambda o: o.T16_C, "magma", "{:.1f}"),
        ("Q_reject to cooling tower [kW]", lambda o: o.Q_reject_kW, "cividis", "{:.0f}"),
        ("solution flow m1 [kg/s]", lambda o: o.m1_kg_s, "cividis", "{:.2f}"),
        ("geothermal flow m_geo [kg/s]", lambda o: o.m_geo_kg_s, "cividis", "{:.2f}"),
        ("crystallization margin dw [-]", lambda o: o.dw_cryst, "RdYlGn", "{:.3f}"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5))
    for ax, (label, value, cmap, fmt) in zip(axes.flat, panels):
        Z = _grid_values(points, r_ref, value, T_vals, P_vals)
        if label.startswith("crystallization"):
            finite = Z[np.isfinite(Z)]
            lim = max(float(np.abs(finite).max()) if finite.size else 0.0, 1.0e-3)
            lo, hi = -lim, lim
        else:
            lo, hi = _finite_range([Z])
        mesh = _annotated_map(ax, Z, T_vals, P_vals, cmap=cmap, vmin=lo, vmax=hi, fmt=fmt, title=label,
                              fontsize=7)
        fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.suptitle(f"COP-optimal operation at m_cw = {r_ref:.2f} x design ({cfg.routing})")
    fig.tight_layout()
    figs.append(("detail", fig))

    # 4) COP vs. cooling-water flow: the trade-off the TEA has to price
    fig, axes = plt.subplots(1, len(P_vals), figsize=(4.0 * len(P_vals), 4.4), sharey=True,
                             squeeze=False)
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(T_vals)))
    for ax, plr in zip(axes[0], sorted(P_vals, reverse=True)):
        for color, T in zip(colors, T_vals):
            pts = sorted((p.m_cw_ratio, p.optimum.COP) for p in points
                         if p.optimum and p.plr == plr and p.T_reject_C == T)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-o", color=color, markersize=3.5, label=f"{T:.0f} °C")
        ax.set_title(f"PLR = {plr:.1f}", fontsize=10)
        ax.set_xlabel("m_cw / m_cw_design [-]")
        ax.grid(alpha=0.4)
    axes[0][0].set_ylabel("COP [-]")
    axes[0][-1].legend(title="T_reject", fontsize=7.5, title_fontsize=8)
    fig.suptitle("COP vs. cooling-water flow")
    fig.tight_layout()
    figs.append(("cop_vs_mcw", fig))

    if save_dir is not None:
        for suffix, f in figs:
            path = save_dir / f"{cfg.output_name}_{suffix}.png"
            f.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {path}")
    if show:
        plt.show()
    return figs


if __name__ == "__main__":
    config = PartLoadMapConfig()

    # All assumptions (design point, grid, bounds) live centrally in
    # PartLoadMapConfig above -- adjust them there, not here.
    plot_dir = Path(__file__).resolve().parent / "Plots"
    plot_dir.mkdir(exist_ok=True)
    csv_file = plot_dir / f"{config.output_name}.csv"

    t_start = time.time()
    nominal, map_points = compute_part_load_map(config, csv_path=csv_file)
    print(f"Total runtime: {(time.time() - t_start) / 3600:.2f} h")
    print_map_table(map_points)
    print(f"CSV saved (written point by point): {csv_file}")
    plot_part_load_map(config, nominal, map_points, save_dir=plot_dir, show=False)
