"""Bilevel optimization: design-point UA values + part-load verification.

Stage 1 (design-point optimization, pinch-point model)
-------------------------------------------------------
Variables   : theta = [dT_min_shex, dT_min_des, dT_min_cond, dT_min_evap, dT_min_abs]
Bounds      : dT_min_i >= floor_i   (lower bound per heat exchanger)
Fixed       : Qevap_spec_kW, T_11_C, T_13_C/T_15_C, T_17_C, and the outlet
              temperature specifications (T12/T14/T16/T18_spec_C)
Objective   : min Sum(UA_i)  (SHEX, desorber, condenser, evaporator, absorber)
Penalty     : if solve_ac() fails to converge or plausibility checks
              (crystallization, concentration hierarchy, mass flow > 0) are violated

Optimizer   : differential_evolution (global, robust against non-convergence
              jumps) + Nelder-Mead polishing (local, with bounds)
Warm start  : the last converged primary vector is passed as x0 to the next
              evaluation; falls back to initial_guess(inputs)

IMPORTANT -- solver settings during optimization vs. final solve:
  During stage 1a/1b the inner solver runs with relaxed settings
  (config.opt_solver_tol / config.opt_max_nfev), because only convergence
  and feasibility need to be checked here, not maximum precision. The
  optimum FOUND is then re-solved ONCE with the strict model defaults
  (solver_tol=1e-9, max_nfev=5000) to precisely determine the final UA
  values. This preserves result quality at the end without every one of
  the thousands of intermediate evaluations needing full precision.

Stage 2 (part-load verification, UA model)
--------------------------------------------
The UA values determined in stage 1 AND the external mass flows occurring
at the design point (m11, m13, m15, m17, m1) are frozen. For defined
boundary-condition scenarios (e.g. extreme winter/summer cases), the plant
is ONLY simulated, NOT re-optimized -- "design at critical point, verify
elsewhere" (flexibility-analysis approach, cf. Swaney & Grossmann 1985 /
Halemane & Grossmann 1983).

ASSUMPTIONS for stage 2 (see chat history):
  1. AC_UA_LMTD returns the same result structure as AC_Pinch_Point.
  2. m1 is carried over constant from the design point in the part-load
     case (cycle_scale_spec_mode="m1"), i.e. fixed-speed solution pump.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    TQDM_AVAILABLE = False

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AC_Pinch_Point import (
    ACInputs as PinchInputs,
    ACResult as PinchResult,
    bounds as pinch_bounds,
    initial_guess as pinch_initial_guess,
    solve_ac as solve_pinch,
)

try:
    from Models.AC_UA_LMTD import (
        ACInputs as UAInputs,
        solve_ac as solve_ua,
    )
    UA_MODEL_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    UA_MODEL_AVAILABLE = False
    _ua_import_error = exc


def _clip_to_bounds(z: np.ndarray, inputs: PinchInputs) -> np.ndarray:
    """Safety net: defensively clips a starting vector to the model bounds.

    Guards against a hard crash of least_squares (ValueError: 'Initial
    guess is outside of provided bounds') if initial_guess() or a
    warm-start value lies just outside the permitted limits.
    """
    lower, upper = pinch_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


# ---------------------------------------------------------------------------
# Configuration -- ADJUST HERE
# ---------------------------------------------------------------------------

@dataclass
class DesignPointConfig:
    """Fixed boundary conditions of the design point (stage 1)."""

    # External inlet temperatures [°C]
    T_11_C: float = 90.0
    T_13_C: float = 25.0
    T_15_C: float = 25.0
    T_17_C: float = 11.0

    # External outlet temperature specifications [°C]
    T12_spec_C: float = 72.0
    T14_spec_C: float = 32.0
    T16_spec_C: float = 32.0
    T18_spec_C: float = 5.0

    # Design evaporator capacity [kW]
    Qevap_spec_kW: float = 40.9

    absorber_condenser_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # Lower bounds of the pinch temperature differences [K]
    dT_floor: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 3.0,
            "des": 3.0,
            "cond": 3.0,
            "evap": 3.0,
            "abs": 5.0,
        }
    )
    # Upper search bound = floor + dT_search_range
    dT_search_range: float = 20.0

    # Weighting of the UA values in the objective function (default: all equal, Sum(UA))
    ua_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 1.0, "des": 1.0, "cond": 1.0, "evap": 1.0, "abs": 1.0,
        }
    )

    # Cheaper solver settings ONLY for the optimization phase (stage 1a/1b).
    # The final point is then re-solved with the strict model defaults.
    # Recommendation: don't cap opt_max_nfev too aggressively (see chat
    # explanation) -- 100-150 is usually a good compromise between speed
    # and reliability, as long as the warm start holds. The diagnostics at
    # the end of optimize_design_point() show how often the limit was
    # actually hit.
    opt_solver_tol: float = 1.0e-5  # 1.0e-6
    opt_max_nfev: int = 100         # 150

    # DE tuning
    de_popsize: int = 12
    de_maxiter: int = 60
    de_workers: int = 1

    # Convergence plot (stage 1)
    make_convergence_plot: bool = True
    convergence_plot_path: str = "Design_Point/Plots/AC_stage1_convergence.png"


@dataclass
class PartLoadScenario:
    name: str
    T_11_C: float
    T_13_C: float
    T_15_C: float
    T_17_C: float


DEFAULT_SCENARIOS: List[PartLoadScenario] = [
    PartLoadScenario(name="extreme_cold", T_11_C=80.0, T_13_C=20.0, T_15_C=20.0, T_17_C=8.0),
    PartLoadScenario(name="extreme_warm", T_11_C=90.0, T_13_C=30.0, T_15_C=30.0, T_17_C=14.0),
]


# ---------------------------------------------------------------------------
# Stage 1: design-point optimization
# ---------------------------------------------------------------------------

THETA_ORDER = ["shex", "des", "cond", "evap", "abs"]


@dataclass
class EvalStats:
    """Collects diagnostics across all objective-function calls.

    NOTE on multiprocessing (de_workers > 1): each worker process has its
    own memory. This statistic (and the WarmStartCache) are then NOT
    reliably summed/shared across all processes -- the numbers/plot then
    reflect only part of the evaluations, and most of the warm-start
    benefit is lost per process. For correct, complete diagnostics/plots:
    use de_workers=1.
    """

    calls: int = 0
    feasible: int = 0
    infeasible: int = 0
    hit_nfev_cap: int = 0
    total_solve_time_s: float = 0.0
    history: List[Tuple[int, float]] = field(default_factory=list)

    def record(self, cost: float, feasible: bool, hit_cap: bool, dt: float) -> None:
        self.calls += 1
        self.total_solve_time_s += dt
        if feasible:
            self.feasible += 1
        else:
            self.infeasible += 1
        if hit_cap:
            self.hit_nfev_cap += 1
        self.history.append((self.calls, cost))

    @property
    def avg_solve_time_s(self) -> float:
        return self.total_solve_time_s / self.calls if self.calls else 0.0


class DEProgress:
    """Progress display for differential_evolution (one line per generation)."""

    def __init__(self, maxiter: int, stats: EvalStats):
        self.stats = stats
        self.maxiter = maxiter
        self.gen = 0
        self.pbar = tqdm(total=maxiter, desc="Stage 1a", unit="Gen") if TQDM_AVAILABLE else None

    def __call__(self, xk, convergence) -> bool:
        self.gen += 1
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix(
                conv=f"{convergence:.2e}",
                calls=self.stats.calls,
                avg_s=f"{self.stats.avg_solve_time_s:.2f}",
            )
        else:
            print(
                f"  Gen {self.gen}/{self.maxiter} | convergence={convergence:.2e} "
                f"| calls={self.stats.calls} | avg={self.stats.avg_solve_time_s:.2f}s"
            )
        return False

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class WarmStartCache:
    """Holds the last converged primary vector for warm starts."""

    def __init__(self) -> None:
        self.z: Optional[np.ndarray] = None

    def get(self, inputs: PinchInputs) -> np.ndarray:
        z = self.z if self.z is not None else pinch_initial_guess(inputs)
        return _clip_to_bounds(z, inputs)

    def update(self, z: np.ndarray) -> None:
        self.z = np.asarray(z, dtype=float).copy()

    def reset(self) -> None:
        self.z = None


def build_pinch_inputs(
    theta: np.ndarray, config: DesignPointConfig, *, fast: bool
) -> PinchInputs:
    """Builds the pinch-point inputs.

    fast=True  : relaxed solver tolerances (stage 1a/1b, many evaluations)
    fast=False : strict model defaults (final solve after optimization)
    """
    dT_shex, dT_des, dT_cond, dT_evap, dT_abs = theta
    kwargs = dict(
        T_11_C=config.T_11_C,
        T_13_C=config.T_13_C,
        T_15_C=config.T_15_C,
        T_17_C=config.T_17_C,
        dT_min_shex=float(dT_shex),
        dT_min_des=float(dT_des),
        dT_min_cond=float(dT_cond),
        dT_min_evap=float(dT_evap),
        dT_min_abs=float(dT_abs),
        absorber_condenser_routing_mode=config.absorber_condenser_routing_mode,
        cycle_scale_spec_mode="Qeva",
        Qevap_spec_kW=config.Qevap_spec_kW,
        desorber_spec_mode="T12",
        T12_spec_C=config.T12_spec_C,
        absorber_spec_mode="T14",
        T14_spec_C=config.T14_spec_C,
        condenser_spec_mode="T16",
        T16_spec_C=config.T16_spec_C,
        evaporator_spec_mode="T18",
        T18_spec_C=config.T18_spec_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.opt_solver_tol
        kwargs["max_nfev"] = config.opt_max_nfev
    return PinchInputs(**kwargs)


def _is_feasible(result: PinchResult) -> bool:
    if not result.solve_info.success or not result.solve_info.final_point_evaluable:
        return False
    if not result.checks:
        return False
    return all(result.checks.values())


def design_point_objective(
    theta: np.ndarray,
    config: DesignPointConfig,
    cache: WarmStartCache,
    stats: EvalStats,
) -> float:
    inputs = build_pinch_inputs(theta, config, fast=True)
    x0 = cache.get(inputs)

    t0 = time.perf_counter()
    result = solve_pinch(inputs, x0=x0)
    dt = time.perf_counter() - t0

    feasible = _is_feasible(result)
    hit_cap = (
        not result.solve_info.success
        and result.solve_info.nfev >= config.opt_max_nfev
    )

    if not feasible:
        penalty = 1.0e4 + 100.0 * result.solve_info.scaled_residual_norm
        stats.record(penalty, feasible=False, hit_cap=hit_cap, dt=dt)
        return float(penalty)

    cache.update(np.array(list(result.primary_variables.values()), dtype=float))

    ua = result.UA_conversion
    total = (
        config.ua_weights["shex"] * ua["UA_shex"]
        + config.ua_weights["des"] * ua["UA_des"]
        + config.ua_weights["cond"] * ua["UA_cond"]
        + config.ua_weights["evap"] * ua["UA_evap"]
        + config.ua_weights["abs"] * ua["UA_abs"]
    )
    stats.record(total, feasible=True, hit_cap=False, dt=dt)
    return float(total)


def _theta_bounds(config: DesignPointConfig) -> List[tuple]:
    return [
        (config.dT_floor[key], config.dT_floor[key] + config.dT_search_range)
        for key in THETA_ORDER
    ]


def plot_convergence(stats: EvalStats, path: str) -> None:
    if not stats.history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed -- skipping convergence plot "
            "(pip install matplotlib)."
        )
        return

    calls, costs = zip(*stats.history)
    costs = np.asarray(costs, dtype=float)
    running_best = np.minimum.accumulate(costs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(calls, costs, s=6, alpha=0.35, label="single evaluation (incl. penalty)")
    ax.plot(calls, running_best, color="tab:red", linewidth=2, label="best Sum(UA) so far")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Objective value [kW/K] (penalty ≈ 1e4)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot saved: {path}")


def optimize_design_point(
    config: DesignPointConfig, *, seed: int = 42, verbose: bool = True
) -> tuple[np.ndarray, PinchResult, EvalStats]:
    cache = WarmStartCache()
    stats = EvalStats()
    bounds = _theta_bounds(config)

    if config.de_workers != 1 and verbose:
        print(
            "Note: de_workers != 1 -- warm-start cache and live diagnostics "
            "(counters/plot) are not reliably complete across multiple "
            "processes. Use de_workers=1 for exact diagnostics/plots."
        )

    t_stage1_start = time.perf_counter()

    if verbose:
        print("Stage 1a: global search (differential_evolution) ...")

    progress = DEProgress(config.de_maxiter, stats)
    updating = "deferred" if config.de_workers != 1 else "immediate"

    t0 = time.perf_counter()
    de_result = differential_evolution(
        design_point_objective,
        bounds=bounds,
        args=(config, cache, stats),
        seed=seed,
        popsize=config.de_popsize,
        maxiter=config.de_maxiter,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=False,
        workers=config.de_workers,
        updating=updating,
        callback=progress,
        disp=False,
    )
    t_de = time.perf_counter() - t0
    progress.close()

    if verbose:
        print(
            f"  DE result: theta = {de_result.x}, Sum(UA) = {de_result.fun:.4f} "
            f"| duration: {t_de/60:.1f} min"
        )
        print("Stage 1b: local polishing (Nelder-Mead) ...")

    t0 = time.perf_counter()
    nm_result = minimize(
        design_point_objective,
        x0=de_result.x,
        args=(config, cache, stats),
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-3, "fatol": 1e-4, "adaptive": True, "maxiter": 500},
    )
    t_nm = time.perf_counter() - t0

    theta_opt = nm_result.x if nm_result.fun <= de_result.fun else de_result.x

    if verbose:
        print(
            f"  Final theta = {theta_opt}, "
            f"Sum(UA) = {min(nm_result.fun, de_result.fun):.4f} | duration: {t_nm:.1f} s"
        )

    if verbose:
        print("Final precision solve (strict tolerances) ...")
    t0 = time.perf_counter()
    inputs_opt = build_pinch_inputs(theta_opt, config, fast=False)
    result_opt = solve_pinch(inputs_opt, x0=cache.get(inputs_opt))
    t_final = time.perf_counter() - t0

    if not _is_feasible(result_opt):
        raise RuntimeError(
            "Final optimization point is not feasible. Please check bounds/boundary conditions."
        )

    t_stage1_total = time.perf_counter() - t_stage1_start

    if verbose:
        print()
        print("Stage 1 timing overview")
        print(f"  1a global search (DE)    : {t_de/60:8.2f} min")
        print(f"  1b local polishing (NM)  : {t_nm:8.2f} s")
        print(f"  final precision solve    : {t_final:8.2f} s")
        print(f"  total stage 1            : {t_stage1_total/60:8.2f} min")
        print()
        print("Evaluation diagnostics")
        print(f"  total calls              : {stats.calls}")
        print(f"  of which feasible        : {stats.feasible}")
        print(f"  of which infeasible/penalty: {stats.infeasible}")
        print(f"  of which hit nfev limit  : {stats.hit_nfev_cap}"
              f" (of opt_max_nfev={config.opt_max_nfev})")
        print(f"  avg solve time/call      : {stats.avg_solve_time_s*1000:.1f} ms")
        if stats.calls and stats.hit_nfev_cap / stats.calls > 0.1:
            print(
                "  WARNING: >10% of calls hit the nfev limit -- consider "
                "raising opt_max_nfev to avoid false infeasibility "
                "classification."
            )

    if config.make_convergence_plot:
        plot_convergence(stats, config.convergence_plot_path)

    return theta_opt, result_opt, stats


def print_design_point_summary(theta: np.ndarray, result: PinchResult) -> None:
    print("=" * 90)
    print("Design-point optimization -- result")
    print("=" * 90)
    for key, val in zip(THETA_ORDER, theta):
        print(f"  dT_min_{key:5s}: {val:8.4f} K")
    print()
    print("UA values [kW/K]")
    for key in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"]:
        print(f"  {key:10s}: {result.UA_conversion[key]:10.4f}")
    total_ua = sum(result.UA_conversion[k] for k in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"])
    print(f"  {'Sum(UA)':10s}: {total_ua:10.4f}")
    print()
    print("External mass flows at the design point [kg/s]")
    for key in ["m11_kg_s", "m13_kg_s", "m15_kg_s", "m17_kg_s"]:
        print(f"  {key:10s}: {result.diagnostics[key]:10.6f}")
    # m1 (internal solution mass flow) is not in diagnostics, but in state
    # "1" (solution after the absorber, before the pump).
    m1 = result.states["1"]["m_kg_s"]
    print(f"  {'m1':10s}: {m1:10.6f}   (internal solution mass flow, relevant for stage 2)")
    print()
    print(f"COP: {result.kpis['COP']:.4f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Stage 2: part-load verification (simulation only, no optimization)
# ---------------------------------------------------------------------------

def build_ua_inputs(
    scenario: PartLoadScenario,
    design_ua: Dict[str, float],
    design_flows: Dict[str, float],
    config: DesignPointConfig,
) -> "UAInputs":
    """Builds the UA-model inputs for a part-load point.

    shex_model="UA" with explicit UA_shex (NOT "NTU"/Effectiveness_shex),
    because otherwise the SHEX UA value optimized in stage 1 would be
    ignored.
    """
    return UAInputs(
        T_11_C=scenario.T_11_C,
        T_13_C=scenario.T_13_C,
        T_15_C=scenario.T_15_C,
        T_17_C=scenario.T_17_C,
        UA_shex=design_ua["UA_shex"],
        UA_des=design_ua["UA_des"],
        UA_cond=design_ua["UA_cond"],
        UA_evap=design_ua["UA_evap"],
        UA_abs=design_ua["UA_abs"],
        m_11=design_flows["m11_kg_s"],
        m_13=design_flows["m13_kg_s"],
        m_15=design_flows["m15_kg_s"],
        absorber_condenser_routing_mode=config.absorber_condenser_routing_mode,
        cycle_scale_spec_mode="m1",
        m1_spec=design_flows["m1_kg_s"],
        evaporator_spec_mode="m17",
        m17_spec=design_flows["m17_kg_s"],
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
        shex_model="UA",
    )


def verify_part_load(
    scenarios: List[PartLoadScenario],
    design_result: PinchResult,
    config: DesignPointConfig,
) -> None:
    if not UA_MODEL_AVAILABLE:
        print(
            "Could not import the UA model "
            f"({_ua_import_error!r}). Skipping stage 2."
        )
        return

    design_ua = {
        k: design_result.UA_conversion[k]
        for k in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"]
    }
    design_flows = {
        "m11_kg_s": design_result.diagnostics["m11_kg_s"],
        "m13_kg_s": design_result.diagnostics["m13_kg_s"],
        "m15_kg_s": design_result.diagnostics["m15_kg_s"],
        "m17_kg_s": design_result.diagnostics["m17_kg_s"],
        "m1_kg_s": design_result.states["1"]["m_kg_s"],
    }

    print("=" * 90)
    print("Stage 2: part-load verification")
    print("=" * 90)

    cache = WarmStartCache()

    for scenario in scenarios:
        t0 = time.perf_counter()
        try:
            inputs = build_ua_inputs(scenario, design_ua, design_flows, config)
        except TypeError as exc:
            print(
                f"[{scenario.name}] Could not construct UAInputs -- "
                f"parameter mismatch with AC_UA_LMTD: {exc}\n"
                "  -> Please share the AC_UA_LMTD.py source so build_ua_inputs()"
                " can be adjusted."
            )
            continue

        x0 = cache.get(inputs)

        try:
            result = solve_ua(inputs, x0=x0)
        except Exception as exc:
            print(f"[{scenario.name}] Error while solving: {exc}")
            continue

        try:
            feasible = _is_feasible(result)
        except AttributeError as exc:
            print(
                f"[{scenario.name}] Result structure of the UA model differs: {exc}\n"
                "  -> Please share the AC_UA_LMTD.py source for comparison."
            )
            continue

        dt = time.perf_counter() - t0
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"\n[{scenario.name}] Status: {status}  ({dt:.2f} s)")

        if feasible:
            cache.update(np.array(list(result.primary_variables.values()), dtype=float))
            print(f"  Q_evap = {result.heat_flows_kW.get('Q_evap', float('nan')):.3f} kW")
            print(f"  COP    = {result.kpis.get('COP', float('nan')):.4f}")
        else:
            print(f"  Solver message: {result.solve_info.message}")
            if result.checks:
                violated = [k for k, v in result.checks.items() if not v]
                print(f"  Violated checks: {violated}")

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = DesignPointConfig()

    t_total_start = time.perf_counter()

    theta_opt, design_result, stats = optimize_design_point(config)
    print_design_point_summary(theta_opt, design_result)

    t0 = time.perf_counter()
    verify_part_load(DEFAULT_SCENARIOS, design_result, config)
    t_stage2 = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start
    print()
    print(f"Stage 2 duration: {t_stage2:.1f} s")
    print(f"Total duration  : {t_total/60:.2f} min")
