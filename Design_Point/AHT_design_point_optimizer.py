"""Design-point optimization for the absorption heat transformer AHT.

Analogous to the chiller's design-point optimizer (AC_design_point_optimizer.py),
but for Models.AHT_Pinch_Point instead of Models.AC_Pinch_Point.

IMPORTANT DIFFERENCE from the chiller -- swapped roles of the components:
  - The absorber delivers the USEFUL HEAT (high temperature level, "product"
    of the AHT). Spec variables: absorber_spec_mode="T12", T12_spec_C
    (useful-heat sink, external inlet/outlet temperature T_11_C -> T12_spec_C).
  - Desorber AND evaporator are BOTH fed from the external waste-heat source
    (medium temperature level) -- hence "desorber_evaporator_routing_mode"
    (parallel / serial), analogous to "absorber_condenser_routing_mode" for
    the chiller, just with the components swapped.
  - The condenser rejects heat at low temperature level (T_17_C -> T18_spec_C).
  - cycle_scale_spec_mode="Qabs" with Qabs_spec_kW: the design useful-heat
    capacity (analogous to Qevap_spec_kW for the chiller) is specified at
    the ABSORBER.

Stage 2 (part-load verification, AHT_UA_LMTD)
------------------------------------------------
Analogous to the chiller: UA values and external mass flows from stage 1
are frozen, and for defined boundary-condition scenarios the plant is ONLY
simulated, NOT re-optimized. Togglable via config.run_stage2.

ASSUMPTION (please verify, since I don't have Models.AHT_UA_LMTD):
structure analogous to Models.AC_UA_LMTD -- desorber/evaporator/condenser
as fixed mass-flow kwargs (m_13, m_15, m_17), ONLY the absorber (the
design-determining component group) selectable via spec_mode "m11"/"T12",
cycle_scale_spec_mode="m6" with m6_spec. If this is wrong: please send
feedback or the source/main script of AHT_UA_LMTD.py, and build_ua_inputs()
will be adjusted exactly.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
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

from Models.AHT_Pinch_Point import (
    AHTInputs,
    bounds as aht_bounds,
    initial_guess as aht_initial_guess,
    solve_aht,
)

try:
    from Models.AHT_UA_LMTD import (
        AHTInputs as UAInputs,
        solve_aht as solve_ua,
    )
    UA_MODEL_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    UA_MODEL_AVAILABLE = False
    _ua_import_error = exc


def _clip_to_bounds(z: np.ndarray, inputs: AHTInputs) -> np.ndarray:
    """Safety net: defensively clips a starting vector to the model bounds."""
    lower, upper = aht_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


# ---------------------------------------------------------------------------
# Configuration -- ADJUST HERE
# ---------------------------------------------------------------------------

@dataclass
class DesignPointConfig:
    """Fixed boundary conditions of the AHT design point."""

    # External inlet temperatures [°C]
    T_11_C: float = 63.0    # useful-heat sink (absorber), cold inlet
    T_13_C: float = 57.0    # waste-heat source (desorber/evaporator, routing-dependent)
    T_15_C: float = 57.0    # waste-heat source (desorber/evaporator, routing-dependent)
    T_17_C: float = 20.0    # reject cooling (condenser), cold inlet

    # External outlet temperature specifications [°C]
    T12_spec_C: float = 67.0    # useful-heat sink, outlet (absorber)
    T14_spec_C: float = 52.0    # waste-heat source, outlet (desorber)
    T16_spec_C: float = 52.0    # waste-heat source, outlet (evaporator)
    T18_spec_C: float = 24.0    # reject cooling, outlet (condenser)

    # Design useful-heat capacity [kW] (at the absorber, NOT the evaporator!)
    Qabs_spec_kW: float = 500.0

    desorber_evaporator_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    dT_floor: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 3.0, "des": 3.0, "cond": 3.0, "evap": 3.0, "abs": 3.0,
        }
    )
    dT_search_range: float = 30.0

    ua_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 1.0, "des": 1.0, "cond": 1.0, "evap": 1.0, "abs": 1.0,
        }
    )

    opt_solver_tol: float = 1.0e-6
    opt_max_nfev: int = 150

    de_popsize: int = 12
    de_maxiter: int = 60
    de_workers: int = 1

    de_patience: Optional[int] = 25
    de_min_improvement: float = 0.01  # kW/K

    # MUST lie well above any plausible real Sum(UA) value -- otherwise an
    # infeasible point with a tiny residual error (penalty cost ~penalty_base,
    # since the residual term is then almost 0) can look CHEAPER than any
    # real, feasible design, and DE then "optimizes" toward a point that
    # violates only a plausibility check (not convergence) (see chat: the DE
    # result was exactly Sum(UA)=old_penalty_base=200 twice -- not a
    # coincidence, but exactly this effect).
    penalty_base: float = 5000.0
    penalty_residual_weight: float = 20.0

    make_convergence_plot: bool = True
    convergence_plot_path: str = "Design_Point/Plots/AHT_stage1_convergence_AHT.png"

    # Optional manual starting vector (internal model units, K/-, order as
    # in primary_variables) as a fallback for the VERY FIRST optimizer call,
    # before the warm-start cache is populated. Useful for difficult/new
    # operating conditions when you already know a hand-tuned, converging
    # solution for a similar operating point (e.g. from your main script or
    # from quick_feasibility_probe()) -- noticeably more reliable than the
    # model's generic initial_guess() heuristic.
    x0_override: Optional[np.ndarray] = None

    # Optional HARD upper bounds per heat exchanger (overrides
    # dT_floor[key] + dT_search_range for the named keys). Useful when the
    # optimization "sticks" to the upper bound (see the diagnostic note at
    # the end of optimize_design_point) -- that indicates too narrow a
    # search space for THIS operating point, not necessarily a warm-start
    # problem.
    #
    # Deliberately set ASYMMETRICALLY here (not a uniform cap for all
    # five!): an earlier investigation (see chat) measured how much each
    # individual pinch costs in achievable GTL, starting from a 3K baseline
    # (T_waste=70°C, approach 4/4/3K):
    #   dT_min_shex: -0.25 K GTL per +2 K pinch  (~0.13 K/K -- barely any effect)
    #   dT_min_des:  -2.63 K GTL per +2 K pinch  (~1.31 K/K)
    #   dT_min_cond: -2.50 K GTL per +2 K pinch  (~1.25 K/K)
    #   dT_min_evap: -2.38 K GTL per +2 K pinch  (~1.19 K/K)
    #   dT_min_abs:  -2.00 K GTL per +2 K pinch  (~1.00 K/K)
    # So SHEX barely costs any GTL (a wider SHEX pinch is usually even
    # BENEFICIAL for the optimization objective Sum(UA): smaller SHEX area,
    # barely any GTL loss) -- hence NOT constrained here (stays at
    # dT_floor+dT_search_range). The other four, on the other hand,
    # effectively share a common "GTL budget": at this operating point
    # (T12_spec_C=67°C, T15_C=57°C) that's about 10 K. The caps below are
    # `floor + budget/sensitivity`, i.e. the pinch increase at which THIS
    # heat exchanger ALONE would use up the entire budget (the other three
    # would then have to stay near the floor) -- a generous but no longer
    # pointlessly wide frame. These sensitivities were measured at ONE
    # other operating point; the order of magnitude/ranking should
    # transfer, but the exact numbers are a starting value, not an exact
    # derivation for THIS operating point.
    dT_upper: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "des": 11.0, "cond": 11.0, "evap": 11.0, "abs": 13.0,
        }
    )

    # Quick feasibility test before the full optimization (see
    # quick_feasibility_probe) -- costs only seconds to a few minutes and
    # would have made the 19h failure recognizable early.
    run_feasibility_probe: bool = True

    # Stage 1b (Nelder-Mead polishing) can be disabled for fast test runs.
    # Can take a very long time without improving anything when the DE
    # result is almost entirely infeasible (e.g. with a very small
    # de_popsize/de_maxiter for testing) -- NM has no early stopping and
    # then runs through to maxiter=500 by pure wandering in the penalty
    # region (see chat: 104 minutes of NM in a test run with only 3
    # feasible points).
    run_local_polish: bool = True

    # Stage 2: part-load verification on/off. If False, verify_part_load()
    # is not called at all (saves the few seconds, mainly useful while
    # experimenting with stage 1 and not currently needing stage 2).
    run_stage2: bool = False


@dataclass
class PartLoadScenario:
    """A boundary operating point to verify (stage 2). ADJUST VALUES --
    currently just placeholders relative to the nominal design point above."""

    name: str
    T_11_C: float
    T_13_C: float
    T_15_C: float
    T_17_C: float


DEFAULT_SCENARIOS: List[PartLoadScenario] = [
    PartLoadScenario(name="extreme_cold", T_11_C=70.0, T_13_C=55.0, T_15_C=55.0, T_17_C=15.0),
    PartLoadScenario(name="extreme_warm", T_11_C=70.0, T_13_C=62.0, T_15_C=62.0, T_17_C=25.0),
]


# ---------------------------------------------------------------------------
# Stage 1: design-point optimization
# ---------------------------------------------------------------------------

THETA_ORDER = ["shex", "des", "cond", "evap", "abs"]


@dataclass
class EvalStats:
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
    """Progress display + early stopping for differential_evolution."""

    def __init__(
        self,
        maxiter: int,
        stats: EvalStats,
        patience: Optional[int] = None,
        min_improvement: float = 0.02,
    ):
        self.stats = stats
        self.maxiter = maxiter
        self.gen = 0
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_history: List[float] = []
        self.pbar = tqdm(total=maxiter, desc="Stage 1a", unit="Gen") if TQDM_AVAILABLE else None

    def __call__(self, xk, convergence) -> bool:
        self.gen += 1
        current_best = min((c for _, c in self.stats.history), default=float("inf"))
        self.best_history.append(current_best)

        stop = False
        if self.patience is not None and len(self.best_history) > self.patience:
            improvement = self.best_history[-self.patience - 1] - self.best_history[-1]
            if improvement < self.min_improvement:
                stop = True

        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix(
                conv=f"{convergence:.2e}",
                best=f"{current_best:.4f}",
                calls=self.stats.calls,
                avg_s=f"{self.stats.avg_solve_time_s:.2f}",
            )
        else:
            print(
                f"  Gen {self.gen}/{self.maxiter} | best={current_best:.4f} "
                f"| convergence={convergence:.2e} | calls={self.stats.calls} "
                f"| avg={self.stats.avg_solve_time_s:.2f}s"
            )

        if stop:
            msg = (
                f"\n  Early stopping: no improvement > {self.min_improvement} kW/K "
                f"for {self.patience} generations -- ending the search."
            )
            if self.pbar is not None:
                self.pbar.write(msg)
            else:
                print(msg)

        return stop

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class WarmStartCache:
    """Holds the last converged primary vector for warm starts.

    fallback: optional manual starting vector (config.x0_override), used as
    long as no converged point is yet in the cache -- more reliable than
    the generic initial_guess() heuristic when you already know a
    hand-tuned solution for a similar operating point.
    """

    def __init__(self, fallback: Optional[np.ndarray] = None) -> None:
        self.z: Optional[np.ndarray] = None
        self.fallback = fallback

    def get(self, inputs: AHTInputs) -> np.ndarray:
        if self.z is not None:
            z = self.z
        elif self.fallback is not None:
            z = self.fallback
        else:
            z = aht_initial_guess(inputs)
        return _clip_to_bounds(z, inputs)

    def update(self, z: np.ndarray) -> None:
        self.z = np.asarray(z, dtype=float).copy()

    def reset(self) -> None:
        self.z = None


def build_aht_inputs(
    theta: np.ndarray, config: DesignPointConfig, *, fast: bool
) -> AHTInputs:
    """Builds the AHT inputs. fast=True -> relaxed solver tolerances (stage 1)."""
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
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="Qabs",
        Qabs_spec_kW=config.Qabs_spec_kW,
        absorber_spec_mode="T12",
        T12_spec_C=config.T12_spec_C,
        desorber_spec_mode="T14",
        T14_spec_C=config.T14_spec_C,
        evaporator_spec_mode="T16",
        T16_spec_C=config.T16_spec_C,
        condenser_spec_mode="T18",
        T18_spec_C=config.T18_spec_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.opt_solver_tol
        kwargs["max_nfev"] = config.opt_max_nfev
    return AHTInputs(**kwargs)


def _is_feasible(result) -> bool:
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
    inputs = build_aht_inputs(theta, config, fast=True)
    x0 = cache.get(inputs)

    t0 = time.perf_counter()
    result = solve_aht(inputs, x0=x0)
    feasible = _is_feasible(result)

    # Fall back to a fresh, generic starting vector if the cached warm
    # start fails. The cache holds only ONE vector -- the one from the
    # last SUCCESSFUL theta. But DE often jumps far around the 5D space
    # between generations (no continuous path), and the solver's basin of
    # attraction is empirically only ~2-4 K wide (see
    # AHT_feasibility_sweep.py). Without this fallback, a poorly
    # warm-started attempt often gets stuck at the nfev limit instead of
    # cleanly converging OR cleanly failing -- this was the main cause of
    # the 72% nfev-limit hits and the 16h runtime in the 500kW test run.
    # aht_initial_guess() doesn't know the current theta, but is often a
    # much better starting point for a NEW theta than a warm start from a
    # completely different theta.
    if not feasible:
        x0_fresh = _clip_to_bounds(aht_initial_guess(inputs), inputs)
        result_fresh = solve_aht(inputs, x0=x0_fresh)
        if _is_feasible(result_fresh):
            result = result_fresh
            feasible = True

    dt = time.perf_counter() - t0

    hit_cap = (
        not result.solve_info.success
        and result.solve_info.nfev >= config.opt_max_nfev
    )

    if not feasible:
        penalty = config.penalty_base + config.penalty_residual_weight * result.solve_info.scaled_residual_norm
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
    result = []
    for key in THETA_ORDER:
        lo = config.dT_floor[key]
        if config.dT_upper and key in config.dT_upper:
            hi = config.dT_upper[key]
        else:
            hi = lo + config.dT_search_range
        result.append((lo, hi))
    return result


def plot_convergence(stats: EvalStats, path: str) -> None:
    if not stats.history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping convergence plot.")
        return

    calls, costs = zip(*stats.history)
    costs = np.asarray(costs, dtype=float)
    running_best = np.minimum.accumulate(costs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(calls, costs, s=6, alpha=0.35, label="single evaluation (incl. penalty)")
    ax.plot(calls, running_best, color="tab:red", linewidth=2, label="best Sum(UA) so far")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Objective value [kW/K]")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot saved: {path}")


def optimize_design_point(
    config: DesignPointConfig, *, seed: int = 42, verbose: bool = True
) -> tuple[np.ndarray, "object", EvalStats]:
    cache = WarmStartCache(fallback=config.x0_override)
    stats = EvalStats()
    bounds = _theta_bounds(config)

    t_stage1_start = time.perf_counter()

    if verbose:
        print("Stage 1a: global search (differential_evolution) ...")

    progress = DEProgress(
        config.de_maxiter, stats, patience=config.de_patience, min_improvement=config.de_min_improvement
    )
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

    if config.run_local_polish:
        if verbose:
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
    else:
        t_nm = 0.0
        theta_opt = de_result.x
        if verbose:
            print("Stage 1b: local polishing (Nelder-Mead) skipped (config.run_local_polish=False).")

    if verbose:
        print("Final precision solve (strict tolerances) ...")

    t0 = time.perf_counter()
    inputs_opt = build_aht_inputs(theta_opt, config, fast=False)
    result_opt = solve_aht(inputs_opt, x0=cache.get(inputs_opt))
    t_final = time.perf_counter() - t0

    final_feasible = _is_feasible(result_opt)
    if not final_feasible:
        if verbose:
            print(
                "  WARNING: precision solve (strict tolerances) is NOT feasible "
                f"(message: '{result_opt.solve_info.message}'). Falling back to "
                "the result with the relaxed optimization tolerances (fast=True), "
                "so you can still see what was found."
            )
        inputs_fallback = build_aht_inputs(theta_opt, config, fast=True)
        result_opt = solve_aht(inputs_fallback, x0=cache.get(inputs_fallback))
        final_feasible = _is_feasible(result_opt)

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
        if stats.calls:
            infeasible_frac = stats.infeasible / stats.calls
            print(f"  infeasible fraction      : {infeasible_frac*100:.1f}%")
            if infeasible_frac > 0.25:
                print(
                    "  NOTE: high infeasible fraction -- dT_search_range/dT_upper "
                    "may be chosen too narrow for this operating point."
                )
        if stats.calls and stats.hit_nfev_cap / stats.calls > 0.1:
            print("  WARNING: >10% of calls hit the nfev limit -- check opt_max_nfev.")

        # Boundary diagnostic: is theta sticking to the upper search bound?
        near_upper = []
        for key, val, (lo, hi) in zip(THETA_ORDER, theta_opt, bounds):
            span = hi - lo
            if span > 0 and (val - lo) / span > 0.9:
                near_upper.append(key)
        if near_upper:
            print(
                f"  NOTE: theta is near the UPPER search bound for {near_upper} -- "
                "this indicates a search space (dT_search_range/dT_upper) chosen "
                "too narrow for this operating point, not necessarily a "
                "warm-start problem."
            )

    if config.make_convergence_plot:
        plot_convergence(stats, config.convergence_plot_path)

    if not final_feasible:
        print(
            "\nWARNING: the fallback solve (relaxed tolerances) is also not "
            "feasible. The returned result does NOT correspond to any valid "
            "solution -- please do NOT use it for UA values. theta_opt "
            "presumably lies outside the actually solvable range (see the "
            "boundary note above)."
        )

    return theta_opt, result_opt, stats


def print_design_point_summary(theta: np.ndarray, result) -> None:
    print("=" * 90)
    print("AHT design-point optimization -- result")
    print("=" * 90)
    for key, val in zip(THETA_ORDER, theta):
        print(f"  dT_min_{key:5s}: {val:8.4f} K")
    print()

    if not _is_feasible(result) or not result.UA_conversion:
        # solve_aht() deliberately returns an EMPTY UA_conversion when
        # final_point_evaluable=False (see Models.AHT_Pinch_Point) -- UA
        # values from an invalid state would be meaningless. theta_opt
        # above is then the best theta FOUND (but not valid), not a design
        # result.
        print(
            "NO valid result (result.solve_info.final_point_evaluable=False "
            "or a plausibility check is violated) -- UA values, mass flows, "
            "and KPIs cannot be meaningfully printed. theta above is merely "
            f"the best theta FOUND, not a valid one "
            f"(solver message: '{result.solve_info.message}')."
        )
        print("=" * 90)
        return

    print("UA values [kW/K]")
    for key in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"]:
        print(f"  {key:10s}: {result.UA_conversion[key]:10.4f}")
    total_ua = sum(result.UA_conversion[k] for k in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"])
    print(f"  {'Sum(UA)':10s}: {total_ua:10.4f}")
    print()
    print("External mass flows at the design point [kg/s]")
    labels = {
        "m11_kg_s": "absorber (useful heat)",
        "m13_kg_s": "desorber (source)",
        "m15_kg_s": "evaporator (source)",
        "m17_kg_s": "condenser (reject cooling)",
    }
    for key, label in labels.items():
        print(f"  {key:10s} [{label:24s}]: {result.diagnostics[key]:10.6f}")
    m6 = result.diagnostics["m6_kg_s"]
    print(f"  {'m6':10s} [internal solution flow ]: {m6:10.6f}")
    print()
    print("KPIs")
    for k, v in result.kpis.items():
        try:
            print(f"  {k:10s}: {float(v):10.4f}")
        except (TypeError, ValueError):
            print(f"  {k:10s}: {v}")
    print("=" * 90)


def quick_feasibility_probe(
    config: DesignPointConfig, thetas: Optional[List[np.ndarray]] = None
) -> None:
    """Quick test (seconds to a few minutes) BEFORE a full optimization.

    Checks a few candidate theta vectors WITH STRICT solver settings
    (fast=False, like the final solve) and shows feasibility + compute
    time. This lets you see within minutes whether
    dT_floor/dT_search_range/dT_upper/x0_override are reasonably chosen
    at all for a new (possibly difficult) operating point -- BEFORE
    investing hours in a full DE search (see the 19h experience with a
    too-narrow/wrong setup).

    Without custom thetas: automatically tests three points (near the
    lower bound, middle, near the upper bound) per heat-exchanger
    combination.
    """
    bounds = _theta_bounds(config)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    if thetas is None:
        thetas = [
            lo + 0.1 * (hi - lo),
            (lo + hi) / 2.0,
            hi - 0.1 * (hi - lo),
        ]

    cache = WarmStartCache(fallback=config.x0_override)

    print("=" * 90)
    print("Quick feasibility test (recommended BEFORE a full optimization)")
    print("=" * 90)
    any_feasible = False
    for theta in thetas:
        inputs = build_aht_inputs(np.asarray(theta, dtype=float), config, fast=False)
        x0 = cache.get(inputs)
        t0 = time.perf_counter()
        result = solve_aht(inputs, x0=x0)
        dt = time.perf_counter() - t0
        feasible = _is_feasible(result)
        any_feasible = any_feasible or feasible
        status = "FEASIBLE  " if feasible else "INFEASIBLE"
        print(
            f"  theta={np.round(theta, 2)} -> {status} "
            f"({dt:.2f}s, nfev={result.solve_info.nfev})"
        )
        if feasible:
            cache.update(np.array(list(result.primary_variables.values()), dtype=float))
        else:
            print(f"    Solver message: {result.solve_info.message}")
    if not any_feasible:
        print(
            "\n  NO test point feasible -- before starting a full optimization,"
            " check the boundary conditions (Qabs_spec_kW, temperature levels)"
            " for fundamental solvability, e.g. with even larger dT_upper"
            " values as a test."
        )
    print("=" * 90)


def sweep_parameter(
    base_config: DesignPointConfig,
    param_name: str,
    values: List[float],
    *,
    verbose: bool = True,
) -> List[Tuple[float, np.ndarray, "object"]]:
    """Continuation/homotopy sweep for sensitivity analyses.

    Varies EXACTLY ONE parameter (e.g. 'T_17_C') over a list of values,
    holds all other boundary conditions fixed, and runs a FULL design-point
    optimization for each value -- but with one crucial difference from
    independent individual runs: the converged theta AND the converged
    primary vector of the previous value are used as the starting point
    (x0_override) for the next value.

    This is exactly the continuation/homotopy principle from the chat
    explanation: small steps from a known-good point instead of starting
    "cold" every time. For a sensitivity analysis over a temperature level
    this is usually both FASTER (good warm start => faster convergence,
    smaller de_maxiter/de_popsize may suffice) and MORE ROBUST (the
    optimizer never starts completely "blind" again).

    NOTE: values should be in ascending OR descending order (monotonic),
    not mixed arbitrarily -- otherwise the steps between consecutive
    values are larger than necessary and the warm-start benefit shrinks.

    Returns: list of (value, theta_opt, result) for each successfully
    solved value. Values for which even the fallback solve was infeasible
    are skipped (with a warning), but the sweep continues.
    """
    results: List[Tuple[float, np.ndarray, "object"]] = []
    x0_carry: Optional[np.ndarray] = base_config.x0_override
    theta_carry: Optional[np.ndarray] = None

    for i, value in enumerate(values):
        if verbose:
            print("\n" + "#" * 90)
            print(f"# Sweep step {i+1}/{len(values)}: {param_name} = {value}")
            print("#" * 90)

        cfg = replace(base_config, **{param_name: value}, x0_override=x0_carry)

        # From the second step onward: narrow the search space around the
        # previous theta (saves time, since we already roughly know where
        # the solution lies) -- only if the caller hasn't already set
        # dT_upper.
        if theta_carry is not None and cfg.dT_upper is None:
            margin = 5.0  # K, deliberately generous around the previous point
            new_floor = {
                k: max(0.5, theta_carry[j] - margin) for j, k in enumerate(THETA_ORDER)
            }
            new_upper = {k: theta_carry[j] + margin for j, k in enumerate(THETA_ORDER)}
            cfg = replace(cfg, dT_floor=new_floor, dT_upper=new_upper)

        try:
            theta_opt, result, _ = optimize_design_point(cfg, verbose=verbose)
        except Exception as exc:  # pragma: no cover
            print(f"  Sweep step {param_name}={value} failed: {exc}")
            continue

        if not _is_feasible(result):
            print(
                f"  WARNING: {param_name}={value} yielded no feasible result "
                "-- skipping, sweep continues."
            )
            continue

        results.append((value, theta_opt, result))
        theta_carry = theta_opt
        x0_carry = np.array(list(result.primary_variables.values()), dtype=float)

    return results


# ---------------------------------------------------------------------------
# Stage 2: part-load verification (simulation only, no optimization)
# ---------------------------------------------------------------------------

def build_ua_inputs(
    scenario: PartLoadScenario,
    design_ua: Dict[str, float],
    design_flows: Dict[str, float],
    config: DesignPointConfig,
) -> "UAInputs":
    """Builds the AHT UA-model inputs for a part-load point.

    ASSUMPTION (see module docstring): desorber/evaporator/condenser as
    fixed mass-flow kwargs (m_13, m_15, m_17); ONLY the absorber (the
    design-determining component group, analogous to the evaporator for
    the chiller) is selectable via absorber_spec_mode="m11"/"T12";
    cycle_scale_spec_mode="m6". shex_model="UA" with explicit UA_shex, so
    the SHEX UA value optimized in stage 1 is not ignored (NOT
    "NTU"/Effectiveness_shex).
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
        m_13=design_flows["m13_kg_s"],
        m_15=design_flows["m15_kg_s"],
        m_17=design_flows["m17_kg_s"],
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="m6",
        m6_spec=design_flows["m6_kg_s"],
        absorber_spec_mode="m11",
        m11_spec=design_flows["m11_kg_s"],
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
        shex_model="UA",
    )


def verify_part_load(
    scenarios: List[PartLoadScenario],
    design_result,
    config: DesignPointConfig,
) -> None:
    if not UA_MODEL_AVAILABLE:
        print(
            "Could not import the AHT UA model "
            f"({_ua_import_error!r}). Skipping stage 2.\n"
            "  -> If the module has a different name/location: adjust the\n"
            "     path/class names in the script, or send me the main script/source."
        )
        return

    if not _is_feasible(design_result) or not design_result.UA_conversion:
        print(
            "Skipping stage 2: the design point from stage 1 is not valid "
            "(no UA_conversion present) -- see the warning from "
            "print_design_point_summary()."
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
        "m6_kg_s": design_result.diagnostics["m6_kg_s"],
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
                f"parameter mismatch with AHT_UA_LMTD: {exc}\n"
                "  -> Please send the AHT_UA_LMTD.py source/main script so"
                " build_ua_inputs() can be adjusted."
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
                f"[{scenario.name}] Result structure of the AHT UA model differs: {exc}\n"
                "  -> Please send the AHT_UA_LMTD.py source for comparison."
            )
            continue

        dt = time.perf_counter() - t0
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"\n[{scenario.name}] Status: {status}  ({dt:.2f} s)")

        if feasible:
            cache.update(np.array(list(result.primary_variables.values()), dtype=float))
            print(f"  Q_abs = {result.heat_flows_kW.get('Q_abs', float('nan')):.3f} kW")
            for k, v in result.kpis.items():
                try:
                    print(f"  {k:6s}= {float(v):.4f}")
                except (TypeError, ValueError):
                    pass
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

    if config.run_feasibility_probe:
        quick_feasibility_probe(config)
        print(
            "\nFeasibility test complete. If everything above (fast) was "
            "INFEASIBLE, adjust bounds/boundary conditions/x0_override "
            "before starting the full optimization (see chat explanation)."
        )

    theta_opt, design_result, stats = optimize_design_point(config)
    print_design_point_summary(theta_opt, design_result)

    if config.run_stage2:
        t0 = time.perf_counter()
        verify_part_load(DEFAULT_SCENARIOS, design_result, config)
        t_stage2 = time.perf_counter() - t0
        print(f"\nStage 2 duration: {t_stage2:.1f} s")
    else:
        print("\nStage 2 skipped (config.run_stage2 = False).")

    t_total = time.perf_counter() - t_total_start
    print(f"Total duration  : {t_total/60:.2f} min")
