"""Operating map of the reversible plant in heat-transformer (AHT) mode.

Goal
----
The reversible plant uses the SAME machine as in chiller (AC) mode: all UA
values are taken 1:1 from the AC (PartLoadMapConfig in AC_part_load_map.py),
the vessels keep their function (desorber stays desorber, ...). Only the
pressure levels swap. This script maps the MAXIMUM heat output of the AHT
over its operating conditions for the techno-economic analysis (TEA).

Plant (heat transformer as booster for the district-heating supply)
-------------------------------------------------------------------
- source loop (from the second geothermal heat exchanger): T13 -> desorber
  -> T14 = T15 -> evaporator -> T16, one stream m13 = m15 (series)
- district heating: T11 = 75 °C (fixed) -> absorber -> T12 = DH supply
  (absorber_spec_mode "T12"; the DH flow m11 follows)
- condenser: cooling water T17 -> T18 (dry cooler)

Stream numbers are the AHT model's own -- they differ from the AC model:

    vessel       UA      AC streams                 AHT streams
    desorber     UA_des  11->12 geothermal drive    13->14 source (series, 1st)
    absorber     UA_abs  13->14 cooling water       11->12 district heating
    condenser    UA_cond 15->16 cooling water       17->18 cooling water
    evaporator   UA_evap 17->18 chilled water       15->16 source (series, 2nd)

Map axes
--------
    T_source   source-loop inlet T13            (e.g. 60-90 °C)
    T_cond     cooling-water inlet T17          (e.g. 5-30 °C)
    T_DH       district-heating supply T12      (e.g. 85/90/95 °C)
    m_cw       cooling-water flow / AC design cooling-water flow (axis, so the
               TEA can price fan and pump power, as in the AC map)

Operation per map point: the plant always runs at MAXIMUM heat output while
holding the DH supply temperature. The decision variable is the solution
circulation m6 (variable-speed solution pump): Q_abs has a clear maximum
over m6 (too little -> too little absorption, too much -> sensible losses).
The source flow is fixed at its hydraulic limit (source_flow_ratio x AC
design desorber flow -- in series the desorber, sized for the smaller AC
drive flow, limits the common stream). The source outlet temperature and
the DH flow are RESULTS; the DH flow is reported relative to the AC design
absorber flow, because it is far below it (20 K spread instead of ~2 K) --
exactly where constant UA values overestimate the real heat transfer.

Numerics
--------
Per point: scan of m6 on a geometric grid (warm-started continuation, stops
once past the maximum), then bounded refinement around the best scan point.
Warm starts come from computed neighbors. Infeasibility is monotone in
"harder" direction (colder source, warmer cooling water): if a point is
infeasible after a feasible one in walk direction, harder points are
skipped. Every solve runs in a worker process with a wall-clock limit;
results are appended to the CSV per point, and a restart resumes from it
(the settings are checked against <name>_config.json; only the grid may
change between runs with the same output_name).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

import Thermodynamic_Properties.libr_props as lp
from Models.AHT_UA_LMTD import (
    AHTInputs,
    AHTResult,
    PRIMARY_VARIABLE_NAMES,
    initial_guess,
    solve_aht,
)
from AC_part_load_map import (
    PartLoadMapConfig,
    TimedSolver,
    _adjacent,
    _annotated_map,
    _finite_range,
    _outward_order,
    solve_design as solve_ac_design,
)

RESIDUAL_TOL = 1.0e-6

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AHTMapConfig:
    # --- Machine: identical to the AC (UA values + design flows from there) --
    ac: PartLoadMapConfig = field(default_factory=PartLoadMapConfig)

    # --- Fixed boundary conditions -------------------------------------------
    T_DH_return_C: float = 75.0          # absorber inlet T11 (after the first geothermal HX)
    routing: str = "series_desorber_to_evaporator"
    source_flow_ratio: float = 1.5       # source flow = ratio x AC design desorber flow (hydraulic limit)
    cp_w_kJkgK: float = 4.2

    # --- Map grid ------------------------------------------------------------
    T_source_values_C: Tuple[float, ...] = (60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0)
    T_cond_values_C: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    T_DH_supply_values_C: Tuple[float, ...] = (85.0, 90.0, 95.0)
    m_cw_ratio_values: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.25, 1.5)   # x AC design cooling-water flow

    # --- Solution circulation m6 (decision variable) ---------------------------
    m6_min_kg_s: float = 0.15
    m6_max_kg_s: float = 8.0
    m6_scan_points: int = 14             # geometric scan grid
    max_consecutive_failures: int = 2
    past_peak_steps: int = 2             # stop the scan after Q_abs fell this many times in a row
    refine_maxiter: int = 10

    # --- Validity ---------------------------------------------------------------
    probe_max_nfev: int = 120
    solve_timeout_s: Optional[float] = 60.0

    # --- Output -----------------------------------------------------------------
    output_name: str = "AHT_part_load_map"


GRID_FIELDS = ("T_source_values_C", "T_cond_values_C", "T_DH_supply_values_C",
               "m_cw_ratio_values", "output_name")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NominalFlows:
    m_source_kg_s: float      # source flow (fixed at its hydraulic limit)
    m_cw_design_kg_s: float   # AC design cooling-water flow (reference of the m_cw axis)
    m_abs_ac_kg_s: float      # AC design absorber flow (reference for the DH flow)


@dataclass(frozen=True)
class AHTPoint:
    m6_kg_s: float
    Q_abs_kW: float
    Q_evap_kW: float
    Q_des_kW: float
    Q_cond_kW: float
    COP: float
    m_DH_kg_s: float          # district-heating flow through the absorber (m11)
    m_source_kg_s: float
    m_cw_kg_s: float
    T14_C: float              # source between desorber and evaporator
    T16_C: float              # source outlet
    T18_C: float              # cooling-water outlet
    FR: float
    dw_cryst: float
    pinch_min_K: float
    pinch_min_at: str
    x: np.ndarray = field(repr=False)


@dataclass
class AHTMapPoint:
    T_source_C: float
    T_cond_C: float
    T_DH_C: float
    m_cw_ratio: float
    optimum: Optional[AHTPoint]
    n_solves: int
    runtime_s: float
    status: str = "ok"        # "ok", "infeasible" or "skipped"
    limit: str = ""           # "capacity" (an easier point in walk direction was feasible) or "unknown"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


def _x_from_result(result: AHTResult) -> np.ndarray:
    return np.array([result.primary_variables[k] for k in PRIMARY_VARIABLE_NAMES], dtype=float)


def _crystallization_margin(result: AHTResult) -> float:
    """Minimum of w_crit(T) - w over the solution states the model checks."""
    margin = math.inf
    for label in ("3", "4", "5", "6", "20"):
        s = result.states[label]
        if s["w_LiBr"] <= 0.57:        # below the range of the crystallization correlation
            continue
        try:
            margin = min(margin, lp.crystallization_limit("T", s["T_K"]) - s["w_LiBr"])
        except Exception:
            return -math.inf
    return margin


def nominal_flows(cfg: AHTMapConfig) -> NominalFlows:
    ac = solve_ac_design(cfg.ac)
    return NominalFlows(
        m_source_kg_s=cfg.source_flow_ratio * ac.m11_kg_s,
        m_cw_design_kg_s=ac.m15_kg_s,
        m_abs_ac_kg_s=ac.m13_kg_s,
    )


def _build_inputs(cfg: AHTMapConfig, nom: NominalFlows, T_source: float, T_cond: float,
                  T_DH: float, m_cw_ratio: float, m6: float) -> AHTInputs:
    ac = cfg.ac
    return AHTInputs(
        T_11_C=cfg.T_DH_return_C,
        T_13_C=T_source if cfg.routing != "series_evaporator_to_desorber" else None,
        T_15_C=T_source if cfg.routing != "series_desorber_to_evaporator" else None,
        T_17_C=T_cond,
        m_13=nom.m_source_kg_s, m_15=nom.m_source_kg_s, m_17=m_cw_ratio * nom.m_cw_design_kg_s,
        UA_cond=ac.UA_cond, UA_evap=ac.UA_evap, UA_abs=ac.UA_abs, UA_des=ac.UA_des,
        UA_shex=ac.UA_shex, shex_model="UA", desorber_evaporator_routing_mode=cfg.routing,
        cycle_scale_spec_mode="m6", m6_spec=m6,
        absorber_spec_mode="T12", T12_spec_C=T_DH,
        cp_w_kJkgK=cfg.cp_w_kJkgK, max_nfev=cfg.probe_max_nfev,
    )


class Evaluator:
    """Solves one AHT operating point, trying several warm starts; counts solves."""

    def __init__(self, cfg: AHTMapConfig, nom: NominalFlows):
        self.cfg = cfg
        self.nom = nom
        self.n_solves = 0
        self.solver = TimedSolver(cfg.solve_timeout_s, solve_fn=solve_aht)

    def __call__(self, T_source: float, T_cond: float, T_DH: float, m_cw_ratio: float, m6: float,
                 x0_candidates: Sequence[np.ndarray]) -> Optional[AHTPoint]:
        try:
            inputs = _build_inputs(self.cfg, self.nom, T_source, T_cond, T_DH, m_cw_ratio, m6)
        except ValueError:
            return None
        for x0 in list(x0_candidates) + [initial_guess(inputs)]:
            self.n_solves += 1
            try:
                result = self.solver(inputs, x0)
            except Exception:
                continue
            info = result.solve_info
            if (not info.success or not info.final_point_evaluable or not result.flash_outputs
                    or info.scaled_residual_norm > RESIDUAL_TOL):
                continue
            if not all(v for k, v in result.checks.items() if k != "crystallization_safe_all_checked_states"):
                continue
            hf = result.heat_flows_kW
            pinches = {k[len("Pinch_"):]: v for k, v in result.lmtd_K.items() if k.startswith("Pinch_")}
            pinch_at = min(pinches, key=pinches.get) if pinches else ""
            return AHTPoint(
                m6_kg_s=m6, Q_abs_kW=hf["Q_abs"], Q_evap_kW=hf["Q_evap"], Q_des_kW=hf["Q_des"],
                Q_cond_kW=hf["Q_cond"], COP=result.kpis["COP"],
                m_DH_kg_s=result.diagnostics["m11_kg_s"], m_source_kg_s=inputs.m_13,
                m_cw_kg_s=inputs.m_17,
                T14_C=result.states["14"]["T_K"] - 273.15, T16_C=result.states["16"]["T_K"] - 273.15,
                T18_C=result.states["18"]["T_K"] - 273.15, FR=result.kpis["FR"],
                dw_cryst=_crystallization_margin(result),
                pinch_min_K=pinches[pinch_at] if pinches else float("nan"), pinch_min_at=pinch_at,
                x=_x_from_result(result),
            )
        return None


# ---------------------------------------------------------------------------
# Maximum heat output over the solution circulation m6
# ---------------------------------------------------------------------------


def maximize_heat_output(
    cfg: AHTMapConfig, solve_at: Callable[[float, Sequence[np.ndarray]], Optional[AHTPoint]],
    x0_candidates: Sequence[np.ndarray], m6_hint: Optional[float] = None,
) -> Optional[AHTPoint]:
    """Scan m6 on a geometric grid (continuation), stop once past the maximum,
    then refine between the neighbors of the best scan point. If a neighbor's
    optimum m6 is known, the scan starts a few steps below it."""
    grid = list(np.geomspace(cfg.m6_min_kg_s, cfg.m6_max_kg_s, cfg.m6_scan_points))
    if m6_hint is not None:
        i_hint = int(np.argmin([abs(math.log(m / m6_hint)) for m in grid]))
        grid = grid[max(i_hint - 3, 0):]

    scan: List[Tuple[int, AHTPoint]] = []
    x_chain = list(x0_candidates)
    failures = drops = 0
    for i, m6 in enumerate(grid):
        op = solve_at(m6, x_chain)
        if op is None:
            failures += 1
            if scan and failures >= cfg.max_consecutive_failures:
                break
            continue
        failures = 0
        if scan and op.Q_abs_kW < scan[-1][1].Q_abs_kW:
            drops += 1
        else:
            drops = 0
        scan.append((i, op))
        x_chain = [op.x]
        if drops >= cfg.past_peak_steps:
            break
    if not scan:
        return None

    i_best, best = max(scan, key=lambda item: item[1].Q_abs_kW)
    lo = grid[max(i_best - 1, 0)]
    hi = grid[min(i_best + 1, len(grid) - 1)]
    if cfg.refine_maxiter > 0 and hi > lo:
        cache: List[AHTPoint] = []

        def objective(m6: float) -> float:
            op = solve_at(m6, [best.x])
            if op is None:
                return 0.0           # Q_abs > 0 for every feasible point
            cache.append(op)
            return -op.Q_abs_kW

        minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                        options={"maxiter": cfg.refine_maxiter, "xatol": 1.0e-3 * hi})
        if cache:
            refined = max(cache, key=lambda op: op.Q_abs_kW)
            if refined.Q_abs_kW > best.Q_abs_kW:
                best = refined
    return best


# ---------------------------------------------------------------------------
# Map computation
# ---------------------------------------------------------------------------


def compute_aht_map(
    cfg: AHTMapConfig, *, csv_path: Optional[Path] = None, resume: bool = True,
) -> Tuple[NominalFlows, List[AHTMapPoint]]:
    nom = nominal_flows(cfg)
    print(f"Machine from the AC design: source flow = {nom.m_source_kg_s:.3f} kg/s "
          f"({cfg.source_flow_ratio:.2f} x AC desorber flow), cooling-water design flow = "
          f"{nom.m_cw_design_kg_s:.3f} kg/s, AC absorber flow = {nom.m_abs_ac_kg_s:.3f} kg/s")
    evaluate = Evaluator(cfg, nom)

    S_vals, K_vals = cfg.T_source_values_C, cfg.T_cond_values_C
    D_vals, C_vals = cfg.T_DH_supply_values_C, cfg.m_cw_ratio_values
    done: Dict[Tuple[int, int, int, int], AHTPoint] = {}       # (i_DH, i_cw, i_src, i_cond)
    failed: set = set()                                        # capacity-limited or skipped keys
    points: List[AHTMapPoint] = []

    previous = _read_csv(csv_path) if (resume and csv_path is not None) else None
    if previous is None:
        if csv_path is not None:
            _write_csv_header(csv_path)
            _write_config(cfg, _config_path(csv_path))
    else:
        _check_config(cfg, _config_path(csv_path))
        print(f"Resuming from {csv_path}: {len(previous)} points already computed")
        for (s, k, d, c), point in previous.items():
            if s in S_vals and k in K_vals and d in D_vals and c in C_vals:
                key = (D_vals.index(d), C_vals.index(c), S_vals.index(s), K_vals.index(k))
                if point.optimum is not None:
                    done[key] = point.optimum
                elif point.limit == "capacity":
                    failed.add(key)

    n_total = len(S_vals) * len(K_vals) * len(D_vals) * len(C_vals)
    t_run = time.time()
    D_order = sorted(range(len(D_vals)), key=lambda i: D_vals[i])          # easy (low supply) first
    C_order = _outward_order(C_vals, 1.0)
    S_order = sorted(range(len(S_vals)), key=lambda i: -S_vals[i])         # easy (hot source) first
    K_order = sorted(range(len(K_vals)), key=lambda i: K_vals[i])          # easy (cold cooling) first

    def report(point: AHTMapPoint, msg: str) -> None:
        print(f"[{len(points):3d}/{n_total}, {(time.time() - t_run) / 60:5.0f} min] "
              f"T_DH={point.T_DH_C:4.1f} m_cw={point.m_cw_ratio:4.2f}x T_src={point.T_source_C:4.1f} "
              f"T_cond={point.T_cond_C:4.1f} °C ({point.runtime_s:5.1f} s)  {msg}", flush=True)

    try:
        for d in D_order:
            for c in C_order:
                slice_feasible = any(key[:2] == (d, c) for key in done)
                for s in S_order:
                    for k in K_order:
                        t0, n0 = time.time(), evaluate.n_solves
                        T_src, T_cond, T_DH, r_cw = S_vals[s], K_vals[k], D_vals[d], C_vals[c]
                        if previous is not None and (T_src, T_cond, T_DH, r_cw) in previous:
                            points.append(previous[(T_src, T_cond, T_DH, r_cw)])
                            continue

                        # Easier neighbors: hotter source, colder cooling water.
                        easier = [(d, c, _adjacent(S_vals, s, +1), k), (d, c, s, _adjacent(K_vals, k, -1))]
                        if any(key in failed for key in easier):
                            failed.add((d, c, s, k))
                            point = AHTMapPoint(T_src, T_cond, T_DH, r_cw, None, 0, 0.0,
                                                status="skipped", limit="capacity")
                            points.append(point)
                            if csv_path is not None:
                                _append_csv_row(csv_path, point)
                            report(point, "skipped (an easier neighbor is beyond the capacity limit)")
                            continue

                        neighbor_keys = [(d, c, s, _adjacent(K_vals, k, -1)), (d, c, _adjacent(S_vals, s, +1), k),
                                         (d, c, s, _adjacent(K_vals, k, +1)), (d, c, _adjacent(S_vals, s, -1), k),
                                         (d, _adjacent(C_vals, c, +1), s, k), (d, _adjacent(C_vals, c, -1), s, k),
                                         (_adjacent(D_vals, d, -1), c, s, k), (_adjacent(D_vals, d, +1), c, s, k)]
                        neighbors = [done[key] for key in neighbor_keys if key in done]
                        cands = [op.x for op in neighbors][:3]
                        m6_hint = neighbors[0].m6_kg_s if neighbors else None

                        optimum = maximize_heat_output(
                            cfg, lambda m6, xs: evaluate(T_src, T_cond, T_DH, r_cw, m6, xs), cands, m6_hint)
                        if optimum is None and m6_hint is not None:       # retry with the full m6 scan
                            optimum = maximize_heat_output(
                                cfg, lambda m6, xs: evaluate(T_src, T_cond, T_DH, r_cw, m6, xs), cands)

                        limit = ""
                        if optimum is not None:
                            done[(d, c, s, k)] = optimum
                            slice_feasible = True
                        elif slice_feasible:
                            limit = "capacity"
                            failed.add((d, c, s, k))
                        else:
                            limit = "unknown"
                        point = AHTMapPoint(T_src, T_cond, T_DH, r_cw, optimum, evaluate.n_solves - n0,
                                            time.time() - t0, status="ok" if optimum else "infeasible",
                                            limit=limit)
                        points.append(point)
                        if csv_path is not None:
                            _append_csv_row(csv_path, point)
                        if optimum is None:
                            msg = f"infeasible ({limit} limit)"
                        else:
                            o = optimum
                            msg = (f"Q_abs={o.Q_abs_kW:6.1f} kW COP={o.COP:.3f} m6={o.m6_kg_s:.2f} "
                                   f"m_DH={o.m_DH_kg_s:.2f} T_src_out={o.T16_C:5.1f} "
                                   f"| pinch {o.pinch_min_K:.2f} K ({o.pinch_min_at}) dw={o.dw_cryst:.3f}")
                        report(point, msg)
    finally:
        evaluate.solver.close()
    print(f"Solver calls: {evaluate.n_solves}, timeouts: {evaluate.solver.n_timeouts}, "
          f"worker restarts: {evaluate.solver.n_worker_restarts}")
    return nom, points


# ---------------------------------------------------------------------------
# CSV, config check
# ---------------------------------------------------------------------------

X_FIELDS = tuple(f"x_{name}" for name in PRIMARY_VARIABLE_NAMES)
CSV_FIELDS = ("T_source_C", "T_cond_C", "T_DH_supply_C", "m_cw_ratio", "status", "limit", "feasible",
              "Q_abs_kW", "COP", "m6_kg_s", "Q_evap_kW", "Q_des_kW", "Q_cond_kW", "m_source_kg_s",
              "T14_C", "T_source_out_C", "m_DH_kg_s", "m_cw_kg_s", "T_cond_out_C", "FR", "dw_cryst",
              "pinch_min_K", "pinch_min_at", "n_solves", "runtime_s") + X_FIELDS


def _csv_row(p: AHTMapPoint) -> Dict[str, object]:
    row: Dict[str, object] = {k: float("nan") for k in CSV_FIELDS}
    row.update({"T_source_C": p.T_source_C, "T_cond_C": p.T_cond_C, "T_DH_supply_C": p.T_DH_C,
                "m_cw_ratio": p.m_cw_ratio, "status": p.status, "limit": p.limit,
                "feasible": p.optimum is not None, "n_solves": p.n_solves,
                "runtime_s": round(p.runtime_s, 1)})
    o = p.optimum
    if o is not None:
        row.update({"Q_abs_kW": o.Q_abs_kW, "COP": o.COP, "m6_kg_s": o.m6_kg_s, "Q_evap_kW": o.Q_evap_kW,
                    "Q_des_kW": o.Q_des_kW, "Q_cond_kW": o.Q_cond_kW, "m_source_kg_s": o.m_source_kg_s,
                    "T14_C": o.T14_C, "T_source_out_C": o.T16_C, "m_DH_kg_s": o.m_DH_kg_s,
                    "m_cw_kg_s": o.m_cw_kg_s, "T_cond_out_C": o.T18_C, "FR": o.FR, "dw_cryst": o.dw_cryst,
                    "pinch_min_K": o.pinch_min_K, "pinch_min_at": o.pinch_min_at})
        row.update(dict(zip(X_FIELDS, o.x.tolist())))
    return row


def _write_csv_header(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_csv_row(path: Path, point: AHTMapPoint) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(_csv_row(point))


def _read_csv(path: Path) -> Optional[Dict[Tuple[float, float, float, float], AHTMapPoint]]:
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            print(f"{path} has different columns -- not resumable, starting from scratch.")
            return None
        rows = list(reader)
    points: Dict[Tuple[float, float, float, float], AHTMapPoint] = {}
    for r in rows:
        num = lambda name: float(r[name])
        key = (num("T_source_C"), num("T_cond_C"), num("T_DH_supply_C"), num("m_cw_ratio"))
        optimum = None
        if r["status"] == "ok":
            optimum = AHTPoint(
                m6_kg_s=num("m6_kg_s"), Q_abs_kW=num("Q_abs_kW"), Q_evap_kW=num("Q_evap_kW"),
                Q_des_kW=num("Q_des_kW"), Q_cond_kW=num("Q_cond_kW"), COP=num("COP"),
                m_DH_kg_s=num("m_DH_kg_s"), m_source_kg_s=num("m_source_kg_s"),
                m_cw_kg_s=num("m_cw_kg_s"), T14_C=num("T14_C"),
                T16_C=num("T_source_out_C"), T18_C=num("T_cond_out_C"), FR=num("FR"),
                dw_cryst=num("dw_cryst"), pinch_min_K=num("pinch_min_K"), pinch_min_at=r["pinch_min_at"],
                x=np.array([num(name) for name in X_FIELDS]),
            )
        points[key] = AHTMapPoint(key[0], key[1], key[2], key[3], optimum, int(r["n_solves"]),
                                  float(r["runtime_s"]), status=r["status"], limit=r["limit"])
    return points


def _config_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_config.json")


def _config_signature(cfg: AHTMapConfig) -> Dict[str, object]:
    """All settings that influence the results (everything but the grid; the
    nested AC settings define the machine and are included completely,
    except the AC's own grid and output name)."""
    signature = {k: v for k, v in asdict(cfg).items() if k not in GRID_FIELDS}
    for k in ("plr_values", "T_reject_values_C", "m_cw_ratio_values", "output_name"):
        signature["ac"].pop(k, None)
    return json.loads(json.dumps(signature))


def _write_config(cfg: AHTMapConfig, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_config_signature(cfg), f, indent=2)


def _check_config(cfg: AHTMapConfig, path: Path) -> None:
    if not path.exists():
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
            f"settings:\n{details}\nSet a new output_name in AHTMapConfig to compute this "
            f"configuration as a separate map (or delete the old CSV to overwrite it)."
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_aht_map(cfg: AHTMapConfig, nom: NominalFlows, points: Sequence[AHTMapPoint], *,
                 save_dir: Optional[Path], show: bool = True):
    import matplotlib.pyplot as plt

    K_vals = sorted(set(cfg.T_cond_values_C))
    S_vals = sorted(set(cfg.T_source_values_C))
    D_vals = sorted(set(cfg.T_DH_supply_values_C))
    C_vals = sorted(set(cfg.m_cw_ratio_values))
    r_ref = min(C_vals, key=lambda r: abs(r - 1.0))
    figs = []

    def grid(T_DH: float, r_cw: float, value: Callable[[AHTPoint], float]) -> np.ndarray:
        Z = np.full((len(S_vals), len(K_vals)), np.nan)
        for p in points:
            if p.optimum and p.T_DH_C == T_DH and math.isclose(p.m_cw_ratio, r_cw):
                Z[S_vals.index(p.T_source_C), K_vals.index(p.T_cond_C)] = value(p.optimum)
        return Z

    panels = [
        ("Q_abs max [kW]", lambda o: o.Q_abs_kW, "viridis", "{:.0f}"),
        ("COP [-]", lambda o: o.COP, "viridis", "{:.3f}"),
        ("source outlet T16 [°C]", lambda o: o.T16_C, "magma", "{:.1f}"),
        ("DH flow / AC absorber flow [-]", lambda o: o.m_DH_kg_s / nom.m_abs_ac_kg_s, "cividis", "{:.2f}"),
        ("solution circulation m6 [kg/s]", lambda o: o.m6_kg_s, "cividis", "{:.2f}"),
    ]
    fig, axes = plt.subplots(len(D_vals), len(panels), figsize=(4.8 * len(panels), 4.0 * len(D_vals)),
                             squeeze=False)
    for col, (label, value, cmap, fmt) in enumerate(panels):
        Zs = [grid(T_DH, r_ref, value) for T_DH in D_vals]
        lo, hi = _finite_range(Zs)
        for row, (T_DH, Z) in enumerate(zip(D_vals, Zs)):
            ax = axes[row][col]
            mesh = _annotated_map(ax, Z, K_vals, S_vals, cmap=cmap, vmin=lo, vmax=hi, fmt=fmt,
                                  title=label if row == 0 else "", fontsize=6.5)
            ax.set_xlabel("cooling-water inlet T17 [°C]")
            ax.set_ylabel(f"DH supply {T_DH:.0f} °C\nsource inlet T13 [°C]" if col == 0 else "")
        fig.colorbar(mesh, ax=axes[:, col].tolist(), shrink=0.8, pad=0.02)
    fig.suptitle(f"AHT at maximum heat output (m_cw = {r_ref:.2f} x design, DH return "
                 f"{cfg.T_DH_return_C:.0f} °C, source {nom.m_source_kg_s:.2f} kg/s, {cfg.routing})")
    figs.append(("overview", fig))

    # Q_abs vs cooling-water flow: what the TEA trades against fan/pump power
    fig, axes = plt.subplots(1, len(D_vals), figsize=(5.0 * len(D_vals), 4.4), sharey=True, squeeze=False)
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(K_vals)))
    T_src_ref = S_vals[len(S_vals) // 2]
    for ax, T_DH in zip(axes[0], D_vals):
        for color, T_cond in zip(colors, K_vals):
            pts = sorted((p.m_cw_ratio, p.optimum.Q_abs_kW) for p in points if p.optimum
                         and p.T_DH_C == T_DH and p.T_cond_C == T_cond and p.T_source_C == T_src_ref)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-o", color=color, markersize=3.5, label=f"{T_cond:.0f} °C")
        ax.set_title(f"DH supply {T_DH:.0f} °C, source {T_src_ref:.0f} °C", fontsize=10)
        ax.set_xlabel("m_cw / m_cw_design [-]")
        ax.grid(alpha=0.4)
    axes[0][0].set_ylabel("Q_abs max [kW]")
    axes[0][-1].legend(title="cooling water", fontsize=7.5, title_fontsize=8)
    fig.tight_layout()
    figs.append(("Qabs_vs_mcw", fig))

    if save_dir is not None:
        for suffix, f in figs:
            path = save_dir / f"{cfg.output_name}_{suffix}.png"
            f.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {path}")
    if show:
        plt.show()
    return figs


if __name__ == "__main__":
    config = AHTMapConfig()

    # All assumptions live centrally in AHTMapConfig (machine: in its nested
    # PartLoadMapConfig of the AC) -- adjust them there, not here.
    plot_dir = Path(__file__).resolve().parent / "Plots"
    plot_dir.mkdir(exist_ok=True)
    csv_file = plot_dir / f"{config.output_name}.csv"

    t_start = time.time()
    nominal, map_points = compute_aht_map(config, csv_path=csv_file)
    print(f"Total runtime: {(time.time() - t_start) / 3600:.2f} h")
    print(f"CSV (written point by point): {csv_file}")
    plot_aht_map(config, nominal, map_points, save_dir=plot_dir, show=False)
