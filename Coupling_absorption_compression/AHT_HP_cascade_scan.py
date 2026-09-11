"""Four-way comparison: AHT / compression HP configurations for lifting
data-center waste heat into a district-heating (DH) network.

Configurations compared
------------------------
    1. DC -> AHT -> HP -> DH   (AHT first, HP tops up the remaining lift)
    2. DC -> AHT -> DH         (AHT alone)
    3. DC -> HP  -> DH         (HP alone)
    4. DC -> HP  -> AHT -> DH  (HP first, AHT tops up the remaining lift)

All four boundary streams are fixed, T-mode (both inlet AND outlet given,
mass flow computed by whichever model owns that stream):
    - DC waste heat   : T_DC_in_C -> T_DC_out_C = T_DC_in_C - dT_DC_C
    - AHT's own reject cooling (independent utility, used in every
      configuration that includes an AHT, regardless of cascade position):
      T_reject_in_C -> T_reject_out_C = T_reject_in_C + dT_reject_C
    - DH network      : T_DH_return_C = T_DH_supply_C - dT_DH_C -> T_DH_supply_C

Configs 2 and 3 (single machine) are each a single, fully-determined design
point -- no free variable. Solve once; if it doesn't converge or fails a
plausibility check (crystallization, pinch, pressure ratio, ...), that
configuration is INFEASIBLE and dropped entirely, per your instruction.

Configs 1 and 4 (cascades) have exactly one free variable: the handover
("mid") temperature at which the first machine's product becomes the
second machine's raw material -- e.g. in config 1, the AHT absorber's own
external loop (T11 -> T12) is directly the HP's evaporator source loop.
This script sweeps T_mid and, for each value, checks whether the split is
achievable and what it costs in electricity.

Simplified captive-loop coupling (by design, not an oversight)
------------------------------------------------------------------
The loop between the two machines (AHT absorber <-> HP evaporator in
config 1; HP condenser <-> AHT desorber/evaporator in config 4) is a real,
physically shared loop -- same water, same mass flow, at both ends. A
fully rigorous treatment would iterate that loop's mass flow and both of
its temperatures until the two sub-models agree (a "tear stream").

This script does NOT do that. Instead:
    - the loop's temperature GLIDE (dT_mid_C) is a fixed assumption, same
      as every other external stream here
    - the two machines are solved in CAUSAL order: whichever one sits
      LAST in the chain (i.e. is fully pinned by the fixed DH boundary
      conditions) is solved FIRST, and its own energy balance tells you
      exactly how much duty (kW) the upstream machine needs to deliver
      into the shared loop -- that duty (not a matched mass flow) is
      passed forward as the upstream machine's own scale spec
      (Qabs_spec_kW for the AHT, Q_cond_kW for the HP)

This sidesteps the need for an iterative solve while still being duty-
consistent at the interface (the whole point of a cascade -- how much
heat moves and at what temperature -- is preserved). What it does NOT
guarantee is that the two machines' independently-computed mass flows for
that shared loop numerically agree (the AHT model uses a constant cp_w
approximation for external streams throughout this repo; the HP model
uses TESPy's real water properties) -- a genuine but, for this first-pass
targeting exercise, secondary concern. If you need the rigorous version
later, this is the one part of the model that would need to become an
iterative (tear-stream) solve instead of two straight-line calls.

Standalone usage
-----------------
    python Coupling_absorption_compression/AHT_HP_cascade_scan.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from Compression_Models.HeatPump_Pinch_Point import HeatPumpInputs, HeatPumpModel
from Models.AHT_Pinch_Point import AHTInputs, AHTResult, PRIMARY_VARIABLE_NAMES, initial_guess, solve_aht
from Design_Point.AHT_duehring_screening import estimate_max_gtl


RESIDUAL_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    T_waste_heat_in_C: float = 65.0
    dT_waste_heat_C: float = 5.0

    T_demand_supply_C: float = 100.0
    dT_demand_C: float = 15.0

    T_reject_in_C: float = 15.0
    dT_reject_C: float = 5.0

    Q_total_kW: float = 500.0
    cp_w_kJkgK: float = 4.18

    dT_mid_C: float = 5.0

    # AHT pinches 
    dT_min_shex: float = 5.0
    dT_min_des: float = 5.0
    dT_min_cond: float = 5.0
    dT_min_evap: float = 5.0
    dT_min_abs: float = 5.0

    min_GTL_K: float = 3.0

    # Relaxed solver tolerances for the T_mid sweeps
    probe_solver_tol: float = 1.0e-6
    probe_max_nfev: int = 300

    # Compression HP assumptions
    refrigerant: str = "R1233zd(E)"
    dT_min_evap_hp: float = 5.0
    dT_min_cond_hp: float = 5.0
    superheat_K: float = 3.0
    subcooling_K: float = 2.0
    T_discharge_max_C: float = 150.0
    PR_max: float = 6.0

    # eta_s(PR) correlation + motor/mechanical efficiency, same generic
    # parabola as Compression_Models/HeatPump_feasibility_sweep.py
    eta_s_max: float = 0.78
    PR_opt: float = 4.0
    eta_s_curvature: float = 0.015
    eta_s_min: float = 0.35
    eta_mech_motor: float = 0.94

    @property
    def T_DC_out_C(self) -> float:
        return self.T_waste_heat_in_C - self.dT_waste_heat_C

    @property
    def T_DH_return_C(self) -> float:
        return self.T_demand_supply_C - self.dT_demand_C

    @property
    def T_reject_out_C(self) -> float:
        return self.T_reject_in_C + self.dT_reject_C


T_MID_MARGIN_C = 2.0   # keep T_mid a bit clear of T_DC_in and T_DH_supply
T_MID_STEP_C = 2.0

plot_name = "Coupling_absorption_compression/Plots/aht_hp_configurations"


# ---------------------------------------------------------------------------
# AHT helpers
# ---------------------------------------------------------------------------

def _is_valid_aht_solution(result: AHTResult) -> bool:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _aht_duehring_T12_max(T13_C: float, T15_C: float, T17_C: float, config: ScenarioConfig) -> Optional[float]:
    """Optimistic (Duehring-correlation) upper bound on the AHT's
    achievable absorber outlet T12 for desorber/evaporator inlet
    T13=T15 and condenser (reject-cooling) inlet T17 -- a pure closed-form
    calculation, no solver involved (see
    Design_Point/AHT_duehring_screening.py). Independent of T11.

    Used as a cheap pre-filter: if even this optimistic ceiling can't
    reach the required T12, the real (pinch) model, which is strictly
    tighter, certainly can't either -- so that point can be skipped
    without a single solver attempt. The real achievable T12 is typically
    well below this bound (it ignores the SHEX pinch), so passing this
    check is necessary but not sufficient for feasibility.
    """
    result = estimate_max_gtl(
        T13_C=T13_C, T15_C=T15_C, T17_C=T17_C,
        dT_min_des=config.dT_min_des, dT_min_evap=config.dT_min_evap,
        dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
    )
    return result.T12_max_C if result.feasible else None


def _build_aht_inputs(
    *, T11_C: float, T12_C: float, T_ext_hot_C: float, T_ext_cold_C: float,
    T17_C: float, T18_C: float, Qabs_kW: float, config: ScenarioConfig,
    fast: bool = False,
) -> AHTInputs:
    """T11/T12 fix the absorber (useful-heat output). T_ext_hot/T_ext_cold
    fix BOTH desorber and evaporator (parallel routing -- the same
    external stream feeds both at the same temperature level). T17/T18
    fix the condenser (AHT's own reject-cooling utility). Qabs_kW is the
    overall plant-size spec.

    fast=True uses relaxed solver_tol/max_nfev (T_mid sweeps, many points,
    some expected to be infeasible); fast=False uses the AHTInputs
    defaults (the single-point solve_aht_alone() call)."""
    kwargs = dict(
        T_11_C=T11_C,
        T_13_C=T_ext_hot_C,
        T_15_C=T_ext_hot_C,
        T_17_C=T17_C,
        dT_min_shex=config.dT_min_shex,
        dT_min_des=config.dT_min_des,
        dT_min_cond=config.dT_min_cond,
        dT_min_evap=config.dT_min_evap,
        dT_min_abs=config.dT_min_abs,
        desorber_evaporator_routing_mode="parallel",
        cycle_scale_spec_mode="Qabs",
        Qabs_spec_kW=Qabs_kW,
        absorber_spec_mode="T12",
        T12_spec_C=T12_C,
        desorber_spec_mode="T14",
        T14_spec_C=T_ext_cold_C,
        evaporator_spec_mode="T16",
        T16_spec_C=T_ext_cold_C,
        condenser_spec_mode="T18",
        T18_spec_C=T18_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
    )
    if fast:
        kwargs["solver_tol"] = config.probe_solver_tol
        kwargs["max_nfev"] = config.probe_max_nfev
    return AHTInputs(**kwargs)


def _solve_aht(inputs: AHTInputs, x0: Optional[np.ndarray] = None):
    """Tries a continued warm start first, falls back to a fresh generic
    guess -- the AHT model's own solver has a narrow basin of attraction
    (see its module docstring), so a single fixed x0 strategy is not
    reliable on its own."""
    if x0 is not None:
        result = solve_aht(inputs, x0=x0)
        if _is_valid_aht_solution(result):
            return result
    return solve_aht(inputs, x0=initial_guess(inputs))


def _locate_aht_point(
    build_inputs_fn, T_target_C: float, *, x0_seed: Optional[np.ndarray] = None,
    span_C: float = 20.0, step_C: float = 1.0,
):
    """Finds a feasible AHT design point at (or, failing that, near)
    T_target_C. Tries T_target_C directly first (continuing from x0_seed
    if given, else a fresh cold start); if that fails, fans out in small
    steps in both directions from T_target_C -- mirrors
    Design_Point/AHT_feasibility_sweep.py's _locate_anchor(), needed
    because the AHT model's basin of attraction is often only ~2-4 K wide,
    so a direct jump to an arbitrary grid point can miss a genuinely
    feasible point.

    build_inputs_fn(T) -> AHTInputs builds the design point for handover
    temperature T (config 1 and config 4 wire T into different
    absorber/desorber/evaporator roles, hence the callback).

    Returns (T_found_C, x0_found, result_found), or (None, None, None) if
    nothing feasible was found anywhere in [T_target_C - span_C, T_target_C + span_C].
    """
    if x0_seed is not None:
        try:
            result = _solve_aht(build_inputs_fn(T_target_C), x0=x0_seed)
        except Exception:
            result = None
        if result is not None and _is_valid_aht_solution(result):
            x0 = np.array([result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES])
            return T_target_C, x0, result

    try:
        result = _solve_aht(build_inputs_fn(T_target_C))
    except Exception:
        result = None
    if result is not None and _is_valid_aht_solution(result):
        x0 = np.array([result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES])
        return T_target_C, x0, result

    n_steps = max(1, int(round(span_C / step_C)))
    for direction in (+1, -1):
        x0_walk = None
        for k in range(1, n_steps + 1):
            candidate = T_target_C + direction * k * step_C
            try:
                if x0_walk is not None:
                    result = _solve_aht(build_inputs_fn(candidate), x0=x0_walk)
                else:
                    result = _solve_aht(build_inputs_fn(candidate))
            except Exception:
                result = None
            if result is None:
                continue
            x0_walk = np.array([result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES])
            if _is_valid_aht_solution(result):
                return candidate, x0_walk, result

    return None, None, None


def _homotopy_walk_aht(
    build_inputs_fn, T_from_C: float, x0_from: np.ndarray, T_to_C: float,
    *, step_initial_C: float = 2.0, step_min_C: float = 0.25, max_steps: int = 100,
):
    """Moves the AHT leg's handover temperature from T_from_C to T_to_C in
    small, adaptive steps (halved on failure, grown on success), carrying
    x0 forward -- same continuation principle as
    Design_Point/AHT_feasibility_sweep.py's _homotopy_walk_T_waste().

    Returns (T_reached_C, x0_reached, result_at_T_reached_or_None, fully_reached).
    """
    direction = 1.0 if T_to_C > T_from_C else -1.0
    T_cur, x0_cur, result_cur = T_from_C, x0_from, None
    step = step_initial_C

    for _ in range(max_steps):
        remaining = direction * (T_to_C - T_cur)
        if remaining <= 1.0e-9:
            return T_cur, x0_cur, result_cur, True

        step_trial = min(step, remaining)
        T_trial = T_cur + direction * step_trial
        try:
            result = _solve_aht(build_inputs_fn(T_trial), x0=x0_cur)
        except Exception:
            result = None

        if result is not None and _is_valid_aht_solution(result):
            T_cur = T_trial
            x0_cur = np.array([result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES])
            result_cur = result
            step = min(step_trial * 1.5, step_initial_C)
        else:
            step = step_trial / 2.0
            if step < step_min_C:
                return T_cur, x0_cur, result_cur, False

    return T_cur, x0_cur, result_cur, False


def _advance_aht_leg(build_inputs_fn, T_target_C, state: dict):
    """Shared per-grid-point driver for both cascade sweeps: tries to
    reach T_target_C by walking forward from the last successful point
    (state['T'], state['x0']) if there is one, and falls back to a fresh
    anchor search directly at T_target_C otherwise (or if the walk gets
    stuck) -- exactly the fallback structure
    Design_Point/AHT_feasibility_sweep.py's
    sweep_relative_lift_window_homotopy() uses.

    Mutates `state` in place (keys 'T', 'x0') to track the last point
    reached, feasible or not, so the next call has the best available
    warm start. Returns (result_at_T_target_or_None, reason_str_or_None).
    """
    aht_result = None

    if state["x0"] is not None:
        T_reached, x0_reached, result_reached, fully_reached = _homotopy_walk_aht(
            build_inputs_fn, state["T"], state["x0"], T_target_C,
        )
        state["T"], state["x0"] = T_reached, x0_reached
        if fully_reached and result_reached is not None:
            aht_result = result_reached

    if aht_result is None:
        T_found, x0_found, result_found = _locate_aht_point(
            build_inputs_fn, T_target_C, x0_seed=state["x0"],
        )
        if T_found is None:
            state["T"], state["x0"] = None, None
            return None, "AHT: no valid solution"
        state["T"], state["x0"] = T_found, x0_found
        if abs(T_found - T_target_C) > 1.0e-6:
            return None, "AHT: no valid solution (nearest feasible point is elsewhere)"
        aht_result = result_found

    return aht_result, None


# ---------------------------------------------------------------------------
# HP helpers
# ---------------------------------------------------------------------------

def _eta_s_from_pr(PR: float, config: ScenarioConfig) -> float:
    eta = config.eta_s_max - config.eta_s_curvature * (PR - config.PR_opt) ** 2
    return float(np.clip(eta, config.eta_s_min, config.eta_s_max))


def _estimate_pr(T_source_out_C: float, T_sink_in_C: float, config: ScenarioConfig) -> float:
    T_evap_sat_K = (T_source_out_C - config.dT_min_evap_hp) + 273.15
    T_cond_sat_K = (T_sink_in_C + config.dT_min_cond_hp + config.subcooling_K) + 273.15
    p_evap = PropsSI("P", "T", T_evap_sat_K, "Q", 1, config.refrigerant)
    p_cond = PropsSI("P", "T", T_cond_sat_K, "Q", 0, config.refrigerant)
    return p_cond / p_evap


def _is_valid_hp_solution(result) -> bool:
    if result.kpis["pressure_ratio"] <= 1.0 + 1.0e-6:
        return False
    if result.kpis["T_lift_external_K"] <= 0.0:
        return False
    if result.kpis["COP"] <= 0.0:
        return False
    for label in ("0", "11", "21"):
        if result.states[label]["m_kg_s"] <= 0.0:
            return False
    return True


def _solve_hp(
    model: HeatPumpModel, *, T_source_in_C: float, T_source_out_C: float,
    T_sink_in_C: float, T_sink_out_C: float, Q_cond_kW: float, config: ScenarioConfig,
):
    PR_est = _estimate_pr(T_source_out_C, T_sink_in_C, config)
    eta_s = _eta_s_from_pr(PR_est, config)
    inputs = HeatPumpInputs(
        T_source_in_C=T_source_in_C,
        T_sink_in_C=T_sink_in_C,
        dT_min_evap=config.dT_min_evap_hp,
        dT_min_cond=config.dT_min_cond_hp,
        refrigerant=config.refrigerant,
        eta_s=eta_s,
        superheat_K=config.superheat_K,
        subcooling_K=config.subcooling_K,
        scale_spec_mode="Q_cond",
        Q_cond_kW=Q_cond_kW,
        source_spec_mode="T",
        T_source_out_C=T_source_out_C,
        sink_spec_mode="T",
        T_sink_out_C=T_sink_out_C,
    )
    try:
        result = model.solve(inputs)
    except Exception as exc:
        # Broad on purpose: TESPy's property backend can raise a raw
        # ValueError/RuntimeError directly out of nw.solve() on bad points
        # (e.g. a property flash failing near the refrigerant's critical
        # point), not just the HeatPumpEvaluationError this module raises
        # itself on non-convergence -- either way, treat it as infeasible.
        return None, f"solver failed: {exc}"

    if not _is_valid_hp_solution(result):
        return None, "spurious/invalid root"
    if result.states["2"]["T_C"] > config.T_discharge_max_C:
        return None, "discharge temperature exceeds limit"
    if result.kpis["pressure_ratio"] > config.PR_max:
        return None, "pressure ratio exceeds single-stage limit"
    return result, None


# ---------------------------------------------------------------------------
# Configuration 2: DC -> AHT -> DH (AHT alone)
# ---------------------------------------------------------------------------

def solve_aht_alone(config: ScenarioConfig) -> dict:
    T12_max = _aht_duehring_T12_max(config.T_waste_heat_in_C, config.T_waste_heat_in_C, config.T_reject_in_C, config)
    if T12_max is None or config.T_demand_supply_C > T12_max:
        ceiling = f"{T12_max:.1f} °C" if T12_max is not None else "not achievable at all (crystallization/pressure)"
        return {
            "feasible": False,
            "reason": (
                f"Duehring optimistic ceiling T12_max={ceiling} < required "
                f"T12={config.T_demand_supply_C:.1f} °C -- definitely infeasible, no solver attempt needed."
            ),
        }

    def build_fn(T12):
        return _build_aht_inputs(
            T11_C=config.T_DH_return_C, T12_C=T12,
            T_ext_hot_C=config.T_waste_heat_in_C, T_ext_cold_C=config.T_DC_out_C,
            T17_C=config.T_reject_in_C, T18_C=config.T_reject_out_C,
            Qabs_kW=config.Q_total_kW, config=config, fast=True,
        )

    # Single fixed point (no sweep, so no warm start to continue from) --
    # still uses the anchor search's fan-out, since a cold start can miss
    # even a genuinely reachable point (see module docstring on the AHT's
    # narrow basin of attraction).
    T_found, _x0, result = _locate_aht_point(build_fn, config.T_demand_supply_C)
    if T_found is None:
        return {"feasible": False, "reason": "no valid AHT solution anywhere near the required T12 (pinch/crystallization infeasible)"}
    if abs(T_found - config.T_demand_supply_C) > 1.0e-6:
        return {
            "feasible": False,
            "reason": (
                f"nearest reachable T12={T_found:.1f} °C, "
                f"{config.T_demand_supply_C - T_found:.1f} K short of the required {config.T_demand_supply_C:.1f} °C"
            ),
        }

    P_el_kW = result.pump_work_kW["W_AHT_total"]
    Q_DC_kW = result.heat_flows_kW["Q_des"] + result.heat_flows_kW["Q_evap"]
    return {
        "feasible": True, "P_el_kW": P_el_kW, "Q_DC_kW": Q_DC_kW,
        "specific_power_kWel_per_kWth": P_el_kW / config.Q_total_kW,
        "AHT_COP": result.kpis.get("COP", float("nan")),
    }


# ---------------------------------------------------------------------------
# Configuration 3: DC -> HP -> DH (HP alone)
# ---------------------------------------------------------------------------

def solve_hp_alone(config: ScenarioConfig) -> dict:
    model = HeatPumpModel(refrigerant=config.refrigerant)
    model.nw.iterinfo = False

    result, reason = _solve_hp(
        model,
        T_source_in_C=config.T_waste_heat_in_C, T_source_out_C=config.T_DC_out_C,
        T_sink_in_C=config.T_DH_return_C, T_sink_out_C=config.T_demand_supply_C,
        Q_cond_kW=config.Q_total_kW, config=config,
    )
    if result is None:
        return {"feasible": False, "reason": reason}

    P_el_kW = result.heat_flows_kW["P_compressor"] / config.eta_mech_motor
    Q_DC_kW = abs(result.heat_flows_kW["Q_evap"])
    return {
        "feasible": True, "P_el_kW": P_el_kW, "Q_DC_kW": Q_DC_kW,
        "specific_power_kWel_per_kWth": P_el_kW / config.Q_total_kW,
        "HP_COP": result.kpis["COP"],
    }


# ---------------------------------------------------------------------------
# Configuration 1: DC -> AHT -> HP -> DH
# ---------------------------------------------------------------------------
#
# The HP sits LAST -- fully pinned by the fixed DH boundary conditions --
# so it is solved FIRST for each T_mid. Its own evaporator duty (Q_evap)
# is exactly how much heat the AHT needs to deliver into the captive loop,
# and is passed forward as the AHT's Qabs_spec_kW (see module docstring).

def sweep_aht_then_hp(config: ScenarioConfig, T_mid_values_C):
    hp_model = HeatPumpModel(refrigerant=config.refrigerant)
    hp_model.nw.iterinfo = False

    # T13=T15=T_DC_in_C is constant throughout this sweep, so the Duehring
    # ceiling only needs computing once (see _aht_duehring_T12_max).
    T12_max_aht = _aht_duehring_T12_max(config.T_waste_heat_in_C, config.T_waste_heat_in_C, config.T_reject_in_C, config)

    records = []
    aht_state = {"T": None, "x0": None}
    for T_mid_C in T_mid_values_C:
        T11_aht = T_mid_C - config.dT_mid_C

        # Cheap pre-filter before either solver runs: if even the
        # optimistic Duehring ceiling can't reach T_mid, don't bother
        # solving the HP leg either.
        if T12_max_aht is None or T_mid_C > T12_max_aht:
            records.append({
                "T_mid_C": T_mid_C, "feasible": False,
                "reason": f"AHT: T_mid > Duehring ceiling T12_max={T12_max_aht}",
            })
            aht_state["T"], aht_state["x0"] = None, None
            continue

        hp_result, hp_reason = _solve_hp(
            hp_model,
            T_source_in_C=T_mid_C, T_source_out_C=T11_aht,
            T_sink_in_C=config.T_DH_return_C, T_sink_out_C=config.T_demand_supply_C,
            Q_cond_kW=config.Q_total_kW, config=config,
        )
        if hp_result is None:
            records.append({"T_mid_C": T_mid_C, "feasible": False, "reason": f"HP: {hp_reason}"})
            continue
        Q_evap_hp = hp_result.heat_flows_kW["Q_evap"]  # negative in TESPy's own sign convention
        P_HP_el_kW = hp_result.heat_flows_kW["P_compressor"] / config.eta_mech_motor

        GTL_est = T_mid_C - config.T_waste_heat_in_C
        if GTL_est < config.min_GTL_K:
            records.append({"T_mid_C": T_mid_C, "feasible": False, "reason": f"AHT: GTL={GTL_est:.1f} K < min_GTL_K"})
            aht_state["T"], aht_state["x0"] = None, None
            continue

        def build_fn(T, _Qabs_kW=abs(Q_evap_hp)):
            return _build_aht_inputs(
                T11_C=T - config.dT_mid_C, T12_C=T,
                T_ext_hot_C=config.T_waste_heat_in_C, T_ext_cold_C=config.T_DC_out_C,
                T17_C=config.T_reject_in_C, T18_C=config.T_reject_out_C,
                Qabs_kW=_Qabs_kW, config=config, fast=True,
            )

        aht_result, aht_reason = _advance_aht_leg(build_fn, T_mid_C, aht_state)
        if aht_result is None:
            records.append({"T_mid_C": T_mid_C, "feasible": False, "reason": aht_reason})
            continue
        P_AHT_kW = aht_result.pump_work_kW["W_AHT_total"]
        # DC feeds the AHT's desorber+evaporator directly in this config
        # (the HP's source is the captive loop, not DC) -- see module docstring.
        Q_DC_kW = aht_result.heat_flows_kW["Q_des"] + aht_result.heat_flows_kW["Q_evap"]

        P_total_kW = P_AHT_kW + P_HP_el_kW
        records.append({
            "T_mid_C": T_mid_C, "feasible": True,
            "Q_AHT_kW": abs(Q_evap_hp), "Q_HP_kW": config.Q_total_kW, "Q_DC_kW": Q_DC_kW,
            "P_AHT_kW": P_AHT_kW, "P_HP_el_kW": P_HP_el_kW, "P_total_kW": P_total_kW,
            "specific_power_kWel_per_kWth": P_total_kW / config.Q_total_kW,
            "AHT_COP": aht_result.kpis.get("COP", float("nan")),
            "HP_COP": hp_result.kpis["COP"],
        })

    return records


# ---------------------------------------------------------------------------
# Configuration 4: DC -> HP -> AHT -> DH
# ---------------------------------------------------------------------------
#
# The AHT sits LAST here -- fully pinned by the fixed DH boundary
# conditions -- so it is solved FIRST for each T_mid. Its own desorber +
# evaporator duty is exactly how much heat the HP needs to deliver into
# the captive loop, and is passed forward as the HP's Q_cond_kW.

def sweep_hp_then_aht(config: ScenarioConfig, T_mid_values_C):
    hp_model = HeatPumpModel(refrigerant=config.refrigerant)
    hp_model.nw.iterinfo = False

    records = []
    aht_state = {"T": None, "x0": None}
    for T_mid_C in T_mid_values_C:
        T14_aht = T_mid_C - config.dT_mid_C

        GTL_est = config.T_demand_supply_C - T_mid_C
        if GTL_est < config.min_GTL_K:
            records.append({"T_mid_C": T_mid_C, "feasible": False, "reason": f"AHT: GTL={GTL_est:.1f} K < min_GTL_K"})
            aht_state["T"], aht_state["x0"] = None, None
            continue

        # T13=T15=T_mid_C here (the AHT's "waste heat" is the captive
        # loop from the HP), so the Duehring ceiling depends on T_mid_C
        # and is cheap enough to recompute per point (closed-form, no solver).
        T12_max_aht = _aht_duehring_T12_max(T_mid_C, T_mid_C, config.T_reject_in_C, config)
        if T12_max_aht is None or config.T_demand_supply_C > T12_max_aht:
            records.append({
                "T_mid_C": T_mid_C, "feasible": False,
                "reason": f"AHT: required T12 > Duehring ceiling T12_max={T12_max_aht}",
            })
            aht_state["T"], aht_state["x0"] = None, None
            continue

        def build_fn(T):
            return _build_aht_inputs(
                T11_C=config.T_DH_return_C, T12_C=config.T_demand_supply_C,
                T_ext_hot_C=T, T_ext_cold_C=T - config.dT_mid_C,
                T17_C=config.T_reject_in_C, T18_C=config.T_reject_out_C,
                Qabs_kW=config.Q_total_kW, config=config, fast=True,
            )

        aht_result, aht_reason = _advance_aht_leg(build_fn, T_mid_C, aht_state)
        if aht_result is None:
            records.append({"T_mid_C": T_mid_C, "feasible": False, "reason": aht_reason})
            continue
        P_AHT_kW = aht_result.pump_work_kW["W_AHT_total"]
        Q_captive_kW = aht_result.heat_flows_kW["Q_des"] + aht_result.heat_flows_kW["Q_evap"]

        hp_result, hp_reason = _solve_hp(
            hp_model,
            T_source_in_C=config.T_waste_heat_in_C, T_source_out_C=config.T_DC_out_C,
            T_sink_in_C=T14_aht, T_sink_out_C=T_mid_C,
            Q_cond_kW=Q_captive_kW, config=config,
        )
        if hp_result is None:
            records.append({"T_mid_C": T_mid_C, "feasible": False, "reason": f"HP: {hp_reason}"})
            continue
        P_HP_el_kW = hp_result.heat_flows_kW["P_compressor"] / config.eta_mech_motor
        # DC feeds the HP's evaporator directly in this config (the AHT's
        # desorber/evaporator draw from the captive loop, not DC).
        Q_DC_kW = abs(hp_result.heat_flows_kW["Q_evap"])

        P_total_kW = P_AHT_kW + P_HP_el_kW
        records.append({
            "T_mid_C": T_mid_C, "feasible": True,
            "Q_AHT_kW": config.Q_total_kW, "Q_HP_kW": Q_captive_kW, "Q_DC_kW": Q_DC_kW,
            "P_AHT_kW": P_AHT_kW, "P_HP_el_kW": P_HP_el_kW, "P_total_kW": P_total_kW,
            "specific_power_kWel_per_kWth": P_total_kW / config.Q_total_kW,
            "AHT_COP": aht_result.kpis.get("COP", float("nan")),
            "HP_COP": hp_result.kpis["COP"],
        })

    return records


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    records_aht_hp, records_hp_aht, aht_alone: dict, hp_alone: dict,
):
    """Two panels sharing the T_mid axis: electrical power demand (the
    usual efficiency metric) and DC waste-heat draw (Q_DC_kW) -- shown
    side by side because a configuration that looks cheapest in
    electricity can require substantially more raw waste heat from the
    data center for the same delivered duty (AHT's COP < 1 means heat
    passed through it is only partially converted, not just "moved")."""
    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(13.5, 6.0))

    series = (
        (records_aht_hp, "tab:blue", "DC -> AHT -> HP -> DH"),
        (records_hp_aht, "tab:green", "DC -> HP -> AHT -> DH"),
    )
    for records, color, label in series:
        feasible = [r for r in records if r["feasible"]]
        if not feasible:
            continue
        T_mid = [r["T_mid_C"] for r in feasible]
        ax_p.plot(T_mid, [r["P_total_kW"] for r in feasible], "o-", color=color, label=label)
        ax_q.plot(T_mid, [r["Q_DC_kW"] for r in feasible], "o-", color=color, label=label)

    for key, label_fmt in (
        (aht_alone, lambda r: f"DC -> AHT -> DH (COP={r['AHT_COP']:.2f})"),
        (hp_alone, lambda r: f"DC -> HP -> DH (COP={r['HP_COP']:.2f})"),
    ):
        if not key["feasible"]:
            continue
        color = "tab:orange" if key is aht_alone else "tab:red"
        ax_p.axhline(key["P_el_kW"], color=color, linestyle="--", label=label_fmt(key))
        ax_q.axhline(key["Q_DC_kW"], color=color, linestyle="--", label=label_fmt(key))

    ax_p.set_xlabel("Handover temperature T_mid [°C]")
    ax_p.set_ylabel("Electrical power demand P_el [kW]")
    ax_p.legend(fontsize=9)
    ax_p.grid(alpha=0.3)

    ax_q.set_xlabel("Handover temperature T_mid [°C]")
    ax_q.set_ylabel("DC waste-heat draw Q_DC [kW]")
    ax_q.legend(fontsize=9)
    ax_q.grid(alpha=0.3)

    fig.tight_layout()

    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{plot_name}.svg")
    fig.savefig(f"{plot_name}.png", dpi=150)
    print(f"Saved plot to {plot_name}.svg / .png")


def _print_sweep_table(name: str, records) -> None:
    print(f"\n{name}:")
    n_feasible = sum(1 for r in records if r["feasible"])
    print(f"  Feasible T_mid points: {n_feasible} / {len(records)}")
    for r in records:
        if r["feasible"]:
            print(
                f"    T_mid={r['T_mid_C']:5.1f} C  P_total={r['P_total_kW']:6.2f} kW  "
                f"Q_DC={r['Q_DC_kW']:7.1f} kW  "
                f"spec.power={r['specific_power_kWel_per_kWth']:.4f}  "
                f"AHT_COP={r['AHT_COP']:.2f}  HP_COP={r['HP_COP']:.2f}"
            )
        else:
            print(f"    T_mid={r['T_mid_C']:5.1f} C  infeasible ({r['reason']})")
    if n_feasible:
        best = min((r for r in records if r["feasible"]), key=lambda r: r["P_total_kW"])
        print(f"  Best: T_mid={best['T_mid_C']:.1f} C, P_total={best['P_total_kW']:.2f} kW")


if __name__ == "__main__":
    config = ScenarioConfig()

    print("=" * 70)
    print(
        f"DC waste heat {config.T_waste_heat_in_C:.1f} -> {config.T_DC_out_C:.1f} C, "
        f"DH network {config.T_DH_return_C:.1f} -> {config.T_demand_supply_C:.1f} C, "
        f"Q_total={config.Q_total_kW:.0f} kW"
    )
    print("=" * 70)

    print("\nConfiguration 2 (DC -> AHT -> DH, AHT alone):")
    aht_alone = solve_aht_alone(config)
    if aht_alone["feasible"]:
        print(
            f"  FEASIBLE -- P_el={aht_alone['P_el_kW']:.2f} kW, Q_DC={aht_alone['Q_DC_kW']:.1f} kW, "
            f"spec.power={aht_alone['specific_power_kWel_per_kWth']:.4f}, "
            f"AHT COP={aht_alone['AHT_COP']:.2f}"
        )
    else:
        print(f"  INFEASIBLE ({aht_alone['reason']}) -- excluded from the comparison.")

    print("\nConfiguration 3 (DC -> HP -> DH, HP alone):")
    hp_alone = solve_hp_alone(config)
    if hp_alone["feasible"]:
        print(
            f"  FEASIBLE -- P_el={hp_alone['P_el_kW']:.2f} kW, Q_DC={hp_alone['Q_DC_kW']:.1f} kW, "
            f"spec.power={hp_alone['specific_power_kWel_per_kWth']:.4f}, "
            f"HP COP={hp_alone['HP_COP']:.2f}"
        )
    else:
        print(f"  INFEASIBLE ({hp_alone['reason']}) -- excluded from the comparison.")

    T_mid_grid = list(np.arange(
        config.T_waste_heat_in_C + T_MID_MARGIN_C,
        config.T_demand_supply_C - T_MID_MARGIN_C + 1.0e-9,
        T_MID_STEP_C,
    ))

    records_aht_hp = sweep_aht_then_hp(config, T_mid_grid)
    _print_sweep_table("Configuration 1 (DC -> AHT -> HP -> DH)", records_aht_hp)

    records_hp_aht = sweep_hp_then_aht(config, T_mid_grid)
    _print_sweep_table("Configuration 4 (DC -> HP -> AHT -> DH)", records_hp_aht)

    plot_comparison(records_aht_hp, records_hp_aht, aht_alone, hp_alone)
