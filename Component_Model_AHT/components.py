"""Individual AHT components, each exposing its own governing equations.

Every function here mirrors one physical piece of the cycle and returns
the quantities the rest of the network needs plus any RESIDUALS it
contributes to the simultaneous system (empty tuple if the component is
purely algebraic, i.e. fully determined by its inputs with no equation
left over for the solver).

Design choice vs. Models/AHT_Pinch_Point.py's _evaluate_model_common:
Four states that used to be resolved by their own nested Brent root-find
on every model evaluation -- T3, T6, T20 (saturation temperature from
pressure + concentration) and T5 (temperature from enthalpy + LiBr mole
fraction) -- are primary unknowns here instead, with their defining
equation exposed as a residual (calc_p_sat_patek(...)-p = 0, or
h_solution_mass_kjkg(...)-h = 0). The outer Newton/least_squares solve
handles them simultaneously with everything else, the same way TESPy
folds component equations into one system instead of nesting solves.

The one exception is the throttle/flash (state 2 -> 1): it has a genuine
flash / no-flash regime switch (see _flash_valve_state in libr_props.py),
which is awkward to fold into a single smooth residual without a proper
complementarity formulation. It keeps its own internal, already-robust
solve (flash_valve_state_2_to_1, unchanged) and is treated as a
self-contained component instead.

All numerical primitives (water/LiBr property functions, smooth_min,
counterflow LMTD, external-stream resolution) are imported from the
existing, validated Models/AHT_Pinch_Point.py and
Thermodynamic_Properties/libr_props.py rather than reimplemented, to
keep this module's own risk surface limited to the *structural* change
(component decomposition + folded-in residuals), not the underlying
thermodynamics.
"""

from __future__ import annotations

from typing import Tuple

import Thermodynamic_Properties.libr_props as lp
from Models.AHT_Pinch_Point import (
    AHTInputs,
    ModelEvaluationError,
    _resolve_absorber_external_stream,
    _resolve_condenser_external_stream,
    _resolve_desorber_external_inlet_temperature,
    _resolve_desorber_external_stream,
    _resolve_evaporator_external_inlet_temperature,
    _resolve_evaporator_external_stream,
    counterflow_lmtd,
    counterflow_lmtd_soft,
    smooth_min,
    water_h_kjkg_PQ,
    water_h_kjkg_PT,
    water_p_sat_from_T,
    water_rho_kgm3_PQ,
    water_T_K_PH,
)


def _lmtd(strict: bool, *, hot_in: float, hot_out: float, cold_in: float, cold_out: float, name: str) -> float:
    if strict:
        return counterflow_lmtd(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out, HX_name=name)
    return counterflow_lmtd_soft(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out)


# ---------------------------------------------------------------------------
# Refrigerant pressure levels
# ---------------------------------------------------------------------------

def pressure_levels(T8: float, T10: float) -> Tuple[float, float]:
    """Low/high refrigerant pressure from the condenser/generator saturation temperatures."""
    p_low = water_p_sat_from_T(T8, Q=0.0)
    p_high = water_p_sat_from_T(T10, Q=1.0)
    return p_low, p_high


# ---------------------------------------------------------------------------
# Saturated solution states (formerly nested Brent solves, now residuals)
# ---------------------------------------------------------------------------

_DT_FOR_SLOPE = 0.01  # K, finite-difference step for the local dp/dT below


def saturated_solution_state(T: float, x_libr_mol: float, p_Pa: float) -> Tuple[float, float, float]:
    """A LiBr solution state on its saturation line at a given pressure.

    Used for states 3, 6, and 20 (desorber outlet, absorber outlet,
    pre-absorption outlet) -- all saturated liquid at p_high or p_low.

    Returns
    -------
    (w_LiBr, h_kJ_kg, residual)
        residual = (calc_p_sat_patek(T, x) - p_Pa) / (dp_sat/dT), i.e. the
        pressure mismatch converted to Kelvin via the local saturation
        slope, not a raw Pascal residual. This matters: dp_sat/dx at a
        typical operating point is ~1e6 Pa per unit x, vs. dp_sat/dT of
        only ~1e3 Pa/K -- a flat Pa-scale residual is roughly 1000x more
        sensitive to x than to T, which starves the other (K- and
        kW-scaled) residuals of influence over x during solving and was
        found (via validate.py) to let the solver settle on a nearby but
        wrong root. Rescaling by the local slope puts this residual back
        on equal footing (Kelvin, same as the pinch residuals) without
        reintroducing a root-find: dp/dT is a single extra direct
        function evaluation, not an iterative solve.
    """
    w = lp.w_libr_from_x(x_libr_mol)
    h = lp.h_solution_mass_kjkg(T, x_libr_mol)
    p_calc = lp.calc_p_sat_patek(T, x_libr_mol)
    dpdT = (lp.calc_p_sat_patek(T + _DT_FOR_SLOPE, x_libr_mol) - p_calc) / _DT_FOR_SLOPE
    residual_K = (p_calc - p_Pa) / dpdT
    return w, h, residual_K


# ---------------------------------------------------------------------------
# Cycle mass flows (algebraic, given m6 and the three concentrations)
# ---------------------------------------------------------------------------

def cycle_mass_flows(
    *, m6: float, w3: float, w6: float, w20: float, strict: bool
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Mass-flow network from the pumped solution flow m6 and concentrations.

    Returns
    -------
    (m4, m3, m2, m1, m7, m20, m19, m21)
    m5 == m4 == m6; m8 == m9 == m10 == m7 (same refrigerant-side flow).
    """
    m5 = m4 = m6
    w3_safe = w3 if strict else max(w3, 1.0e-9)
    m3 = m4 * w6 / w3_safe
    m2 = m1 = m3
    m7 = m3 - m6

    if strict and m7 <= 0.0:
        raise ModelEvaluationError(f"Refrigerant mass flow not positive: m7={m7:.6f} kg/s.")

    w20_safe = w20 if strict else max(w20, 1.0e-12)
    m20 = m4 * w6 / w20_safe
    m19 = m20 - m4
    m10_safe = m7 if strict else (m7 if abs(m7) > 1.0e-12 else 1.0e-12)
    beta = m19 / m10_safe
    m21 = m7 - m19

    if strict and not (0.0 <= beta <= 1.0):
        raise ModelEvaluationError(f"Pre-absorption fraction beta outside [0,1]: beta={beta:.6f}.")
    if strict and m21 < 0.0:
        raise ModelEvaluationError(f"m21 negative: {m21:.6f} kg/s.")

    return m4, m3, m2, m1, m7, m20, m19, m21


# ---------------------------------------------------------------------------
# Solution pump 6 -> 5
# ---------------------------------------------------------------------------

def solution_pump(
    *, T6: float, x6: float, h6: float, p_low: float, p_high: float, m6: float, T5: float
) -> Tuple[float, float, float]:
    """Pumps the strong solution from p_low to p_high.

    Returns
    -------
    (h5, W_sol_pump_kW, residual)
        residual = (h_solution_mass_kjkg(T5, x6) - h5) / cp(T5, x6), i.e.
        the enthalpy mismatch converted to Kelvin via the local specific
        heat, so it sits on the same K scale as the other residuals
        (see saturated_solution_state's docstring for why this matters).
    """
    rho6 = lp.rho_solution_mass(T6, x6)
    v6 = 1.0 / rho6
    W_sol_pump = m6 * v6 * (p_high - p_low) / 1000.0
    h5 = h6 + W_sol_pump / m6
    cp5 = lp.cp_solution_mass_kjkgk(T5, x6)
    residual_K = (lp.h_solution_mass_kjkg(T5, x6) - h5) / cp5
    return h5, W_sol_pump, residual_K


# ---------------------------------------------------------------------------
# Solution heat exchanger (SHEX): 3 -> 2 and 5 -> 4
# ---------------------------------------------------------------------------

def shex(
    *, T2: float, T3: float, T4: float, T5: float, x3: float, h2: float, h3: float, h4: float, h5: float,
    m3: float, m4: float, dT_min_shex: float, strict: bool,
) -> Tuple[float, float, float, float]:
    """Counterflow solution/solution heat exchanger.

    Returns
    -------
    (Q_shex, pinch_shex, lmtd_shex, residual_energy)
    """
    Q_shex_hot = m3 * (h3 - h2)
    Q_shex_cold = m4 * (h4 - h5)
    if strict and Q_shex_hot <= 0.0:
        raise ModelEvaluationError(f"Q_shex_hot not positive: {Q_shex_hot:.6f} kW.")
    if strict and Q_shex_cold <= 0.0:
        raise ModelEvaluationError(f"Q_shex_cold not positive: {Q_shex_cold:.6f} kW.")

    pinch_shex = smooth_min(T3 - T4, T2 - T5, k=50.0)
    lmtd_shex = _lmtd(strict, hot_in=T3, hot_out=T2, cold_in=T5, cold_out=T4, name="SHEX")
    residual_energy = Q_shex_hot - Q_shex_cold
    return Q_shex_hot, pinch_shex, lmtd_shex, residual_energy


# ---------------------------------------------------------------------------
# Throttle / isenthalpic flash 2 -> 1 (kept as its own internal solve)
# ---------------------------------------------------------------------------

def throttle_flash(*, p_low: float, h2: float, m2: float, x3: float, strict: bool) -> dict:
    """Isenthalpic flash throttle. Not folded into the simultaneous system

    (see module docstring) -- it still resolves its own T1/x1/flash
    fraction internally via flash_valve_state_2_to_1, exactly as in the
    reference model.
    """
    if strict:
        return lp.flash_valve_state_2_to_1(p_out_pa=p_low, h2_kJkg=h2, m2_kg_s=m2, x2_libr_mol=x3)
    try:
        return lp.flash_valve_state_2_to_1(p_out_pa=p_low, h2_kJkg=h2, m2_kg_s=m2, x2_libr_mol=x3)
    except Exception:
        from Models.AHT_Pinch_Point import water_T_sat_from_p

        T1 = water_T_sat_from_p(p_low, Q=0.0)
        w3 = lp.w_libr_from_x(x3)
        return {
            "T1_K": T1, "x1_LiBr_mol": x3, "w1_LiBr": w3,
            "m1_sol_kg_s": m2, "m1_flash_kg_s": 0.0,
            "h1_sol_kJ_kg": h2, "h1_flash_kJ_kg": float("nan"), "h1_mix_kJ_kg": h2,
            "flash_fraction": 0.0,
        }


# ---------------------------------------------------------------------------
# Desorber: external hot stream 13 -> 14, internal cold side 1 -> 6
# ---------------------------------------------------------------------------

def desorber(
    *, inputs: AHTInputs, T13_in: float, T1: float, T6: float, Q_des: float, strict: bool
) -> Tuple[float, float, float, float]:
    """External-stream resolution + pinch for the desorber.

    Returns
    -------
    (T14, m13, pinch_des, lmtd_des)
    """
    m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=strict, T13=T13_in)
    lmtd_des = _lmtd(strict, hot_in=T13_in, hot_out=T14, cold_in=T1, cold_out=T6, name="Desorber")
    pinch_des = smooth_min(T13_in - T6, T14 - T1, k=50.0)
    return T14, m13, pinch_des, lmtd_des


def desorber_duty(*, m6: float, h6: float, m7: float, h7: float, m1: float, h1: float, strict: bool) -> float:
    """Desorber heat duty from the internal (solution + refrigerant) side energy balance."""
    Q_des = m6 * h6 + m7 * h7 - m1 * h1
    if strict and Q_des <= 0.0:
        raise ModelEvaluationError(f"Desorber heat flow not positive: Q_des={Q_des:.6f} kW.")
    return Q_des


# ---------------------------------------------------------------------------
# Condenser: 7 -> 8, external stream 17 -> 18
# ---------------------------------------------------------------------------

def condenser(
    *, inputs: AHTInputs, T7: float, T8: float, h7: float, h8: float, m7: float, p_low: float, strict: bool
) -> Tuple[float, float, float, float, float]:
    """Returns (Q_cond, T18, m17, pinch_cond, lmtd_cond)."""
    Q_cond = m7 * (h7 - h8)
    if strict and Q_cond <= 0.0:
        raise ModelEvaluationError(f"Condenser heat flow not positive: Q_cond={Q_cond:.6f} kW.")

    m17, T18 = _resolve_condenser_external_stream(inputs, Q_cond, strict=strict)

    h_g_low = water_h_kjkg_PQ(p_low, Q=1.0)
    Q_desuperheat = max(0.0, min(m7 * (h7 - h_g_low), Q_cond))
    T18_sat = T18 - Q_desuperheat / (m17 * inputs.cp_w_kJkgK)

    pinch_cond = smooth_min(
        smooth_min(T7 - T18, T8 - T18_sat, k=50.0),
        T8 - inputs.T_17,
        k=50.0,
    )
    lmtd_cond = _lmtd(strict, hot_in=T8, hot_out=T8, cold_in=inputs.T_17, cold_out=T18, name="Condenser")
    return Q_cond, T18, m17, pinch_cond, lmtd_cond


# ---------------------------------------------------------------------------
# Refrigerant pump 8 -> 9
# ---------------------------------------------------------------------------

def refrigerant_pump(*, p_low: float, p_high: float, h8: float, m8: float) -> Tuple[float, float, float]:
    """Returns (h9, T9, W_ref_pump_kW)."""
    rho8 = water_rho_kgm3_PQ(p_low, Q=0.0)
    v8 = 1.0 / rho8
    W_ref_pump = m8 * v8 * (p_high - p_low) / 1000.0
    h9 = h8 + (W_ref_pump / m8 if abs(m8) > 1.0e-12 else 0.0)
    T9 = water_T_K_PH(p_high, h9)
    return h9, T9, W_ref_pump


# ---------------------------------------------------------------------------
# Evaporator: 9 -> 10, external stream 15 -> 16
# ---------------------------------------------------------------------------

def evaporator(
    *, inputs: AHTInputs, T15_in: float, T10: float, T9: float, h9: float, h10: float, h_f_high: float,
    m9: float, strict: bool,
) -> Tuple[float, float, float, float, float]:
    """Returns (Q_evap, T16, m15, pinch_evap, lmtd_evap)."""
    Q_evap = m9 * (h10 - h9)
    if strict and Q_evap <= 0.0:
        raise ModelEvaluationError(f"Evaporator heat flow not positive: Q_evap={Q_evap:.6f} kW.")

    m15, T16 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=strict, T15=T15_in)

    Q_subcool = max(0.0, min(m9 * (h_f_high - h9), Q_evap))
    T16_sat = T16 + Q_subcool / (m15 * inputs.cp_w_kJkgK)

    pinch_evap = smooth_min(
        smooth_min(T15_in - T10, T16_sat - T10, k=50.0),
        T16 - T9,
        k=50.0,
    )
    lmtd_evap = _lmtd(strict, hot_in=T15_in, hot_out=T16, cold_in=T10, cold_out=T10, name="Evaporator")
    return Q_evap, T16, m15, pinch_evap, lmtd_evap


# ---------------------------------------------------------------------------
# Adiabatic pre-absorption mixer: 4 + 19 -> 20
# ---------------------------------------------------------------------------

def preabsorption_mixer(*, m4: float, h4: float, m19: float, h10: float, m20: float, h20: float) -> float:
    """LiBr balance is exact by construction (see cycle_mass_flows); this
    returns the one residual left over: the adiabatic-mixing energy balance."""
    return m4 * h4 + m19 * h10 - m20 * h20


# ---------------------------------------------------------------------------
# Absorber: global energy balance, local LMTD referenced to state 20
# ---------------------------------------------------------------------------

def absorber(
    *, inputs: AHTInputs, T20: float, T3: float, m10: float, h10: float, m4: float, h4: float,
    m3: float, h3: float, strict: bool,
) -> Tuple[float, float, float, float, float]:
    """Returns (Q_abs, T12, m11, pinch_abs, lmtd_abs)."""
    Q_abs = m10 * h10 + m4 * h4 - m3 * h3
    if strict and Q_abs <= 0.0:
        raise ModelEvaluationError(f"Absorber heat flow not positive: Q_abs={Q_abs:.6f} kW.")

    m11, T12 = _resolve_absorber_external_stream(inputs, Q_abs, strict=strict)

    pinch_abs = smooth_min(T20 - T12, T3 - inputs.T_11, k=50.0)
    lmtd_abs = _lmtd(strict, hot_in=T20, hot_out=T3, cold_in=inputs.T_11, cold_out=T12, name="Absorber")
    return Q_abs, T12, m11, pinch_abs, lmtd_abs


__all__ = [
    "pressure_levels",
    "saturated_solution_state",
    "cycle_mass_flows",
    "solution_pump",
    "shex",
    "throttle_flash",
    "desorber",
    "desorber_duty",
    "condenser",
    "refrigerant_pump",
    "evaporator",
    "preabsorption_mixer",
    "absorber",
    "resolve_desorber_external_inlet_temperature",
    "resolve_evaporator_external_inlet_temperature",
]

resolve_desorber_external_inlet_temperature = _resolve_desorber_external_inlet_temperature
resolve_evaporator_external_inlet_temperature = _resolve_evaporator_external_inlet_temperature
