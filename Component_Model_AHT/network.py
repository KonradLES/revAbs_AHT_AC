"""Assembles the AHT components into one simultaneous residual system.

Unknown vector (11, vs. 7 in Models/AHT_Pinch_Point.py):
    [T8, T10, x3, x6, x20, T2, T4, T3, T6, T20, T5]

T3, T6, T20, T5 used to be resolved by their own nested Brent root-find
inside every model evaluation. Here they are ordinary unknowns, and their
defining equations become 4 extra residuals -- the same total system,
solved simultaneously by one scipy.optimize.least_squares call instead of
1 (outer) + up to 4 (inner, per outer iteration) nested solves.

Not folded in: the throttle/flash (state 2 -> 1). See components.py's
module docstring for why (flash / no-flash regime switch).

Not yet supported: cycle_scale_spec_mode="Qdes_eva" (the reference
model's probe-based resolution). Only "m6" and "Qabs" are implemented
here; "Qdes_eva" would need the same recursive-probe treatment and was
left out of this first prototype to keep scope bounded.

This is a parallel, additional model -- it does not replace
Models/AHT_Pinch_Point.py. Use validate.py to compare the two on the same
inputs before trusting this one for anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

import Thermodynamic_Properties.libr_props as lp
from Models.AHT_Pinch_Point import AHTInputs, ModelEvaluationError, water_h_kjkg_PQ, water_h_kjkg_PT

from . import components as comp

UNKNOWN_NAMES = ["T8", "T10", "x3", "x6", "x20", "T2", "T4", "T3", "T6", "T20", "T5"]
TEMPERATURE_INDICES = (0, 1, 5, 6, 7, 8, 9, 10)  # all except x3, x6, x20

RESIDUAL_NAMES = [
    "R1_SHEX_energy",
    "R2_SHEX_pinch",
    "R3_preabs_energy",
    "R4_desorber_pinch",
    "R5_condenser_pinch",
    "R6_evaporator_pinch",
    "R7_absorber_pinch",
    "R8_T3_saturation",
    "R9_T6_saturation",
    "R10_T20_saturation",
    "R11_T5_enthalpy",
]

_RESIDUAL_SCALES = np.array([
    100.0,   # R1_SHEX_energy, kW
    1.0,     # R2_SHEX_pinch, K
    100.0,   # R3_preabs_energy, kW
    1.0,     # R4_desorber_pinch, K
    1.0,     # R5_condenser_pinch, K
    1.0,     # R6_evaporator_pinch, K
    1.0,     # R7_absorber_pinch, K
    1.0,     # R8_T3_saturation, K-equivalent (see components.saturated_solution_state)
    1.0,     # R9_T6_saturation, K-equivalent
    1.0,     # R10_T20_saturation, K-equivalent
    1.0,     # R11_T5_enthalpy, K-equivalent (see components.solution_pump)
])


def temperatures_C_to_K(z_user: np.ndarray) -> np.ndarray:
    z = np.asarray(z_user, dtype=float).copy()
    z[list(TEMPERATURE_INDICES)] += 273.15
    return z


def temperatures_K_to_C(z_internal: np.ndarray) -> np.ndarray:
    z = np.asarray(z_internal, dtype=float).copy()
    z[list(TEMPERATURE_INDICES)] -= 273.15
    return z


def _resolve_cycle_scale(inputs: AHTInputs, *, w3: float, w6: float, h3: float, h4: float, h10: float, strict: bool) -> float:
    """m6 resolution for "m6" and "Qabs" modes only -- see module docstring."""
    if inputs.cycle_scale_spec_mode == "m6":
        m6 = float(inputs.m6_spec)
        if strict and m6 <= 0.0:
            raise ModelEvaluationError("Pumped solution mass flow m6 must be positive.")
        return m6

    if inputs.cycle_scale_spec_mode != "Qabs":
        raise ValueError(
            f"cycle_scale_spec_mode={inputs.cycle_scale_spec_mode!r} is not supported by this "
            "prototype yet -- only 'm6' and 'Qabs' are implemented."
        )

    w3_safe = w3 if strict else max(w3, 1.0e-9)
    ratio = w6 / w3_safe
    denominator = (ratio - 1.0) * h10 + h4 - ratio * h3

    if strict:
        if abs(denominator) <= 1.0e-12:
            raise ModelEvaluationError("Cannot resolve cycle scaling from Q_abs: denominator near zero.")
        m6 = float(inputs.Qabs_spec_kW) / denominator
        if m6 <= 0.0:
            raise ModelEvaluationError(f"Computed m6 not positive: m6={m6:.6f} kg/s.")
        return m6

    denom_safe = denominator if abs(denominator) > 1.0e-12 else (1.0e-12 if denominator >= 0.0 else -1.0e-12)
    return float(inputs.Qabs_spec_kW) / denom_safe


@dataclass(frozen=True)
class NetworkEvaluation:
    unknowns: Dict[str, float]
    residuals: Dict[str, float]
    heat_flows_kW: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    m6: float


def evaluate(z: np.ndarray, inputs: AHTInputs, *, strict: bool) -> NetworkEvaluation:
    T8, T10, x3, x6, x20, T2, T4, T3, T6, T20, T5 = map(float, z)

    p_low, p_high = comp.pressure_levels(T8, T10)
    if p_high <= p_low:
        raise ModelEvaluationError(f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa).")

    w3, h3, r_T3 = comp.saturated_solution_state(T3, x3, p_high)
    w6, h6, r_T6 = comp.saturated_solution_state(T6, x6, p_low)
    w20, h20, r_T20 = comp.saturated_solution_state(T20, x20, p_high)

    if strict and not (w6 > w3 > 0.0):
        raise ModelEvaluationError(f"Concentration hierarchy violated: w6={w6:.6f}, w3={w3:.6f}.")
    if strict and not (w6 > w20 > w3):
        raise ModelEvaluationError(f"Implausible pre-absorption state: w6={w6:.6f}, w20={w20:.6f}, w3={w3:.6f}.")

    h4 = lp.h_solution_mass_kjkg(T4, x6)
    h10 = water_h_kjkg_PQ(p_high, Q=1.0)

    m6 = _resolve_cycle_scale(inputs, w3=w3, w6=w6, h3=h3, h4=h4, h10=h10, strict=strict)
    m4, m3, m2, m1, m7, m20, m19, m21 = comp.cycle_mass_flows(m6=m6, w3=w3, w6=w6, w20=w20, strict=strict)
    m8 = m9 = m10 = m7

    h2 = lp.h_solution_mass_kjkg(T2, x3)
    h5, W_sol_pump, r_T5 = comp.solution_pump(T6=T6, x6=x6, h6=h6, p_low=p_low, p_high=p_high, m6=m6, T5=T5)

    Q_shex, pinch_shex, lmtd_shex, r_shex_energy = comp.shex(
        T2=T2, T3=T3, T4=T4, T5=T5, x3=x3, h2=h2, h3=h3, h4=h4, h5=h5,
        m3=m3, m4=m4, dT_min_shex=inputs.dT_min_shex, strict=strict,
    )
    r_shex_pinch = pinch_shex - inputs.dT_min_shex

    h1 = h2
    flash = comp.throttle_flash(p_low=p_low, h2=h2, m2=m2, x3=x3, strict=strict)
    T1 = flash["T1_K"]

    T7 = T1 + inputs.desorber_vapor_superheat_K
    h7 = water_h_kjkg_PT(p_low, T7)
    Q_des = comp.desorber_duty(m6=m6, h6=h6, m7=m7, h7=h7, m1=m1, h1=h1, strict=strict)

    h8 = water_h_kjkg_PQ(p_low, Q=0.0)
    Q_cond, T18, m17, pinch_cond, lmtd_cond = comp.condenser(
        inputs=inputs, T7=T7, T8=T8, h7=h7, h8=h8, m7=m7, p_low=p_low, strict=strict
    )
    r_cond_pinch = pinch_cond - inputs.dT_min_cond

    h9, T9, W_ref_pump = comp.refrigerant_pump(p_low=p_low, p_high=p_high, h8=h8, m8=m8)
    h_f_high = water_h_kjkg_PQ(p_high, Q=0.0)

    # External desorber/evaporator routing (parallel only in this prototype;
    # the serial routing modes from the reference model are not yet ported).
    if inputs.desorber_evaporator_routing_mode != "parallel":
        raise ValueError(
            "This prototype only supports desorber_evaporator_routing_mode='parallel' so far."
        )
    T13_in = comp.resolve_desorber_external_inlet_temperature(inputs)
    T15_in = comp.resolve_evaporator_external_inlet_temperature(inputs)

    T14, m13, pinch_des, lmtd_des = comp.desorber(
        inputs=inputs, T13_in=T13_in, T1=T1, T6=T6, Q_des=Q_des, strict=strict
    )
    r_des_pinch = pinch_des - inputs.dT_min_des

    Q_evap, T16, m15, pinch_evap, lmtd_evap = comp.evaporator(
        inputs=inputs, T15_in=T15_in, T10=T10, T9=T9, h9=h9, h10=h10, h_f_high=h_f_high, m9=m9, strict=strict
    )
    r_evap_pinch = pinch_evap - inputs.dT_min_evap

    r_preabs_energy = comp.preabsorption_mixer(m4=m4, h4=h4, m19=m19, h10=h10, m20=m20, h20=h20)

    Q_abs, T12, m11, pinch_abs, lmtd_abs = comp.absorber(
        inputs=inputs, T20=T20, T3=T3, m10=m10, h10=h10, m4=m4, h4=h4, m3=m3, h3=h3, strict=strict
    )
    r_abs_pinch = pinch_abs - inputs.dT_min_abs

    residuals = dict(zip(
        RESIDUAL_NAMES,
        [
            r_shex_energy, r_shex_pinch, r_preabs_energy, r_des_pinch,
            r_cond_pinch, r_evap_pinch, r_abs_pinch, r_T3, r_T6, r_T20, r_T5,
        ],
    ))

    unknowns = dict(zip(UNKNOWN_NAMES, [T8, T10, x3, x6, x20, T2, T4, T3, T6, T20, T5]))

    heat_flows_kW = {
        "Q_shex": Q_shex, "Q_des": Q_des, "Q_cond": Q_cond, "Q_evap": Q_evap, "Q_abs": Q_abs,
    }

    states = {
        "1": {"T_K": T1, "p_Pa": p_low, "m_kg_s": m1, "h_kJ_kg": h1, "x_LiBr_mol": x3},
        "2": {"T_K": T2, "p_Pa": p_high, "m_kg_s": m2, "h_kJ_kg": h2, "x_LiBr_mol": x3},
        "3": {"T_K": T3, "p_Pa": p_high, "m_kg_s": m3, "h_kJ_kg": h3, "x_LiBr_mol": x3},
        "4": {"T_K": T4, "p_Pa": p_high, "m_kg_s": m4, "h_kJ_kg": h4, "x_LiBr_mol": x6},
        "5": {"T_K": T5, "p_Pa": p_high, "m_kg_s": m6, "h_kJ_kg": h5, "x_LiBr_mol": x6},
        "6": {"T_K": T6, "p_Pa": p_low, "m_kg_s": m6, "h_kJ_kg": h6, "x_LiBr_mol": x6},
        "7": {"T_K": T7, "p_Pa": p_low, "m_kg_s": m7, "h_kJ_kg": h7},
        "8": {"T_K": T8, "p_Pa": p_low, "m_kg_s": m8, "h_kJ_kg": h8},
        "9": {"T_K": T9, "p_Pa": p_high, "m_kg_s": m9, "h_kJ_kg": h9},
        "10": {"T_K": T10, "p_Pa": p_high, "m_kg_s": m10, "h_kJ_kg": h10},
        "20": {"T_K": T20, "p_Pa": p_high, "m_kg_s": m20, "h_kJ_kg": h20, "x_LiBr_mol": x20},
        "12": {"T_K": T12, "m_kg_s": m11},
        "14": {"T_K": T14, "m_kg_s": m13},
        "16": {"T_K": T16, "m_kg_s": m15},
        "18": {"T_K": T18, "m_kg_s": m17},
    }

    return NetworkEvaluation(unknowns=unknowns, residuals=residuals, heat_flows_kW=heat_flows_kW, states=states, m6=m6)


def _residual_array(ev: NetworkEvaluation) -> np.ndarray:
    raw = np.array([ev.residuals[name] for name in RESIDUAL_NAMES], dtype=float)
    return raw / _RESIDUAL_SCALES


def residual_vector(z: np.ndarray, inputs: AHTInputs) -> np.ndarray:
    try:
        ev = evaluate(z, inputs, strict=True)
        return _residual_array(ev)
    except Exception:
        try:
            ev = evaluate(z, inputs, strict=False)
            return _residual_array(ev)
        except Exception:
            return np.full(len(RESIDUAL_NAMES), inputs.penalty_level, dtype=float)


def _bootstrap_aux(z7: np.ndarray, inputs: AHTInputs) -> np.ndarray:
    """Computes T3, T6, T20, T5 for a given (T8,T10,x3,x6,x20,T2,T4) via the
    reference model's own nested solves.

    Used twice: once to build the very first initial guess, and a few more
    times during solve()'s short warm-start phase (see its docstring) --
    NOT on every residual evaluation, which is the whole point of this
    prototype. A handful of these calls per solve() is a small, bounded
    cost; one per residual evaluation (the reference model's approach) is
    not.
    """
    T8, T10, x3, x6, x20, T2, T4 = map(float, z7)
    from Models.AHT_Pinch_Point import water_p_sat_from_T as _p_sat

    p_low = _p_sat(T8, Q=0.0)
    p_high = _p_sat(T10, Q=1.0)

    T3 = lp.T_sat_solution_from_p_x(p_high, x3)
    T6 = lp.T_sat_solution_from_p_x(p_low, x6)
    T20 = lp.T_sat_solution_from_p_x(p_high, x20)

    h6 = lp.h_solution_mass_kjkg(T6, x6)
    rho6 = lp.rho_solution_mass(T6, x6)
    m6_guess = float(inputs.m6_spec) if inputs.cycle_scale_spec_mode == "m6" else 1.0
    W_sol_pump = m6_guess / rho6 * (p_high - p_low) / 1000.0
    h5 = h6 + W_sol_pump / m6_guess
    try:
        T5 = lp.T_from_h_x_mass(h5, x6)
    except Exception:
        T5 = T6

    return np.array([T8, T10, x3, x6, x20, T2, T4, T3, T6, T20, T5], dtype=float)


def initial_guess(inputs: AHTInputs) -> np.ndarray:
    """Same reference guess as Models/AHT_Pinch_Point.initial_guess, with
    T3, T6, T20, T5 bootstrapped once via _bootstrap_aux."""
    T_evap_ref = inputs.evaporator_temperature_reference
    z7 = np.array([
        inputs.T_17 + 8.0,
        T_evap_ref - 8.0,
        0.23, 0.30, 0.27,
        T_evap_ref,
        inputs.T_11 + 8.0,
    ])
    return _bootstrap_aux(z7, inputs)


def bounds(inputs: AHTInputs, x0: np.ndarray | None = None, margin: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds for the 11 unknowns.

    T8, T10, x3, x6, x20, T2, T4 use the exact same bounds as
    Models/AHT_Pinch_Point.bounds(). T3, T6, T20, T5 -- the 4 unknowns
    that used to be implicitly, tightly constrained by being *computed*
    from (p_low/p_high, x) rather than searched -- get a +-60 K window
    around the initial guess instead of a wide, generic bound.

    This matters because the underlying 7-equation system is highly
    nonlinear and not provably unique: validate.py found a second,
    independently physically-valid root (passes the reference model's own
    strict checks) reachable from a wide-bounded start. Tying T3/T6/T20/T5
    to a window around a good initial guess keeps the solver in the same
    basin the reference model converges to, for a like-for-like
    comparison -- it does not, by itself, prove global uniqueness.
    """
    T_evap_ref = inputs.evaporator_temperature_reference
    T_des_ref = inputs.desorber_temperature_reference

    if x0 is None:
        x0 = initial_guess(inputs)
    T3_guess, T6_guess, T20_guess, T5_guess = x0[7], x0[8], x0[9], x0[10]

    lower = np.array([
        inputs.T_17 + 0.5, inputs.T_17 + 5.0, 0.05, 0.08, 0.05, inputs.T_17 + 1.0, inputs.T_17 + 1.0,
        T3_guess - margin, T6_guess - margin, T20_guess - margin, T5_guess - margin,
    ])
    upper = np.array([
        min(T_des_ref - 1.0, 420.0), min(T_evap_ref - 0.5, 500.0), 0.34, 0.39, 0.39, 500.0, 500.0,
        T3_guess + margin, T6_guess + margin, T20_guess + margin, T5_guess + margin,
    ])
    return lower, upper


def _solve_once(inputs: AHTInputs, z_start: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    z_start = np.clip(z_start, lower, upper)
    return least_squares(
        lambda z: residual_vector(z, inputs),
        z_start,
        bounds=(lower, upper),
        method="trf",
        xtol=inputs.solver_tol,
        ftol=inputs.solver_tol,
        gtol=inputs.solver_tol,
        max_nfev=inputs.max_nfev,
    )


def solve(inputs: AHTInputs, x0: np.ndarray | None = None, warm_start_iters: int = 3):
    """Solves the 11-unknown simultaneous system.

    Short warm-start phase first (see module docstring's condition-number
    finding in validate.py / the conversation that led here): T3, T6, T20,
    T5 are re-bootstrapped via the reference model's nested solves a
    handful of times (warm_start_iters, default 3) and each time used to
    re-center a TIGHT (+-8 K) bound window for a quick simultaneous solve
    of just the 7 core unknowns' neighborhood. This keeps the search in
    the same basin the reference model converges to -- starting the full,
    loosely-bounded 11-var system directly from the generic initial guess
    was found to converge to a different (also mathematically valid, but
    wrong) nearby root, because the free T3/T6/T20/T5 parametrization has
    a soft (high-condition-number, not singular) direction the reduced
    7-var reference formulation doesn't have.

    A handful of bootstrap calls here (warm_start_iters + 1) is still far
    fewer than the reference model's one bootstrap per residual
    evaluation, so most of the original performance motivation survives;
    it is no longer a pure "zero nested solves" simultaneous system,
    though.
    """
    if x0 is None:
        x0 = initial_guess(inputs)
    z7 = x0[:7].copy()

    result = None
    for _ in range(warm_start_iters):
        z_guess = _bootstrap_aux(z7, inputs)
        lower, upper = bounds(inputs, x0=z_guess, margin=8.0)
        result = _solve_once(inputs, z_guess, lower, upper)
        z7 = result.x[:7]

    z_guess = _bootstrap_aux(z7, inputs)
    lower, upper = bounds(inputs, x0=z_guess, margin=60.0)
    result = _solve_once(inputs, z_guess, lower, upper)

    ev = evaluate(result.x, inputs, strict=True)
    return result, ev
