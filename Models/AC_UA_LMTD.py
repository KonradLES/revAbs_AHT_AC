"""AC simulation with 6 primary unknowns (UA/LMTD formulation).

Unlike the pinch-point variant, the four main heat exchangers (desorber,
condenser, evaporator, absorber) are specified by fixed UA values, and the
residuals directly enforce Q - UA*LMTD = 0. The external mass flows m_11,
m_13, m_15 (desorber, absorber, condenser) are fixed inputs.

SHEX modeling is variable:
- shex_model = "UA": UA_shex is given, residual is Q - UA_shex*LMTD_shex
- shex_model = "NTU": Effectiveness_shex is given, residual is
  Q - Effectiveness_shex * Q_shex_max

External evaporator stream specification is variable:
- evaporator_spec_mode = "m17": m17_spec is given, T18 is computed
- evaporator_spec_mode = "T18": T18_spec_C is given, m17 is computed

Cycle scaling is variable:
- cycle_scale_spec_mode = "m1": m1_spec is given
- cycle_scale_spec_mode = "Qeva": Qevap_spec_kW is given, m1 is computed

Model assumptions
------------------
- working fluid pair: H2O/LiBr
- steady-state operation
- no pressure losses in components or piping
- isenthalpic solution throttle with local flash calculation
- adiabatic pre-absorption ahead of the absorber is modeled explicitly
- external fluids are described with constant cp_w

Primary solver variables
-------------------------
z = [T8, T10, x4, x1, T3, T5]

Internal units
--------------
- temperature: K
- pressure: Pa
- mass flow: kg/s
- specific enthalpy: kJ/kg
- heat flow / power: kW (= kJ/s)
- UA: kW/K

Strategy for unphysical intermediate states
--------------------------------------------
During iteration, the solver (trf) inevitably visits points where
temperature differences in heat exchangers become negative. The strict
final evaluation raises a ModelEvaluationError there, which is correct
for judging the physical validity of the final point.

For the solver path, this module evaluates the same model core in a
robust variant:
  - identical equation structure as the strict final evaluation
  - counterflow_lmtd_soft instead of counterflow_lmtd: returns
    min(dT1, dT2) instead of raising when dT <= 0
  - no raises for negative heat flows, negative mass flows, etc.
  - residuals Q - UA*LMTD_soft stay large and correctly signed,
    so the solver gets a real gradient signal back toward the
    physically valid region

This robust variant has the same roots as the strict evaluation
(LMTD_soft = LMTD when both dT > 0), so it converges to the same
physical solution.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    import CoolProp.CoolProp as CP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CoolProp is not installed. Install it e.g. with `pip install CoolProp`."
    ) from exc

import Thermodynamic_Properties.libr_props as lp

PRIMARY_VARIABLE_NAMES = ["T8", "T10", "x4", "x1", "T3", "T5"]
RESIDUAL_NAMES = [
    "R1_SHEX_energy",
    "R2_SHEX_UA",
    "R4_desorber_UA",
    "R5_condenser_UA",
    "R6_evaporator_UA",
    "R7_absorber_UA",
]

PRIMARY_TEMPERATURE_INDICES = (0, 1, 4, 5)


def kelvin_to_celsius(T_K: float) -> float:
    return float(T_K) - 273.15


def celsius_to_kelvin(T_C: float) -> float:
    return float(T_C) + 273.15


def primary_temperatures_C_to_K(
    z_user: np.ndarray | list[float] | tuple[float, ...]
) -> np.ndarray:
    """Converts the temperature components of the primary vector from degC to K."""
    z_internal = np.asarray(z_user, dtype=float).copy()
    z_internal[list(PRIMARY_TEMPERATURE_INDICES)] += 273.15
    return z_internal


def primary_temperatures_K_to_C(
    z_internal: np.ndarray | list[float] | tuple[float, ...]
) -> np.ndarray:
    """Converts the temperature components of the primary vector from K to degC."""
    z_user = np.asarray(z_internal, dtype=float).copy()
    z_user[list(PRIMARY_TEMPERATURE_INDICES)] -= 273.15
    return z_user


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ACInputs:
    # External inlet temperatures [degC]
    T_11_C: float
    T_13_C: float | None
    T_15_C: float | None
    T_17_C: float

    # External mass flows [kg/s]
    m_11: float
    m_13: float
    m_15: float

    # UA values [kW/K]
    UA_cond: float
    UA_evap: float
    UA_abs: float
    UA_des: float
    UA_shex: float | None = None
    Effectiveness_shex: float | None = None

    _: KW_ONLY

    # SHEX specification mode:
    # - "UA":  UA_shex is used (default)
    # - "NTU": shex_effectiveness is used
    shex_model: str = "UA"

    # External heat sink/source routing of desorber and evaporator:
    # - "parallel": default case with separate T_13 and T_15 given
    # - "series_absorber_to_condenser": internally T15 = T14
    # - "series_condenser_to_absorber": internally T13 = T16
    absorber_condenser_routing_mode: str = "parallel"

    # Cycle scaling specification:
    # - "m1": m1_spec is given
    # - "Qevap": Qevap_spec_kW is given, m1 is computed
    cycle_scale_spec_mode: str = "m1"
    m1_spec: float | None = None
    Qevap_spec_kW: float | None = None

    # External evaporator stream specification:
    # - "m17": m17_spec is given, T18 is computed
    # - "T18": T18_spec_C is given, m17 is computed
    evaporator_spec_mode: str = "m17"
    m17_spec: float | None = None
    T18_spec_C: float | None = None

    # External fluid: water
    cp_w_kJkgK: float = 4.2

    # Desorber vapor outlet: default is saturated vapor at the
    # low-pressure level
    desorber_vapor_superheat_K: float = 0.0

    # Solver
    solver_tol: float = 1.0e-9
    max_nfev: int = 5000
    penalty_level: float = 1.0e6

    def __post_init__(self) -> None:
        if self.absorber_condenser_routing_mode not in {
            "parallel",
            "series_absorber_to_condenser",
            "series_condenser_to_absorber",
        }:
            raise ValueError(
                "absorber_condenser_routing_mode must be 'parallel', "
                "'series_absorber_to_condenser', or 'series_condenser_to_absorber'."
            )

        if self.absorber_condenser_routing_mode == "parallel":
            if self.T_13_C is None:
                raise ValueError(
                    "For absorber_condenser_routing_mode='parallel', T_13_C must be given."
                )
            if self.T_15_C is None:
                raise ValueError(
                    "For absorber_condenser_routing_mode='parallel', T_15_C must be given."
                )
        elif self.absorber_condenser_routing_mode == "series_absorber_to_condenser":
            if self.T_13_C is None:
                raise ValueError(
                    "For absorber_condenser_routing_mode='series_absorber_to_condenser', "
                    "T_13_C must be given."
                )
            if self.T_15_C is not None:
                raise ValueError(
                    "For absorber_condenser_routing_mode='series_absorber_to_condenser', T_15_C "
                    "must not be set; internally T15 = T14."
                )
        else:
            if self.T_15_C is None:
                raise ValueError(
                    "For absorber_condenser_routing_mode='series_condenser_to_absorber', "
                    "T_15_C must be given; internally T13 = T16."
                )
            if self.T_13_C is not None:
                raise ValueError(
                    "For absorber_condenser_routing_mode='series_condenser_to_absorber', T_13_C "
                    "must not be set; internally T13 = T16."
                )

        if self.cycle_scale_spec_mode not in {"m1", "Qeva"}:
            raise ValueError("cycle_scale_spec_mode must be 'm1' or 'Qeva'.")
        if self.cycle_scale_spec_mode == "m1":
            if self.m1_spec is None:
                raise ValueError("For cycle_scale_spec_mode='m1', m1_spec must be given.")
            if self.Qevap_spec_kW is not None:
                raise ValueError(
                    "For cycle_scale_spec_mode='m1', Qeva_spec_kW must not be set."
                )
            if self.m1_spec <= 0.0:
                raise ValueError("For cycle_scale_spec_mode='m1', m1_spec > 0 must hold.")
        else:
            if self.Qevap_spec_kW is None:
                raise ValueError(
                    "For cycle_scale_spec_mode='Qeva', Qeva_spec_kW must be given."
                )
            if self.m1_spec is not None:
                raise ValueError(
                    "For cycle_scale_spec_mode='Qeva', m1_spec must not be set."
                )
            if self.Qevap_spec_kW <= 0.0:
                raise ValueError("For cycle_scale_spec_mode='Qeva', Qeva_spec_kW > 0 must hold.")

        if self.evaporator_spec_mode not in {"m17", "T18"}:
            raise ValueError("evaporator_spec_mode must be 'm17' or 'T18'.")
        if self.evaporator_spec_mode == "m17":
            if self.m17_spec is None:
                raise ValueError("For evaporator_spec_mode='m17', m17_spec must be given.")
            if self.T18_spec_C is not None:
                raise ValueError(
                    "For evaporator_spec_mode='m17', T18_spec_C must not be set."
                )
            if self.m17_spec <= 0.0:
                raise ValueError("For evaporator_spec_mode='m17', m17_spec > 0 must hold.")
        else:
            if self.T18_spec_C is None:
                raise ValueError("For evaporator_spec_mode='T18', T18_spec_C must be given.")
            if self.m17_spec is not None:
                raise ValueError(
                    "For evaporator_spec_mode='T18', m17_spec must not be set."
                )
            if self.T18_spec_C >= self.T_17_C:
                raise ValueError(
                    "For evaporator_spec_mode='T18', T18_spec_C < T_17_C must hold."
                )
        if self.shex_model not in {"UA", "NTU"}:
            raise ValueError("shex_model must be 'UA' or 'NTU'.")
        if self.shex_model == "UA":
            if self.UA_shex is None:
                raise ValueError("For shex_model='UA', UA_shex must be given.")
            if self.UA_shex <= 0.0:
                raise ValueError("For shex_model='UA', UA > 0 must hold.")
        else:
            if self.Effectiveness_shex is None:
                raise ValueError("For shex_model='NTU', Effectiveness_shex must be given.")
            if self.UA_shex is not None:
                raise ValueError(
                    "For shex_model='NTU', UA_shex must not be set."
                )

    @property
    def T_17(self) -> float:
        return celsius_to_kelvin(self.T_17_C)

    @property
    def T18_spec(self) -> float:
        if self.T18_spec_C is None:
            raise AttributeError("T18_spec_C is not set for this specification.")
        return celsius_to_kelvin(self.T18_spec_C)

    @property
    def T_13(self) -> float:
        if self.T_13_C is None:
            raise AttributeError(
                "T_13_C is not set for the chosen routing mode."
            )
        return celsius_to_kelvin(self.T_13_C)

    @property
    def T_15(self) -> float:
        if self.T_15_C is None:
            raise AttributeError(
                "T_15_C is not set for the chosen routing mode."
            )
        return celsius_to_kelvin(self.T_15_C)

    @property
    def T_11(self) -> float:
        return celsius_to_kelvin(self.T_11_C)

    @property
    def uses_serial_absorber_to_condenser_routing(self) -> bool:
        return self.absorber_condenser_routing_mode == "series_absorber_to_condenser"

    @property
    def uses_serial_condenser_to_absorber_routing(self) -> bool:
        return self.absorber_condenser_routing_mode == "series_condenser_to_absorber"

    @property
    def uses_any_serial_absorber_condenser_routing(self) -> bool:
        return self.absorber_condenser_routing_mode in {
            "series_absorber_to_condenser",
            "series_condenser_to_absorber",
        }

    @property
    def condenser_temperature_reference(self) -> float:
        """Reference temperature for initial guesses and bounds of the condenser."""
        if self.uses_serial_absorber_to_condenser_routing:
            return self.T_13
        return self.T_15

    @property
    def absorber_temperature_reference(self) -> float:
        """Reference temperature for initial guesses and bounds of the absorber."""
        if self.uses_serial_condenser_to_absorber_routing:
            return self.T_15
        return self.T_13


@dataclass(frozen=True)
class SolveInfo:
    success: bool
    status: int
    message: str
    cost: float
    scaled_residual_norm: float
    raw_residual_norm: float | None
    nfev: int
    final_point_evaluable: bool
    final_evaluation_error: str | None


@dataclass(frozen=True)
class ModelEvaluation:
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    pump_work_W: Dict[str, float]
    lmtd_K: Dict[str, float]
    compositions: Dict[str, float]
    flash_outputs: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    checks: Dict[str, bool]
    validity_messages: List[str]


@dataclass(frozen=True)
class ACResult:
    inputs: ACInputs
    solve_info: SolveInfo
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    pump_work_W: Dict[str, float]
    lmtd_K: Dict[str, float]
    compositions: Dict[str, float]
    flash_outputs: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    checks: Dict[str, bool]
    validity_messages: List[str]

@dataclass(frozen=True)
class ModelTrace:
    primary_variables: Dict[str, float]
    values: Dict[str, float]
    stage: str
    success: bool
    error_type: str | None
    error_message: str | None

class ModelEvaluationError(RuntimeError):
    """Internal error during model evaluation."""


# ---------------------------------------------------------------------------
# Water property functions (CoolProp wrapper)
# ---------------------------------------------------------------------------

def water_h_kjkg_PT(P_pa: float, T_K: float) -> float:
    return CP.PropsSI("H", "P", P_pa, "T", T_K, "Water") / 1000.0


def water_h_kjkg_PQ(P_pa: float, Q: float) -> float:
    return CP.PropsSI("H", "P", P_pa, "Q", Q, "Water") / 1000.0


def water_T_K_PH(P_pa: float, h_kjkg: float) -> float:
    return CP.PropsSI("T", "P", P_pa, "H", h_kjkg * 1000.0, "Water")


def water_p_sat_from_T(T_K: float, Q: float) -> float:
    return CP.PropsSI("P", "T", T_K, "Q", Q, "Water")


def water_T_sat_from_p(P_pa: float, Q: float) -> float:
    return CP.PropsSI("T", "P", P_pa, "Q", Q, "Water")


def water_rho_kgm3_PQ(P_pa: float, Q: float) -> float:
    return CP.PropsSI("D", "P", P_pa, "Q", Q, "Water")


# ---------------------------------------------------------------------------
# General helper functions
# ---------------------------------------------------------------------------

def lmtd(delta_T_1: float, delta_T_2: float) -> float:
    """Strict LMTD: raises ModelEvaluationError for dT <= 0."""
    if delta_T_1 <= 0.0 or delta_T_2 <= 0.0:
        raise ModelEvaluationError(
            f"LMTD undefined because delta_T_1={delta_T_1:.6f} K or"
            f" delta_T_2={delta_T_2:.6f} K is not positive."
        )
    if math.isclose(delta_T_1, delta_T_2, rel_tol=1.0e-10, abs_tol=1.0e-10):
        return 0.5 * (delta_T_1 + delta_T_2)
    return (delta_T_1 - delta_T_2) / math.log(delta_T_1 / delta_T_2)


def lmtd_soft(delta_T_1: float, delta_T_2: float) -> float:
    """Robust LMTD for the solver path.

    For dT > 0: identical to lmtd().
    For dT <= 0: returns min(dT1, dT2) (negative/zero).

    This keeps the residual Q - UA*LMTD_soft defined and continuous
    everywhere. Negative dT drives LMTD_soft negative, so the residual
    grows large and positive, giving the solver a real gradient signal
    back toward the physically valid region.

    Same roots as lmtd(): whenever the solution is physical (dT > 0),
    lmtd() == lmtd_soft(), so the solution itself is unaffected.
    """
    if delta_T_1 <= 0.0 or delta_T_2 <= 0.0:
        return min(delta_T_1, delta_T_2)
    if math.isclose(delta_T_1, delta_T_2, rel_tol=1.0e-10, abs_tol=1.0e-10):
        return 0.5 * (delta_T_1 + delta_T_2)
    return (delta_T_1 - delta_T_2) / math.log(delta_T_1 / delta_T_2)


def counterflow_lmtd(hot_in: float, hot_out: float, cold_in: float, cold_out: float) -> float:
    return lmtd(hot_in - cold_out, hot_out - cold_in)


def counterflow_lmtd_soft(hot_in: float, hot_out: float, cold_in: float, cold_out: float) -> float:
    return lmtd_soft(hot_in - cold_out, hot_out - cold_in)


def heating_outlet_temperature(T_in: float, Q_kW: float, m_kg_s: float, cp_kJkgK: float) -> float:
    if m_kg_s <= 0.0 or cp_kJkgK <= 0.0:
        raise ModelEvaluationError("External heat capacity flow rate must be positive.")
    return T_in + Q_kW / (m_kg_s * cp_kJkgK)


def cooling_outlet_temperature(T_in: float, Q_kW: float, m_kg_s: float, cp_kJkgK: float) -> float:
    if m_kg_s <= 0.0 or cp_kJkgK <= 0.0:
        raise ModelEvaluationError("External heat capacity flow rate must be positive.")
    return T_in - Q_kW / (m_kg_s * cp_kJkgK)

def water_throttle_state(p_out_pa: float, h_in_kJkg: float) -> Dict[str, float]:
    """Isenthalpic throttle for the pure refrigerant (water), condenser -> evaporator.

    Unlike the solution throttle (LiBr/H2O, see lp.flash_valve_state_5_to_6),
    no mixture calculation is needed here: since this is pure water, a simple
    vapor-quality calculation at p_out_pa suffices.
    """
    h_f = water_h_kjkg_PQ(p_out_pa, Q=0.0)
    h_g = water_h_kjkg_PQ(p_out_pa, Q=1.0)
    T_sat = water_T_sat_from_p(p_out_pa, Q=0.0)

    if h_g <= h_f:
        # degenerate case (numerical), should not occur in practice
        return {"T9_K": T_sat, "q9": 0.0, "h9_kJ_kg": h_in_kJkg}
 
    if h_in_kJkg <= h_f:
        q = 0.0
        T = T_sat
    elif h_in_kJkg >= h_g:
        q = 1.0
        T = water_T_K_PH(p_out_pa, h_in_kJkg)
    else:
        q = (h_in_kJkg - h_f) / (h_g - h_f)
        T = T_sat
 
    return {"T9_K": T, "q9": q, "h9_kJ_kg": h_in_kJkg}

def _penalty_vector(size: int, level: float) -> np.ndarray:
    return np.full(size, level, dtype=float)


def _residual_scales(m1: float) -> np.ndarray:
    """Scaling of the six energetic residuals [kW]."""
    return np.array(
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        dtype=float,
    )


def _state_dict(
    T_K: float,
    p_Pa: float | None = None,
    m_kg_s: float | None = None,
    h_kJ_kg: float | None = None,
    x_LiBr_mol: float | None = None,
    w_LiBr: float | None = None,
) -> Dict[str, float]:
    state: Dict[str, float] = {"T_K": float(T_K)}
    if p_Pa is not None:
        state["p_Pa"] = float(p_Pa)
    if m_kg_s is not None:
        state["m_kg_s"] = float(m_kg_s)
    if h_kJ_kg is not None:
        state["h_kJ_kg"] = float(h_kJ_kg)
    if x_LiBr_mol is not None:
        state["x_LiBr_mol"] = float(x_LiBr_mol)
    if w_LiBr is not None:
        state["w_LiBr"] = float(w_LiBr)
    return state


def _calculate_kpis(
    *,
    Q_des: float,
    Q_evap: float,
    T12: float,
    T_hs_ref: float,
    m1: float,
    m7: float,
) -> Dict[str, float]:
    denominator_cop = Q_des
    cop = float("nan") if abs(denominator_cop) <= 1.0e-12 else Q_evap / denominator_cop
    fr = float("nan") if abs(m7) <= 1.0e-12 else m1 / m7

    return {
        "COP": cop,
        "GTL_K": T12 - T_hs_ref,
        "FR": fr,
    }


def _resolve_cycle_scale(
    inputs: ACInputs, *, w4: float, w1: float, h9: float, h10: float, strict: bool
) -> float:
    """Resolves the cycle scaling to the pumped solution mass flow m1."""
    if inputs.cycle_scale_spec_mode == "m1":
        m1 = float(inputs.m1_spec)  # guaranteed by __post_init__
        if strict and m1 <= 0.0:
            raise ModelEvaluationError("Pumped solution mass flow m1 must be positive.")
        return m1

    w4_balance = w4 if strict else max(w4, 1.0e-9)
    ratio = w1 / w4_balance
    denominator = (1 - ratio) * (h10 - h9)

    if strict:
        if abs(denominator) <= 1.0e-12:
            raise ModelEvaluationError(
                "Cannot resolve cycle scaling from Q_eva because the denominator is near zero."
            )
        m1 = float(inputs.Qevap_spec_kW) / denominator
        if m1 <= 0.0:
            raise ModelEvaluationError(
                f"Computed solution mass flow m1 is not positive: m1={m1:.6f} kg/s."
            )
        return m1

    denominator_safe = denominator
    if abs(denominator_safe) <= 1.0e-12:
        denominator_safe = 1.0e-12 if denominator_safe >= 0.0 else -1.0e-12
    return float(inputs.Qevap_spec_kW) / denominator_safe


def _resolve_evaporator_external_stream(
    inputs: ACInputs, Q_evap: float, *, strict: bool
) -> tuple[float, float]:
    """Resolves the evaporator specification to internal working quantities.

    Returns
    -------
    (m17, T18)
    """
    if inputs.evaporator_spec_mode == "m17":
        m17 = float(inputs.m17_spec)  # guaranteed by __post_init__
        if strict:
            T18 = cooling_outlet_temperature(inputs.T_17, Q_evap, m17, inputs.cp_w_kJkgK)
        else:
            T18 = inputs.T_17 - Q_evap / (m17 * inputs.cp_w_kJkgK)
        return m17, T18

    T18 = inputs.T18_spec
    delta_T = inputs.T_17 - T18
    if strict:
        if delta_T <= 0.0:
            raise ModelEvaluationError(
                "For evaporator_spec_mode='T18', T17 > T18 must hold."
            )
        if inputs.cp_w_kJkgK <= 0.0:
            raise ModelEvaluationError("External heat capacity flow rate must be positive.")
    m17 = Q_evap / (inputs.cp_w_kJkgK * delta_T)
    return m17, T18


def _resolve_condenser_external_inlet_temperature(inputs: ACInputs, T14: float | None = None) -> float:
    """Resolves the external condenser inlet temperature.

    - parallel: T15 is read from the inputs
    - series_absorber_to_condenser: T15 equals the external absorber outlet T14
    - series_condenser_to_absorber: T15 stays an external input
    """
    if inputs.uses_serial_absorber_to_condenser_routing:
        if T14 is None:
            raise ModelEvaluationError("T14 must be known for series_absorber_to_condenser.")
        return T14
    return inputs.T_15


def _resolve_absorber_external_inlet_temperature(inputs: ACInputs, T16: float | None = None) -> float:
    """Resolves the external absorber inlet temperature.

    - parallel: T13 is read from the inputs
    - series_absorber_to_condenser: T13 stays an external input
    - series_condenser_to_absorber: T13 equals the external condenser outlet T16
    """
    if inputs.uses_serial_condenser_to_absorber_routing:
        if T16 is None:
            raise ModelEvaluationError("T16 must be known for series_condenser_to_absorber.")
        return T16
    return inputs.T_13


# ---------------------------------------------------------------------------
# Solver helper functions
# ---------------------------------------------------------------------------

def initial_guess(inputs: ACInputs) -> np.ndarray:
    """Heuristic initial guess for the 6-dimensional solver vector."""
    T_cond_ref = inputs.condenser_temperature_reference
    T_abs_ref = inputs.absorber_temperature_reference
    return np.array(
        [
            T_cond_ref + 15.0,      # T8
            inputs.T_17 - 8.0,      # T10
            0.243,                   # x4
            0.22,                  # x1
            T_abs_ref + 12.0,       # T3
            inputs.T_11 - 47.0,     # T5
        ],
        dtype=float,
    )


def bounds(inputs: ACInputs) -> Tuple[np.ndarray, np.ndarray]:
    T_cond_ref = inputs.condenser_temperature_reference
    T_abs_ref = inputs.absorber_temperature_reference
    lower = np.array(
        [
            inputs.T_17 + 1.0,      # T8
            274.15,                 # T10
            0.08,                   # x4
            0.05,                   # x1
            inputs.T_17 + 1.0,      # T3
            inputs.T_17 + 1.0,      # T5
        ],
        dtype=float,
    )
    upper = np.array(
        [
            min(inputs.T_11 - 1.0, 420.0),   # T8
            min(inputs.T_17 + 0.5, 500.0),   # T10
            0.39,                           # x4
            0.34,                           # x1
            500.0,                          # T3
            500.0,                          # T5
        ],
        dtype=float,
    )
    return lower, upper



# ---------------------------------------------------------------------------
# Shared model core (strict for final evaluation, robust for solver path)
# ---------------------------------------------------------------------------

class _SoftResidualVector(RuntimeError):
    """Internal exception: the robust residual vector is already determined."""

    def __init__(self, residuals_scaled: np.ndarray):
        super().__init__("Soft residual vector ready.")
        self.residuals_scaled = np.asarray(residuals_scaled, dtype=float)


def _counterflow_lmtd_mode(
    *, strict: bool, hot_in: float, hot_out: float, cold_in: float, cold_out: float
) -> float:
    if strict:
        return counterflow_lmtd(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out)
    return counterflow_lmtd_soft(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out)


def _scaled_residual_array(model: ModelEvaluation) -> np.ndarray:
    return np.array([model.residuals_scaled[name] for name in RESIDUAL_NAMES], dtype=float)


def _evaluate_model_common(z: np.ndarray, inputs: ACInputs, *, strict: bool) -> ModelEvaluation:
    """Shared model core for the strict final evaluation and the robust solver path.

    strict=True:
        - identical behavior to the former evaluate_model() function
        - raises ModelEvaluationError for unphysical states

    strict=False:
        - identical equation structure
        - robust variants only where needed for the solver path
        - no raises for negative heat flows / mass flows
        - counterflow_lmtd_soft instead of counterflow_lmtd
        - fallback for T2
        - direct residual vector on a fundamental pressure violation p_high <= p_low
    """
    T8, T10, x4, x1, T3, T5 = map(float, z)

    # ------------------------------------------------------------------
    # 1) Refrigerant pressure levels
    # ------------------------------------------------------------------
    p_low = water_p_sat_from_T(T10, Q=1.0)
    p_high = water_p_sat_from_T(T8, Q=0.0)
    if p_high <= p_low:
        if strict:
            raise ModelEvaluationError(f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa).")
        pen = (p_low - p_high + 100.0) / 100.0
        raise _SoftResidualVector(np.full(len(RESIDUAL_NAMES), pen, dtype=float))

    # ------------------------------------------------------------------
    # 2) Saturated solution states, concentrations, and early property values
    # ------------------------------------------------------------------
    T1 = lp.T_sat_solution_from_p_x(p_low, x1)
    T4 = lp.T_sat_solution_from_p_x(p_high, x4)

    w1 = lp.w_libr_from_x(x1)
    w4 = lp.w_libr_from_x(x4)

    if strict and not (w4 > w1 > 0.0):
        raise ModelEvaluationError(
            f"Concentration hierarchy violated: w4={w4:.6f}, w1={w1:.6f}. Expected w4 > w1."
        )

    h3 = lp.h_solution_mass_kjkg(T3, x1)
    h4 = lp.h_solution_mass_kjkg(T4, x4)
    T7 = lp.T_sat_solution_from_p_x(p_high, x1) + inputs.desorber_vapor_superheat_K
    h7 = water_h_kjkg_PT(p_high, T7)
    h8 = water_h_kjkg_PQ(p_high, Q=0.0)
    h10 = water_h_kjkg_PQ(p_low, Q=1.0)

    # ------------------------------------------------------------------
    # 9) Throttle 8 -> 9
    # ------------------------------------------------------------------
    h9 = h8
    refrigerant_throttle = water_throttle_state(p_low, h9)
    T9 = refrigerant_throttle["T9_K"]
    q9 = refrigerant_throttle["q9"]

    # ------------------------------------------------------------------
    # 3) Cycle scaling and mass flows
    # ------------------------------------------------------------------
    m1 = _resolve_cycle_scale(inputs, w4=w4, w1=w1, h9=h9, h10=h10, strict=strict)
    m2 = m3 = m1
    w4_for_balance = w4 if strict else max(w4, 1.0e-9)
    m4 = m3 * w1 / w4_for_balance
    m5 = m6 = m4
    m7 = m8 = m9 = m10 = m1 - m4

    if strict and m7 <= 0.0:
        raise ModelEvaluationError(f"Refrigerant mass flow not positive: m7={m7:.6f} kg/s.")

    if strict:
        if m10 <= 0.0:
            raise ModelEvaluationError(f"m10 not positive: m10={m10:.6f} kg/s.")

    # ------------------------------------------------------------------
    # 4) Solution enthalpies and solution pump 1 -> 2
    # ------------------------------------------------------------------
    h5 = lp.h_solution_mass_kjkg(T5, x4)
    h1 = lp.h_solution_mass_kjkg(T1, x1)

    rho1 = lp.rho_solution_mass(T1, x1)
    v1 = 1.0 / rho1
    W_sol_pump = m1 * v1 * (p_high - p_low) / 1000.0  # kW
    h2 = h1 + W_sol_pump / m1
    if strict:
        T2 = lp.T_from_h_x_mass(h2, x1)
    else:
        try:
            T2 = lp.T_from_h_x_mass(h2, x1)
        except Exception:
            T2 = T1

    # ------------------------------------------------------------------
    # 5) Solution heat exchanger (SHEX): 3 -> 2 and 5 -> 4
    # ------------------------------------------------------------------
    Q_shex_hot = m4 * (h4 - h5)
    Q_shex_cold = m1 * (h3 - h2)
    if strict and Q_shex_hot <= 0.0:
        raise ModelEvaluationError(f"Q_shex_hot not positive: {Q_shex_hot:.6f} kW.")
    if strict and Q_shex_cold <= 0.0:
        raise ModelEvaluationError(f"Q_shex_cold not positive: {Q_shex_cold:.6f} kW.")
    Q_shex = Q_shex_hot

    # LMTD is computed independently of the specification mode, since T2..T5 are always known
    lmtd_shex = _counterflow_lmtd_mode(
        strict=strict, hot_in=T4, hot_out=T5, cold_in=T2, cold_out=T3
    )

    # SHEX: heat-transfer residual (R2) depending on the specification mode
    if inputs.shex_model == "UA":
        R2_shex = Q_shex - inputs.UA_shex * lmtd_shex
        UA_shex_calc = inputs.UA_shex
    else:
        # effectiveness-NTU method
        C23 = (h2 - h3) / (T2 - T3) if abs(T2 - T3) > 1.0e-12 else float("nan")
        C45 = (h4 - h5) / (T4 - T5) if abs(T4 - T5) > 1.0e-12 else float("nan")
        C2_3 = m1 * C23
        C4_5 = m4 * C45
        C_min = min(C2_3, C4_5)
        Q_shex_max = C_min * (T4 - T2)
        R2_shex = Q_shex - inputs.Effectiveness_shex * Q_shex_max
        # UA back-calculated from LMTD (informational only, not part of the residuals)
        if strict and lmtd_shex <= 0.0:
            raise ModelEvaluationError(
                f"LMTD_shex not positive, cannot back-calculate UA: {lmtd_shex:.6f} K."
            )
        UA_shex_calc = Q_shex / lmtd_shex if lmtd_shex > 0.0 else float("nan")

    # SHEX pinch residual: smallest temperature gap = dT_min_shex
    dT_shex_hot_end  = T4 - T3   # hot in  / cold out
    dT_shex_cold_end = T5 - T2   # hot out / cold in
    pinch_shex = min(dT_shex_hot_end, dT_shex_cold_end)

    # ------------------------------------------------------------------
    # 6) Throttle 5 -> 6 (isenthalpic, T6 from flash throttle)
    # ------------------------------------------------------------------
    h6 = h5
    if strict:
        flash = lp.flash_valve_state_5_to_6(
            p_out_pa=p_low,
            h5_kJkg=h5,
            m5_kg_s=m5,
            x5_libr_mol=x4,
        )
        T6 = flash["T6_K"]
    else:
        try:
            flash = lp.flash_valve_state_5_to_6(
                p_out_pa=p_low,
                h5_kJkg=h5,
                m5_kg_s=m5,
                x5_libr_mol=x4,
            )
            T6 = flash["T6_K"]
        except Exception:
            T6 = water_T_sat_from_p(p_low, Q=0.0)
            flash = {
                "T6_K": T6,
                "x6_LiBr_mol": x4,
                "w6_LiBr": w4,
                "m6_sol_kg_s": m4,
                "m6_flash_kg_s": 0.0,
                "h6_sol_kJ_kg": h6,
                "h6_flash_kJ_kg": float("nan"),
                "h6_mix_kJ_kg": h6,
                "flash_fraction": 0.0,
            }
    flash_outputs = {key: float(value) for key, value in flash.items()}

    # ------------------------------------------------------------------
    # 7) Refrigerant vapor path 7, and external hot side of desorber/evaporator
    # ------------------------------------------------------------------
    Q_des = - m1 * h3 + m7 * h7 + m4 * h4
    if strict and Q_des <= 0.0:
        raise ModelEvaluationError(f"Desorber heat flow not positive: Q_des={Q_des:.6f} kW.")

    if strict:
        T12 = cooling_outlet_temperature(inputs.T_11, Q_des, inputs.m_11, inputs.cp_w_kJkgK)
    else:
        T12 = inputs.T_11 - Q_des / (inputs.m_11 * inputs.cp_w_kJkgK)

    lmtd_des = _counterflow_lmtd_mode(
        strict=strict, hot_in=inputs.T_11, hot_out=T12, cold_in=T7, cold_out=T4
    )

    # Desorber pinch: min of both ends
    dT_des_hot_end  = inputs.T_11 - T4          # hot in / cold out
    dT_des_cold_end = T12  - T7  # hot out / cold in
    pinch_des  = min(dT_des_hot_end,  dT_des_cold_end)

    # ------------------------------------------------------------------
    # 10) Evaporator 9 -> 10 and coupled external hot side
    # ------------------------------------------------------------------
    Q_evap = m9 * (h10 - h9)
    if strict and Q_evap <= 0.0:
        raise ModelEvaluationError(f"Evaporator heat flow not positive: Q_evap={Q_evap:.6f} kW.")
    m17, T18 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=strict)

    lmtd_evap = _counterflow_lmtd_mode(
        strict=strict, hot_in=inputs.T_17, hot_out=T18, cold_in=T9, cold_out=T10
    )

    # Heat share for subcooling:
    h_f_high = water_h_kjkg_PQ(p_high, Q=0.0)

    Q_subcool = m9 * (h_f_high - h9)
    Q_subcool = max(0.0, min(Q_subcool, Q_evap))

    # Chilled-water return temperature at the transition
    # subcooling -> evaporation
    T18_sat = T18 + Q_subcool / (
        m17 * inputs.cp_w_kJkgK
    )

    # Pinch candidates
    dT_evap_in = inputs.T_17 - T10
    dT_evap_sat = T18_sat - T10
    dT_evap_out = T18 - T9

    pinch_evap = min(dT_evap_in, dT_evap_sat, dT_evap_out)

    # ------------------------------------------------------------------
    # 12) Absorber (global energy balance, local LMTD)
    # ------------------------------------------------------------------
    Q_abs = m10 * h10 + m4 * h6 - m1 * h1
    if strict and Q_abs <= 0.0:
        raise ModelEvaluationError(f"Absorber heat flow not positive: Q_abs={Q_abs:.6f} kW.")

    # ------------------------------------------------------------------
    # 8) Condenser 7 -> 8
    # ------------------------------------------------------------------
    Q_cond = m7 * (h7 - h8)
    if strict and Q_cond <= 0.0:
        raise ModelEvaluationError(f"Condenser heat flow not positive: Q_cond={Q_cond:.6f} kW.")
    if inputs.uses_serial_condenser_to_absorber_routing:
        T15_in = _resolve_condenser_external_inlet_temperature(inputs)
        if strict:
            T16 = heating_outlet_temperature(T15_in, Q_cond, inputs.m_15, inputs.cp_w_kJkgK)
        else:
            T16 = T15_in + Q_cond / (inputs.m_15 * inputs.cp_w_kJkgK)
        lmtd_cond = _counterflow_lmtd_mode(
            strict=strict, hot_in=T8, hot_out=T8, cold_in=T15_in, cold_out=T16
        )
        T13_in = _resolve_absorber_external_inlet_temperature(inputs, T16)
        if strict:
            T14 = heating_outlet_temperature(T13_in, Q_abs, inputs.m_13, inputs.cp_w_kJkgK)
        else:
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        lmtd_abs = _counterflow_lmtd_mode(
            strict=strict, hot_in=T6, hot_out=T1, cold_in=T13_in, cold_out=T14
        )
    else:
        T13_in = _resolve_absorber_external_inlet_temperature(inputs)
        if strict:
            T14 = heating_outlet_temperature(T13_in, Q_abs, inputs.m_13, inputs.cp_w_kJkgK)
        else:
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        lmtd_abs = _counterflow_lmtd_mode(
            strict=strict, hot_in=T6, hot_out=T1, cold_in=T13_in, cold_out=T14
        )
        T15_in = _resolve_condenser_external_inlet_temperature(inputs, T14)
        if strict:
            T16 = heating_outlet_temperature(T15_in, Q_cond, inputs.m_15, inputs.cp_w_kJkgK)
        else:
            T16 = T15_in + Q_cond / (inputs.m_15 * inputs.cp_w_kJkgK)
        lmtd_cond = _counterflow_lmtd_mode(
            strict=strict, hot_in=T8, hot_out=T8, cold_in=T15_in, cold_out=T16
        )

    # Absorber pinch: min of both ends (location depends on operating point)
    dT_abs_hot_end  = T6 - T14   # hot in / cold out
    dT_abs_cold_end = T1 - T13_in   # hot out / cold in
    pinch_abs  = min(dT_abs_hot_end,  dT_abs_cold_end)

    # Heat share for desuperheating:
    h_g_high = water_h_kjkg_PQ(p_high, Q=1.0)

    Q_desuperheat = m7 * (h7 - h_g_high)
    Q_desuperheat = max(0.0, min(Q_desuperheat, Q_cond))

    # Cooling-water temperature at the transition
    # desuperheating -> condensation
    T16_sat = T16 - Q_desuperheat / (
        inputs.m_15 * inputs.cp_w_kJkgK
    )

    # Pinch candidates
    dT_cond_in = T7 - T16
    dT_cond_sat = T8 - T16_sat
    dT_cond_out = T8 - T15_in
    pinch_cond = min(dT_cond_in, dT_cond_sat, dT_cond_out)

    # ------------------------------------------------------------------
    # 13) Residuals of the 6x6 system
    # ------------------------------------------------------------------
    residuals_raw_array = np.array(
        [
            Q_shex_hot - Q_shex_cold,
            R2_shex,
            Q_des - inputs.UA_des * lmtd_des,
            Q_cond - inputs.UA_cond * lmtd_cond,
            Q_evap - inputs.UA_evap * lmtd_evap,
            Q_abs - inputs.UA_abs * lmtd_abs,
        ],
        dtype=float,
    )
    scales = _residual_scales(m1)
    residuals_scaled_array = residuals_raw_array / scales

    residuals_raw = dict(zip(RESIDUAL_NAMES, residuals_raw_array.tolist()))
    residuals_scaled = dict(zip(RESIDUAL_NAMES, residuals_scaled_array.tolist()))

    # ------------------------------------------------------------------
    # 14) State validation and plausibility checks
    # ------------------------------------------------------------------
    validity_messages: List[str] = []
    crystallization_safe_all = True
    for label, T_state, w_state in [
        ("1 liquid phase after throttle", flash["T6_K"], flash["w6_LiBr"]),
        ("1", T1, w1), ("2", T2, w1), ("3", T3, w1), ("4", T4, w4)
    ]:
        validity = lp.validate_solution_state(T_state, w_state, label=f"State {label}")
        validity_messages.append(validity.message)
        crystallization_safe_all = crystallization_safe_all and validity.crystallization_safe

    checks = {
        "p_high_gt_p_low": p_high > p_low,
        "w4_gt_w1": w4 > w1,
        "m7_positive": m7 > 0.0,
        "absorber_condenser_temperature_coupling_ok": (
            (abs(T15_in - T14) <= 1.0e-12)
            if inputs.uses_serial_absorber_to_condenser_routing
            else ((abs(T13_in - T16) <= 1.0e-12) if inputs.uses_serial_condenser_to_absorber_routing else True)
        ),
        "crystallization_safe_all_checked_states": crystallization_safe_all,
    }

    diagnostics = {
        "p_low_Pa": p_low,
        "p_high_Pa": p_high,
        "pressure_ratio_high_over_low": p_high / p_low,
        "T18_K": T18,
        "m6_kg_s": m6,
        "m17_kg_s": m17,
        "T13_K": T13_in,
        "T14_K": T14,
        "T15_K": T15_in,
        "T16_K": T16,
        "T12_K": T12,
        "deltaT_shex_1_K": dT_shex_hot_end,
        "deltaT_shex_2_K": dT_shex_cold_end,
        "deltaT_des_1_K": dT_des_hot_end,
        "deltaT_des_2_K": dT_des_cold_end,
        "deltaT_cond_1_K": dT_cond_in,
        "deltaT_cond_2_K": dT_cond_out,
        "deltaT_evap_1_K": dT_evap_in,
        "deltaT_evap_2_K": dT_evap_out,
        "deltaT_abs_1_K": dT_abs_hot_end,
        "deltaT_abs_2_K": dT_abs_cold_end,
    }

    # ------------------------------------------------------------------
    # 15) State dictionary
    # ------------------------------------------------------------------
    states = {
        "1":  _state_dict(T1,          p_Pa=p_low,  m_kg_s=m1,  h_kJ_kg=h1,  x_LiBr_mol=x1,  w_LiBr=w1),
        "2":  _state_dict(T2,          p_Pa=p_high, m_kg_s=m2,  h_kJ_kg=h2,  x_LiBr_mol=x1,  w_LiBr=w1),
        "3":  _state_dict(T3,          p_Pa=p_high, m_kg_s=m3,  h_kJ_kg=h3,  x_LiBr_mol=x1,  w_LiBr=w1),
        "4":  _state_dict(T4,          p_Pa=p_high, m_kg_s=m4,  h_kJ_kg=h4,  x_LiBr_mol=x4,  w_LiBr=w4),
        "5":  _state_dict(T5,          p_Pa=p_high, m_kg_s=m5,  h_kJ_kg=h5,  x_LiBr_mol=x4,  w_LiBr=w4),
        "6":  _state_dict(T6,          p_Pa=p_low,  m_kg_s=m6,  h_kJ_kg=h6,  x_LiBr_mol=x4,  w_LiBr=w4),
        "7":  _state_dict(T7,          p_Pa=p_high,  m_kg_s=m7,  h_kJ_kg=h7,  x_LiBr_mol=0.0, w_LiBr=0.0),
        "8":  _state_dict(T8,          p_Pa=p_high,  m_kg_s=m8,  h_kJ_kg=h8,  x_LiBr_mol=0.0, w_LiBr=0.0),
        "9":  _state_dict(T9,          p_Pa=p_low, m_kg_s=m9,  h_kJ_kg=h9,  x_LiBr_mol=0.0, w_LiBr=0.0),
        "10": _state_dict(T10,         p_Pa=p_low, m_kg_s=m10, h_kJ_kg=h10, x_LiBr_mol=0.0, w_LiBr=0.0),
        "11": _state_dict(inputs.T_11, m_kg_s=inputs.m_11),
        "12": _state_dict(T12,         m_kg_s=inputs.m_11),
        "13": _state_dict(T13_in,       m_kg_s=inputs.m_13),
        "14": _state_dict(T14,         m_kg_s=inputs.m_13),
        "15": _state_dict(T15_in,       m_kg_s=inputs.m_15),
        "16": _state_dict(T16,         m_kg_s=inputs.m_15),
        "17": _state_dict(inputs.T_17, m_kg_s=m17),
        "18": _state_dict(T18,         m_kg_s=m17),
    }

    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, [T8, T10, x4, x1, T3, T5]))
    if inputs.uses_serial_absorber_to_condenser_routing:
        T_hs_ref = inputs.T_13
    else:
        T_hs_ref = T15_in
    kpis = _calculate_kpis(
        Q_des=Q_des,
        Q_evap=Q_evap,
        T12=T12,
        T_hs_ref=T_hs_ref,
        m1=m1,
        m7=m7,
    )

    return ModelEvaluation(
        primary_variables=primary_variables,
        states=states,
        heat_flows_kW={
            "Q_shex": Q_shex,
            "Q_des": Q_des,
            "Q_cond": Q_cond,
            "Q_evap": Q_evap,
            "Q_abs": Q_abs,
        },
        kpis=kpis,
        pump_work_W={
            "W_sol_pump": W_sol_pump*1000,
        },
        lmtd_K={
            "LMTD_shex": lmtd_shex,
            "LMTD_des": lmtd_des,
            "LMTD_cond": lmtd_cond,
            "LMTD_evap": lmtd_evap,
            "LMTD_abs": lmtd_abs,
            "UA_shex_calc": UA_shex_calc,
            "Pinch_shex": pinch_shex,
            "Pinch_des": pinch_des,
            "Pinch_cond": pinch_cond,
            "Pinch_evap": pinch_evap,
            "Pinch_abs": pinch_abs,
        },
        compositions={
            "x1_LiBr_mol": x1,
            "x4_LiBr_mol": x4,
            "w1_LiBr": w1,
            "w4_LiBr": w4,
        },
        flash_outputs=flash_outputs,
        residuals_raw=residuals_raw,
        residuals_scaled=residuals_scaled,
        diagnostics=diagnostics,
        checks=checks,
        validity_messages=validity_messages,
    )


def evaluate_model(z: np.ndarray, inputs: ACInputs) -> ModelEvaluation:
    """Computes all states, component quantities, and residuals for a variable vector.

    This public variant is the strict final evaluation and raises
    ModelEvaluationError for unphysical states.
    """
    return _evaluate_model_common(z, inputs, strict=True)


# ---------------------------------------------------------------------------
# Solver interface
# ---------------------------------------------------------------------------

def residual_vector(z: np.ndarray, inputs: ACInputs) -> np.ndarray:
    """Residual vector for least_squares.

    Fast path: strict evaluate_model() evaluation.
    Fallback: the same model core in the robust solver variant (strict=False).

    This way the solver uses the same equation structure as the final
    evaluation; only the robustifications needed for the solver path
    still differ.
    """
    try:
        model = evaluate_model(z, inputs)
        return _scaled_residual_array(model)
    except ModelEvaluationError:
        try:
            model = _evaluate_model_common(z, inputs, strict=False)
            return _scaled_residual_array(model)
        except _SoftResidualVector as exc:
            return exc.residuals_scaled
        except Exception:
            return _penalty_vector(len(RESIDUAL_NAMES), inputs.penalty_level)
    except Exception:
        return _penalty_vector(len(RESIDUAL_NAMES), inputs.penalty_level)


def try_evaluate_model(
    z: np.ndarray, inputs: ACInputs
) -> tuple[ModelEvaluation | None, str | None]:
    try:
        model = evaluate_model(z, inputs)
        return model, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def solve_ac(inputs: ACInputs, x0: np.ndarray | None = None) -> ACResult:
    if x0 is None:
        x0 = initial_guess(inputs)

    lower, upper = bounds(inputs)
    lsq = least_squares(
        fun=lambda zz: residual_vector(zz, inputs),
        x0=np.asarray(x0, dtype=float),
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=inputs.solver_tol,
        xtol=inputs.solver_tol,
        gtol=inputs.solver_tol,
        max_nfev=inputs.max_nfev,
        verbose=0,
    )

    scaled_residual_norm = float(np.linalg.norm(lsq.fun, ord=2))
    z_final = np.asarray(lsq.x, dtype=float)
    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, map(float, z_final)))

    model, final_evaluation_error = try_evaluate_model(z_final, inputs)
    final_point_evaluable = model is not None

    if final_point_evaluable:
        raw_residual_norm = float(
            np.linalg.norm(
                np.array([model.residuals_raw[name] for name in RESIDUAL_NAMES], dtype=float),
                ord=2,
            )
        )
    else:
        raw_residual_norm = None

    solve_info = SolveInfo(
        success=bool(lsq.success),
        status=int(lsq.status),
        message=str(lsq.message),
        cost=float(lsq.cost),
        scaled_residual_norm=scaled_residual_norm,
        raw_residual_norm=raw_residual_norm,
        nfev=int(lsq.nfev),
        final_point_evaluable=final_point_evaluable,
        final_evaluation_error=final_evaluation_error,
    )

    if model is None:
        return ACResult(
            inputs=inputs,
            solve_info=solve_info,
            primary_variables=primary_variables,
            states={},
            heat_flows_kW={},
            kpis={},
            pump_work_W={},
            lmtd_K={},
            compositions={},
            flash_outputs={},
            residuals_raw={},
            residuals_scaled={},
            diagnostics={},
            checks={},
            validity_messages=[],
        )

    return ACResult(
        inputs=inputs,
        solve_info=solve_info,
        primary_variables=model.primary_variables,
        states=model.states,
        heat_flows_kW=model.heat_flows_kW,
        kpis=model.kpis,
        pump_work_W=model.pump_work_W,
        lmtd_K=model.lmtd_K,
        compositions=model.compositions,
        flash_outputs=model.flash_outputs,
        residuals_raw=model.residuals_raw,
        residuals_scaled=model.residuals_scaled,
        diagnostics=model.diagnostics,
        checks=model.checks,
        validity_messages=model.validity_messages,
    )


# ---------------------------------------------------------------------------
# Debugging aid: trace for initial-guess analysis
# ---------------------------------------------------------------------------

def trace_model(z: np.ndarray, inputs: ACInputs) -> ModelTrace:
    """Evaluates the model step by step and returns all intermediate results.
    Useful for diagnosing initial-guess problems."""
    T8, T10, x4, x1, T3, T5 = map(float, z)

    values: Dict[str, float] = {}
    primary_variables = dict(
        zip(PRIMARY_VARIABLE_NAMES, [T8, T10, x4, x1, T3, T5])
    )
    stage = "initial"

    try:
        stage = "pressure_levels"
        p_low = water_p_sat_from_T(T10, Q=1.0)
        p_high = water_p_sat_from_T(T8, Q=0.0)
        values["p_low_Pa"] = p_low
        values["p_high_Pa"] = p_high

        if p_high <= p_low:
            raise ModelEvaluationError(
                f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa)."
            )

        stage = "solution_saturation_states"
        T4 = lp.T_sat_solution_from_p_x(p_high, x4)
        T1 = lp.T_sat_solution_from_p_x(p_low, x1)
        values["T4_K"] = T4
        values["T1_K"] = T1

        w4 = lp.w_libr_from_x(x4)
        w1 = lp.w_libr_from_x(x1)
        values["w4_LiBr"] = w4
        values["w1_LiBr"] = w1

        if not (w4 > w1 > 0.0):
            raise ModelEvaluationError(
                f"Concentration hierarchy violated: w4={w4:.6f}, w1={w1:.6f}."
            )

        stage = "cycle_scale"
        h3 = lp.h_solution_mass_kjkg(T3, x1)
        h4 = lp.h_solution_mass_kjkg(T4, x4)
        T7 = lp.T_sat_solution_from_p_x(p_high, x1) + inputs.desorber_vapor_superheat_K
        h7 = water_h_kjkg_PT(p_high, T7)
        h8 = water_h_kjkg_PQ(p_high, Q=0.0)
        h10 = water_h_kjkg_PQ(p_low, Q=1.0)
        h9 = h8
        refrigerant_throttle = water_throttle_state(p_low, h9)
        T9 = refrigerant_throttle["T9_K"]
        m1 = _resolve_cycle_scale(inputs, w4=w4, w1=w1, h9=h9, h10=h10, strict=True)
        values["h3_kJ_kg"] = h3
        values["h4_kJ_kg"] = h4
        values["h7_kJ_kg"] = h7
        values["h8_kJ_kg"] = h8
        values["h10_kJ_kg"] = h10
        values["m1_kg_s"] = m1
        values["T7_K"] = T7


        stage = "mass_flows"
        m3 = m1
        m4 = m3 * w1 / w4
        m7 = m1 - m4
        m10 = m7
        values["m3_kg_s"] = m3
        values["m7_kg_s"] = m7

        if m7 <= 0.0:
            raise ModelEvaluationError(f"m7={m7:.6f} kg/s not positive.")

        stage = "solution_pump"
        h5 = lp.h_solution_mass_kjkg(T5, x4)
        h1 = lp.h_solution_mass_kjkg(T1, x1)
        rho1 = lp.rho_solution_mass(T1, x1)
        v1 = 1.0 / rho1
        W_sol_pump = m1 * v1 * (p_high - p_low) / 1000.0  # kW
        h2 = h1 + W_sol_pump / m1
        T2 = lp.T_from_h_x_mass(h2, x1)
        values.update({
            "h1_kJ_kg": h1,
            "h2_kJ_kg": h2,
            "W_sol_pump_kW": W_sol_pump,
            "h5_kJ_kg": h5,
            "T2_K": T2,
        })

        stage = "shex"
        Q_shex_hot = m4 * (h4 - h5)
        Q_shex_cold = m3 * (h3 - h2)
        values["Q_shex_hot_kW"] = Q_shex_hot
        values["Q_shex_cold_kW"] = Q_shex_cold
        values["deltaT_shex_1_K"] = T4 - T3
        values["deltaT_shex_2_K"] = T5 - T2

        if Q_shex_hot <= 0.0:
            raise ModelEvaluationError(f"Q_shex_hot={Q_shex_hot:.4f} kW not positive.")
        if Q_shex_cold <= 0.0:
            raise ModelEvaluationError(f"Q_shex_cold={Q_shex_cold:.4f} kW not positive.")

        lmtd_shex = counterflow_lmtd(hot_in=T4, hot_out=T5, cold_in=T2, cold_out=T3)
        values["LMTD_shex_K"] = lmtd_shex

        stage = "throttle_solution"
        h6 = h5
        flash = lp.flash_valve_state_5_to_6(
            p_out_pa=p_low,
            h5_kJkg=h5,
            m5_kg_s=m4,
            x5_libr_mol=x4,
        )
        T6 = flash["T6_K"]
        values["h6_kJ_kg"] = h6
        values["T6_K"] = T6
        for key, value in flash.items():
            values[f"flash_{key}"] = float(value)

        stage = "hot_side_coupling"
        Q_abs = m10 * h10 + m4 * h6 - m1 * h1
        values["Q_abs_kW"] = Q_abs
        if Q_abs <= 0.0:
            raise ModelEvaluationError(f"Q_abs={Q_abs:.4f} kW not positive.")

        stage = "evaporator"
        Q_evap = m7 * (h10 - h9)
        if Q_evap <= 0.0:
            raise ModelEvaluationError(f"Q_evap={Q_evap:.4f} kW not positive.")
        m17, T18 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=True)
        lmtd_evap = _counterflow_lmtd_mode(
            strict=True, hot_in=inputs.T_17, hot_out=T18, cold_in=T9, cold_out=T10
        )

        values["Q_evap_kW"] = Q_evap
        values["m17_kg_s"] = m17
        values["T18_K"] = T18
        values["deltaT_evap_1_K"] = T18 - T9
        values["deltaT_evap_2_K"] = inputs.T_17 - T10
        values["LMTD_evap_K"] = lmtd_evap

        stage = "condenser"
        if inputs.uses_serial_condenser_to_absorber_routing:
            T15_in = _resolve_condenser_external_inlet_temperature(inputs)
        else:
            T13_in = _resolve_absorber_external_inlet_temperature(inputs)
            T14_seed = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
            T15_in = _resolve_condenser_external_inlet_temperature(inputs, T14_seed)

        Q_cond = m7 * (h7 - h8)
        T16 = T15_in + Q_cond / (inputs.m_15 * inputs.cp_w_kJkgK)
        values["Q_cond_kW"] = Q_cond
        values["T16_K"] = T16
        values["T15_K"] = T15_in
        values["deltaT_cond_1_K"] =T8 - T16
        values["deltaT_cond_2_K"] = T8 - T15_in

        if Q_cond <= 0.0:
            raise ModelEvaluationError(f"Q_cond={Q_cond:.4f} kW not positive.")

        lmtd_cond = counterflow_lmtd(hot_in=T8, hot_out=T8, cold_in=T15_in, cold_out=T16)
        values["LMTD_cond_K"] = lmtd_cond

        stage = "throttle_refrigerant"
        values["h9_kJ_kg"] = h9
        values["T9_K"] = T9

        stage = "absorber"
        if inputs.uses_serial_condenser_to_absorber_routing:
            T13_in = _resolve_absorber_external_inlet_temperature(inputs, T16)
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        else:
            T13_in = _resolve_absorber_external_inlet_temperature(inputs)
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        values["T13_K"] = T13_in
        values["T14_K"] = T14
        values["deltaT_abs_1_K"] = T6 - T14
        values["deltaT_abs_2_K"] = T1 - T13_in

        lmtd_abs = counterflow_lmtd(
            hot_in=T6, hot_out=T1, cold_in=T13_in, cold_out=T14
        )
        values["LMTD_abs_K"] = lmtd_abs

        stage = "desorber"
        Q_des = m4 * h4 + m7 * h7 - m1 * h3
        if Q_des <= 0.0:
            raise ModelEvaluationError(f"Q_des={Q_des:.4f} kW not positive.")
        values["Q_des_kW"] = Q_des
        T12 = inputs.T_11 - Q_des / (inputs.m_11 * inputs.cp_w_kJkgK)

        values["T12_K"] = T12
        values["deltaT_des_1_K"] = inputs.T_11 - T4
        values["deltaT_des_2_K"] = T12 - T7

        lmtd_des = counterflow_lmtd(
            hot_in=inputs.T_11, hot_out=T12, cold_in=T7, cold_out=T4
        )
        values["LMTD_des_K"] = lmtd_des

        return ModelTrace(
            primary_variables=primary_variables,
            values=values,
            stage=stage,
            success=True,
            error_type=None,
            error_message=None,
        )

    except Exception as exc:
        return ModelTrace(
            primary_variables=primary_variables,
            values=values,
            stage=stage,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _is_absolute_temperature_key(key: str) -> bool:
    return key.startswith("T")


def _display_key(key: str) -> str:
    if key.endswith("_K") and _is_absolute_temperature_key(key):
        return f"{key[:-2]}_C"
    return key


def _display_value_and_unit(key: str, value: float) -> tuple[float, str]:
    if _is_absolute_temperature_key(key):
        return kelvin_to_celsius(value), "°C"
    if key.endswith("_Pa"):
        return value, "Pa"
    if key.endswith("_kg_s"):
        return value, "kg/s"
    if key.endswith("_kJ_kg"):
        return value, "kJ/kg"
    if key.endswith("_kW"):
        return value, "kW"
    if key.endswith("_K"):
        return value, "K"
    return value, "-"


def _format_state_line(state_id: str, state: Dict[str, float]) -> str:
    def maybe_temperature() -> str:
        if "T_K" not in state:
            return "-"
        return f"{kelvin_to_celsius(state['T_K']):.3f} °C"

    def maybe(key: str, fmt: str) -> str:
        return fmt.format(state[key]) if key in state else "-"

    return (
        f"{state_id:>3s} | "
        f"T = {maybe_temperature()} | "
        f"p = {maybe('p_Pa', '{:.3e}')} Pa | "
        f"m = {maybe('m_kg_s', '{:.6f}')} kg/s | "
        f"h = {maybe('h_kJ_kg', '{:.6f}')} kJ/kg | "
        f"x = {maybe('x_LiBr_mol', '{:.6f}')} | "
        f"w = {maybe('w_LiBr', '{:.6f}')}"
    )


def print_trace(trace: ModelTrace) -> None:
    print("=" * 110)
    print("Initial-guess trace / model-point trace")
    print("=" * 110)

    print("Primary variables")
    for key, value in trace.primary_variables.items():
        if _is_absolute_temperature_key(key):
            display_value, unit = kelvin_to_celsius(value), "°C"
        else:
            display_value, unit = value, "-"
        print(f"  {key:12s}: {display_value:14.6f} {unit}")
    print()

    print(f"Evaluation status : {trace.success}")
    print(f"Last stage        : {trace.stage}")
    if trace.error_type is not None:
        print(f"Error type        : {trace.error_type}")
    if trace.error_message is not None:
        print(f"Error message     : {trace.error_message}")
    print()

    print("Computed quantities up to the point of failure")
    for key, value in trace.values.items():
        display_key = _display_key(key)
        display_value, unit = _display_value_and_unit(key, value)
        print(f"  {display_key:28s}: {display_value:14.6f} {unit}")
    print("=" * 110)


def print_summary(result: ACResult) -> None:
    print("=" * 110)
    print("AC simulation - results overview (6 primary unknowns)")
    print("=" * 110)

    print("Solver information")
    print(f"  Success                : {result.solve_info.success}")
    print(f"  Status                 : {result.solve_info.status}")
    print(f"  Message                : {result.solve_info.message}")
    print(f"  Function evaluations   : {result.solve_info.nfev}")
    print(f"  least_squares cost     : {result.solve_info.cost:.6e}")
    print(f"  Scaled residual norm   : {result.solve_info.scaled_residual_norm:.6e}")
    if result.solve_info.raw_residual_norm is None:
        print("  Raw residual norm      : n/a (final point not physically evaluable)")
    else:
        print(f"  Raw residual norm      : {result.solve_info.raw_residual_norm:.6e}")
    print(f"  Final point physically evaluable: {result.solve_info.final_point_evaluable}")
    if result.solve_info.final_evaluation_error is not None:
        print(f"  Final-point evaluation error   : {result.solve_info.final_evaluation_error}")
    print()

    if not result.solve_info.final_point_evaluable:
        print("No physically evaluable model solution available.")
        print("=" * 110)
        return

    print("Primary solver variables")
    for name in PRIMARY_VARIABLE_NAMES:
        if _is_absolute_temperature_key(name):
            display_value, unit = kelvin_to_celsius(result.primary_variables[name]), "°C"
        else:
            display_value, unit = result.primary_variables[name], "-"
        print(f"  {name:8s}: {display_value:12.6f} {unit}")
    print()

    print("Heat flows [kW]")
    for key, value in result.heat_flows_kW.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("KPIs")
    for key, value in result.kpis.items():
        unit = "K" if key.endswith("_K") else "-"
        print(f"  {key:12s}: {value:12.6f} {unit}")
    print()

    print("Pump work [kW]")
    for key, value in result.pump_work_W.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("LMTD [K]")
    for key, value in result.lmtd_K.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("Flash throttle 5 -> 6 (local sub-calculation; T6 is used in the desorber)")
    for key, value in result.flash_outputs.items():
        display_key = _display_key(key)
        display_value, unit = _display_value_and_unit(key, value)
        print(f"  {display_key:28s}: {display_value:14.6f} {unit}")
    print()

    print("Residuals")
    for name in RESIDUAL_NAMES:
        raw = result.residuals_raw[name]
        scaled = result.residuals_scaled[name]
        print(f"  {name:18s}: raw = {raw:14.6e} | scaled = {scaled:14.6e}")
    print()

    print("Diagnostics")
    for key, value in result.diagnostics.items():
        display_key = _display_key(key)
        display_value, unit = _display_value_and_unit(key, value)
        print(f"  {display_key:28s}: {display_value:14.6f} {unit}")
    print()

    print("Plausibility checks")
    for key, value in result.checks.items():
        print(f"  {key:35s}: {value}")
    print()

    print("States")
    for state_id in sorted(result.states, key=lambda s: (len(s), s)):
        print(_format_state_line(state_id, result.states[state_id]))
    print()

    print("Validity messages")
    for msg in result.validity_messages:
        print(f"  - {msg}")

    print("=" * 110)


__all__ = [
    "ACInputs",
    "ACResult",
    "SolveInfo",
    "ModelTrace",
    "PRIMARY_VARIABLE_NAMES",
    "RESIDUAL_NAMES",
    "PRIMARY_TEMPERATURE_INDICES",
    "kelvin_to_celsius",
    "celsius_to_kelvin",
    "primary_temperatures_C_to_K",
    "primary_temperatures_K_to_C",
    "bounds",
    "initial_guess",
    "evaluate_model",
    "trace_model",
    "residual_vector",
    "solve_ac",
    "print_trace",
    "print_summary",
]
