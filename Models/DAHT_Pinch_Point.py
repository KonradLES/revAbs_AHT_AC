"""Double-lift absorption heat transformer (DAHT) model.

Topology confirmed against the actual experimental prototype in Lubis
et al., "Experimental performance of a double-lift absorption heat
transformer for manufacturing-process steam generation", Energy
Conversion and Management 148 (2017) 267-278 (Figs. 3 and 5, Table 4):

Topology re-verified directly against Lubis et al.'s Fig. 3 (conceptual
block diagram) and Fig. 5 (instrumented system flow diagram + Duhring
chart), which the user supplied as images -- this superseded an earlier
reading based on text and the effectiveness formulas alone, which had
misplaced the two solution economizers (see below).

- ONE desorber/generator (G, p_low) and ONE condenser (C, p_low),
  exactly as in the single-stage model.
- ONE refrigerant separator EH, grouped with AH in one physical unit at
  p_high; ONE "absorbing evaporator" AL, grouped with EL in one physical
  unit at p_mid (Fig. 3's two dashed boxes; the Duhring chart's T_EH and
  T_EL horizontal bands confirm these are genuinely different pressure
  levels, not just different concentrations at one pressure).
- ONE solution loop with a SINGLE pump (p_low -> p_high) and TWO
  economizers, positioned in SERIES but at DIFFERENT points along the
  loop -- confirmed by Fig. 5's layout, where SH is drawn directly
  between the EH+AH block and the EL+AL block (recovering heat from AH's
  own outlet before AL), and SL is drawn between the EL+AL block and the
  C+G block (recovering heat from AL's own outlet before the final
  throttle back to the desorber):
      desorber -> pump -> SL (cold) -> T4 -> SH (cold) -> T5
      -> AH (p_high, EXTERNAL, the useful high-temperature output)
      -> SH (hot: cools AH's own outlet) -> T4b
      -> throttle (p_high -> p_mid, no significant flash assumed)
      -> AL (p_mid, INTERNAL, drives EH)
      -> SL (hot: cools AL's own outlet) -> T2
      -> throttle (p_mid -> p_low, with flash) -> desorber
  An earlier version of this module put BOTH economizers before AH (both
  fed by AL's outlet in series) and let AL's inlet come straight off
  AH's outlet with no economizer in between at all -- removing exactly
  the independently-adjustable intermediate temperature the real design
  has at that point, which is consistent with the solver's persistent
  collapse (T10_EL/T10_EH and the concentrations all being pulled
  together) in that version.
  The concentrated solution meets AH FIRST -- this is what lets AH
  reach the highest temperature ("delivers the concentrated solution
  from the generator to the high-temperature absorber ... to maximize
  the temperature of the output heat at AH", Lubis et al. Sec. 2.2).
  Table 4 confirms this ordering: "middle solution concentration (at
  the outlet of the AH)" is between "strong" (generator/AH inlet) and
  "weak" (AL outlet).
- The condensed refrigerant splits into TWO separately pumped streams
  (Fig. 5 shows a single refrigerant pump RP feeding the whole EH/AH
  assembly, with EL fed via a throttle/needle valve off the same
  pumped stream rather than a second independent pump -- a minor
  plumbing simplification this module still makes, of little
  thermodynamic consequence for a liquid pump/throttle):
    * branch A: p_low -> p_mid -> EL (evaporator), driven by external
      heat. Its vapor is absorbed in AL.
    * branch B: p_low -> p_high -> EH (evaporator): AL's rejected heat,
      instead of leaving the system, evaporates this stream. Its vapor
      is absorbed in AH, releasing the final useful output.
- External heat is routed in SERIES through the generator and then EL
  ("The hot water passes through the generator (G), and subsequently
  the first evaporator (EL)", Lubis et al. Sec. 2.2) -- EL's external
  inlet temperature equals the generator's external outlet T14, not an
  independent input.

This module is a monolithic script in the same style as
Models/AHT_Pinch_Point.py, importing only the generic, validated
helpers from there (property wrappers, LMTD, external stream
resolution, the throttle/flash routine).

Simplifications (documented, candidates for a later iteration):
- No adiabatic pre-absorption refinement in either absorber.
- The AH -> AL throttle (p_high -> p_mid, after SH's hot side) is
  treated as non-flashing (solution enthalpy/temperature approximately
  pressure-independent for a liquid with no phase change); only the
  larger AL -> desorber throttle (p_mid -> p_low) uses the full flash
  routine, exactly like the single-stage reference model's one throttle.
- TWO independent refrigerant pumps (branch A to p_mid, branch B to
  p_high), each fully evaporating in its own evaporator -- not Fig. 5's
  single pump RP feeding a shared coil in AL that only partially
  evaporates, with the leftover liquid cascading down to EL and then to
  the condenser (see the note above). This was a deliberate choice, not
  an oversight: replicating that cascade would mean modelling EH as a
  genuine flash separator with an EH->EL->condenser liquid return path,
  which is a materially bigger restructuring than "two more heat
  exchangers" -- and this module's convergence was hard-won already (see
  git history / the conversation that produced it). Decided against it
  for now; the two-branch split is mass- and energy-balance equivalent,
  it just can't host RH/RL.
- No RH/RL refrigerant-side economizers (Fig. 5), for the same reason:
  they sit on the cascading liquid-return path above, which this
  module's two-branch refrigerant split doesn't have -- there is no
  "returning" refrigerant stream in this topology for them to recover
  heat from without inventing one. Expected impact is secondary and
  one-sided (this module's COP is, if anything, a slight
  underestimate): RH/RL only recover sensible heat over a few K on
  liquid refrigerant, versus SL/SH's much larger recovery on the
  solution side, which IS modelled.
- cycle_scale_spec_mode supports "m_gen", "QAH", and "Qwaste" (= Q_des +
  Q_EL, the total waste-heat input -- typically the most natural way to
  size a design point) -- no probe-based "Qdes_eva" analogue beyond that.

Internal units: temperature K, pressure Pa, mass flow kg/s, specific
enthalpy kJ/kg, heat flow kW -- identical convention to
Models/AHT_Pinch_Point.py.

Primary solver variables
-------------------------
z = [T8, T10_EL, T10_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5]

T8     condenser / p_low saturation temperature
T10_EL evaporator EL / p_mid saturation vapor temperature
T10_EH evaporator EH / p_high saturation vapor temperature
x_gen  desorber outlet (strong) solution mole fraction -- SL's cold inlet
x_mid  AH outlet (middle) / AL inlet solution mole fraction
x_dil  AL outlet (weak) solution mole fraction -- desorber's inlet
T2     SL hot-side outlet (before the throttle back to the desorber)
T4     SL cold-side outlet / SH cold-side inlet
T4b    SH hot-side outlet -- AL's actual inlet (after the p_high->p_mid throttle)
T5     SH cold-side outlet -- AH's actual inlet
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, least_squares, minimize, root_scalar

import Thermodynamic_Properties.libr_props as lp
from Models.AHT_Pinch_Point import (
    ModelEvaluationError,
    _resolve_absorber_external_stream,
    _resolve_condenser_external_stream,
    _resolve_desorber_external_stream,
    _resolve_evaporator_external_stream,
    celsius_to_kelvin,
    kelvin_to_celsius,
    counterflow_lmtd,
    counterflow_lmtd_soft,
    smooth_min,
    water_h_kjkg_PQ,
    water_h_kjkg_PT,
    water_p_sat_from_T,
    water_rho_kgm3_PQ,
    water_T_K_PH,
)

PRIMARY_VARIABLE_NAMES = ["T8", "T10_EL", "T10_EH", "x_gen", "x_mid", "x_dil", "T2", "T4", "T4b", "T5"]
RESIDUAL_NAMES = [
    "R1_SL_energy",
    "R2_SL_pinch",
    "R3_desorber_pinch",
    "R4_condenser_pinch",
    "R5_EL_pinch",
    "R6_AH_pinch",
    "R7_AL_EH_pinch",
    "R8_AL_EH_energy",
    "R9_SH_energy",
    "R10_SH_pinch",
]
_TEMPERATURE_INDICES = (0, 1, 2, 6, 7, 8, 9)  # T8, T10_EL, T10_EH, T2, T4, T4b, T5


def primary_temperatures_C_to_K(z_user) -> np.ndarray:
    z = np.asarray(z_user, dtype=float).copy()
    z[list(_TEMPERATURE_INDICES)] += 273.15
    return z


def primary_temperatures_K_to_C(z_internal) -> np.ndarray:
    z = np.asarray(z_internal, dtype=float).copy()
    z[list(_TEMPERATURE_INDICES)] -= 273.15
    return z


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAHTInputs:
    # External inlet temperatures [degC]
    T_11_C: float  # AH heat sink inlet (useful, high-T output)
    T_13_C: float  # Desorber heat source inlet (routed in series into EL)
    T_17_C: float  # Condenser heat sink inlet

    # Pinch (minimum approach temperature) values [K]
    dT_min_SL: float  # low-temp solution economizer: pump_out->T4 / AL_out->T2
    dT_min_des: float
    dT_min_cond: float
    dT_min_EL: float
    dT_min_AL_EH: float
    dT_min_AH: float
    dT_min_SH: float  # high-temp solution economizer: T4->T5 / AH_out->T4b

    _: KW_ONLY

    cycle_scale_spec_mode: str = "m_gen"  # "m_gen", "QAH", or "Qwaste"
    m_gen_spec: float | None = None
    QAH_spec_kW: float | None = None
    Qwaste_spec_kW: float | None = None  # target total waste-heat input, Q_des + Q_EL

    # Attribute names below (absorber_spec_mode, evaporator_spec_mode, ...)
    # match Models.AHT_Pinch_Point's reused _resolve_*_external_stream
    # helpers exactly -- they read these names off `inputs` directly.
    absorber_spec_mode: str = "m11"  # "m11" or "T12"; drives AH (the useful output)
    m11_spec: float | None = None
    T12_spec_C: float | None = None

    desorber_spec_mode: str = "m13"  # "m13" or "T14"
    m13_spec: float | None = None
    T14_spec_C: float | None = None

    evaporator_spec_mode: str = "m15"  # "m15" or "T16"; drives EL
    m15_spec: float | None = None
    T16_spec_C: float | None = None

    condenser_spec_mode: str = "m17"  # "m17" or "T18"
    m17_spec: float | None = None
    T18_spec_C: float | None = None

    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    solver_tol: float = 1.0e-9
    max_nfev: int = 8000
    penalty_level: float = 1.0e6

    def __post_init__(self) -> None:
        for name, val in [
            ("dT_min_SL", self.dT_min_SL),
            ("dT_min_des", self.dT_min_des),
            ("dT_min_cond", self.dT_min_cond),
            ("dT_min_EL", self.dT_min_EL),
            ("dT_min_AL_EH", self.dT_min_AL_EH),
            ("dT_min_AH", self.dT_min_AH),
            ("dT_min_SH", self.dT_min_SH),
        ]:
            if val <= 0.0:
                raise ValueError(f"{name} must be positive, but is {val}.")

        if self.cycle_scale_spec_mode not in {"m_gen", "QAH", "Qwaste"}:
            raise ValueError("cycle_scale_spec_mode must be 'm_gen', 'QAH', or 'Qwaste'.")
        _scale_specs = {
            "m_gen": self.m_gen_spec,
            "QAH": self.QAH_spec_kW,
            "Qwaste": self.Qwaste_spec_kW,
        }
        for name, val in _scale_specs.items():
            if name == self.cycle_scale_spec_mode:
                if val is None or val <= 0.0:
                    raise ValueError(f"For cycle_scale_spec_mode='{name}', {name}_spec (or _kW) > 0 must be given.")
            elif val is not None:
                raise ValueError(f"For cycle_scale_spec_mode='{self.cycle_scale_spec_mode}', {name}_spec (or _kW) must not be set.")

        self._check_spec_pair("absorber_spec_mode", self.absorber_spec_mode, "m11", self.m11_spec, "T12", self.T12_spec_C)
        self._check_spec_pair("desorber_spec_mode", self.desorber_spec_mode, "m13", self.m13_spec, "T14", self.T14_spec_C)
        self._check_spec_pair("evaporator_spec_mode", self.evaporator_spec_mode, "m15", self.m15_spec, "T16", self.T16_spec_C)
        self._check_spec_pair("condenser_spec_mode", self.condenser_spec_mode, "m17", self.m17_spec, "T18", self.T18_spec_C)

    @staticmethod
    def _check_spec_pair(field_name, mode, m_name, m_val, T_name, T_val) -> None:
        if mode not in {m_name, T_name}:
            raise ValueError(f"{field_name} must be '{m_name}' or '{T_name}'.")
        if mode == m_name:
            if m_val is None or m_val <= 0.0:
                raise ValueError(f"For {field_name}='{m_name}', {m_name}_spec > 0 must be given.")
            if T_val is not None:
                raise ValueError(f"For {field_name}='{m_name}', {T_name}_spec_C must not be set.")
        else:
            if T_val is None:
                raise ValueError(f"For {field_name}='{T_name}', {T_name}_spec_C must be given.")
            if m_val is not None:
                raise ValueError(f"For {field_name}='{T_name}', {m_name}_spec must not be set.")

    @property
    def T_11(self) -> float:
        return celsius_to_kelvin(self.T_11_C)

    @property
    def T_13(self) -> float:
        return celsius_to_kelvin(self.T_13_C)

    @property
    def T_17(self) -> float:
        return celsius_to_kelvin(self.T_17_C)

    @property
    def T12_spec(self) -> float:
        return celsius_to_kelvin(self.T12_spec_C)

    @property
    def T14_spec(self) -> float:
        return celsius_to_kelvin(self.T14_spec_C)

    @property
    def T16_spec(self) -> float:
        return celsius_to_kelvin(self.T16_spec_C)

    @property
    def T18_spec(self) -> float:
        return celsius_to_kelvin(self.T18_spec_C)


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
class DAHTEvaluation:
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    mass_flows_kg_s: Dict[str, float]
    pinch_temperatures_K: Dict[str, float]
    compositions: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    checks: Dict[str, bool]


@dataclass(frozen=True)
class DAHTResult:
    inputs: DAHTInputs
    solve_info: SolveInfo
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    mass_flows_kg_s: Dict[str, float]
    pinch_temperatures_K: Dict[str, float]
    compositions: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    checks: Dict[str, bool]


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------


class _SoftResidualVector(RuntimeError):
    def __init__(self, residuals_scaled: np.ndarray):
        super().__init__("Soft residual vector ready.")
        self.residuals_scaled = np.asarray(residuals_scaled, dtype=float)


def _residual_scales(m_gen: float) -> np.ndarray:
    return np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 1.0], dtype=float)


def _counterflow_lmtd_mode(*, strict: bool, hot_in, hot_out, cold_in, cold_out, HX_name) -> float:
    if strict:
        return counterflow_lmtd(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out, HX_name=HX_name)
    return counterflow_lmtd_soft(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out)


def initial_guess(inputs: DAHTInputs) -> np.ndarray:
    """Derives a starting point analytically from inputs' own pinch/spec
    targets, rather than from fixed Celsius offsets.

    An earlier version used offsets calibrated against Lubis et al. 2017
    Table 4 (real prototype, T13~88 degC), e.g. T5 = T13+80. Those offsets
    silently assumed operating levels close to Lubis's own absolute
    temperatures. For a very different window (e.g. T13=60, T11=85 degC),
    they landed 50-80 K away from the physically consistent point -- still
    numerically a valid starting vector, but far outside the solver's
    practical basin of convergence given how expensive each residual
    evaluation is (nested Brent root-finds), so solve times went from tens
    of seconds to unusable (10+ minutes, sometimes not converging within
    max_nfev at all).

    This version instead inverts the actual physical constraints: x_gen
    from the desorber's own pinch-limited boiling point, p_high (and hence
    T10_EH) from the concentration and temperature actually required at AH,
    T10_EL from EL's pinch targets -- so it tracks whatever operating
    window `inputs` describes instead of only the one it was validated on.
    """
    T17 = inputs.T_17_C
    T13 = inputs.T_13_C
    T11 = inputs.T_11_C

    # --- condenser / p_low ---------------------------------------------
    T18_target = inputs.T18_spec_C if inputs.condenser_spec_mode == "T18" else T17 + 8.0
    T8_C = T18_target + inputs.dT_min_cond + 2.0
    T8 = celsius_to_kelvin(T8_C)
    p_low = water_p_sat_from_T(T8, Q=0.0)

    # --- desorber -> x_gen -----------------------------------------------
    T14_target = inputs.T14_spec_C if inputs.desorber_spec_mode == "T14" else T13 - 5.0
    T_gen_out_C = min(T14_target, T13) - inputs.dT_min_des - 1.0
    T_gen_out = celsius_to_kelvin(T_gen_out_C)
    x_gen = root_scalar(
        lambda x: lp.T_sat_solution_from_p_x(p_low, x) - T_gen_out, bracket=[0.02, 0.38], method="brentq"
    ).root
    w_gen = lp.w_libr_from_x(x_gen)

    # --- concentration spread (Lubis-like relative spacing, wt%) ---------
    w_mid = w_gen - 0.023
    w_dil = w_mid - 0.0155
    x_mid = lp.x_from_w_libr(w_mid)
    x_dil = lp.x_from_w_libr(w_dil)

    # --- EL -> T10_EL / p_mid --------------------------------------------
    T16_target = inputs.T16_spec_C if inputs.evaporator_spec_mode == "T16" else T14_target - 8.0
    T10_EL_C = min(T14_target, T16_target) - inputs.dT_min_EL - 2.0
    T10_EL = celsius_to_kelvin(T10_EL_C)
    p_mid = water_p_sat_from_T(T10_EL, Q=1.0)

    # --- AH -> p_high / T10_EH: solve for the pressure level at which
    #     x_mid actually saturates at AH's required outlet temperature ---
    T12_target = inputs.T12_spec_C if inputs.absorber_spec_mode == "T12" else T11 + 15.0
    T_mid_out_C = max(T12_target, T11) + inputs.dT_min_AH + 2.0
    T_mid_out = celsius_to_kelvin(T_mid_out_C)
    p_high = lp.calc_p_sat_patek(T_mid_out, x_mid)
    T10_EH = root_scalar(
        lambda T: water_p_sat_from_T(T, Q=1.0) - p_high, bracket=[T8, celsius_to_kelvin(200.0)], method="brentq"
    ).root
    # For a small requested lift (T12 close to T11/T13), the AH-derived
    # p_high can come out BELOW p_mid -- physically nonsensical (AH must be
    # the highest pressure level) and fatal for the solver's ordering
    # checks. This is only a starting guess, so simply clip it away from
    # p_mid rather than solving the (harder) fully self-consistent case.
    T10_EH = max(T10_EH, T10_EL + 5.0)

    # --- remaining states --------------------------------------------------
    T_AL_out = lp.T_sat_solution_from_p_x(p_mid, x_dil)
    T4b_C = max(kelvin_to_celsius(T10_EH) + inputs.dT_min_AL_EH + 3.0, kelvin_to_celsius(T_AL_out) - 15.0)
    T4b = celsius_to_kelvin(T4b_C)
    T5 = celsius_to_kelvin(T_mid_out_C)  # AH inlet, close to (slightly below) T_mid_out target
    T2 = celsius_to_kelvin(T_gen_out_C + 5.0)
    T4 = celsius_to_kelvin(kelvin_to_celsius(T_AL_out) - inputs.dT_min_SL - 3.0)

    return np.array([T8, T10_EL, T10_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5], dtype=float)


def bounds(inputs: DAHTInputs) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            inputs.T_17 + 0.5,
            inputs.T_17 + 5.0,
            inputs.T_17 + 10.0,
            0.05,
            0.05,
            0.05,
            inputs.T_17 + 1.0,
            inputs.T_17 + 1.0,
            inputs.T_17 + 1.0,
            inputs.T_17 + 1.0,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            min(inputs.T_13 - 1.0, 420.0),
            min(inputs.T_13 - 0.5, 420.0),
            500.0,
            0.39,
            0.39,
            0.39,
            500.0,
            500.0,
            500.0,
            500.0,
        ],
        dtype=float,
    )
    return lower, upper


# ---------------------------------------------------------------------------
# Cycle scale resolution (m_gen)
# ---------------------------------------------------------------------------


def _resolve_cycle_scale(z: np.ndarray, inputs: DAHTInputs, *, strict: bool) -> float:
    if inputs.cycle_scale_spec_mode == "m_gen":
        m_gen = float(inputs.m_gen_spec)
        if strict and m_gen <= 0.0:
            raise ModelEvaluationError("Solution mass flow m_gen must be positive.")
        return m_gen

    target_kW = inputs.QAH_spec_kW if inputs.cycle_scale_spec_mode == "QAH" else inputs.Qwaste_spec_kW
    probe_inputs = replace(
        inputs, cycle_scale_spec_mode="m_gen", m_gen_spec=1.0, QAH_spec_kW=None, Qwaste_spec_kW=None
    )
    try:
        probe = _evaluate_model_common(z, probe_inputs, strict=False)
    except _SoftResidualVector:
        if strict:
            raise ModelEvaluationError(
                f"Cannot resolve cycle scaling from {inputs.cycle_scale_spec_mode} because the probe "
                "evaluation hit an invalid pressure level."
            )
        return 1.0e-9

    if inputs.cycle_scale_spec_mode == "QAH":
        q_per_unit = probe.heat_flows_kW["Q_AH"]
    else:
        q_per_unit = probe.heat_flows_kW["Q_des"] + probe.heat_flows_kW["Q_EL"]

    if strict:
        if q_per_unit <= 1.0e-12:
            raise ModelEvaluationError(
                f"Cannot resolve cycle scaling from {inputs.cycle_scale_spec_mode}: specific duty not "
                f"positive ({q_per_unit:.6f})."
            )
        m_gen = float(target_kW) / q_per_unit
        if m_gen <= 0.0:
            raise ModelEvaluationError(f"Computed m_gen not positive: {m_gen:.6f} kg/s.")
        return m_gen

    q_safe = q_per_unit if q_per_unit > 1.0e-12 else 1.0e-12
    return float(target_kW) / q_safe


# ---------------------------------------------------------------------------
# Shared model core
# ---------------------------------------------------------------------------


def _evaluate_model_common(z: np.ndarray, inputs: DAHTInputs, *, strict: bool) -> DAHTEvaluation:
    T8, T10_EL, T10_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5 = map(float, z)

    # 1) Pressure levels -----------------------------------------------------
    p_low = water_p_sat_from_T(T8, Q=0.0)
    p_mid = water_p_sat_from_T(T10_EL, Q=1.0)
    p_high = water_p_sat_from_T(T10_EH, Q=1.0)
    if not (p_high > p_mid > p_low):
        if strict:
            raise ModelEvaluationError(
                f"Pressure levels not strictly increasing: p_low={p_low:.3e}, p_mid={p_mid:.3e}, "
                f"p_high={p_high:.3e} Pa."
            )
        pen = (max(p_low - p_mid, p_mid - p_high) + 100.0) / 100.0
        raise _SoftResidualVector(np.full(len(RESIDUAL_NAMES), pen, dtype=float))

    # 2) Saturated solution states -------------------------------------------
    w_gen = lp.w_libr_from_x(x_gen)
    w_mid = lp.w_libr_from_x(x_mid)
    w_dil = lp.w_libr_from_x(x_dil)

    if strict and not (w_gen > w_mid > w_dil > 0.0):
        raise ModelEvaluationError(
            f"Concentration hierarchy violated: w_gen={w_gen:.6f}, w_mid={w_mid:.6f}, w_dil={w_dil:.6f}."
        )

    T_gen_out = lp.T_sat_solution_from_p_x(p_low, x_gen)
    T_AL_out = lp.T_sat_solution_from_p_x(p_mid, x_dil)

    h_gen_out = lp.h_solution_mass_kjkg(T_gen_out, x_gen)
    h_AL_out = lp.h_solution_mass_kjkg(T_AL_out, x_dil)

    h4 = lp.h_solution_mass_kjkg(T4, x_gen)
    h5 = lp.h_solution_mass_kjkg(T5, x_gen)
    h10_EL = water_h_kjkg_PQ(p_mid, Q=1.0)
    h10_EH = water_h_kjkg_PQ(p_high, Q=1.0)

    # 3) Cycle scaling and mass flows ----------------------------------------
    m_gen = _resolve_cycle_scale(z, inputs, strict=strict)

    w_mid_safe = w_mid if strict else max(w_mid, 1.0e-9)
    w_dil_safe = w_dil if strict else max(w_dil, 1.0e-9)
    m_mid = m_gen * w_gen / w_mid_safe
    mB = m_mid - m_gen           # EH's vapor, absorbed by AH
    m_dil = m_mid * w_mid / w_dil_safe
    mA = m_dil - m_mid           # EL's vapor, absorbed by AL
    m_vap_des = m_dil - m_gen

    if strict and mA <= 0.0:
        raise ModelEvaluationError(f"EL refrigerant flow not positive: mA={mA:.6f} kg/s.")
    if strict and mB <= 0.0:
        raise ModelEvaluationError(f"EH refrigerant flow not positive: mB={mB:.6f} kg/s.")

    mA_safe = mA if strict else (mA if abs(mA) > 1.0e-9 else 1.0e-9)
    mB_safe = mB if strict else (mB if abs(mB) > 1.0e-9 else 1.0e-9)

    # 4) Solution pump (single, p_low -> p_high) -----------------------------
    rho_gen = lp.rho_solution_mass(T_gen_out, x_gen)
    W_pump = m_gen / rho_gen * (p_high - p_low) / 1000.0
    h_pump_out = h_gen_out + W_pump / m_gen
    if strict:
        T_pump_out = lp.T_from_h_x_mass(h_pump_out, x_gen)
    else:
        try:
            T_pump_out = lp.T_from_h_x_mass(h_pump_out, x_gen)
        except Exception:
            T_pump_out = T_gen_out

    # 5) SL (low-temp economizer, between the AL/EL and C/G levels per Fig. 5):
    #    hot (T_AL_out -> T2, x_dil, m_dil) / cold (T_pump_out -> T4, x_gen, m_gen)
    h2 = lp.h_solution_mass_kjkg(T2, x_dil)
    Q_SL_hot = m_dil * (h_AL_out - h2)
    Q_SL_cold = m_gen * (h4 - h_pump_out)
    if strict and Q_SL_hot <= 0.0:
        raise ModelEvaluationError(f"Q_SL_hot not positive: {Q_SL_hot:.6f} kW.")
    if strict and Q_SL_cold <= 0.0:
        raise ModelEvaluationError(f"Q_SL_cold not positive: {Q_SL_cold:.6f} kW.")

    pinch_SL = smooth_min(
        smooth_min(T_AL_out - T4, T2 - T_pump_out, k=50.0), T4 - T_pump_out, k=50.0
    )
    lmtd_SL = _counterflow_lmtd_mode(strict=strict, hot_in=T_AL_out, hot_out=T2, cold_in=T_pump_out, cold_out=T4, HX_name="SL")

    # 6) AH (external -- the useful high-T output), fed by T5 (post-SL+SH) --
    T_mid_out = lp.T_sat_solution_from_p_x(p_high, x_mid)
    h_mid_out = lp.h_solution_mass_kjkg(T_mid_out, x_mid)

    Q_AH = m_gen * h5 + mB * h10_EH - m_mid * h_mid_out
    if strict and Q_AH <= 0.0:
        raise ModelEvaluationError(f"AH heat flow not positive: Q_AH={Q_AH:.6f} kW.")

    m11, T12 = _resolve_absorber_external_stream(inputs, Q_AH, strict=strict)
    pinch_AH = smooth_min(smooth_min(T5 - T12, T_mid_out - inputs.T_11, k=50.0), T_mid_out - T5, k=50.0)
    lmtd_AH = _counterflow_lmtd_mode(strict=strict, hot_in=T5, hot_out=T_mid_out, cold_in=inputs.T_11, cold_out=T12, HX_name="AH")

    # 7) SH (high-temp economizer, between the EH/AH and AL/EL levels per
    #    Fig. 5 -- sits directly between AH's outlet and AL's inlet, NOT
    #    before AH on both sides as an earlier version of this module had
    #    it): hot (T_mid_out -> T4b, x_mid, m_mid) / cold (T4 -> T5, x_gen, m_gen)
    h4b = lp.h_solution_mass_kjkg(T4b, x_mid)  # T4b now plays the role "T4b": SH's hot outlet
    Q_SH_hot = m_mid * (h_mid_out - h4b)
    Q_SH_cold = m_gen * (h5 - h4)
    if strict and Q_SH_hot <= 0.0:
        raise ModelEvaluationError(f"Q_SH_hot not positive: {Q_SH_hot:.6f} kW.")
    if strict and Q_SH_cold <= 0.0:
        raise ModelEvaluationError(f"Q_SH_cold not positive: {Q_SH_cold:.6f} kW.")

    pinch_SH = smooth_min(
        smooth_min(T_mid_out - T5, T4b - T4, k=50.0), T5 - T4, k=50.0
    )
    lmtd_SH = _counterflow_lmtd_mode(strict=strict, hot_in=T_mid_out, hot_out=T4b, cold_in=T4, cold_out=T5, HX_name="SH")

    # 8) AH -> AL throttle (p_high -> p_mid), no significant flash assumed;
    #    AL's actual inlet is SH's hot outlet (T4b), not T_mid_out directly.
    T_AL_in = T4b
    h_AL_in = h4b

    # 9) AL (internal -- drives EH) ------------------------------------------
    Q_AL = m_mid * h_AL_in + mA * h10_EL - m_dil * h_AL_out
    if strict and Q_AL <= 0.0:
        raise ModelEvaluationError(f"AL heat flow not positive: Q_AL={Q_AL:.6f} kW.")

    pinch_AL_EH = smooth_min(
        smooth_min(T_AL_in - T10_EH, T_AL_out - T10_EH, k=50.0), T_AL_out - T_AL_in, k=50.0
    )

    # 9) Desorber (p_low, external T13) --------------------------------------
    flash = lp.flash_valve_state_2_to_1(p_out_pa=p_low, h2_kJkg=h2, m2_kg_s=m_dil, x2_libr_mol=x_dil)
    T1 = flash["T1_K"]
    h1 = h2
    T7 = T1 + inputs.desorber_vapor_superheat_K
    h7 = water_h_kjkg_PT(p_low, T7)

    Q_des = m_gen * h_gen_out + m_vap_des * h7 - m_dil * h1
    if strict and Q_des <= 0.0:
        raise ModelEvaluationError(f"Desorber heat flow not positive: Q_des={Q_des:.6f} kW.")

    m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=strict, T13=inputs.T_13)
    pinch_des = smooth_min(inputs.T_13 - T_gen_out, T14 - T1, k=50.0)
    lmtd_des = _counterflow_lmtd_mode(strict=strict, hot_in=inputs.T_13, hot_out=T14, cold_in=T1, cold_out=T_gen_out, HX_name="Desorber")

    # 10) Condenser (p_low, external T17) ------------------------------------
    h8 = water_h_kjkg_PQ(p_low, Q=0.0)
    Q_cond = m_vap_des * (h7 - h8)
    if strict and Q_cond <= 0.0:
        raise ModelEvaluationError(f"Condenser heat flow not positive: Q_cond={Q_cond:.6f} kW.")

    m17, T18 = _resolve_condenser_external_stream(inputs, Q_cond, strict=strict)
    h_g_low = water_h_kjkg_PQ(p_low, Q=1.0)
    Q_desuperheat = max(0.0, min(m_vap_des * (h7 - h_g_low), Q_cond))
    T18_sat = T18 - Q_desuperheat / (m17 * inputs.cp_w_kJkgK)
    pinch_cond = smooth_min(smooth_min(T7 - T18, T8 - T18_sat, k=50.0), T8 - inputs.T_17, k=50.0)
    lmtd_cond = _counterflow_lmtd_mode(strict=strict, hot_in=T8, hot_out=T8, cold_in=inputs.T_17, cold_out=T18, HX_name="Condenser")

    # 11) Refrigerant pump A (p_low -> p_mid) + EL (external T15 = T14) -----
    rho8 = water_rho_kgm3_PQ(p_low, Q=0.0)
    v8 = 1.0 / rho8
    W_pumpA = mA * v8 * (p_mid - p_low) / 1000.0
    h9_EL = h8 + (W_pumpA / mA if strict else W_pumpA / mA_safe)
    T9_EL = water_T_K_PH(p_mid, h9_EL)

    Q_EL = mA * (h10_EL - h9_EL)
    if strict and Q_EL <= 0.0:
        raise ModelEvaluationError(f"EL heat flow not positive: Q_EL={Q_EL:.6f} kW.")

    T15_in = T14  # series routing: EL's external inlet is the desorber's external outlet
    m15, T16 = _resolve_evaporator_external_stream(inputs, Q_EL, strict=strict, T15=T15_in)
    h_f_mid = water_h_kjkg_PQ(p_mid, Q=0.0)
    Q_subcool = max(0.0, min(mA * (h_f_mid - h9_EL), Q_EL))
    T16_sat = T16 + Q_subcool / (m15 * inputs.cp_w_kJkgK)
    pinch_EL = smooth_min(
        smooth_min(T15_in - T10_EL, T16_sat - T10_EL, k=50.0), T16 - T9_EL, k=50.0
    )
    lmtd_EL = _counterflow_lmtd_mode(strict=strict, hot_in=T15_in, hot_out=T16, cold_in=T10_EL, cold_out=T10_EL, HX_name="EL")

    # 12) Refrigerant pump B (p_low -> p_high) + EH energy balance ----------
    W_pumpB = mB * v8 * (p_high - p_low) / 1000.0
    h9_EH = h8 + (W_pumpB / mB if strict else W_pumpB / mB_safe)
    T9_EH = water_T_K_PH(p_high, h9_EH)

    Q_EH_required = mB * (h10_EH - h9_EH)
    r_AL_EH_energy = Q_EH_required - Q_AL

    # 13) Residuals -----------------------------------------------------------
    residuals_raw_array = np.array(
        [
            Q_SL_hot - Q_SL_cold,
            pinch_SL - inputs.dT_min_SL,
            pinch_des - inputs.dT_min_des,
            pinch_cond - inputs.dT_min_cond,
            pinch_EL - inputs.dT_min_EL,
            pinch_AH - inputs.dT_min_AH,
            pinch_AL_EH - inputs.dT_min_AL_EH,
            r_AL_EH_energy,
            Q_SH_hot - Q_SH_cold,
            pinch_SH - inputs.dT_min_SH,
        ],
        dtype=float,
    )
    scales = _residual_scales(m_gen)
    residuals_scaled_array = residuals_raw_array / scales

    residuals_raw = dict(zip(RESIDUAL_NAMES, residuals_raw_array.tolist()))
    residuals_scaled = dict(zip(RESIDUAL_NAMES, residuals_scaled_array.tolist()))

    checks = {
        "pressures_increasing": p_high > p_mid > p_low,
        "concentrations_decreasing": w_gen > w_mid > w_dil,
        "mA_positive": mA > 0.0,
        "mB_positive": mB > 0.0,
        "mass_balance_consistent": abs((mA + mB) - m_vap_des) <= 1.0e-6 * max(1.0, m_vap_des),
    }

    kpis = {
        "COP": (Q_AH / (Q_des + Q_EL)) if (Q_des + Q_EL) > 1.0e-12 else float("nan"),
        "GTL_K": T12 - inputs.T_13,  # gross temperature lift vs. the waste-heat source
    }

    diagnostics = {
        "p_low_Pa": p_low,
        "p_mid_Pa": p_mid,
        "p_high_Pa": p_high,
        "T12_K": T12,
        "T14_K": T14,
        "T15_K": T15_in,
        "T16_K": T16,
        "T18_K": T18,
        "m_gen_kg_s": m_gen,
    }

    states = {
        "gen_out": {"T_K": T_gen_out, "p_Pa": p_low, "m_kg_s": m_gen, "h_kJ_kg": h_gen_out, "x_LiBr_mol": x_gen},
        "pump_out": {"T_K": T_pump_out, "p_Pa": p_high, "m_kg_s": m_gen, "h_kJ_kg": h_pump_out, "x_LiBr_mol": x_gen},
        "SL_cold_out (4)": {"T_K": T4, "p_Pa": p_high, "m_kg_s": m_gen, "h_kJ_kg": h4, "x_LiBr_mol": x_gen},
        "AH_in (5)": {"T_K": T5, "p_Pa": p_high, "m_kg_s": m_gen, "h_kJ_kg": h5, "x_LiBr_mol": x_gen},
        "AH_out (mid)": {"T_K": T_mid_out, "p_Pa": p_high, "m_kg_s": m_mid, "h_kJ_kg": h_mid_out, "x_LiBr_mol": x_mid},
        "AL_in (T4b, SH hot outlet)": {"T_K": T_AL_in, "p_Pa": p_mid, "m_kg_s": m_mid, "h_kJ_kg": h_AL_in, "x_LiBr_mol": x_mid},
        "AL_out (dil)": {"T_K": T_AL_out, "p_Pa": p_mid, "m_kg_s": m_dil, "h_kJ_kg": h_AL_out, "x_LiBr_mol": x_dil},
        "SL_hot_out (2)": {"T_K": T2, "p_Pa": p_mid, "m_kg_s": m_dil, "h_kJ_kg": h2, "x_LiBr_mol": x_dil},
        "des_in (1)": {"T_K": T1, "p_Pa": p_low, "m_kg_s": m_dil, "h_kJ_kg": h1, "x_LiBr_mol": x_dil},
        "des_vapor (7)": {"T_K": T7, "p_Pa": p_low, "m_kg_s": m_vap_des, "h_kJ_kg": h7},
        "cond_out (8)": {"T_K": T8, "p_Pa": p_low, "m_kg_s": m_vap_des, "h_kJ_kg": h8},
        "EL_in (9)": {"T_K": T9_EL, "p_Pa": p_mid, "m_kg_s": mA, "h_kJ_kg": h9_EL},
        "EL_out (10)": {"T_K": T10_EL, "p_Pa": p_mid, "m_kg_s": mA, "h_kJ_kg": h10_EL},
        "EH_in (9b)": {"T_K": T9_EH, "p_Pa": p_high, "m_kg_s": mB, "h_kJ_kg": h9_EH},
        "EH_out (10b)": {"T_K": T10_EH, "p_Pa": p_high, "m_kg_s": mB, "h_kJ_kg": h10_EH},
        "11": {"T_K": inputs.T_11, "m_kg_s": m11},
        "12": {"T_K": T12, "m_kg_s": m11},
        "13": {"T_K": inputs.T_13, "m_kg_s": m13},
        "14": {"T_K": T14, "m_kg_s": m13},
        "15": {"T_K": T15_in, "m_kg_s": m15},
        "16": {"T_K": T16, "m_kg_s": m15},
        "17": {"T_K": inputs.T_17, "m_kg_s": m17},
        "18": {"T_K": T18, "m_kg_s": m17},
    }

    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, [T8, T10_EL, T10_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5]))

    return DAHTEvaluation(
        primary_variables=primary_variables,
        states=states,
        heat_flows_kW={
            "Q_SL": Q_SL_hot,
            "Q_SH": Q_SH_hot,
            "Q_des": Q_des,
            "Q_cond": Q_cond,
            "Q_EL": Q_EL,
            "Q_AL": Q_AL,
            "Q_EH": Q_EH_required,
            "Q_AH": Q_AH,
        },
        kpis=kpis,
        mass_flows_kg_s={
            "m_gen": m_gen,
            "m_mid": m_mid,
            "m_dil": m_dil,
            "mA": mA,
            "mB": mB,
            "m_vap_des": m_vap_des,
            "m11": m11,
            "m13": m13,
            "m15": m15,
            "m17": m17,
        },
        pinch_temperatures_K={
            "pinch_SL_K": pinch_SL,
            "pinch_SH_K": pinch_SH,
            "pinch_des_K": pinch_des,
            "pinch_cond_K": pinch_cond,
            "pinch_EL_K": pinch_EL,
            "pinch_AH_K": pinch_AH,
            "pinch_AL_EH_K": pinch_AL_EH,
        },
        compositions={
            "x_gen_LiBr_mol": x_gen,
            "x_mid_LiBr_mol": x_mid,
            "x_dil_LiBr_mol": x_dil,
            "w_gen_LiBr": w_gen,
            "w_mid_LiBr": w_mid,
            "w_dil_LiBr": w_dil,
        },
        residuals_raw=residuals_raw,
        residuals_scaled=residuals_scaled,
        diagnostics=diagnostics,
        checks=checks,
    )


def evaluate_model(z: np.ndarray, inputs: DAHTInputs) -> DAHTEvaluation:
    return _evaluate_model_common(z, inputs, strict=True)


def _scaled_residual_array(model: DAHTEvaluation) -> np.ndarray:
    return np.array([model.residuals_scaled[name] for name in RESIDUAL_NAMES], dtype=float)


def _penalty_vector(size: int, level: float) -> np.ndarray:
    return np.full(size, level, dtype=float)


def residual_vector(z: np.ndarray, inputs: DAHTInputs) -> np.ndarray:
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


def try_evaluate_model(z: np.ndarray, inputs: DAHTInputs) -> tuple[DAHTEvaluation | None, str | None]:
    try:
        return evaluate_model(z, inputs), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _tight_bounds(inputs: DAHTInputs, x0: np.ndarray, *, T_margin: float, x_margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds window centered on x0, clipped to the model's physical bounds.

    Used for a short warm-start phase: a tightly bounded solve is much less
    prone to wandering into a distant, unphysical root of this coupled
    system than starting directly with the wide bounds() window.
    """
    lower_phys, upper_phys = bounds(inputs)
    margin = np.array([T_margin] * 3 + [x_margin] * 3 + [T_margin] * 4, dtype=float)
    lower = np.maximum(lower_phys, x0 - margin)
    upper = np.minimum(upper_phys, x0 + margin)
    return lower, upper


def _gap_to_physical(z_gap: np.ndarray) -> np.ndarray:
    """Converts a gap-parametrized vector to the physical z used by the model.

    z_gap = [T8, T10_EL, gap_EH, x_gen, gap_mid, gap_dil, T2, T4, T4b, T5]
    with T10_EH = T10_EL + gap_EH, x_mid = x_gen - gap_mid,
    x_dil = x_mid - gap_dil.
    """
    T8, T10_EL, gap_EH, x_gen, gap_mid, gap_dil, T2, T4, T4b, T5 = z_gap
    x_mid = x_gen - gap_mid
    x_dil = x_mid - gap_dil
    return np.array([T8, T10_EL, T10_EL + gap_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5], dtype=float)


def _physical_to_gap(z_phys: np.ndarray) -> np.ndarray:
    T8, T10_EL, T10_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5 = z_phys
    return np.array(
        [T8, T10_EL, T10_EH - T10_EL, x_gen, x_gen - x_mid, x_mid - x_dil, T2, T4, T4b, T5],
        dtype=float,
    )


def _gap_bounds(inputs: DAHTInputs, *, margin_T: float, margin_x: float) -> Tuple[np.ndarray, np.ndarray]:
    lower_phys, upper_phys = bounds(inputs)
    lower = _physical_to_gap(lower_phys)
    upper = _physical_to_gap(upper_phys)
    # The gap entries (indices 2, 4, 5) don't come from a meaningful
    # physical-bounds transform above -- set them explicitly instead.
    lower[2] = margin_T
    upper[2] = 200.0
    lower[4] = margin_x
    upper[4] = 0.15
    lower[5] = margin_x
    upper[5] = 0.15
    return lower, upper


def solve_daht_gap_parametrized(
    inputs: DAHTInputs,
    x0: np.ndarray | None = None,
    *,
    warm_start_iters: int = 3,
    margin_T: float = 2.0,
    margin_x: float = 0.003,
) -> DAHTResult:
    """Solves the same system as solve_daht(), structurally forbidding the
    collapsed-pressure / crossed-concentration root by solving for the
    GAPS (T10_EH-T10_EL, x_gen-x_mid, x_mid-x_dil) instead of the raw
    values, with those gaps bounded away from zero.

    solve_daht_constrained()'s general-purpose NonlinearConstraint via
    trust-constr was tried first per the same idea, but empirically made
    things worse from an already-good starting point on this problem
    (trust-constr's BFGS/CG machinery seems confused by the very different
    scales of the temperature and concentration unknowns, despite
    Bounds/NonlinearConstraint in principle allowing exactly this). This
    reparametrization achieves the same structural guarantee more simply:
    ordinary box bounds on the gap variables, solved with the same fast,
    well-scaled least_squares 'trf' path as solve_daht().
    """
    if x0 is None:
        x0 = initial_guess(inputs)
    x0_gap = _physical_to_gap(np.asarray(x0, dtype=float))

    lower_phys, upper_phys = _gap_bounds(inputs, margin_T=margin_T, margin_x=margin_x)
    z_gap = np.clip(x0_gap, lower_phys, upper_phys)

    def _residual_gap(zz_gap: np.ndarray) -> np.ndarray:
        return residual_vector(_gap_to_physical(zz_gap), inputs)

    for _ in range(warm_start_iters):
        T_margin = np.array([15.0, 15.0, 30.0], dtype=float)
        x_margin = np.array([0.04, 0.02, 0.02], dtype=float)
        margin = np.concatenate([T_margin, x_margin, [15.0, 15.0, 15.0, 15.0]])
        lower = np.maximum(lower_phys, z_gap - margin)
        upper = np.minimum(upper_phys, z_gap + margin)
        z_gap = np.clip(z_gap, lower, upper)
        lsq_warm = least_squares(
            fun=_residual_gap, x0=z_gap, bounds=(lower, upper), method="trf", x_scale="jac",
            ftol=inputs.solver_tol, xtol=inputs.solver_tol, gtol=inputs.solver_tol, max_nfev=2000,
        )
        z_gap = np.asarray(lsq_warm.x, dtype=float)

    lsq = least_squares(
        fun=_residual_gap, x0=z_gap, bounds=(lower_phys, upper_phys), method="trf", x_scale="jac",
        ftol=inputs.solver_tol, xtol=inputs.solver_tol, gtol=inputs.solver_tol, max_nfev=inputs.max_nfev,
    )

    return _build_result(
        z_final=_gap_to_physical(np.asarray(lsq.x, dtype=float)),
        inputs=inputs,
        success=bool(lsq.success),
        status=int(lsq.status),
        message=str(lsq.message),
        cost=float(lsq.cost),
        scaled_residual_norm=float(np.linalg.norm(lsq.fun, ord=2)),
        nfev=int(lsq.nfev),
    )


def solve_daht_robust(inputs: DAHTInputs, x0: np.ndarray | None = None) -> DAHTResult:
    """Best-effort solve: tries the fast unconstrained solve_daht() first
    (which converges cleanly for well-matched, T-spec-driven design points),
    and only falls back to the slower, ordering-safe
    solve_daht_gap_parametrized() -- started from a corrected point with
    x_gen and x_dil swapped -- if that first attempt lands on the
    wrong-ordering "shadow" root (see module history / DAHT_main_Pinch_Point.py
    for the story behind this).
    """
    result = solve_daht(inputs, x0=x0, warm_start_iters=3)
    if result.solve_info.final_point_evaluable:
        return result

    z = np.array([result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES], dtype=float)
    idx = {n: i for i, n in enumerate(PRIMARY_VARIABLE_NAMES)}
    z_fix = z.copy()
    z_fix[idx["x_gen"]], z_fix[idx["x_dil"]] = z[idx["x_dil"]], z[idx["x_gen"]]

    for margin_x in (0.001, 0.0005, 0.0002):
        fallback = solve_daht_gap_parametrized(
            inputs, x0=z_fix, warm_start_iters=1, margin_T=0.5, margin_x=margin_x
        )
        if fallback.solve_info.final_point_evaluable:
            return fallback

    return fallback


def _ordering_constraint_values(z: np.ndarray) -> np.ndarray:
    """Pressure- and concentration-ordering gaps, each required to be > a margin.

    least_squares (used for the fast warm-start and polish passes) has no
    concept of an inequality constraint -- it only minimizes the sum of
    squared residuals, unconstrained beyond the box bounds(). That leaves a
    mathematically valid but unphysical root reachable: e.g. T10_EL and
    T10_EH collapsing together while x_mid drifts past x_gen, which can
    satisfy every pinch/energy residual to within tolerance without
    representing a real double-lift operating point (observed repeatedly
    during development, reproducibly, from multiple starting points).
    solve_daht_constrained() guards against this with an explicit
    NonlinearConstraint built from this function; ordinary solve_daht()
    does not enforce it and should be checked against .checks afterwards.
    """
    T8, T10_EL, T10_EH, x_gen, x_mid, x_dil, T2, T4, T4b, T5 = z
    return np.array([T10_EL - T8, T10_EH - T10_EL, x_gen - x_mid, x_mid - x_dil])


def solve_daht(inputs: DAHTInputs, x0: np.ndarray | None = None, warm_start_iters: int = 3) -> DAHTResult:
    """Solves the 10-unknown simultaneous system with plain least_squares.

    A short warm-start phase (tightly bounded around the current point)
    runs first to keep the trust-region solve in the basin of the intended
    physical root before the final, widely bounded solve. This does NOT
    enforce the pressure-/concentration-ordering constraints -- check
    result.checks afterwards, or use solve_daht_constrained() instead,
    which does enforce them (at a significant speed cost).
    """
    if x0 is None:
        x0 = initial_guess(inputs)
    x0 = np.asarray(x0, dtype=float)

    lower_phys, upper_phys = bounds(inputs)
    z = np.clip(x0, lower_phys, upper_phys)

    for _ in range(warm_start_iters):
        lower, upper = _tight_bounds(inputs, z, T_margin=15.0, x_margin=0.04)
        z = np.clip(z, lower, upper)
        lsq_warm = least_squares(
            fun=lambda zz: residual_vector(zz, inputs),
            x0=z,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=inputs.solver_tol,
            xtol=inputs.solver_tol,
            gtol=inputs.solver_tol,
            max_nfev=2000,
        )
        z = np.asarray(lsq_warm.x, dtype=float)

    lsq = least_squares(
        fun=lambda zz: residual_vector(zz, inputs),
        x0=z,
        bounds=(lower_phys, upper_phys),
        method="trf",
        x_scale="jac",
        ftol=inputs.solver_tol,
        xtol=inputs.solver_tol,
        gtol=inputs.solver_tol,
        max_nfev=inputs.max_nfev,
    )

    scaled_residual_norm = float(np.linalg.norm(lsq.fun, ord=2))
    return _build_result(
        z_final=np.asarray(lsq.x, dtype=float),
        inputs=inputs,
        success=bool(lsq.success),
        status=int(lsq.status),
        message=str(lsq.message),
        cost=float(lsq.cost),
        scaled_residual_norm=scaled_residual_norm,
        nfev=int(lsq.nfev),
    )


def _build_result(
    *, z_final: np.ndarray, inputs: DAHTInputs, success: bool, status: int, message: str,
    cost: float, scaled_residual_norm: float, nfev: int,
) -> DAHTResult:
    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, map(float, z_final)))

    model, final_evaluation_error = try_evaluate_model(z_final, inputs)
    final_point_evaluable = model is not None

    if final_point_evaluable:
        raw_residual_norm = float(
            np.linalg.norm(
                np.array([model.residuals_raw[name] for name in RESIDUAL_NAMES], dtype=float), ord=2
            )
        )
    else:
        raw_residual_norm = None

    solve_info = SolveInfo(
        success=success,
        status=status,
        message=message,
        cost=cost,
        scaled_residual_norm=scaled_residual_norm,
        raw_residual_norm=raw_residual_norm,
        nfev=nfev,
        final_point_evaluable=final_point_evaluable,
        final_evaluation_error=final_evaluation_error,
    )

    if model is None:
        return DAHTResult(
            inputs=inputs, solve_info=solve_info, primary_variables=primary_variables,
            states={}, heat_flows_kW={}, kpis={}, mass_flows_kg_s={}, pinch_temperatures_K={},
            compositions={}, residuals_raw={}, residuals_scaled={}, diagnostics={}, checks={},
        )

    return DAHTResult(
        inputs=inputs,
        solve_info=solve_info,
        primary_variables=model.primary_variables,
        states=model.states,
        heat_flows_kW=model.heat_flows_kW,
        kpis=model.kpis,
        mass_flows_kg_s=model.mass_flows_kg_s,
        pinch_temperatures_K=model.pinch_temperatures_K,
        compositions=model.compositions,
        residuals_raw=model.residuals_raw,
        residuals_scaled=model.residuals_scaled,
        diagnostics=model.diagnostics,
        checks=model.checks,
    )


def solve_daht_constrained(
    inputs: DAHTInputs,
    x0: np.ndarray | None = None,
    *,
    warm_start_iters: int = 2,
    margin_T: float = 2.0,
    margin_x: float = 0.003,
    maxiter: int = 300,
) -> DAHTResult:
    """Solves the same system as solve_daht(), but with the pressure- and
    concentration-ordering gaps enforced as hard inequality constraints
    (see _ordering_constraint_values) via scipy's trust-constr, instead of
    hoping the unconstrained least-squares solve avoids the degenerate
    collapsed-pressure root on its own.

    Runs a cheap least_squares warm start first (same as solve_daht, but
    fewer iterations) to get into a reasonable neighborhood, then a
    trust-constr polish. trust-constr is markedly more expensive per
    function evaluation than least_squares' 'trf' (it estimates a
    Hessian via BFGS on top of the constraint Jacobian), so this is
    noticeably slower -- budget noticeably more wall-clock time than
    solve_daht() for the same inputs.
    """
    if x0 is None:
        x0 = initial_guess(inputs)
    x0 = np.asarray(x0, dtype=float)

    lower_phys, upper_phys = bounds(inputs)
    z = np.clip(x0, lower_phys, upper_phys)

    for _ in range(warm_start_iters):
        lower, upper = _tight_bounds(inputs, z, T_margin=15.0, x_margin=0.04)
        z = np.clip(z, lower, upper)
        lsq_warm = least_squares(
            fun=lambda zz: residual_vector(zz, inputs),
            x0=z,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=inputs.solver_tol,
            xtol=inputs.solver_tol,
            gtol=inputs.solver_tol,
            max_nfev=1500,
        )
        z = np.asarray(lsq_warm.x, dtype=float)

    def _cost(zz: np.ndarray) -> float:
        r = residual_vector(zz, inputs)
        return float(np.dot(r, r))

    nlc = NonlinearConstraint(
        _ordering_constraint_values,
        [margin_T, margin_T, margin_x, margin_x],
        [np.inf, np.inf, np.inf, np.inf],
    )
    res = minimize(
        _cost,
        z,
        method="trust-constr",
        bounds=Bounds(lower_phys, upper_phys),
        constraints=[nlc],
        options={"maxiter": maxiter, "gtol": 1.0e-10, "xtol": 1.0e-12, "verbose": 0},
    )

    return _build_result(
        z_final=np.asarray(res.x, dtype=float),
        inputs=inputs,
        success=bool(res.get("success", False)),
        status=int(res.get("status", -1)),
        message=str(res.get("message", "")),
        cost=float(res.fun),
        scaled_residual_norm=float(np.sqrt(max(res.fun, 0.0))),
        nfev=int(res.get("nfev", 0)),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(result: DAHTResult) -> None:
    print("=" * 110)
    print("DAHT simulation - results overview (10 primary unknowns)")
    print("=" * 110)

    print("Solver information")
    print(f"  Success                : {result.solve_info.success}")
    print(f"  Status                 : {result.solve_info.status}")
    print(f"  Message                : {result.solve_info.message}")
    print(f"  Function evaluations   : {result.solve_info.nfev}")
    print(f"  Cost                   : {result.solve_info.cost:.6e}")
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
        if name.startswith("x_"):
            print(f"  {name:8s}: {result.primary_variables[name]:12.6f} -")
        else:
            print(f"  {name:8s}: {kelvin_to_celsius(result.primary_variables[name]):12.6f} degC")
    print()

    print("Heat flows [kW]")
    for key, value in result.heat_flows_kW.items():
        print(f"  {key:8s}: {value:12.6f}")
    print()

    print("KPIs")
    for key, value in result.kpis.items():
        unit = "K" if key.endswith("_K") else "-"
        print(f"  {key:24s}: {value:12.6f} {unit}")
    print()

    print("Mass flows [kg/s]")
    for key, value in result.mass_flows_kg_s.items():
        print(f"  {key:8s}: {value:12.6f}")
    print()

    print("Pinch temperatures [K]")
    for key, value in result.pinch_temperatures_K.items():
        print(f"  {key:16s}: {value:12.6f}")
    print()

    print("Compositions")
    for key, value in result.compositions.items():
        print(f"  {key:16s}: {value:12.6f}")
    print()

    print("Residuals")
    for name in RESIDUAL_NAMES:
        raw = result.residuals_raw[name]
        scaled = result.residuals_scaled[name]
        print(f"  {name:18s}: raw = {raw:14.6e} | scaled = {scaled:14.6e}")
    print()

    print("Checks")
    for key, value in result.checks.items():
        print(f"  {key:32s}: {value}")
    print("=" * 110)
