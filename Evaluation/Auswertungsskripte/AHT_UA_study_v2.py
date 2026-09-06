from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _Old.AHT_main_3 import build_example_inputs
import Thermodynamic_Properties.libr_props as lp

from _Old.AHT_simulation_3 import (
    PRIMARY_VARIABLE_NAMES,
    bounds,
    kelvin_to_celsius,
    primary_temperatures_C_to_K,
    solve_awt,
)


SCRIPT_NAME = "AHT_UA_study_v1"
ROOT_OUTPUT_DIR = Path(__file__).resolve().parent / "AHT_outputs" / SCRIPT_NAME
COMPARISON_DIRNAME = "config_comparison"
OUTER_VARIATION_COMPARISON_DIRNAME = "outer_variation_comparison"
CSV_FILENAME = "operating_points.csv"
SUMMARY_FILENAME = "summary.txt"
QABS_OPTIMUM_SUMMARY_FILENAME = "qabs_optimum_summary.txt"
QABS_OPTIMA_CSV_FILENAME = "qabs_optima.csv"
QABS_OPTIMA_OUTER_PLOT_FILENAME = "qabs_optima_over_m6_by_outer_variation.png"
QABS_REFERENCE_UA_PLOT_FILENAME = "qabs_at_reference_m6_over_varied_ua.png"
COP_REFERENCE_UA_PLOT_FILENAME = "cop_at_reference_m6_over_varied_ua.png"
QABS_MAX_UA_PLOT_FILENAME = "qabs_max_over_varied_ua.png"
COP_AT_QABS_MAX_UA_PLOT_FILENAME = "cop_at_qabs_max_over_varied_ua.png"
QABS_MAX_AND_COP_AT_QABS_MAX_UA_PLOT_FILENAME = "qabs_max_and_cop_at_qabs_max_over_varied_ua.png"
QABS_AND_COP_REFERENCE_UA_PLOT_FILENAME = "qabs_and_cop_at_reference_m6_over_varied_ua.png"
QABS_CURVES_BY_UA_PLOT_FILENAME = "qabs_over_m6_by_varied_ua_and_config.png"
PINCH_OPTIMUM_UA_PLOT_FILENAME = "pinch_min_at_qabs_max_over_varied_ua.png"
PINCH_REFERENCE_UA_PLOT_FILENAME = "pinch_min_at_reference_m6_over_varied_ua.png"
PINCH_HEATMAP_VALUE_FILENAME = "heatmap_pinch_min_over_m6_and_varied_ua.png"
PINCH_HEATMAP_UNIT_FILENAME = "heatmap_pinch_limiting_unit_over_m6_and_varied_ua.png"
CRYST_TEMP_OPTIMUM_UA_PLOT_FILENAME = "crystallization_temperature_margin_min_at_qabs_max_over_varied_ua.png"
CRYST_TEMP_REFERENCE_UA_PLOT_FILENAME = "crystallization_temperature_margin_min_at_reference_m6_over_varied_ua.png"
CRYST_TEMP_HEATMAP_VALUE_FILENAME = "heatmap_crystallization_temperature_margin_min_over_m6_and_varied_ua.png"
CRYST_TEMP_HEATMAP_STATE_FILENAME = "heatmap_crystallization_temperature_margin_limiting_state_over_m6_and_varied_ua.png"
CRYST_W_OPTIMUM_UA_PLOT_FILENAME = "crystallization_concentration_margin_min_at_qabs_max_over_varied_ua.png"
CRYST_W_REFERENCE_UA_PLOT_FILENAME = "crystallization_concentration_margin_min_at_reference_m6_over_varied_ua.png"
CRYST_W_HEATMAP_VALUE_FILENAME = "heatmap_crystallization_concentration_margin_min_over_m6_and_varied_ua.png"
CRYST_W_HEATMAP_STATE_FILENAME = "heatmap_crystallization_concentration_margin_limiting_state_over_m6_and_varied_ua.png"

SERIAL_EXTERNAL_MASSFLOW_FACTOR = 2.0

OUTER_VARIATION_MODE_UA_SCALE_PERCENT = "ua_scale_percent"
OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT = "single_ua_scale_percent"
OUTER_VARIATION_MODE_HOT_TEMPERATURE_C = "hot_temperature_C"
OUTER_VARIATION_MODE_T17_TEMPERATURE_C = "t17_temperature_C"

UA_FIELD_NAMES = ("UA_cond", "UA_evap", "UA_abs", "UA_des", "UA_shex")
PINCH_UNIT_ORDER = ("absorber", "desorber", "condenser", "evaporator", "shex")
PINCH_UNIT_LABELS = {
    "absorber": "Absorber",
    "desorber": "Desorber",
    "condenser": "Kondensator",
    "evaporator": "Verdampfer",
    "shex": "SHEX",
}
PINCH_UNIT_CODES = {unit: idx + 1 for idx, unit in enumerate(PINCH_UNIT_ORDER)}
PINCH_CODE_LABELS = {0: "invalid", **{code: PINCH_UNIT_LABELS[unit] for unit, code in PINCH_UNIT_CODES.items()}}

CRYSTALLIZATION_STATE_IDS = ("1", "2", "3", "4", "5", "6", "20")
CRYSTALLIZATION_STATE_CODES = {state_id: idx + 1 for idx, state_id in enumerate(CRYSTALLIZATION_STATE_IDS)}
CRYSTALLIZATION_CODE_LABELS = {0: "invalid", **{code: f"Zustand {state_id}" for state_id, code in CRYSTALLIZATION_STATE_CODES.items()}}

FAIL_OK = 0
FAIL_SOLVER = 1
FAIL_RESIDUAL = 2
FAIL_PRESSURE = 3
FAIL_MASSFLOW = 4
FAIL_CRYSTALLIZATION = 5
FAIL_DESORBER_DT = 6
FAIL_EVAPORATOR_DT = 7
FAIL_ABSORBER_DT = 8
FAIL_CONDENSER_DT = 9
FAIL_SHEX_DT = 10
FAIL_OTHER = 99

FAILURE_LABELS = {
    FAIL_OK: "zulässig",
    FAIL_SOLVER: "Solver / Endpunkt",
    FAIL_RESIDUAL: "Residuen zu groß",
    FAIL_PRESSURE: "Druckniveau",
    FAIL_MASSFLOW: "Massenstrom",
    FAIL_CRYSTALLIZATION: "Kristallisation",
    FAIL_DESORBER_DT: "Desorber ΔT",
    FAIL_EVAPORATOR_DT: "Verdampfer ΔT",
    FAIL_ABSORBER_DT: "Absorber ΔT",
    FAIL_CONDENSER_DT: "Kondensator ΔT",
    FAIL_SHEX_DT: "SHEX ΔT",
    FAIL_OTHER: "Sonstiges",
}


@dataclass(frozen=True)
class AnalysisConfig:
    outer_variation_mode: str = OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT
    m6_values: np.ndarray = field(
        default_factory=lambda: np.arange(0.4, 1.1, 0.01, dtype=float)
    )
    ua_scale_percent_values: np.ndarray = field(
        default_factory=lambda: np.arange(50.0, 151.0, 10.0, dtype=float)
    )
    single_ua_target: str = "UA_shex"
    single_ua_scale_percent_values: np.ndarray = field(
        default_factory=lambda: np.arange(50.0, 151.0, 10.0, dtype=float)
    )
    hot_temperature_values_C: np.ndarray = field(
        default_factory=lambda: np.arange(100.0, 141.0, 5.0, dtype=float)
    )
    t17_temperature_values_C: np.ndarray = field(
        default_factory=lambda: np.arange(15.0, 46.0, 5.0, dtype=float)
    )
    reference_m6: float = 1.0
    reference_x0_C: np.ndarray = field(
        default_factory=lambda: np.array([
            55.0,   # T8  [°C]
            101.0,  # T10 [°C]
            0.23,   # x3  [-]
            0.27,   # x6  [-]
            0.26,   # x20 [-]
            121.0,  # T2  [°C]
            150.0,  # T4  [°C]
            0.15,   # beta [-]
        ], dtype=float)
    )
    deltaT_margin_K: float = 0.5
    scaled_residual_tol: float = 1.0e-5
    solver_tol_map: float = 1.0e-7
    max_nfev_map: int = 300
    print_attempts: bool = True
    output_root_dir: Path = ROOT_OUTPUT_DIR


@dataclass(frozen=True)
class RoutingCase:
    key: str
    label: str
    folder_name: str
    routing_mode: str
    varied_m6_axis_label: str = "m6 [kg/s]"


CONFIG = AnalysisConfig()
ROUTING_CASES = [
    RoutingCase(
        key="parallel",
        label="Parallel (T13 = T15)",
        folder_name="01_parallel",
        routing_mode="parallel",
    ),
    RoutingCase(
        key="series_desorber_to_evaporator",
        label="Serie Desorber → Verdampfer",
        folder_name="02_series_desorber_to_evaporator",
        routing_mode="series_desorber_to_evaporator",
    ),
    RoutingCase(
        key="series_evaporator_to_desorber",
        label="Serie Verdampfer → Desorber",
        folder_name="03_series_evaporator_to_desorber",
        routing_mode="series_evaporator_to_desorber",
    ),
]
CASE_BY_KEY = {case.key: case for case in ROUTING_CASES}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_temperature_key(key: str) -> bool:
    return key.startswith("T") and key.endswith("_K")


def _stable_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred_prefixes = [
        "config_",
        "input_",
        "solve_",
        "status_",
        "failure_",
        "feasible_",
        "primary_",
        "heat_",
        "kpi_",
        "pump_",
        "lmtd_",
        "flash_",
        "residual_raw_",
        "residual_scaled_",
        "diagnostic_",
        "pinch_",
        "crystallization_",
        "check_",
        "state_",
        "validity_",
    ]
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    def rank(key: str) -> tuple[int, str]:
        for idx, prefix in enumerate(preferred_prefixes):
            if key.startswith(prefix):
                return idx, key
        return len(preferred_prefixes), key

    return sorted(keys, key=rank)


def _save_csv(rows: list[dict[str, Any]], filepath: Path) -> None:
    if not rows:
        raise ValueError("Keine Daten zum Schreiben vorhanden.")
    fieldnames = _stable_fieldnames(rows)
    with filepath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def build_reference_x0() -> np.ndarray:
    return primary_temperatures_C_to_K(CONFIG.reference_x0_C)


def sanitize_initial_guess(inputs, x0: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    z = np.asarray(x0, dtype=float).copy()
    lower, upper = bounds(inputs)
    lower_safe = np.asarray(lower, dtype=float) + eps
    upper_safe = np.asarray(upper, dtype=float) - eps
    midpoint = 0.5 * (lower_safe + upper_safe)
    invalid = lower_safe >= upper_safe
    lower_safe[invalid] = midpoint[invalid]
    upper_safe[invalid] = midpoint[invalid]
    return np.clip(z, lower_safe, upper_safe)


def _base_hot_value_C(base_inputs) -> float:
    if base_inputs.T_13_C is not None:
        return float(base_inputs.T_13_C)
    if base_inputs.T_15_C is not None:
        return float(base_inputs.T_15_C)
    raise ValueError("Weder T_13_C noch T_15_C sind in den Basis-Inputs gesetzt.")


def _validate_outer_variation_mode(mode: str) -> str:
    if mode not in {
        OUTER_VARIATION_MODE_UA_SCALE_PERCENT,
        OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT,
        OUTER_VARIATION_MODE_HOT_TEMPERATURE_C,
        OUTER_VARIATION_MODE_T17_TEMPERATURE_C,
    }:
        raise ValueError(
            "outer_variation_mode muss 'ua_scale_percent', 'single_ua_scale_percent', 'hot_temperature_C' oder 't17_temperature_C' sein."
        )
    return mode


def _validate_ua_field_name(ua_field_name: str) -> str:
    if ua_field_name not in UA_FIELD_NAMES:
        raise ValueError(f"ua_field_name muss einer von {UA_FIELD_NAMES} sein.")
    return ua_field_name


def _outer_variation_values(cfg: AnalysisConfig) -> list[float]:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return [float(v) for v in cfg.ua_scale_percent_values]
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return [float(v) for v in cfg.single_ua_scale_percent_values]
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return [float(v) for v in cfg.hot_temperature_values_C]
    return [float(v) for v in cfg.t17_temperature_values_C]


def _outer_variation_label(mode: str, *, ua_field_name: str | None = None) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "UA-Skalierung [%]"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"Skalierung { _validate_ua_field_name(ua_field_name or '') } [%]"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return "Abwärmetemperatur [°C]"
    return "Kühltemperatur T17 [°C]"


def _outer_variation_title(mode: str, *, ua_field_name: str | None = None) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "UA-Skalierungen"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"Skalierungen von {_validate_ua_field_name(ua_field_name or '')}"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return "Abwärmetemperaturen"
    return "Kühltemperaturen T17"


def _outer_variation_series_label(mode: str, value: float, *, ua_field_name: str | None = None) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return f"UA {float(value):.1f} %"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"{_validate_ua_field_name(ua_field_name or '')} {float(value):.1f} %"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return f"T_hot {float(value):.1f} °C"
    return f"T17 {float(value):.1f} °C"


def _outer_variation_display_string(mode: str, value: float, *, ua_field_name: str | None = None) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return f"UA-Skalierung: {float(value):.1f} %"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"Skalierung {_validate_ua_field_name(ua_field_name or '')}: {float(value):.1f} %"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return f"Abwärmetemperatur: {float(value):.1f} °C"
    return f"Kühltemperatur T17: {float(value):.1f} °C"


def _outer_variation_folder_name(mode: str, value: float, *, ua_field_name: str | None = None) -> str:
    safe = f"{float(value):.1f}".replace("-", "m").replace(".", "p")
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return f"ua_scale_{safe}pct"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"{_validate_ua_field_name(ua_field_name or '').lower()}_{safe}pct"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return f"hot_{safe}C"
    return f"t17_{safe}C"


def _ua_scale_factor_from_percent(ua_scale_percent: float) -> float:
    return float(ua_scale_percent) / 100.0


def _scaled_ua_kwargs(base_inputs, ua_scale_percent: float) -> dict[str, Any]:
    ua_scale_factor = _ua_scale_factor_from_percent(ua_scale_percent)
    return {
        "UA_cond": ua_scale_factor * float(base_inputs.UA_cond),
        "UA_evap": ua_scale_factor * float(base_inputs.UA_evap),
        "UA_abs": ua_scale_factor * float(base_inputs.UA_abs),
        "UA_des": ua_scale_factor * float(base_inputs.UA_des),
        "UA_shex": ua_scale_factor * float(base_inputs.UA_shex),
    }


def _scaled_single_ua_kwargs(base_inputs, ua_field_name: str, ua_scale_percent: float) -> dict[str, Any]:
    field_name = _validate_ua_field_name(ua_field_name)
    ua_scale_factor = _ua_scale_factor_from_percent(ua_scale_percent)
    return {field_name: ua_scale_factor * float(getattr(base_inputs, field_name))}


def _varied_ua_value_from_inputs(inputs, cfg: AnalysisConfig) -> float:
    field_name = _validate_ua_field_name(cfg.single_ua_target)
    return float(getattr(inputs, field_name))


def _base_absorber_spec_kwargs(base_inputs) -> dict[str, Any]:
    absorber_spec_mode = getattr(base_inputs, "absorber_spec_mode", None)

    if absorber_spec_mode == "T12":
        if getattr(base_inputs, "T12_spec_C", None) is None:
            raise ValueError(
                "build_example_inputs() liefert absorber_spec_mode='T12', aber keinen gültigen T12-Sollwert."
            )
        return {
            "absorber_spec_mode": "T12",
            "T12_spec_C": float(base_inputs.T12_spec_C),
            "m11_spec": None,
        }

    if absorber_spec_mode == "m11":
        if getattr(base_inputs, "m11_spec", None) is None:
            raise ValueError(
                "build_example_inputs() liefert absorber_spec_mode='m11', aber keinen gültigen m11-Sollwert."
            )
        return {
            "absorber_spec_mode": "m11",
            "m11_spec": float(base_inputs.m11_spec),
            "T12_spec_C": None,
        }

    raise ValueError(
        "build_example_inputs() muss für diese Studie absorber_spec_mode='T12' oder 'm11' liefern."
    )


def build_inputs_for_point(case: RoutingCase, m6_value: float, cfg: AnalysisConfig, *, outer_value: float):
    base = build_example_inputs()
    outer_mode = _validate_outer_variation_mode(cfg.outer_variation_mode)

    hot_C = _base_hot_value_C(base)
    T17_C = float(base.T_17_C)
    kwargs: dict[str, Any] = dict(
        cycle_scale_spec_mode="m6",
        m6_spec=float(m6_value),
        Qabs_spec_kW=None,
        desorber_evaporator_routing_mode=case.routing_mode,
        T_17_C=T17_C,
        solver_tol=float(cfg.solver_tol_map),
        max_nfev=int(cfg.max_nfev_map),
    )
    kwargs.update(_base_absorber_spec_kwargs(base))

    if outer_mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        kwargs.update(_scaled_ua_kwargs(base, outer_value))
    elif outer_mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        kwargs.update(_scaled_single_ua_kwargs(base, cfg.single_ua_target, outer_value))
    elif outer_mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        hot_C = float(outer_value)
    elif outer_mode == OUTER_VARIATION_MODE_T17_TEMPERATURE_C:
        T17_C = float(outer_value)
        kwargs["T_17_C"] = T17_C
    else:
        raise ValueError(f"Unbekannter outer_variation_mode: {outer_mode}")

    if case.routing_mode == "parallel":
        kwargs["T_13_C"] = hot_C
        kwargs["T_15_C"] = hot_C
    elif case.routing_mode == "series_desorber_to_evaporator":
        kwargs["T_13_C"] = hot_C
        kwargs["T_15_C"] = None
        kwargs["m_13"] = SERIAL_EXTERNAL_MASSFLOW_FACTOR * float(base.m_13)
        kwargs["m_15"] = SERIAL_EXTERNAL_MASSFLOW_FACTOR * float(base.m_15)
    elif case.routing_mode == "series_evaporator_to_desorber":
        kwargs["T_13_C"] = None
        kwargs["T_15_C"] = hot_C
        kwargs["m_13"] = SERIAL_EXTERNAL_MASSFLOW_FACTOR * float(base.m_13)
        kwargs["m_15"] = SERIAL_EXTERNAL_MASSFLOW_FACTOR * float(base.m_15)
    else:
        raise ValueError(f"Unbekannter routing_mode: {case.routing_mode}")

    return replace(base, **kwargs)


def _primary_vector_from_result(result) -> np.ndarray:
    return np.array([result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float)


def _temperature_differences(result) -> dict[str, float]:
    d = result.diagnostics
    return {
        "deltaT_shex_1_K": float(d["deltaT_shex_1_K"]),
        "deltaT_shex_2_K": float(d["deltaT_shex_2_K"]),
        "deltaT_des_1_K": float(d["deltaT_des_1_K"]),
        "deltaT_des_2_K": float(d["deltaT_des_2_K"]),
        "deltaT_cond_1_K": float(d["deltaT_cond_1_K"]),
        "deltaT_cond_2_K": float(d["deltaT_cond_2_K"]),
        "deltaT_evap_1_K": float(d["deltaT_evap_1_K"]),
        "deltaT_evap_2_K": float(d["deltaT_evap_2_K"]),
        "deltaT_abs_1_K": float(d["deltaT_abs_1_K"]),
        "deltaT_abs_2_K": float(d["deltaT_abs_2_K"]),
    }


def _unitwise_min_temperature_differences(result) -> dict[str, float]:
    d = result.diagnostics
    return {
        "desorber": float(min(d["deltaT_des_1_K"], d["deltaT_des_2_K"])),
        "evaporator": float(min(d["deltaT_evap_1_K"], d["deltaT_evap_2_K"])),
        "absorber": float(min(d["deltaT_abs_1_K"], d["deltaT_abs_2_K"])),
        "condenser": float(min(d["deltaT_cond_1_K"], d["deltaT_cond_2_K"])),
        "shex": float(min(d["deltaT_shex_1_K"], d["deltaT_shex_2_K"])),
    }




def _crystallization_margins_from_result(result) -> dict[str, Any]:
    temperature_candidates: list[tuple[float, str, float, float, float, float]] = []
    concentration_candidates: list[tuple[float, str, float, float, float, float]] = []

    for state_id in CRYSTALLIZATION_STATE_IDS:
        state = result.states.get(state_id, {})
        T_K = _float_or_nan(state.get("T_K"))
        w = _float_or_nan(state.get("w_LiBr"))
        if not (math.isfinite(T_K) and math.isfinite(w) and w > 0.0):
            continue
        try:
            T_crit_C = float(lp.crystallization_limit("w", w))
            T_crit_K = T_crit_C + 273.15
            w_crit = float(lp.crystallization_limit("T", T_K))
        except Exception:
            continue

        temperature_margin_K = T_K - T_crit_K
        concentration_margin = w_crit - w
        if math.isfinite(temperature_margin_K):
            temperature_candidates.append((temperature_margin_K, state_id, T_K, w, T_crit_K, w_crit))
        if math.isfinite(concentration_margin):
            concentration_candidates.append((concentration_margin, state_id, T_K, w, T_crit_K, w_crit))

    if temperature_candidates:
        temp_margin, temp_state, temp_T_K, temp_w, temp_Tcrit_K, temp_wcrit = min(temperature_candidates, key=lambda item: item[0])
    else:
        temp_margin, temp_state, temp_T_K, temp_w, temp_Tcrit_K, temp_wcrit = (float("nan"), "invalid", float("nan"), float("nan"), float("nan"), float("nan"))

    if concentration_candidates:
        w_margin, w_state, w_T_K, w_w, w_Tcrit_K, w_wcrit = min(concentration_candidates, key=lambda item: item[0])
    else:
        w_margin, w_state, w_T_K, w_w, w_Tcrit_K, w_wcrit = (float("nan"), "invalid", float("nan"), float("nan"), float("nan"), float("nan"))

    return {
        "crystallization_temperature_margin_min_K": float(temp_margin),
        "crystallization_temperature_margin_limiting_state": str(temp_state),
        "crystallization_temperature_margin_limiting_state_code": int(CRYSTALLIZATION_STATE_CODES.get(str(temp_state), 0)),
        "crystallization_temperature_margin_limiting_T_C": kelvin_to_celsius(float(temp_T_K)) if math.isfinite(_float_or_nan(temp_T_K)) else float("nan"),
        "crystallization_temperature_margin_limiting_w_LiBr": float(temp_w),
        "crystallization_temperature_margin_limiting_Tcrit_C": kelvin_to_celsius(float(temp_Tcrit_K)) if math.isfinite(_float_or_nan(temp_Tcrit_K)) else float("nan"),
        "crystallization_temperature_margin_limiting_wcrit_LiBr": float(temp_wcrit),
        "crystallization_concentration_margin_min": float(w_margin),
        "crystallization_concentration_margin_limiting_state": str(w_state),
        "crystallization_concentration_margin_limiting_state_code": int(CRYSTALLIZATION_STATE_CODES.get(str(w_state), 0)),
        "crystallization_concentration_margin_limiting_T_C": kelvin_to_celsius(float(w_T_K)) if math.isfinite(_float_or_nan(w_T_K)) else float("nan"),
        "crystallization_concentration_margin_limiting_w_LiBr": float(w_w),
        "crystallization_concentration_margin_limiting_Tcrit_C": kelvin_to_celsius(float(w_Tcrit_K)) if math.isfinite(_float_or_nan(w_Tcrit_K)) else float("nan"),
        "crystallization_concentration_margin_limiting_wcrit_LiBr": float(w_wcrit),
    }

def determine_limiting_pinch(result) -> tuple[str, float, str]:
    pinch_map = _temperature_differences(result)
    limiting_key = min(pinch_map, key=pinch_map.get)
    limiting_value = float(pinch_map[limiting_key])
    limiting_unit = {
        "deltaT_shex_1_K": "shex",
        "deltaT_shex_2_K": "shex",
        "deltaT_des_1_K": "desorber",
        "deltaT_des_2_K": "desorber",
        "deltaT_cond_1_K": "condenser",
        "deltaT_cond_2_K": "condenser",
        "deltaT_evap_1_K": "evaporator",
        "deltaT_evap_2_K": "evaporator",
        "deltaT_abs_1_K": "absorber",
        "deltaT_abs_2_K": "absorber",
    }[limiting_key]
    return limiting_key, limiting_value, limiting_unit


def classify_failure(result, *, deltaT_margin_K: float, scaled_residual_tol: float) -> tuple[int, str, str]:
    if not result.solve_info.success:
        return FAIL_SOLVER, "solver_failed", "unknown"
    if not result.solve_info.final_point_evaluable:
        return FAIL_SOLVER, "final_point_not_evaluable", "unknown"
    if result.solve_info.scaled_residual_norm >= scaled_residual_tol:
        return FAIL_RESIDUAL, "scaled_residual_too_large", "unknown"

    checks = result.checks
    if not checks.get("p_high_gt_p_low", False):
        return FAIL_PRESSURE, "pressure_level_invalid", "pressure"
    if not checks.get("m7_positive", False) or not checks.get("m21_nonnegative", False):
        return FAIL_MASSFLOW, "mass_flow_invalid", "massflow"
    if not checks.get("crystallization_safe_all_checked_states", False):
        return FAIL_CRYSTALLIZATION, "crystallization_risk", "crystallization"

    dt_unit_min = _unitwise_min_temperature_differences(result)
    limiting_unit = min(dt_unit_min, key=dt_unit_min.get)
    limiting_value = dt_unit_min[limiting_unit]

    if limiting_value < 0.0:
        unit_code = {
            "desorber": FAIL_DESORBER_DT,
            "evaporator": FAIL_EVAPORATOR_DT,
            "absorber": FAIL_ABSORBER_DT,
            "condenser": FAIL_CONDENSER_DT,
            "shex": FAIL_SHEX_DT,
        }
        return unit_code[limiting_unit], f"{limiting_unit}_deltaT_negative", limiting_unit

    if limiting_value < deltaT_margin_K:
        return FAIL_OK, f"feasible_but_near_limit_{limiting_unit}", limiting_unit

    return FAIL_OK, "feasible", limiting_unit


def evaluate_point(case: RoutingCase, m6_value: float, *, x0: np.ndarray, cfg: AnalysisConfig, outer_value: float) -> tuple[dict[str, Any], np.ndarray | None]:
    outer_mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    inputs = build_inputs_for_point(case, m6_value, cfg, outer_value=outer_value)
    lower, upper = bounds(inputs)

    hot_C = _base_hot_value_C(inputs)
    row: dict[str, Any] = {
        "config_case_key": case.key,
        "config_case_label": case.label,
        "config_routing_mode": case.routing_mode,
        "input_m6_kg_s": float(m6_value),
        "input_fixed_T17_C": float(inputs.T_17_C),
        "input_fixed_hot_C": float(hot_C),
        "input_fixed_T13_C": float(inputs.T_13_C) if inputs.T_13_C is not None else float("nan"),
        "input_fixed_T15_C": float(inputs.T_15_C) if inputs.T_15_C is not None else float("nan"),
        "input_fixed_m13_kg_s": float(inputs.m_13),
        "input_fixed_m15_kg_s": float(inputs.m_15),
        "input_fixed_m17_kg_s": float(inputs.m_17),
        "input_outer_variation_mode": outer_mode,
        "input_outer_variation_value": float(outer_value),
        "input_outer_variation_label": _outer_variation_label(outer_mode, ua_field_name=cfg.single_ua_target),
        "input_ua_scale_percent": float(outer_value) if outer_mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT else float("nan"),
        "input_ua_scale_factor": _ua_scale_factor_from_percent(outer_value) if outer_mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT else float("nan"),
        "input_single_ua_target": cfg.single_ua_target if outer_mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT else "",
        "input_single_ua_scale_percent": float(outer_value) if outer_mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT else float("nan"),
        "input_varied_UA_value": _varied_ua_value_from_inputs(inputs, cfg) if outer_mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT else float("nan"),
        "input_hot_variation_C": float(outer_value) if outer_mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C else float("nan"),
        "input_t17_variation_C": float(outer_value) if outer_mode == OUTER_VARIATION_MODE_T17_TEMPERATURE_C else float("nan"),
        "input_scaled_UA_cond": float(inputs.UA_cond),
        "input_scaled_UA_evap": float(inputs.UA_evap),
        "input_scaled_UA_abs": float(inputs.UA_abs),
        "input_scaled_UA_des": float(inputs.UA_des),
        "input_scaled_UA_shex": float(inputs.UA_shex),
        "solve_success": False,
        "solve_status": -999,
        "solve_message": "",
        "solve_nfev": 0,
        "solve_cost": float("nan"),
        "solve_scaled_residual_norm": float("nan"),
        "solve_raw_residual_norm": float("nan"),
        "status_final_point_evaluable": False,
        "status_final_evaluation_error": "",
        "failure_code": FAIL_OTHER,
        "failure_reason": "not_classified",
        "failure_label": FAILURE_LABELS[FAIL_OTHER],
        "feasible_numeric": False,
        "feasible_physical": False,
        "feasible_margin": False,
        "pinch_deltaT_min_global_K": float("nan"),
        "pinch_limiting_unit": "unknown",
        "pinch_limiting_key": "invalid",
        "pinch_limiting_unit_code": 0,
        "pinch_min_absorber_K": float("nan"),
        "pinch_min_desorber_K": float("nan"),
        "pinch_min_condenser_K": float("nan"),
        "pinch_min_evaporator_K": float("nan"),
        "pinch_min_shex_K": float("nan"),
        "crystallization_temperature_margin_min_K": float("nan"),
        "crystallization_temperature_margin_limiting_state": "invalid",
        "crystallization_temperature_margin_limiting_state_code": 0,
        "crystallization_temperature_margin_limiting_T_C": float("nan"),
        "crystallization_temperature_margin_limiting_w_LiBr": float("nan"),
        "crystallization_temperature_margin_limiting_Tcrit_C": float("nan"),
        "crystallization_temperature_margin_limiting_wcrit_LiBr": float("nan"),
        "crystallization_concentration_margin_min": float("nan"),
        "crystallization_concentration_margin_limiting_state": "invalid",
        "crystallization_concentration_margin_limiting_state_code": 0,
        "crystallization_concentration_margin_limiting_T_C": float("nan"),
        "crystallization_concentration_margin_limiting_w_LiBr": float("nan"),
        "crystallization_concentration_margin_limiting_Tcrit_C": float("nan"),
        "crystallization_concentration_margin_limiting_wcrit_LiBr": float("nan"),
    }

    if np.any(np.asarray(lower, dtype=float) >= np.asarray(upper, dtype=float)):
        row["failure_code"] = FAIL_SOLVER
        row["failure_reason"] = "invalid_solver_bounds"
        row["failure_label"] = FAILURE_LABELS[FAIL_SOLVER]
        row["solve_message"] = "invalid_solver_bounds"
        row["status_final_evaluation_error"] = "invalid_solver_bounds"
        return row, None

    z0 = sanitize_initial_guess(inputs, x0)
    result = solve_awt(inputs, x0=z0)

    row["solve_success"] = bool(result.solve_info.success)
    row["solve_status"] = int(result.solve_info.status)
    row["solve_message"] = str(result.solve_info.message)
    row["solve_nfev"] = int(result.solve_info.nfev)
    row["solve_cost"] = float(result.solve_info.cost)
    row["solve_scaled_residual_norm"] = float(result.solve_info.scaled_residual_norm)
    row["solve_raw_residual_norm"] = (
        float(result.solve_info.raw_residual_norm)
        if result.solve_info.raw_residual_norm is not None
        else float("nan")
    )
    row["status_final_point_evaluable"] = bool(result.solve_info.final_point_evaluable)
    row["status_final_evaluation_error"] = (
        "" if result.solve_info.final_evaluation_error is None else str(result.solve_info.final_evaluation_error)
    )

    x_next: np.ndarray | None = None

    if result.solve_info.final_point_evaluable:
        failure_code, failure_reason, _ = classify_failure(
            result,
            deltaT_margin_K=cfg.deltaT_margin_K,
            scaled_residual_tol=cfg.scaled_residual_tol,
        )
        pinch_key, pinch_value, pinch_unit = determine_limiting_pinch(result)
        dt_unit_min = _unitwise_min_temperature_differences(result)
        crystallization_margins = _crystallization_margins_from_result(result)
        row["pinch_deltaT_min_global_K"] = float(pinch_value)
        row["pinch_limiting_unit"] = pinch_unit
        row["pinch_limiting_key"] = pinch_key
        row["pinch_limiting_unit_code"] = int(PINCH_UNIT_CODES.get(pinch_unit, 0))
        row["pinch_min_absorber_K"] = float(dt_unit_min["absorber"])
        row["pinch_min_desorber_K"] = float(dt_unit_min["desorber"])
        row["pinch_min_condenser_K"] = float(dt_unit_min["condenser"])
        row["pinch_min_evaporator_K"] = float(dt_unit_min["evaporator"])
        row["pinch_min_shex_K"] = float(dt_unit_min["shex"])
        row.update(crystallization_margins)
        row["failure_code"] = int(failure_code)
        row["failure_reason"] = str(failure_reason)
        row["failure_label"] = FAILURE_LABELS.get(failure_code, FAILURE_LABELS[FAIL_OTHER])
        row["feasible_numeric"] = bool(
            result.solve_info.success
            and result.solve_info.final_point_evaluable
            and result.solve_info.scaled_residual_norm < cfg.scaled_residual_tol
        )
        row["feasible_physical"] = bool(failure_code == FAIL_OK)
        row["feasible_margin"] = bool(
            row["feasible_physical"] and float(row["pinch_deltaT_min_global_K"]) >= cfg.deltaT_margin_K
        )

        for name, value in result.primary_variables.items():
            if name.startswith("T"):
                row[f"primary_{name}_C"] = kelvin_to_celsius(float(value))
            else:
                row[f"primary_{name}"] = float(value)

        for key, value in result.heat_flows_kW.items():
            row[f"heat_{key}_kW"] = float(value)
        for key, value in result.kpis.items():
            row[f"kpi_{key}"] = float(value)
        for key, value in result.pump_work_kW.items():
            row[f"pump_{key}_kW"] = float(value)
        for key, value in result.lmtd_K.items():
            row[f"lmtd_{key}_K"] = float(value)
        for key, value in result.flash_outputs.items():
            if _is_temperature_key(key):
                row[f"flash_{key[:-2]}_C"] = kelvin_to_celsius(float(value))
            else:
                row[f"flash_{key}"] = float(value)
        for key, value in result.residuals_raw.items():
            row[f"residual_raw_{key}"] = float(value)
        for key, value in result.residuals_scaled.items():
            row[f"residual_scaled_{key}"] = float(value)
        for key, value in result.diagnostics.items():
            if _is_temperature_key(key):
                row[f"diagnostic_{key[:-2]}_C"] = kelvin_to_celsius(float(value))
            else:
                row[f"diagnostic_{key}"] = float(value)
        for key, value in result.checks.items():
            row[f"check_{key}"] = bool(value)
        for state_id, state in result.states.items():
            for state_key, state_value in state.items():
                if state_key == "T_K":
                    row[f"state_{state_id}_T_C"] = kelvin_to_celsius(float(state_value))
                else:
                    row[f"state_{state_id}_{state_key}"] = float(state_value)
        for idx, message in enumerate(result.validity_messages, start=1):
            row[f"validity_message_{idx}"] = message
        row["validity_message_count"] = len(result.validity_messages)

        x_next = _primary_vector_from_result(result)
    else:
        row["failure_code"] = FAIL_SOLVER
        row["failure_reason"] = "final_point_not_evaluable"
        row["failure_label"] = FAILURE_LABELS[FAIL_SOLVER]

    return row, x_next


def _m6_sequence_center_out(values: np.ndarray, reference_value: float) -> tuple[list[float], list[float]]:
    vals = [float(v) for v in values]
    left = [v for v in vals if v < reference_value]
    right = [v for v in vals if v > reference_value]
    return left[::-1], right


def _evaluate_and_store(
    *,
    case: RoutingCase,
    m6_value: float,
    x0: np.ndarray,
    cfg: AnalysisConfig,
    outer_value: float,
    rows: list[dict[str, Any]],
    solved_store: dict[float, np.ndarray],
    verbose_prefix: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if cfg.print_attempts:
        print(f"    löse Punkt m6={m6_value:8.5f} kg/s | start={verbose_prefix}")
    row, x_next = evaluate_point(case, m6_value, x0=x0, cfg=cfg, outer_value=outer_value)
    rows.append(row)
    if x_next is not None:
        solved_store[float(m6_value)] = np.array(x_next, dtype=float, copy=True)
    marker = "OK+" if bool(row["feasible_margin"]) else ("OK" if bool(row["feasible_physical"]) else "--")
    cop = _float_or_nan(row.get("kpi_COP"))
    q_abs = _float_or_nan(row.get("heat_Q_abs_kW"))
    fr = _float_or_nan(row.get("kpi_FR"))
    print(
        f"  m6={m6_value:8.5f} kg/s | {marker:>3s} | COP={cop:>8.4f} | FR={fr:>8.4f} | "
        f"Q_abs={q_abs:>9.4f} kW | reason={row['failure_reason']}"
    )
    return row, x_next


def run_case(case: RoutingCase, cfg: AnalysisConfig, *, outer_value: float) -> tuple[list[dict[str, Any]], Path, dict[str, Any] | None]:
    if not np.any(np.isclose(np.asarray(cfg.m6_values, dtype=float), float(cfg.reference_m6), rtol=0.0, atol=1.0e-12)):
        raise ValueError("reference_m6 muss im m6-Raster enthalten sein.")

    case_dir = cfg.output_root_dir / case.folder_name
    _ensure_dir(case_dir)

    rows: list[dict[str, Any]] = []
    solved_store: dict[float, np.ndarray] = {}

    print("=" * 110)
    print(f"Auswertung Konfiguration: {case.label}")
    print(f"Routing-Mode: {case.routing_mode}")
    print(
        f"m6-Raster: {float(np.min(cfg.m6_values)):.5f} bis {float(np.max(cfg.m6_values)):.5f} kg/s"
    )
    print(f"Referenzpunkt: m6={cfg.reference_m6:.5f} kg/s")
    print(_outer_variation_display_string(cfg.outer_variation_mode, outer_value, ua_field_name=cfg.single_ua_target))
    print("Ablauf: Referenzpunkt -> Sweep nach oben -> Sweep nach unten")
    print("=" * 110)

    reference_x0 = build_reference_x0()
    _, x_ref = _evaluate_and_store(
        case=case,
        m6_value=float(cfg.reference_m6),
        x0=reference_x0,
        cfg=cfg,
        outer_value=outer_value,
        rows=rows,
        solved_store=solved_store,
        verbose_prefix="reference_x0",
    )
    if x_ref is None:
        raise RuntimeError(
            f"Referenzpunkt für Konfiguration '{case.key}' konnte nicht physikalisch auswertbar gelöst werden."
        )

    left_values_desc, right_values_asc = _m6_sequence_center_out(cfg.m6_values, cfg.reference_m6)

    prev_valid_right = np.array(x_ref, dtype=float, copy=True)
    for m6_value in right_values_asc:
        _, x_next = _evaluate_and_store(
            case=case,
            m6_value=float(m6_value),
            x0=prev_valid_right,
            cfg=cfg,
            outer_value=outer_value,
            rows=rows,
            solved_store=solved_store,
            verbose_prefix="previous_valid_right",
        )
        if x_next is not None:
            prev_valid_right = np.array(x_next, dtype=float, copy=True)

    prev_valid_left = np.array(x_ref, dtype=float, copy=True)
    for m6_value in left_values_desc:
        _, x_next = _evaluate_and_store(
            case=case,
            m6_value=float(m6_value),
            x0=prev_valid_left,
            cfg=cfg,
            outer_value=outer_value,
            rows=rows,
            solved_store=solved_store,
            verbose_prefix="previous_valid_left",
        )
        if x_next is not None:
            prev_valid_left = np.array(x_next, dtype=float, copy=True)

    rows_sorted = sorted(rows, key=lambda row: float(row["input_m6_kg_s"]))
    optimum = _determine_qabs_optimum(rows_sorted, cfg)
    _save_csv(rows_sorted, case_dir / CSV_FILENAME)
    _write_summary(rows_sorted, case, cfg, case_dir / SUMMARY_FILENAME, outer_value=outer_value)
    _write_qabs_optimum_summary(optimum, case, cfg, case_dir / QABS_OPTIMUM_SUMMARY_FILENAME, outer_value=outer_value)
    print(f"CSV gespeichert unter: {case_dir / CSV_FILENAME}")
    if optimum is None:
        print("Q_abs-Optimum: kein zulässiger Betriebspunkt gefunden.")
    else:
        print(
            f"Q_abs-Optimum: m6={float(optimum['input_m6_kg_s']):.6f} kg/s | "
            f"Q_abs={float(optimum['heat_Q_abs_kW']):.6f} kW | "
            f"Randoptimum={bool(optimum['optimum_at_any_m6_boundary'])}"
        )
    return rows_sorted, case_dir, optimum


def _write_summary(
    rows: list[dict[str, Any]],
    case: RoutingCase,
    cfg: AnalysisConfig,
    filepath: Path,
    *,
    outer_value: float,
) -> None:
    total = len(rows)
    feasible_numeric = sum(bool(row.get("feasible_numeric", False)) for row in rows)
    feasible_physical = sum(bool(row.get("feasible_physical", False)) for row in rows)
    feasible_margin = sum(bool(row.get("feasible_margin", False)) for row in rows)
    base = build_example_inputs()
    hot_C = _base_hot_value_C(base)
    lines = [
        f"Konfiguration: {case.label}",
        "=" * 72,
        f"m6-Bereich: {float(np.min(cfg.m6_values)):.5f} bis {float(np.max(cfg.m6_values)):.5f} kg/s",
        f"Referenzpunkt: m6={cfg.reference_m6:.5f} kg/s",
        f"Basis-T17: {base.T_17_C:.3f} °C",
        f"Basis-Hot-Niveau: {hot_C:.3f} °C",
        _outer_variation_display_string(cfg.outer_variation_mode, outer_value, ua_field_name=cfg.single_ua_target),
        f"deltaT_margin_K: {cfg.deltaT_margin_K:.3f}",
        f"scaled_residual_tol: {cfg.scaled_residual_tol:.3e}",
        "-",
        f"Punkte gesamt: {total}",
        f"feasible_numeric: {feasible_numeric}",
        f"feasible_physical: {feasible_physical}",
        f"feasible_margin: {feasible_margin}",
        "-",
        "Failure-Code-Verteilung:",
    ]
    for code in sorted(FAILURE_LABELS):
        count = sum(int(row.get("failure_code", FAIL_OTHER)) == code for row in rows)
        lines.append(f"  {code:>2d} | {FAILURE_LABELS[code]:20s}: {count}")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _determine_qabs_optimum(rows: list[dict[str, Any]], cfg: AnalysisConfig) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if bool(row.get("feasible_physical", False))
        and math.isfinite(_float_or_nan(row.get("heat_Q_abs_kW")))
        and math.isfinite(_float_or_nan(row.get("input_m6_kg_s")))
    ]
    if not candidates:
        return None

    optimum_row = max(
        candidates,
        key=lambda row: (_float_or_nan(row.get("heat_Q_abs_kW")), -_float_or_nan(row.get("input_m6_kg_s"))),
    )
    m6_opt = float(optimum_row["input_m6_kg_s"])
    qabs_opt = float(optimum_row["heat_Q_abs_kW"])
    is_at_lower_boundary = bool(np.isclose(m6_opt, float(np.min(cfg.m6_values)), rtol=0.0, atol=1.0e-12))
    is_at_upper_boundary = bool(np.isclose(m6_opt, float(np.max(cfg.m6_values)), rtol=0.0, atol=1.0e-12))

    return {
        "config_case_key": str(optimum_row.get("config_case_key", "")),
        "config_case_label": str(optimum_row.get("config_case_label", "")),
        "input_outer_variation_mode": str(optimum_row.get("input_outer_variation_mode", "")),
        "input_outer_variation_value": _float_or_nan(optimum_row.get("input_outer_variation_value")),
        "input_outer_variation_label": str(optimum_row.get("input_outer_variation_label", "")),
        "input_ua_scale_percent": _float_or_nan(optimum_row.get("input_ua_scale_percent")),
        "input_ua_scale_factor": _float_or_nan(optimum_row.get("input_ua_scale_factor")),
        "input_single_ua_target": str(optimum_row.get("input_single_ua_target", "")),
        "input_single_ua_scale_percent": _float_or_nan(optimum_row.get("input_single_ua_scale_percent")),
        "input_varied_UA_value": _float_or_nan(optimum_row.get("input_varied_UA_value")),
        "input_hot_variation_C": _float_or_nan(optimum_row.get("input_hot_variation_C")),
        "input_t17_variation_C": _float_or_nan(optimum_row.get("input_t17_variation_C")),
        "input_m6_kg_s": m6_opt,
        "heat_Q_abs_kW": qabs_opt,
        "kpi_COP": _float_or_nan(optimum_row.get("kpi_COP")),
        "kpi_FR": _float_or_nan(optimum_row.get("kpi_FR")),
        "heat_Q_des_kW": _float_or_nan(optimum_row.get("heat_Q_des_kW")),
        "heat_Q_evap_kW": _float_or_nan(optimum_row.get("heat_Q_evap_kW")),
        "heat_Q_cond_kW": _float_or_nan(optimum_row.get("heat_Q_cond_kW")),
        "state_7_m_kg_s": _float_or_nan(optimum_row.get("state_7_m_kg_s")),
        "state_21_m_kg_s": _float_or_nan(optimum_row.get("state_21_m_kg_s")),
        "primary_beta": _float_or_nan(optimum_row.get("primary_beta")),
        "pinch_deltaT_min_global_K": _float_or_nan(optimum_row.get("pinch_deltaT_min_global_K")),
        "pinch_limiting_unit": str(optimum_row.get("pinch_limiting_unit", "")),
        "pinch_limiting_key": str(optimum_row.get("pinch_limiting_key", "")),
        "pinch_limiting_unit_code": int(_float_or_nan(optimum_row.get("pinch_limiting_unit_code"))) if math.isfinite(_float_or_nan(optimum_row.get("pinch_limiting_unit_code"))) else 0,
        "pinch_min_absorber_K": _float_or_nan(optimum_row.get("pinch_min_absorber_K")),
        "pinch_min_desorber_K": _float_or_nan(optimum_row.get("pinch_min_desorber_K")),
        "pinch_min_condenser_K": _float_or_nan(optimum_row.get("pinch_min_condenser_K")),
        "pinch_min_evaporator_K": _float_or_nan(optimum_row.get("pinch_min_evaporator_K")),
        "pinch_min_shex_K": _float_or_nan(optimum_row.get("pinch_min_shex_K")),
        "crystallization_temperature_margin_min_K": _float_or_nan(optimum_row.get("crystallization_temperature_margin_min_K")),
        "crystallization_temperature_margin_limiting_state": str(optimum_row.get("crystallization_temperature_margin_limiting_state", "")),
        "crystallization_temperature_margin_limiting_state_code": int(_float_or_nan(optimum_row.get("crystallization_temperature_margin_limiting_state_code"))) if math.isfinite(_float_or_nan(optimum_row.get("crystallization_temperature_margin_limiting_state_code"))) else 0,
        "crystallization_temperature_margin_limiting_T_C": _float_or_nan(optimum_row.get("crystallization_temperature_margin_limiting_T_C")),
        "crystallization_temperature_margin_limiting_w_LiBr": _float_or_nan(optimum_row.get("crystallization_temperature_margin_limiting_w_LiBr")),
        "crystallization_temperature_margin_limiting_Tcrit_C": _float_or_nan(optimum_row.get("crystallization_temperature_margin_limiting_Tcrit_C")),
        "crystallization_temperature_margin_limiting_wcrit_LiBr": _float_or_nan(optimum_row.get("crystallization_temperature_margin_limiting_wcrit_LiBr")),
        "crystallization_concentration_margin_min": _float_or_nan(optimum_row.get("crystallization_concentration_margin_min")),
        "crystallization_concentration_margin_limiting_state": str(optimum_row.get("crystallization_concentration_margin_limiting_state", "")),
        "crystallization_concentration_margin_limiting_state_code": int(_float_or_nan(optimum_row.get("crystallization_concentration_margin_limiting_state_code"))) if math.isfinite(_float_or_nan(optimum_row.get("crystallization_concentration_margin_limiting_state_code"))) else 0,
        "crystallization_concentration_margin_limiting_T_C": _float_or_nan(optimum_row.get("crystallization_concentration_margin_limiting_T_C")),
        "crystallization_concentration_margin_limiting_w_LiBr": _float_or_nan(optimum_row.get("crystallization_concentration_margin_limiting_w_LiBr")),
        "crystallization_concentration_margin_limiting_Tcrit_C": _float_or_nan(optimum_row.get("crystallization_concentration_margin_limiting_Tcrit_C")),
        "crystallization_concentration_margin_limiting_wcrit_LiBr": _float_or_nan(optimum_row.get("crystallization_concentration_margin_limiting_wcrit_LiBr")),
        "optimum_at_lower_m6_boundary": is_at_lower_boundary,
        "optimum_at_upper_m6_boundary": is_at_upper_boundary,
        "optimum_at_any_m6_boundary": bool(is_at_lower_boundary or is_at_upper_boundary),
        "n_feasible_physical_points": int(len(candidates)),
    }


def _write_qabs_optimum_summary(
    optimum: dict[str, Any] | None,
    case: RoutingCase,
    cfg: AnalysisConfig,
    filepath: Path,
    *,
    outer_value: float,
) -> None:
    lines = [
        f"Konfiguration: {case.label}",
        "=" * 72,
        f"m6-Bereich: {float(np.min(cfg.m6_values)):.5f} bis {float(np.max(cfg.m6_values)):.5f} kg/s",
        f"Referenzpunkt: m6={cfg.reference_m6:.5f} kg/s",
        _outer_variation_display_string(cfg.outer_variation_mode, float(optimum["input_outer_variation_value"]), ua_field_name=cfg.single_ua_target) if optimum is not None else _outer_variation_display_string(cfg.outer_variation_mode, outer_value, ua_field_name=cfg.single_ua_target),
        "-",
    ]

    if optimum is None:
        lines.append("Kein physikalisch zulässiger Betriebspunkt mit endlichem Q_abs gefunden.")
    else:
        lines.extend([
            f"Q_abs-Optimum bei m6           : {float(optimum['input_m6_kg_s']):.6f} kg/s",
            f"Maximales Q_abs                : {float(optimum['heat_Q_abs_kW']):.6f} kW",
            f"COP am Optimum                 : {float(optimum['kpi_COP']):.6f}",
            f"FR am Optimum                  : {float(optimum['kpi_FR']):.6f}",
            f"Q_des am Optimum               : {float(optimum['heat_Q_des_kW']):.6f} kW",
            f"Q_evap am Optimum              : {float(optimum['heat_Q_evap_kW']):.6f} kW",
            f"Q_cond am Optimum              : {float(optimum['heat_Q_cond_kW']):.6f} kW",
            f"m7 am Optimum                  : {float(optimum['state_7_m_kg_s']):.6f} kg/s",
            f"m21 am Optimum                 : {float(optimum['state_21_m_kg_s']):.6f} kg/s",
            f"beta am Optimum                : {float(optimum['primary_beta']):.6f}",
            f"minimaler Pinch am Optimum     : {_float_or_nan(optimum.get('pinch_deltaT_min_global_K')):.6f} K ({optimum.get('pinch_limiting_unit', 'unknown')})",
            f"Pinch Absorber am Optimum      : {_float_or_nan(optimum.get('pinch_min_absorber_K')):.6f} K",
            f"Pinch Desorber am Optimum      : {_float_or_nan(optimum.get('pinch_min_desorber_K')):.6f} K",
            f"Pinch Kondensator am Optimum   : {_float_or_nan(optimum.get('pinch_min_condenser_K')):.6f} K",
            f"Pinch Verdampfer am Optimum    : {_float_or_nan(optimum.get('pinch_min_evaporator_K')):.6f} K",
            f"Pinch SHEX am Optimum          : {_float_or_nan(optimum.get('pinch_min_shex_K')):.6f} K",
            f"Krist.-T-Abstand am Optimum    : {_float_or_nan(optimum.get('crystallization_temperature_margin_min_K')):.6f} K (Zustand {optimum.get('crystallization_temperature_margin_limiting_state', 'invalid')})",
            f"Krist.-w-Abstand am Optimum    : {_float_or_nan(optimum.get('crystallization_concentration_margin_min')):.6f} (Zustand {optimum.get('crystallization_concentration_margin_limiting_state', 'invalid')})",
            f"Optimum am Rand des m6-Rasters : {bool(optimum['optimum_at_any_m6_boundary'])}",
            f"Feasible Punkte für Suche      : {int(optimum['n_feasible_physical_points'])}",
        ])

    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_qabs_optima_csv(optima: list[dict[str, Any]], filepath: Path) -> None:
    if not optima:
        return
    fieldnames = _stable_fieldnames(optima)
    with filepath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(optima)


def _comparison_plot_over_m6(case_rows: dict[str, list[dict[str, Any]]], value_key: str, ylabel: str, filepath: Path) -> None:
    plt.figure(figsize=(8.5, 6.0))
    for case in ROUTING_CASES:
        rows = case_rows.get(case.key, [])
        rows = sorted(rows, key=lambda row: float(row["input_m6_kg_s"]))
        xs = [float(row["input_m6_kg_s"]) for row in rows]
        ys = [
            float(row[value_key])
            if bool(row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(row.get(value_key)))
            else float("nan")
            for row in rows
        ]
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, label=case.label)
    plt.xlabel("m6 [kg/s]")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} über m6")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def create_comparison_plots(case_rows: dict[str, list[dict[str, Any]]], cfg: AnalysisConfig) -> None:
    comparison_dir = cfg.output_root_dir / COMPARISON_DIRNAME
    _ensure_dir(comparison_dir)
    _comparison_plot_over_m6(case_rows, "kpi_COP", "COP [-]", comparison_dir / "comparison_COP_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "kpi_FR", "FR [-]", comparison_dir / "comparison_FR_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "heat_Q_abs_kW", "Q_abs [kW]", comparison_dir / "comparison_Qabs_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "heat_Q_des_kW", "Q_des [kW]", comparison_dir / "comparison_Qdes_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "heat_Q_evap_kW", "Q_evap [kW]", comparison_dir / "comparison_Qevap_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "heat_Q_cond_kW", "Q_cond [kW]", comparison_dir / "comparison_Qcond_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "state_7_m_kg_s", "m7 [kg/s]", comparison_dir / "comparison_m7_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "state_21_m_kg_s", "m21 [kg/s]", comparison_dir / "comparison_m21_vs_m6.png")
    _comparison_plot_over_m6(case_rows, "primary_beta", "beta [-]", comparison_dir / "comparison_beta_vs_m6.png")


def _plot_qabs_optima_over_m6_by_outer_variation(optima: list[dict[str, Any]], cfg: AnalysisConfig, filepath: Path) -> None:
    from matplotlib.lines import Line2D

    outer_mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    valid_optima = [
        opt for opt in optima
        if math.isfinite(_float_or_nan(opt.get("input_m6_kg_s")))
        and math.isfinite(_float_or_nan(opt.get("heat_Q_abs_kW")))
        and math.isfinite(_float_or_nan(opt.get("input_outer_variation_value")))
    ]

    plt.figure(figsize=(9.5, 6.5))
    if not valid_optima:
        plt.text(0.5, 0.5, "Keine gültigen Q_abs-Optima verfügbar.", ha="center", va="center", transform=plt.gca().transAxes)
        plt.xlabel("m6 [kg/s]")
        plt.ylabel("Q_abs [kW]")
        plt.title(f"Q_abs-Optima über m6 für verschiedene {_outer_variation_title(outer_mode, ua_field_name=cfg.single_ua_target)}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close()
        return

    unique_values = sorted({float(opt["input_outer_variation_value"]) for opt in valid_optima})
    cmap = plt.get_cmap("viridis", max(len(unique_values), 1))
    value_to_color = {value: cmap(idx) for idx, value in enumerate(unique_values)}
    marker_map = {
        "parallel": "o",
        "series_desorber_to_evaporator": "s",
        "series_evaporator_to_desorber": "^",
    }

    for opt in valid_optima:
        outer_value = float(opt["input_outer_variation_value"])
        case_key = str(opt.get("config_case_key", ""))
        marker = marker_map.get(case_key, "o")
        color = value_to_color[outer_value]
        plt.plot(
            [float(opt["input_m6_kg_s"])],
            [float(opt["heat_Q_abs_kW"])],
            linestyle="None",
            marker=marker,
            markersize=8.0,
            color=color,
        )

    plt.xlabel("m6 [kg/s]")
    plt.ylabel("Q_abs [kW]")
    plt.title(f"Q_abs-Optima über m6 für verschiedene {_outer_variation_title(outer_mode, ua_field_name=cfg.single_ua_target)}")
    plt.grid(True, alpha=0.3)

    color_handles = [
        Line2D([], [], linestyle="None", marker="o", markersize=8.0, color=value_to_color[value], label=_outer_variation_series_label(outer_mode, value, ua_field_name=cfg.single_ua_target))
        for value in unique_values
    ]
    marker_handles = [
        Line2D([], [], linestyle="None", marker=marker_map[case.key], markersize=8.0, color="black", label=case.label)
        for case in ROUTING_CASES
    ]

    ax = plt.gca()
    legend_colors = ax.legend(handles=color_handles, title=_outer_variation_label(outer_mode, ua_field_name=cfg.single_ua_target), loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(legend_colors)
    ax.legend(handles=marker_handles, title="Konfiguration", loc="upper left", bbox_to_anchor=(1.02, 0.45))
    plt.subplots_adjust(right=0.70)
    plt.savefig(filepath, dpi=200)
    plt.close()




def _reference_m6_row(rows: list[dict[str, Any]], reference_m6: float) -> dict[str, Any] | None:
    for row in rows:
        if math.isclose(_float_or_nan(row.get("input_m6_kg_s")), float(reference_m6), rel_tol=0.0, abs_tol=1.0e-12):
            return row
    return None


def _ua_reference_plot_x(row: dict[str, Any], cfg: AnalysisConfig, outer_value: float) -> float:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return float(outer_value)
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return _float_or_nan(row.get("input_varied_UA_value"))
    return float("nan")


def _ua_reference_plot_xlabel(cfg: AnalysisConfig) -> str:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "gemeinsame UA-Skalierung [%]"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"{_validate_ua_field_name(cfg.single_ua_target)} [kW/K]"
    return "äußerer Variationswert"


def _plot_qabs_over_varied_ua_at_reference_m6(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    filepath: Path,
) -> None:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    plt.figure(figsize=(8.5, 6.0))

    if mode not in {OUTER_VARIATION_MODE_UA_SCALE_PERCENT, OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT}:
        plt.text(0.5, 0.5, "Dieser Plot ist nur für UA-Variationsmodi definiert.", ha="center", va="center", transform=plt.gca().transAxes)
        plt.xlabel(_outer_variation_label(mode, ua_field_name=cfg.single_ua_target))
        plt.ylabel("Q_abs [kW]")
        plt.title(f"Q_abs bei m6=reference_m6 für {_outer_variation_title(mode, ua_field_name=cfg.single_ua_target)}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close()
        return

    outer_values_sorted = sorted(all_case_rows.keys())
    for case in ROUTING_CASES:
        xs: list[float] = []
        ys: list[float] = []
        for outer_value in outer_values_sorted:
            rows = all_case_rows.get(float(outer_value), {}).get(case.key, [])
            ref_row = _reference_m6_row(rows, cfg.reference_m6)
            if ref_row is None:
                continue
            x_val = _ua_reference_plot_x(ref_row, cfg, float(outer_value))
            y_val = (
                _float_or_nan(ref_row.get("heat_Q_abs_kW"))
                if bool(ref_row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(ref_row.get("heat_Q_abs_kW")))
                else float("nan")
            )
            xs.append(x_val)
            ys.append(y_val)

        order = np.argsort(np.asarray(xs, dtype=float)) if xs else []
        xs_sorted = [xs[idx] for idx in order] if len(xs) else []
        ys_sorted = [ys[idx] for idx in order] if len(xs) else []
        plt.plot(xs_sorted, ys_sorted, marker="o", linewidth=1.2, markersize=3.0, label=case.label)

    plt.xlabel(_ua_reference_plot_xlabel(cfg))
    plt.ylabel("Q_abs [kW]")
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        title = f"Q_abs bei m6={cfg.reference_m6:.5f} kg/s über gemeinsamer UA-Skalierung"
    else:
        title = f"Q_abs bei m6={cfg.reference_m6:.5f} kg/s über {_validate_ua_field_name(cfg.single_ua_target)}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()



def _outer_x_from_row(row: dict[str, Any], cfg: AnalysisConfig, outer_value: float) -> float:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        x_val = _float_or_nan(row.get("input_varied_UA_value"))
        return x_val if math.isfinite(x_val) else float(outer_value)
    return float(outer_value)


def _outer_x_from_optimum(optimum: dict[str, Any], cfg: AnalysisConfig) -> float:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        x_val = _float_or_nan(optimum.get("input_varied_UA_value"))
        return x_val if math.isfinite(x_val) else _float_or_nan(optimum.get("input_outer_variation_value"))
    return _float_or_nan(optimum.get("input_outer_variation_value"))


def _outer_plot_xlabel(cfg: AnalysisConfig) -> str:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "gemeinsame UA-Skalierung [%]"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"{_validate_ua_field_name(cfg.single_ua_target)} [kW/K]"
    return _outer_variation_label(mode, ua_field_name=cfg.single_ua_target)


def _outer_plot_title_suffix(cfg: AnalysisConfig) -> str:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "gemeinsamer UA-Skalierung"
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return _validate_ua_field_name(cfg.single_ua_target)
    return _outer_variation_title(mode, ua_field_name=cfg.single_ua_target)


def _outer_legend_label_for_x_value(cfg: AnalysisConfig, x_value: float) -> str:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_SINGLE_UA_SCALE_PERCENT:
        return f"{_validate_ua_field_name(cfg.single_ua_target)} {float(x_value):.3g} kW/K"
    return _outer_variation_series_label(mode, float(x_value), ua_field_name=cfg.single_ua_target)


def _outer_pairs_for_plot(all_case_rows: dict[float, dict[str, list[dict[str, Any]]]], cfg: AnalysisConfig) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for outer_value in sorted(all_case_rows.keys()):
        x_val = float(outer_value)
        for case in ROUTING_CASES:
            rows = all_case_rows.get(float(outer_value), {}).get(case.key, [])
            if rows:
                x_candidate = _outer_x_from_row(rows[0], cfg, float(outer_value))
                if math.isfinite(x_candidate):
                    x_val = x_candidate
                    break
        pairs.append((float(outer_value), float(x_val)))
    return sorted(pairs, key=lambda item: item[1])


def _plot_metric_over_varied_ua_at_reference_m6(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    value_key: str,
    ylabel: str,
    filepath: Path,
    *,
    title: str,
) -> None:
    plt.figure(figsize=(8.5, 6.0))
    outer_pairs = _outer_pairs_for_plot(all_case_rows, cfg)

    for case in ROUTING_CASES:
        xs: list[float] = []
        ys: list[float] = []
        for outer_value, x_val in outer_pairs:
            rows = all_case_rows.get(float(outer_value), {}).get(case.key, [])
            ref_row = _reference_m6_row(rows, cfg.reference_m6)
            if ref_row is None:
                continue
            y_val = _float_or_nan(ref_row.get(value_key))
            if not (bool(ref_row.get("feasible_physical", False)) and math.isfinite(y_val)):
                y_val = float("nan")
            xs.append(float(x_val))
            ys.append(float(y_val))
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, label=case.label)

    plt.xlabel(_outer_plot_xlabel(cfg))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _plot_metric_over_varied_ua_at_optimum(
    optima: list[dict[str, Any]],
    cfg: AnalysisConfig,
    value_key: str,
    ylabel: str,
    filepath: Path,
    *,
    title: str,
) -> None:
    plt.figure(figsize=(8.5, 6.0))
    for case in ROUTING_CASES:
        case_optima = [opt for opt in optima if str(opt.get("config_case_key", "")) == case.key]
        xy = []
        for opt in case_optima:
            x_val = _outer_x_from_optimum(opt, cfg)
            y_val = _float_or_nan(opt.get(value_key))
            if math.isfinite(x_val) and math.isfinite(y_val):
                xy.append((x_val, y_val))
        xy.sort(key=lambda item: item[0])
        plt.plot(
            [item[0] for item in xy],
            [item[1] for item in xy],
            marker="o",
            linewidth=1.2,
            markersize=3.0,
            label=case.label,
        )
    plt.xlabel(_outer_plot_xlabel(cfg))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _plot_dual_axis_optimum_metrics_over_varied_ua(
    optima: list[dict[str, Any]],
    cfg: AnalysisConfig,
    left_key: str,
    left_label: str,
    right_key: str,
    right_label: str,
    filepath: Path,
    *,
    title: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9.0, 6.0))
    ax2 = ax1.twinx()
    line_styles = {"left": "-", "right": "--"}
    handles: list[Any] = []
    labels: list[str] = []

    for case in ROUTING_CASES:
        case_optima = [opt for opt in optima if str(opt.get("config_case_key", "")) == case.key]
        left_xy: list[tuple[float, float]] = []
        right_xy: list[tuple[float, float]] = []
        for opt in case_optima:
            x_val = _outer_x_from_optimum(opt, cfg)
            left_val = _float_or_nan(opt.get(left_key))
            right_val = _float_or_nan(opt.get(right_key))
            if math.isfinite(x_val) and math.isfinite(left_val):
                left_xy.append((x_val, left_val))
            if math.isfinite(x_val) and math.isfinite(right_val):
                right_xy.append((x_val, right_val))
        left_xy.sort(key=lambda item: item[0])
        right_xy.sort(key=lambda item: item[0])
        h1, = ax1.plot([x for x, _ in left_xy], [y for _, y in left_xy], marker="o", linestyle=line_styles["left"], linewidth=1.2, markersize=3.0, label=f"{case.label} – {left_label}")
        h2, = ax2.plot([x for x, _ in right_xy], [y for _, y in right_xy], marker="s", linestyle=line_styles["right"], linewidth=1.2, markersize=3.0, label=f"{case.label} – {right_label}")
        handles.extend([h1, h2])
        labels.extend([h1.get_label(), h2.get_label()])

    ax1.set_xlabel(_outer_plot_xlabel(cfg))
    ax1.set_ylabel(left_label)
    ax2.set_ylabel(right_label)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def _plot_dual_axis_reference_metrics_over_varied_ua(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    left_key: str,
    left_label: str,
    right_key: str,
    right_label: str,
    filepath: Path,
    *,
    title: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9.0, 6.0))
    ax2 = ax1.twinx()
    outer_pairs = _outer_pairs_for_plot(all_case_rows, cfg)
    handles: list[Any] = []
    labels: list[str] = []

    for case in ROUTING_CASES:
        xs_left: list[float] = []
        ys_left: list[float] = []
        xs_right: list[float] = []
        ys_right: list[float] = []
        for outer_value, x_val in outer_pairs:
            rows = all_case_rows.get(float(outer_value), {}).get(case.key, [])
            ref_row = _reference_m6_row(rows, cfg.reference_m6)
            if ref_row is None:
                continue
            left_val = _float_or_nan(ref_row.get(left_key))
            right_val = _float_or_nan(ref_row.get(right_key))
            if bool(ref_row.get("feasible_physical", False)) and math.isfinite(left_val):
                xs_left.append(float(x_val)); ys_left.append(left_val)
            if bool(ref_row.get("feasible_physical", False)) and math.isfinite(right_val):
                xs_right.append(float(x_val)); ys_right.append(right_val)
        h1, = ax1.plot(xs_left, ys_left, marker="o", linestyle="-", linewidth=1.2, markersize=3.0, label=f"{case.label} – {left_label}")
        h2, = ax2.plot(xs_right, ys_right, marker="s", linestyle="--", linewidth=1.2, markersize=3.0, label=f"{case.label} – {right_label}")
        handles.extend([h1, h2])
        labels.extend([h1.get_label(), h2.get_label()])

    ax1.set_xlabel(_outer_plot_xlabel(cfg))
    ax1.set_ylabel(left_label)
    ax2.set_ylabel(right_label)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def _plot_qabs_over_m6_by_varied_ua_and_config(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    filepath: Path,
) -> None:
    from matplotlib.lines import Line2D

    outer_pairs = _outer_pairs_for_plot(all_case_rows, cfg)
    plt.figure(figsize=(10.5, 6.8))
    cmap = plt.get_cmap("viridis", max(len(outer_pairs), 1))
    outer_to_color = {outer_value: cmap(idx) for idx, (outer_value, _) in enumerate(outer_pairs)}
    linestyle_map = {
        "parallel": "-",
        "series_desorber_to_evaporator": "--",
        "series_evaporator_to_desorber": ":",
    }

    for outer_value, _ in outer_pairs:
        color = outer_to_color[outer_value]
        for case in ROUTING_CASES:
            rows = sorted(all_case_rows.get(float(outer_value), {}).get(case.key, []), key=lambda row: _float_or_nan(row.get("input_m6_kg_s")))
            xs = [_float_or_nan(row.get("input_m6_kg_s")) for row in rows]
            ys = [
                _float_or_nan(row.get("heat_Q_abs_kW"))
                if bool(row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(row.get("heat_Q_abs_kW")))
                else float("nan")
                for row in rows
            ]
            plt.plot(xs, ys, color=color, linestyle=linestyle_map.get(case.key, "-"), linewidth=1.2)

    plt.xlabel("m6 [kg/s]")
    plt.ylabel("Q_abs [kW]")
    plt.title(f"Q_abs über m6 für verschiedene {_outer_variation_title(cfg.outer_variation_mode, ua_field_name=cfg.single_ua_target)}")
    plt.grid(True, alpha=0.3)

    color_handles = [
        Line2D([], [], color=outer_to_color[outer_value], linestyle="-", label=_outer_legend_label_for_x_value(cfg, x_value))
        for outer_value, x_value in outer_pairs
    ]
    linestyle_handles = [
        Line2D([], [], color="black", linestyle=linestyle_map[case.key], label=case.label)
        for case in ROUTING_CASES
    ]
    ax = plt.gca()
    legend_colors = ax.legend(handles=color_handles, title=_outer_plot_xlabel(cfg), loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(legend_colors)
    ax.legend(handles=linestyle_handles, title="Konfiguration", loc="upper left", bbox_to_anchor=(1.02, 0.45))
    plt.subplots_adjust(right=0.70)
    plt.savefig(filepath, dpi=200)
    plt.close()


def _heatmap_matrix_for_case(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    case_key: str,
    value_key: str,
    *,
    require_feasible: bool = True,
) -> tuple[np.ndarray, list[float], list[float]]:
    outer_pairs = _outer_pairs_for_plot(all_case_rows, cfg)
    m6_values = [float(v) for v in cfg.m6_values]
    matrix = np.full((len(m6_values), len(outer_pairs)), np.nan, dtype=float)

    for x_idx, (outer_value, _x_val) in enumerate(outer_pairs):
        rows = all_case_rows.get(float(outer_value), {}).get(case_key, [])
        for y_idx, m6_value in enumerate(m6_values):
            row = _reference_m6_row(rows, m6_value)
            if row is None:
                continue
            val = _float_or_nan(row.get(value_key))
            if require_feasible and not bool(row.get("feasible_physical", False)):
                val = float("nan")
            matrix[y_idx, x_idx] = val if math.isfinite(val) else float("nan")
    x_values = [x_val for _outer_value, x_val in outer_pairs]
    return matrix, x_values, m6_values


def _plot_heatmap_metric_over_m6_and_varied_ua(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    value_key: str,
    cbar_label: str,
    filepath: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(ROUTING_CASES), figsize=(5.4 * len(ROUTING_CASES), 5.6), squeeze=False)
    axes_flat = axes[0]
    images = []

    for ax, case in zip(axes_flat, ROUTING_CASES):
        matrix, x_values, m6_values = _heatmap_matrix_for_case(all_case_rows, cfg, case.key, value_key, require_feasible=True)
        if x_values and m6_values:
            extent = [min(x_values), max(x_values), min(m6_values), max(m6_values)]
        else:
            extent = [0.0, 1.0, 0.0, 1.0]
        image = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto", extent=extent)
        images.append(image)
        ax.set_title(case.label)
        ax.set_xlabel(_outer_plot_xlabel(cfg))
        ax.set_ylabel("m6 [kg/s]")
        ax.grid(False)

    fig.suptitle(title)
    if images:
        fig.colorbar(images[0], ax=axes_flat.tolist(), label=cbar_label, shrink=0.88)
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_category_over_m6_and_varied_ua(
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]],
    cfg: AnalysisConfig,
    value_key: str,
    code_labels: dict[int, str],
    filepath: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(ROUTING_CASES), figsize=(5.4 * len(ROUTING_CASES), 5.6), squeeze=False)
    axes_flat = axes[0]
    max_code = max(code_labels.keys()) if code_labels else 0
    cmap = plt.get_cmap("tab10", max(max_code + 1, 1))
    images = []

    for ax, case in zip(axes_flat, ROUTING_CASES):
        matrix, x_values, m6_values = _heatmap_matrix_for_case(all_case_rows, cfg, case.key, value_key, require_feasible=True)
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
        if x_values and m6_values:
            extent = [min(x_values), max(x_values), min(m6_values), max(m6_values)]
        else:
            extent = [0.0, 1.0, 0.0, 1.0]
        image = ax.imshow(matrix, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=-0.5, vmax=max_code + 0.5)
        images.append(image)
        ax.set_title(case.label)
        ax.set_xlabel(_outer_plot_xlabel(cfg))
        ax.set_ylabel("m6 [kg/s]")
        ax.grid(False)

    fig.suptitle(title)
    if images:
        cbar = fig.colorbar(images[0], ax=axes_flat.tolist(), ticks=list(sorted(code_labels.keys())), shrink=0.88)
        cbar.ax.set_yticklabels([code_labels[code] for code in sorted(code_labels.keys())])
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> dict[float, dict[str, list[dict[str, Any]]]]:
    _ensure_dir(CONFIG.output_root_dir)
    outer_mode = _validate_outer_variation_mode(CONFIG.outer_variation_mode)
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]] = {}
    all_optima: list[dict[str, Any]] = []

    for outer_value in _outer_variation_values(CONFIG):
        loop_root_dir = CONFIG.output_root_dir / _outer_variation_folder_name(outer_mode, outer_value, ua_field_name=CONFIG.single_ua_target)
        loop_cfg = replace(CONFIG, output_root_dir=loop_root_dir)
        _ensure_dir(loop_cfg.output_root_dir)

        print("#" * 110)
        print(f"Äußerer Variationsdurchlauf: {_outer_variation_display_string(outer_mode, outer_value, ua_field_name=CONFIG.single_ua_target)}")
        print("#" * 110)

        case_rows: dict[str, list[dict[str, Any]]] = {}
        optima_this_outer_value: list[dict[str, Any]] = []
        for case in ROUTING_CASES:
            rows, _, optimum = run_case(case, loop_cfg, outer_value=outer_value)
            case_rows[case.key] = rows
            if optimum is not None:
                optima_this_outer_value.append(optimum)
                all_optima.append(optimum)

        create_comparison_plots(case_rows, loop_cfg)
        comparison_dir = loop_cfg.output_root_dir / COMPARISON_DIRNAME
        _ensure_dir(comparison_dir)
        _write_qabs_optima_csv(optima_this_outer_value, comparison_dir / QABS_OPTIMA_CSV_FILENAME)
        all_case_rows[outer_value] = case_rows

    outer_comparison_dir = CONFIG.output_root_dir / OUTER_VARIATION_COMPARISON_DIRNAME
    _ensure_dir(outer_comparison_dir)
    _write_qabs_optima_csv(all_optima, outer_comparison_dir / QABS_OPTIMA_CSV_FILENAME)
    _plot_qabs_optima_over_m6_by_outer_variation(all_optima, CONFIG, outer_comparison_dir / QABS_OPTIMA_OUTER_PLOT_FILENAME)
    _plot_qabs_over_varied_ua_at_reference_m6(all_case_rows, CONFIG, outer_comparison_dir / QABS_REFERENCE_UA_PLOT_FILENAME)
    _plot_metric_over_varied_ua_at_reference_m6(
        all_case_rows, CONFIG, "kpi_COP", "COP [-]",
        outer_comparison_dir / COP_REFERENCE_UA_PLOT_FILENAME,
        title=f"COP bei m6={CONFIG.reference_m6:.5f} kg/s über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_optimum(
        all_optima, CONFIG, "heat_Q_abs_kW", "Q_abs,max [kW]",
        outer_comparison_dir / QABS_MAX_UA_PLOT_FILENAME,
        title=f"Q_abs,max über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_optimum(
        all_optima, CONFIG, "kpi_COP", "COP bei Q_abs,max [-]",
        outer_comparison_dir / COP_AT_QABS_MAX_UA_PLOT_FILENAME,
        title=f"COP bei Q_abs,max über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_dual_axis_optimum_metrics_over_varied_ua(
        all_optima, CONFIG, "heat_Q_abs_kW", "Q_abs,max [kW]", "kpi_COP", "COP bei Q_abs,max [-]",
        outer_comparison_dir / QABS_MAX_AND_COP_AT_QABS_MAX_UA_PLOT_FILENAME,
        title=f"Q_abs,max und COP bei Q_abs,max über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_dual_axis_reference_metrics_over_varied_ua(
        all_case_rows, CONFIG, "heat_Q_abs_kW", "Q_abs [kW]", "kpi_COP", "COP [-]",
        outer_comparison_dir / QABS_AND_COP_REFERENCE_UA_PLOT_FILENAME,
        title=f"Q_abs und COP bei m6={CONFIG.reference_m6:.5f} kg/s über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_qabs_over_m6_by_varied_ua_and_config(
        all_case_rows, CONFIG, outer_comparison_dir / QABS_CURVES_BY_UA_PLOT_FILENAME,
    )
    _plot_metric_over_varied_ua_at_optimum(
        all_optima, CONFIG, "pinch_deltaT_min_global_K", "minimaler Pinch [K]",
        outer_comparison_dir / PINCH_OPTIMUM_UA_PLOT_FILENAME,
        title=f"minimaler Pinch bei Q_abs,max über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_reference_m6(
        all_case_rows, CONFIG, "pinch_deltaT_min_global_K", "minimaler Pinch [K]",
        outer_comparison_dir / PINCH_REFERENCE_UA_PLOT_FILENAME,
        title=f"minimaler Pinch bei m6={CONFIG.reference_m6:.5f} kg/s über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_heatmap_metric_over_m6_and_varied_ua(
        all_case_rows, CONFIG, "pinch_deltaT_min_global_K", "minimaler Pinch [K]",
        outer_comparison_dir / PINCH_HEATMAP_VALUE_FILENAME,
        title=f"minimaler Pinch über m6 und {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_heatmap_category_over_m6_and_varied_ua(
        all_case_rows, CONFIG, "pinch_limiting_unit_code", PINCH_CODE_LABELS,
        outer_comparison_dir / PINCH_HEATMAP_UNIT_FILENAME,
        title=f"limitierender Pinch-Apparat über m6 und {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_optimum(
        all_optima, CONFIG, "crystallization_temperature_margin_min_K", "minimaler Kristallisations-T-Abstand [K]",
        outer_comparison_dir / CRYST_TEMP_OPTIMUM_UA_PLOT_FILENAME,
        title=f"minimaler Kristallisations-T-Abstand bei Q_abs,max über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_reference_m6(
        all_case_rows, CONFIG, "crystallization_temperature_margin_min_K", "minimaler Kristallisations-T-Abstand [K]",
        outer_comparison_dir / CRYST_TEMP_REFERENCE_UA_PLOT_FILENAME,
        title=f"minimaler Kristallisations-T-Abstand bei m6={CONFIG.reference_m6:.5f} kg/s über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_heatmap_metric_over_m6_and_varied_ua(
        all_case_rows, CONFIG, "crystallization_temperature_margin_min_K", "minimaler Kristallisations-T-Abstand [K]",
        outer_comparison_dir / CRYST_TEMP_HEATMAP_VALUE_FILENAME,
        title=f"minimaler Kristallisations-T-Abstand über m6 und {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_heatmap_category_over_m6_and_varied_ua(
        all_case_rows, CONFIG, "crystallization_temperature_margin_limiting_state_code", CRYSTALLIZATION_CODE_LABELS,
        outer_comparison_dir / CRYST_TEMP_HEATMAP_STATE_FILENAME,
        title=f"limitierender Kristallisations-Zustand (T-Abstand) über m6 und {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_optimum(
        all_optima, CONFIG, "crystallization_concentration_margin_min", "minimaler Kristallisations-w-Abstand [-]",
        outer_comparison_dir / CRYST_W_OPTIMUM_UA_PLOT_FILENAME,
        title=f"minimaler Kristallisations-w-Abstand bei Q_abs,max über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_metric_over_varied_ua_at_reference_m6(
        all_case_rows, CONFIG, "crystallization_concentration_margin_min", "minimaler Kristallisations-w-Abstand [-]",
        outer_comparison_dir / CRYST_W_REFERENCE_UA_PLOT_FILENAME,
        title=f"minimaler Kristallisations-w-Abstand bei m6={CONFIG.reference_m6:.5f} kg/s über {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_heatmap_metric_over_m6_and_varied_ua(
        all_case_rows, CONFIG, "crystallization_concentration_margin_min", "minimaler Kristallisations-w-Abstand [-]",
        outer_comparison_dir / CRYST_W_HEATMAP_VALUE_FILENAME,
        title=f"minimaler Kristallisations-w-Abstand über m6 und {_outer_plot_title_suffix(CONFIG)}",
    )
    _plot_heatmap_category_over_m6_and_varied_ua(
        all_case_rows, CONFIG, "crystallization_concentration_margin_limiting_state_code", CRYSTALLIZATION_CODE_LABELS,
        outer_comparison_dir / CRYST_W_HEATMAP_STATE_FILENAME,
        title=f"limitierender Kristallisations-Zustand (w-Abstand) über m6 und {_outer_plot_title_suffix(CONFIG)}",
    )

    print("=" * 110)
    print("Alle äußeren Variations- und Konfigurationsdurchläufe abgeschlossen.")
    print(f"Ausgabeordner: {CONFIG.output_root_dir}")
    print("=" * 110)
    return all_case_rows


if __name__ == "__main__":
    main()
