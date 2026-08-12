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
from _Old.AHT_simulation_3 import (
    PRIMARY_VARIABLE_NAMES,
    bounds,
    kelvin_to_celsius,
    primary_temperatures_C_to_K,
    solve_awt,
)


SCRIPT_NAME = "AHT_m6_study_v5"
ROOT_OUTPUT_DIR = Path(__file__).resolve().parent / "AHT_outputs" / SCRIPT_NAME
COMPARISON_DIRNAME = "config_comparison"
OUTER_VARIATION_COMPARISON_DIRNAME = "outer_variation_comparison"
CSV_FILENAME = "operating_points.csv"
SUMMARY_FILENAME = "summary.txt"
QABS_OPTIMUM_SUMMARY_FILENAME = "qabs_optimum_summary.txt"
QABS_OPTIMA_CSV_FILENAME = "qabs_optima.csv"
QABS_OPTIMA_OUTER_PLOT_FILENAME = "qabs_optima_over_m6_by_outer_variation.png"

SERIAL_EXTERNAL_MASSFLOW_FACTOR = 2.0

OUTER_VARIATION_MODE_UA_SCALE_PERCENT = "ua_scale_percent"
OUTER_VARIATION_MODE_HOT_TEMPERATURE_C = "hot_temperature_C"
OUTER_VARIATION_MODE_T17_TEMPERATURE_C = "t17_temperature_C"

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
    outer_variation_mode: str = OUTER_VARIATION_MODE_UA_SCALE_PERCENT
    m6_values: np.ndarray = field(
        default_factory=lambda: np.arange(0.4, 1.1, 0.01, dtype=float)
    )
    ua_scale_percent_values: np.ndarray = field(
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
        OUTER_VARIATION_MODE_HOT_TEMPERATURE_C,
        OUTER_VARIATION_MODE_T17_TEMPERATURE_C,
    }:
        raise ValueError(
            "outer_variation_mode muss 'ua_scale_percent', 'hot_temperature_C' oder 't17_temperature_C' sein."
        )
    return mode


def _outer_variation_values(cfg: AnalysisConfig) -> list[float]:
    mode = _validate_outer_variation_mode(cfg.outer_variation_mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return [float(v) for v in cfg.ua_scale_percent_values]
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return [float(v) for v in cfg.hot_temperature_values_C]
    return [float(v) for v in cfg.t17_temperature_values_C]


def _outer_variation_label(mode: str) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "UA-Skalierung [%]"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return "Abwärmetemperatur [°C]"
    return "Kühltemperatur T17 [°C]"


def _outer_variation_title(mode: str) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return "UA-Skalierungen"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return "Abwärmetemperaturen"
    return "Kühltemperaturen T17"


def _outer_variation_series_label(mode: str, value: float) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return f"UA {float(value):.1f} %"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return f"T_hot {float(value):.1f} °C"
    return f"T17 {float(value):.1f} °C"


def _outer_variation_display_string(mode: str, value: float) -> str:
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return f"UA-Skalierung: {float(value):.1f} %"
    if mode == OUTER_VARIATION_MODE_HOT_TEMPERATURE_C:
        return f"Abwärmetemperatur: {float(value):.1f} °C"
    return f"Kühltemperatur T17: {float(value):.1f} °C"


def _outer_variation_folder_name(mode: str, value: float) -> str:
    safe = f"{float(value):.1f}".replace("-", "m").replace(".", "p")
    mode = _validate_outer_variation_mode(mode)
    if mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT:
        return f"ua_scale_{safe}pct"
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
        "input_outer_variation_label": _outer_variation_label(outer_mode),
        "input_ua_scale_percent": float(outer_value) if outer_mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT else float("nan"),
        "input_ua_scale_factor": _ua_scale_factor_from_percent(outer_value) if outer_mode == OUTER_VARIATION_MODE_UA_SCALE_PERCENT else float("nan"),
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
        row["pinch_deltaT_min_global_K"] = float(pinch_value)
        row["pinch_limiting_unit"] = pinch_unit
        row["pinch_limiting_key"] = pinch_key
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
    print(_outer_variation_display_string(cfg.outer_variation_mode, outer_value))
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
        _outer_variation_display_string(cfg.outer_variation_mode, outer_value),
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
        _outer_variation_display_string(cfg.outer_variation_mode, float(optimum["input_outer_variation_value"])) if optimum is not None else _outer_variation_display_string(cfg.outer_variation_mode, outer_value),
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
        plt.title(f"Q_abs-Optima über m6 für verschiedene {_outer_variation_title(outer_mode)}")
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
    plt.title(f"Q_abs-Optima über m6 für verschiedene {_outer_variation_title(outer_mode)}")
    plt.grid(True, alpha=0.3)

    color_handles = [
        Line2D([], [], linestyle="None", marker="o", markersize=8.0, color=value_to_color[value], label=_outer_variation_series_label(outer_mode, value))
        for value in unique_values
    ]
    marker_handles = [
        Line2D([], [], linestyle="None", marker=marker_map[case.key], markersize=8.0, color="black", label=case.label)
        for case in ROUTING_CASES
    ]

    ax = plt.gca()
    legend_colors = ax.legend(handles=color_handles, title=_outer_variation_label(outer_mode), loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(legend_colors)
    ax.legend(handles=marker_handles, title="Konfiguration", loc="upper left", bbox_to_anchor=(1.02, 0.45))
    plt.subplots_adjust(right=0.70)
    plt.savefig(filepath, dpi=200)
    plt.close()


def main() -> dict[float, dict[str, list[dict[str, Any]]]]:
    _ensure_dir(CONFIG.output_root_dir)
    outer_mode = _validate_outer_variation_mode(CONFIG.outer_variation_mode)
    all_case_rows: dict[float, dict[str, list[dict[str, Any]]]] = {}
    all_optima: list[dict[str, Any]] = []

    for outer_value in _outer_variation_values(CONFIG):
        loop_root_dir = CONFIG.output_root_dir / _outer_variation_folder_name(outer_mode, outer_value)
        loop_cfg = replace(CONFIG, output_root_dir=loop_root_dir)
        _ensure_dir(loop_cfg.output_root_dir)

        print("#" * 110)
        print(f"Äußerer Variationsdurchlauf: {_outer_variation_display_string(outer_mode, outer_value)}")
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

    print("=" * 110)
    print("Alle äußeren Variations- und Konfigurationsdurchläufe abgeschlossen.")
    print(f"Ausgabeordner: {CONFIG.output_root_dir}")
    print("=" * 110)
    return all_case_rows


if __name__ == "__main__":
    main()
