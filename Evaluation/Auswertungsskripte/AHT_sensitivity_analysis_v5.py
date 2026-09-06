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
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from _Old.AHT_main_3 import build_example_inputs
from _Old.AHT_simulation_3 import (
    PRIMARY_VARIABLE_NAMES,
    bounds,
    kelvin_to_celsius,
    primary_temperatures_C_to_K,
    solve_awt,
)


SCRIPT_NAME = "AHT_sensitivity_analysis_v5"
ROOT_OUTPUT_DIR = Path(__file__).resolve().parent / "AHT_outputs" / SCRIPT_NAME
COMPARISON_DIRNAME = "config_comparison"
CSV_FILENAME = "operating_points.csv"
SUMMARY_FILENAME = "summary.txt"

SERIAL_EXTERNAL_MASSFLOW_FACTOR = 2.0

SELECTED_T17_CURVES = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
SELECTED_HOT_CURVES = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
COMPARISON_T17_C = 30.0
COMPARISON_HOT_INPUT_C = 120.0

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

PINCH_LABELS = {
    "deltaT_shex_1_K": "SHEX ΔT1",
    "deltaT_shex_2_K": "SHEX ΔT2",
    "deltaT_des_1_K": "Desorber ΔT1",
    "deltaT_des_2_K": "Desorber ΔT2",
    "deltaT_cond_1_K": "Kondensator ΔT1",
    "deltaT_cond_2_K": "Kondensator ΔT2",
    "deltaT_evap_1_K": "Verdampfer ΔT1",
    "deltaT_evap_2_K": "Verdampfer ΔT2",
    "deltaT_abs_1_K": "Absorber ΔT1",
    "deltaT_abs_2_K": "Absorber ΔT2",
}
PINCH_CODE_MAP = {key: idx + 1 for idx, key in enumerate(PINCH_LABELS)}
PINCH_CODE_INVALID = 0
PINCH_UNIT_FROM_KEY = {
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
}


@dataclass(frozen=True)
class AnalysisConfig:
    hot_values_C: np.ndarray = field(
        default_factory=lambda: np.arange(100.0, 136.0, 2.5, dtype=float)
    )
    T17_values_C: np.ndarray = field(
        default_factory=lambda: np.arange(15.0, 46.0, 2.5, dtype=float)
    )
    reference_hot_C: float = 120.0
    reference_T17_C: float = 30.0
    reference_x0_C: np.ndarray = field(
        default_factory=lambda: np.array([55.0, 101.0, 0.23, 0.27, 0.26, 121.0, 150.0, 0.20], dtype=float)
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
    varied_hot_label: str
    varied_hot_axis_label: str
    varied_hot_column_label: str


CONFIG = AnalysisConfig()
ROUTING_CASES = [
    RoutingCase(
        key="parallel",
        label="Parallel (T13 = T15)",
        folder_name="01_parallel",
        routing_mode="parallel",
        varied_hot_label="T_hot = T13 = T15",
        varied_hot_axis_label="T_hot = T13 = T15 [°C]",
        varied_hot_column_label="T_hot",
    ),
    RoutingCase(
        key="series_desorber_to_evaporator",
        label="Serie Desorber → Verdampfer",
        folder_name="02_series_desorber_to_evaporator",
        routing_mode="series_desorber_to_evaporator",
        varied_hot_label="T13",
        varied_hot_axis_label="T13 [°C]",
        varied_hot_column_label="T13",
    ),
    RoutingCase(
        key="series_evaporator_to_desorber",
        label="Serie Verdampfer → Desorber",
        folder_name="03_series_evaporator_to_desorber",
        routing_mode="series_evaporator_to_desorber",
        varied_hot_label="T15",
        varied_hot_axis_label="T15 [°C]",
        varied_hot_column_label="T15",
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
        "derived_",
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


def build_inputs_for_point(case: RoutingCase, hot_C: float, T17_C: float, cfg: AnalysisConfig):
    base = build_example_inputs()

    kwargs: dict[str, Any] = dict(
        desorber_evaporator_routing_mode=case.routing_mode,
        T_17_C=float(T17_C),
        solver_tol=float(cfg.solver_tol_map),
        max_nfev=int(cfg.max_nfev_map),
    )

    if case.routing_mode == "parallel":
        kwargs["T_13_C"] = float(hot_C)
        kwargs["T_15_C"] = float(hot_C)

    elif case.routing_mode == "series_desorber_to_evaporator":
        kwargs["T_13_C"] = float(hot_C)
        kwargs["T_15_C"] = None
        kwargs["m_13"] = SERIAL_EXTERNAL_MASSFLOW_FACTOR * float(base.m_13)
        kwargs["m_15"] = SERIAL_EXTERNAL_MASSFLOW_FACTOR * float(base.m_15)

    elif case.routing_mode == "series_evaporator_to_desorber":
        kwargs["T_13_C"] = None
        kwargs["T_15_C"] = float(hot_C)
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


def compute_deltaT_min_global(result) -> float:
    return min(_temperature_differences(result).values())


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
    limiting_unit = PINCH_UNIT_FROM_KEY[limiting_key]
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


def evaluate_point(case: RoutingCase, hot_C: float, T17_C: float, *, x0: np.ndarray, cfg: AnalysisConfig) -> tuple[dict[str, Any], np.ndarray | None]:
    inputs = build_inputs_for_point(case, hot_C, T17_C, cfg)
    lower, upper = bounds(inputs)

    row: dict[str, Any] = {
        "config_case_key": case.key,
        "config_case_label": case.label,
        "config_routing_mode": case.routing_mode,
        "input_T17_C": float(T17_C),
        "input_hot_C": float(hot_C),
        "input_hot_label": case.varied_hot_label,
        "input_T13_given_C": float(hot_C) if case.routing_mode in {"parallel", "series_desorber_to_evaporator"} else float("nan"),
        "input_T15_given_C": float(hot_C) if case.routing_mode in {"parallel", "series_evaporator_to_desorber"} else float("nan"),
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
        "pinch_limiting_label": "invalid",
        "pinch_limiting_code": int(PINCH_CODE_INVALID),
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
        failure_code, failure_reason, limiting_unit = classify_failure(
            result,
            deltaT_margin_K=cfg.deltaT_margin_K,
            scaled_residual_tol=cfg.scaled_residual_tol,
        )
        pinch_key, pinch_value, pinch_unit = determine_limiting_pinch(result)
        row["pinch_deltaT_min_global_K"] = float(pinch_value)
        row["pinch_limiting_unit"] = pinch_unit
        row["pinch_limiting_key"] = pinch_key
        row["pinch_limiting_label"] = PINCH_LABELS[pinch_key]
        row["pinch_limiting_code"] = int(PINCH_CODE_MAP[pinch_key])
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

        w1 = _float_or_nan(row.get("state_1_w_LiBr"))
        w6 = _float_or_nan(row.get("state_6_w_LiBr"))
        w3 = _float_or_nan(row.get("state_3_w_LiBr"))
        w4 = _float_or_nan(row.get("state_4_w_LiBr"))
        w20 = _float_or_nan(row.get("state_20_w_LiBr"))

        row["derived_delta_w_H2O_des"] = ((1.0 - w1) - (1.0 - w6)) if math.isfinite(w1) and math.isfinite(w6) else float("nan")
        row["derived_delta_w_H2O_abs_tot"] = ((1.0 - w3) - (1.0 - w4)) if math.isfinite(w3) and math.isfinite(w4) else float("nan")
        row["derived_delta_w_H2O_preabs"] = ((1.0 - w20) - (1.0 - w4)) if math.isfinite(w20) and math.isfinite(w4) else float("nan")
        row["derived_delta_w_H2O_abs"] = ((1.0 - w3) - (1.0 - w4)) if math.isfinite(w3) and math.isfinite(w4) else float("nan")

        for idx, message in enumerate(result.validity_messages, start=1):
            row[f"validity_message_{idx}"] = message
        row["validity_message_count"] = len(result.validity_messages)

        x_next = _primary_vector_from_result(result)
    else:
        row["failure_code"] = FAIL_SOLVER
        row["failure_reason"] = "final_point_not_evaluable"
        row["failure_label"] = FAILURE_LABELS[FAIL_SOLVER]
        row["feasible_numeric"] = False
        row["feasible_physical"] = False
        row["feasible_margin"] = False

    return row, x_next


def _hot_sequence_center_out(values: np.ndarray, reference_value: float) -> tuple[list[float], list[float]]:
    vals = [float(v) for v in values]
    left = [v for v in vals if v < reference_value]
    right = [v for v in vals if v > reference_value]
    return left[::-1], right


def _t17_offsets(values: np.ndarray, reference_value: float) -> list[int]:
    vals = [int(round(v - reference_value)) for v in values if not math.isclose(float(v), reference_value)]
    positive = sorted([v for v in vals if v > 0])
    negative = sorted([v for v in vals if v < 0], reverse=True)
    merged: list[int] = []
    idx = 0
    while idx < max(len(positive), len(negative)):
        if idx < len(positive):
            merged.append(positive[idx])
        if idx < len(negative):
            merged.append(negative[idx])
        idx += 1
    return merged


def _choose_start_from_store(store: dict[tuple[float, float], np.ndarray], hot_C: float, T17_C: float) -> np.ndarray | None:
    key = (float(T17_C), float(hot_C))
    value = store.get(key)
    if value is None:
        return None
    return np.array(value, dtype=float, copy=True)

def _solved_hot_values_for_T17(store: dict[tuple[float, float], np.ndarray], T17_C: float) -> list[float]:
    values = []
    for t17, hot in store.keys():
        if math.isclose(float(t17), float(T17_C), rel_tol=0.0, abs_tol=1.0e-12):
            values.append(float(hot))
    values.sort()
    return values


def _find_first_available_hot_in_order(
    store: dict[tuple[float, float], np.ndarray],
    source_T17_C: float,
    ordered_hots: list[float],
) -> float | None:
    for hot_C in ordered_hots:
        if _choose_start_from_store(store, hot_C, source_T17_C) is not None:
            return float(hot_C)
    return None


def _evaluate_and_store(
    *,
    case: RoutingCase,
    hot_C: float,
    T17_C: float,
    x0: np.ndarray,
    cfg: AnalysisConfig,
    rows: list[dict[str, Any]],
    solved_store: dict[tuple[float, float], np.ndarray],
    verbose_prefix: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if cfg.print_attempts:
        print(f"    löse Punkt {case.varied_hot_label}={hot_C:6.2f} °C, T17={T17_C:6.2f} °C | start={verbose_prefix}")
    row, x_next = evaluate_point(case, hot_C, T17_C, x0=x0, cfg=cfg)
    rows.append(row)
    if x_next is not None:
        solved_store[(float(T17_C), float(hot_C))] = np.array(x_next, dtype=float, copy=True)
    marker = "OK+" if bool(row["feasible_margin"]) else ("OK" if bool(row["feasible_physical"]) else "--")
    cop = _float_or_nan(row.get("kpi_COP", float("nan")))
    q_abs = _float_or_nan(row.get("heat_Q_abs_kW", float("nan")))
    dtmin = _float_or_nan(row.get("pinch_deltaT_min_global_K", float("nan")))
    print(
        f"  {case.varied_hot_column_label}={hot_C:6.2f} °C | {marker:>3s} | "
        f"COP={cop:>8.4f} | Q_abs={q_abs:>9.4f} kW | ΔTmin={dtmin:>8.4f} K | "
        f"reason={row['failure_reason']}"
    )
    return row, x_next


def _run_reference_row(
    case: RoutingCase,
    cfg: AnalysisConfig,
    rows: list[dict[str, Any]],
    solved_store: dict[tuple[float, float], np.ndarray],
) -> np.ndarray:
    reference_T17 = float(cfg.reference_T17_C)
    reference_hot = float(cfg.reference_hot_C)
    reference_x0 = build_reference_x0()

    print("-" * 110)
    print(f"Referenzpunkt: T17={reference_T17:.2f} °C, {case.varied_hot_label}={reference_hot:.2f} °C")
    _, x_ref = _evaluate_and_store(
        case=case,
        hot_C=reference_hot,
        T17_C=reference_T17,
        x0=reference_x0,
        cfg=cfg,
        rows=rows,
        solved_store=solved_store,
        verbose_prefix="reference_x0",
    )
    if x_ref is None:
        raise RuntimeError(
            f"Referenzpunkt für Konfiguration '{case.key}' konnte nicht physikalisch auswertbar gelöst werden."
        )

    left_values_desc, right_values_asc = _hot_sequence_center_out(cfg.hot_values_C, reference_hot)

    prev_valid_right = np.array(x_ref, dtype=float, copy=True)
    for hot_C in right_values_asc:
        _, x_next = _evaluate_and_store(
            case=case,
            hot_C=float(hot_C),
            T17_C=reference_T17,
            x0=prev_valid_right,
            cfg=cfg,
            rows=rows,
            solved_store=solved_store,
            verbose_prefix="previous_valid_reference_row_right",
        )
        if x_next is not None:
            prev_valid_right = np.array(x_next, dtype=float, copy=True)

    prev_valid_left = np.array(x_ref, dtype=float, copy=True)
    for hot_C in left_values_desc:
        _, x_next = _evaluate_and_store(
            case=case,
            hot_C=float(hot_C),
            T17_C=reference_T17,
            x0=prev_valid_left,
            cfg=cfg,
            rows=rows,
            solved_store=solved_store,
            verbose_prefix="previous_valid_reference_row_left",
        )
        if x_next is not None:
            prev_valid_left = np.array(x_next, dtype=float, copy=True)

    return np.array(x_ref, dtype=float, copy=True)


def _run_branch_row(
    *,
    case: RoutingCase,
    cfg: AnalysisConfig,
    rows: list[dict[str, Any]],
    solved_store: dict[tuple[float, float], np.ndarray],
    T17_C: float,
    source_T17_C: float,
    direction: str,
    branch_label: str,
) -> None:
    hot_values = [float(v) for v in cfg.hot_values_C]
    if direction == "forward":
        ordered_hots = hot_values
        direction_label = "40→80"
    elif direction == "reverse":
        ordered_hots = hot_values[::-1]
        direction_label = "80→40"
    else:
        raise ValueError(f"Ungültige Richtung: {direction}")

    anchor_hot = _find_first_available_hot_in_order(solved_store, source_T17_C, ordered_hots)
    if anchor_hot is None:
        solved_source_hots = _solved_hot_values_for_T17(solved_store, source_T17_C)
        raise RuntimeError(
            f"Kein vertikaler Startwert für Zeile T17={T17_C:.2f} °C verfügbar. "
            f"In Quellzeile T17={source_T17_C:.2f} °C existieren keine gelösten Hot-Werte. "
            f"Gelöste Hot-Werte dort: {solved_source_hots}"
        )

    start_x0 = _choose_start_from_store(solved_store, anchor_hot, source_T17_C)
    if start_x0 is None:
        raise RuntimeError(
            f"Interner Fehler: Anker-Hot {anchor_hot:.2f} °C in Quellzeile T17={source_T17_C:.2f} °C nicht verfügbar."
        )

    anchor_idx = ordered_hots.index(float(anchor_hot))
    primary_segment = ordered_hots[anchor_idx:]
    secondary_segment = list(reversed(ordered_hots[:anchor_idx]))

    print("-" * 110)
    print(
        f"Zeile T17={T17_C:.2f} °C | Branch={branch_label} | Richtung={direction_label} | "
        f"vertikaler Start von T17={source_T17_C:.2f} °C bei Hot={anchor_hot:.2f} °C"
    )

    anchor_solution_current_row = None
    prev_valid = np.array(start_x0, dtype=float, copy=True)
    for idx, hot_C in enumerate(primary_segment):
        verbose = (
            f"vertical_same_hot_from_T17_{source_T17_C:.2f}" if idx == 0 else "previous_valid_same_row"
        )
        _, x_next = _evaluate_and_store(
            case=case,
            hot_C=float(hot_C),
            T17_C=float(T17_C),
            x0=prev_valid,
            cfg=cfg,
            rows=rows,
            solved_store=solved_store,
            verbose_prefix=verbose,
        )
        if idx == 0 and x_next is not None:
            anchor_solution_current_row = np.array(x_next, dtype=float, copy=True)
        if x_next is not None:
            prev_valid = np.array(x_next, dtype=float, copy=True)

    if secondary_segment and anchor_solution_current_row is not None:
        prev_valid = np.array(anchor_solution_current_row, dtype=float, copy=True)
        for hot_C in secondary_segment:
            _, x_next = _evaluate_and_store(
                case=case,
                hot_C=float(hot_C),
                T17_C=float(T17_C),
                x0=prev_valid,
                cfg=cfg,
                rows=rows,
                solved_store=solved_store,
                verbose_prefix="previous_valid_same_row_secondary_branch",
            )
            if x_next is not None:
                prev_valid = np.array(x_next, dtype=float, copy=True)


def run_case(case: RoutingCase, cfg: AnalysisConfig) -> tuple[list[dict[str, Any]], Path]:
    if float(cfg.reference_hot_C) not in {float(v) for v in cfg.hot_values_C}:
        raise ValueError("reference_hot_C muss im Hot-Raster enthalten sein.")
    if float(cfg.reference_T17_C) not in {float(v) for v in cfg.T17_values_C}:
        raise ValueError("reference_T17_C muss im T17-Raster enthalten sein.")

    case_dir = cfg.output_root_dir / case.folder_name
    _ensure_dir(case_dir)

    rows: list[dict[str, Any]] = []
    solved_store: dict[tuple[float, float], np.ndarray] = {}

    print("=" * 110)
    print(f"Auswertung Konfiguration: {case.label}")
    print(f"Routing-Mode: {case.routing_mode}")
    print(
        f"Raster: {case.varied_hot_label} {float(np.min(cfg.hot_values_C)):.1f} bis {float(np.max(cfg.hot_values_C)):.1f} °C, "
        f"T17 {float(np.min(cfg.T17_values_C)):.1f} bis {float(np.max(cfg.T17_values_C)):.1f} °C"
    )
    print(
        f"Referenzpunkt: T17={cfg.reference_T17_C:.1f} °C, {case.varied_hot_label}={cfg.reference_hot_C:.1f} °C"
    )
    print("Ablauf: Referenzpunkt -> Referenzzeile -> Expansion oberhalb/unterhalb -> snake je Zeile")
    print("Keine Notfall-Fallbacks mit initial_guess() oder weiteren alternativen Startwerten.")
    print("=" * 110)

    _run_reference_row(case, cfg, rows, solved_store)

    upper_values = [float(v) for v in cfg.T17_values_C if float(v) > float(cfg.reference_T17_C)]
    lower_values = [float(v) for v in cfg.T17_values_C if float(v) < float(cfg.reference_T17_C)]
    upper_values.sort()
    lower_values.sort(reverse=True)

    max_len = max(len(upper_values), len(lower_values))
    previous_upper_T17 = float(cfg.reference_T17_C)
    previous_lower_T17 = float(cfg.reference_T17_C)

    for idx in range(max_len):
        if idx < len(upper_values):
            T17_upper = upper_values[idx]
            direction_upper = "reverse" if idx % 2 == 0 else "forward"
            _run_branch_row(
                case=case,
                cfg=cfg,
                rows=rows,
                solved_store=solved_store,
                T17_C=T17_upper,
                source_T17_C=previous_upper_T17,
                direction=direction_upper,
                branch_label="upper",
            )
            previous_upper_T17 = T17_upper

        if idx < len(lower_values):
            T17_lower = lower_values[idx]
            direction_lower = "forward" if idx % 2 == 0 else "reverse"
            _run_branch_row(
                case=case,
                cfg=cfg,
                rows=rows,
                solved_store=solved_store,
                T17_C=T17_lower,
                source_T17_C=previous_lower_T17,
                direction=direction_lower,
                branch_label="lower",
            )
            previous_lower_T17 = T17_lower

    rows_sorted = sorted(rows, key=lambda row: (float(row["input_T17_C"]), float(row["input_hot_C"])))
    _save_csv(rows_sorted, case_dir / CSV_FILENAME)
    _write_summary(rows_sorted, case, cfg, case_dir / SUMMARY_FILENAME)
    create_case_plots(rows_sorted, case, cfg, case_dir)
    print(f"CSV gespeichert unter: {case_dir / CSV_FILENAME}")
    print(f"Plots gespeichert unter: {case_dir}")
    return rows_sorted, case_dir


def _write_summary(rows: list[dict[str, Any]], case: RoutingCase, cfg: AnalysisConfig, filepath: Path) -> None:
    total = len(rows)
    feasible_numeric = sum(bool(row.get("feasible_numeric", False)) for row in rows)
    feasible_physical = sum(bool(row.get("feasible_physical", False)) for row in rows)
    feasible_margin = sum(bool(row.get("feasible_margin", False)) for row in rows)
    lines = [
        f"Konfiguration: {case.label}",
        "=" * 72,
        f"Hot-Bereich: {float(np.min(cfg.hot_values_C)):.1f} bis {float(np.max(cfg.hot_values_C)):.1f} °C",
        f"T17-Bereich: {float(np.min(cfg.T17_values_C)):.1f} bis {float(np.max(cfg.T17_values_C)):.1f} °C",
        f"Referenzpunkt: T17={cfg.reference_T17_C:.1f} °C, {case.varied_hot_label}={cfg.reference_hot_C:.1f} °C",
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


def _rows_to_grid(rows: list[dict[str, Any]], cfg: AnalysisConfig, value_key: str, *, fill_value: float = np.nan) -> np.ndarray:
    x_vals = [float(v) for v in cfg.hot_values_C]
    y_vals = [float(v) for v in cfg.T17_values_C]
    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}
    grid = np.full((len(y_vals), len(x_vals)), fill_value, dtype=float)
    for row in rows:
        x = float(row["input_hot_C"])
        y = float(row["input_T17_C"])
        if x in x_index and y in y_index:
            try:
                grid[y_index[y], x_index[x]] = float(row.get(value_key, fill_value))
            except Exception:
                grid[y_index[y], x_index[x]] = fill_value
    return grid


def _extent(cfg: AnalysisConfig) -> list[float]:
    return [
        float(np.min(cfg.hot_values_C)),
        float(np.max(cfg.hot_values_C)),
        float(np.min(cfg.T17_values_C)),
        float(np.max(cfg.T17_values_C)),
    ]


def _plot_common(case: RoutingCase, title: str) -> None:
    plt.xlabel(case.varied_hot_axis_label)
    plt.ylabel("T17 [°C]")
    plt.title(title)


def _plot_lines_over_hot(rows: list[dict[str, Any]], case: RoutingCase, cfg: AnalysisConfig, value_key: str, ylabel: str, filepath: Path) -> None:
    plt.figure(figsize=(8.5, 6.0))
    for T17_C in SELECTED_T17_CURVES:
        subset = [row for row in rows if math.isclose(float(row["input_T17_C"]), T17_C)]
        subset.sort(key=lambda row: float(row["input_hot_C"]))
        xs = [float(row["input_hot_C"]) for row in subset]
        ys = [
            float(row[value_key]) if bool(row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(row.get(value_key))) else float("nan")
            for row in subset
        ]
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, label=f"T17={T17_C:.0f} °C")
    plt.xlabel(case.varied_hot_axis_label)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} über {case.varied_hot_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _plot_lines_over_T17(rows: list[dict[str, Any]], case: RoutingCase, cfg: AnalysisConfig, value_key: str, ylabel: str, filepath: Path) -> None:
    plt.figure(figsize=(8.5, 6.0))
    for hot_C in SELECTED_HOT_CURVES:
        subset = [row for row in rows if math.isclose(float(row["input_hot_C"]), hot_C)]
        subset.sort(key=lambda row: float(row["input_T17_C"]))
        xs = [float(row["input_T17_C"]) for row in subset]
        ys = [
            float(row[value_key]) if bool(row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(row.get(value_key))) else float("nan")
            for row in subset
        ]
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, label=f"{case.varied_hot_column_label}={hot_C:.0f} °C")
    plt.xlabel("T17 [°C]")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} über T17")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _plot_heatmap(rows: list[dict[str, Any]], case: RoutingCase, cfg: AnalysisConfig, value_key: str, title: str, cbar_label: str, filepath: Path, *, mask_to_feasible: bool = True) -> None:
    grid = _rows_to_grid(rows, cfg, value_key)
    feasible_grid = _rows_to_grid(rows, cfg, "feasible_physical", fill_value=0.0)
    if mask_to_feasible:
        grid = np.where(feasible_grid > 0.5, grid, np.nan)
    masked = np.ma.masked_invalid(grid)
    plt.figure(figsize=(8.0, 6.0))
    image = plt.imshow(masked, origin="lower", aspect="auto", extent=_extent(cfg))
    plt.colorbar(image, label=cbar_label)
    X, Y = np.meshgrid(cfg.hot_values_C, cfg.T17_values_C)
    plt.contour(X, Y, feasible_grid, levels=[0.5], linewidths=1.0)
    _plot_common(case, title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _plot_pinch_id(rows: list[dict[str, Any]], case: RoutingCase, cfg: AnalysisConfig, filepath: Path) -> None:
    grid = _rows_to_grid(rows, cfg, "pinch_limiting_code", fill_value=float(PINCH_CODE_INVALID))
    labels = ["invalid"] + [PINCH_LABELS[key] for key in PINCH_LABELS]
    cmap = ListedColormap([
        "#000000",
        "#8c6d31",
        "#c49c94",
        "#ff7f0e",
        "#ffbb78",
        "#e377c2",
        "#f7b6d2",
        "#1f77b4",
        "#aec7e8",
        "#9467bd",
        "#c5b0d5",
    ])
    norm = BoundaryNorm(np.arange(-0.5, len(labels) + 0.5, 1.0), cmap.N)
    plt.figure(figsize=(9.0, 6.5))
    image = plt.imshow(grid, origin="lower", aspect="auto", extent=_extent(cfg), cmap=cmap, norm=norm)
    cbar = plt.colorbar(image, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    _plot_common(case, "Pinch-ID-Karte")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _get_reference_row(rows: list[dict[str, Any]], *, hot_C: float, T17_C: float) -> dict[str, Any] | None:
    matches = [
        row for row in rows
        if math.isclose(float(row["input_hot_C"]), float(hot_C))
        and math.isclose(float(row["input_T17_C"]), float(T17_C))
        and bool(row.get("feasible_physical", False))
    ]
    return matches[0] if matches else None


DUHRING_STATE_IDS = ["1", "6", "20", "3", "10", "8", "4"]
DUHRING_REFRIGERANT_PATH = ["3", "10", "8", "1"]
DUHRING_SOLUTION_PATH = ["1", "6", "20", "3", "1"]


def _duhring_state_values(row: dict[str, Any], state_id: str) -> tuple[float, float, float] | None:
    T_C = _float_or_nan(row.get(f"state_{state_id}_T_C"))
    p_kPa = _float_or_nan(row.get(f"state_{state_id}_p_Pa")) / 1000.0
    w = _float_or_nan(row.get(f"state_{state_id}_w_LiBr"))
    if not (math.isfinite(T_C) and math.isfinite(p_kPa) and math.isfinite(w)):
        return None
    return T_C, p_kPa, w


def _plot_duhring_connection(
    row: dict[str, Any],
    state_ids: list[str],
    *,
    color: str,
    linewidth: float,
) -> bool:
    coords: list[tuple[float, float]] = []
    for state_id in state_ids:
        values = _duhring_state_values(row, state_id)
        if values is None:
            return False
        T_C, p_kPa, _ = values
        coords.append((T_C, p_kPa))

    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    plt.plot(xs, ys, color=color, linestyle="-", linewidth=linewidth)
    return True


def _duhring_mass_fraction_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for state_id in DUHRING_STATE_IDS:
        values = _duhring_state_values(row, state_id)
        if values is not None:
            labels.append(f"w{state_id}={values[2]:.4f}")
    return labels


def _add_duhring_series_to_plot(
    row: dict[str, Any],
    *,
    color: str,
    case_label: str,
) -> list[Any]:
    from matplotlib.lines import Line2D

    handles: list[Any] = []
    values_available = [_duhring_state_values(row, state_id) for state_id in DUHRING_STATE_IDS]
    if not all(value is not None for value in values_available):
        return handles

    _plot_duhring_connection(
        row,
        DUHRING_SOLUTION_PATH,
        color=color,
        linewidth=1.6,
    )
    _plot_duhring_connection(
        row,
        DUHRING_REFRIGERANT_PATH,
        color=color,
        linewidth=1.6,
    )

    xs = []
    ys = []
    for state_id in DUHRING_STATE_IDS:
        T_C, p_kPa, _ = _duhring_state_values(row, state_id)  # type: ignore[misc]
        xs.append(T_C)
        ys.append(p_kPa)
    plt.plot(xs, ys, marker="o", linestyle="None", color=color)

    for state_id in DUHRING_STATE_IDS:
        T_C, p_kPa, _ = _duhring_state_values(row, state_id)  # type: ignore[misc]
        plt.annotate(
            state_id,
            xy=(T_C, p_kPa),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )

    handles.append(Line2D([], [], color=color, marker="o", linewidth=1.6, label=case_label))
    for label in _duhring_mass_fraction_labels(row):
        handles.append(Line2D([], [], linestyle="None", color=color, label=label))

    return handles


def _finalize_duhring_plot(
    *,
    title: str,
    filepath: Path,
    legend_handles: list[Any],
    plotted_any: bool,
    fallback_text: str,
) -> None:
    fig = plt.gcf()
    ax = plt.gca()

    ax.set_xlabel("Temperatur [°C]")
    ax.set_ylabel("Druck [kPa]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if plotted_any and legend_handles:
        # Plotfläche links bewusst etwas schmaler setzen
        ax.set_position([0.08, 0.12, 0.58, 0.78])

        # Eigene Achse nur für die Legende rechts
        ax_leg = fig.add_axes([0.70, 0.12, 0.27, 0.78])
        ax_leg.axis("off")
        ax_leg.legend(
            handles=legend_handles,
            loc="center left",
            frameon=True,
            borderaxespad=0.0,
            handlelength=2.0,
            handletextpad=0.8,
            labelspacing=0.6,
        )
    else:
        # Ohne Legende den Plot mittiger und breiter setzen
        ax.set_position([0.10, 0.12, 0.82, 0.78])
        ax.text(
            0.5,
            0.5,
            fallback_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def _case_plot_duhring_solution_cycle(rows: list[dict[str, Any]], case: RoutingCase, filepath: Path) -> None:
    plt.figure(figsize=(9.0, 6.5))
    row = _get_reference_row(rows, hot_C=COMPARISON_HOT_INPUT_C, T17_C=COMPARISON_T17_C)
    legend_handles: list[Any] = []
    plotted_any = False
    if row is not None:
        legend_handles = _add_duhring_series_to_plot(row, color="tab:blue", case_label=case.label)
        plotted_any = bool(legend_handles)

    _finalize_duhring_plot(
        title=(
            "Dühring-Plot des Kreislaufs\n"
            f"{case.label} bei T17={COMPARISON_T17_C:.0f} °C und {case.varied_hot_label}={COMPARISON_HOT_INPUT_C:.0f} °C"
        ),
        filepath=filepath,
        legend_handles=legend_handles,
        plotted_any=plotted_any,
        fallback_text=(
            "Kein physikalisch zulässiger Punkt gefunden\n"
            f"für T17={COMPARISON_T17_C:.0f} °C und {case.varied_hot_label}={COMPARISON_HOT_INPUT_C:.0f} °C."
        ),
    )


def create_case_plots(rows: list[dict[str, Any]], case: RoutingCase, cfg: AnalysisConfig, case_dir: Path) -> None:
    _plot_lines_over_hot(rows, case, cfg, "kpi_COP", "COP [-]", case_dir / "plot_COP_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "heat_Q_abs_kW", "Q_abs [kW]", case_dir / "plot_Qabs_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "kpi_FR", "FR [-]", case_dir / "plot_FR_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "state_21_m_kg_s", "m21 [kg/s]", case_dir / "plot_m21_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "state_7_m_kg_s", "m7 [kg/s]", case_dir / "plot_m7_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "primary_beta", "beta [-]", case_dir / "plot_beta_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "derived_delta_w_H2O_des", "Δw_H2O,des [-]", case_dir / "plot_delta_w_H2O_des_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "derived_delta_w_H2O_abs_tot", "Δw_H2O,abs_tot [-]", case_dir / "plot_delta_w_H2O_abs_tot_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "derived_delta_w_H2O_preabs", "Δw_H2O,preabs [-]", case_dir / "plot_delta_w_H2O_preabs_vs_hot_selected_T17.png")
    _plot_lines_over_hot(rows, case, cfg, "derived_delta_w_H2O_abs", "Δw_H2O,abs [-]", case_dir / "plot_delta_w_H2O_abs_vs_hot_selected_T17.png")

    _plot_lines_over_T17(rows, case, cfg, "kpi_COP", "COP [-]", case_dir / "plot_COP_vs_T17_selected_hot.png")
    _plot_lines_over_T17(rows, case, cfg, "heat_Q_abs_kW", "Q_abs [kW]", case_dir / "plot_Qabs_vs_T17_selected_hot.png")
    _plot_lines_over_T17(rows, case, cfg, "kpi_FR", "FR [-]", case_dir / "plot_FR_vs_T17_selected_hot.png")

    _plot_heatmap(rows, case, cfg, "kpi_COP", "COP-Heatmap", "COP [-]", case_dir / "map_COP.png")
    _plot_heatmap(rows, case, cfg, "pinch_deltaT_min_global_K", "Heatmap ΔT_min", "ΔT_min [K]", case_dir / "map_deltaT_min.png", mask_to_feasible=False)
    _plot_heatmap(rows, case, cfg, "heat_Q_abs_kW", "Heatmap Q_abs", "Q_abs [kW]", case_dir / "map_Qabs.png")
    _plot_pinch_id(rows, case, cfg, case_dir / "map_pinch_id.png")
    _case_plot_duhring_solution_cycle(rows, case, case_dir / "plot_duhring_solution_cycle_at_comparison_point.png")


def _comparison_plot_over_hot(case_rows: dict[str, list[dict[str, Any]]], value_key: str, ylabel: str, filepath: Path) -> None:
    plt.figure(figsize=(8.5, 6.0))
    for case in ROUTING_CASES:
        rows = [
            row for row in case_rows[case.key]
            if math.isclose(float(row["input_T17_C"]), COMPARISON_T17_C)
        ]
        rows.sort(key=lambda row: float(row["input_hot_C"]))
        xs = [float(row["input_hot_C"]) for row in rows]
        ys = [
            float(row[value_key]) if bool(row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(row.get(value_key))) else float("nan")
            for row in rows
        ]
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, label=case.label)
    plt.xlabel("vorgegebener Hot-Eingang [°C]")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} über Hot bei T17={COMPARISON_T17_C:.0f} °C")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _comparison_plot_over_T17(case_rows: dict[str, list[dict[str, Any]]], value_key: str, ylabel: str, filepath: Path) -> None:
    plt.figure(figsize=(8.5, 6.0))
    for case in ROUTING_CASES:
        rows = [
            row for row in case_rows[case.key]
            if math.isclose(float(row["input_hot_C"]), COMPARISON_HOT_INPUT_C)
        ]
        rows.sort(key=lambda row: float(row["input_T17_C"]))
        xs = [float(row["input_T17_C"]) for row in rows]
        ys = [
            float(row[value_key]) if bool(row.get("feasible_physical", False)) and math.isfinite(_float_or_nan(row.get(value_key))) else float("nan")
            for row in rows
        ]
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, label=case.label)
    plt.xlabel("T17 [°C]")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} über T17 bei Hot={COMPARISON_HOT_INPUT_C:.0f} °C")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _comparison_plot_cop_over_fr(case_rows: dict[str, list[dict[str, Any]]], filepath: Path) -> None:
    plt.figure(figsize=(7.5, 6.0))
    for case in ROUTING_CASES:
        matches = [
            row for row in case_rows[case.key]
            if math.isclose(float(row["input_hot_C"]), COMPARISON_HOT_INPUT_C)
            and math.isclose(float(row["input_T17_C"]), COMPARISON_T17_C)
            and bool(row.get("feasible_physical", False))
        ]
        if not matches:
            continue
        row = matches[0]
        fr = _float_or_nan(row.get("kpi_FR"))
        cop = _float_or_nan(row.get("kpi_COP"))
        plt.plot([fr], [cop], marker="o", linestyle="None", markersize=8, label=case.label)
    plt.xlabel("FR [-]")
    plt.ylabel("COP [-]")
    plt.title(f"COP über FR bei Hot={COMPARISON_HOT_INPUT_C:.0f} °C und T17={COMPARISON_T17_C:.0f} °C")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def _comparison_plot_duhring_solution_cycle(case_rows: dict[str, list[dict[str, Any]]], filepath: Path) -> None:
    plt.figure(figsize=(9.0, 6.5))

    serial_cases = [
        ("series_desorber_to_evaporator", "tab:blue"),
        ("series_evaporator_to_desorber", "tab:red"),
    ]

    legend_handles: list[Any] = []
    plotted_any = False
    for case_key, color in serial_cases:
        row = _get_reference_row(
            case_rows.get(case_key, []),
            hot_C=COMPARISON_HOT_INPUT_C,
            T17_C=COMPARISON_T17_C,
        )
        if row is None:
            continue

        case_label = CASE_BY_KEY[case_key].label
        handles = _add_duhring_series_to_plot(
            row,
            color=color,
            case_label=case_label,
        )
        if handles:
            legend_handles.extend(handles)
            plotted_any = True

    _finalize_duhring_plot(
        title=(
            "Dühring-Plot des Kreislaufs\n"
            f"beide Serienkonfigurationen bei T17={COMPARISON_T17_C:.0f} °C und T13 bzw. T15={COMPARISON_HOT_INPUT_C:.0f} °C"
        ),
        filepath=filepath,
        legend_handles=legend_handles,
        plotted_any=plotted_any,
        fallback_text=(
            "Keine physikalisch zulässigen Vergleichspunkte gefunden\n"
            f"für T17={COMPARISON_T17_C:.0f} °C und T13 bzw. T15={COMPARISON_HOT_INPUT_C:.0f} °C."
        ),
    )


def create_comparison_plots(case_rows: dict[str, list[dict[str, Any]]], cfg: AnalysisConfig) -> None:
    comparison_dir = cfg.output_root_dir / COMPARISON_DIRNAME
    _ensure_dir(comparison_dir)
    _comparison_plot_over_hot(case_rows, "kpi_COP", "COP [-]", comparison_dir / "comparison_COP_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_T17(case_rows, "kpi_COP", "COP [-]", comparison_dir / "comparison_COP_vs_T17_at_comparison_hot.png")
    _comparison_plot_over_hot(case_rows, "heat_Q_abs_kW", "Q_abs [kW]", comparison_dir / "comparison_Qabs_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_T17(case_rows, "heat_Q_abs_kW", "Q_abs [kW]", comparison_dir / "comparison_Qabs_vs_T17_at_comparison_hot.png")
    _comparison_plot_over_hot(case_rows, "heat_Q_cond_kW", "Q_cond [kW]", comparison_dir / "comparison_Qcond_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_T17(case_rows, "heat_Q_cond_kW", "Q_cond [kW]", comparison_dir / "comparison_Qcond_vs_T17_at_comparison_hot.png")
    _comparison_plot_over_hot(case_rows, "heat_Q_evap_kW", "Q_evap [kW]", comparison_dir / "comparison_Qevap_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_T17(case_rows, "heat_Q_evap_kW", "Q_evap [kW]", comparison_dir / "comparison_Qevap_vs_T17_at_comparison_hot.png")
    _comparison_plot_over_hot(case_rows, "heat_Q_des_kW", "Q_des [kW]", comparison_dir / "comparison_Qdes_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_T17(case_rows, "heat_Q_des_kW", "Q_des [kW]", comparison_dir / "comparison_Qdes_vs_T17_at_comparison_hot.png")
    _comparison_plot_over_hot(case_rows, "kpi_FR", "FR [-]", comparison_dir / "comparison_FR_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_T17(case_rows, "kpi_FR", "FR [-]", comparison_dir / "comparison_FR_vs_T17_at_comparison_hot.png")
    _comparison_plot_over_hot(case_rows, "state_21_m_kg_s", "m21 [kg/s]", comparison_dir / "comparison_m21_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_hot(case_rows, "state_7_m_kg_s", "m7 [kg/s]", comparison_dir / "comparison_m7_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_hot(case_rows, "primary_beta", "beta [-]", comparison_dir / "comparison_beta_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_hot(case_rows, "derived_delta_w_H2O_des", "Δw_H2O,des [-]", comparison_dir / "comparison_delta_w_H2O_des_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_hot(case_rows, "derived_delta_w_H2O_abs_tot", "Δw_H2O,abs_tot [-]", comparison_dir / "comparison_delta_w_H2O_abs_tot_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_hot(case_rows, "derived_delta_w_H2O_preabs", "Δw_H2O,preabs [-]", comparison_dir / "comparison_delta_w_H2O_preabs_vs_hot_at_comparison_T17.png")
    _comparison_plot_over_hot(case_rows, "derived_delta_w_H2O_abs", "Δw_H2O,abs [-]", comparison_dir / "comparison_delta_w_H2O_abs_vs_hot_at_comparison_T17.png")
    _comparison_plot_cop_over_fr(case_rows, comparison_dir / "comparison_COP_vs_FR_at_comparison_point.png")
    _comparison_plot_duhring_solution_cycle(case_rows, comparison_dir / "comparison_duhring_solution_cycle_at_comparison_point.png")


def main() -> dict[str, list[dict[str, Any]]]:
    _ensure_dir(CONFIG.output_root_dir)
    case_rows: dict[str, list[dict[str, Any]]] = {}
    for case in ROUTING_CASES:
        rows, _ = run_case(case, CONFIG)
        case_rows[case.key] = rows
    create_comparison_plots(case_rows, CONFIG)
    print("=" * 110)
    print("Alle Konfigurationen abgeschlossen.")
    print(f"Ausgabeordner: {CONFIG.output_root_dir}")
    print("=" * 110)
    return case_rows


if __name__ == "__main__":
    main()
