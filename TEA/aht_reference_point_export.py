"""Exports the AWT (winter heat-transformer) reference operating point as
an editable JSON, mirroring design_point_export.py's ac_design_point.json
-- same numbered-temperature convention (T11-T18) as the model itself and
as the AC side, instead of the earlier T_drive_C/T_cond_C naming.

Unlike the AC side, there is no independent UA optimization here (the "one
design point" premise: the AWT reuses the AC's UA set 1:1, by component
name -- UA_cond(AWT)=UA_cond(AC) etc., see the chat discussion). What IS
calibrated here is which external mass flows (m11, m13, m17) make the
GIVEN temperature targets (T11->T12 absorber, T13->T16 drive pair,
T17->T18 condenser) and the given absorber duty self-consistent with that
UA set -- via calibrate_reference_flows(), which inverts the targets
instead of guessing the flows directly.

To change the winter operating point: edit "boundary_conditions" and
"targets" in the exported JSON (or the AHTReferencePointConfig defaults
below) and re-run this script. "UA_kW_K", "reference_flows_kg_s" and
"resulting" are outputs, not inputs -- don't hand-edit those.

Why Qabs_reference_kW defaults to 600, not a smaller number matching the
winter use case more closely: this UA set was optimized for 500 kW of AC
cooling, i.e. it is quite large. Direct testing (random multi-start,
hundreds of tries) showed that a much smaller target duty (e.g. the
150 kW first discussed) is THERMODYNAMICALLY INFEASIBLE with this specific
UA set at sensible temperature levels -- not a solver issue, confirmed via
the free (not AC-derived) pinch model, which finds 150 kW perfectly
feasible with a ~4-5x SMALLER, purpose-fit UA set. This hardware, sized
for 500 kW of summer cooling, cannot be throttled down to 150 kW of winter
heat transformation at these temperatures -- it wants to run much closer
to its own scale (confirmed feasible: 300-700 kW; 800 kW already failed).
600 kW is a round number safely inside that confirmed-feasible window.

Standalone usage
-----------------
    python TEA/aht_reference_point_export.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AHT_UA_LMTD import (
    AHTInputs,
    PRIMARY_VARIABLE_NAMES,
    bounds as aht_bounds,
    initial_guess as aht_initial_guess,
    solve_aht,
)
from TEA.design_point_export import DEFAULT_EXPORT_PATH as AC_DESIGN_POINT_PATH, load_design_point

RESIDUAL_TOL = 1.0e-6
DEFAULT_AHT_REFERENCE_PATH = Path(__file__).resolve().parent / "design_point" / "aht_reference_point.json"


@dataclass
class AHTReferencePointConfig:
    # --- External inlet temperatures [°C] -- EDIT HERE ---
    T_11_C: float = 80.0      # Netzruecklauf into the absorber
    T_13_C: float = 75.0      # geothermal "drive" feeding desorber+evaporator
    T_17_C: float = 20.0      # ambient reject into the condenser

    # --- Target external outlet temperatures [°C] -- EDIT HERE ---
    T12_target_C: float = 90.0     # absorber outlet (Netzvorlauf)
    T16_target_C: float = 63.15    # drive-pair final outlet (desorber->evaporator, series)
    T18_target_C: float = 23.92    # condenser outlet

    # --- Target absorber duty [kW] -- EDIT HERE ---
    Qabs_reference_kW: float = 600.0

    desorber_evaporator_routing_mode: str = "series_desorber_to_evaporator"
    cp_w_kJkgK: float = 4.18

    # Calibration solver settings
    max_iter: int = 15
    tol_K: float = 0.02
    solver_tol: float = 1.0e-9
    max_nfev: int = 5000


def _routing_temperature_kwargs(routing_mode: str, T_13_C: float) -> dict:
    """T_13_C/T_15_C for the given routing_mode, with T_13_C as the ONE
    fixed geothermal drive temperature: for "parallel" it feeds both
    independent streams; for a series mode it is the inlet of whichever
    component comes FIRST in that chain, the other temperature is None
    (computed internally, T15=T14 or T13=T16). Mirrors
    ac_characteristic_map.py's _routing_temperature_kwargs. Shared with
    aht_characteristic_map.py (imported from here).
    """
    if routing_mode == "series_desorber_to_evaporator":
        return dict(T_13_C=T_13_C, T_15_C=None)
    if routing_mode == "series_evaporator_to_desorber":
        return dict(T_13_C=None, T_15_C=T_13_C)
    return dict(T_13_C=T_13_C, T_15_C=T_13_C)


def _is_valid(result) -> bool:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def calibrate_reference_flows(ua: dict, config: AHTReferencePointConfig, *, verbose: bool = True) -> dict:
    """Backs out m11/m13/m17 reference flows from config's given external
    temperature targets + target absorber duty, instead of guessing them.

    m11 follows directly from the absorber's own energy balance
    (Qabs/(cp*(T12-T11))) -- no iteration needed, since absorber_spec_mode
    ="m11" with Qabs targeted via cycle_scale_spec_mode="Qabs" makes this
    exact by construction. m13(=m15) and m17 need outer fixed-point
    iteration: the desorber+evaporator duty (Q_des+Q_evap) and the
    condenser duty (Q_cond) are themselves OUTPUTS of the cycle, not
    independent inputs, so hitting a target outlet temperature
    (T16_target_C, T18_target_C) means re-solving with an updated flow
    each round:
        m13_(n+1) = (Q_des+Q_evap)_n / (cp * (T_13_C - T16_target_C))
        m17_(n+1) = Q_cond_n        / (cp * (T18_target_C - T_17_C))
    which converges quickly in practice because m13/m17 only weakly
    perturb Q_des+Q_evap/Q_cond through the UA-pinch equations.

    Returns a dict with the three reference flows and, on success, the
    resulting T12_C/T16_C/T18_C/COP for you to check against config's own
    targets.
    """
    cp = config.cp_w_kJkgK
    m11 = config.Qabs_reference_kW / (cp * (config.T12_target_C - config.T_11_C))

    # Initial guesses only (refined below from the actual solved Q's): the
    # drive side must supply Qabs/COP, not Qabs itself, and the condenser
    # rejects roughly Qabs/COP - Qabs -- assume a typical AHT COP~0.5 to
    # seed both (a factor-of-2 miss here, i.e. assuming Q_total_drive==
    # Qabs, previously stranded the iteration completely: the resulting
    # m13 came out ~2x too small to ever hit the target T16).
    assumed_cop = 0.5
    Q_total_drive_guess = config.Qabs_reference_kW / assumed_cop
    Q_cond_guess = Q_total_drive_guess - config.Qabs_reference_kW
    m13 = Q_total_drive_guess / (cp * (config.T_13_C - config.T16_target_C))
    m17 = Q_cond_guess / (cp * (config.T18_target_C - config.T_17_C))

    result = None
    x0 = None
    for iteration in range(config.max_iter):
        inputs = AHTInputs(
            T_11_C=config.T_11_C,
            **_routing_temperature_kwargs(config.desorber_evaporator_routing_mode, config.T_13_C),
            T_17_C=config.T_17_C,
            m_13=m13, m_15=m13, m_17=m17,
            UA_cond=ua["UA_cond"], UA_evap=ua["UA_evap"], UA_abs=ua["UA_abs"],
            UA_des=ua["UA_des"], UA_shex=ua["UA_shex"],
            desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
            cycle_scale_spec_mode="Qabs", Qabs_spec_kW=config.Qabs_reference_kW,
            absorber_spec_mode="m11", m11_spec=m11,
            cp_w_kJkgK=cp, solver_tol=config.solver_tol, max_nfev=config.max_nfev,
        )
        lower, upper = aht_bounds(inputs)
        if x0 is None:
            x0 = aht_initial_guess(inputs)
        else:
            x0 = np.clip(x0, lower + 1.0e-6, upper - 1.0e-6)
        try:
            result = solve_aht(inputs, x0=x0)
        except Exception as exc:
            return {"success": False, "message": f"solve_aht raised: {exc}", "iteration": iteration}

        if not _is_valid(result):
            # Near the edge of this UA set's feasible envelope the basin
            # of attraction around the physical solution can be small --
            # fall back to a bounded random multi-start before giving up
            # on this (m13, m17) pair entirely.
            rng = np.random.default_rng(0)
            for _ in range(200):
                x0_random = lower + rng.random(len(lower)) * (upper - lower)
                try:
                    result = solve_aht(inputs, x0=x0_random)
                except Exception:
                    continue
                if _is_valid(result):
                    break

        if not _is_valid(result):
            violated = [k for k, v in (result.checks or {}).items() if not v]
            return {
                "success": False,
                "message": (
                    f"Not feasible/converged even after a 200-point random "
                    f"multi-start (residual={result.solve_info.scaled_residual_norm:.3e}, "
                    f"violated checks: {violated})."
                ),
                "iteration": iteration,
                "m11_reference_kg_s": m11, "m13_reference_kg_s": m13, "m17_reference_kg_s": m17,
            }

        x0 = np.array([result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES])
        T16_out = result.diagnostics["T16_K"] - 273.15
        T18_out = result.diagnostics["T18_K"] - 273.15
        T12_out = result.diagnostics["T12_K"] - 273.15

        if verbose:
            print(
                f"  iter {iteration}: m13={m13:.4f} kg/s, m17={m17:.4f} kg/s -> "
                f"T16={T16_out:.3f} C (target {config.T16_target_C}), "
                f"T18={T18_out:.3f} C (target {config.T18_target_C}), "
                f"T12={T12_out:.3f} C, COP={result.kpis['COP']:.4f}"
            )

        if abs(T16_out - config.T16_target_C) < config.tol_K and abs(T18_out - config.T18_target_C) < config.tol_K:
            return {
                "success": True,
                "m11_reference_kg_s": m11,
                "m13_reference_kg_s": m13,
                "m17_reference_kg_s": m17,
                "T12_C": T12_out,
                "T16_C": T16_out,
                "T18_C": T18_out,
                "COP": result.kpis["COP"],
                "iterations": iteration + 1,
            }

        Q_total_drive = result.heat_flows_kW["Q_des"] + result.heat_flows_kW["Q_evap"]
        Q_cond = result.heat_flows_kW["Q_cond"]
        m13 = Q_total_drive / (cp * (config.T_13_C - config.T16_target_C))
        m17 = Q_cond / (cp * (config.T18_target_C - config.T_17_C))

    return {
        "success": False,
        "message": f"Did not converge within {config.max_iter} iterations.",
        "m11_reference_kg_s": m11, "m13_reference_kg_s": m13, "m17_reference_kg_s": m17,
    }


def export_reference_point(
    config: AHTReferencePointConfig = AHTReferencePointConfig(),
    *,
    ac_design_path: Path = AC_DESIGN_POINT_PATH,
    path: Path = DEFAULT_AHT_REFERENCE_PATH,
) -> dict:
    ac_design = load_design_point(ac_design_path)
    ua = ac_design["UA_kW_K"]

    calib = calibrate_reference_flows(ua, config)
    if not calib.get("success"):
        raise RuntimeError(f"AHT reference point calibration failed: {calib.get('message')}")

    payload = {
        "boundary_conditions": {
            "T_11_C": config.T_11_C,
            "T_13_C": config.T_13_C,
            "T_17_C": config.T_17_C,
            "desorber_evaporator_routing_mode": config.desorber_evaporator_routing_mode,
            "cp_w_kJkgK": config.cp_w_kJkgK,
        },
        "targets": {
            "T12_target_C": config.T12_target_C,
            "T16_target_C": config.T16_target_C,
            "T18_target_C": config.T18_target_C,
            "Qabs_reference_kW": config.Qabs_reference_kW,
        },
        "UA_kW_K": ua,
        "ua_source": str(ac_design_path),
        "reference_flows_kg_s": {
            "m11": calib["m11_reference_kg_s"],
            "m13": calib["m13_reference_kg_s"],
            "m17": calib["m17_reference_kg_s"],
        },
        "resulting": {
            "T12_C": calib["T12_C"],
            "T16_C": calib["T16_C"],
            "T18_C": calib["T18_C"],
            "COP": calib["COP"],
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"AHT reference point exported: {path}")
    return payload


def load_reference_point(path: Path = DEFAULT_AHT_REFERENCE_PATH) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    export_reference_point()
