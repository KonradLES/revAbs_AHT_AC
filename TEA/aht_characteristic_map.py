"""Normalized AHT (winter heat-transformer) characteristic map, reusing the
SAME fixed UA geometry as the AC map ("ein Auslegungspunkt, zwei
Kennfelder").

2D raster over (T_17_C, load_fraction), implementing the control strategy
you chose in chat ("Antriebsseite per Bypass konstant, Produktlast
variiert Absorber-Massenstrom"):

    FIXED  : T_13_C (geothermal return feeding the paired desorber+
             evaporator -- the AHT's "antriebsseite", held constant,
             mirroring the AC's constant T11)
             T_11_C (Netzruecklauf into the absorber)
    SWEPT  : T_17_C (condenser reject to ambient -- the AHT's single,
             ambient-driven side, mirroring the AC's T_reject sweep)
             load_fraction = Q_abs / Q_abs_reference, in (0, 1]

Routing of the desorber+evaporator drive pair follows
desorber_evaporator_routing_mode (default "series_desorber_to_evaporator":
one shared geothermal stream, desorber first then evaporator, T15=T14
internally -- the common industrial arrangement, and what actually
happens physically with one geothermal loop). T_13_C then feeds the
single external inlet of whichever component comes first in the chain
(see _routing_temperature_kwargs, imported from aht_reference_point_export
.py); T14 (desorber outlet / evaporator inlet) and T16 (evaporator outlet
-- the actual geothermal injection-temperature contribution) are model
outputs.

Mirrors the AC map's mechanics exactly, with pair/single roles swapped
(see Design_Point/AC_feasibility_sweep.py's module docstring table):
  - absorber_spec_mode="m11": m11 = load_fraction * m11_reference_kg_s is
    GIVEN directly (mirrors the AC's desorber m_11 bypass); T12 (Netz-
    Vorlauf) is a model OUTPUT, not fixed.
  - cycle_scale_spec_mode="Qabs": Qabs_spec_kW = load_fraction *
    Qabs_reference_kW targets the absorber duty via the internal cycle
    scale m6 (mirrors the AC's Qevap target).
  - m_13 = m_15 = load_fraction * m13_reference_kg_s (the geothermal
    "Bypass-Anteil, Antrieb"), mirroring the AC's m_11 scaling.
  - Unlike the AC's evaporator, the UA/LMTD model gives the condenser NO
    spec-mode choice: m_17 is a required FIXED field (T18 is always a
    model output). m_17 = load_fraction * m17_reference_kg_s (same
    "scale every external loop with load_fraction" convention used for
    the AC's absorber/condenser flows); it is not geothermally
    constrained, so it is not tracked as a map output.

Reference point (T_11_C/T_13_C/Qabs_reference_kW/the three reference
flows) is NOT configured inline here anymore -- it is calibrated and
exported to JSON by aht_reference_point_export.py (mirroring
design_point_export.py's ac_design_point.json), so you can edit the
winter temperature levels directly in that JSON. See that module's
docstring for why Qabs_reference_kW is 600 (not the smaller winter duty
first discussed) and how the reference flows were derived rather than
guessed. Use config_from_reference_point() below to turn the loaded JSON
into an AHTMapConfig for build_map().

Standalone usage
-----------------
    python TEA/aht_characteristic_map.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AHT_UA_LMTD import (
    AHTInputs,
    PRIMARY_VARIABLE_NAMES,
    bounds as aht_bounds,
    initial_guess as aht_initial_guess,
    solve_aht,
)
from TEA.aht_reference_point_export import (
    _routing_temperature_kwargs,
    load_reference_point,
)
from TEA.characteristic_map import CharacteristicMap, sweep_raster
from TEA.design_point_export import DEFAULT_EXPORT_PATH, load_design_point

RESIDUAL_TOL = 1.0e-6
DEFAULT_MAP_PATH = Path(__file__).resolve().parent / "maps" / "aht_characteristic_map.npz"


@dataclass
class AHTMapConfig:
    # --- Fixed boundary conditions (normally loaded from
    # aht_reference_point.json via config_from_reference_point(), not set
    # here directly) ---
    T_11_C: float = 80.0          # Netzruecklauf into the absorber [°C]
    T_13_C: float = 75.0          # geothermal return feeding des+evap [°C]
    desorber_evaporator_routing_mode: str = "series_desorber_to_evaporator"

    # --- Reference duty: WHAT load_fraction=1.0 means for this Kennfeld ---
    # See aht_reference_point_export.py for how these were calibrated.
    Qabs_reference_kW: float = 600.0
    m11_reference_kg_s: float = 14.354066985645934
    m13_reference_kg_s: float = 24.036699612439822
    m17_reference_kg_s: float = 36.04507517719856  # condenser (ambient reject) cooling-loop design flow

    # --- Sweep grid ---
    T_17_values_C: Tuple[float, ...] = tuple(np.arange(-5.0, 21.0, 2.0))
    load_fraction_values: Tuple[float, ...] = tuple(np.round(np.arange(0.15, 1.0001, 0.05), 3))
    min_load_fraction: float = 0.15

    solver_tol: float = 1.0e-7
    max_nfev: int = 400


def config_from_reference_point(
    ref: dict,
    *,
    T_17_values_C: Tuple[float, ...] = tuple(np.arange(-5.0, 21.0, 2.0)),
    load_fraction_values: Tuple[float, ...] = tuple(np.round(np.arange(0.15, 1.0001, 0.05), 3)),
    min_load_fraction: float = 0.15,
    solver_tol: float = 1.0e-7,
    max_nfev: int = 400,
) -> AHTMapConfig:
    """Builds an AHTMapConfig from a loaded aht_reference_point.json (see
    aht_reference_point_export.load_reference_point()). The sweep grid
    (T_17_values_C/load_fraction_values) is not part of that JSON -- it
    describes the Kennfeld you want, not the calibrated reference point --
    so it stays a normal keyword argument here with the usual defaults.
    """
    bc = ref["boundary_conditions"]
    targets = ref["targets"]
    flows = ref["reference_flows_kg_s"]
    return AHTMapConfig(
        T_11_C=bc["T_11_C"],
        T_13_C=bc["T_13_C"],
        desorber_evaporator_routing_mode=bc["desorber_evaporator_routing_mode"],
        Qabs_reference_kW=targets["Qabs_reference_kW"],
        m11_reference_kg_s=flows["m11"],
        m13_reference_kg_s=flows["m13"],
        m17_reference_kg_s=flows["m17"],
        T_17_values_C=T_17_values_C,
        load_fraction_values=load_fraction_values,
        min_load_fraction=min_load_fraction,
        solver_tol=solver_tol,
        max_nfev=max_nfev,
    )


def _clip_to_bounds(z: np.ndarray, inputs: AHTInputs) -> np.ndarray:
    lower, upper = aht_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


def _build_inputs(T_17_C: float, load_fraction: float, design: dict, config: AHTMapConfig) -> AHTInputs:
    ua = design["UA_kW_K"]

    return AHTInputs(
        T_11_C=config.T_11_C,
        **_routing_temperature_kwargs(config.desorber_evaporator_routing_mode, config.T_13_C),
        T_17_C=T_17_C,
        m_13=load_fraction * config.m13_reference_kg_s,
        m_15=load_fraction * config.m13_reference_kg_s,
        m_17=load_fraction * config.m17_reference_kg_s,
        UA_cond=ua["UA_cond"],
        UA_evap=ua["UA_evap"],
        UA_abs=ua["UA_abs"],
        UA_des=ua["UA_des"],
        UA_shex=ua["UA_shex"],
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="Qabs",
        Qabs_spec_kW=load_fraction * config.Qabs_reference_kW,
        absorber_spec_mode="m11",
        m11_spec=load_fraction * config.m11_reference_kg_s,
        solver_tol=config.solver_tol,
        max_nfev=config.max_nfev,
    )


def _is_valid(result) -> bool:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def build_map(design: dict, config: AHTMapConfig = AHTMapConfig(), *, verbose: bool = True) -> CharacteristicMap:
    Qabs_ref = config.Qabs_reference_kW

    def cold_start(T_17_C: float, load_fraction: float) -> np.ndarray:
        return aht_initial_guess(_build_inputs(T_17_C, load_fraction, design, config))

    def solve_point(T_17_C: float, load_fraction: float, x0: np.ndarray):
        inputs = _build_inputs(T_17_C, load_fraction, design, config)
        x0_clipped = _clip_to_bounds(x0, inputs)
        try:
            result = solve_aht(inputs, x0=x0_clipped)
        except Exception:
            return None
        if not _is_valid(result):
            return None
        x_sol = np.array([result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES])
        values = {
            "Q_abs_frac": result.heat_flows_kW["Q_abs"] / Qabs_ref,
            "COP": result.kpis["COP"],
            # m_13 = m_15 are FIXED inputs to this UA/LMTD model (not solved
            # for), so m13_frac == load_fraction exactly by construction.
            "m13_frac": load_fraction,
            # T16, not T14: with series_desorber_to_evaporator, T14 is only
            # the INTERSTAGE point (desorber outlet = evaporator inlet);
            # T16 is the actual final geothermal outlet after BOTH
            # components, i.e. the real injection-temperature contribution.
            "T_geo_out_C": result.diagnostics["T16_K"] - 273.15,
            "T12_C": result.diagnostics["T12_K"] - 273.15,        # Netzvorlauf (resulting)
        }
        return x_sol, values

    grids = sweep_raster(
        np.asarray(config.T_17_values_C, dtype=float),
        np.asarray(config.load_fraction_values, dtype=float),
        solve_point=solve_point,
        cold_start=cold_start,
        field_names=["Q_abs_frac", "COP", "m13_frac", "T_geo_out_C", "T12_C"],
        verbose=verbose,
    )

    return CharacteristicMap(
        T_ambient_C=np.asarray(config.T_17_values_C, dtype=float),
        load_fraction=np.asarray(config.load_fraction_values, dtype=float),
        fields=grids,
        meta={
            "mode": "AHT",
            "design_point_source": str(DEFAULT_EXPORT_PATH),
            "Qabs_reference_kW": Qabs_ref,
            "T_11_C": config.T_11_C,
            "T_13_C": config.T_13_C,
            "min_load_fraction": config.min_load_fraction,
        },
    )


if __name__ == "__main__":
    design = load_design_point()
    ref = load_reference_point()
    config = config_from_reference_point(ref)
    cmap = build_map(design, config)
    cmap.save(DEFAULT_MAP_PATH)
    print(f"AHT characteristic map saved: {DEFAULT_MAP_PATH}")
