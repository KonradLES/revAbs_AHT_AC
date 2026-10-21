"""Exports the ONE AC design-point UA set that both Kennfelder reuse.

Core assumption of the whole TEA pipeline (see chat discussion with the
user): the reversible machine is always sized against the COOLING demand,
so there is exactly one design point (the AC/chiller design point). Its UA
set (UA_shex/des/cond/evap/abs) is reused UNCHANGED as the AHT (winter)
geometry -- no independent AHT design-point optimization is run. This
module runs the existing bilevel optimizer once and dumps the result to
JSON so ac_characteristic_map.py / aht_characteristic_map.py don't each
have to re-run the (minutes-long) DE optimization.

Standalone usage
-----------------
    python TEA/design_point_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Design_Point.AC_design_point_optimizer import (
    DesignPointConfig,
    optimize_design_point,
    print_design_point_summary,
)

DEFAULT_EXPORT_PATH = Path(__file__).resolve().parent / "design_point" / "ac_design_point.json"


def export_design_point(
    config: DesignPointConfig, path: Path = DEFAULT_EXPORT_PATH, *, seed: int = 42
) -> dict:
    theta_opt, result, _stats = optimize_design_point(config, seed=seed)
    print_design_point_summary(theta_opt, result)

    ua = result.UA_conversion
    payload = {
        "boundary_conditions": {
            "T_11_C": config.T_11_C,
            "T_13_C": config.T_13_C,
            "T_15_C": config.T_15_C,
            "T_17_C": config.T_17_C,
            "T12_spec_C": config.T12_spec_C,
            "T14_spec_C": config.T14_spec_C,
            "T16_spec_C": config.T16_spec_C,
            "T18_spec_C": config.T18_spec_C,
            "Qevap_spec_kW": config.Qevap_spec_kW,
            "absorber_condenser_routing_mode": config.absorber_condenser_routing_mode,
            "cp_w_kJkgK": config.cp_w_kJkgK,
        },
        "dT_min": {k: float(v) for k, v in zip(["shex", "des", "cond", "evap", "abs"], theta_opt)},
        "UA_kW_K": {
            "UA_shex": ua["UA_shex"],
            "UA_des": ua["UA_des"],
            "UA_cond": ua["UA_cond"],
            "UA_evap": ua["UA_evap"],
            "UA_abs": ua["UA_abs"],
        },
        "design_flows_kg_s": {
            "m11": result.diagnostics["m11_kg_s"],
            "m13": result.diagnostics["m13_kg_s"],
            "m15": result.diagnostics["m15_kg_s"],
            "m17": result.diagnostics["m17_kg_s"],
            "m1": result.states["1"]["m_kg_s"],
        },
        "Q_evap_design_kW": result.heat_flows_kW["Q_evap"],
        "COP_design": result.kpis["COP"],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Design point exported: {path}")
    return payload


def load_design_point(path: Path = DEFAULT_EXPORT_PATH) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    export_design_point(DesignPointConfig())
