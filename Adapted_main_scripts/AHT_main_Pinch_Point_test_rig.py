"""Entry point for the AHT simulation with 7 primary unknowns.

Absorber specification is explicitly selectable:
- ABSORBER_SPEC_MODE = "m11" -> give m11_spec, T12 is computed
- ABSORBER_SPEC_MODE = "T12" -> give T12_spec_C, m11 is computed

Desorber, evaporator, and condenser specification are fixed to their mass-flow
mode below (m13/m15/m17), matching the fixed external mass flows used by the
sibling UA/LMTD test rig.

Cycle scaling is explicitly selectable:
- CYCLE_SCALE_SPEC_MODE = "m6"   -> give m6_spec
- CYCLE_SCALE_SPEC_MODE = "Qabs" -> give Qabs_spec_kW, m6 is computed

External thermal routing of desorber and evaporator is selectable:
- DESORBER_EVAPORATOR_ROUTING_MODE = "parallel" -> T_13_C and T_15_C are given
- DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator" -> internally T15 = T14
- DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber" -> internally T13 = T16
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AHT_Pinch_Point import (
    AHTInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_aht,
    trace_model,
)

# ABSORBER_SPEC_MODE = "m11"
ABSORBER_SPEC_MODE = "T12"

# CYCLE_SCALE_SPEC_MODE = "m6"
CYCLE_SCALE_SPEC_MODE = "Qabs"

DESORBER_EVAPORATOR_ROUTING_MODE = "parallel"
#DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator"
# DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber"


def build_example_inputs() -> AHTInputs:
    common_kwargs = dict(
        T_11_C=85.0,   # 135, 60, 80
        #T_13_C=None,   # 120, 60
        #T_15_C=None,   # 120, 60
        T_17_C=20.0,   # 30, 20, 20
        dT_min_shex=1.2,    # 1.2
        dT_min_des=1.9,     # 1.9
        dT_min_cond=4.8,    # 4.8
        dT_min_evap=6.6,    # 6.6
        dT_min_abs=5.6,     # 5.6
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        absorber_spec_mode=ABSORBER_SPEC_MODE,
        desorber_spec_mode="m13",
        evaporator_spec_mode="m15",
        condenser_spec_mode="m17",
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
        desorber_evaporator_routing_mode=DESORBER_EVAPORATOR_ROUTING_MODE,
    )

    if DESORBER_EVAPORATOR_ROUTING_MODE == "parallel":
        common_kwargs["T_13_C"] = 65.0   # 120,60, 65
        common_kwargs["T_15_C"] = 65.0  # 120, 60, 65
    elif DESORBER_EVAPORATOR_ROUTING_MODE == "series_desorber_to_evaporator":
        common_kwargs["T_13_C"] = 65.0 # 120, 65
        common_kwargs["T_15_C"] = None
    elif DESORBER_EVAPORATOR_ROUTING_MODE == "series_evaporator_to_desorber":
        common_kwargs["T_13_C"] = None
        common_kwargs["T_15_C"] = 65.0 # 120, 65
    else:
        raise ValueError(
            "DESORBER_EVAPORATOR_ROUTING_MODE must be 'parallel', "
            "'series_desorber_to_evaporator', or 'series_evaporator_to_desorber'."
        )

    spec_kwargs: dict[str, float] = {
        "m13_spec": 4,   # 4, 0.2, 4
        "m15_spec": 4,   # 4, 0.2, 4
        "m17_spec": 4,   # 4, 0.2, 4
    }

    if ABSORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 4  # 4, 0.2
    elif ABSORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 90  # 146, 80
    else:
        raise ValueError("ABSORBER_SPEC_MODE must be 'm11' or 'T12'.")

    if CYCLE_SCALE_SPEC_MODE == "m6":
        spec_kwargs["m6_spec"] = 0.45  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qabs":
        spec_kwargs["Qabs_spec_kW"] = 60  # 184, 6.9
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE must be 'm6' or 'Qabs'.")

    return AHTInputs(
        **common_kwargs,
        **spec_kwargs,
    )


if __name__ == "__main__":
    inputs = build_example_inputs()

    # Initial vector in the order:
    # [T8, T10, x3, x6, x20, T2, T4]
    #
    # Temperature initial guesses given by the user in degC.
    # Conversion to the internal model units [K] happens directly below.
    x0 = primary_temperatures_C_to_K(
        np.array(
            [30.1458, 57.2958, 0.1787, 0.2003, 0.1981, 66.1183, 85.6828],
            dtype=float,
        )
    )

    trace = trace_model(x0, inputs)
    #print_trace(trace)

    result = solve_aht(inputs, x0=x0)
    print_summary(result)
