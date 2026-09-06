"""Entry point for the AC simulation with 6 primary unknowns.

Desorber specification is explicitly selectable:
- DESORBER_SPEC_MODE = "m11" -> give m11_spec, T12 is computed
- DESORBER_SPEC_MODE = "T12" -> give T12_spec_C, m11 is computed

Absorber specification is explicitly selectable:
- ABSORBER_SPEC_MODE = "m13" -> give m13_spec, T14 is computed
- ABSORBER_SPEC_MODE = "T14" -> give T14_spec_C, m13 is computed

Condenser specification is explicitly selectable:
- CONDENSER_SPEC_MODE = "m15" -> give m15_spec, T16 is computed
- CONDENSER_SPEC_MODE = "T16" -> give T16_spec_C, m15 is computed

Evaporator specification is explicitly selectable:
- EVAPORATOR_SPEC_MODE = "m17" -> give m17_spec, T18 is computed
- EVAPORATOR_SPEC_MODE = "T18" -> give T18_spec_C, m17 is computed

Cycle scaling is explicitly selectable:
- CYCLE_SCALE_SPEC_MODE = "m1"   -> give m1_spec
- CYCLE_SCALE_SPEC_MODE = "Qeva" -> give Qevap_spec_kW, m1 is computed

External thermal routing of absorber and condenser is selectable:
- ABSORBER_CONDENSER_ROUTING_MODE = "parallel" -> T_13_C and T_15_C are given
- ABSORBER_CONDENSER_ROUTING_MODE = "series_absorber_to_condenser" -> internally T15 = T14
- ABSORBER_CONDENSER_ROUTING_MODE = "series_condenser_to_absorber" -> internally T13 = T16
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AC_Pinch_Point import (
    ACInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_ac,
    trace_model,
)

# DESORBER_SPEC_MODE = "m11"
DESORBER_SPEC_MODE = "T12"

# ABSORBER_SPEC_MODE = "m13"
ABSORBER_SPEC_MODE = "T14"

# CONDENSER_SPEC_MODE = "m15"
CONDENSER_SPEC_MODE = "T16"

# EVAPORATOR_SPEC_MODE = "m17"
EVAPORATOR_SPEC_MODE = "T18"

# CYCLE_SCALE_SPEC_MODE = "m1"
CYCLE_SCALE_SPEC_MODE = "Qeva"

ABSORBER_CONDENSER_ROUTING_MODE = "parallel"
#ABSORBER_CONDENSER_ROUTING_MODE = "series_absorber_to_condenser"
# ABSORBER_CONDENSER_ROUTING_MODE = "series_condenser_to_absorber"

def build_example_inputs() -> ACInputs:
    common_kwargs = dict(
        T_11_C=90.0,   # 135, 60, 80
        T_17_C=11.0,   # 30, 20, 20
        dT_min_shex=2,    # 20.1
        dT_min_des=5,     # 10.8, 1.68
        dT_min_cond=5,   # 5.4, 0.3
        dT_min_evap=5,   # 2.23, 2.8
        dT_min_abs=5,    # 7.82, 3.9
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        desorber_spec_mode=DESORBER_SPEC_MODE,
        absorber_spec_mode=ABSORBER_SPEC_MODE,
        condenser_spec_mode=CONDENSER_SPEC_MODE,
        evaporator_spec_mode=EVAPORATOR_SPEC_MODE,
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
        absorber_condenser_routing_mode=ABSORBER_CONDENSER_ROUTING_MODE,
    )
    
    spec_kwargs: dict[str, float] = {}
    
    if DESORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 0.7
    elif DESORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 72
    else:
        raise ValueError("DESORBER_SPEC_MODE must be 'm11' or 'T12'.")

    if ABSORBER_SPEC_MODE == "m13":
        spec_kwargs["m13_spec"] = 1.7
    elif ABSORBER_SPEC_MODE == "T14":
        spec_kwargs["T14_spec_C"] = 32
    else:
        raise ValueError("ABSORBER_SPEC_MODE must be 'm13' or 'T14'.")

    if CONDENSER_SPEC_MODE == "m15":
        spec_kwargs["m15_spec"] = 1.5
    elif CONDENSER_SPEC_MODE == "T16":
        spec_kwargs["T16_spec_C"] = 32
    else:
        raise ValueError("CONDENSER_SPEC_MODE must be 'm15' or 'T16'.")

    if EVAPORATOR_SPEC_MODE == "m17":
        spec_kwargs["m17_spec"] = 1.6
    elif EVAPORATOR_SPEC_MODE == "T18":
        spec_kwargs["T18_spec_C"] = 5
    else:
        raise ValueError("EVAPORATOR_SPEC_MODE must be 'm17' or 'T18'.")

    if ABSORBER_CONDENSER_ROUTING_MODE == "parallel":
        common_kwargs["T_13_C"] = 25.0   # 120,60, 65
        common_kwargs["T_15_C"] = 25.0  # 120, 60, 65
    elif ABSORBER_CONDENSER_ROUTING_MODE == "series_absorber_to_condenser":
        common_kwargs["T_13_C"] = 25.0 # 120, 65
        common_kwargs["T_15_C"] = None
    elif ABSORBER_CONDENSER_ROUTING_MODE == "series_condenser_to_absorber":
        common_kwargs["T_13_C"] = None
        common_kwargs["T_15_C"] = 25.0 # 120, 65
    else:
        raise ValueError(
            "ABSORBER_CONDENSER_ROUTING_MODE must be 'parallel', "
            "'series_absorber_to_condenser', or 'series_condenser_to_absorber'."
        )

    if CYCLE_SCALE_SPEC_MODE == "m1":
        spec_kwargs["m1_spec"] = 0.37  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qeva":
        spec_kwargs["Qevap_spec_kW"] = 40.9  # 184, 6.9
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE must be 'm1' or 'Qeva'.")

    return ACInputs(
        **common_kwargs,
        **spec_kwargs,
    )


if __name__ == "__main__":
    inputs = build_example_inputs()

    # Initial vector in the order:
    # [T8, T10, x4, x1, T3, T5]
    #
    # Temperature initial guesses given by the user in degC.
    # Conversion to the internal model units [K] happens directly below.
    x0 = primary_temperatures_C_to_K(
        np.array(
            [
                32.3,   # T8  [°C] 55, 29.98, 30
                2.2,   # T10 [°C] 101, 50.02, 55
                0.2388,    # x4  [-] 0.23, 0.15, 0.23
                0.2185,    # x1  [-] 0.27, 0.18, 0.27
                76.7,   # T3  [°C] 121, 59.50, 70
                42,   # T5  [°C] 150, 68.98, 80
            ],
            dtype=float,
        )
    )

    trace = trace_model(x0, inputs)
    #print_trace(trace)

    result = solve_ac(inputs, x0=x0)
    print_summary(result)
