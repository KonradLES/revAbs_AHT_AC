"""Entry point for the AC simulation with 6 primary unknowns (UA/LMTD formulation).

Evaporator specification is explicitly selectable:
- EVAPORATOR_SPEC_MODE = "m17" -> give m17_spec, T18 is computed
- EVAPORATOR_SPEC_MODE = "T18" -> give T18_spec_C, m17 is computed

Cycle scaling is explicitly selectable:
- CYCLE_SCALE_SPEC_MODE = "m1"   -> give m1_spec
- CYCLE_SCALE_SPEC_MODE = "Qeva" -> give Qevap_spec_kW, m1 is computed

SHEX modeling is explicitly selectable:
- SHEX_MODEL_MODE = "UA"  -> give UA_shex
- SHEX_MODEL_MODE = "NTU" -> give Effectiveness_shex

External thermal routing of absorber and condenser is selectable:
- ABSORBER_CONDENSER_ROUTING_MODE = "parallel" -> T_13_C and T_15_C are given
- ABSORBER_CONDENSER_ROUTING_MODE = "series_absorber_to_condenser" -> internally T15 = T14
- ABSORBER_CONDENSER_ROUTING_MODE = "series_condenser_to_absorber" -> internally T13 = T16
"""

from __future__ import annotations

import numpy as np

from Models.AC_UA_LMTD import (
    ACInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_ac,
    trace_model,
)

EVAPORATOR_SPEC_MODE = "m17"
#EVAPORATOR_SPEC_MODE = "T18"

CYCLE_SCALE_SPEC_MODE = "m1"
# CYCLE_SCALE_SPEC_MODE = "Qeva"

# SHEX_MODEL_MODE = "UA"
SHEX_MODEL_MODE = "NTU"

ABSORBER_CONDENSER_ROUTING_MODE = "parallel"
#ABSORBER_CONDENSER_ROUTING_MODE = "series_absorber_to_condenser"
# ABSORBER_CONDENSER_ROUTING_MODE = "series_condenser_to_absorber"

def build_example_inputs() -> ACInputs:
    common_kwargs = dict(
        T_11_C=100.0,   # 135, 60, 80
        #T_13_C=None,   # 120, 60
        #T_15_C=None,   # 120, 60
        T_17_C=10.0,   # 30, 20, 20
        m_11=1,      # 4, 0.2, 4
        m_13=0.28,      # 4, 0.2, 4
        m_15=0.28,      # 4, 0.2, 4
        UA_cond=1.2,  # 10, 1.0025, 25.2578
        UA_evap=2.25,  # 15, 1.5079, 11.3518
        UA_abs=1.8,      # 10, 1.5, 8.1355
        UA_des=1.0,   # 25, 2.4895, 10.4058
        cp_w_kJkgK=4.2,
        desorber_vapor_superheat_K=0.0,
        shex_model=SHEX_MODEL_MODE,
        evaporator_spec_mode=EVAPORATOR_SPEC_MODE,
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
        absorber_condenser_routing_mode=ABSORBER_CONDENSER_ROUTING_MODE,
    )

    if SHEX_MODEL_MODE == "UA":
        common_kwargs["UA_shex"] = 3.105/22.96   
    elif SHEX_MODEL_MODE == "NTU":
        common_kwargs["Effectiveness_shex"] = 0.64
    else:
        raise ValueError(
            "SHEX_MODEL_MODE must be 'UA' or 'NTU'."
        )

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

    spec_kwargs: dict[str, float] = {}

    if EVAPORATOR_SPEC_MODE == "m17":
        spec_kwargs["m17_spec"] = 0.4  # 4, 0.2
    elif EVAPORATOR_SPEC_MODE == "T18":
        spec_kwargs["T18_spec_C"] = 146  # 146, 80
    else:
        raise ValueError("EVAPORATOR_SPEC_MODE must be 'm17' or 'T18'.")

    if CYCLE_SCALE_SPEC_MODE == "m1":
        spec_kwargs["m1_spec"] = 0.05  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qeva":
        spec_kwargs["Qevap_spec_kW"] = 6.67  # 184, 6.9
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
                40.06,   # T8  [°C] 55, 29.98, 30
                1.39,   # T10 [°C] 101, 50.02, 55
                0.262,    # x4  [-] 0.23, 0.15, 0.23
                0.243,    # x1  [-] 0.27, 0.18, 0.27
                63.61,   # T3  [°C] 121, 59.50, 70
                53.11,   # T5  [°C] 150, 68.98, 80
            ],
            dtype=float,
        )
    )

    trace = trace_model(x0, inputs)
    #print_trace(trace)

    result = solve_ac(inputs, x0=x0)
    print_summary(result)
