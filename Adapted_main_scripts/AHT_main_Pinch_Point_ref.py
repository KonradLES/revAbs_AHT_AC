"""Entry point for the AHT simulation with 7 primary unknowns.

Absorber specification is explicitly selectable:
- ABSORBER_SPEC_MODE = "m11" -> give m11_spec, T12 is computed
- ABSORBER_SPEC_MODE = "T12" -> give T12_spec_C, m11 is computed

Desorber specification is explicitly selectable:
- DESORBER_SPEC_MODE = "m13" -> give m13_spec, T14 is computed
- DESORBER_SPEC_MODE = "T14" -> give T14_spec_C, m13 is computed

Evaporator specification is explicitly selectable:
- EVAPORATOR_SPEC_MODE = "m15" -> give m15_spec, T16 is computed
- EVAPORATOR_SPEC_MODE = "T16" -> give T16_spec_C, m15 is computed

Condenser specification is explicitly selectable:
- CONDENSER_SPEC_MODE = "m17" -> give m17_spec, T18 is computed
- CONDENSER_SPEC_MODE = "T18" -> give T18_spec_C, m17 is computed

Cycle scaling is explicitly selectable:
- CYCLE_SCALE_SPEC_MODE = "m6"   -> give m6_spec
- CYCLE_SCALE_SPEC_MODE = "Qabs" -> give Qabs_spec_kW, m6 is computed

External thermal routing of desorber and evaporator is selectable:
- DESORBER_EVAPORATOR_ROUTING_MODE = "parallel" -> T_13_C and T_15_C are given
- DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator" -> internally T15 = T14
- DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber" -> internally T13 = T16
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from Models.AHT_Pinch_Point import (
    AHTInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_aht,
    trace_model,
)
from Postprocessing.AHT_QT_Plot import plot_qt_diagrams
from Postprocessing.AHT_Duehring_Plot import plot_duehring_operating_point

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
# Generate Q-T diagrams (pinch analysis) after solving?
ENABLE_QT_PLOT = True
QT_PLOT_SAVE_PATH = "Postprocessing/Plots/AHT_QT_Diagramme.png"  # None to skip saving
# Generate a Duehring diagram with the operating point marked, after solving?
ENABLE_DUEHRING_PLOT = True
DUEHRING_PLOT_SAVE_PATH = "Postprocessing/Plots/AHT_Duehring_Diagramm.png"  # None to skip saving
DUEHRING_PLOT_VARIANT = "mass"  # "mass" or "mole"
# ----------------------------------------------------------------------------

# CYCLE_SCALE_SPEC_MODE = "m6"
CYCLE_SCALE_SPEC_MODE = "Qabs"

# ABSORBER_SPEC_MODE = "m11"
ABSORBER_SPEC_MODE = "T12"

# DESORBER_SPEC_MODE = "m13"
DESORBER_SPEC_MODE = "T14"

# EVAPORATOR_SPEC_MODE = "m15"
EVAPORATOR_SPEC_MODE = "T16"

# CONDENSER_SPEC_MODE = "m17"
CONDENSER_SPEC_MODE = "T18"

DESORBER_EVAPORATOR_ROUTING_MODE = "parallel"
#DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator"
# DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber"


def build_example_inputs() -> AHTInputs:
    common_kwargs = dict(
        T_11_C=135.0,   # 135, 60, 80
        T_17_C=30.0,   # 30, 20, 20
        dT_min_shex=4.282178,    # 4.3
        dT_min_des=6.257224,     # 6.3
        dT_min_cond=14.224404,   # 25.1
        dT_min_evap=8.618245,   # 7.73
        dT_min_abs=17.836021,    # 17.8
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        absorber_spec_mode=ABSORBER_SPEC_MODE,
        desorber_spec_mode=DESORBER_SPEC_MODE,
        evaporator_spec_mode=EVAPORATOR_SPEC_MODE,
        condenser_spec_mode=CONDENSER_SPEC_MODE,
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
        desorber_evaporator_routing_mode=DESORBER_EVAPORATOR_ROUTING_MODE,
    )

    spec_kwargs: dict[str, float] = {}

    if CYCLE_SCALE_SPEC_MODE == "m6":
        spec_kwargs["m6_spec"] = 1.0  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qabs":
        spec_kwargs["Qabs_spec_kW"] = 184.4  # 184.4, 6.9
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE must be 'm6' or 'Qabs'.")

    if ABSORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 4  # 4, 0.2
    elif ABSORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 146.02  # 146.02, 80
    else:
        raise ValueError("ABSORBER_SPEC_MODE must be 'm11' or 'T12'.")

    if DESORBER_SPEC_MODE == "m13":
        spec_kwargs["m13_spec"] = 4
    elif DESORBER_SPEC_MODE == "T14":
        spec_kwargs["T14_spec_C"] = 108.92  # 108.92
    else:
        raise ValueError("DESORBER_SPEC_MODE must be 'm13' or 'T14'.")

    if EVAPORATOR_SPEC_MODE == "m15":
        spec_kwargs["m15_spec"] = 4
    elif EVAPORATOR_SPEC_MODE == "T16":
        spec_kwargs["T16_spec_C"] = 108.80 # 108.80
    else:
        raise ValueError("EVAPORATOR_SPEC_MODE must be 'm15' or 'T16'.")

    if CONDENSER_SPEC_MODE == "m17":
        spec_kwargs["m17_spec"] = 4
    elif CONDENSER_SPEC_MODE == "T18":
        spec_kwargs["T18_spec_C"] = 41.26  # 41.26
    else:
        raise ValueError("CONDENSER_SPEC_MODE must be 'm17' or 'T18'.")


    if DESORBER_EVAPORATOR_ROUTING_MODE == "parallel":
        common_kwargs["T_13_C"] = 120.0   # 120,60, 65
        common_kwargs["T_15_C"] = 120.0  # 120, 60, 65
    elif DESORBER_EVAPORATOR_ROUTING_MODE == "series_desorber_to_evaporator":
        common_kwargs["T_13_C"] = 120.0 # 120, 65
        common_kwargs["T_15_C"] = None
    elif DESORBER_EVAPORATOR_ROUTING_MODE == "series_evaporator_to_desorber":
        common_kwargs["T_13_C"] = None
        common_kwargs["T_15_C"] = 120.0 # 120, 65
    else:
        raise ValueError(
            "DESORBER_EVAPORATOR_ROUTING_MODE must be 'parallel', "
            "'series_desorber_to_evaporator', or 'series_evaporator_to_desorber'."
        )

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
            [
                55,   # T8  [°C] 55, 29.98, 30          - T17 + 25
                101,   # T10 [°C] 101, 50.02, 55        - T15 - 20
                0.23,    # x3  [-] 0.23, 0.15, 0.23
                0.27,    # x6  [-] 0.27, 0.18, 0.27
                0.26,   # x20 [-] 0.26, 0.17, 0.26
                121,   # T2  [°C] 121, 59.50, 70        - T11 - 15
                150,   # T4  [°C] 150, 68.98, 80        - T11 + 15
            ],
            dtype=float,
        )
    )

    trace = trace_model(x0, inputs)
    #print_trace(trace)

    result = solve_aht(inputs, x0=x0)
    print_summary(result)

    if ENABLE_QT_PLOT:
        plot_qt_diagrams(result, save_path=QT_PLOT_SAVE_PATH)
    
    if ENABLE_DUEHRING_PLOT:
        plot_duehring_operating_point(
            result,
            variant=DUEHRING_PLOT_VARIANT,
            save_path=DUEHRING_PLOT_SAVE_PATH,
        )