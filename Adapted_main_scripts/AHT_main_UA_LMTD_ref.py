"""Einstiegspunkt für die AHT-Simulation mit 8 primären Unbekannten.

Die Absorber-Spezifikation ist explizit wählbar:
- ABSORBER_SPEC_MODE = "m11"  -> m11_spec vorgeben, T12 wird berechnet
- ABSORBER_SPEC_MODE = "T12"  -> T12_spec_C vorgeben, m11 wird berechnet

Die Kreislaufskalierung ist explizit wählbar:
- CYCLE_SCALE_SPEC_MODE = "m6"   -> m6_spec vorgeben
- CYCLE_SCALE_SPEC_MODE = "Qabs" -> Qabs_spec_kW vorgeben, m6 wird berechnet

Die externe thermische Verschaltung von Desorber und Verdampfer ist wählbar:
- DESORBER_EVAPORATOR_ROUTING_MODE = "parallel" -> T_13_C und T_15_C werden vorgegeben
- DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator" -> intern gilt T15 = T14
- DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber" -> intern gilt T13 = T16
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from Models.AHT_UA_LMTD import (
    AHTInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_aht,
    trace_model,
)

ABSORBER_SPEC_MODE = "m11"
#ABSORBER_SPEC_MODE = "T12"

CYCLE_SCALE_SPEC_MODE = "m6"
# CYCLE_SCALE_SPEC_MODE = "Qabs"

SHEX_MODEL_MODE = "UA"
# SHEX_MODEL_MODE = "NTU"

DESORBER_EVAPORATOR_ROUTING_MODE = "parallel"
#DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator"
# DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber"ö

def build_example_inputs() -> AHTInputs:
    common_kwargs = dict(
        T_11_C=135.0,   # 135, 60, 80
        #T_13_C=None,   # 120, 60
        #T_15_C=None,   # 120, 60
        T_17_C=30.0,   # 30, 20, 20
        m_13=4,      # 4, 0.2, 4
        m_15=4,      # 4, 0.2, 4
        m_17=4,      # 4, 0.2, 4
        UA_cond=10,  # 10, 1.0025, 25.2578
        UA_evap=15,  # 15, 1.5079, 11.3518
        UA_abs=10,      # 10, 1.5, 8.1355
        UA_des=25,   # 25, 2.4895, 10.4058s
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        shex_model=SHEX_MODEL_MODE,
        absorber_spec_mode=ABSORBER_SPEC_MODE,
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
        desorber_evaporator_routing_mode=DESORBER_EVAPORATOR_ROUTING_MODE,
    )

    if SHEX_MODEL_MODE == "UA":
        common_kwargs["UA_shex"] = 70.8/6.43   
    elif SHEX_MODEL_MODE == "NTU":
        common_kwargs["Effectiveness_shex"] = 0.9
    else:
        raise ValueError(
            "SHEX_MODEL_MODE muss 'UA' oder 'NTU' sein."
        )

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
            "DESORBER_EVAPORATOR_ROUTING_MODE muss 'parallel', "
            "'series_desorber_to_evaporator' oder 'series_evaporator_to_desorber' sein."
        )

    spec_kwargs: dict[str, float] = {}

    if ABSORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 4  # 4, 0.2
    elif ABSORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 146  # 146, 80
    else:
        raise ValueError("ABSORBER_SPEC_MODE muss 'm11' oder 'T12' sein.")

    if CYCLE_SCALE_SPEC_MODE == "m6":
        spec_kwargs["m6_spec"] = 1  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qabs":
        spec_kwargs["Qabs_spec_kW"] = 184  # 184, 6.9
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE muss 'm6' oder 'Qabs' sein.")

    return AHTInputs(
        **common_kwargs,
        **spec_kwargs,
    )


if __name__ == "__main__":
    inputs = build_example_inputs()

    # Startvektor in der Reihenfolge:
    # [T8, T10, x3, x6, x20, T2, T4]
    #
    # Benutzerangabe der Temperatur-Startwerte in °C.
    # Die Konvertierung in die internen Modell-Einheiten [K] erfolgt direkt darunter.
    x0 = primary_temperatures_C_to_K(
        np.array(
            [
                55,   # T8  [°C] 55, 29.98, 30
                101,   # T10 [°C] 101, 50.02, 55
                0.23,    # x3  [-] 0.23, 0.15, 0.23
                0.27,    # x6  [-] 0.27, 0.18, 0.27
                0.26,   # x20 [-] 0.26, 0.17, 0.26
                121,   # T2  [°C] 121, 59.50, 70
                150,   # T4  [°C] 150, 68.98, 80
            ],
            dtype=float,
        )
    )

    trace = trace_model(x0, inputs)
    #print_trace(trace)

    result = solve_aht(inputs, x0=x0)
    print_summary(result)
