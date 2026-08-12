"""Einstiegspunkt für die AWT-Simulation mit 8 primären Unbekannten.

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

import numpy as np

from Models.AHT_UA_LMTD import (
    AWTInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_awt,
    trace_model,
)

# ABSORBER_SPEC_MODE = "m11"
ABSORBER_SPEC_MODE = "T12"

# CYCLE_SCALE_SPEC_MODE = "m6"
CYCLE_SCALE_SPEC_MODE = "Qabs"

DESORBER_EVAPORATOR_ROUTING_MODE = "parallel"
#DESORBER_EVAPORATOR_ROUTING_MODE = "series_desorber_to_evaporator"
# DESORBER_EVAPORATOR_ROUTING_MODE = "series_evaporator_to_desorber"


def build_example_inputs() -> AWTInputs:
    common_kwargs = dict(
        T_11_C=85.0,
        # T_13_C=65.0,
        # T_15_C=65.0,
        T_17_C=20.0,
        m_13=4.0,
        m_15=4.0,
        m_17=4.0,
        UA_cond=10.0,
        UA_evap=15.0,
        UA_abs=10.0,
        UA_des=25.0,
        UA_shex=70.8 / 6.43,
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        absorber_spec_mode=ABSORBER_SPEC_MODE,
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
            "DESORBER_EVAPORATOR_ROUTING_MODE muss 'parallel', "
            "'series_desorber_to_evaporator' oder 'series_evaporator_to_desorber' sein."
        )

    spec_kwargs: dict[str, float] = {}

    if ABSORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 4  # 4, 0.2
    elif ABSORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 90  # 146, 80
    else:
        raise ValueError("ABSORBER_SPEC_MODE muss 'm11' oder 'T12' sein.")

    if CYCLE_SCALE_SPEC_MODE == "m6":
        spec_kwargs["m6_spec"] = 0.45  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qabs":
        spec_kwargs["Qabs_spec_kW"] = 60  # 184, 6.9
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE muss 'm6' oder 'Qabs' sein.")

    return AWTInputs(
        **common_kwargs,
        **spec_kwargs,
    )


if __name__ == "__main__":
    inputs = build_example_inputs()

    # Startvektor in der Reihenfolge:
    # [T8, T10, x3, x6, x20, T2, T4, beta]
    #
    # Benutzerangabe der Temperatur-Startwerte in °C.
    # Die Konvertierung in die internen Modell-Einheiten [K] erfolgt direkt darunter.
    x0 = primary_temperatures_C_to_K(
        np.array(
            [30.1458, 57.2958, 0.1787, 0.2003, 0.1981, 66.1183, 85.6828, 0.0905],
            dtype=float,
        )
    )

    trace = trace_model(x0, inputs)
    #print_trace(trace)

    result = solve_awt(inputs, x0=x0)
    print_summary(result)
