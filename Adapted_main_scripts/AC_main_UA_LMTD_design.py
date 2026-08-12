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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AC_UA_LMTD import (
    AKMInputs,
    primary_temperatures_C_to_K,
    print_summary,
    print_trace,
    solve_awt,
    trace_model,
)

EVAPORATOR_SPEC_MODE = "m17"
#EVAPORATOR_SPEC_MODE = "T18"

# CYCLE_SCALE_SPEC_MODE = "m1"
CYCLE_SCALE_SPEC_MODE = "Qeva"

# SHEX_MODEL_MODE = "UA"
SHEX_MODEL_MODE = "NTU"

ABSORBER_CONDENSER_ROUTING_MODE = "parallel"
#ABSORBER_CONDENSER_ROUTING_MODE = "series_absorber_to_condenser"
# ABSORBER_CONDENSER_ROUTING_MODE = "series_condenser_to_absorber"

def build_example_inputs() -> AKMInputs:
    common_kwargs = dict(
        T_11_C=90.0,   # 135, 60, 80
        #T_13_C=None,   # 120, 60
        #T_15_C=None,   # 120, 60
        T_17_C=11.0,   # 30, 20, 20
        m_11=0.7,      # 4, 0.2, 4
        m_13=1.7,      # 4, 0.2, 4
        m_15=1.5,      # 4, 0.2, 4
        UA_cond=19.88064798,  # 10, 1.0025, 25.2578
        UA_evap=7.805985208,  # 15, 1.5079, 11.3518
        UA_abs=10.03107026,      # 10, 1.5, 8.1355
        UA_des=9.935330421,   # 25, 2.4895, 10.4058
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        shex_model=SHEX_MODEL_MODE,
        evaporator_spec_mode=EVAPORATOR_SPEC_MODE,
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
        absorber_condenser_routing_mode=ABSORBER_CONDENSER_ROUTING_MODE,
    )

    if SHEX_MODEL_MODE == "UA":
        common_kwargs["UA_shex"] = 10.105   
    elif SHEX_MODEL_MODE == "NTU":
        common_kwargs["Effectiveness_shex"] = 0.9
    else:
        raise ValueError(
            "SHEX_MODEL_MODE muss 'UA' oder 'NTU' sein."
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
            "ABSORBER_CONDENSER_ROUTING_MODE muss 'parallel', "
            "'series_absorber_to_condenser' oder 'series_condenser_to_absorber' sein."
        )

    spec_kwargs: dict[str, float] = {}

    if EVAPORATOR_SPEC_MODE == "m17":
        spec_kwargs["m17_spec"] = 1.6  # 4, 0.2
    elif EVAPORATOR_SPEC_MODE == "T18":
        spec_kwargs["T18_spec_C"] = 5  # 146, 80
    else:
        raise ValueError("EVAPORATOR_SPEC_MODE muss 'm17' oder 'T18' sein.")

    if CYCLE_SCALE_SPEC_MODE == "m1":
        spec_kwargs["m1_spec"] = 0.37  # 1, 0.05, 0.236
    elif CYCLE_SCALE_SPEC_MODE == "Qeva":
        spec_kwargs["Qevap_spec_kW"] = 40.9  # 184, 6.9
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE muss 'm1' oder 'Qeva' sein.")

    return AKMInputs(
        **common_kwargs,
        **spec_kwargs,
    )


if __name__ == "__main__":
    inputs = build_example_inputs()

    # Startvektor in der Reihenfolge:
    # [T8, T10, x4, x1, T3, T5]
    #
    # Benutzerangabe der Temperatur-Startwerte in °C.
    # Die Konvertierung in die internen Modell-Einheiten [K] erfolgt direkt darunter.
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

    result = solve_awt(inputs, x0=x0)
    print_summary(result)
