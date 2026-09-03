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
# Q-T-Diagramme (Pinch-Analyse) nach der Lösung erzeugen?
ENABLE_QT_PLOT = True
QT_PLOT_SAVE_PATH = "Postprocessing/Plots/AHT_QT_Diagramme.png"  # None, um nicht zu speichern
# Dühring-Diagramm mit eingezeichnetem Betriebspunkt nach der Lösung erzeugen?
ENABLE_DUEHRING_PLOT = True
DUEHRING_PLOT_SAVE_PATH = "Postprocessing/Plots/AHT_Duehring_Diagramm.png"  # None, um nicht zu speichern
DUEHRING_PLOT_VARIANT = "mass"  # "mass" oder "mole"
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
        raise ValueError("CYCLE_SCALE_SPEC_MODE muss 'm6' oder 'Qabs' sein.")
    
    if ABSORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 4  # 4, 0.2
    elif ABSORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 146.02  # 146.02, 80
    else:
        raise ValueError("ABSORBER_SPEC_MODE muss 'm11' oder 'T12' sein.")

    if DESORBER_SPEC_MODE == "m13":
        spec_kwargs["m13_spec"] = 4  
    elif DESORBER_SPEC_MODE == "T14":
        spec_kwargs["T14_spec_C"] = 108.92  # 108.92
    else:
        raise ValueError("DESORBER_SPEC_MODE muss 'm13' oder 'T14' sein.")
        
    if EVAPORATOR_SPEC_MODE == "m15":
        spec_kwargs["m15_spec"] = 4 
    elif EVAPORATOR_SPEC_MODE == "T16":
        spec_kwargs["T16_spec_C"] = 108.80 # 108.80
    else:
        raise ValueError("EVAPORATOR_SPEC_MODE muss 'm15' oder 'T16' sein.")
    
    if CONDENSER_SPEC_MODE == "m17":
        spec_kwargs["m17_spec"] = 4  
    elif CONDENSER_SPEC_MODE == "T18":
        spec_kwargs["T18_spec_C"] = 41.26  # 41.26
    else:
        raise ValueError("CONDENSER_SPEC_MODE muss 'm17' oder 'T18' sein.")
    

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