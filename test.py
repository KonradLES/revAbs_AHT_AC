try:
    import CoolProp.CoolProp as CP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CoolProp ist nicht installiert. Installation z. B. mit `pip install CoolProp`."
    ) from exc

import Thermodynamic_Properties.libr_props as lp

def water_p_sat_from_T(T_K: float, Q: float) -> float:
    return CP.PropsSI("P", "T", T_K, "Q", Q, "Water")

T8 = 32+273.15
x1=0.2185
p_high = water_p_sat_from_T(T8, Q=0.0)
T7 = lp.T_sat_solution_from_p_x(p_high, x1) 

print(f"T7: {T7-273.15:.2f} °C, p_high: {p_high} Pa")