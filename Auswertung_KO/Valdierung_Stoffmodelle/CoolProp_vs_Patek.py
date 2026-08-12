import numpy as np
import CoolProp.CoolProp as CP

# Eigene Stofffunktionen
from Thermodynamic_Properties.libr_props import (
    w_libr_from_x,
    x_from_w_libr,
    h_solution_mass_kjkg,
    s_solution_mass_kjkgK,
    cp_solution_mass_kjkgk,
    rho_solution_mass,
)

# ==========================================
# Testbereiche
# ==========================================

temperatures = [293.15, 313.15, 333.15, 353.15, 373.15]   # K
mass_fractions = [0.45, 0.50, 0.55, 0.60]                 # LiBr-Massenanteil

print("=" * 110)
print(
    f"{'T [°C]':>7} {'w':>6} "
    f"{'h err [%]':>12} "
    f"{'s err [%]':>12} "
    f"{'cp err [%]':>12} "
    f"{'rho err [%]':>12}"
)
print("=" * 110)

for T in temperatures:
    for w in mass_fractions:

        x = x_from_w_libr(w)

        # -----------------------------
        # Eigene Patek-Implementierung
        # -----------------------------
        h_pat = h_solution_mass_kjkg(T, x)          # kJ/kg
        s_pat = s_solution_mass_kjkgK(T, x)         # kJ/kg/K
        cp_pat = cp_solution_mass_kjkgk(T, x)       # kJ/kg/K
        rho_pat = rho_solution_mass(T, x)           # kg/m³

        # -----------------------------
        # CoolProp
        # -----------------------------
        fluid = f"INCOMP::LiBr[{w}]"

        h_cp = CP.PropsSI("H", "T", T, "P", 101325, fluid) / 1000.0
        s_cp = CP.PropsSI("S", "T", T, "P", 101325, fluid) / 1000.0
        cp_cp = CP.PropsSI("C", "T", T, "P", 101325, fluid) / 1000.0
        rho_cp = CP.PropsSI("D", "T", T, "P", 101325, fluid)

        # -----------------------------
        # relative Fehler
        # -----------------------------
print(
    f"T={T-273.15:5.1f}°C  w={w:.2f}\n"
    f"  h:   Patek={h_pat:8.2f}   CP={h_cp:8.2f}   Δ={h_pat-h_cp:8.2f} kJ/kg\n"
    f"  s:   Patek={s_pat:8.4f}   CP={s_cp:8.4f}   Δ={s_pat-s_cp:8.4f} kJ/kgK\n"
    f"  cp:  Patek={cp_pat:8.4f}  CP={cp_cp:8.4f}  err={100*(cp_pat-cp_cp)/cp_cp:7.3f}%\n"
    f"  rho: Patek={rho_pat:8.2f} CP={rho_cp:8.2f} err={100*(rho_pat-rho_cp)/rho_cp:7.3f}%\n"
)
print(f"fluid = {fluid}")
print(f"h_cp   = {h_cp}")
print(f"s_cp   = {s_cp}")
print(f"cp_cp  = {cp_cp}")
print(f"rho_cp = {rho_cp}")

T1 = 333.15
T2 = 373.15

dh_pat = h_solution_mass_kjkg(T2, x) - h_solution_mass_kjkg(T1, x)

dh_cp = (
    CP.PropsSI("H","T",T2,"P",101325,fluid)
    - CP.PropsSI("H","T",T1,"P",101325,fluid)
) / 1000

print(dh_pat, dh_cp)