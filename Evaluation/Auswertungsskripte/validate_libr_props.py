"""
Validierung der LiBr/H2O-Stofffunktionen nach Patek (2006)
===========================================================
Referenz: Patek & Klomfar, Int. J. Refrigeration 29 (2006) 566–578
          Tabelle 9 — Validierungspunkte für Computerprogramme

Getestete Funktionen (molare Basis, direkte Patek-Gleichungen):
    calc_p_sat_patek    →  Sättigungsdruck      [Pa]
    calc_rho_molar_patek →  Molare Dichte        [mol/m³]
    calc_cp_molar_patek  →  Molare cp            [J/mol/K]
    calc_h_molar_patek   →  Molare Enthalpie     [J/mol]

Hinweis: Dieses Skript erwartet die Datei "libr_props.py" im selben Ordner.
         Benennen Sie "libr_props (2).py" entsprechend um.
"""

from Thermodynamic_Properties.libr_props import (
    calc_p_sat_patek,
    calc_rho_molar_patek,
    calc_cp_molar_patek,
    calc_h_molar_patek,
    T_sat_solution_from_p_x,  
    T_from_h_x_mass,           
    mixture_molar_mass,        # wird für h-Einheitenumrechnung benötigt
)

# ── Referenzpunkte aus Patek (2006), Tabelle 9 ───────────────────────────────
#  Format: (x [-], T [K], p [Pa], rho [mol/m³], cp [J/mol/K], h [J/mol])
REFERENZPUNKTE = [
    (0.05, 300,   3025.1805, 54148.9, 69.931,  1603.9),
    (0.05, 450, 835097.47,   48984.9, 74.047, 12189.0),
    (0.10, 300,   2286.4858, 52985.4, 65.520,  1445.1),
    (0.10, 450, 647702.12,   48550.2, 70.305, 11555.2),
    (0.30, 350,   2237.3986, 47826.4, 66.597,  9072.1),
    # x=0.40 ist die exakte Obergrenze — libr_props.py verwendet X_MAX = 0.399999
    # (der Term (0.4-x) wird im Nenner singulär), daher x=0.399999 als Näherung:
    (0.399999, 450,  43075.149,  45941.8, 70.294, 21024.4),
]

# ── Ausgabe ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("  Validierung libr_props.py  vs.  Patek (2006) Tabelle 9")
print("=" * 72)

for (x, T, p_ref, rho_ref, cp_ref, h_ref) in REFERENZPUNKTE:

    p_calc   = calc_p_sat_patek(T, x)
    rho_calc = calc_rho_molar_patek(T, x)
    cp_calc  = calc_cp_molar_patek(T, x)
    h_calc   = calc_h_molar_patek(T, x)

    dp   = (p_calc   - p_ref)   / p_ref   * 100
    drho = (rho_calc - rho_ref) / rho_ref * 100
    dcp  = (cp_calc  - cp_ref)  / cp_ref  * 100
    dh   = (h_calc   - h_ref)   / h_ref   * 100

    print(f"\n  x = {x:.2f},  T = {T:.0f} K")
    print(f"  {'Größe':<10} {'Referenz':>14} {'Berechnet':>14} {'Abw. [%]':>10}")
    print(f"  {'-'*50}")
    print(f"  {'p  [Pa]':<10} {p_ref:>14.4f} {p_calc:>14.4f} {dp:>+10.4f}")
    print(f"  {'ρ[mol/m³]':<10} {rho_ref:>14.1f} {rho_calc:>14.1f} {drho:>+10.4f}")
    print(f"  {'cp[J/molK]':<10} {cp_ref:>14.3f} {cp_calc:>14.3f} {dcp:>+10.4f}")
    print(f"  {'h  [J/mol]':<10} {h_ref:>14.1f} {h_calc:>14.1f} {dh:>+10.4f}")

print("\n" + "=" * 72)

# ── Inverse Funktionen ────────────────────────────────────────────────────────
# T_sat_solution_from_p_x(p, x)  →  T [K]   aus p [Pa] und x [-]
# T_from_h_x_mass(h_kjkg, x)     →  T [K]   aus h [kJ/kg] und x [-]
#
# Vorgehen: Referenzwerte p_ref bzw. h_ref als Eingabe nutzen,
#           berechnetes T mit dem bekannten T_ref vergleichen.

print("\n" + "=" * 72)
print("  Inverse Funktionen")
print("=" * 72)
print(f"\n  {'Funktion':<28} {'x':>6} {'T[K]':>6}  {'T_ref [K]':>10} {'T_calc [K]':>12} {'Abw. [K]':>10}")
print(f"  {'-' * 66}")

for (x, T, p_ref, rho_ref, cp_ref, h_ref) in REFERENZPUNKTE:

    # --- T_sat_solution_from_p_x ---
    # Eingabe: p_ref [Pa] und x [-]  →  Ausgabe: T [K]
    T_calc_p = T_sat_solution_from_p_x(p_ref, x)
    dT_p = T_calc_p - T                          # absolute Abweichung in K

    # --- T_from_h_x_mass ---
    # Die Funktion erwartet h in kJ/kg, die Referenztabelle gibt h in J/mol.
    # Umrechnung: h [kJ/kg] = h [J/mol] / (M_mix [kg/mol] * 1000)
    M_mix = mixture_molar_mass(x)
    h_kjkg = h_ref / (M_mix * 1000.0)

    # Eingabe: h_kjkg [kJ/kg] und x [-]  →  Ausgabe: T [K]
    T_calc_h = T_from_h_x_mass(h_kjkg, x)
    dT_h = T_calc_h - T                          # absolute Abweichung in K

    print(f"  {'T_sat_solution_from_p_x':<28} {x:>6.4f} {T:>6.0f}  {T:>10.4f} {T_calc_p:>12.4f} {dT_p:>+10.6f}")
    print(f"  {'T_from_h_x_mass':<28} {x:>6.4f} {T:>6.0f}  {T:>10.4f} {T_calc_h:>12.4f} {dT_h:>+10.6f}")
    print()

print("=" * 72)
print("  Erwartete Abweichung: < 0.001 K  (numerische Inversionsgenauigkeit)")
print("=" * 72)

print("  Fertig. Erwartete Abweichungen (Patek 2006):")
print("    p: ±2.1%  |  ρ: ±0.5%  |  cp: ±2%  |  h: ±10 kJ/kg")
print("=" * 72)
