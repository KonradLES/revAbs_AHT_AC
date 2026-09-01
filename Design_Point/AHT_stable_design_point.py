"""
Findet für einen fest vorgegebenen Betriebspunkt (externe Temperaturen + Leistung)
über eine Kontinuitaets-/Homotopie-Strategie:

  (a) einen stabilen Startvektor x0 fuer die 8 primaeren Unbekannten, und
  (b) den kleinsten simultan erreichbaren Satz an minimalen Pinch-Temperatur-
      differenzen (dT_min) fuer die 5 Waermeuebertrager (SHEX, Desorber,
      Kondensator, Verdampfer, Absorber),

sodass Betriebspunkt + x0 + dT_min anschliessend als Startpunkt in den
Bilevel-Optimierer uebergeben werden koennen.

Vorgehen (Homotopie):
  - Start bei sehr lockeren (grossen) dT_min-Werten -> Gleichungssystem ist
    "weich" und konvergiert praktisch unabhaengig vom Startvektor.
  - Homotopie-Parameter t in [0, 1] interpoliert linear zwischen den
    lockeren Startwerten (t=0) und den gewuenschten "Floor"-Zielwerten (t=1).
  - t wird schrittweise erhoeht, jede konvergierte Loesung dient als Warmstart
    (x0) fuer den naechsten, etwas schwierigeren Schritt.
  - Schlaegt ein Schritt fehl, wird die Schrittweite halbiert (Bisektion).
    Wird die Schrittweite zu klein, gilt der zuletzt konvergierte Punkt als
    die (praktische) Pinch-Grenze fuer diesen Betriebspunkt.

Diese Version verwendet direkt die echten Schnittstellen aus AHT_Pinch_Point.py:
  - initial_guess(inputs)       -> generische Startwert-Heuristik des Modells
  - solve_awt(...) -> AWTResult mit .primary_variables (dict!), .solve_info
    (success, final_point_evaluable, scaled_residual_norm) und .checks
  - Ein Loesungspunkt gilt hier erst dann als "konvergiert", wenn ALLE der
    folgenden Bedingungen erfuellt sind (rein scipy-success reicht nicht,
    da trf auch an einem unphysikalischen/instabilen Punkt "erfolgreich"
    terminieren kann, wenn die Soft-Residuen dort zufaellig klein sind):
      1. solve_info.success            == True
      2. solve_info.final_point_evaluable == True  (strenge Endauswertung ok)
      3. solve_info.scaled_residual_norm <= RESIDUAL_TOL
      4. alle result.checks-Werte      == True     (Konzentrationsreihenfolge,
         Kristallisationssicherheit, Massenbilanzen etc.)
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import dataclasses

import numpy as np

from Models.AHT_Pinch_Point import (
    AWTInputs,
    AWTResult,
    PRIMARY_VARIABLE_NAMES,
    initial_guess,
    primary_temperatures_K_to_C,
    print_summary,
    solve_awt,
)

# Schwelle fuer die skalierte Residuumsnorm, ab der ein Punkt als
# "konvergiert" gilt (nicht nur "von scipy terminiert").
RESIDUAL_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# 1) Betriebspunkt festlegen (bleibt ueber den gesamten Lauf KONSTANT)
#    -> hier deine externen Temperaturen und deine geforderte Leistung eintragen
# ---------------------------------------------------------------------------
OPERATING_POINT = dict(
    T_11_C=75.0,
    T_13_C=60.0,
    T_15_C=60.0,
    T_17_C=20.0,
    Qabs_spec_kW=500.4,
    T12_spec_C=80.02,
    T14_spec_C=53.92,
    T16_spec_C=53.80,
    T18_spec_C=26.26,
    cp_w_kJkgK=4.18,
    desorber_vapor_superheat_K=0.0,
    absorber_spec_mode="T12",
    desorber_spec_mode="T14",
    evaporator_spec_mode="T16",
    condenser_spec_mode="T18",
    cycle_scale_spec_mode="Qabs",
    desorber_evaporator_routing_mode="parallel",
)

# ---------------------------------------------------------------------------
# 2) Homotopie-Einstellungen fuer die Pinch-Temperaturdifferenzen
# ---------------------------------------------------------------------------
# "Locker" = garantiert unproblematischer Startpunkt (nicht aendern, ausser
# selbst das konvergiert bei dir nicht -> dann hier weiter erhoehen)
DT_MIN_LOOSE = dict(
    dT_min_shex=65.0,
    dT_min_des=65.0,
    dT_min_cond=65.0,
    dT_min_evap=65.0,
    dT_min_abs=65.0,
)

# "Floor" = das, was du eigentlich erreichen willst (ruhig aggressiv/klein
# waehlen - das Skript findet automatisch die reale Grenze, falls das nicht
# vollstaendig erreichbar ist)
DT_MIN_FLOOR = dict(
    dT_min_shex=1.0,
    dT_min_des=1.0,
    dT_min_cond=1.0,
    dT_min_evap=1.0,
    dT_min_abs=1.0,
)

T_STEP_INITIAL = 0.25       # initialer Homotopie-Schritt (Anteil von [0,1])
T_STEP_MIN = 1e-3           # Abbruch, wenn Schrittweite kleiner wird
MAX_ITERATIONS = 500


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _x0_from_result(result: AWTResult) -> np.ndarray:
    """Extrahiert den Loesungsvektor aus einem AWTResult in der Reihenfolge
    von PRIMARY_VARIABLE_NAMES, damit er direkt als x0 fuer den naechsten
    solve_awt-Aufruf (Warmstart) verwendet werden kann.
    """
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES],
        dtype=float,
    )


def _is_valid_solution(result: AWTResult) -> bool:
    """Strenge Konvergenzpruefung, siehe Modulbeschreibung oben."""
    info = result.solve_info
    if not info.success:
        return False
    if not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _with_dT_min(inputs: AWTInputs, dT_min: dict) -> AWTInputs:
    """Erzeugt eine Kopie von `inputs` mit aktualisierten dT_min-Werten."""
    try:
        return dataclasses.replace(inputs, **dT_min)
    except TypeError:
        # Fallback, falls AWTInputs kein dataclass ist
        data = vars(inputs).copy()
        data.update(dT_min)
        return AWTInputs(**data)


def _interp_dT_min(t: float) -> dict:
    """Lineare Interpolation zwischen lockeren und Ziel-dT_min-Werten."""
    return {
        key: (1.0 - t) * DT_MIN_LOOSE[key] + t * DT_MIN_FLOOR[key]
        for key in DT_MIN_LOOSE
    }


def _try_solve(inputs: AWTInputs, x0: np.ndarray):
    """Ruft solve_awt auf und prueft das Ergebnis streng (siehe _is_valid_solution).
    solve_awt wirft bei Nichtkonvergenz KEINE Exception, sondern liefert ein
    AWTResult mit entsprechend gesetzten solve_info-/checks-Feldern - deshalb
    wird hier trotzdem defensiv try/except verwendet (z. B. falls AWTInputs
    selbst schon bei der Konstruktion einen ValueError wirft).
    """
    try:
        result = solve_awt(inputs, x0=x0)
    except Exception as exc:
        print(f"    -> Exception bei solve_awt: {exc}")
        return False, None

    if not _is_valid_solution(result):
        info = result.solve_info
        print(
            f"    -> nicht konvergiert (success={info.success}, "
            f"final_point_evaluable={info.final_point_evaluable}, "
            f"residual_norm={info.scaled_residual_norm:.3e})"
        )
        if result.checks and not all(result.checks.values()):
            failed = [k for k, v in result.checks.items() if not v]
            print(f"       fehlgeschlagene Plausibilitaetschecks: {failed}")
        return False, None

    return True, result


def find_stable_operating_point(base_inputs: AWTInputs):
    """Homotopie-Lauf.

    Returns:
        best_inputs: AWTInputs mit den kleinsten erreichten dT_min
        best_x0: zugehoeriger konvergierter Loesungsvektor (Warmstart-faehig)
        best_dT: dict der erreichten dT_min-Werte
        best_result: letztes erfolgreiches solve_awt-Ergebnis
    """
    x0 = initial_guess(base_inputs)

    t = 0.0
    t_step = T_STEP_INITIAL

    best_inputs = _with_dT_min(base_inputs, _interp_dT_min(t))
    ok, result = _try_solve(best_inputs, x0)
    if not ok:
        raise RuntimeError(
            "Bereits die lockeren dT_min-Startwerte (DT_MIN_LOOSE) konvergieren "
            "nicht. Bitte DT_MIN_LOOSE weiter erhoehen (z. B. auf 35-40 K) oder "
            "den Betriebspunkt in OPERATING_POINT pruefen."
        )
    best_x0 = _x0_from_result(result)
    best_dT = _interp_dT_min(t)
    best_result = result

    print(f"[t=0.00] OK   dT_min={best_dT}")

    iteration = 0
    while t < 1.0 and iteration < MAX_ITERATIONS:
        iteration += 1
        t_trial = min(1.0, t + t_step)
        dT_trial = _interp_dT_min(t_trial)
        trial_inputs = _with_dT_min(base_inputs, dT_trial)

        print(f"[t={t_trial:.4f}] versuche dT_min={dT_trial}")
        ok, result = _try_solve(trial_inputs, best_x0)

        if ok:
            t = t_trial
            best_inputs = trial_inputs
            best_x0 = _x0_from_result(result)
            best_dT = dT_trial
            best_result = result
            # nach Erfolg Schrittweite wieder vorsichtig vergroessern
            t_step = min(T_STEP_INITIAL, t_step * 1.5)
            print("    -> OK")
        else:
            t_step /= 2.0
            if t_step < T_STEP_MIN:
                print(
                    f"\nAbbruch: minimale Schrittweite unterschritten. "
                    f"Letzter stabiler Punkt bei t={t:.4f}."
                )
                break

    if t < 1.0 - 1e-9:
        print(
            "\nHINWEIS: Die gewuenschten Floor-Werte (DT_MIN_FLOOR) sind fuer "
            "diesen Betriebspunkt NICHT vollstaendig erreichbar.\n"
            "Die unten ausgegebenen dT_min sind die kleinsten simultan "
            "erreichbaren Werte (praktische Pinch-Grenze dieses Betriebspunkts)."
        )
    else:
        print("\nZiel-dT_min (DT_MIN_FLOOR) vollstaendig erreicht.")

    return best_inputs, best_x0, best_dT, best_result


if __name__ == "__main__":
    base_inputs = AWTInputs(**OPERATING_POINT, **DT_MIN_LOOSE)

    stable_inputs, stable_x0, stable_dT_min, result = find_stable_operating_point(
        base_inputs
    )

    print("\n" + "=" * 70)
    print("ERGEBNIS - stabiler Betriebspunkt gefunden")
    print("=" * 70)

    print("\nMinimale simultan erreichbare Pinch-Temperaturdifferenzen:")
    for k, v in stable_dT_min.items():
        print(f"  {k:15s} = {v:.3f} K")

    stable_x0_C = primary_temperatures_K_to_C(stable_x0)
    print("\nStabiler Startvektor x0:")
    print(f"  intern (K bzw. -)      : {stable_x0}")
    print(f"  lesbar (Grad C bzw. -) : {dict(zip(PRIMARY_VARIABLE_NAMES, stable_x0_C))}")

    print("\nVollstaendige Ergebniszusammenfassung:")
    print_summary(result)