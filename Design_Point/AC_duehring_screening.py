"""Reines Dühring-/Gleichgewichts-Screening für die AKM (single-effect
Absorptionskältemaschine) -- OHNE Solver.

Drittes Skript in der Dühring-Screening-Familie, nach AHT_duehring_screening.py
(single-lift AWT) und AHT_duehring_screening_double_lift.py (double-lift AWT).
Gleiche Philosophie, gleiche Bausteine (Pátek-Korrelationen aus
Thermodynamic_Properties.libr_props, Wasser-Sättigung aus Models.AHT_Pinch_Point,
Kristallisationsgrenze nach Albers/Boryta), aber andere Verschaltung: bei der
AKM liegen Desorber UND Kondensator auf der HOHEN Druckseite (p_high), Absorber
UND Verdampfer auf der NIEDRIGEN Druckseite (p_low) -- genau umgekehrt zum AWT.
Die "freie", sich ergebende Grösse ist hier nicht mehr die Absorbertemperatur
(wie beim AWT), sondern die Verdampfertemperatur T_evap, also die eigentliche
Kälteleistungs-Temperatur.

Kein Massen-/Energiebilanz-Solve, keine Kreislaufskalierung, kein Zirkulations-
verhältnis, kein SHEX -- exakt dieselben Vereinfachungen wie in den beiden AWT-
Skripten. Ergebnis ist wieder eine bewusst OPTIMISTISCHE obere/untere Schranke.

Physikalisches Bild
--------------------
- Desorber (Generator) + Kondensator bei p_high: Antriebswärme bei T_gen treibt
  die Lösung im Desorber aus (Konzentration x_strong), der Dampf kondensiert im
  Kondensator gegen Rückkühlwasser bei T_rueck.
- Verdampfer + Absorber bei p_low: Das Kältemittel (Wasser) verdampft bei
  T_evap und liefert die Kälteleistung. Die (unverdünnte) starke Lösung x_strong
  kommt -- ohne SHEX -- direkt vom Desorber zum Absorber, wird dort vom
  Rückkühlwasser (ebenfalls bei T_rueck) gekühlt und absorbiert den Dampf.

Zwei Rechenrichtungen werden angeboten, weil sie unterschiedliche Fragen
beantworten und unterschiedlich mit der Kristallisationsgrenze umgehen (siehe
unten):

1) VORWÄRTS -- estimate_min_evap_temperature(T_gen, T_rueck):
   "Was ist die tiefstmögliche Verdampfertemperatur bei gegebener Antriebs-
   und Rückkühltemperatur?" Direktes Analogon zur Fragestellung der beiden
   AWT-Skripte. ACHTUNG: bei den meisten realistischen (T_gen, T_rueck)-
   Kombinationen sackt die reine Gleichgewichtsschranke sehr schnell weit
   unter 0 °C und liefert daher als Diagramm wenig Information -- deshalb
   gibt es dafür hier bewusst keinen Standard-Plot mehr. Die Funktion bleibt
   aber verfügbar, z.B. für Einzelauswertungen oder um die Kristallisations-
   grenze an einem bestimmten Punkt zu prüfen.

2) RÜCKWÄRTS -- estimate_min_generator_temperature(T_evap_target, T_rueck):
   "Welche Antriebstemperatur brauche ich mindestens, um bei gegebener Rück-
   kühltemperatur eine bestimmte Ziel-Verdampfertemperatur noch zu erreichen?"
   Das ist praktisch meist die relevantere Frage (Vorauswahl: reicht meine
   Abwärmequelle für die geforderte Kühltemperatur, in Abhängigkeit von der
   Aussen-/Rückkühltemperatur übers Jahr?) und liefert i.d.R. deutlich mehr
   Struktur im Diagramm als Variante 1.

Kristallisation: zwei strukturell unterschiedliche Fälle
---------------------------------------------------------
Ohne SHEX ist der kälteste Punkt im Kreislauf, an dem die Lösung noch die
volle Konzentration x_strong trägt, NICHT (wie beim AWT) der Desorberaustritt,
sondern der ABSORBER: die vom heissen Desorber kommende starke Lösung wird
dort direkt vom (kalten) Rückkühlwasser gekühlt. Das ist genau der aus der
Praxis bekannte Kristallisationsfall (kaltes Rückkühlwasser im Winter). Die
beiden Rechenrichtungen behandeln eine Verletzung dieser Grenze deshalb
unterschiedlich:

- VORWÄRTS: x_strong ist unabhängig von T_rueck aus dem Desorber (T_gen,
  p_high) bestimmt. Ist es am Absorber (T_rueck-abhängig) zu konzentriert,
  wird x_strong -- wie in den AWT-Skripten -- auf die Löslichkeitsgrenze
  geklemmt und die Rechnung läuft mit der geklemmten (schwächeren) Lösung
  weiter (crystallization_limited=True). Sinnvoll, weil der Desorber schlicht
  mehr Konzentration liefern KÖNNTE, als am Absorber sicher ankommen darf.

- RÜCKWÄRTS: x_strong wird hier direkt AUS der Absorber-Randbedingung
  (T_evap_target, T_rueck) bestimmt, unabhängig von T_gen. Ist die dafür
  nötige Konzentration bei T_rueck bereits unlöslich, gibt es KEIN T_gen, das
  daran etwas ändert -- die Kombination (T_evap_target, T_rueck) ist dann
  strukturell infeasible, nicht nur "geklemmt".

Was hier NICHT abgebildet wird (bewusst, für Geschwindigkeit, siehe auch die
beiden AWT-Skripte): Massenstromaufteilung/Zirkulationsverhältnis, SHEX-
Wärmerückgewinnung, realer Gegenstrom-Temperaturverlauf über die Wärme-
übertrager (nur der jeweils bindende Pinch-Punkt), UA-Werte/Baugrösse.
Insbesondere die fehlende SHEX-Vorkühlung der starken Lösung macht die hier
berechnete Kristallisationsgefahr am Absorber KONSERVATIVER (pessimistischer)
als eine reale Anlage mit SHEX -- umgekehrt zur sonstigen "optimistischen
obere-Schranke"-Natur dieser Skriptfamilie. Das lohnt sich im Hinterkopf zu
behalten, wenn du die beiden Effekte gegeneinander abwägst.

Namenskonvention: T_gen/T_cond/T_abs/T_evap statt fester T-Nummern, da die
State-Point-Nummerierung deines AKM-Solvermodells hier nicht bekannt ist --
lässt sich 1:1 umbenennen, falls du eine feste Konvention hast.

Aufruf als Skript
-----------------
    python Design_Point/AKM_duehring_screening.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

import Thermodynamic_Properties.libr_props as lp
from Models.AHT_Pinch_Point import (
    celsius_to_kelvin,
    kelvin_to_celsius,
    water_p_sat_from_T,
    water_T_sat_from_p,
)

X_LO = 1.0e-6
X_HI = lp.X_MAX_PAT - 1.0e-6


# ---------------------------------------------------------------------------
# Kernfunktionen (Inversion der Dühring-Beziehung, in beide Richtungen)
# ---------------------------------------------------------------------------

def concentration_for_boiling_point(p_pa: float, T_target_K: float) -> float:
    """Invertiert T_sat_solution_from_p_x nach x: liefert x, sodass die
    Lösung bei p_pa genau bei T_target_K siedet. Identisch zu den
    gleichnamigen Funktionen in den beiden AWT-Skripten."""

    def f(x: float) -> float:
        return lp.T_sat_solution_from_p_x(p_pa, x) - T_target_K

    f_lo = f(X_LO)
    f_hi = f(X_HI)
    if f_lo > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K liegt unterhalb des Siedepunkts von "
            f"reinem Wasser bei p={p_pa:.1f} Pa -- keine Aufkonzentration möglich."
        )
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K ist bei p={p_pa:.1f} Pa mit keiner "
            f"LiBr-Konzentration im gültigen Bereich [{X_LO:.2e}, {X_HI:.6f}] "
            "erreichbar."
        )
    return float(brentq(f, X_LO, X_HI))


def pressure_for_boiling_point(
    x: float, T_target_K: float, p_floor: float = 1.0e-3
) -> float:
    """Invertiert T_sat_solution_from_p_x nach p (bei fester Konzentration x):
    liefert p, sodass die Lösung mit Konzentration x genau bei T_target_K
    siedet/im Gleichgewicht ist. Wird für die AKM gebraucht, um aus der
    Absorber-Randbedingung (T_abs_max, x_strong) den Verdampferdruck p_low zu
    bestimmen -- die Umkehrung dessen, was concentration_for_boiling_point
    macht.

    Obere Klammergrenze: der Druck, bei dem REINES Wasser exakt bei
    T_target_K siedet (Q=0.0) -- wegen der Siedepunktserhöhung durch LiBr
    siedet die Lösung bei DIESEM Druck erst bei einer höheren Temperatur,
    d.h. f(p_hi) = T_sat_solution(p_hi, x) - T_target_K > 0 für x > 0.

    Untere Klammergrenze: NICHT fest (z.B. 1 Pa), sondern adaptiv gesucht.
    Für schwache Konzentrationen (x nahe 0, z.B. verdünnte Lösung bei
    niedrigem T_gen) liegt der gesuchte Druck oft schon bei einigen 1000 Pa
    -- eine feste, sehr niedrige untere Grenze fällt dann leicht unter den
    gültigen Patek-Temperaturbereich [T_MIN_PAT, T_MAX_PAT] und
    T_sat_solution_from_p_x() wirft dort PropertyError, statt einen (sehr
    negativen) Wert zurückzugeben. Die Suche unten nutzt genau dieses
    Verhalten: sie bisEziert geometrisch (Drücke überspannen leicht mehrere
    Grössenordnungen) so lange, bis ein Druck gefunden ist, an dem die
    Auswertung noch gültig ist UND einen negativen Wert liefert -- dieser
    Punkt ist dann eine sichere, auswertbare untere Klammergrenze für
    brentq()."""

    p_hi = water_p_sat_from_T(T_target_K, Q=0.0)

    def f(p: float) -> float:
        return lp.T_sat_solution_from_p_x(p, x) - T_target_K

    try:
        f_hi = f(p_hi)
    except lp.PropertyError as exc:
        raise ValueError(
            f"T_target={T_target_K:.3f} K bei x={x:.4f} nicht auswertbar "
            f"(obere Klammergrenze p_hi={p_hi:.1f} Pa): {exc}"
        ) from exc
    if f_hi <= 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K ist für x={x:.4f} bereits am "
            f"Siedepunkt reinen Wassers (p_hi={p_hi:.1f} Pa) nicht "
            "überschritten -- unerwarteter/inkonsistenter Zustand."
        )

    low, high = p_floor, p_hi
    p_valid_negative = None
    for _ in range(80):
        mid = (low * high) ** 0.5  # geometrisches Mittel (Drücke über mehrere Grössenordnungen)
        try:
            f_mid = f(mid)
        except lp.PropertyError:
            # mid liegt unterhalb des gültigen Patek-Temperaturbereichs für
            # dieses x -- untere Grenze anheben.
            low = mid
            continue
        if f_mid >= 0.0:
            high = mid
        else:
            p_valid_negative = mid
            break
    else:
        p_valid_negative = None

    if p_valid_negative is None:
        raise ValueError(
            f"T_target={T_target_K:.3f} K ist für x={x:.4f} im gültigen "
            "Patek-Temperaturbereich mit keinem Druck erreichbar (auch die "
            "adaptive Suche nach einer unteren Klammergrenze ist nicht "
            "konvergiert)."
        )

    return float(brentq(f, p_valid_negative, p_hi))


# ---------------------------------------------------------------------------
# 1) VORWÄRTS: tiefstmögliche Verdampfertemperatur bei (T_gen, T_rueck)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinEvapResult:
    T_gen_C: float
    T_rueck_C: float
    dT_min_des: float
    dT_min_cond: float
    dT_min_abs: float

    feasible: bool
    message: str

    p_high_Pa: float = float("nan")
    p_low_Pa: float = float("nan")
    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_abs_max_C: float = float("nan")
    T_evap_min_C: float = float("nan")

    crystallization_safe: bool = True
    crystallization_message: str = ""
    crystallization_limited: bool = False


def estimate_min_evap_temperature(
    T_gen_C: float,
    T_rueck_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> MinEvapResult:
    """Optimistische UNTERE Schranke für die erreichbare Verdampfertemperatur
    bei gegebener Antriebstemperatur T_gen und Rückkühltemperatur T_rueck
    (gilt für Absorber und Kondensator gleichermassen). Siehe Modul-Docstring
    für die Kristallisationsbehandlung (geklemmt, nicht infeasible)."""

    common = dict(
        T_gen_C=T_gen_C, T_rueck_C=T_rueck_C,
        dT_min_des=dT_min_des, dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
    )

    # 1) Kondensatordruck aus T_rueck (Pinch: Kondensator muss wärmer sein
    #    als das Kühlwasser)
    T_cond_K = celsius_to_kelvin(T_rueck_C) + dT_min_cond
    p_high = water_p_sat_from_T(T_cond_K, Q=0.0)

    # 2) Desorber-Gleichgewicht: x_strong aus T_gen und p_high
    T_gen_eff_K = celsius_to_kelvin(T_gen_C) - dT_min_des
    try:
        x_strong = concentration_for_boiling_point(p_high, T_gen_eff_K)
    except ValueError as exc:
        return MinEvapResult(**common, feasible=False, message=str(exc), p_high_Pa=p_high)

    w_strong = lp.w_libr_from_x(x_strong)

    # 3) Absorber-Randbedingung: bestmögliche (kälteste) Absorbertemperatur
    #    durch das Rückkühlwasser
    T_abs_max_K = celsius_to_kelvin(T_rueck_C) + dT_min_abs
    T_abs_max_C = kelvin_to_celsius(T_abs_max_K)

    # 4) Kristallisationscheck an GENAU diesem Punkt: unverdünnte x_strong,
    #    gekühlt auf T_abs_max -- der kälteste Punkt im Kreislauf bei dieser
    #    Konzentration (kein SHEX modelliert).
    validity = lp.validate_solution_state(
        T_abs_max_K, w_strong, label="Absorbereintritt (AKM-Dühring-Screening)"
    )

    crystallization_limited = False
    if validity.crystallization_checked and not validity.crystallization_safe:
        # Klemmen auf die Löslichkeitsgrenze bei T_abs_max -- identische
        # Bisektionslogik wie in den beiden AWT-Skripten, nur an einer
        # anderen Temperatur ausgewertet.
        w_lo, w_hi = 0.57, w_strong
        for _ in range(60):
            w_mid = 0.5 * (w_lo + w_hi)
            if lp.validate_solution_state(T_abs_max_K, w_mid).crystallization_safe:
                w_lo = w_mid
            else:
                w_hi = w_mid

        x_strong = lp.x_from_w_libr(w_lo)
        w_strong = lp.w_libr_from_x(x_strong)
        validity = lp.validate_solution_state(
            T_abs_max_K, w_strong,
            label="Absorbereintritt (AKM-Dühring-Screening, an Löslichkeitsgrenze geklemmt)",
        )
        crystallization_limited = True

    partial_common = dict(
        **common, p_high_Pa=p_high,
        x_strong=x_strong, w_strong=w_strong, T_abs_max_C=T_abs_max_C,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
        crystallization_limited=crystallization_limited,
    )

    if not validity.crystallization_safe:
        # Sollte nach dem Klemmen praktisch nicht mehr auftreten, aber zur
        # Sicherheit (z.B. falls schon w=0.57 selbst unsicher wäre).
        return MinEvapResult(
            **partial_common, feasible=False,
            message=f"Kristallisationsrisiko (auch nach Klemmen): {validity.message}",
        )

    # 5) Verdampferdruck p_low aus (x_strong, T_abs_max) -- Umkehrung der
    #    Dühring-Beziehung nach p.
    try:
        p_low = pressure_for_boiling_point(x_strong, T_abs_max_K)
    except ValueError as exc:
        return MinEvapResult(**partial_common, feasible=False, message=str(exc))

    if p_low >= p_high:
        return MinEvapResult(
            **partial_common, feasible=False, p_low_Pa=p_low,
            message=(
                f"p_low ({p_low:.0f} Pa) >= p_high ({p_high:.0f} Pa): kein "
                "Druckgefälle -- T_gen zu niedrig relativ zu T_rueck."
            ),
        )

    # 6) Verdampfungstemperatur des reinen Kältemittels bei p_low
    T_evap_min_K = water_T_sat_from_p(p_low, Q=1.0)
    T_evap_min_C = kelvin_to_celsius(T_evap_min_K)

    message = (
        "OK (an Löslichkeitsgrenze geklemmt)." if crystallization_limited
        else "OK (optimistische untere Schranke)."
    )
    return MinEvapResult(
        **partial_common, feasible=True, message=message,
        p_low_Pa=p_low, T_evap_min_C=T_evap_min_C,
    )


# ---------------------------------------------------------------------------
# 2) RÜCKWÄRTS: minimale Antriebstemperatur für eine Ziel-Verdampfertemperatur
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinGenResult:
    T_evap_target_C: float
    T_rueck_C: float
    dT_min_des: float
    dT_min_cond: float
    dT_min_abs: float

    feasible: bool
    message: str

    p_high_Pa: float = float("nan")
    p_low_Pa: float = float("nan")
    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_abs_max_C: float = float("nan")
    T_gen_min_C: float = float("nan")

    crystallization_safe: bool = True
    crystallization_message: str = ""


def estimate_min_generator_temperature(
    T_evap_target_C: float,
    T_rueck_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> MinGenResult:
    """Minimale Antriebstemperatur T_gen, die -- rein gleichgewichtsseitig --
    nötig ist, um bei gegebener Rückkühltemperatur T_rueck die Ziel-
    Verdampfertemperatur T_evap_target noch zu erreichen. Anders als bei
    estimate_min_evap_temperature() führt eine Kristallisationsverletzung
    hier zu einem echten Infeasible (siehe Modul-Docstring) statt zu einem
    Klemmen, weil x_strong hier direkt aus der Absorber-Randbedingung folgt,
    unabhängig von T_gen."""

    common = dict(
        T_evap_target_C=T_evap_target_C, T_rueck_C=T_rueck_C,
        dT_min_des=dT_min_des, dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
    )

    # 1) Verdampferdruck direkt aus der Ziel-Verdampfertemperatur
    p_low = water_p_sat_from_T(celsius_to_kelvin(T_evap_target_C), Q=1.0)

    # 2) Absorber-Randbedingung durch das Rückkühlwasser
    T_abs_max_K = celsius_to_kelvin(T_rueck_C) + dT_min_abs
    T_abs_max_C = kelvin_to_celsius(T_abs_max_K)

    # 3) Welche Konzentration wird am Absorber mindestens gebraucht, um bei
    #    p_low noch zu absorbieren, wenn der Absorber nicht kälter als
    #    T_abs_max werden kann?
    try:
        x_strong = concentration_for_boiling_point(p_low, T_abs_max_K)
    except ValueError as exc:
        return MinGenResult(
            **common, feasible=False, message=str(exc),
            p_low_Pa=p_low, T_abs_max_C=T_abs_max_C,
        )

    w_strong = lp.w_libr_from_x(x_strong)

    # 4) Kristallisationscheck an exakt diesem (T_abs_max, w_strong) -- HIER
    #    kein Klemmen: wenn das bereits unsicher ist, hilft kein T_gen der
    #    Welt (x_strong ist ja schon die gesuchte, feste Randbedingung).
    validity = lp.validate_solution_state(
        T_abs_max_K, w_strong,
        label="Absorbereintritt (AKM-Dühring-Screening, rückwärts)",
    )

    partial_common = dict(
        **common, p_low_Pa=p_low, x_strong=x_strong, w_strong=w_strong,
        T_abs_max_C=T_abs_max_C,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
    )

    if not validity.crystallization_safe:
        return MinGenResult(
            **partial_common, feasible=False,
            message=(
                f"T_evap_target={T_evap_target_C:.2f} °C ist bei "
                f"T_rueck={T_rueck_C:.2f} °C mit KEINEM T_gen erreichbar -- "
                f"Kristallisationsrisiko am Absorber: {validity.message}"
            ),
        )

    # 5) Kondensatordruck (unabhängig von T_evap_target, nur von T_rueck)
    T_cond_K = celsius_to_kelvin(T_rueck_C) + dT_min_cond
    p_high = water_p_sat_from_T(T_cond_K, Q=0.0)

    if p_high <= p_low:
        return MinGenResult(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=(
                f"p_high ({p_high:.0f} Pa) <= p_low ({p_low:.0f} Pa): "
                "T_evap_target liegt nicht unter der Kondensatorseite -- "
                "unplausible Kombination."
            ),
        )

    # 6) Nötige Desorbertemperatur, um x_strong bei p_high zu erzeugen
    try:
        T_gen_min_K = lp.T_sat_solution_from_p_x(p_high, x_strong) + dT_min_des
    except Exception as exc:
        return MinGenResult(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=f"Desorber-Gleichgewichtstemperatur nicht berechenbar: {exc}",
        )

    T_gen_min_C = kelvin_to_celsius(T_gen_min_K)
    return MinGenResult(
        **partial_common, feasible=True, message="OK (optimistische untere Schranke).",
        p_high_Pa=p_high, T_gen_min_C=T_gen_min_C,
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_generator_temperature(
    T_gen_values_C: Sequence[float],
    T_rueck_C: float,
    **kwargs,
) -> List[MinEvapResult]:
    """Vorwärts: T_evap_min über T_gen, bei fester Rückkühltemperatur."""
    return [estimate_min_evap_temperature(t, T_rueck_C, **kwargs) for t in T_gen_values_C]


def sweep_recool_temperature(
    T_rueck_values_C: Sequence[float],
    T_evap_target_C: float,
    **kwargs,
) -> List[MinGenResult]:
    """Rückwärts: T_gen_min über T_rueck, bei fester Ziel-Verdampfertemperatur."""
    return [estimate_min_generator_temperature(T_evap_target_C, t, **kwargs) for t in T_rueck_values_C]


def print_min_evap_table(results: Sequence[MinEvapResult]) -> None:
    print("=" * 110)
    print(
        f"{'T_gen[C]':>9} {'T_rueck[C]':>10} {'x_strong':>9} "
        f"{'T_evap_min[C]':>14} {'Kristall.':>10}  Hinweis"
    )
    print("-" * 110)
    for r in results:
        if not r.crystallization_safe:
            krist = "RISIKO"
        elif r.crystallization_limited:
            krist = "geklemmt"
        else:
            krist = "sicher"
        print(
            f"{r.T_gen_C:9.2f} {r.T_rueck_C:10.2f} {r.x_strong:9.4f} "
            f"{r.T_evap_min_C:14.2f} {krist:>10}  {r.message}"
        )
    print("=" * 110)


def print_min_gen_table(results: Sequence[MinGenResult]) -> None:
    print("=" * 110)
    print(
        f"{'T_evap_Ziel[C]':>15} {'T_rueck[C]':>10} {'x_strong':>9} "
        f"{'T_gen_min[C]':>13} {'Kristall.':>10}  Hinweis"
    )
    print("-" * 110)
    for r in results:
        krist = "sicher" if r.crystallization_safe else "RISIKO"
        print(
            f"{r.T_evap_target_C:15.2f} {r.T_rueck_C:10.2f} {r.x_strong:9.4f} "
            f"{r.T_gen_min_C:13.2f} {krist:>10}  {r.message}"
        )
    print("=" * 110)


def plot_min_gen_vs_recool_temperature(
    T_rueck_values_C: Sequence[float],
    T_evap_target_values_C: Sequence[float],
    *,
    save_path: Optional[str] = "Design_Point/Plots/duehring_screening_AKM_Tgen_vs_Trueck.png",
    show: bool = True,
    **kwargs,
):
    """RÜCKWÄRTS-Plot (empfohlen als Hauptdiagramm): T_gen_min vs. T_rueck,
    eine Kurve je Ziel-Verdampfertemperatur T_evap_target."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    all_results = {}
    for T_evap_target_C in T_evap_target_values_C:
        results = sweep_recool_temperature(T_rueck_values_C, T_evap_target_C, **kwargs)
        all_results[T_evap_target_C] = results

        x = np.array([r.T_rueck_C for r in results])
        y = np.array([r.T_gen_min_C for r in results])
        ok = np.array([r.feasible for r in results])

        (line,) = ax.plot(x[ok], y[ok], "o-", label=f"T_evap,Ziel = {T_evap_target_C:.0f} °C")
        if np.any(~ok):
            ax.plot(
                x[~ok], y[~ok], "x", color=line.get_color(), markersize=8, markeredgewidth=2,
            )

    ax.set_xlabel("Rückkühltemperatur T_rueck (Absorber & Kondensator) [°C]")
    ax.set_ylabel("Minimal nötige Antriebstemperatur T_gen,min [°C]\n(optimistische untere Schranke)")
    ax.set_title("AKM Dühring-Screening: nötige Antriebstemperatur vs. Rückkühltemperatur")
    ax.grid(alpha=0.4)
    ax.legend(title="× = mit keinem T_gen erreichbar\n(Kristallisation am Absorber)")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")
    if show:
        plt.show()
    return fig, ax, all_results


if __name__ == "__main__":
    DT_MIN_DES = 5.0
    DT_MIN_COND = 5.0
    DT_MIN_ABS = 5.0

    # Rückwärts: die praktisch relevantere Darstellung (siehe Modul-Docstring)
    T_RUECK_RANGE_C = list(np.arange(15.0, 40.0, 2.5))
    T_EVAP_TARGET_CURVES_C = [3.0, 5.0, 7.0, 10.0]

    print("AKM Dühring-Screening (rückwärts) für T_evap_target = 5 °C:")
    results_bwd = sweep_recool_temperature(
        T_RUECK_RANGE_C, T_evap_target_C=5.0,
        dT_min_des=DT_MIN_DES, dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )
    print_min_gen_table(results_bwd)

    plot_min_gen_vs_recool_temperature(
        T_RUECK_RANGE_C, T_EVAP_TARGET_CURVES_C,
        dT_min_des=DT_MIN_DES, dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )