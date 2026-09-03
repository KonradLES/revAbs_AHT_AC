"""Reines Dühring-/Gleichgewichts-Screening für den DOUBLE-LIFT AWT -- OHNE Solver.

Direktes Analogon zu AHT_duehring_screening.py (single-lift), erweitert um eine
zweite Verdampfer-Absorber-Stufe. Beantwortet dieselbe Frage wie das Single-Lift-
Skript ("welcher GTL ist bei welcher Abwärmetemperatur thermodynamisch überhaupt
maximal drin?"), diesmal für die zweistufige ("double-lift") AWT-Bauart, bevor
irgendein Pinch-Modell oder Optimierer angeworfen wird. Nutzt ausschliesslich:

  - die LiBr/H2O-Gleichgewichtsbeziehung (Pátek-Korrelationen in
    Thermodynamic_Properties.libr_props),
  - die Kristallisationsgrenze nach Albers/Boryta,
  - die Wasser-Sättigungsfunktionen aus Models.AHT_Pinch_Point (identische
    CoolProp-Quelle wie im eigentlichen Solvermodell, daher konsistent).

Kein Massen-/Energiebilanz-Solve, keine Kreislaufskalierung, kein Zirkulations-
verhältnis -- exakt dieselben Vereinfachungen wie im single-lift Skript, siehe
dessen Docstring. Das Ergebnis ist auch hier bewusst eine OPTIMISTISCHE
Abschätzung (Pinch nur am jeweils bindenden Ende, nicht über den vollen
Gegenstrom-Temperaturverlauf).

Physikalisches Bild (Double-Lift AWT, "serial flow" Bauart nach Saito et al.
2015 / Lubis et al. 2017, siehe auch den Übersichtsartikel Cudok et al. 2021,
"Absorption heat transformer - state-of-the-art of industrial applications",
Renew. Sustain. Energy Rev. 141, 110757, Fig. 2 rechte Seite)
-------------------------------------------------------------------------------
Ein Double-Lift-AWT hat -- im Unterschied zum single-lift AWT mit 2 Druck-
niveaus -- DREI Druckniveaus, aber weiterhin nur EINEN Desorber und EINEN
Kondensator:

  - Desorber (G) + Kondensator (C) auf dem NIEDRIGSTEN Druckniveau p_low:
    Abwärme bei T13 treibt die Lösung im Desorber aus (wie im single-lift
    Fall), der Dampf kondensiert bei T17 (Rückkühlung). Der Desorber liefert
    die stark aufkonzentrierte Lösung x_strong -- identisch zur single-lift
    Herleitung (Schritt 1-3 unten sind wortwörtlich dieselben Gleichungen).

  - Verdampfer/Absorber-Verbund NIEDRIGER Stufe (EL/AL) auf mittlerem
    Druckniveau p_mid: EL wird -- wie der einzelne Verdampfer im single-lift
    Fall -- von der externen Abwärme bei T15 gespeist. AL absorbiert diesen
    Dampf mit der (in dieser Näherung unverdünnten) starken Lösung x_strong
    und liefert dabei eine erste angehobene Temperatur T_AL -- das ist exakt
    der "GTL" des single-lift Falls, hier aber nur die ERSTE von zwei Stufen.

  - Verdampfer/Absorber-Verbund HOHER Stufe (EH/AH) auf dem höchsten Druck-
    niveau p_high: Der Clou des Double-Lift-Zyklus ist, dass AL NICHT die
    Nutzwärme nach aussen abgibt, sondern intern EH antreibt ("the low
    pressure absorber AL drives the higher pressure evaporator EH by internal
    heat exchange", Cudok et al. 2021). AH absorbiert diesen zweiten Dampf-
    strom -- wieder mit x_strong -- und liefert erst hier, bei T_AH, die
    tatsächlich nutzbare Wärme. Der Gesamt-GTL bezogen auf die Abwärme T15
    ist damit die Summe zweier "Lifts" (T_AL - T15) + (T_AH - T_AL).

Vereinfachung ggü. der realen "serial flow" Schaltung (WICHTIG, unbedingt
lesen)
-------------------------------------------------------------------------------
In der realen serial-flow Schaltung durchläuft EIN Lösungsstrom zuerst AH
(dort noch unverdünnt, x_strong) und erst danach -- über ein Druckminderventil
auf p_mid entspannt und bereits teilweise verdünnt -- AL. Für AL stünde also
real eine SCHWÄCHERE Konzentration zur Verfügung als x_strong, was den in AL
erreichbaren GTL1 gegenüber der hier berechneten Schranke reduzieren würde.

Diese Kopplung liesse sich nur mit einer Massenbilanz (Zirkulationsverhältnis)
auflösen -- exakt das, was auch das single-lift Skript bewusst weglässt. Um
beide Skripte strukturell und im Vereinfachungsgrad vergleichbar zu halten,
wird hier stattdessen die (parallele) Näherung getroffen, dass sowohl AL als
auch AH mit der vollen, unverdünnten Konzentration x_strong aus dem Desorber
gespeist werden (entspricht in der Literatur der "parallel feed" Variante
des Double-/Dual-Absorption-Heat-Transformers). Das macht die hier berechnete
GTL_total-Schranke NOCH optimistischer als eine reale serial-flow Auslegung
liefern würde -- also weiterhin eine gültige, nur eben etwas grosszügigere
obere Schranke. Das volle Pinch-Modell mit Massenbilanz liefert danach die
tatsächlich erreichbare, engere Grenze (analog zur Rolle von
AHT_feasibility_sweep.py / AHT_design_point_optimizer.py für den single-lift
Fall).

Kristallisation wird -- wie im single-lift Skript -- nur am Desorberaustritt
(T_gen, w_strong) geprüft: das ist der kälteste Punkt im gesamten Kreislauf,
an dem die Lösung die Konzentration x_strong trägt, und damit der bindende
Kristallisationscheck (AL und AH liegen bei gleicher Konzentration, aber
höherer Temperatur, also unkritischer).

Was hier NICHT abgebildet wird (bewusst, für Geschwindigkeit, siehe auch
single-lift Skript):
  - Massenstromaufteilung / Zirkulationsverhältnis (FR) je Stufe
  - die reale Verdünnung der Lösung beim Durchlauf AH -> AL (siehe oben)
  - SHEX-Wärmerückgewinnung, Vorabsorption
  - der tatsächliche Temperaturverlauf über die Wärmeübertrager (nur der
    jeweils bindende Pinch-Punkt wird betrachtet, nicht LMTD/Gegenstrom)
  - UA-Werte / Baugrösse

-> Ergebnis ist eine OBERE Schranke, optimistischer als die reale serial-flow
   Bauart. Ein vollständiges Pinch-Modell mit Massenbilanz liefert danach die
   tatsächlich erreichbare, engere Grenze.

Aufruf als Skript
-----------------
    python Design_Point/AHT_duehring_screening_double_lift.py
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
)

X_LO = 1.0e-6
X_HI = lp.X_MAX_PAT - 1.0e-6


# ---------------------------------------------------------------------------
# Kernfunktionen
# ---------------------------------------------------------------------------

def concentration_for_boiling_point(p_pa: float, T_target_K: float) -> float:
    """Invertiert T_sat_solution_from_p_x: liefert x, sodass die Lösung bei
    p_pa genau bei T_target_K siedet. Wirft ValueError, wenn T_target_K
    ausserhalb des bei diesem Druck erreichbaren Bereichs liegt.

    Identisch zur gleichnamigen Funktion in AHT_duehring_screening.py
    (single-lift) -- hier dupliziert, damit dieses Skript eigenständig
    lauffähig bleibt, ohne einen Import aus dem single-lift Modul zu
    benötigen."""

    def f(x: float) -> float:
        return lp.T_sat_solution_from_p_x(p_pa, x) - T_target_K

    f_lo = f(X_LO)
    f_hi = f(X_HI)
    if f_lo > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K liegt unterhalb des Siedepunkts von "
            f"reinem Wasser bei p={p_pa:.1f} Pa -- keine Aufkonzentration möglich "
            "(Antriebstemperatur zu niedrig relativ zu diesem Druckniveau)."
        )
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"T_target={T_target_K:.3f} K ist bei p={p_pa:.1f} Pa mit keiner "
            f"LiBr-Konzentration im gültigen Bereich [{X_LO:.2e}, {X_HI:.6f}] "
            "erreichbar (Antriebstemperatur zu hoch / ausserhalb Patek-Bereich)."
        )
    return float(brentq(f, X_LO, X_HI))


@dataclass(frozen=True)
class DuehringScreeningResultDoubleLift:
    T13_C: float
    T15_C: float
    T17_C: float
    dT_min_des: float
    dT_min_evap: float
    dT_min_cond: float
    dT_min_abs: float
    dT_min_evap2: float
    dT_min_abs2: float

    feasible: bool
    message: str

    p_low_Pa: float = float("nan")
    p_mid_Pa: float = float("nan")
    p_high_Pa: float = float("nan")

    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_gen_C: float = float("nan")

    # Stufe 1 (EL/AL, mittleres Druckniveau)
    T10L_C: float = float("nan")
    T_AL_max_C: float = float("nan")
    GTL1_max_K: float = float("nan")

    # Stufe 2 (EH/AH, höchstes Druckniveau) -- von AL intern angetrieben
    T10H_C: float = float("nan")
    T_AH_max_C: float = float("nan")
    GTL2_max_K: float = float("nan")

    # Gesamt-Lift bezogen auf die Abwärme T15 (= GTL1_max_K + GTL2_max_K)
    GTL_total_max_K: float = float("nan")

    crystallization_safe: bool = True
    crystallization_message: str = ""
    # True, wenn x_strong auf die Löslichkeitsgrenze geklemmt wurde -- siehe
    # Kommentar in estimate_max_gtl_double_lift() bzw. im single-lift Skript.
    crystallization_limited: bool = False


def estimate_max_gtl_double_lift(
    T13_C: float,
    T15_C: float,
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    dT_min_evap2: float = 5.0,
    dT_min_abs2: float = 5.0,
) -> DuehringScreeningResultDoubleLift:
    """Optimistische obere Schranke für den erreichbaren Gesamt-GTL eines
    Double-Lift-AWT, siehe Modul-Docstring.

    dT_min_des/_evap/_cond/_abs beziehen sich -- wie im single-lift Skript --
    auf Desorber, Verdampfer der Stufe 1 (EL), Kondensator bzw. Absorber der
    Stufe 1 (AL). dT_min_evap2/_abs2 sind die analogen Pinch-Annahmen für den
    intern angetriebenen Verdampfer (EH) bzw. Endabsorber (AH) der Stufe 2.
    """

    common = dict(
        T13_C=T13_C, T15_C=T15_C, T17_C=T17_C,
        dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
        dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
        dT_min_evap2=dT_min_evap2, dT_min_abs2=dT_min_abs2,
    )

    # 1) Kondensatordruck aus T17 (Pinch am kalten Ende) -- identisch single-lift
    T8_K = celsius_to_kelvin(T17_C) + dT_min_cond
    p_low = water_p_sat_from_T(T8_K, Q=0.0)

    # 2) Verdampferdruck Stufe 1 aus T15 (Pinch am heissen Ende, optimistisch)
    T10L_K = celsius_to_kelvin(T15_C) - dT_min_evap
    try:
        p_mid = water_p_sat_from_T(T10L_K, Q=1.0)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False,
            message=f"Verdampferdruck Stufe 1 (EL) nicht berechenbar: {exc}",
            p_low_Pa=p_low,
        )

    if p_mid <= p_low:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False,
            message=(
                f"p_mid ({p_mid:.0f} Pa) <= p_low ({p_low:.0f} Pa): "
                "Abwärmetemperatur T15 zu niedrig relativ zur Rückkühlung T17 "
                "-- Stufe 1 (EL/AL) kann so nicht angetrieben werden."
            ),
            p_low_Pa=p_low, p_mid_Pa=p_mid,
        )

    # 3) Desorber-Gleichgewicht: x_strong aus T13 und p_low -- identisch single-lift
    T_gen_K = celsius_to_kelvin(T13_C) - dT_min_des
    try:
        x_strong = concentration_for_boiling_point(p_low, T_gen_K)
    except ValueError as exc:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False, message=str(exc),
            p_low_Pa=p_low, p_mid_Pa=p_mid,
            T_gen_C=kelvin_to_celsius(T_gen_K),
            T10L_C=kelvin_to_celsius(T10L_K),
        )

    w_strong = lp.w_libr_from_x(x_strong)

    # 4) Kristallisationscheck am Desorberaustritt (kältester Punkt bei x_strong)
    validity = lp.validate_solution_state(
        T_gen_K, w_strong, label="Desorberaustritt (Double-Lift-Dühring-Screening)"
    )

    # 4b) Auf die Löslichkeitsgrenze klemmen, statt den Punkt zu verwerfen --
    # identische Logik wie im single-lift Skript (dort ausführlich kommentiert).
    crystallization_limited = False
    if validity.crystallization_checked and not validity.crystallization_safe:
        w_lo, w_hi = 0.57, w_strong
        for _ in range(60):
            w_mid = 0.5 * (w_lo + w_hi)
            if lp.validate_solution_state(T_gen_K, w_mid).crystallization_safe:
                w_lo = w_mid
            else:
                w_hi = w_mid

        x_strong = lp.x_from_w_libr(w_lo)
        w_strong = lp.w_libr_from_x(x_strong)
        validity = lp.validate_solution_state(
            T_gen_K, w_strong,
            label="Desorberaustritt (Double-Lift-Dühring-Screening, an Löslichkeitsgrenze geklemmt)",
        )
        crystallization_limited = True

    # 5) Stufe 1: AL mit x_strong bei p_mid -> T_AL_max (= "GTL1", analog zum
    #    T12_max des single-lift Skripts)
    try:
        T_AL_solution_K = lp.T_sat_solution_from_p_x(p_mid, x_strong)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **common, feasible=False,
            message=f"Absorber-Gleichgewichtstemperatur Stufe 1 (AL) nicht berechenbar: {exc}",
            p_low_Pa=p_low, p_mid_Pa=p_mid,
            x_strong=x_strong, w_strong=w_strong,
            T_gen_C=kelvin_to_celsius(T_gen_K), T10L_C=kelvin_to_celsius(T10L_K),
            crystallization_safe=validity.crystallization_safe,
            crystallization_message=validity.message,
            crystallization_limited=crystallization_limited,
        )

    T_AL_max_K = T_AL_solution_K - dT_min_abs
    T_AL_max_C = kelvin_to_celsius(T_AL_max_K)
    GTL1_max_K = T_AL_max_C - T15_C

    partial_common = dict(
        **common,
        p_low_Pa=p_low, p_mid_Pa=p_mid,
        x_strong=x_strong, w_strong=w_strong,
        T_gen_C=kelvin_to_celsius(T_gen_K), T10L_C=kelvin_to_celsius(T10L_K),
        T_AL_max_C=T_AL_max_C, GTL1_max_K=GTL1_max_K,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
        crystallization_limited=crystallization_limited,
    )

    if not validity.crystallization_safe:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False,
            message=f"Kristallisationsrisiko: {validity.message}",
        )
    if GTL1_max_K <= 0.0:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False,
            message=(
                f"T_AL_max ({T_AL_max_C:.2f} °C) liegt nicht über T15 "
                f"({T15_C:.2f} °C) -- Stufe 1 liefert keinen positiven Lift, "
                "Stufe 2 kann so nicht angetrieben werden."
            ),
        )

    # 6) Verdampferdruck Stufe 2 (EH): angetrieben von AL, Pinch am heissen Ende
    T10H_K = T_AL_max_K - dT_min_evap2
    try:
        p_high = water_p_sat_from_T(T10H_K, Q=1.0)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False,
            message=f"Verdampferdruck Stufe 2 (EH) nicht berechenbar: {exc}",
        )

    if p_high <= p_mid:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=(
                f"p_high ({p_high:.0f} Pa) <= p_mid ({p_mid:.0f} Pa): "
                "T_AL_max zu niedrig relativ zu p_mid, um EH (Stufe 2) über den "
                "hier angesetzten Pinch dT_min_evap2 anzutreiben."
            ),
        )

    # 7) Stufe 2: AH mit x_strong bei p_high -> T_AH_max (finale Nutzwärme)
    try:
        T_AH_solution_K = lp.T_sat_solution_from_p_x(p_high, x_strong)
    except Exception as exc:
        return DuehringScreeningResultDoubleLift(
            **partial_common, feasible=False, p_high_Pa=p_high,
            message=f"Absorber-Gleichgewichtstemperatur Stufe 2 (AH) nicht berechenbar: {exc}",
        )

    T_AH_max_K = T_AH_solution_K - dT_min_abs2
    T_AH_max_C = kelvin_to_celsius(T_AH_max_K)
    GTL2_max_K = T_AH_max_C - T_AL_max_C
    GTL_total_max_K = T_AH_max_C - T15_C

    feasible = validity.crystallization_safe and GTL1_max_K > 0.0 and GTL2_max_K > 0.0
    if GTL2_max_K <= 0.0:
        message = (
            f"T_AH_max ({T_AH_max_C:.2f} °C) liegt nicht über T_AL_max "
            f"({T_AL_max_C:.2f} °C) -- Stufe 2 liefert keinen positiven "
            "Zusatzlift."
        )
    elif crystallization_limited:
        message = "OK (an Löslichkeitsgrenze geklemmt, nicht voller Desorber-Pinch ausgenutzt)."
    else:
        message = "OK (optimistische obere Schranke, parallel-feed Näherung)."

    return DuehringScreeningResultDoubleLift(
        **partial_common, feasible=feasible, message=message,
        p_high_Pa=p_high, T10H_C=kelvin_to_celsius(T10H_K),
        T_AH_max_C=T_AH_max_C, GTL2_max_K=GTL2_max_K,
        GTL_total_max_K=GTL_total_max_K,
    )


# ---------------------------------------------------------------------------
# Sweep über Abwärmetemperatur (T13 = T15, "parallel"-Fall, wie single-lift)
# ---------------------------------------------------------------------------

def sweep_waste_heat_temperature_double_lift(
    T_waste_values_C: Sequence[float],
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    dT_min_evap2: float = 5.0,
    dT_min_abs2: float = 5.0,
) -> List[DuehringScreeningResultDoubleLift]:
    """Setzt T13 = T15 = T_waste (parallele Verschaltung von Desorber und
    Verdampfer der Stufe 1) und wertet estimate_max_gtl_double_lift() für
    jeden Wert aus T_waste_values_C aus."""
    return [
        estimate_max_gtl_double_lift(
            T13_C=t, T15_C=t, T17_C=T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
            dT_min_evap2=dT_min_evap2, dT_min_abs2=dT_min_abs2,
        )
        for t in T_waste_values_C
    ]


def print_results_table(results: Sequence[DuehringScreeningResultDoubleLift]) -> None:
    print("=" * 140)
    print(
        f"{'T_waste[C]':>10} {'T17[C]':>7} {'x_strong':>9} "
        f"{'T_AL_max[C]':>12} {'GTL1[K]':>8} "
        f"{'T_AH_max[C]':>12} {'GTL2[K]':>8} {'GTL_tot[K]':>11} "
        f"{'Kristall.':>10}  Hinweis"
    )
    print("-" * 140)
    for r in results:
        if not r.crystallization_safe:
            krist = "RISIKO"
        elif r.crystallization_limited:
            krist = "geklemmt"
        else:
            krist = "sicher"
        print(
            f"{r.T13_C:10.2f} {r.T17_C:7.2f} {r.x_strong:9.4f} "
            f"{r.T_AL_max_C:12.2f} {r.GTL1_max_K:8.2f} "
            f"{r.T_AH_max_C:12.2f} {r.GTL2_max_K:8.2f} {r.GTL_total_max_K:11.2f} "
            f"{krist:>10}  {r.message}"
        )
    print("=" * 140)


def plot_gtl_vs_waste_heat_double_lift(
    T_waste_values_C: Sequence[float],
    T17_values_C: Sequence[float],
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    dT_min_evap2: float = 5.0,
    dT_min_abs2: float = 5.0,
    save_path: Optional[str] = "Design_Point/Plots/duehring_screening_GTL_double_lift_3K.png",
    show: bool = True,
):
    """Eine Kurve GTL_total_max vs. Abwärmetemperatur je T17-Wert. Infeasible/
    Kristallisationsrisiko-Punkte werden als leere Marker dargestellt --
    Aufbau identisch zu plot_gtl_vs_waste_heat() im single-lift Skript."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    all_results = {}
    for T17_C in T17_values_C:
        results = sweep_waste_heat_temperature_double_lift(
            T_waste_values_C, T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
            dT_min_evap2=dT_min_evap2, dT_min_abs2=dT_min_abs2,
        )
        all_results[T17_C] = results

        x = np.array([r.T13_C for r in results])
        y = np.array([r.GTL_total_max_K for r in results])
        ok = np.array([r.feasible for r in results])

        (line,) = ax.plot(x[ok], y[ok], "o-", label=f"T17 = {T17_C:.0f} °C")
        if np.any(~ok):
            ax.plot(
                x[~ok], y[~ok], "x", color=line.get_color(),
                markersize=8, markeredgewidth=2,
            )

    ax.set_xlabel("Abwärmetemperatur T13 = T15 [°C]")
    ax.set_ylabel(
        "Maximaler Gesamt-GTL = T_AH,max - T15 [K]\n"
        "(optimistische obere Schranke, parallel-feed Näherung)"
    )
    ax.set_title("Double-Lift Dühring-Screening: theoretisch maximaler Gesamt-GTL vs. Abwärmetemperatur")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.grid(alpha=0.4)
    ax.legend(title="× = Kristallisationsrisiko\noder GTL ≤ 0 in einer Stufe")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")
    if show:
        plt.show()

    return fig, ax, all_results


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # HIER ANPASSEN: Pinch-Annahmen für die optimistische Abschätzung
    # ------------------------------------------------------------------
    DT_MIN_DES = 5.0
    DT_MIN_EVAP = 5.0
    DT_MIN_COND = 5.0
    DT_MIN_ABS = 5.0
    DT_MIN_EVAP2 = 5.0
    DT_MIN_ABS2 = 5.0

    T_WASTE_RANGE_C = list(np.arange(40.0, 85.0, 5.0))
    T17_CURVES_C = [15.0, 20.0, 25.0]

    print("Double-Lift Dühring-Screening für T17 = 20 °C:")
    results = sweep_waste_heat_temperature_double_lift(
        T_WASTE_RANGE_C, T17_C=20.0,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
        dT_min_evap2=DT_MIN_EVAP2, dT_min_abs2=DT_MIN_ABS2,
    )
    print_results_table(results)

    plot_gtl_vs_waste_heat_double_lift(
        T_WASTE_RANGE_C, T17_CURVES_C,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
        dT_min_evap2=DT_MIN_EVAP2, dT_min_abs2=DT_MIN_ABS2,
    )