"""Reines Dühring-/Gleichgewichts-Screening für den AWT -- OHNE Solver.

Beantwortet die Frage "welcher GTL ist bei welcher Abwärmetemperatur
*thermodynamisch überhaupt maximal drin*", bevor irgendein Pinch-Modell
oder Optimierer angeworfen wird. Nutzt ausschliesslich:

  - die LiBr/H2O-Gleichgewichtsbeziehung (Pátek-Korrelationen in
    Thermodynamic_Properties.libr_props),
  - die Kristallisationsgrenze nach Albers/Boryta,
  - die Wasser-Sättigungsfunktionen aus Models.AHT_Pinch_Point (identische
    CoolProp-Quelle wie im eigentlichen Solvermodell, daher konsistent).

Kein Massen-/Energiebilanz-Solve, keine Kreislaufskalierung, kein Zirkulations-
verhältnis. Das Ergebnis ist bewusst eine OPTIMISTISCHE Abschätzung (Pinch nur
am jeweils bindenden Ende, nicht über den vollen Gegenstrom-Temperaturverlauf):

Physikalisches Bild (AWT, wie in Models.AHT_Pinch_Point umgesetzt)
-------------------------------------------------------------------
- Desorber + Kondensator liegen auf der NIEDRIGEN Druckseite (p_low):
  Abwärme bei T13 treibt die Lösung im Desorber aus, der Dampf kondensiert
  bei T17 (Rückkühlung).
- Verdampfer + Absorber liegen auf der HOHEN Druckseite (p_high):
  Abwärme bei T15 verdampft das Kältemittel bei p_high (deshalb höheres
  Druckniveau als der Kondensator!), der Dampf wird im Absorber von der im
  Desorber aufkonzentrierten ("starken") Lösung absorbiert und liefert
  Nutzwärme bei T12 > T15 -- das ist die eigentliche "Transformer"-Anhebung.

Kernidee der Abschätzung
------------------------
1. p_low aus T17 (Kondensator-Pinch am kalten Ende: T8 = T17 + dT_min_cond).
2. p_high aus T15 (Verdampfer-Pinch am heissen Ende, optimistisch:
   T10 = T15 - dT_min_evap).
3. Die im Desorber maximal erreichbare Konzentration x_strong ergibt sich
   direkt aus der Dühring-Gleichgewichtsbedingung
       T_sat_solution(p_low, x_strong) = T13 - dT_min_des
   (die Lösung siedet bis zu der Konzentration auf, deren Siedepunkt bei
   p_low gerade der Antriebstemperatur minus Pinch entspricht).
4. Kristallisationscheck bei (T13 - dT_min_des, w_strong).
5. Die maximal lieferbare Nutzwärmetemperatur ergibt sich aus dem
   Siedepunkt DERSELBEN (starken) Konzentration bei p_high, abzüglich des
   Absorber-Pinch:
       T12_max = T_sat_solution(p_high, x_strong) - dT_min_abs
       GTL_max = T12_max - T15

Diese Kette braucht kein Zirkulationsverhältnis (m_stark/m_schwach), weil
für die maximal mögliche Temperatur im Absorber die am *heissesten*
eintretende (= stärkste) Lösung massgeblich ist -- exakt der Zustand, den
Schritt 3 liefert. Das deckt sich mit der klassischen graphischen
Dühring-Auslegung (zwei Isothermen + zwei Isosteren), wie sie auch in
Postprocessing/AHT_Duehring_Plot.py als Sechseck dargestellt wird.

Was hier NICHT abgebildet wird (bewusst, für Geschwindigkeit):
  - Massenstromaufteilung / Zirkulationsverhältnis (FR)
  - SHEX-Wärmerückgewinnung, Vorabsorption
  - Der tatsächliche Temperaturverlauf über die Wärmeübertrager (nur der
    jeweils bindende Pinch-Punkt wird betrachtet, nicht LMTD/Gegenstrom)
  - UA-Werte / Baugrösse

-> Ergebnis ist eine OBERE Schranke. Das volle Pinch-Modell
   (AHT_feasibility_sweep.py, AHT_design_point_optimizer.py) liefert
   danach die tatsächlich erreichbare, engere Grenze.

Aufruf als Skript
-----------------
    python Design_Point/AHT_duehring_screening.py
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
# Kernfunktionen
# ---------------------------------------------------------------------------

def concentration_for_boiling_point(p_pa: float, T_target_K: float) -> float:
    """Invertiert T_sat_solution_from_p_x: liefert x, sodass die Lösung bei
    p_pa genau bei T_target_K siedet. Wirft ValueError, wenn T_target_K
    ausserhalb des bei diesem Druck erreichbaren Bereichs liegt."""

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
class DuehringScreeningResult:
    T13_C: float
    T15_C: float
    T17_C: float
    dT_min_des: float
    dT_min_evap: float
    dT_min_cond: float
    dT_min_abs: float

    feasible: bool
    message: str

    p_low_Pa: float = float("nan")
    p_high_Pa: float = float("nan")
    x_strong: float = float("nan")
    w_strong: float = float("nan")
    T_gen_C: float = float("nan")
    T10_C: float = float("nan")
    T12_max_C: float = float("nan")
    GTL_max_K: float = float("nan")
    crystallization_safe: bool = True
    crystallization_message: str = ""
    # True, wenn x_strong auf die Löslichkeitsgrenze geklemmt wurde (siehe
    # estimate_max_gtl()): T12_max_K/GTL_max_K sind dann NICHT mehr durch den
    # Desorber-Pinch (volle Antriebstemperatur ausgenutzt), sondern durch die
    # Kristallisationsgrenze limitiert -- weiterhin eine gültige, nur eben
    # löslichkeits- statt temperaturlimitierte obere Schranke.
    crystallization_limited: bool = False


def estimate_max_gtl(
    T13_C: float,
    T15_C: float,
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> DuehringScreeningResult:
    """Optimistische obere Schranke für den erreichbaren GTL, siehe Modul-Docstring."""

    common = dict(
        T13_C=T13_C, T15_C=T15_C, T17_C=T17_C,
        dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
        dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
    )

    # 1) Kondensatordruck aus T17 (Pinch am kalten Ende)
    T8_K = celsius_to_kelvin(T17_C) + dT_min_cond
    p_low = water_p_sat_from_T(T8_K, Q=0.0)

    # 2) Verdampferdruck aus T15 (Pinch am heissen Ende, optimistisch)
    T10_K = celsius_to_kelvin(T15_C) - dT_min_evap
    try:
        p_high = water_p_sat_from_T(T10_K, Q=1.0)
    except Exception as exc:
        return DuehringScreeningResult(
            **common, feasible=False,
            message=f"Verdampferdruck nicht berechenbar: {exc}",
            p_low_Pa=p_low,
        )

    if p_high <= p_low:
        return DuehringScreeningResult(
            **common, feasible=False,
            message=(
                f"p_high ({p_high:.0f} Pa) <= p_low ({p_low:.0f} Pa): "
                "Abwärmetemperatur T15 zu niedrig relativ zur Rückkühlung T17 "
                "-- dieses Druckverhältnis kann keinen AWT antreiben."
            ),
            p_low_Pa=p_low, p_high_Pa=p_high,
        )

    # 3) Desorber-Gleichgewicht: x_strong aus T13 und p_low
    T_gen_K = celsius_to_kelvin(T13_C) - dT_min_des
    try:
        x_strong = concentration_for_boiling_point(p_low, T_gen_K)
    except ValueError as exc:
        return DuehringScreeningResult(
            **common, feasible=False, message=str(exc),
            p_low_Pa=p_low, p_high_Pa=p_high,
            T_gen_C=kelvin_to_celsius(T_gen_K),
            T10_C=kelvin_to_celsius(T10_K),
        )

    w_strong = lp.w_libr_from_x(x_strong)

    # 4) Kristallisationscheck am Desorberaustritt
    validity = lp.validate_solution_state(
        T_gen_K, w_strong, label="Desorberaustritt (Dühring-Screening)"
    )

    # 4b) Auf die Löslichkeitsgrenze klemmen, statt den Punkt zu verwerfen.
    #
    # x_strong oben ist die Konzentration, deren Siedepunkt bei p_low GENAU
    # der vollen Antriebstemperatur (T13 - dT_min_des) entspricht. Ist diese
    # Konzentration nicht löslich, kann die reale Lösung dort NICHT hin
    # aufkonzentriert werden -- sie kristallisiert vorher aus und bleibt an
    # der Löslichkeitsgrenze stehen. Der Desorber ist dann nicht mehr
    # Pinch-, sondern löslichkeitslimitiert (es steht mehr Antriebstemperatur
    # zur Verfügung, als genutzt werden kann). GTL_max wird deshalb aus DER
    # geklemmten Konzentration neu bestimmt -- weiterhin eine gültige obere
    # Schranke, nur mit anderer bindender Nebenbedingung.
    crystallization_limited = False
    if validity.crystallization_checked and not validity.crystallization_safe:
        # Bisektion DIREKT auf der echten Sicherheitsprüfung (nicht nur auf
        # der T->w-Korrelation): die beiden Kristallisationskorrelationen
        # (T_cr(w) und w_cr(T)) sind unabhängige Fits, keine exakten
        # Inversen voneinander -- bei höheren Temperaturen weicht das um
        # mehr als 1 K auseinander. w=0.57 ist per Definition immer sicher
        # (siehe validate_solution_state), w_strong ist hier per Vorbedingung
        # unsicher -- klassische Bisektion konvergiert auf die tatsächliche
        # Grenze, unabhängig davon, welche der beiden Korrelationen bindet.
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
            label="Desorberaustritt (Dühring-Screening, an Löslichkeitsgrenze geklemmt)",
        )
        crystallization_limited = True

    # 5) Absorber: gleiche (starke, ggf. geklemmte) Konzentration bei p_high -> T12_max
    try:
        T_abs_solution_K = lp.T_sat_solution_from_p_x(p_high, x_strong)
    except Exception as exc:
        return DuehringScreeningResult(
            **common, feasible=False,
            message=f"Absorber-Gleichgewichtstemperatur nicht berechenbar: {exc}",
            p_low_Pa=p_low, p_high_Pa=p_high,
            x_strong=x_strong, w_strong=w_strong,
            T_gen_C=kelvin_to_celsius(T_gen_K), T10_C=kelvin_to_celsius(T10_K),
            crystallization_safe=validity.crystallization_safe,
            crystallization_message=validity.message,
            crystallization_limited=crystallization_limited,
        )

    T12_max_K = T_abs_solution_K - dT_min_abs
    T12_max_C = kelvin_to_celsius(T12_max_K)
    GTL_max_K = T12_max_C - T15_C

    feasible = validity.crystallization_safe and GTL_max_K > 0.0
    if not validity.crystallization_safe:
        message = f"Kristallisationsrisiko: {validity.message}"
    elif GTL_max_K <= 0.0:
        message = (
            f"T12_max ({T12_max_C:.2f} °C) liegt nicht über T15 ({T15_C:.2f} °C) "
            "-- kein positiver GTL erreichbar."
        )
    elif crystallization_limited:
        message = "OK (an Löslichkeitsgrenze geklemmt, nicht voller Desorber-Pinch ausgenutzt)."
    else:
        message = "OK (optimistische obere Schranke)."

    return DuehringScreeningResult(
        **common, feasible=feasible, message=message,
        p_low_Pa=p_low, p_high_Pa=p_high,
        x_strong=x_strong, w_strong=w_strong,
        T_gen_C=kelvin_to_celsius(T_gen_K), T10_C=kelvin_to_celsius(T10_K),
        T12_max_C=T12_max_C, GTL_max_K=GTL_max_K,
        crystallization_safe=validity.crystallization_safe,
        crystallization_message=validity.message,
        crystallization_limited=crystallization_limited,
    )


# ---------------------------------------------------------------------------
# Sweep über Abwärmetemperatur (T13 = T15, "parallel"-Fall)
# ---------------------------------------------------------------------------

def sweep_waste_heat_temperature(
    T_waste_values_C: Sequence[float],
    T17_C: float,
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
) -> List[DuehringScreeningResult]:
    """Setzt T13 = T15 = T_waste (parallele Verschaltung) und wertet
    estimate_max_gtl() für jeden Wert aus T_waste_values_C aus."""
    return [
        estimate_max_gtl(
            T13_C=t, T15_C=t, T17_C=T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
        )
        for t in T_waste_values_C
    ]


def print_results_table(results: Sequence[DuehringScreeningResult]) -> None:
    print("=" * 100)
    print(
        f"{'T_waste[C]':>10} {'T17[C]':>7} {'x_strong':>9} {'w_strong':>9} "
        f"{'T12_max[C]':>11} {'GTL_max[K]':>11} {'Kristall.':>10}  Hinweis"
    )
    print("-" * 100)
    for r in results:
        if not r.crystallization_safe:
            krist = "RISIKO"
        elif r.crystallization_limited:
            krist = "geklemmt"
        else:
            krist = "sicher"
        print(
            f"{r.T13_C:10.2f} {r.T17_C:7.2f} {r.x_strong:9.4f} {r.w_strong:9.4f} "
            f"{r.T12_max_C:11.2f} {r.GTL_max_K:11.2f} {krist:>10}  {r.message}"
        )
    print("=" * 100)


def plot_gtl_vs_waste_heat(
    T_waste_values_C: Sequence[float],
    T17_values_C: Sequence[float],
    *,
    dT_min_des: float = 5.0,
    dT_min_evap: float = 5.0,
    dT_min_cond: float = 5.0,
    dT_min_abs: float = 5.0,
    save_path: Optional[str] = "Design_Point/Plots/duehring_screening_GTL_3K.png",
    show: bool = True,
):
    """Eine Kurve GTL_max vs. Abwärmetemperatur je T17-Wert. Infeasible/
    Kristallisationsrisiko-Punkte werden als leere Marker dargestellt."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    all_results = {}
    for T17_C in T17_values_C:
        results = sweep_waste_heat_temperature(
            T_waste_values_C, T17_C,
            dT_min_des=dT_min_des, dT_min_evap=dT_min_evap,
            dT_min_cond=dT_min_cond, dT_min_abs=dT_min_abs,
        )
        all_results[T17_C] = results

        x = np.array([r.T13_C for r in results])
        y = np.array([r.GTL_max_K for r in results])
        ok = np.array([r.feasible for r in results])

        (line,) = ax.plot(x[ok], y[ok], "o-", label=f"T17 = {T17_C:.0f} °C")
        if np.any(~ok):
            ax.plot(
                x[~ok], y[~ok], "x", color=line.get_color(),
                markersize=8, markeredgewidth=2,
            )

    ax.set_xlabel("Abwärmetemperatur T13 = T15 [°C]")
    ax.set_ylabel("Maximaler GTL = T12,max - T15 [K]\n(optimistische obere Schranke)")
    ax.set_title("Dühring-Screening: theoretisch maximaler GTL vs. Abwärmetemperatur")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.grid(alpha=0.4)
    ax.legend(title="× = Kristallisationsrisiko\noder GTL ≤ 0")

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
    DT_MIN_DES = 3.0
    DT_MIN_EVAP = 3.0
    DT_MIN_COND = 3.0
    DT_MIN_ABS = 3.0

    T_WASTE_RANGE_C = list(np.arange(40.0, 100.0, 5.0))
    T17_CURVES_C = [15.0, 20.0, 25.0]

    print("Dühring-Screening für T17 = 20 °C:")
    results = sweep_waste_heat_temperature(
        T_WASTE_RANGE_C, T17_C=20.0,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )
    print_results_table(results)

    plot_gtl_vs_waste_heat(
        T_WASTE_RANGE_C, T17_CURVES_C,
        dT_min_des=DT_MIN_DES, dT_min_evap=DT_MIN_EVAP,
        dT_min_cond=DT_MIN_COND, dT_min_abs=DT_MIN_ABS,
    )
