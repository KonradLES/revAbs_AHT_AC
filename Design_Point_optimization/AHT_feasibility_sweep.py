"""Schnelle Pinch-Feasibility-Karte für den AWT -- NUR Simulation, KEINE Optimierung.

Unterschied zu AHT_design_point_optimizer.py
---------------------------------------------
Der Bilevel-Optimierer dort minimiert Sum(UA) über 5 dT_min-Werte je
Betriebspunkt (DE + Nelder-Mead) -- mächtig, aber teuer (Minuten pro Punkt).
Für die Frage "wie weit kann ich die externen Temperaturen überhaupt
verschieben, damit die Anlage noch läuft" ist das Overkill.

Dieses Skript hält dT_min FEST auf eure real angenommenen/gebauten
Pinch-Werte (kein Optimierungsziel!) und sucht für ein Raster von
Abwärmetemperaturen (T13 = T15, parallele Verschaltung) das GESAMTE
feasible T12-Fenster [T12_min, T12_max] -- nicht nur das Maximum --, für
das solve_awt() eine gültige (feasible) Lösung liefert. Jeder Punkt kostet
nur eine Handvoll solve_awt()-Aufrufe (Millisekunden bis Sekunden) statt
einer vollen DE-Suche.

Warum ein FENSTER und nicht nur ein Maximum
--------------------------------------------
Naive Annahme (erste Version dieses Skripts): T12 nahe T_11_C ist immer der
"leichteste" Fall und damit ein sicherer Startpunkt für die Suche nach dem
Maximum. Das stimmt NICHT bei hohen Abwärmetemperaturen: mit steigendem
T_waste steigt der Hochdruck (Verdampfer/Absorber-Seite) so weit, dass
selbst die verdünnteste Lösung im Absorber nicht mehr bei niedrigen
Temperaturen (nahe T_11_C) kondensieren/absorbieren kann -- die gesamte
feasible Zone wandert nach oben. Ein fixer Startpunkt bei T_11_C+1°C ist
dort selbst schon unlösbar, und eine reine "Maximum"-Suche würde fälschlich
"gar nichts gefunden" melden, obwohl ein Fenster weiter oben existiert.

Vorgehen
--------
- Externe Austrittstemperaturen von Desorber/Verdampfer/Kondensator werden
  über feste ANNÄHERUNGS-Deltas relativ zur (geschwenkten) Abwärme- bzw.
  Rückkühltemperatur vorgegeben (T14 = T_waste - dT_approach_des usw.) --
  das ist eine Design-Annahme (typische externe Spreizung), keine
  Modellgrenze. Passt sie an eure tatsächliche Netz-/Kreislaufauslegung an.
- Für jede Abwärmetemperatur wird zunächst per Anker-Suche (ausgehend vom
  vorherigen Ergebnis, alternierend nach oben/unten) EIN feasibler T12-Wert
  lokalisiert. Von dort wird sowohl nach oben (Maximum) als auch nach unten
  (Minimum) per Expansion + Bisektion das Fenster bestimmt.
- "Feasible" bedeutet hier (identisch zur strengen Prüfung in
  AHT_stable_design_point.py): solve_info.success, final_point_evaluable,
  scaled_residual_norm <= RESIDUAL_TOL, und alle result.checks == True.

Ergebnis: für jede Abwärmetemperatur das GESAMTE erreichbare GTL-Fenster bei
EURER gewählten (nicht optimierten) Pinch-Güte -- direkt vergleichbar mit
der optimistischen oberen Schranke aus AHT_duehring_screening.py (die nur
die obere Kante liefert, siehe deren Docstring).

Erst für die 3-5 daraus ausgewählten, tatsächlich interessanten
Betriebspunkte lohnt sich der volle UA-Optimierer.

Aufruf als Skript
-----------------
    python Design_Point_optimization/AHT_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from Models.AHT_Pinch_Point import (
    AWTInputs,
    AWTResult,
    PRIMARY_VARIABLE_NAMES,
    initial_guess,
    solve_awt,
)

RESIDUAL_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------

@dataclass
class FeasibilitySweepConfig:
    # Feste Randbedingungen
    T_11_C: float = 70.0     # Nutzwärmesenke, kalter Eintritt
    T_17_C: float = 15.0     # Rückkühlung, kalter Eintritt
    Qabs_spec_kW: float = 500.0

    # Eure real angenommenen/gebauten Pinch-Werte -- NICHT optimiert.
    dT_min_shex: float = 3.0    # 5.0
    dT_min_des: float = 3.0     # 5.0   
    dT_min_cond: float = 3.0    # 5.0
    dT_min_evap: float = 3.0    # 5.0
    dT_min_abs: float = 3.0     # 5.0

    # Externe Spreizung (Design-Annahme, siehe Moduldocstring)
    dT_approach_des_C: float = 4.0    # 7.0 T14 = T_waste - dT_approach_des_C
    dT_approach_evap_C: float = 4.0   # 7.0 T16 = T_waste - dT_approach_evap_C
    dT_approach_cond_C: float = 3.0   # 6.0 T18 = T_17_C + dT_approach_cond_C

    desorber_evaporator_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # Bisektionssteuerung für T12_spec. step_C klein gehalten (siehe
    # anchor_search_step_C oben) -- dasselbe Basin-Argument gilt auch für
    # die Fenster-Expansion in _expand_and_bisect.
    T12_search_margin_C: float = 1.0   # Startabstand oberhalb T_11_C (nur 1. Punkt)
    T12_step_C: float = 2.0            # Expansionsschritt
    T12_bisect_tol_C: float = 0.2      # Abbruchbreite der Bisektion
    max_expand_steps: int = 40   # bei step_C=2.0 -> bis zu 80 K Reichweite
    max_bisect_steps: int = 25

    # Anker-Suche (siehe _locate_anchor): wie weit als Kontinuitäts-Walk um
    # den Schätzwert herum nach einem ERSTEN feasiblen T12 gesucht wird.
    # step_C bewusst klein: empirisch ist das Einzugsgebiet eines Warmstarts
    # oft nur ~2-4 K breit (siehe Chat-Diagnose) -- ein gröberes Raster
    # überspringt echte Lösungen, auch mit Kontinuitäts-Chaining.
    anchor_search_span_C: float = 60.0
    anchor_search_step_C: float = 1.0

    # Gelockerte Solver-Toleranzen für die Probe-Solves (Anker-Suche,
    # Expansion, Bisektion). Mit den strengen AWTInputs-Defaults
    # (solver_tol=1e-9, max_nfev=5000) kann JEDER einzelne fehlschlagende
    # Versuch bis zu 5000 Iterationen brauchen, bevor er als infeasible
    # erkannt wird -- bei ~100 Versuchen/Punkt summiert sich das schnell zu
    # Stunden. Analog zum fast=True/False-Muster in
    # AHT_design_point_optimizer.py: hier schnell/locker suchen, danach
    # EINMAL je Fenstergrenze streng nachrechnen (siehe find_feasible_window).
    probe_solver_tol: float = 1.0e-6
    probe_max_nfev: int = 300


@dataclass(frozen=True)
class FeasibilityPoint:
    T_waste_C: float
    T12_min_C: float
    T12_max_C: float
    GTL_min_K: float
    GTL_max_K: float
    feasible: bool
    message: str
    result: Optional[AWTResult] = None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _build_inputs(
    T_waste_C: float, T12_spec_C: float, config: FeasibilitySweepConfig, *,
    fast: bool = True, T11_C: Optional[float] = None,
) -> AWTInputs:
    """T11_C überschreibt config.T_11_C für einen einzelnen Aufruf -- genutzt
    von probe_minimum_output_temperature(), wo T11 selbst gesucht wird statt
    fix zu sein (siehe dortigen Docstring)."""
    kwargs = dict(
        T_11_C=T11_C if T11_C is not None else config.T_11_C,
        T_13_C=T_waste_C,
        T_15_C=T_waste_C,
        T_17_C=config.T_17_C,
        dT_min_shex=config.dT_min_shex,
        dT_min_des=config.dT_min_des,
        dT_min_cond=config.dT_min_cond,
        dT_min_evap=config.dT_min_evap,
        dT_min_abs=config.dT_min_abs,
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="Qabs",
        Qabs_spec_kW=config.Qabs_spec_kW,
        absorber_spec_mode="T12",
        T12_spec_C=T12_spec_C,
        desorber_spec_mode="T14",
        T14_spec_C=T_waste_C - config.dT_approach_des_C,
        evaporator_spec_mode="T16",
        T16_spec_C=T_waste_C - config.dT_approach_evap_C,
        condenser_spec_mode="T18",
        T18_spec_C=config.T_17_C + config.dT_approach_cond_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.probe_solver_tol
        kwargs["max_nfev"] = config.probe_max_nfev
    return AWTInputs(**kwargs)


def _is_valid_solution(result: AWTResult) -> bool:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False
    if not result.checks or not all(result.checks.values()):
        return False
    return True


def _x0_from_result(result: AWTResult) -> np.ndarray:
    return np.array(
        [result.primary_variables[name] for name in PRIMARY_VARIABLE_NAMES], dtype=float
    )


def _try_solve(
    T_waste_C: float, T12_spec_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True, T11_C: Optional[float] = None,
) -> Tuple[bool, Optional[AWTResult]]:
    try:
        inputs = _build_inputs(T_waste_C, T12_spec_C, config, fast=fast, T11_C=T11_C)
    except ValueError:
        return False, None
    try:
        result = solve_awt(inputs, x0=x0)
    except Exception:
        return False, None
    if not _is_valid_solution(result):
        return False, None
    return True, result


def _solve_raw(
    T_waste_C: float, T12_spec_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True, T11_C: Optional[float] = None,
) -> Optional[AWTResult]:
    """Wie _try_solve(), gibt aber IMMER das Result zurück (auch wenn nicht
    'valid' nach _is_valid_solution) -- nur None bei echtem Fehler
    (ValueError/Exception). Für Warmstart-Ketten: der Lösungsvektor eines
    nicht ganz konvergierten Solves ist meist trotzdem ein deutlich besserer
    Startpunkt für den NÄCHSTEN, benachbarten Versuch als ein genereller
    Heuristik-Guess -- siehe _locate_anchor()."""
    try:
        inputs = _build_inputs(T_waste_C, T12_spec_C, config, fast=fast, T11_C=T11_C)
    except ValueError:
        return None
    try:
        return solve_awt(inputs, x0=x0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fensterbestimmung: [T12_min, T12_max] für eine gegebene Abwärmetemperatur
# ---------------------------------------------------------------------------

def _locate_anchor(
    T_waste_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, guess_C: float,
) -> Tuple[Optional[float], Optional[AWTResult]]:
    """Sucht EINEN feasiblen T12-Wert, als Kontinuitäts-WALK in kleinen
    Schritten von guess_C aus (beide Richtungen) -- NICHT als unabhängige
    Sprünge mit demselben Startvektor.

    WICHTIG (empirisch bestätigt, siehe Chat): das Einzugsgebiet eines
    gegebenen Warmstarts ist oft nur ~2-4 K breit -- ein Kandidat 2 K daneben
    kann von genau demselben x0 aus glatt konvergieren, während einer 5-20 K
    weiter weg tagelang divergiert, OBWOHL dort ebenfalls eine gültige
    Lösung existiert. Ein Sprung-Scan mit fixem x0_seed trifft solche
    schmalen Becken nur zufällig. Der Walk reicht deshalb den Lösungsvektor
    JEDES Versuchs weiter (auch wenn er (noch) nicht "valid" ist, siehe
    _solve_raw) -- exakt das Kontinuitätsprinzip aus
    AHT_stable_design_point.py, nur über T12 statt über dT_min.
    """
    lo_bound = config.T_11_C + 1.0e-3
    step = config.anchor_search_step_C
    n_steps = max(1, int(round(config.anchor_search_span_C / step)))

    if guess_C > lo_bound:
        result = _solve_raw(T_waste_C, guess_C, x0_seed, config)
        if result is not None and _is_valid_solution(result):
            return guess_C, result

    for direction in (+1, -1):
        x0_walk = x0_seed
        for k in range(1, n_steps + 1):
            candidate = guess_C + direction * k * step
            if candidate <= lo_bound:
                break

            result = _solve_raw(T_waste_C, candidate, x0_walk, config)
            if result is None:
                try:
                    fresh_inputs = _build_inputs(T_waste_C, candidate, config)
                    x0_fresh = initial_guess(fresh_inputs)
                except ValueError:
                    continue
                result = _solve_raw(T_waste_C, candidate, x0_fresh, config)
                if result is None:
                    continue

            x0_walk = _x0_from_result(result)  # Kette weiterreichen, auch wenn nicht "valid"
            if _is_valid_solution(result):
                return candidate, result

    return None, None


def _bisect_boundary(
    feasible_T: float, x0_feasible: np.ndarray, infeasible_T: float,
    T_waste_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Bisektiert zwischen einem bekannt feasiblen und einem bekannt
    infeasiblen T12-Wert (Reihenfolge/Richtung beliebig) und gibt den
    zuletzt feasiblen Wert + zugehörigen Warmstart-Vektor zurück."""
    lo_feasible, x0_lo = feasible_T, x0_feasible
    hi_infeasible = infeasible_T
    for _ in range(config.max_bisect_steps):
        if abs(hi_infeasible - lo_feasible) < config.T12_bisect_tol_C:
            break
        mid = 0.5 * (lo_feasible + hi_infeasible)
        ok, result = _try_solve(T_waste_C, mid, x0_lo, config)
        if ok:
            lo_feasible = mid
            x0_lo = _x0_from_result(result)
        else:
            hi_infeasible = mid
    return lo_feasible, x0_lo


def _expand_and_bisect(
    anchor_T: float, x0_anchor: np.ndarray, direction: int,
    T_waste_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Expandiert von anchor_T aus in Richtung `direction` (+1 = Maximum
    suchen, -1 = Minimum suchen), bis infeasible, dann Bisektion auf die
    Grenze. Bricht am harten Rand T_11_C ab (T12 muss > T_11_C sein).
    Gibt (Grenzwert, zugehöriger Warmstart-Vektor) zurück."""
    lo_bound = config.T_11_C + 1.0e-3
    feasible_T = anchor_T
    x0_feasible = x0_anchor
    for _ in range(config.max_expand_steps):
        candidate = feasible_T + direction * config.T12_step_C
        if direction < 0 and candidate <= lo_bound:
            ok, result = _try_solve(T_waste_C, lo_bound, x0_feasible, config)
            return (lo_bound, _x0_from_result(result)) if ok else (feasible_T, x0_feasible)
        ok, result = _try_solve(T_waste_C, candidate, x0_feasible, config)
        if ok:
            feasible_T = candidate
            x0_feasible = _x0_from_result(result)
        else:
            return _bisect_boundary(feasible_T, x0_feasible, candidate, T_waste_C, config)
    return feasible_T, x0_feasible  # max_expand_steps erreicht, siehe Aufrufer-Warnung


def _refine_boundary(
    T_waste_C: float, T12_C: float, x0_seed: np.ndarray, config: FeasibilitySweepConfig,
) -> Optional[AWTResult]:
    """Ein abschliessender Solve mit strengen (AWTInputs-Default-)Toleranzen
    an einer per Fast-Probing gefundenen Fenstergrenze, für belastbare
    KPIs/UA-Werte im zurückgegebenen Result. Fällt bei Fehlschlag auf den
    gelockerten Solve zurück (Toleranzunterschied ist bei
    T12_bisect_tol_C=0.2 K i.d.R. irrelevant)."""
    ok, result = _try_solve(T_waste_C, T12_C, x0_seed, config, fast=False)
    if ok:
        return result
    ok, result = _try_solve(T_waste_C, T12_C, x0_seed, config, fast=True)
    return result if ok else None


def find_feasible_window(
    T_waste_C: float,
    config: FeasibilitySweepConfig,
    x0_seed: np.ndarray,
    T12_anchor_guess_C: Optional[float] = None,
) -> FeasibilityPoint:
    """Lokalisiert einen Anker und bestimmt davon ausgehend das gesamte
    feasible T12-Fenster [T12_min, T12_max]. Die eigentliche Suche läuft mit
    gelockerten Toleranzen (config.probe_*, siehe Moduldocstring); an den
    beiden gefundenen Grenzen wird danach je einmal streng nachgerechnet."""

    guess = (
        T12_anchor_guess_C if T12_anchor_guess_C is not None
        else config.T_11_C + config.T12_search_margin_C
    )

    anchor_T, anchor_result = _locate_anchor(T_waste_C, config, x0_seed, guess)
    if anchor_T is None:
        return FeasibilityPoint(
            T_waste_C=T_waste_C, T12_min_C=float("nan"), T12_max_C=float("nan"),
            GTL_min_K=float("nan"), GTL_max_K=float("nan"), feasible=False,
            message=(
                f"Keine feasible Lösung bei T_waste={T_waste_C:.2f} °C gefunden "
                f"(Anker-Suche um {guess:.2f} °C ± {config.anchor_search_span_C:.0f} K) "
                "-- dieser Betriebspunkt scheint ausserhalb des lösbaren Bereichs "
                "zu liegen (siehe AHT_duehring_screening.py zur Vorprüfung)."
            ),
        )

    x0_anchor = _x0_from_result(anchor_result)
    T12_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_waste_C, config)
    T12_min, _x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_waste_C, config)

    result_max = _refine_boundary(T_waste_C, T12_max, x0_max, config)

    return FeasibilityPoint(
        T_waste_C=T_waste_C, T12_min_C=T12_min, T12_max_C=T12_max,
        GTL_min_K=T12_min - T_waste_C, GTL_max_K=T12_max - T_waste_C,
        feasible=True, message="OK", result=result_max if result_max is not None else anchor_result,
    )


# ---------------------------------------------------------------------------
# Kleinster ECHTER Hub (GTL > 0), unabhängig von einer fixen T_11_C
# ---------------------------------------------------------------------------
#
# find_feasible_window() hält T_11_C aus der Config fest (eure reale
# Senkentemperatur). Das beantwortet "erreicht diese Abwärme MEINE
# geforderte Senkentemperatur" -- bei niedrigem T_waste kann das schlicht
# mit "nein" beantwortet sein, weil T_11_C zu hoch angesetzt ist, nicht weil
# die Anlage grundsätzlich keinen Hub liefern könnte.
#
# WICHTIGE KORREKTUR (eine erste, isolierte Version dieser Analyse war
# fehlerhaft): T_11_C beeinflusst NUR den externen Absorber-Massenstrom
# (m11 = Q_abs / (cp * (T12 - T11)), siehe _resolve_absorber_external_stream
# in Models.AHT_Pinch_Point) -- keine einzige interne Pinch-/Konzentrations-
# Nebenbedingung hängt von T11 ab. Ein Versuch, T11 = T12 - lift_margin_C zu
# setzen, hat deshalb "Lösungen" mit T12 < T_waste zugelassen: mathematisch
# zulässig, aber physikalisch sinnlos -- ein Wärmetransformator, der nicht
# transformiert (kein Hub gegenüber der Antriebstemperatur). Die tatsächlich
# harte, physikalisch sinnvolle Randbedingung ist GTL > 0, also
# T12 > T_waste_C (T_waste = T13 = T15) -- NICHT T12 > T11.
#
# Ein zweiter, isolierter Anlauf (eigene Expansion+Bisektion ab T_waste,
# ohne Warmstart-Kette) fand für 45/50/55°C fälschlich "GAR KEIN Hub
# möglich" -- das war ein SOLVER-Konvergenzproblem (kalter Start weit vom
# tatsächlichen Betriebspunkt), keine echte physikalische Grenze: die
# Dühring-Obergrenze (AHT_duehring_screening.py) zeigt für diesen Bereich
# durchaus zweistellige GTL-Werte als thermodynamisch möglich. Genau dieses
# Muster - generische Startwerte scheitern bei "schwierigen" (hier: sehr
# niedriges GTL) Betriebspunkten - ist der Grund, warum
# AHT_stable_design_point.py überhaupt Kontinuität/Warmstart-Ketten nutzt.
#
# sweep_true_lift_window() macht es deshalb NICHT nochmal isoliert, sondern
# wiederverwendet die bereits Warmstart-verkettete find_feasible_window() /
# sweep_feasibility()-Maschinerie 1:1 -- nur mit T_11_C durch einen
# niedrigen, für die Machbarkeit irrelevanten Platzhalter ersetzt -- und
# filtert das gefundene Fenster anschliessend auf den physikalisch
# sinnvollen Teil (T12 > T_waste_C).

@dataclass(frozen=True)
class TrueLiftPoint:
    T_waste_C: float
    lift_feasible: bool          # gibt es überhaupt ein T12 > T_waste_C im Fenster?
    GTL_min_K: float = float("nan")
    GTL_max_K: float = float("nan")
    raw_point: Optional[FeasibilityPoint] = None
    message: str = ""


def _duehring_initial_guess_C(
    T_waste_C: float, config: FeasibilitySweepConfig, *, fraction: float = 0.6
) -> Optional[float]:
    """Liefert T_waste_C + fraction * GTL_max(Dühring-Screening) als groben,
    aber grössenordnungsmässig richtigen T12-Schätzwert für den allerersten
    Punkt einer Suche -- siehe sweep_feasibility()-Docstring, warum ein
    generischer Schätzwert (z.B. T_11_C+margin) bei niedrigem T_waste um
    Grössenordnungen daneben liegen kann. fraction<1, weil das reale Fenster
    mit Pinch/Approach unter der optimistischen Dühring-Obergrenze liegt,
    aber in derselben Grössenordnung."""
    try:
        from AHT_duehring_screening import estimate_max_gtl
    except ImportError:
        try:
            from Design_Point_optimization.AHT_duehring_screening import estimate_max_gtl
        except ImportError:
            return None

    duehring = estimate_max_gtl(
        T13_C=T_waste_C, T15_C=T_waste_C, T17_C=config.T_17_C,
        dT_min_des=config.dT_min_des, dT_min_evap=config.dT_min_evap,
        dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
    )
    if duehring.feasible and duehring.GTL_max_K > 0:
        return T_waste_C + fraction * duehring.GTL_max_K
    return None


def sweep_true_lift_window(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    T11_placeholder_C: Optional[float] = None,
) -> List[TrueLiftPoint]:
    """Wie sweep_feasibility(), aber mit T_11_C durch einen niedrigen
    Platzhalter ersetzt (siehe Moduldocstring) und auf GTL > 0 gefiltert.

    T_waste_values_C trotzdem mit einem eher hohen, "leichten" Wert beginnen
    und absteigen -- die Warmstart-Kette in sweep_feasibility() trägt danach
    Schritt für Schritt weiter; der Startschätzwert für den allerersten
    Punkt kommt aus _duehring_initial_guess_C().
    """
    from dataclasses import replace

    T11_fixed = T11_placeholder_C if T11_placeholder_C is not None else config.T_17_C + 3.0
    config_probe = replace(config, T_11_C=T11_fixed)

    initial_guess_C = _duehring_initial_guess_C(T_waste_values_C[0], config) if T_waste_values_C else None

    raw_points = sweep_feasibility(
        T_waste_values_C, config_probe, initial_anchor_guess_C=initial_guess_C
    )

    results: List[TrueLiftPoint] = []
    for p in raw_points:
        if not p.feasible:
            results.append(TrueLiftPoint(
                T_waste_C=p.T_waste_C, lift_feasible=False, message=p.message,
            ))
            continue

        meaningful_min = max(p.T12_min_C, p.T_waste_C)
        meaningful_max = p.T12_max_C
        if meaningful_max <= p.T_waste_C:
            results.append(TrueLiftPoint(
                T_waste_C=p.T_waste_C, lift_feasible=False, raw_point=p,
                message=(
                    f"Selbst das rechnerische Maximum T12={p.T12_max_C:.2f} °C liegt "
                    f"nicht über T_waste={p.T_waste_C:.2f} °C -- diese Abwärmetemperatur "
                    "lässt mit den aktuellen Pinch-/Approach-Annahmen KEINEN echten Hub zu."
                ),
            ))
        else:
            results.append(TrueLiftPoint(
                T_waste_C=p.T_waste_C, lift_feasible=True, raw_point=p,
                GTL_min_K=meaningful_min - p.T_waste_C,
                GTL_max_K=meaningful_max - p.T_waste_C,
                message="OK",
            ))

    return results


def print_true_lift_table(points: Sequence[TrueLiftPoint]) -> None:
    print("=" * 90)
    print(f"{'T_waste[C]':>10} {'GTL_min[K]':>11} {'GTL_max[K]':>11} {'Status':>8}")
    print("-" * 90)
    for p in points:
        status = "OK" if p.lift_feasible else "KEIN HUB"
        gtl_min = f"{p.GTL_min_K:11.2f}" if p.lift_feasible else " " * 11
        gtl_max = f"{p.GTL_max_K:11.2f}" if p.lift_feasible else " " * 11
        print(f"{p.T_waste_C:10.2f} {gtl_min} {gtl_max} {status:>8}")
        if not p.lift_feasible:
            print(f"             -> {p.message}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Sweep über Abwärmetemperatur
# ---------------------------------------------------------------------------

def sweep_feasibility(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    initial_anchor_guess_C: Optional[float] = None,
) -> List[FeasibilityPoint]:
    """Warmstart-verkettete Fenstersuche über ein T_waste-Raster.

    T_waste_values_C sollte monoton (auf- oder absteigend) sein, damit der
    Warmstart von Punkt zu Punkt trägt -- analog zu sweep_parameter() in
    AHT_design_point_optimizer.py. Der Anker-Schätzwert für Punkt i+1 ist
    das zuletzt gefundene T12_max von Punkt i.

    initial_anchor_guess_C: Startschätzwert NUR für den allerersten Punkt
    (danach übernimmt die Warmstart-Kette). Ohne Angabe wird
    config.T_11_C + T12_search_margin_C verwendet -- das ist bei niedrigem
    T_waste_C oft SEHR weit von der tatsächlichen Lösung entfernt (siehe
    Chat-Diagnose: ein Kontinuitäts-Walk kann eine ~60 K-Lücke durch einen
    lösungsfreien Bereich nicht überbrücken, selbst mit kleinen Schritten).
    Für einen guten Schätzwert eignet sich z.B. die optimistische
    Dühring-Obergrenze aus AHT_duehring_screening.estimate_max_gtl() --
    siehe sweep_true_lift_window(), die genau das tut.
    """
    points: List[FeasibilityPoint] = []
    x0_carry: Optional[np.ndarray] = None
    anchor_guess_carry: Optional[float] = initial_anchor_guess_C

    for i, T_waste_C in enumerate(T_waste_values_C):
        if x0_carry is None:
            seed_guess_C = (
                anchor_guess_carry if anchor_guess_carry is not None
                else config.T_11_C + config.T12_search_margin_C
            )
            seed_inputs = _build_inputs(T_waste_C, seed_guess_C, config)
            x0_carry = initial_guess(seed_inputs)

        point = find_feasible_window(
            T_waste_C, config, x0_carry, T12_anchor_guess_C=anchor_guess_carry
        )
        points.append(point)

        status = "OK" if point.feasible else "FEHLGESCHLAGEN"
        print(
            f"[{i+1}/{len(T_waste_values_C)}] T_waste={T_waste_C:6.2f} °C -> "
            f"T12 in [{point.T12_min_C:6.2f}, {point.T12_max_C:6.2f}] °C, "
            f"GTL in [{point.GTL_min_K:6.2f}, {point.GTL_max_K:6.2f}] K "
            f"[{status}] {point.message if not point.feasible else ''}"
        )

        if point.feasible and point.result is not None:
            x0_carry = _x0_from_result(point.result)
            anchor_guess_carry = point.T12_max_C

    return points


# ---------------------------------------------------------------------------
# Fenster mit MITSKALIERENDER Mindest-Nutztemperatur (T11 = T_waste + Hub)
# ---------------------------------------------------------------------------
#
# sweep_feasibility() hält T_11_C für ALLE T_waste-Werte auf demselben
# absoluten Wert fest (z.B. immer 70°C) -- das erzwingt bei niedrigem
# T_waste einen unrealistisch grossen Hub und lässt das Fenster dort
# unnötig "unlösbar" wirken.
#
# sweep_relative_lift_window() setzt T_11_C stattdessen PRO PUNKT als
# T_waste_C + min_lift_offset_C: eine Design-VORGABE, die mit der
# Abwärmetemperatur mitskaliert ("ich will mindestens X Kelvin Hub über die
# jeweilige Abwärme"), statt ein fixes Absolutziel. Das gefundene Fenster
# [T12_min, T12_max] ist dann direkt als "welche Nutztemperatur kann ich mir
# bei dieser Abwärmetemperatur sinnvoll aussuchen" lesbar -- min_lift_offset_C
# ist dabei nur ein technischer Ankerpunkt für die Suche, keine reale
# Anforderung; die eigentliche Entscheidung (welches T11 ihr real wählt)
# trefft ihr anhand des ganzen abgelesenen Fensters.

def sweep_relative_lift_window(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    min_lift_offset_C: float = 5.0,
) -> List[FeasibilityPoint]:
    """Wie sweep_feasibility(), aber T_11_C wird für jeden Punkt individuell
    als T_waste_C + min_lift_offset_C gesetzt statt global fix (siehe
    Abschnitts-Docstring). Nutzt denselben Dühring-informierten
    Startschätzwert für den allerersten Punkt wie sweep_true_lift_window().
    """
    from dataclasses import replace

    points: List[FeasibilityPoint] = []
    x0_carry: Optional[np.ndarray] = None
    anchor_guess_carry: Optional[float] = None

    for i, T_waste_C in enumerate(T_waste_values_C):
        T11_this = T_waste_C + min_lift_offset_C
        config_point = replace(config, T_11_C=T11_this)

        if x0_carry is None:
            seed_guess_C = (
                anchor_guess_carry if anchor_guess_carry is not None
                else _duehring_initial_guess_C(T_waste_C, config)
                or T11_this + config.T12_search_margin_C
            )
            seed_inputs = _build_inputs(T_waste_C, seed_guess_C, config_point)
            x0_carry = initial_guess(seed_inputs)

        point = find_feasible_window(
            T_waste_C, config_point, x0_carry, T12_anchor_guess_C=anchor_guess_carry
        )
        points.append(point)

        status = "OK" if point.feasible else "FEHLGESCHLAGEN"
        print(
            f"[{i+1}/{len(T_waste_values_C)}] T_waste={T_waste_C:6.2f} °C "
            f"(T11={T11_this:6.2f} °C) -> "
            f"T12 in [{point.T12_min_C:6.2f}, {point.T12_max_C:6.2f}] °C, "
            f"GTL in [{point.GTL_min_K:6.2f}, {point.GTL_max_K:6.2f}] K "
            f"[{status}] {point.message if not point.feasible else ''}"
        )

        if point.feasible and point.result is not None:
            x0_carry = _x0_from_result(point.result)
            anchor_guess_carry = point.T12_max_C

    return points


# ---------------------------------------------------------------------------
# Adaptive Homotopie über T_waste (nicht nur über T12)
# ---------------------------------------------------------------------------
#
# sweep_relative_lift_window() macht pro T_waste-Punkt EINEN Sprung (mit
# Kontinuitäts-Walk nur in T12, T_waste bleibt bei diesem Walk fest). Das
# reicht, solange der SPRUNG zwischen zwei T_waste-Werten selbst klein genug
# ist -- ist er es nicht (siehe Chat: schon 10 K T_waste-Sprung kann die
# Kette reissen lassen, obwohl ausreichend thermodynamischer Spielraum laut
# Dühring-Screening vorhanden ist), bleibt der generische initial_guess()-
# Kaltstart in einem falschen Zustand stecken (nachgewiesen: alle
# Pinch-Residuen bleiben deutlich von 0 entfernt, obwohl scipy "success"
# meldet -- ein Scheinkonvergenzpunkt, kein echter).
#
# Die Funktionen hier lösen das analog zu AHT_stable_design_point.py, nur
# mit T_waste (statt dT_min) als Homotopie-Parameter: (T_waste, T12) werden
# GEMEINSAM in kleinen, adaptiven Schritten bewegt (GTL = T12 - T_waste
# dabei näherungsweise konstant gehalten), mit automatischer
# Schrittweitenhalbierung bei Fehlschlag und Vergrösserung bei Erfolg.

def _window_at_point(
    T_waste_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, anchor_guess_C: float,
) -> Optional[Tuple[FeasibilityPoint, np.ndarray]]:
    """Wie find_feasible_window(), gibt aber zusätzlich den Warmstart-Vektor
    am T12_min zurück (für die Fortsetzung der Homotopie zum nächsten
    T_waste-Ziel). None bei Fehlschlag der Anker-Suche."""
    anchor_T, anchor_result = _locate_anchor(T_waste_C, config, x0_seed, anchor_guess_C)
    if anchor_T is None:
        return None

    x0_anchor = _x0_from_result(anchor_result)
    T12_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_waste_C, config)
    T12_min, x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_waste_C, config)
    result_max = _refine_boundary(T_waste_C, T12_max, x0_max, config)

    point = FeasibilityPoint(
        T_waste_C=T_waste_C, T12_min_C=T12_min, T12_max_C=T12_max,
        GTL_min_K=T12_min - T_waste_C, GTL_max_K=T12_max - T_waste_C,
        feasible=True, message="OK", result=result_max if result_max is not None else anchor_result,
    )
    return point, x0_min


def _homotopy_walk_T_waste(
    T_waste_from_C: float, T12_from_C: float, x0_from: np.ndarray, T_waste_to_C: float,
    config: FeasibilitySweepConfig, min_lift_offset_C: float,
    *, step_initial_C: float = 3.0, step_min_C: float = 0.25, max_steps: int = 200,
) -> Tuple[float, float, np.ndarray, bool]:
    """Bewegt (T_waste, T12) gemeinsam von (T_waste_from_C, T12_from_C) nach
    T_waste_to_C, GTL = T12 - T_waste dabei konstant gehalten (mindestens
    min_lift_offset_C). Schrittweite wird bei Fehlschlag halbiert (Abbruch
    unter step_min_C -> gibt den weitesten erreichten Punkt zurück, analog
    zur "praktischen Grenze" in AHT_stable_design_point.py), bei Erfolg
    wieder vergrössert (gedeckelt auf step_initial_C).

    Rückgabe: (T_waste_erreicht, T12_erreicht, x0_erreicht, ziel_voll_erreicht)
    """
    GTL_hold = max(T12_from_C - T_waste_from_C, min_lift_offset_C)
    direction = 1.0 if T_waste_to_C > T_waste_from_C else -1.0

    T_waste_cur, T12_cur, x0_cur = T_waste_from_C, T12_from_C, x0_from
    step = step_initial_C

    for _ in range(max_steps):
        remaining = direction * (T_waste_to_C - T_waste_cur)
        if remaining <= 1.0e-9:
            return T_waste_cur, T12_cur, x0_cur, True

        step_trial = min(step, remaining)
        T_waste_trial = T_waste_cur + direction * step_trial
        T12_trial = T_waste_trial + GTL_hold
        T11_trial = T_waste_trial + min_lift_offset_C
        config_trial = replace(config, T_11_C=T11_trial)

        result = _solve_raw(T_waste_trial, T12_trial, x0_cur, config_trial)
        if result is not None and _is_valid_solution(result):
            T_waste_cur, T12_cur = T_waste_trial, T12_trial
            x0_cur = _x0_from_result(result)
            step = min(step_trial * 1.5, step_initial_C)
        else:
            step = step_trial / 2.0
            if step < step_min_C:
                return T_waste_cur, T12_cur, x0_cur, False

    return T_waste_cur, T12_cur, x0_cur, False


def sweep_relative_lift_window_homotopy(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    min_lift_offset_C: float = 5.0,
    homotopy_step_initial_C: float = 3.0,
    homotopy_step_min_C: float = 0.25,
) -> List[FeasibilityPoint]:
    """Wie sweep_relative_lift_window(), aber mit adaptiver Homotopie
    ZWISCHEN den Rasterpunkten (siehe Abschnitts-Docstring) statt eines
    einzelnen Sprungs. T_waste_values_C in Wanderreihenfolge angeben (z.B.
    absteigend von einem bekannt robusten Startwert wie 65°C).
    """
    points: List[FeasibilityPoint] = []
    T_waste_cur: Optional[float] = None
    T12_cur: Optional[float] = None
    x0_cur: Optional[np.ndarray] = None

    for i, T_waste_target in enumerate(T_waste_values_C):
        if x0_cur is None:
            T11_first = T_waste_target + min_lift_offset_C
            config_first = replace(config, T_11_C=T11_first)
            guess_C = (
                _duehring_initial_guess_C(T_waste_target, config)
                or T11_first + config.T12_search_margin_C
            )
            seed_inputs = _build_inputs(T_waste_target, guess_C, config_first)
            x0_seed = initial_guess(seed_inputs)
            outcome = _window_at_point(T_waste_target, config_first, x0_seed, guess_C)
            reported_T_waste = T_waste_target
        else:
            reached_T_waste, reached_T12, reached_x0, fully_reached = _homotopy_walk_T_waste(
                T_waste_cur, T12_cur, x0_cur, T_waste_target, config, min_lift_offset_C,
                step_initial_C=homotopy_step_initial_C, step_min_C=homotopy_step_min_C,
            )
            if not fully_reached:
                print(
                    f"  [Homotopie] Ziel {T_waste_target:.2f} °C nicht vollständig erreicht, "
                    f"angehalten bei {reached_T_waste:.2f} °C "
                    f"(Schrittweite unter {homotopy_step_min_C:.2f} K gefallen)."
                )
            T11_target = reached_T_waste + min_lift_offset_C
            config_target = replace(config, T_11_C=T11_target)
            # Sicherheitsabstand: reached_T12 kann (durch Bisektionstoleranz/
            # Rundung) sehr nah an T11_target liegen -- ein Anker direkt auf
            # der Grenze verpasst schmale Fenster, siehe Chat-Diagnose bei
            # T_waste=49.38°C (echtes, aber nur ~1K breites Fenster wurde
            # vom 2K-Schritt übersprungen). Anker bewusst spürbar oberhalb
            # T11_target ansetzen.
            anchor_guess_C = max(reached_T12, T11_target + 1.0)
            outcome = _window_at_point(reached_T_waste, config_target, reached_x0, anchor_guess_C)
            reported_T_waste = reached_T_waste

        if outcome is None:
            points.append(FeasibilityPoint(
                T_waste_C=reported_T_waste, T12_min_C=float("nan"), T12_max_C=float("nan"),
                GTL_min_K=float("nan"), GTL_max_K=float("nan"), feasible=False,
                message=f"Auch die Anker-Suche bei {reported_T_waste:.2f} °C fand keine Lösung.",
            ))
            print(f"[{i+1}/{len(T_waste_values_C)}] T_waste={reported_T_waste:6.2f} °C -> FEHLGESCHLAGEN")
            continue

        point, x0_min = outcome
        points.append(point)
        print(
            f"[{i+1}/{len(T_waste_values_C)}] T_waste={reported_T_waste:6.2f} °C -> "
            f"T12 in [{point.T12_min_C:6.2f}, {point.T12_max_C:6.2f}] °C, "
            f"GTL in [{point.GTL_min_K:6.2f}, {point.GTL_max_K:6.2f}] K [OK]"
        )

        T_waste_cur = reported_T_waste
        T12_cur = point.T12_min_C
        x0_cur = x0_min

    return points


def print_sweep_table(points: Sequence[FeasibilityPoint]) -> None:
    print("=" * 100)
    print(
        f"{'T_waste[C]':>10} {'T12_min[C]':>11} {'T12_max[C]':>11} "
        f"{'GTL_min[K]':>11} {'GTL_max[K]':>11} {'Status':>8}"
    )
    print("-" * 100)
    for p in points:
        status = "OK" if p.feasible else "FAIL"
        print(
            f"{p.T_waste_C:10.2f} {p.T12_min_C:11.2f} {p.T12_max_C:11.2f} "
            f"{p.GTL_min_K:11.2f} {p.GTL_max_K:11.2f} {status:>8}"
        )
    print("=" * 100)


def plot_feasibility_sweep(
    points: Sequence[FeasibilityPoint],
    *,
    duehring_reference: Optional[Sequence] = None,
    save_path: Optional[str] = "Design_Point_optimization/feasibility_sweep_GTL.png",
    show: bool = True,
):
    """Plottet das feasible GTL-Fenster vs. Abwärmetemperatur; optional
    Vergleich mit der optimistischen Dühring-Obergrenze (Liste von
    DuehringScreeningResult aus
    AHT_duehring_screening.sweep_waste_heat_temperature())."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ok_points = [p for p in points if p.feasible]
    x = np.array([p.T_waste_C for p in ok_points])
    y_min = np.array([p.GTL_min_K for p in ok_points])
    y_max = np.array([p.GTL_max_K for p in ok_points])

    ax.fill_between(
        x, y_min, y_max, color="tab:blue", alpha=0.18,
        label="feasibles GTL-Fenster (Pinch-Modell, fixe dT_min)",
    )
    ax.plot(x, y_max, "o-", color="tab:blue", label="GTL_max")
    ax.plot(x, y_min, "o--", color="tab:blue", linewidth=1.2, label="GTL_min")

    fail_points = [p for p in points if not p.feasible]
    if fail_points:
        xf = np.array([p.T_waste_C for p in fail_points])
        ax.plot(
            xf, np.zeros_like(xf), "x", color="tab:red", markersize=8,
            markeredgewidth=2, label="nicht lösbar",
        )

    if duehring_reference is not None:
        xr = np.array([r.T13_C for r in duehring_reference])
        yr = np.array([r.GTL_max_K for r in duehring_reference])
        okr = np.array([r.feasible for r in duehring_reference])
        ax.plot(
            xr[okr], yr[okr], "--", color="0.4",
            label="Dühring-Obergrenze (optimistisch)",
        )

    ax.set_xlabel("Abwärmetemperatur T13 = T15 [°C]")
    ax.set_ylabel("GTL [K]")
    ax.set_title("Pinch-Feasibility-Sweep: erreichbares GTL-Fenster vs. Abwärmetemperatur")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    config = FeasibilitySweepConfig()

    # ------------------------------------------------------------------
    # Hauptauswertung: für jede Abwärmetemperatur T_waste in [45, 85]°C das
    # erreichbare Nutztemperatur-/GTL-Fenster, mit einer MITSKALIERENDEN
    # Mindest-Nutztemperatur T11 = T_waste + MIN_LIFT_OFFSET_C (Design-
    # Vorgabe "mindestens X Kelvin Hub über die jeweilige Abwärme"), statt
    # eines global fixen T_11_C (siehe sweep_relative_lift_window()-
    # Abschnitts-Docstring, warum das bei niedrigem T_waste sonst unnötig
    # als "unlösbar" erscheint). Rückkühltemperatur T_17_C bleibt konstant
    # (config.T_17_C) -- als nächsten Schritt könnt ihr die auch variieren.
    #
    # MODERATE Pinch-Werte (5K überall) statt der scharfen config-Defaults
    # (3K): mit sehr scharfen Pinch-Werten kann das feasible T12-Fenster so
    # schmal werden, dass es vom Suchraster übersprungen wird (siehe
    # Chat-Diagnose). Die scharfen 3K-Werte sind ein BAUKRITERIUM für die
    # spätere UA-Feinauslegung eines konkret ausgewählten Punktes, keine
    # Voraussetzung für diese Explorations-Karte.
    #
    # Externe Approach-Werte auf 4K/4K/3K reduziert (statt 7K/7K/6K): mit
    # den grösseren Approach-Werten schliesst sich das feasible Fenster
    # (bei MIN_LIFT_OFFSET_C=5K) bereits zwischen 55°C und 65°C -- ein
    # ECHTER Umkehrpunkt (Fensterbreite geht reproduzierbar gegen 0, nicht
    # nur ein übersprungenes schmales Fenster). Mit den kleineren
    # Approach-Werten (= mehr externer Massenstrom, weniger "aufgebrauchtes"
    # thermisches Budget für die 5 internen 5K-Pinches) verschiebt sich
    # dieser Umkehrpunkt runter auf ca. 49-50°C -- siehe Chat, dort auch,
    # wie man das mit noch kleineren Approach-Werten oder einem kleineren
    # MIN_LIFT_OFFSET_C weiter nach unten schieben kann.
    #
    # sweep_relative_lift_window_homotopy() statt sweep_relative_lift_window():
    # nutzt eine ADAPTIVE Homotopie zwischen den Rasterpunkten (Schrittweite
    # wird bei Fehlschlag automatisch halbiert, bei Erfolg vergrössert,
    # analog zu AHT_stable_design_point.py) -- notwendig, weil auch
    # T_waste-Sprünge von nur 10K die Warmstart-Kette reissen lassen können,
    # obwohl ausreichend Lösungsraum existiert (siehe Chat-Diagnose:
    # Scheinkonvergenz mit deutlich verletzten Pinch-Residuen trotz
    # scipy-"success").
    # ------------------------------------------------------------------
    from dataclasses import replace as _replace

    config_explore = _replace(
        config,
        dT_min_shex=5.0, dT_min_des=5.0, dT_min_cond=5.0, dT_min_evap=5.0, dT_min_abs=5.0,
        dT_approach_des_C=4.0, dT_approach_evap_C=4.0, dT_approach_cond_C=3.0,
    )

    MIN_LIFT_OFFSET_C = 5.0
    T_WASTE_RANGE_C = list(np.arange(85.0, 44.0, -5.0))  # absteigend: 85 -> 50 (49-50°C ist die reale Grenze)

    print(f"Erreichbares Nutztemperatur-Fenster, T11 = T_waste + {MIN_LIFT_OFFSET_C:.0f} K")
    print(f"(T_17_C = {config_explore.T_17_C:.1f} °C konstant)")
    points = sweep_relative_lift_window_homotopy(
        T_WASTE_RANGE_C, config_explore, min_lift_offset_C=MIN_LIFT_OFFSET_C
    )
    print_sweep_table(points)

    duehring_reference = None
    try:
        from AHT_duehring_screening import sweep_waste_heat_temperature

        duehring_reference = sweep_waste_heat_temperature(
            T_WASTE_RANGE_C, T17_C=config_explore.T_17_C,
            dT_min_des=config_explore.dT_min_des, dT_min_evap=config_explore.dT_min_evap,
            dT_min_cond=config_explore.dT_min_cond, dT_min_abs=config_explore.dT_min_abs,
        )
    except ImportError:
        pass

    plot_feasibility_sweep(points, duehring_reference=duehring_reference)
