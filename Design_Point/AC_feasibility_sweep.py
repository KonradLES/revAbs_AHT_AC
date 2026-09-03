"""Schnelle Pinch-Feasibility-Karte für die AKM -- NUR Simulation, KEINE Optimierung.

Analogon zu AHT_feasibility_sweep.py, aber für die Absorptionskältemaschine.

Kalibrierungsstand (WICHTIG vor dem ersten eigenen Lauf lesen)
------------------------------------------------------------------
Zwei reale Ursachen wurden gefunden und hier behoben (keine reine
Approach-Kalibrierung, wie in einer früheren Version dieses Kommentars
vermutet):

1) Models.AC_Pinch_Point.initial_guess() setzte den Kaltstart-Schätzwert
   ursprünglich auf x4=0.22, x1=0.243 -- also x4 < x1, bereits VERKEHRT herum
   relativ zur überall geforderten Konzentrationshierarchie x4 > x1
   (Desorber-Austritt muss konzentrierter sein als Absorber-Austritt). Von
   diesem Startpunkt aus blieb der Solver praktisch immer auf dem falschen
   Zweig hängen ("Konzentrationshierarchie verletzt" oder ein
   Scheinkonvergenzpunkt mit Residuum >> 0), egal wie klein die Suchschritte
   waren -- ein reiner Kontinuitäts-Walk kann eine falsche Wurzel nicht
   reparieren. INZWISCHEN IN Models/AC_Pinch_Point.py SELBST BEHOBEN (x4/x1
   dort vertauscht) -- dieses Skript ruft initial_guess() deshalb wieder
   direkt auf, ohne eigene Korrektur. Mit diesem einen Fix konvergieren die
   allermeisten Punkte bereits ohne jede weitere Klimmzüge auf Residuum
   ~1e-10.
2) Der Verdampfer hat eine harte Modellgrenze T10 >= 1 °C (kein Eis
   modelliert). Bei T18_spec_C=5.0 und dT_min_evap=5.0 wird die interne
   Verdampfungstemperatur rechnerisch auf ~0 °C gezwungen -- exakt auf/unter
   dieser Grenze, also strukturell unlösbar. AC_design_point_optimizer.py
   verwendet deshalb selbst schon einen kleineren Verdampfer-Pinch (Floor
   3.0 K) als die übrigen Wärmeübertrager; dT_min_evap sollte generell
   spürbar kleiner als (T18_spec_C - 1 °C) bleiben.
Ebenfalls wichtig: dT_approach_evap_C bestimmt T17 = T18_spec_C +
dT_approach_evap_C, und Models.AC_Pinch_Point.initial_guess() schätzt daraus
T10 ~= T17 - 8 K -- bei T17 < ca. 9 °C verletzt schon dieser Schätzwert die
obige 1°C-Grenze. dT_approach_evap_C deshalb bei niedrigem T18_spec_C nicht
zu klein wählen (>= 5.0 empfohlen).
Unterschied zu AC_design_point_optimizer.py
--------------------------------------------
Der Bilevel-Optimierer dort minimiert Sum(UA) über 5 dT_min-Werte je
Betriebspunkt (DE + Nelder-Mead) -- mächtig, aber teuer (Minuten pro Punkt).
Für die Frage "wie heiss muss mein Antriebswasser bei einer gegebenen
Rückkühltemperatur mindestens sein" ist das Overkill.

Dieses Skript hält dT_min FEST auf real angenommene/gebaute Pinch-Werte
(kein Optimierungsziel!) und sucht für ein Raster von Rückkühltemperaturen
(T13 = T15, parallele Verschaltung von Absorber und Kondensator) das GESAMTE
feasible T11-Fenster [T11_min, T11_max] -- nicht nur das Minimum. Jeder Punkt
kostet nur eine Handvoll solve_awt()-Aufrufe statt einer vollen DE-Suche.

Rollentausch gegenüber dem AWT (wichtig für das Verständnis)
--------------------------------------------------------------
Bei der AKM liegen die Druckniveaus GENAU UMGEKEHRT zum AWT: Desorber und
Kondensator auf der HOHEN Druckseite, Absorber und Verdampfer auf der
NIEDRIGEN. Das vertauscht auch, welche zwei Apparate ein gemeinsames
externes Temperaturniveau teilen ("Paar") und welche zwei unabhängig
spezifiziert werden ("Einzeln"):

                    AWT (Wärmetransformator)   AKM (Kältemaschine)
    Paar (gemeinsame externe Temperatur):
        Desorber + Verdampfer <-> T_waste       Absorber + Kondensator <-> T_rueck
    Einzeln (unabhängig vorgegeben):
        Absorber (Produkt)  : T11 -> T12         Desorber (Antrieb): T11 -> T12
        Kondensator (Abwurf): T17 -> T18          Verdampfer (Produkt): T17 -> T18

Deshalb wird hier -- als Analogon zu T_waste beim AWT -- die Rückkühl-
temperatur T_rueck (= T13 = T15) durchfahren, T18 (Verdampferaustritt,
"Nutzkälte") bleibt FEST, und gesucht wird das Fenster der Generator-
eintrittstemperatur T11 (Desorber-Antrieb) -- direktes Analogon zum
T12-Fenster beim AWT, nur dass hier T11 (statt T12) die frei variierte,
unabhängige Eingangsgrösse ist: T12 wird für jeden Kandidaten T11 über einen
festen Approach (dT_approach_des_C) mitgeführt, T11 selbst bleibt der
Suchparameter (siehe _build_inputs).

Wichtig zu wissen für die Interpretation
------------------------------------------
- Anders als beim AWT (wo T11 auf die interne Machbarkeit praktisch keinen
  Einfluss hat) bestimmt T11 hier UNMITTELBAR das Desorber-Gleichgewicht
  (höheres T11 -> höhere Konzentration x4 möglich) und damit auch, wie nah
  die Anlage an ihrer Kristallisationsgrenze operiert (state "6", der
  kälteste Punkt mit voller Konzentration x4, kurz vor dem Absorber). Ein zu
  NIEDRIGES T11 macht die Anlage pinch-/druckseitig infeasible (T11_min, die
  hier interessierende Grösse); ein zu HOHES T11 kann umgekehrt
  Kristallisation am SHEX-Austritt provozieren (T11_max). Das Fenster
  [T11_min, T11_max] kann sich daher -- anders als beim AWT, wo "kleiner
  Pinch überall" immer das grösstmögliche Fenster ergibt -- bei sehr tiefen
  Rückkühltemperaturen von OBEN her schliessen.
- Der Solver-Warmstart (x0) hat nur ein schmales Einzugsgebiet (oft nur
  ~2-4 K in T11, teils auch in T_rueck selbst). Ein zu grosser Sprung lässt
  den Solver in einem Scheinkonvergenzpunkt landen (scipy meldet "success",
  obwohl die Pinch-Residuen deutlich von 0 abweichen). Deshalb arbeiten alle
  Suchfunktionen hier mit kleinschrittigem Kontinuitäts-Walk bzw. adaptiver
  Homotopie (Schrittweite halbieren bei Fehlschlag, vergrössern bei Erfolg)
  -- exakt analog zu AHT_feasibility_sweep.py, nur über T11/T_rueck statt
  T12/T_waste.
- Manche Fenster sind sehr schmal (<2 K) kurz bevor ein Betriebspunkt an
  seine tatsächliche Machbarkeitsgrenze stösst. Ein zu grobes Suchraster
  kann solche Fenster überspringen und fälschlich "nicht lösbar" melden.

Empfehlung
----------
Erst mit diesem Skript den Grobverlauf der nötigen Generatoreintritts-
temperatur über T_rueck kartieren (moderate Pinch-/Approach-Werte, siehe
Konfiguration unten). Erst für die 3-5 daraus ausgewählten, tatsächlich
interessanten Betriebspunkte lohnt sich der volle UA-Optimierer
(AC_design_point_optimizer.py).

Aufruf als Skript
-----------------
    python Design_Point/AC_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from Models.AC_Pinch_Point import (
    AKMInputs,
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
    # --- Betriebspunkt-Randbedingungen -------------------------------------
    T18_spec_C: float = 5.0     # 5.0 feste Verdampferaustrittstemperatur (Nutzkälte)
    Qevap_spec_kW: float = 40.9

    dT_min_shex: float = 5.0
    dT_min_des: float = 5.0
    dT_min_cond: float = 5.0
    dT_min_evap: float = 3.0    # T18 - dT_min_evap >= 0 °C, sonst Kaltstartfehler
    dT_min_abs: float = 5.0

    # --- Externe Approach-Werte (Design-Annahme) ----------------------------
    dT_approach_des_C: float = 4.0 # 18.0
    dT_approach_abs_C: float = 3.0  # 7.0
    dT_approach_cond_C: float = 3.0 # 7.0
    dT_approach_evap_C: float = 4.0 # 6.0 Wenn zu klein, kann T10 < 0 °C werden (harte Modellgrenze Verdampfer)

    absorber_condenser_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # -------------------------------------------------------------------
    # Such-/Solver-Parameter 
    # -------------------------------------------------------------------
    T11_search_margin_C: float = 40.0   # Startabstand oberhalb T_rueck (nur 1. Punkt, falls kein Dühring-Schätzwert)
    T11_step_C: float = 2.0            # Expansionsschritt für die Fenstersuche
    T11_bisect_tol_C: float = 0.2      # Abbruchbreite der Bisektion
    max_expand_steps: int = 40         # bei T11_step_C=2.0 -> bis zu 80 K Reichweite
    max_bisect_steps: int = 25

    anchor_search_span_C: float = 60.0
    anchor_search_step_C: float = 1.0

    probe_solver_tol: float = 1.0e-6
    probe_max_nfev: int = 300

# ---------------------------------------------------------------------------
# Such-Raster für den Sweep
# ---------------------------------------------------------------------------
T_RUECK_START_C = 15.0   # numerisch unproblematischer Startwert [°C]
T_RUECK_END_C = 35.0     # höchste GEWÜNSCHTE Rückkühltemperatur [°C] 
T_RUECK_STEP_C = 2.5     # Rasterabstand [K]

plot_name = "AC_feasibility_sweep_ex_8_5_5_5"

ENABLE_DUEHRING_MULTI_PLOT = True
duehring_plot_name = "AC_duehring_multi_process_ex_8_5_5_5"

ENABLE_QT_MULTI_PDF = True
qt_pdf_name = "AC_qt_multi_process_ex_8_5_5_5"

MULTI_PLOT_EVERY_NTH = 2

@dataclass(frozen=True)
class FeasibilityPoint:
    T_reject_C: float
    T11_min_C: float
    T11_max_C: float
    dT_drive_min_K: float
    dT_drive_max_K: float
    feasible: bool
    message: str
    result: Optional[AWTResult] = None


# ---------------------------------------------------------------------------
# Solve-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _build_inputs(
    T_reject_C: float, T11_C: float, config: FeasibilitySweepConfig, *, fast: bool = True,
) -> AKMInputs:
    kwargs = dict(
        T_11_C=T11_C,
        T_13_C=T_reject_C,
        T_15_C=T_reject_C,
        T_17_C=config.T18_spec_C + config.dT_approach_evap_C,
        dT_min_shex=config.dT_min_shex,
        dT_min_des=config.dT_min_des,
        dT_min_cond=config.dT_min_cond,
        dT_min_evap=config.dT_min_evap,
        dT_min_abs=config.dT_min_abs,
        absorber_condenser_routing_mode=config.absorber_condenser_routing_mode,
        cycle_scale_spec_mode="Qeva",
        Qevap_spec_kW=config.Qevap_spec_kW,
        desorber_spec_mode="T12",
        T12_spec_C=T11_C - config.dT_approach_des_C,
        absorber_spec_mode="T14",
        T14_spec_C=T_reject_C + config.dT_approach_abs_C,
        condenser_spec_mode="T16",
        T16_spec_C=T_reject_C + config.dT_approach_cond_C,
        evaporator_spec_mode="T18",
        T18_spec_C=config.T18_spec_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.probe_solver_tol
        kwargs["max_nfev"] = config.probe_max_nfev
    return AKMInputs(**kwargs)


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
    T_reject_C: float, T11_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True,
) -> Tuple[bool, Optional[AWTResult]]:
    try:
        inputs = _build_inputs(T_reject_C, T11_C, config, fast=fast)
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
    T_reject_C: float, T11_C: float, x0: np.ndarray, config: FeasibilitySweepConfig,
    *, fast: bool = True,
) -> Optional[AWTResult]:
    """Wie _try_solve(), gibt aber IMMER das Result zurück (auch wenn nicht
    'valid' nach _is_valid_solution) -- nur None bei echtem Fehler
    (ValueError/Exception). Für Warmstart-Ketten: der Lösungsvektor eines
    nicht ganz konvergierten Solves ist meist trotzdem ein deutlich besserer
    Startpunkt für den NÄCHSTEN, benachbarten Versuch als ein genereller
    Heuristik-Guess -- siehe _locate_anchor()."""
    try:
        inputs = _build_inputs(T_reject_C, T11_C, config, fast=fast)
    except ValueError:
        return None
    try:
        return solve_awt(inputs, x0=x0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fensterbestimmung: [T11_min, T11_max] für eine gegebene Rückkühltemperatur
# ---------------------------------------------------------------------------

def _locate_anchor(
    T_reject_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, guess_C: float,
) -> Tuple[Optional[float], Optional[AWTResult]]:
    """Sucht EINEN feasiblen T11-Wert, als Kontinuitäts-WALK in kleinen
    Schritten von guess_C aus (beide Richtungen) -- NICHT als unabhängige
    Sprünge mit demselben Startvektor. Siehe AHT_feasibility_sweep.py für
    die ausführliche Begründung des Kontinuitätsprinzips."""
    lo_bound = T_reject_C + 1.0e-3  # T11 muss über der Rückkühltemperatur liegen (Druckgefälle)
    step = config.anchor_search_step_C
    n_steps = max(1, int(round(config.anchor_search_span_C / step)))

    if guess_C > lo_bound:
        result = _solve_raw(T_reject_C, guess_C, x0_seed, config)
        if result is not None and _is_valid_solution(result):
            return guess_C, result

    for direction in (+1, -1):
        x0_walk = x0_seed
        for k in range(1, n_steps + 1):
            candidate = guess_C + direction * k * step
            if candidate <= lo_bound:
                break

            result = _solve_raw(T_reject_C, candidate, x0_walk, config)
            if result is None or not _is_valid_solution(result):
                # Zusätzlich zum Kontinuitäts-Schritt IMMER auch einen
                # frischen Kaltstart an genau diesem Kandidaten probieren --
                # der verkettete Warmstart kann in Einzelfällen an einem
                # ungünstigen Punkt hängen bleiben, ein frischer Kaltstart
                # (initial_guess()) findet von dort oft leichter zurück auf
                # den richtigen Zweig.
                try:
                    fresh_inputs = _build_inputs(T_reject_C, candidate, config)
                    x0_fresh = initial_guess(fresh_inputs)
                except ValueError:
                    x0_fresh = None
                if x0_fresh is not None:
                    fresh_result = _solve_raw(T_reject_C, candidate, x0_fresh, config)
                    if fresh_result is not None and (
                        result is None or _is_valid_solution(fresh_result)
                    ):
                        result = fresh_result

            if result is None:
                continue

            x0_walk = _x0_from_result(result)  # Kette weiterreichen, auch wenn nicht "valid"
            if _is_valid_solution(result):
                return candidate, result

    return None, None


def _bisect_boundary(
    feasible_T: float, x0_feasible: np.ndarray, infeasible_T: float,
    T_reject_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Bisektiert zwischen einem bekannt feasiblen und einem bekannt
    infeasiblen T11-Wert (Reihenfolge/Richtung beliebig) und gibt den
    zuletzt feasiblen Wert + zugehörigen Warmstart-Vektor zurück."""
    lo_feasible, x0_lo = feasible_T, x0_feasible
    hi_infeasible = infeasible_T
    for _ in range(config.max_bisect_steps):
        if abs(hi_infeasible - lo_feasible) < config.T11_bisect_tol_C:
            break
        mid = 0.5 * (lo_feasible + hi_infeasible)
        ok, result = _try_solve(T_reject_C, mid, x0_lo, config)
        if ok:
            lo_feasible = mid
            x0_lo = _x0_from_result(result)
        else:
            hi_infeasible = mid
    return lo_feasible, x0_lo


def _expand_and_bisect(
    anchor_T: float, x0_anchor: np.ndarray, direction: int,
    T_reject_C: float, config: FeasibilitySweepConfig,
) -> Tuple[float, np.ndarray]:
    """Expandiert von anchor_T aus in Richtung `direction` (+1 = Maximum
    suchen, -1 = Minimum suchen), bis infeasible, dann Bisektion auf die
    Grenze. Bricht in Richtung -1 am harten Rand T_reject_C ab (T11 muss >
    T_reject_C sein). Gibt (Grenzwert, zugehöriger Warmstart-Vektor) zurück."""
    lo_bound = T_reject_C + 1.0e-3
    feasible_T = anchor_T
    x0_feasible = x0_anchor
    for _ in range(config.max_expand_steps):
        candidate = feasible_T + direction * config.T11_step_C
        if direction < 0 and candidate <= lo_bound:
            ok, result = _try_solve(T_reject_C, lo_bound, x0_feasible, config)
            return (lo_bound, _x0_from_result(result)) if ok else (feasible_T, x0_feasible)
        ok, result = _try_solve(T_reject_C, candidate, x0_feasible, config)
        if ok:
            feasible_T = candidate
            x0_feasible = _x0_from_result(result)
        else:
            return _bisect_boundary(feasible_T, x0_feasible, candidate, T_reject_C, config)
    return feasible_T, x0_feasible  # max_expand_steps erreicht, siehe Aufrufer-Warnung


def _refine_boundary(
    T_reject_C: float, T11_C: float, x0_seed: np.ndarray, config: FeasibilitySweepConfig,
) -> Optional[AWTResult]:
    """Ein abschliessender Solve mit strengen (AKMInputs-Default-)Toleranzen
    an einer per Fast-Probing gefundenen Fenstergrenze, für belastbare
    KPIs/UA-Werte im zurückgegebenen Result. Fällt bei Fehlschlag auf den
    gelockerten Solve zurück (Toleranzunterschied ist bei
    T11_bisect_tol_C=0.2 K i.d.R. irrelevant)."""
    ok, result = _try_solve(T_reject_C, T11_C, x0_seed, config, fast=False)
    if ok:
        return result
    ok, result = _try_solve(T_reject_C, T11_C, x0_seed, config, fast=True)
    return result if ok else None


def find_feasible_window(
    T_reject_C: float,
    config: FeasibilitySweepConfig,
    x0_seed: np.ndarray,
    T11_anchor_guess_C: Optional[float] = None,
) -> FeasibilityPoint:
    """Lokalisiert einen Anker und bestimmt davon ausgehend das gesamte
    feasible T11-Fenster [T11_min, T11_max]. Die eigentliche Suche läuft mit
    gelockerten Toleranzen (config.probe_*); an der gefundenen Minimum-
    Grenze (die hier interessierende "minimale Generatoreintrittstemperatur")
    wird danach streng nachgerechnet."""

    guess = (
        T11_anchor_guess_C if T11_anchor_guess_C is not None
        else T_reject_C + config.T11_search_margin_C
    )

    anchor_T, anchor_result = _locate_anchor(T_reject_C, config, x0_seed, guess)
    if anchor_T is None:
        return FeasibilityPoint(
            T_reject_C=T_reject_C, T11_min_C=float("nan"), T11_max_C=float("nan"),
            dT_drive_min_K=float("nan"), dT_drive_max_K=float("nan"), feasible=False,
            message=(
                f"Keine feasible Lösung bei T_rueck={T_reject_C:.2f} °C gefunden "
                f"(Anker-Suche um {guess:.2f} °C ± {config.anchor_search_span_C:.0f} K) "
                "-- dieser Betriebspunkt scheint ausserhalb des lösbaren Bereichs "
                "zu liegen (siehe AC_duehring_screening.py zur Vorprüfung)."
            ),
        )

    x0_anchor = _x0_from_result(anchor_result)
    T11_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_reject_C, config)
    T11_min, x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_reject_C, config)

    result_min = _refine_boundary(T_reject_C, T11_min, x0_min, config)

    return FeasibilityPoint(
        T_reject_C=T_reject_C, T11_min_C=T11_min, T11_max_C=T11_max,
        dT_drive_min_K=T11_min - T_reject_C, dT_drive_max_K=T11_max - T_reject_C,
        feasible=True, message="OK", result=result_min if result_min is not None else anchor_result,
    )


def _duehring_initial_guess_C(
    T_reject_C: float, config: FeasibilitySweepConfig, *, margin_C: float = 25.0
) -> Optional[float]:
    """Liefert T_gen_min(Dühring-Screening) + margin_C als groben, aber
    grössenordnungsmässig richtigen T11-Schätzwert für den allerersten Punkt
    einer Suche. Ein generischer Schätzwert (z.B. T_reject_C+margin) kann bei
    hoher Rückkühltemperatur um Grössenordnungen daneben liegen; margin_C>0,
    weil das reale (Pinch-)Minimum über der optimistischen Dühring-
    Untergrenze liegt, aber in derselben Grössenordnung. margin_C=25 ist kein
    Kalibrierungs-Zufallswert: am Betriebspunkt T_rueck=25°C/T18=5°C lag das
    reale Pinch-Modell-Minimum empirisch ca. 27 K über der optimistischen
    Dühring-Schätzung (63 °C Dühring vs. ~90 °C real)."""
    try:
        from AC_duehring_screening import estimate_min_generator_temperature
    except ImportError:
        try:
            from Design_Point.AC_duehring_screening import estimate_min_generator_temperature
        except ImportError:
            return None

    T_evap_target_C = config.T18_spec_C - config.dT_min_evap
    try:
        duehring = estimate_min_generator_temperature(
            T_evap_target_C=T_evap_target_C, T_rueck_C=T_reject_C,
            dT_min_des=config.dT_min_des, dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
        )
    except Exception:
        # Die Dühring-Korrelationen (Patek) können an Rand-/Extremwerten
        # (sehr niedriger Druck, Konzentration nahe 0) intern eine
        # ValueError/PropertyError werfen -- dann einfach ohne Schätzwert
        # weitermachen (Aufrufer fällt auf T_reject_C+Margin zurück) statt
        # den ganzen Sweep abzubrechen.
        return None
    if duehring.feasible:
        return duehring.T_gen_min_C + margin_C
    return None


# ---------------------------------------------------------------------------
# Sweep: adaptive Homotopie über T_rueck (empfohlene, robusteste Variante)
# ---------------------------------------------------------------------------
#
# (T_rueck, T11) werden GEMEINSAM in kleinen, adaptiven Schritten bewegt
# (dT_drive = T11 - T_rueck dabei näherungsweise konstant gehalten), mit
# automatischer Schrittweitenhalbierung bei Fehlschlag und Vergrösserung bei
# Erfolg -- notwendig, weil auch T_rueck-Sprünge von wenigen K die Warmstart-
# Kette reissen lassen können, obwohl ausreichend Lösungsraum existiert
# (Scheinkonvergenz: scipy meldet "success", aber die Pinch-Residuen weichen
# deutlich von 0 ab). Siehe AHT_feasibility_sweep.py für das Vorbild.

def _window_at_point(
    T_reject_C: float, config: FeasibilitySweepConfig, x0_seed: np.ndarray, anchor_guess_C: float,
) -> Optional[Tuple[FeasibilityPoint, np.ndarray]]:
    """Wie find_feasible_window(), gibt aber zusätzlich den Warmstart-Vektor
    am T11_min zurück (für die Fortsetzung der Homotopie zum nächsten
    T_rueck-Ziel). None bei Fehlschlag der Anker-Suche."""
    anchor_T, anchor_result = _locate_anchor(T_reject_C, config, x0_seed, anchor_guess_C)
    if anchor_T is None:
        return None

    x0_anchor = _x0_from_result(anchor_result)
    T11_max, x0_max = _expand_and_bisect(anchor_T, x0_anchor, +1, T_reject_C, config)
    T11_min, x0_min = _expand_and_bisect(anchor_T, x0_anchor, -1, T_reject_C, config)
    result_min = _refine_boundary(T_reject_C, T11_min, x0_min, config)

    point = FeasibilityPoint(
        T_reject_C=T_reject_C, T11_min_C=T11_min, T11_max_C=T11_max,
        dT_drive_min_K=T11_min - T_reject_C, dT_drive_max_K=T11_max - T_reject_C,
        feasible=True, message="OK", result=result_min if result_min is not None else anchor_result,
    )
    return point, x0_min


def _homotopy_walk_T_reject(
    T_reject_from_C: float, T11_from_C: float, x0_from: np.ndarray, T_reject_to_C: float,
    config: FeasibilitySweepConfig,
    *, step_initial_C: float = 3.0, step_min_C: float = 0.25, max_steps: int = 200,
) -> Tuple[float, float, np.ndarray, bool]:
    """Bewegt (T_rueck, T11) gemeinsam von (T_reject_from_C, T11_from_C) nach
    T_reject_to_C, dT_drive = T11 - T_rueck dabei konstant gehalten.
    Schrittweite wird bei Fehlschlag halbiert (Abbruch unter step_min_C ->
    gibt den weitesten erreichten Punkt zurück, ein echter Umkehrpunkt),
    bei Erfolg wieder vergrössert (gedeckelt auf step_initial_C).

    Rückgabe: (T_rueck_erreicht, T11_erreicht, x0_erreicht, ziel_voll_erreicht)
    """
    dT_drive_hold = T11_from_C - T_reject_from_C
    direction = 1.0 if T_reject_to_C > T_reject_from_C else -1.0

    T_reject_cur, T11_cur, x0_cur = T_reject_from_C, T11_from_C, x0_from
    step = step_initial_C

    for _ in range(max_steps):
        remaining = direction * (T_reject_to_C - T_reject_cur)
        if remaining <= 1.0e-9:
            return T_reject_cur, T11_cur, x0_cur, True

        step_trial = min(step, remaining)
        T_reject_trial = T_reject_cur + direction * step_trial
        T11_trial = T_reject_trial + dT_drive_hold

        result = _solve_raw(T_reject_trial, T11_trial, x0_cur, config)
        if result is not None and _is_valid_solution(result):
            T_reject_cur, T11_cur = T_reject_trial, T11_trial
            x0_cur = _x0_from_result(result)
            step = min(step_trial * 1.5, step_initial_C)
        else:
            step = step_trial / 2.0
            if step < step_min_C:
                return T_reject_cur, T11_cur, x0_cur, False

    return T_reject_cur, T11_cur, x0_cur, False


def sweep_min_generator_temperature_homotopy(
    T_reject_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    homotopy_step_initial_C: float = 3.0,
    homotopy_step_min_C: float = 0.25,
) -> List[FeasibilityPoint]:
    """Bestimmt für jede Rückkühltemperatur in T_reject_values_C das feasible
    T11-Fenster, mit adaptiver Homotopie ZWISCHEN den Rasterpunkten (siehe
    Abschnitts-Docstring) -- die empfohlene, robusteste Variante.
    T_reject_values_C in Wanderreihenfolge angeben (z.B. absteigend von
    einem warmen, unproblematischen Startwert, siehe Modul-Docstring)."""

    points: List[FeasibilityPoint] = []
    T_reject_cur: Optional[float] = None
    T11_cur: Optional[float] = None
    x0_cur: Optional[np.ndarray] = None

    for i, T_reject_target in enumerate(T_reject_values_C):
        outcome = None
        reported_T_reject = T_reject_target

        if x0_cur is not None:
            reached_T_reject, reached_T11, reached_x0, fully_reached = _homotopy_walk_T_reject(
                T_reject_cur, T11_cur, x0_cur, T_reject_target, config,
                step_initial_C=homotopy_step_initial_C, step_min_C=homotopy_step_min_C,
            )
            if fully_reached:
                # Sicherheitsabstand: reached_T11 kann (durch Bisektionstoleranz/
                # Rundung) sehr nah an der T_reject-Grenze liegen -- ein Anker
                # direkt auf der Grenze verpasst schmale Fenster. Anker bewusst
                # spürbar oberhalb ansetzen.
                anchor_guess_C = max(reached_T11, reached_T_reject + 1.0)
                outcome = _window_at_point(reached_T_reject, config, reached_x0, anchor_guess_C)
                reported_T_reject = reached_T_reject
            else:
                # Diagonaler Kontinuitäts-Walk (dT_drive konstant) kann schon
                # nach dem ersten Schritt hängen bleiben, wenn der Startpunkt
                # T11_min selbst ein bisektierter Fenster-RAND ist (numerisch
                # empfindlicher als ein Punkt im Fensterinneren) -- das ist
                # KEIN Zeichen für echte Infeasibility am Ziel, siehe
                # Modul-Docstring. Statt den hängengebliebenen Zwischenpunkt
                # erneut zu melden, versuchen wir direkt am Ziel einen
                # frischen, Dühring-geführten Anker (wie beim allerersten
                # Punkt) -- das ist die robustere Fallback-Strategie.
                print(
                    f"  [Homotopie] Ziel {T_reject_target:.2f} °C nicht per Kontinuitäts-Walk "
                    f"erreicht (angehalten bei {reached_T_reject:.2f} °C) -- versuche frischen "
                    "Anker direkt am Ziel."
                )

        if outcome is None:
            guess_C = (
                _duehring_initial_guess_C(T_reject_target, config)
                or T_reject_target + config.T11_search_margin_C
            )
            seed_inputs = _build_inputs(T_reject_target, guess_C, config)
            x0_seed = initial_guess(seed_inputs)
            outcome = _window_at_point(T_reject_target, config, x0_seed, guess_C)
            reported_T_reject = T_reject_target

        if outcome is None:
            points.append(FeasibilityPoint(
                T_reject_C=reported_T_reject, T11_min_C=float("nan"), T11_max_C=float("nan"),
                dT_drive_min_K=float("nan"), dT_drive_max_K=float("nan"), feasible=False,
                message=f"Auch die Anker-Suche bei {reported_T_reject:.2f} °C fand keine Lösung.",
            ))
            print(f"[{i+1}/{len(T_reject_values_C)}] T_rueck={reported_T_reject:6.2f} °C -> FEHLGESCHLAGEN")
            continue

        point, x0_min = outcome
        points.append(point)
        print(
            f"[{i+1}/{len(T_reject_values_C)}] T_rueck={reported_T_reject:6.2f} °C -> "
            f"T11 in [{point.T11_min_C:6.2f}, {point.T11_max_C:6.2f}] °C, "
            f"dT_drive in [{point.dT_drive_min_K:6.2f}, {point.dT_drive_max_K:6.2f}] K [OK]"
        )

        T_reject_cur = reported_T_reject
        T11_cur = point.T11_min_C
        x0_cur = x0_min

    return points


# ---------------------------------------------------------------------------
# Ausgabe: Tabelle + Plot
# ---------------------------------------------------------------------------

def print_sweep_table(points: Sequence[FeasibilityPoint]) -> None:
    print("=" * 100)
    print(
        f"{'T_rueck[C]':>10} {'T11_min[C]':>11} {'T11_max[C]':>11} "
        f"{'dT_drv_min[K]':>13} {'dT_drv_max[K]':>13} {'Status':>8}"
    )
    print("-" * 100)
    for p in points:
        status = "OK" if p.feasible else "FAIL"
        print(
            f"{p.T_reject_C:10.2f} {p.T11_min_C:11.2f} {p.T11_max_C:11.2f} "
            f"{p.dT_drive_min_K:13.2f} {p.dT_drive_max_K:13.2f} {status:>8}"
        )
    print("=" * 100)


def plot_feasibility_sweep(
    points: Sequence[FeasibilityPoint],
    *,
    duehring_reference: Optional[Sequence] = None,
    save_path: Optional[str] = f"Design_Point/Plots/{plot_name}.png",
    show: bool = True,
):
    """Plottet das feasible T11-Fenster vs. Rückkühltemperatur; optional
    Vergleich mit der optimistischen Dühring-Untergrenze (Liste von
    MinGenResult aus AC_duehring_screening.sweep_recool_temperature())."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ok_points = [p for p in points if p.feasible]
    x = np.array([p.T_reject_C for p in ok_points])
    y_min = np.array([p.T11_min_C for p in ok_points])
    y_max = np.array([p.T11_max_C for p in ok_points])

    ax.fill_between(
        x, y_min, y_max, color="tab:orange", alpha=0.18,
        label="feasibles T11-Fenster (Pinch-Modell, fixe dT_min)",
    )
    ax.plot(x, y_min, "o-", color="tab:orange", label="T11_min (minimale Generatoreintrittstemperatur)")
    ax.plot(x, y_max, "o--", color="tab:orange", linewidth=1.2, label="T11_max")

    fail_points = [p for p in points if not p.feasible]
    if fail_points:
        xf = np.array([p.T_reject_C for p in fail_points])
        ax.plot(
            xf, np.zeros_like(xf), "x", color="tab:red", markersize=8,
            markeredgewidth=2, label="nicht lösbar",
        )

    if duehring_reference is not None:
        xr = np.array([r.T_rueck_C for r in duehring_reference])
        yr = np.array([r.T_gen_min_C for r in duehring_reference])
        okr = np.array([r.feasible for r in duehring_reference])
        ax.plot(
            xr[okr], yr[okr], "--", color="0.4",
            label="Dühring-Untergrenze (optimistisch)",
        )

    ax.set_xlabel("Rückkühltemperatur T13 = T15 [°C]")
    ax.set_ylabel("Generatoreintrittstemperatur T11 [°C]")
    ax.set_title("AKM Pinch-Feasibility-Sweep: minimale Generatoreintrittstemperatur vs. Rückkühltemperatur")
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

    # Alle physikalischen Annahmen (Pinch-Werte, Approach-Werte, T18_spec_C)
    # stehen zentral in FeasibilitySweepConfig oben -- dort anpassen, nicht
    # hier.
    T_RUECK_RANGE_C = list(
        np.arange(T_RUECK_START_C, T_RUECK_END_C - 0.5 * T_RUECK_STEP_C, -abs(T_RUECK_STEP_C))
        if T_RUECK_END_C < T_RUECK_START_C else
        np.arange(T_RUECK_START_C, T_RUECK_END_C + 0.5 * T_RUECK_STEP_C, abs(T_RUECK_STEP_C))
    )

    print(f"Minimal nötige Generatoreintrittstemperatur T11 vs. Rückkühltemperatur")
    print(f"(T18_spec_C = {config.T18_spec_C:.1f} °C konstant, feste Verdampferaustrittstemperatur)")
    points = sweep_min_generator_temperature_homotopy(T_RUECK_RANGE_C, config)
    print_sweep_table(points)

    duehring_reference = None
    try:
        from AC_duehring_screening import sweep_recool_temperature
    except ImportError:
        try:
            from Design_Point.AC_duehring_screening import sweep_recool_temperature
        except ImportError:
            sweep_recool_temperature = None
    if sweep_recool_temperature is not None:
        duehring_reference = sweep_recool_temperature(
            T_RUECK_RANGE_C, T_evap_target_C=config.T18_spec_C - config.dT_min_evap,
            dT_min_des=config.dT_min_des, dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
        )

    plot_feasibility_sweep(points, duehring_reference=duehring_reference)

    # Zusatzauswertungen: nutzen die oben bereits berechneten `points` weiter
    # (keine erneute Sweep-Berechnung). Lazy Import, damit die beiden
    # Skripte selbst weiterhin eigenständig importierbar/ausführbar bleiben.
    if ENABLE_DUEHRING_MULTI_PLOT:
        from Design_Point.Visualization_Scripts.AC_duehring_multi_process_plot import (
            select_and_plot_duehring,
        )
        select_and_plot_duehring(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{duehring_plot_name}.png",
        )

    if ENABLE_QT_MULTI_PDF:
        from Design_Point.Visualization_Scripts.AC_qt_multi_process_plot import (
            select_and_plot_qt_pdf,
        )
        select_and_plot_qt_pdf(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{qt_pdf_name}.pdf",
        )
