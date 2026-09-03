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
feasible T12-Fenster [T12_min, T12_max] -- nicht nur das Maximum. Jeder
Punkt kostet nur eine Handvoll solve_awt()-Aufrufe statt einer vollen
DE-Suche.

Kernkonzept: Mindest-Hub statt fixer Zieltemperatur
----------------------------------------------------
Statt einer global fixen Nutzwärmesenke T_11_C wird T11 PRO PUNKT als
`T_waste_C + min_lift_offset_C` gesetzt -- eine Design-Vorgabe ("ich will
mindestens X Kelvin Hub über die jeweilige Abwärme"), die mit T_waste
mitskaliert, statt ein fixes Absolutziel zu erzwingen (das bei niedrigem
T_waste einen unrealistisch grossen Hub verlangen würde). Das gefundene
Fenster [T12_min, T12_max] ist direkt als "welche Nutztemperatur kann ich
mir bei dieser Abwärmetemperatur sinnvoll aussuchen" lesbar.

Wichtig zu wissen für die Interpretation
------------------------------------------
- GTL (Gross Temperature Lift) = T12 - T_waste ist die physikalisch
  sinnvolle Kenngrösse eines Wärmetransformators (T12 muss > T_waste sein,
  sonst "transformiert" die Anlage nichts). T11 selbst hat auf die interne
  Machbarkeit praktisch KEINEN Einfluss -- es bestimmt nur den externen
  Absorber-Massenstrom m11 = Q_abs/(cp·(T12-T11)), siehe
  _resolve_absorber_external_stream in Models.AHT_Pinch_Point.
- Der Solver-Warmstart (x0) hat nur ein schmales Einzugsgebiet (oft nur
  ~2-4 K in T12, teils auch in T_waste selbst). Ein zu grosser Sprung lässt
  den Solver in einem Scheinkonvergenzpunkt landen (scipy meldet "success",
  obwohl die Pinch-Residuen deutlich von 0 abweichen). Deshalb arbeiten
  alle Suchfunktionen hier mit kleinschrittigem Kontinuitäts-Walk bzw.
  adaptiver Homotopie (Schrittweite halbieren bei Fehlschlag, vergrössern
  bei Erfolg) -- analog zu AHT_stable_design_point.py, nur über T12/T_waste
  statt über dT_min.
- Manche Fenster sind sehr schmal (<2 K) kurz bevor ein Betriebspunkt an
  seine tatsächliche Machbarkeitsgrenze stösst (die Fensterbreite geht dort
  reproduzierbar gegen 0 -- ein echter Umkehrpunkt, kein Suchraster-
  Artefakt). Ein zu grobes Suchraster kann solche Fenster überspringen und
  fälschlich "nicht lösbar" meldet.

Empfehlung
----------
Erst mit diesem Skript den Grobverlauf des erreichbaren Fensters über
T_waste kartieren (moderate Pinch-/Approach-Werte, siehe Konfiguration
unten). Erst für die 3-5 daraus ausgewählten, tatsächlich interessanten
Betriebspunkte lohnt sich der volle UA-Optimierer (AHT_design_point_optimizer.py).

Aufruf als Skript
-----------------
    python Design_Point/AHT_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, replace
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
    # --- Betriebspunkt-Randbedingungen -------------------------------------
    T_11_C: float = 70.0     # Nutzwärmesenke, kalter Eintritt (nur von
                              # sweep_feasibility() verwendet, siehe dort)
    T_17_C: float = 20.0     # 15.0  Rückkühlung, kalter Eintritt
    Qabs_spec_kW: float = 500.0

    # --- Mindest-Hub-Vorgabe (T11 = T_waste + min_lift_offset_C) -----------
    # Das ist der Wert, den man i.d.R. zuerst anpassen will: "wie viel
    # Kelvin Hub über die Abwärmetemperatur will ich mindestens erreichen".
    # Nur ein technischer Ankerpunkt für die Suche, keine reale Anforderung
    # -- die eigentliche Wahl von T11/T12 trefft ihr anhand des ganzen
    # gefundenen Fensters. Genutzt von sweep_relative_lift_window() und
    # sweep_relative_lift_window_homotopy() (per min_lift_offset_C-Argument
    # dort überschreibbar).
    min_lift_offset_C: float = 2.0

    # --- Pinch-Werte (Design-Annahme, kein Optimierungsziel) ---------------
    # Minimale Temperaturdifferenz am "Pinch Point" jedes Wärmeübertragers
    # (siehe Modul-Docstring / Chat: kleiner = grösserer & teurerer Apparat,
    # aber näher am thermodynamischen Optimum). 5 K ist ein moderater,
    # robust auffindbarer Wert für die Exploration -- für die reale
    # UA-Feinauslegung eines konkret ausgewählten Punktes ggf. schärfer
    # (kleiner) ansetzen, siehe AHT_design_point_optimizer.py.
    dT_min_shex: float = 5.0    # 3.0
    dT_min_des: float = 5.0     # 3.0
    dT_min_cond: float = 5.0    # 3.0
    dT_min_evap: float = 5.0    # 3.0
    dT_min_abs: float = 5.0     # 3.0

    # --- Externe Approach-Werte (Design-Annahme) ----------------------------
    # Externe Austrittstemperaturen von Desorber/Verdampfer/Kondensator,
    # relativ zur Abwärme- bzw. Rückkühltemperatur vorgegeben:
    #   T14 = T_waste - dT_approach_des_C   T16 = T_waste - dT_approach_evap_C
    #   T18 = T_17_C  + dT_approach_cond_C
    # Kleinere Werte = mehr externer Massenstrom = mehr "thermisches Budget"
    # für die Pinch-Werte oben, verschiebt aber auch die Machbarkeitsgrenze
    # bei niedrigem T_waste nach unten (siehe Modul-Docstring).
    dT_approach_des_C: float = 4.0      # 4.0
    dT_approach_evap_C: float = 4.0     # 4.0
    dT_approach_cond_C: float = 3.0     # 3.0

    desorber_evaporator_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # -------------------------------------------------------------------
    # Such-/Solver-Parameter -- i.d.R. NICHT anfassen
    # -------------------------------------------------------------------
    # Alle Schrittweiten sind bewusst klein gehalten: das Einzugsgebiet
    # eines Warmstarts ist empirisch oft nur ~2-4 K breit (siehe
    # Modul-Docstring); ein gröberes Raster überspringt echte, aber schmale
    # Lösungsfenster.
    T12_search_margin_C: float = 1.0   # Startabstand oberhalb T_11_C (nur 1. Punkt)
    T12_step_C: float = 2.0            # Expansionsschritt für die Fenstersuche
    T12_bisect_tol_C: float = 0.2      # Abbruchbreite der Bisektion
    max_expand_steps: int = 40         # bei T12_step_C=2.0 -> bis zu 80 K Reichweite
    max_bisect_steps: int = 25

    anchor_search_span_C: float = 60.0
    anchor_search_step_C: float = 1.0

    # Gelockerte Solver-Toleranzen für die Probe-Solves (Anker-Suche,
    # Expansion, Bisektion). Mit den strengen AWTInputs-Defaults
    # (solver_tol=1e-9, max_nfev=5000) kann jeder fehlschlagende Versuch
    # bis zu 5000 Iterationen brauchen -- bei ~100 Versuchen/Punkt summiert
    # sich das zu Stunden. Analog zum fast=True/False-Muster in
    # AHT_design_point_optimizer.py: schnell/locker suchen, an den beiden
    # gefundenen Fenstergrenzen danach je einmal streng nachrechnen (siehe
    # _refine_boundary).
    probe_solver_tol: float = 1.0e-6
    probe_max_nfev: int = 300
# ---------------------------------------------------------------------------
# Such-Raster für den Sweep -- HIER ANPASSEN
# ---------------------------------------------------------------------------
# Abwärmetemperaturen, die untersucht werden sollen. Die Homotopie startet
# beim höchsten Wert (T_WASTE_START_C) kalt und wandert von dort SCHRITT
# FÜR SCHRITT abwärts bis T_WASTE_END_C -- deshalb sollte T_WASTE_START_C
# ein unproblematischer, hoher Wert bleiben, auch wenn ihr hauptsächlich an
# tieferen Temperaturen interessiert seid. Wie tief T_WASTE_END_C sinnvoll
# gehen kann, hängt von min_lift_offset_C und den Pinch-/Approach-Werten in
# FeasibilitySweepConfig ab -- irgendwann schliesst sich das Fenster an
# einem echten Umkehrpunkt (siehe Modul-Docstring); die Homotopie bricht
# dort automatisch ab und meldet den zuletzt erreichten Wert.
T_WASTE_START_C = 85.0   # höchste untersuchte Abwärmetemperatur [°C]
T_WASTE_END_C = 40.0     # tiefste GEWÜNSCHTE Abwärmetemperatur [°C] (evtl. nicht erreichbar, s.o.)
T_WASTE_STEP_C = 5.0     # Rasterabstand [K]

plot_name = "feasibility_sweep_10_PP_5"

# Zusatzauswertungen aus DEMSELBEN Sweep, ohne ihn erneut zu rechnen (siehe
# __main__ unten) -- jeweils per ENABLE_*-Schalter einzeln abschaltbar. Die
# zugrundeliegenden Skripte (AHT_duehring_multi_process_plot.py /
# AHT_qt_multi_process_plot.py) bleiben auch eigenständig lauffähig.
ENABLE_DUEHRING_MULTI_PLOT = True
duehring_plot_name = "duehring_multi_process_10_PP_5"

ENABLE_QT_MULTI_PDF = True
qt_pdf_name = "qt_multi_process_10_PP_5"

# Wie viele der untersuchten Abwärmetemperaturen in den beiden
# Zusatzauswertungen eingezeichnet werden (2 = jede zweite) -- bei zu vielen
# überlagerten Prozessen wird das Dühring-Diagramm unleserlich.
MULTI_PLOT_EVERY_NTH = 2


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
# Solve-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _build_inputs(
    T_waste_C: float, T12_spec_C: float, config: FeasibilitySweepConfig, *,
    fast: bool = True, T11_C: Optional[float] = None,
) -> AWTInputs:
    """T11_C überschreibt config.T_11_C für einen einzelnen Aufruf -- genutzt
    von den relative-lift-Funktionen, wo T11 pro Punkt aus T_waste_C
    abgeleitet wird statt fix zu sein."""
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

    Das Einzugsgebiet eines gegebenen Warmstarts ist oft nur ~2-4 K breit:
    ein Kandidat 2 K daneben kann von genau demselben x0 aus glatt
    konvergieren, während einer 5-20 K weiter weg divergiert, OBWOHL dort
    ebenfalls eine gültige Lösung existiert. Der Walk reicht deshalb den
    Lösungsvektor JEDES Versuchs weiter (auch wenn er (noch) nicht "valid"
    ist, siehe _solve_raw) -- exakt das Kontinuitätsprinzip aus
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
    gelockerten Toleranzen (config.probe_*); an den beiden gefundenen
    Grenzen wird danach je einmal streng nachgerechnet."""

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


def _duehring_initial_guess_C(
    T_waste_C: float, config: FeasibilitySweepConfig, *, fraction: float = 0.6
) -> Optional[float]:
    """Liefert T_waste_C + fraction * GTL_max(Dühring-Screening) als groben,
    aber grössenordnungsmässig richtigen T12-Schätzwert für den allerersten
    Punkt einer Suche. Ein generischer Schätzwert (z.B. T_11_C+margin) kann
    bei niedrigem T_waste um Grössenordnungen daneben liegen; fraction<1,
    weil das reale (Pinch-)Fenster unter der optimistischen
    Dühring-Obergrenze liegt, aber in derselben Grössenordnung."""
    try:
        from AHT_duehring_screening import estimate_max_gtl
    except ImportError:
        try:
            from Design_Point.AHT_duehring_screening import estimate_max_gtl
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


# ---------------------------------------------------------------------------
# Sweep A: fixe, absolute Senkentemperatur T_11_C für alle T_waste-Werte
# ---------------------------------------------------------------------------

def sweep_feasibility(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    initial_anchor_guess_C: Optional[float] = None,
) -> List[FeasibilityPoint]:
    """Warmstart-verkettete Fenstersuche über ein T_waste-Raster, mit einer
    für ALLE Punkte FIXEN Senkentemperatur config.T_11_C (z.B. "ich habe
    eine reale Anwendung, die genau 70°C braucht"). Bei niedrigem T_waste
    kann das schlicht "kein Fenster" ergeben, weil T_11_C zu hoch angesetzt
    ist -- nicht, weil die Anlage grundsätzlich keinen Hub liefern könnte.
    Für "was ist bei dieser Abwärmetemperatur überhaupt sinnvoll erreichbar"
    eignet sich sweep_relative_lift_window[_homotopy]() besser.

    T_waste_values_C sollte monoton (auf- oder absteigend) sein, damit der
    Warmstart von Punkt zu Punkt trägt -- analog zu sweep_parameter() in
    AHT_design_point_optimizer.py. Der Anker-Schätzwert für Punkt i+1 ist
    das zuletzt gefundene T12_max von Punkt i.

    initial_anchor_guess_C: Startschätzwert NUR für den allerersten Punkt.
    Ohne Angabe wird config.T_11_C + T12_search_margin_C verwendet -- das
    kann bei niedrigem T_waste_C SEHR weit von der tatsächlichen Lösung
    entfernt sein (ein Kontinuitäts-Walk kann eine grosse Lücke durch einen
    lösungsfreien Bereich nicht überbrücken, selbst mit kleinen Schritten).
    Ein guter Schätzwert kommt z.B. aus _duehring_initial_guess_C().
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
# Sweep B: mitskalierende Mindest-Nutztemperatur T11 = T_waste + Hub
# ---------------------------------------------------------------------------

def sweep_relative_lift_window(
    T_waste_values_C: Sequence[float],
    config: FeasibilitySweepConfig,
    *,
    min_lift_offset_C: Optional[float] = None,
) -> List[FeasibilityPoint]:
    """Wie sweep_feasibility(), aber T_11_C wird für jeden Punkt individuell
    als T_waste_C + min_lift_offset_C gesetzt statt global fix (siehe
    Modul-Docstring "Kernkonzept"). min_lift_offset_C ohne Angabe:
    config.min_lift_offset_C.

    Macht pro T_waste-Punkt EINEN Sprung (mit Kontinuitäts-Walk nur in T12,
    T_waste bleibt dabei fest). Reicht, solange der Sprung zwischen zwei
    T_waste-Werten selbst klein genug ist -- für grössere Sprünge (z.B.
    10 K+) sweep_relative_lift_window_homotopy() verwenden, die das
    automatisch mit adaptiver Schrittweite abfängt.
    """
    offset = min_lift_offset_C if min_lift_offset_C is not None else config.min_lift_offset_C

    points: List[FeasibilityPoint] = []
    x0_carry: Optional[np.ndarray] = None
    anchor_guess_carry: Optional[float] = None

    for i, T_waste_C in enumerate(T_waste_values_C):
        T11_this = T_waste_C + offset
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
# Sweep C: wie B, aber mit adaptiver Homotopie ZWISCHEN den Rasterpunkten
# ---------------------------------------------------------------------------
#
# (T_waste, T12) werden GEMEINSAM in kleinen, adaptiven Schritten bewegt
# (GTL = T12 - T_waste dabei näherungsweise konstant gehalten), mit
# automatischer Schrittweitenhalbierung bei Fehlschlag und Vergrösserung
# bei Erfolg -- notwendig, weil auch T_waste-Sprünge von nur ~10 K die
# Warmstart-Kette reissen lassen können, obwohl ausreichend Lösungsraum
# existiert (Scheinkonvergenz: scipy meldet "success", aber die
# Pinch-Residuen weichen deutlich von 0 ab). Das ist die empfohlene,
# robusteste Variante -- siehe __main__ unten.

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
    min_lift_offset_C: Optional[float] = None,
    homotopy_step_initial_C: float = 3.0,
    homotopy_step_min_C: float = 0.25,
) -> List[FeasibilityPoint]:
    """Wie sweep_relative_lift_window(), aber mit adaptiver Homotopie
    ZWISCHEN den Rasterpunkten (siehe Abschnitts-Docstring) statt eines
    einzelnen Sprungs -- die empfohlene, robusteste Variante. T_waste_values_C
    in Wanderreihenfolge angeben (z.B. absteigend von einem hohen,
    unproblematischen Startwert). min_lift_offset_C ohne Angabe:
    config.min_lift_offset_C.
    """
    offset = min_lift_offset_C if min_lift_offset_C is not None else config.min_lift_offset_C

    points: List[FeasibilityPoint] = []
    T_waste_cur: Optional[float] = None
    T12_cur: Optional[float] = None
    x0_cur: Optional[np.ndarray] = None

    for i, T_waste_target in enumerate(T_waste_values_C):
        if x0_cur is None:
            T11_first = T_waste_target + offset
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
                T_waste_cur, T12_cur, x0_cur, T_waste_target, config, offset,
                step_initial_C=homotopy_step_initial_C, step_min_C=homotopy_step_min_C,
            )
            if not fully_reached:
                print(
                    f"  [Homotopie] Ziel {T_waste_target:.2f} °C nicht vollständig erreicht, "
                    f"angehalten bei {reached_T_waste:.2f} °C "
                    f"(Schrittweite unter {homotopy_step_min_C:.2f} K gefallen)."
                )
            T11_target = reached_T_waste + offset
            config_target = replace(config, T_11_C=T11_target)
            # Sicherheitsabstand: reached_T12 kann (durch Bisektionstoleranz/
            # Rundung) sehr nah an T11_target liegen -- ein Anker direkt auf
            # der Grenze verpasst schmale Fenster (die kurz vor einer echten
            # Machbarkeitsgrenze auftreten können, siehe Modul-Docstring).
            # Anker bewusst spürbar oberhalb T11_target ansetzen.
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


# ---------------------------------------------------------------------------
# Ausgabe: Tabelle + Plot
# ---------------------------------------------------------------------------

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
    save_path: Optional[str] = f"Design_Point/Plots/{plot_name}.png",
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

    # Alle physikalischen Annahmen (Pinch-Werte, Approach-Werte,
    # Mindest-Hub-Vorgabe min_lift_offset_C, T_17_C) stehen zentral in
    # FeasibilitySweepConfig oben -- dort anpassen, nicht hier.
    T_WASTE_RANGE_C = list(
        np.arange(T_WASTE_START_C, T_WASTE_END_C - 0.5 * T_WASTE_STEP_C, -T_WASTE_STEP_C)
    )

    print(f"Erreichbares Nutztemperatur-Fenster, T11 = T_waste + {config.min_lift_offset_C:.0f} K")
    print(f"(T_17_C = {config.T_17_C:.1f} °C konstant)")
    points = sweep_relative_lift_window_homotopy(T_WASTE_RANGE_C, config)
    print_sweep_table(points)

    duehring_reference = None
    try:
        from AHT_duehring_screening import sweep_waste_heat_temperature

        duehring_reference = sweep_waste_heat_temperature(
            T_WASTE_RANGE_C, T17_C=config.T_17_C,
            dT_min_des=config.dT_min_des, dT_min_evap=config.dT_min_evap,
            dT_min_cond=config.dT_min_cond, dT_min_abs=config.dT_min_abs,
        )
    except ImportError:
        pass

    plot_feasibility_sweep(points, duehring_reference=duehring_reference)

    # Zusatzauswertungen: nutzen die oben bereits berechneten `points` weiter
    # (keine erneute Sweep-Berechnung). Lazy Import, damit die beiden
    # Skripte selbst weiterhin eigenständig importierbar/ausführbar bleiben.
    if ENABLE_DUEHRING_MULTI_PLOT:
        try:
            from Design_Point.Visualization_Scripts.AHT_duehring_multi_process_plot import select_and_plot_duehring
        except ImportError:
            from Design_Point.Visualization_Scripts.AHT_duehring_multi_process_plot import (
                select_and_plot_duehring,
            )
        select_and_plot_duehring(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{duehring_plot_name}.png",
        )

    if ENABLE_QT_MULTI_PDF:
        try:
            from Design_Point.Visualization_Scripts.AHT_qt_multi_process_plot import select_and_plot_qt_pdf
        except ImportError:
            from Design_Point.Visualization_Scripts.AHT_qt_multi_process_plot import (
                select_and_plot_qt_pdf,
            )
        select_and_plot_qt_pdf(
            points, every_nth=MULTI_PLOT_EVERY_NTH,
            save_path=f"Design_Point/Plots/{qt_pdf_name}.pdf",
        )
