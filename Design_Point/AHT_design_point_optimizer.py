"""Designpoint-Optimierung für den Absorptionswärmetransformator AHT.

Analog zum Designpoint-Optimierer der Kältemaschine (AC_design_point_optimizer.py),
aber für Models.AHT_Pinch_Point statt Models.AC_Pinch_Point.

WICHTIGER UNTERSCHIED zur Kältemaschine -- vertauschte Rollen der Apparate:
  - Absorber liefert die NUTZWÄRME (hohes Temperaturniveau, "Produkt" des AHT).
    Spec-Variablen: absorber_spec_mode="T12", T12_spec_C (Nutzwärmesenke,
    externe Eintritts-/Austrittstemperatur T_11_C -> T12_spec_C).
  - Desorber UND Verdampfer werden BEIDE von der externen Abwärmequelle
    gespeist (mittleres Temperaturniveau) -- daher
    "desorber_evaporator_routing_mode" (parallel / seriell), analog zur
    "absorber_condenser_routing_mode" bei der Kältemaschine, nur mit
    vertauschten Apparaten.
  - Kondensator rückkühlt auf niedrigem Temperaturniveau (T_17_C -> T18_spec_C).
  - cycle_scale_spec_mode="Qabs" mit Qabs_spec_kW: die Design-Nutzwärmeleistung
    (Analog zu Qevap_spec_kW bei der Kältemaschine) wird am ABSORBER vorgegeben.

Stufe 2 (Teillast-Verifikation, AHT_UA_LMTD)
---------------------------------------------
Analog zur Kältemaschine: UA-Werte und externe Massenströme aus Stufe 1 werden
eingefroren, für definierte Randbedingungs-Szenarien wird NUR simuliert, NICHT
erneut optimiert. Per config.run_stage2 an-/abschaltbar.

ANNAHME (bitte prüfen, da mir Models.AHT_UA_LMTD nicht vorliegt): Struktur
analog zu Models.AC_UA_LMTD -- Desorber/Verdampfer/Kondensator als feste
Massenstrom-Kwargs (m_13, m_15, m_17), NUR der Absorber (designbestimmende
Apparategruppe) mit spec_mode "m11"/"T12" wählbar, cycle_scale_spec_mode="m6"
mit m6_spec. Falls falsch: bitte Rückmeldung bzw. Quelltext/Main-Skript von
AHT_UA_LMTD.py schicken, dann passe ich build_ua_inputs() exakt an.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    TQDM_AVAILABLE = False

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models.AHT_Pinch_Point import (
    AHTInputs,
    bounds as aht_bounds,
    initial_guess as aht_initial_guess,
    solve_aht,
)

try:
    from Models.AHT_UA_LMTD import (
        AHTInputs as UAInputs,
        solve_aht as solve_ua,
    )
    UA_MODEL_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    UA_MODEL_AVAILABLE = False
    _ua_import_error = exc


def _clip_to_bounds(z: np.ndarray, inputs: AHTInputs) -> np.ndarray:
    """Sicherheitsnetz: clippt einen Startvektor defensiv auf die Modell-Bounds."""
    lower, upper = aht_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------

@dataclass
class DesignPointConfig:
    """Fixierte Randbedingungen des AHT-Designpunkts."""

    # Externe Eintrittstemperaturen [°C]
    T_11_C: float = 63.0    # Nutzwärmesenke (Absorber), kalter Eintritt
    T_13_C: float = 57.0    # Abwärmequelle (Desorber/Verdampfer, routing-abhängig)
    T_15_C: float = 57.0    # Abwärmequelle (Desorber/Verdampfer, routing-abhängig)
    T_17_C: float = 20.0    # Rückkühlung (Kondensator), kalter Eintritt

    # Externe Austrittstemperatur-Spezifikationen [°C]
    T12_spec_C: float = 67.0    # Nutzwärmesenke, Austritt (Absorber)
    T14_spec_C: float = 52.0    # Abwärmequelle, Austritt (Desorber)
    T16_spec_C: float = 52.0    # Abwärmequelle, Austritt (Verdampfer)
    T18_spec_C: float = 24.0    # Rückkühlung, Austritt (Kondensator)

    # Design-Nutzwärmeleistung [kW] (am Absorber, NICHT am Verdampfer!)
    Qabs_spec_kW: float = 500.0

    desorber_evaporator_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    dT_floor: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 3.0, "des": 3.0, "cond": 3.0, "evap": 3.0, "abs": 3.0,
        }
    )
    dT_search_range: float = 30.0

    ua_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 1.0, "des": 1.0, "cond": 1.0, "evap": 1.0, "abs": 1.0,
        }
    )

    opt_solver_tol: float = 1.0e-6
    opt_max_nfev: int = 150

    de_popsize: int = 12
    de_maxiter: int = 60
    de_workers: int = 1

    de_patience: Optional[int] = 25
    de_min_improvement: float = 0.01  # kW/K

    # MUSS deutlich über jedem plausiblen echten Sum(UA)-Wert liegen --
    # sonst kann ein infeasibler Punkt mit winzigem Restfehler (Strafkosten
    # ~penalty_base, da der residual-Anteil dann fast 0 ist) GÜNSTIGER
    # aussehen als jedes echte, feasible Design, und DE "optimiert" dann in
    # Richtung eines Punktes, der nur eine Plausibilitätsprüfung (nicht die
    # Konvergenz) verletzt (siehe Chat: DE-Ergebnis war zweimal exakt
    # Sum(UA)=alter_penalty_base=200 -- kein Zufall, sondern genau dieser
    # Effekt).
    penalty_base: float = 5000.0
    penalty_residual_weight: float = 20.0

    make_convergence_plot: bool = True
    convergence_plot_path: str = "Design_Point/Plots/AHT_stage1_convergence_AHT.png"

    # Optionaler manueller Startvektor (interne Modell-Einheiten, K/-, Reihenfolge
    # wie primary_variables) als Fallback für den ALLERERSTEN Optimierer-Aufruf,
    # bevor der Warmstart-Cache befüllt ist. Sinnvoll bei schwierigen/neuen
    # Betriebsbedingungen, wenn du bereits eine handgetunte, konvergierende
    # Lösung für einen ähnlichen Betriebspunkt kennst (z.B. aus deinem
    # main-Skript oder aus quick_feasibility_probe()) -- deutlich zuverlässiger
    # als die generische initial_guess()-Heuristik des Modells.
    x0_override: Optional[np.ndarray] = None

    # Optionale HARTE obere Schranken je Wärmeübertrager (überschreibt
    # dT_floor[key] + dT_search_range für die genannten Keys). Nützlich, wenn
    # die Optimierung an der oberen Grenze "klebt" (siehe Diagnose-Hinweis am
    # Ende von optimize_design_point) -- das zeigt einen zu engen Suchraum für
    # DIESEN Betriebspunkt an, nicht zwingend ein Warmstart-Problem.
    #
    # HIER bewusst ASYMMETRISCH gesetzt (nicht ein einheitlicher Cap für alle
    # fünf!): eine frühere Untersuchung (siehe Chat) hat gemessen, wie stark
    # jeder einzelne Pinch den erreichbaren GTL kostet, ausgehend von einer
    # 3K-Basis (T_waste=70°C, Approach 4/4/3K):
    #   dT_min_shex: -0.25 K GTL pro +2 K Pinch  (~0.13 K/K -- kaum Einfluss)
    #   dT_min_des:  -2.63 K GTL pro +2 K Pinch  (~1.31 K/K)
    #   dT_min_cond: -2.50 K GTL pro +2 K Pinch  (~1.25 K/K)
    #   dT_min_evap: -2.38 K GTL pro +2 K Pinch  (~1.19 K/K)
    #   dT_min_abs:  -2.00 K GTL pro +2 K Pinch  (~1.00 K/K)
    # SHEX kostet also kaum GTL (ein weiter SHEX-Pinch ist meist sogar
    # GÜNSTIG fürs Optimierungsziel Sum(UA): kleinere SHEX-Fläche, kaum
    # GTL-Verlust) -- deshalb hier NICHT eingeschränkt (bleibt bei
    # dT_floor+dT_search_range). Die anderen vier teilen sich dagegen
    # effektiv ein gemeinsames "GTL-Budget": bei diesem Betriebspunkt
    # (T12_spec_C=67°C, T15_C=57°C) sind das ca. 10 K. Die Caps unten sind
    # `floor + budget/Sensitivität`, also die Pinch-Erhöhung, bei der DIESER
    # Wärmeübertrager ALLEIN das gesamte Budget aufbrauchen würde (die
    # anderen drei müssten dann nahe dem Floor bleiben) -- ein grosszügiger,
    # aber nicht mehr sinnlos weiter Rahmen. Diese Sensitivitäten wurden bei
    # EINEM anderen Betriebspunkt gemessen; die Grössenordnung/Reihenfolge
    # sollte übertragbar sein, die genauen Zahlen sind ein Startwert, keine
    # exakte Herleitung für DIESEN Betriebspunkt.
    dT_upper: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "des": 11.0, "cond": 11.0, "evap": 11.0, "abs": 13.0,
        }
    )

    # Schneller Feasibility-Test vor der vollen Optimierung (siehe
    # quick_feasibility_probe) -- kostet nur Sekunden bis wenige Minuten und
    # hätte den 19h-Fehlschlag früh erkennbar gemacht.
    run_feasibility_probe: bool = True

    # Stufe 1b (Nelder-Mead-Politur) für schnelle Test-Läufe abschaltbar.
    # Kann bei fast durchgehend infeasiblem DE-Ergebnis (z.B. bei sehr
    # kleinem de_popsize/de_maxiter zum Testen) sehr lange brauchen, ohne
    # etwas zu verbessern -- NM hat kein Early-Stopping und läuft dann bis
    # maxiter=500 durch reines Herumirren im Penalty-Bereich (siehe Chat:
    # 104 Minuten NM bei einem Testlauf mit nur 3 feasiblen Punkten).
    run_local_polish: bool = True

    # Stufe 2: Teillast-Verifikation an/aus. Wenn False, wird verify_part_load()
    # gar nicht erst aufgerufen (spart die paar Sekunden, hauptsächlich nützlich
    # während du an Stufe 1 experimentierst und Stufe 2 gerade nicht brauchst).
    run_stage2: bool = False


@dataclass
class PartLoadScenario:
    """Ein zu verifizierender Randbetriebspunkt (Stufe 2). WERTE ANPASSEN --
    aktuell nur Platzhalter relativ zum Nominal-Designpunkt oben."""

    name: str
    T_11_C: float
    T_13_C: float
    T_15_C: float
    T_17_C: float


DEFAULT_SCENARIOS: List[PartLoadScenario] = [
    PartLoadScenario(name="Extremfall_kalt", T_11_C=70.0, T_13_C=55.0, T_15_C=55.0, T_17_C=15.0),
    PartLoadScenario(name="Extremfall_warm", T_11_C=70.0, T_13_C=62.0, T_15_C=62.0, T_17_C=25.0),
]


# ---------------------------------------------------------------------------
# Stufe 1: Designpoint-Optimierung
# ---------------------------------------------------------------------------

THETA_ORDER = ["shex", "des", "cond", "evap", "abs"]


@dataclass
class EvalStats:
    calls: int = 0
    feasible: int = 0
    infeasible: int = 0
    hit_nfev_cap: int = 0
    total_solve_time_s: float = 0.0
    history: List[Tuple[int, float]] = field(default_factory=list)

    def record(self, cost: float, feasible: bool, hit_cap: bool, dt: float) -> None:
        self.calls += 1
        self.total_solve_time_s += dt
        if feasible:
            self.feasible += 1
        else:
            self.infeasible += 1
        if hit_cap:
            self.hit_nfev_cap += 1
        self.history.append((self.calls, cost))

    @property
    def avg_solve_time_s(self) -> float:
        return self.total_solve_time_s / self.calls if self.calls else 0.0


class DEProgress:
    """Fortschrittsanzeige + Early Stopping für differential_evolution."""

    def __init__(
        self,
        maxiter: int,
        stats: EvalStats,
        patience: Optional[int] = None,
        min_improvement: float = 0.02,
    ):
        self.stats = stats
        self.maxiter = maxiter
        self.gen = 0
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_history: List[float] = []
        self.pbar = tqdm(total=maxiter, desc="Stufe 1a", unit="Gen") if TQDM_AVAILABLE else None

    def __call__(self, xk, convergence) -> bool:
        self.gen += 1
        current_best = min((c for _, c in self.stats.history), default=float("inf"))
        self.best_history.append(current_best)

        stop = False
        if self.patience is not None and len(self.best_history) > self.patience:
            improvement = self.best_history[-self.patience - 1] - self.best_history[-1]
            if improvement < self.min_improvement:
                stop = True

        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix(
                conv=f"{convergence:.2e}",
                best=f"{current_best:.4f}",
                calls=self.stats.calls,
                avg_s=f"{self.stats.avg_solve_time_s:.2f}",
            )
        else:
            print(
                f"  Gen {self.gen}/{self.maxiter} | best={current_best:.4f} "
                f"| convergence={convergence:.2e} | Aufrufe={self.stats.calls} "
                f"| avg={self.stats.avg_solve_time_s:.2f}s"
            )

        if stop:
            msg = (
                f"\n  Early Stopping: seit {self.patience} Generationen "
                f"Verbesserung < {self.min_improvement} kW/K -- Suche wird beendet."
            )
            if self.pbar is not None:
                self.pbar.write(msg)
            else:
                print(msg)

        return stop

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class WarmStartCache:
    """Hält den letzten konvergierten Primärvektor für Warmstarts.

    fallback: optionaler manueller Startvektor (config.x0_override), der
    verwendet wird, solange noch kein konvergierter Punkt im Cache ist --
    zuverlässiger als die generische initial_guess()-Heuristik, wenn du
    bereits eine handgetunte Lösung für einen ähnlichen Betriebspunkt kennst.
    """

    def __init__(self, fallback: Optional[np.ndarray] = None) -> None:
        self.z: Optional[np.ndarray] = None
        self.fallback = fallback

    def get(self, inputs: AHTInputs) -> np.ndarray:
        if self.z is not None:
            z = self.z
        elif self.fallback is not None:
            z = self.fallback
        else:
            z = aht_initial_guess(inputs)
        return _clip_to_bounds(z, inputs)

    def update(self, z: np.ndarray) -> None:
        self.z = np.asarray(z, dtype=float).copy()

    def reset(self) -> None:
        self.z = None


def build_aht_inputs(
    theta: np.ndarray, config: DesignPointConfig, *, fast: bool
) -> AHTInputs:
    """Baut die AHT-Inputs. fast=True -> gelockerte Solver-Toleranzen (Stufe 1)."""
    dT_shex, dT_des, dT_cond, dT_evap, dT_abs = theta
    kwargs = dict(
        T_11_C=config.T_11_C,
        T_13_C=config.T_13_C,
        T_15_C=config.T_15_C,
        T_17_C=config.T_17_C,
        dT_min_shex=float(dT_shex),
        dT_min_des=float(dT_des),
        dT_min_cond=float(dT_cond),
        dT_min_evap=float(dT_evap),
        dT_min_abs=float(dT_abs),
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="Qabs",
        Qabs_spec_kW=config.Qabs_spec_kW,
        absorber_spec_mode="T12",
        T12_spec_C=config.T12_spec_C,
        desorber_spec_mode="T14",
        T14_spec_C=config.T14_spec_C,
        evaporator_spec_mode="T16",
        T16_spec_C=config.T16_spec_C,
        condenser_spec_mode="T18",
        T18_spec_C=config.T18_spec_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.opt_solver_tol
        kwargs["max_nfev"] = config.opt_max_nfev
    return AHTInputs(**kwargs)


def _is_feasible(result) -> bool:
    if not result.solve_info.success or not result.solve_info.final_point_evaluable:
        return False
    if not result.checks:
        return False
    return all(result.checks.values())


def design_point_objective(
    theta: np.ndarray,
    config: DesignPointConfig,
    cache: WarmStartCache,
    stats: EvalStats,
) -> float:
    inputs = build_aht_inputs(theta, config, fast=True)
    x0 = cache.get(inputs)

    t0 = time.perf_counter()
    result = solve_aht(inputs, x0=x0)
    feasible = _is_feasible(result)

    # Rückfall auf einen frischen, generischen Startvektor, wenn der
    # gecachte Warmstart scheitert. Der Cache hält nur EINEN Vektor -- den
    # vom letzten ERFOLGREICHEN Theta. DE springt zwischen Generationen aber
    # oft weit im 5D-Raum herum (kein kontinuierlicher Pfad), und das
    # Solver-Einzugsgebiet ist empirisch nur ~2-4 K breit (siehe
    # AHT_feasibility_sweep.py). Ohne Rückfall hängt ein schlecht
    # warmgestarteter Versuch oft bis zum nfev-Limit fest, statt sauber zu
    # konvergieren ODER sauber zu scheitern -- das war die Hauptursache für
    # die 72% nfev-Limit-Treffer und die 16h-Laufzeit im 500kW-Testlauf.
    # aht_initial_guess() kennt das aktuelle Theta nicht, ist aber oft ein
    # deutlich besserer Startpunkt für ein NEUES Theta als ein Warmstart von
    # einem ganz anderen Theta.
    if not feasible:
        x0_fresh = _clip_to_bounds(aht_initial_guess(inputs), inputs)
        result_fresh = solve_aht(inputs, x0=x0_fresh)
        if _is_feasible(result_fresh):
            result = result_fresh
            feasible = True

    dt = time.perf_counter() - t0

    hit_cap = (
        not result.solve_info.success
        and result.solve_info.nfev >= config.opt_max_nfev
    )

    if not feasible:
        penalty = config.penalty_base + config.penalty_residual_weight * result.solve_info.scaled_residual_norm
        stats.record(penalty, feasible=False, hit_cap=hit_cap, dt=dt)
        return float(penalty)

    cache.update(np.array(list(result.primary_variables.values()), dtype=float))

    ua = result.UA_conversion
    total = (
        config.ua_weights["shex"] * ua["UA_shex"]
        + config.ua_weights["des"] * ua["UA_des"]
        + config.ua_weights["cond"] * ua["UA_cond"]
        + config.ua_weights["evap"] * ua["UA_evap"]
        + config.ua_weights["abs"] * ua["UA_abs"]
    )
    stats.record(total, feasible=True, hit_cap=False, dt=dt)
    return float(total)


def _theta_bounds(config: DesignPointConfig) -> List[tuple]:
    result = []
    for key in THETA_ORDER:
        lo = config.dT_floor[key]
        if config.dT_upper and key in config.dT_upper:
            hi = config.dT_upper[key]
        else:
            hi = lo + config.dT_search_range
        result.append((lo, hi))
    return result


def plot_convergence(stats: EvalStats, path: str) -> None:
    if not stats.history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nicht installiert -- Konvergenzplot wird übersprungen.")
        return

    calls, costs = zip(*stats.history)
    costs = np.asarray(costs, dtype=float)
    running_best = np.minimum.accumulate(costs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(calls, costs, s=6, alpha=0.35, label="Einzelauswertung (inkl. Penalty)")
    ax.plot(calls, running_best, color="tab:red", linewidth=2, label="Bestes Sum(UA) bisher")
    ax.set_xlabel("Funktionsauswertung")
    ax.set_ylabel("Zielfunktionswert [kW/K]")
    ax.set_yscale("log")
    ax.set_title("AHT -- Stufe 1: Konvergenzverlauf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Konvergenzplot gespeichert: {path}")


def optimize_design_point(
    config: DesignPointConfig, *, seed: int = 42, verbose: bool = True
) -> tuple[np.ndarray, "object", EvalStats]:
    cache = WarmStartCache(fallback=config.x0_override)
    stats = EvalStats()
    bounds = _theta_bounds(config)

    t_stage1_start = time.perf_counter()

    if verbose:
        print("Stufe 1a: globale Suche (differential_evolution) ...")

    progress = DEProgress(
        config.de_maxiter, stats, patience=config.de_patience, min_improvement=config.de_min_improvement
    )
    updating = "deferred" if config.de_workers != 1 else "immediate"

    t0 = time.perf_counter()
    de_result = differential_evolution(
        design_point_objective,
        bounds=bounds,
        args=(config, cache, stats),
        seed=seed,
        popsize=config.de_popsize,
        maxiter=config.de_maxiter,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=False,
        workers=config.de_workers,
        updating=updating,
        callback=progress,
        disp=False,
    )
    t_de = time.perf_counter() - t0
    progress.close()

    if verbose:
        print(
            f"  DE-Ergebnis: theta = {de_result.x}, Sum(UA) = {de_result.fun:.4f} "
            f"| Dauer: {t_de/60:.1f} min"
        )

    if config.run_local_polish:
        if verbose:
            print("Stufe 1b: lokale Politur (Nelder-Mead) ...")

        t0 = time.perf_counter()
        nm_result = minimize(
            design_point_objective,
            x0=de_result.x,
            args=(config, cache, stats),
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-3, "fatol": 1e-4, "adaptive": True, "maxiter": 500},
        )
        t_nm = time.perf_counter() - t0

        theta_opt = nm_result.x if nm_result.fun <= de_result.fun else de_result.x

        if verbose:
            print(
                f"  Finales theta = {theta_opt}, "
                f"Sum(UA) = {min(nm_result.fun, de_result.fun):.4f} | Dauer: {t_nm:.1f} s"
            )
    else:
        t_nm = 0.0
        theta_opt = de_result.x
        if verbose:
            print("Stufe 1b: lokale Politur (Nelder-Mead) übersprungen (config.run_local_polish=False).")

    if verbose:
        print("Finaler Präzisions-Solve (strenge Toleranzen) ...")

    t0 = time.perf_counter()
    inputs_opt = build_aht_inputs(theta_opt, config, fast=False)
    result_opt = solve_aht(inputs_opt, x0=cache.get(inputs_opt))
    t_final = time.perf_counter() - t0

    final_feasible = _is_feasible(result_opt)
    if not final_feasible:
        if verbose:
            print(
                "  WARNUNG: Präzisions-Solve (strenge Toleranzen) ist NICHT feasible "
                f"(Nachricht: '{result_opt.solve_info.message}'). Falle zurück auf das "
                "Ergebnis mit den gelockerten Optimierungs-Toleranzen (fast=True), "
                "damit du trotzdem sehen kannst, was gefunden wurde."
            )
        inputs_fallback = build_aht_inputs(theta_opt, config, fast=True)
        result_opt = solve_aht(inputs_fallback, x0=cache.get(inputs_fallback))
        final_feasible = _is_feasible(result_opt)

    t_stage1_total = time.perf_counter() - t_stage1_start

    if verbose:
        print()
        print("Zeitübersicht Stufe 1")
        print(f"  1a Globale Suche (DE)   : {t_de/60:8.2f} min")
        print(f"  1b Lokale Politur (NM)  : {t_nm:8.2f} s")
        print(f"  Finaler Präzisions-Solve: {t_final:8.2f} s")
        print(f"  Gesamt Stufe 1          : {t_stage1_total/60:8.2f} min")
        print()
        print("Auswertungs-Diagnostik")
        print(f"  Gesamtaufrufe               : {stats.calls}")
        print(f"  davon feasible               : {stats.feasible}")
        print(f"  davon infeasible/Penalty     : {stats.infeasible}")
        print(f"  davon nfev-Limit erreicht    : {stats.hit_nfev_cap}"
              f" (von opt_max_nfev={config.opt_max_nfev})")
        print(f"  mittlere Solve-Zeit/Aufruf   : {stats.avg_solve_time_s*1000:.1f} ms")
        if stats.calls:
            infeasible_frac = stats.infeasible / stats.calls
            print(f"  Infeasible-Anteil            : {infeasible_frac*100:.1f}%")
            if infeasible_frac > 0.25:
                print(
                    "  HINWEIS: hoher Infeasible-Anteil -- dT_search_range/dT_upper "
                    "evtl. zu eng gewählt für diesen Betriebspunkt."
                )
        if stats.calls and stats.hit_nfev_cap / stats.calls > 0.1:
            print("  WARNUNG: >10% der Aufrufe erreichen das nfev-Limit -- opt_max_nfev prüfen.")

        # Boundary-Diagnose: klebt theta an der oberen Suchgrenze?
        near_upper = []
        for key, val, (lo, hi) in zip(THETA_ORDER, theta_opt, bounds):
            span = hi - lo
            if span > 0 and (val - lo) / span > 0.9:
                near_upper.append(key)
        if near_upper:
            print(
                f"  HINWEIS: theta liegt für {near_upper} nahe der OBEREN Suchgrenze -- "
                "das deutet auf einen zu eng gewählten Suchraum (dT_search_range/dT_upper) "
                "für diesen Betriebspunkt hin, nicht zwingend auf ein Warmstart-Problem."
            )

    if config.make_convergence_plot:
        plot_convergence(stats, config.convergence_plot_path)

    if not final_feasible:
        print(
            "\nACHTUNG: Auch der Fallback-Solve (gelockerte Toleranzen) ist nicht "
            "feasible. Das zurückgegebene Ergebnis entspricht KEINER validen Lösung -- "
            "bitte NICHT für UA-Werte verwenden. theta_opt liegt vermutlich außerhalb "
            "des tatsächlich lösbaren Bereichs (siehe Boundary-Hinweis oben)."
        )

    return theta_opt, result_opt, stats


def print_design_point_summary(theta: np.ndarray, result) -> None:
    print("=" * 90)
    print("AHT Designpoint-Optimierung -- Ergebnis")
    print("=" * 90)
    for key, val in zip(THETA_ORDER, theta):
        print(f"  dT_min_{key:5s}: {val:8.4f} K")
    print()

    if not _is_feasible(result) or not result.UA_conversion:
        # solve_aht() liefert bei final_point_evaluable=False bewusst ein
        # LEERES UA_conversion (siehe Models.AHT_Pinch_Point) -- UA-Werte aus
        # einem ungültigen Zustand wären bedeutungslos. theta_opt oben ist
        # dann das beste GEFUNDENE (aber nicht valide) theta, kein
        # Auslegungsergebnis.
        print(
            "KEIN valides Ergebnis (result.solve_info.final_point_evaluable=False "
            "oder ein Plausibilitätscheck ist verletzt) -- UA-Werte, Massenströme "
            "und KPIs können nicht sinnvoll ausgegeben werden. theta oben ist "
            f"lediglich das beste GEFUNDENE, nicht valide theta "
            f"(Solver-Nachricht: '{result.solve_info.message}')."
        )
        print("=" * 90)
        return

    print("UA-Werte [kW/K]")
    for key in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"]:
        print(f"  {key:10s}: {result.UA_conversion[key]:10.4f}")
    total_ua = sum(result.UA_conversion[k] for k in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"])
    print(f"  {'Sum(UA)':10s}: {total_ua:10.4f}")
    print()
    print("Externe Massenströme am Designpunkt [kg/s]")
    labels = {
        "m11_kg_s": "Absorber (Nutzwärme)",
        "m13_kg_s": "Desorber (Quelle)",
        "m15_kg_s": "Verdampfer (Quelle)",
        "m17_kg_s": "Kondensator (Rückkühlung)",
    }
    for key, label in labels.items():
        print(f"  {key:10s} [{label:24s}]: {result.diagnostics[key]:10.6f}")
    m6 = result.diagnostics["m6_kg_s"]
    print(f"  {'m6':10s} [interner Lösungsstrom  ]: {m6:10.6f}")
    print()
    print("KPIs")
    for k, v in result.kpis.items():
        try:
            print(f"  {k:10s}: {float(v):10.4f}")
        except (TypeError, ValueError):
            print(f"  {k:10s}: {v}")
    print("=" * 90)


def quick_feasibility_probe(
    config: DesignPointConfig, thetas: Optional[List[np.ndarray]] = None
) -> None:
    """Schneller Test (Sekunden bis wenige Minuten) VOR einer vollen Optimierung.

    Prüft ein paar Kandidaten-Theta-Vektoren MIT STRENGEN Solver-Einstellungen
    (fast=False, wie der finale Solve) und zeigt Feasibility + Rechenzeit.
    Damit siehst du innerhalb von Minuten, ob dT_floor/dT_search_range/dT_upper/
    x0_override für einen neuen (evtl. schwierigen) Betriebspunkt überhaupt
    sinnvoll gewählt sind -- BEVOR du Stunden in eine volle DE-Suche investierst
    (siehe die 19h-Erfahrung mit einem zu engen/falschen Setup).

    Ohne eigene thetas: testet automatisch drei Punkte (nah an der unteren
    Schranke, Mitte, nah an der oberen Schranke) je Wärmeübertrager-Kombination.
    """
    bounds = _theta_bounds(config)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    if thetas is None:
        thetas = [
            lo + 0.1 * (hi - lo),
            (lo + hi) / 2.0,
            hi - 0.1 * (hi - lo),
        ]

    cache = WarmStartCache(fallback=config.x0_override)

    print("=" * 90)
    print("Schneller Feasibility-Test (empfohlen VOR einer vollen Optimierung)")
    print("=" * 90)
    any_feasible = False
    for theta in thetas:
        inputs = build_aht_inputs(np.asarray(theta, dtype=float), config, fast=False)
        x0 = cache.get(inputs)
        t0 = time.perf_counter()
        result = solve_aht(inputs, x0=x0)
        dt = time.perf_counter() - t0
        feasible = _is_feasible(result)
        any_feasible = any_feasible or feasible
        status = "FEASIBLE  " if feasible else "INFEASIBLE"
        print(
            f"  theta={np.round(theta, 2)} -> {status} "
            f"({dt:.2f}s, nfev={result.solve_info.nfev})"
        )
        if feasible:
            cache.update(np.array(list(result.primary_variables.values()), dtype=float))
        else:
            print(f"    Solver-Nachricht: {result.solve_info.message}")
    if not any_feasible:
        print(
            "\n  KEIN Testpunkt feasible -- bevor du eine volle Optimierung startest,"
            " prüfe Randbedingungen (Qabs_spec_kW, Temperaturniveaus) auf grundsätzliche"
            " Lösbarkeit, z.B. mit noch größeren dT_upper-Werten als Test."
        )
    print("=" * 90)


def sweep_parameter(
    base_config: DesignPointConfig,
    param_name: str,
    values: List[float],
    *,
    verbose: bool = True,
) -> List[Tuple[float, np.ndarray, "object"]]:
    """Kontinuitäts-/Homotopie-Sweep für Sensitivitätsanalysen.

    Verändert GENAU EINEN Parameter (z.B. 'T_17_C') über eine Werteliste,
    hält alle anderen Randbedingungen fix, und führt für jeden Wert eine
    VOLLE Designpoint-Optimierung durch -- aber mit einem entscheidenden
    Unterschied zu unabhängigen Einzelläufen: das konvergierte theta UND der
    konvergierte Primärvektor des vorherigen Werts werden als Startpunkt
    (x0_override) für den nächsten Wert verwendet.

    Das ist genau das Kontinuitäts-/Homotopie-Prinzip aus der Chat-Erklärung:
    kleine Schritte von einem bekannten guten Punkt aus, statt jedes Mal "kalt"
    zu starten. Für eine Sensitivitätsanalyse über ein Temperaturniveau ist das
    i.d.R. sowohl SCHNELLER (guter Warmstart => schnellere Konvergenz, evtl.
    kleinere de_maxiter/de_popsize ausreichend) als auch ROBUSTER (der
    Optimierer startet nie mehr komplett "blind").

    HINWEIS: values sollte in aufsteigender ODER absteigender Reihenfolge sein
    (monoton), nicht wild gemischt -- sonst sind die Schritte zwischen
    aufeinanderfolgenden Werten größer als nötig und der Warmstart-Vorteil
    schrumpft.

    Rückgabe: Liste von (wert, theta_opt, result) für jeden erfolgreich
    gelösten Wert. Werte, bei denen auch der Fallback-Solve infeasible war,
    werden übersprungen (mit Warnung), der Sweep läuft aber weiter.
    """
    results: List[Tuple[float, np.ndarray, "object"]] = []
    x0_carry: Optional[np.ndarray] = base_config.x0_override
    theta_carry: Optional[np.ndarray] = None

    for i, value in enumerate(values):
        if verbose:
            print("\n" + "#" * 90)
            print(f"# Sweep-Schritt {i+1}/{len(values)}: {param_name} = {value}")
            print("#" * 90)

        cfg = replace(base_config, **{param_name: value}, x0_override=x0_carry)

        # Ab dem zweiten Schritt: Suchraum um das vorherige theta herum
        # verengen (spart Zeit, da wir schon wissen, wo die Lösung ungefähr
        # liegt) -- nur wenn der Aufrufer nicht ohnehin dT_upper gesetzt hat.
        if theta_carry is not None and cfg.dT_upper is None:
            margin = 5.0  # K, bewusst grosszügig um den vorherigen Punkt herum
            new_floor = {
                k: max(0.5, theta_carry[j] - margin) for j, k in enumerate(THETA_ORDER)
            }
            new_upper = {k: theta_carry[j] + margin for j, k in enumerate(THETA_ORDER)}
            cfg = replace(cfg, dT_floor=new_floor, dT_upper=new_upper)

        try:
            theta_opt, result, _ = optimize_design_point(cfg, verbose=verbose)
        except Exception as exc:  # pragma: no cover
            print(f"  Sweep-Schritt {param_name}={value} fehlgeschlagen: {exc}")
            continue

        if not _is_feasible(result):
            print(
                f"  WARNUNG: {param_name}={value} lieferte kein feasibles Ergebnis "
                "-- wird übersprungen, Sweep läuft weiter."
            )
            continue

        results.append((value, theta_opt, result))
        theta_carry = theta_opt
        x0_carry = np.array(list(result.primary_variables.values()), dtype=float)

    return results


# ---------------------------------------------------------------------------
# Stufe 2: Teillast-Verifikation (nur Simulation, keine Optimierung)
# ---------------------------------------------------------------------------

def build_ua_inputs(
    scenario: PartLoadScenario,
    design_ua: Dict[str, float],
    design_flows: Dict[str, float],
    config: DesignPointConfig,
) -> "UAInputs":
    """Baut die AHT-UA-Modell-Inputs fuer einen Teillastpunkt.

    ANNAHME (siehe Moduldocstring): Desorber/Verdampfer/Kondensator als feste
    Massenstrom-Kwargs (m_13, m_15, m_17); NUR der Absorber (designbestimmende
    Apparategruppe, analog zum Verdampfer bei der Kältemaschine) ist über
    absorber_spec_mode="m11"/"T12" wählbar; cycle_scale_spec_mode="m6".
    shex_model="UA" mit explizitem UA_shex, damit der in Stufe 1 optimierte
    SHEX-UA-Wert nicht ignoriert wird (NICHT "NTU"/Effectiveness_shex).
    """
    return UAInputs(
        T_11_C=scenario.T_11_C,
        T_13_C=scenario.T_13_C,
        T_15_C=scenario.T_15_C,
        T_17_C=scenario.T_17_C,
        UA_shex=design_ua["UA_shex"],
        UA_des=design_ua["UA_des"],
        UA_cond=design_ua["UA_cond"],
        UA_evap=design_ua["UA_evap"],
        UA_abs=design_ua["UA_abs"],
        m_13=design_flows["m13_kg_s"],
        m_15=design_flows["m15_kg_s"],
        m_17=design_flows["m17_kg_s"],
        desorber_evaporator_routing_mode=config.desorber_evaporator_routing_mode,
        cycle_scale_spec_mode="m6",
        m6_spec=design_flows["m6_kg_s"],
        absorber_spec_mode="m11",
        m11_spec=design_flows["m11_kg_s"],
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
        shex_model="UA",
    )


def verify_part_load(
    scenarios: List[PartLoadScenario],
    design_result,
    config: DesignPointConfig,
) -> None:
    if not UA_MODEL_AVAILABLE:
        print(
            "AHT-UA-Modell konnte nicht importiert werden "
            f"({_ua_import_error!r}). Stufe 2 wird übersprungen.\n"
            "  -> Falls das Modul anders heißt/liegt: Pfad/Klassennamen im Skript anpassen,\n"
            "     oder mir das Main-Skript/den Quelltext schicken."
        )
        return

    if not _is_feasible(design_result) or not design_result.UA_conversion:
        print(
            "Stufe 2 übersprungen: der Designpunkt aus Stufe 1 ist nicht valide "
            "(kein UA_conversion vorhanden) -- siehe Warnung von "
            "print_design_point_summary()."
        )
        return

    design_ua = {
        k: design_result.UA_conversion[k]
        for k in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"]
    }
    design_flows = {
        "m11_kg_s": design_result.diagnostics["m11_kg_s"],
        "m13_kg_s": design_result.diagnostics["m13_kg_s"],
        "m15_kg_s": design_result.diagnostics["m15_kg_s"],
        "m17_kg_s": design_result.diagnostics["m17_kg_s"],
        "m6_kg_s": design_result.diagnostics["m6_kg_s"],
    }

    print("=" * 90)
    print("Stufe 2: Teillast-Verifikation")
    print("=" * 90)

    cache = WarmStartCache()

    for scenario in scenarios:
        t0 = time.perf_counter()
        try:
            inputs = build_ua_inputs(scenario, design_ua, design_flows, config)
        except TypeError as exc:
            print(
                f"[{scenario.name}] Konnte UAInputs nicht erzeugen -- "
                f"Parameter-Mismatch mit AHT_UA_LMTD: {exc}\n"
                "  -> Bitte AHT_UA_LMTD.py-Quelltext/Main-Skript schicken, dann"
                " passe ich build_ua_inputs() an."
            )
            continue

        x0 = cache.get(inputs)

        try:
            result = solve_ua(inputs, x0=x0)
        except Exception as exc:
            print(f"[{scenario.name}] Fehler beim Lösen: {exc}")
            continue

        try:
            feasible = _is_feasible(result)
        except AttributeError as exc:
            print(
                f"[{scenario.name}] Ergebnisstruktur des AHT-UA-Modells weicht ab: {exc}\n"
                "  -> Bitte AHT_UA_LMTD.py-Quelltext schicken zum Abgleich."
            )
            continue

        dt = time.perf_counter() - t0
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"\n[{scenario.name}] Status: {status}  ({dt:.2f} s)")

        if feasible:
            cache.update(np.array(list(result.primary_variables.values()), dtype=float))
            print(f"  Q_abs = {result.heat_flows_kW.get('Q_abs', float('nan')):.3f} kW")
            for k, v in result.kpis.items():
                try:
                    print(f"  {k:6s}= {float(v):.4f}")
                except (TypeError, ValueError):
                    pass
        else:
            print(f"  Solver-Nachricht: {result.solve_info.message}")
            if result.checks:
                verletzte = [k for k, v in result.checks.items() if not v]
                print(f"  Verletzte Checks: {verletzte}")

    print("=" * 90)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = DesignPointConfig()

    t_total_start = time.perf_counter()

    if config.run_feasibility_probe:
        quick_feasibility_probe(config)
        print(
            "\nFeasibility-Test abgeschlossen. Falls oben (fast) alles INFEASIBLE war,"
            " Bounds/Randbedingungen/x0_override anpassen, bevor die volle Optimierung"
            " gestartet wird (siehe Chat-Erklärung)."
        )

    theta_opt, design_result, stats = optimize_design_point(config)
    print_design_point_summary(theta_opt, design_result)

    if config.run_stage2:
        t0 = time.perf_counter()
        verify_part_load(DEFAULT_SCENARIOS, design_result, config)
        t_stage2 = time.perf_counter() - t0
        print(f"\nStufe 2 Dauer : {t_stage2:.1f} s")
    else:
        print("\nStufe 2 übersprungen (config.run_stage2 = False).")

    t_total = time.perf_counter() - t_total_start
    print(f"Gesamtdauer   : {t_total/60:.2f} min")