"""Bilevel-Optimierung: Designpoint-UA-Werte + Teillast-Verifikation.

Stufe 1 (Designpoint-Optimierung, Pinch-Point-Modell)
------------------------------------------------------
Variablen   : theta = [dT_min_shex, dT_min_des, dT_min_cond, dT_min_evap, dT_min_abs]
Bounds      : dT_min_i >= floor_i   (untere Schranke je Wärmeübertrager)
Fixiert     : Qevap_spec_kW, T_11_C, T_13_C/T_15_C, T_17_C sowie die
              Austrittstemperatur-Spezifikationen (T12/T14/T16/T18_spec_C)
Ziel        : min Sum(UA_i)  (SHEX, Desorber, Kondensator, Verdampfer, Absorber)
Penalty     : falls solve_awt() nicht konvergiert oder Plausibilitätschecks
              (Kristallisation, Konzentrationshierarchie, Massenstrom > 0) verletzt sind

Optimierer  : differential_evolution (global, robust gegen Nichtkonvergenz-Sprünge)
              + Nelder-Mead-Politur (lokal, mit Bounds)
Warmstart   : letzter konvergierter Primärvektor wird als x0 der nächsten
              Auswertung übergeben; Fallback auf initial_guess(inputs)

WICHTIG -- Solver-Einstellungen während der Optimierung vs. finaler Solve:
  Während Stufe 1a/1b läuft der innere Solver mit gelockerten Einstellungen
  (config.opt_solver_tol / config.opt_max_nfev), weil hier nur Konvergenz und
  Feasibility geprüft werden müssen, nicht maximale Präzision. Der GEFUNDENE
  Optimalpunkt wird danach EINMAL mit den strengen Modell-Defaults
  (solver_tol=1e-9, max_nfev=5000) nachgerechnet, um die finalen UA-Werte
  präzise zu bestimmen. So bleibt die Ergebnisgüte am Ende erhalten, ohne dass
  jede der tausenden Zwischenauswertungen die volle Präzision braucht.

Stufe 2 (Teillast-Verifikation, UA-Modell)
--------------------------------------------
Die in Stufe 1 bestimmten UA-Werte UND die am Designpunkt anfallenden externen
Massenströme (m11, m13, m15, m17, m1) werden eingefroren. Für definierte
Randbedingungs-Szenarien (z. B. Extremfall Winter/Sommer) wird NUR simuliert,
NICHT erneut optimiert -- "design at critical point, verify elsewhere"
(Flexibility-Analysis-Ansatz, vgl. Swaney & Grossmann 1985 / Halemane &
Grossmann 1983).

ANNAHMEN für Stufe 2 (siehe Chat-Verlauf):
  1. AC_UA_LMTD liefert dieselbe Ergebnisstruktur wie AC_Pinch_Point.
  2. m1 wird im Teillastfall konstant vom Designpunkt übernommen
     (cycle_scale_spec_mode="m1"), d. h. drehzahlfeste Lösungspumpe.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
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

from Models.AC_Pinch_Point import (
    AKMInputs as PinchInputs,
    AWTResult as PinchResult,
    bounds as pinch_bounds,
    initial_guess as pinch_initial_guess,
    solve_awt as solve_pinch,
)

try:
    from Models.AC_UA_LMTD import (
        AKMInputs as UAInputs,
        solve_awt as solve_ua,
    )
    UA_MODEL_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    UA_MODEL_AVAILABLE = False
    _ua_import_error = exc


def _clip_to_bounds(z: np.ndarray, inputs: PinchInputs) -> np.ndarray:
    """Sicherheitsnetz: clippt einen Startvektor defensiv auf die Modell-Bounds.

    Schützt vor einem harten Absturz von least_squares (ValueError: 'Initial
    guess is outside of provided bounds'), falls initial_guess() oder ein
    Warmstart-Wert knapp außerhalb der zulässigen Grenzen liegt.
    """
    lower, upper = pinch_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------

@dataclass
class DesignPointConfig:
    """Fixierte Randbedingungen des Designpunkts (Stufe 1)."""

    # Externe Eintrittstemperaturen [°C]
    T_11_C: float = 90.0
    T_13_C: float = 25.0
    T_15_C: float = 25.0
    T_17_C: float = 11.0

    # Externe Austrittstemperatur-Spezifikationen [°C]
    T12_spec_C: float = 72.0
    T14_spec_C: float = 32.0
    T16_spec_C: float = 32.0
    T18_spec_C: float = 5.0

    # Design-Verdampferleistung [kW]
    Qevap_spec_kW: float = 40.9

    absorber_condenser_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # Untere Schranken der Pinch-Temperaturdifferenzen [K]
    dT_floor: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 3.0,
            "des": 3.0,
            "cond": 3.0,
            "evap": 3.0,
            "abs": 5.0,
        }
    )
    # Obere Suchgrenze = floor + dT_search_range
    dT_search_range: float = 20.0

    # Gewichtung der UA-Werte in der Zielfunktion (Standard: alle gleich, ΣUA)
    ua_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 1.0, "des": 1.0, "cond": 1.0, "evap": 1.0, "abs": 1.0,
        }
    )

    # Günstigere Solver-Einstellungen NUR für die Optimierungsphase (Stufe 1a/1b).
    # Der finale Punkt wird danach mit den strengen Modell-Defaults nachgerechnet.
    # Empfehlung: opt_max_nfev nicht zu aggressiv kappen (siehe Chat-Erklärung) --
    # 100-150 ist idR ein guter Kompromiss aus Geschwindigkeit und Verlässlichkeit,
    # solange der Warmstart greift. Diagnostik am Ende von optimize_design_point()
    # zeigt an, wie oft das Limit tatsächlich erreicht wurde.
    opt_solver_tol: float = 1.0e-6
    opt_max_nfev: int = 150

    # DE-Tuning
    de_popsize: int = 12
    de_maxiter: int = 60
    de_workers: int = 1  # >1 = Multiprocessing, siehe Hinweise im Chat/Docstring

    # Konvergenzplot (Stufe 1)
    make_convergence_plot: bool = True
    convergence_plot_path: str = "stage1_convergence.png"


@dataclass
class PartLoadScenario:
    name: str
    T_11_C: float
    T_13_C: float
    T_15_C: float
    T_17_C: float


DEFAULT_SCENARIOS: List[PartLoadScenario] = [
    PartLoadScenario(name="Extremfall_kalt", T_11_C=80.0, T_13_C=20.0, T_15_C=20.0, T_17_C=8.0),
    PartLoadScenario(name="Extremfall_warm", T_11_C=90.0, T_13_C=30.0, T_15_C=30.0, T_17_C=14.0),
]


# ---------------------------------------------------------------------------
# Stufe 1: Designpoint-Optimierung
# ---------------------------------------------------------------------------

THETA_ORDER = ["shex", "des", "cond", "evap", "abs"]


@dataclass
class EvalStats:
    """Sammelt Diagnostik über alle Zielfunktionsaufrufe hinweg.

    HINWEIS zu Multiprocessing (de_workers > 1): Jeder Worker-Prozess hat
    eigenen Speicher. Diese Statistik (und der WarmStartCache) werden dann
    NICHT mehr zuverlässig über alle Prozesse hinweg aufsummiert bzw. geteilt
    -- die Zahlen/der Plot spiegeln dann nur einen Teil der Auswertungen wider,
    und der Warmstart-Vorteil geht pro Prozess größtenteils verloren. Für
    korrekte, vollständige Diagnostik/Plots: de_workers=1 verwenden.
    """

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
    """Fortschrittsanzeige für differential_evolution (eine Zeile pro Generation)."""

    def __init__(self, maxiter: int, stats: EvalStats):
        self.stats = stats
        self.maxiter = maxiter
        self.gen = 0
        self.pbar = tqdm(total=maxiter, desc="Stufe 1a", unit="Gen") if TQDM_AVAILABLE else None

    def __call__(self, xk, convergence) -> bool:
        self.gen += 1
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix(
                conv=f"{convergence:.2e}",
                calls=self.stats.calls,
                avg_s=f"{self.stats.avg_solve_time_s:.2f}",
            )
        else:
            print(
                f"  Gen {self.gen}/{self.maxiter} | convergence={convergence:.2e} "
                f"| Aufrufe={self.stats.calls} | avg={self.stats.avg_solve_time_s:.2f}s"
            )
        return False

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class WarmStartCache:
    """Hält den letzten konvergierten Primärvektor für Warmstarts."""

    def __init__(self) -> None:
        self.z: Optional[np.ndarray] = None

    def get(self, inputs: PinchInputs) -> np.ndarray:
        z = self.z if self.z is not None else pinch_initial_guess(inputs)
        return _clip_to_bounds(z, inputs)

    def update(self, z: np.ndarray) -> None:
        self.z = np.asarray(z, dtype=float).copy()

    def reset(self) -> None:
        self.z = None


def build_pinch_inputs(
    theta: np.ndarray, config: DesignPointConfig, *, fast: bool
) -> PinchInputs:
    """Baut die Pinch-Point-Inputs.

    fast=True  : gelockerte Solver-Toleranzen (Stufe 1a/1b, viele Auswertungen)
    fast=False : strenge Modell-Defaults (finaler Solve nach der Optimierung)
    """
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
        absorber_condenser_routing_mode=config.absorber_condenser_routing_mode,
        cycle_scale_spec_mode="Qeva",
        Qevap_spec_kW=config.Qevap_spec_kW,
        desorber_spec_mode="T12",
        T12_spec_C=config.T12_spec_C,
        absorber_spec_mode="T14",
        T14_spec_C=config.T14_spec_C,
        condenser_spec_mode="T16",
        T16_spec_C=config.T16_spec_C,
        evaporator_spec_mode="T18",
        T18_spec_C=config.T18_spec_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )
    if fast:
        kwargs["solver_tol"] = config.opt_solver_tol
        kwargs["max_nfev"] = config.opt_max_nfev
    return PinchInputs(**kwargs)


def _is_feasible(result: PinchResult) -> bool:
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
    inputs = build_pinch_inputs(theta, config, fast=True)
    x0 = cache.get(inputs)

    t0 = time.perf_counter()
    result = solve_pinch(inputs, x0=x0)
    dt = time.perf_counter() - t0

    feasible = _is_feasible(result)
    hit_cap = (
        not result.solve_info.success
        and result.solve_info.nfev >= config.opt_max_nfev
    )

    if not feasible:
        penalty = 1.0e4 + 100.0 * result.solve_info.scaled_residual_norm
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
    return [
        (config.dT_floor[key], config.dT_floor[key] + config.dT_search_range)
        for key in THETA_ORDER
    ]


def plot_convergence(stats: EvalStats, path: str) -> None:
    if not stats.history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib ist nicht installiert -- Konvergenzplot wird "
            "übersprungen (pip install matplotlib)."
        )
        return

    calls, costs = zip(*stats.history)
    costs = np.asarray(costs, dtype=float)
    running_best = np.minimum.accumulate(costs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(calls, costs, s=6, alpha=0.35, label="Einzelauswertung (inkl. Penalty)")
    ax.plot(calls, running_best, color="tab:red", linewidth=2, label="Bestes Sum(UA) bisher")
    ax.set_xlabel("Funktionsauswertung")
    ax.set_ylabel("Zielfunktionswert [kW/K] (Penalty ≈ 1e4)")
    ax.set_yscale("log")
    ax.set_title("Stufe 1: Konvergenzverlauf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Konvergenzplot gespeichert: {path}")


def optimize_design_point(
    config: DesignPointConfig, *, seed: int = 42, verbose: bool = True
) -> tuple[np.ndarray, PinchResult, EvalStats]:
    cache = WarmStartCache()
    stats = EvalStats()
    bounds = _theta_bounds(config)

    if config.de_workers != 1 and verbose:
        print(
            "Hinweis: de_workers != 1 -- Warmstart-Cache und Live-Diagnostik "
            "(Zähler/Plot) sind über mehrere Prozesse hinweg nicht zuverlässig "
            "vollständig. Für exakte Diagnostik/Plot de_workers=1 verwenden."
        )

    t_stage1_start = time.perf_counter()

    if verbose:
        print("Stufe 1a: globale Suche (differential_evolution) ...")

    progress = DEProgress(config.de_maxiter, stats)
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

    if verbose:
        print("Finaler Präzisions-Solve (strenge Toleranzen) ...")
    t0 = time.perf_counter()
    inputs_opt = build_pinch_inputs(theta_opt, config, fast=False)
    result_opt = solve_pinch(inputs_opt, x0=cache.get(inputs_opt))
    t_final = time.perf_counter() - t0

    if not _is_feasible(result_opt):
        raise RuntimeError(
            "Finaler Optimierungspunkt ist nicht feasible. Bitte Bounds/Randbedingungen prüfen."
        )

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
        if stats.calls and stats.hit_nfev_cap / stats.calls > 0.1:
            print(
                "  WARNUNG: >10% der Aufrufe erreichen das nfev-Limit -- "
                "opt_max_nfev evtl. erhöhen, um falsche Infeasibility-Klassifikation "
                "zu vermeiden."
            )

    if config.make_convergence_plot:
        plot_convergence(stats, config.convergence_plot_path)

    return theta_opt, result_opt, stats


def print_design_point_summary(theta: np.ndarray, result: PinchResult) -> None:
    print("=" * 90)
    print("Designpoint-Optimierung -- Ergebnis")
    print("=" * 90)
    for key, val in zip(THETA_ORDER, theta):
        print(f"  dT_min_{key:5s}: {val:8.4f} K")
    print()
    print("UA-Werte [kW/K]")
    for key in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"]:
        print(f"  {key:10s}: {result.UA_conversion[key]:10.4f}")
    total_ua = sum(result.UA_conversion[k] for k in ["UA_shex", "UA_des", "UA_cond", "UA_evap", "UA_abs"])
    print(f"  {'Sum(UA)':10s}: {total_ua:10.4f}")
    print()
    print("Externe Massenströme am Designpunkt [kg/s]")
    for key in ["m11_kg_s", "m13_kg_s", "m15_kg_s", "m17_kg_s"]:
        print(f"  {key:10s}: {result.diagnostics[key]:10.6f}")
    # m1 (interner Lösungsmassenstrom) steckt nicht in diagnostics, sondern im
    # Zustand "1" (Loesung nach Absorber, vor Pumpe).
    m1 = result.states["1"]["m_kg_s"]
    print(f"  {'m1':10s}: {m1:10.6f}   (interner Lösungsmassenstrom, für Stufe 2 relevant)")
    print()
    print(f"COP: {result.kpis['COP']:.4f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Stufe 2: Teillast-Verifikation (nur Simulation, keine Optimierung)
# ---------------------------------------------------------------------------

def build_ua_inputs(
    scenario: PartLoadScenario,
    design_ua: Dict[str, float],
    design_flows: Dict[str, float],
    config: DesignPointConfig,
) -> "UAInputs":
    """Baut die UA-Modell-Inputs fuer einen Teillastpunkt.

    shex_model="UA" mit explizitem UA_shex (NICHT "NTU"/Effectiveness_shex),
    weil sonst der in Stufe 1 optimierte SHEX-UA-Wert ignoriert würde.
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
        m_11=design_flows["m11_kg_s"],
        m_13=design_flows["m13_kg_s"],
        m_15=design_flows["m15_kg_s"],
        absorber_condenser_routing_mode=config.absorber_condenser_routing_mode,
        cycle_scale_spec_mode="m1",
        m1_spec=design_flows["m1_kg_s"],
        evaporator_spec_mode="m17",
        m17_spec=design_flows["m17_kg_s"],
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
        shex_model="UA",
    )


def verify_part_load(
    scenarios: List[PartLoadScenario],
    design_result: PinchResult,
    config: DesignPointConfig,
) -> None:
    if not UA_MODEL_AVAILABLE:
        print(
            "UA-Modell konnte nicht importiert werden "
            f"({_ua_import_error!r}). Stufe 2 wird übersprungen."
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
        "m1_kg_s": design_result.states["1"]["m_kg_s"],
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
                f"Parameter-Mismatch mit AC_UA_LMTD: {exc}\n"
                "  -> Bitte AC_UA_LMTD.py-Quelltext schicken, dann passe ich"
                " build_ua_inputs() an."
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
                f"[{scenario.name}] Ergebnisstruktur des UA-Modells weicht ab: {exc}\n"
                "  -> Bitte AC_UA_LMTD.py-Quelltext schicken zum Abgleich."
            )
            continue

        dt = time.perf_counter() - t0
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        print(f"\n[{scenario.name}] Status: {status}  ({dt:.2f} s)")

        if feasible:
            cache.update(np.array(list(result.primary_variables.values()), dtype=float))
            print(f"  Q_evap = {result.heat_flows_kW.get('Q_evap', float('nan')):.3f} kW")
            print(f"  COP    = {result.kpis.get('COP', float('nan')):.4f}")
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

    theta_opt, design_result, stats = optimize_design_point(config)
    print_design_point_summary(theta_opt, design_result)

    t0 = time.perf_counter()
    verify_part_load(DEFAULT_SCENARIOS, design_result, config)
    t_stage2 = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start
    print()
    print(f"Stufe 2 Dauer : {t_stage2:.1f} s")
    print(f"Gesamtdauer   : {t_total/60:.2f} min")