"""Designpoint-Optimierung für den Absorptionswärmetransformator (AWT/AHT).

Analog zum Designpoint-Optimierer der Kältemaschine (AC_design_point_optimizer.py),
aber für Models.AHT_Pinch_Point statt Models.AC_Pinch_Point.

WICHTIGER UNTERSCHIED zur Kältemaschine -- vertauschte Rollen der Apparate:
  - Absorber liefert die NUTZWÄRME (hohes Temperaturniveau, "Produkt" des AWT).
    Spec-Variablen: absorber_spec_mode="T12", T12_spec_C (Nutzwärmesenke,
    externe Eintritts-/Austrittstemperatur T_11_C -> T12_spec_C).
  - Desorber UND Verdampfer werden BEIDE von der externen Abwärmequelle
    gespeist (mittleres Temperaturniveau) -- daher
    "desorber_evaporator_routing_mode" (parallel / seriell), analog zur
    "absorber_condenser_routing_mode" bei der Kältemaschine, nur mit
    vertauschten Apparaten.
  - Kondensator rückkühlt auf niedrigem Temperaturniveau (T_17_C -> T18_spec_C).
  - cycle_scale_spec_mode="Qabs" mit Qabs_spec_kW: die Design-Nutzwärmeleistung
    (Analog zu Qevap_spec_kW bei der Kältemaschine) wird am ABSORBER vorgegeben,
    nicht am Verdampfer.

ANNAHMEN (siehe Chat-Nachricht) -- unbedingt beim ersten Testlauf prüfen:
  1. AHT_Pinch_Point stellt bounds(inputs) und initial_guess(inputs) mit
     identischer Signatur wie AC_Pinch_Point bereit.
  2. Die Ergebnisstruktur (solve_info, checks, UA_conversion, diagnostics,
     states, kpis) ist analog zu AC_Pinch_Point/AWTResult aufgebaut.
  3. Der interne, gepumpte Lösungsmassenstrom heißt vermutlich "m6"
     (passend zu cycle_scale_spec_mode="m6") -- wird defensiv gesucht,
     nicht hart vorausgesetzt (siehe _get_internal_solution_flow()).
  4. UA_conversion enthält dieselben Schlüssel wie bei der Kältemaschine:
     UA_shex, UA_des, UA_cond, UA_evap, UA_abs.

Stufe 2 (Teillast-Verifikation mit einem AHT-UA-Modell) ist hier noch NICHT
enthalten -- sobald ein AHT_UA_LMTD.py-Äquivalent existiert, kann das analog
zur Kältemaschine ergänzt werden.
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

from Models.AHT_Pinch_Point import (
    AWTInputs,
    bounds as awt_bounds,
    initial_guess as awt_initial_guess,
    solve_awt,
)


def _clip_to_bounds(z: np.ndarray, inputs: AWTInputs) -> np.ndarray:
    """Sicherheitsnetz: clippt einen Startvektor defensiv auf die Modell-Bounds
    (siehe gleichnamige Funktion / Bugfix-Historie im AKM-Optimierer)."""
    lower, upper = awt_bounds(inputs)
    eps = 1.0e-6
    return np.clip(z, lower + eps, upper - eps)


# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------

@dataclass
class DesignPointConfig:
    """Fixierte Randbedingungen des AWT-Designpunkts.

    Defaultwerte aus deinem main-Skript übernommen (T_11_C=135 etc.).
    """

    # Externe Eintrittstemperaturen [°C]
    T_11_C: float = 135.0   # Nutzwärmesenke (Absorber), kalter Eintritt
    T_13_C: float = 120.0   # Abwärmequelle (Desorber/Verdampfer, routing-abhängig)
    T_15_C: float = 120.0   # Abwärmequelle (Desorber/Verdampfer, routing-abhängig)
    T_17_C: float = 30.0    # Rückkühlung (Kondensator), kalter Eintritt

    # Externe Austrittstemperatur-Spezifikationen [°C]
    T12_spec_C: float = 146.02   # Nutzwärmesenke, Austritt (Absorber)
    T14_spec_C: float = 108.92   # Abwärmequelle, Austritt (Desorber)
    T16_spec_C: float = 108.80   # Abwärmequelle, Austritt (Verdampfer)
    T18_spec_C: float = 41.26    # Rückkühlung, Austritt (Kondensator)

    # Design-Nutzwärmeleistung [kW] (am Absorber, NICHT am Verdampfer!)
    Qabs_spec_kW: float = 184.4

    desorber_evaporator_routing_mode: str = "parallel"
    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # Untere Schranken der Pinch-Temperaturdifferenzen [K].
    # Bewusst generisch niedrig (3 K) gelassen -- dein main-Skript zeigt, dass
    # Kondensator/Verdampfer/Absorber beim AWT typischerweise deutlich größere
    # Pinches brauchen als bei der Kältemaschine (Beispielwerte bis ~25 K).
    # Der Optimierer soll das selbst finden, nicht durch eine zu enge Floor
    # künstlich verzerrt werden.
    dT_floor: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 3.0,
            "des": 3.0,
            "cond": 3.0,
            "evap": 3.0,
            "abs": 3.0,
        }
    )
    # Obere Suchgrenze = floor + dT_search_range. Größer als beim AKM-Optimierer
    # (20 K), weil deine Beispielwerte für cond/evap/abs schon bis ~25 K reichen.
    dT_search_range: float = 30.0

    # Gewichtung der UA-Werte in der Zielfunktion (Standard: alle gleich, ΣUA)
    ua_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "shex": 1.0, "des": 1.0, "cond": 1.0, "evap": 1.0, "abs": 1.0,
        }
    )

    # Solver-Einstellungen während der Optimierungsphase (siehe Erklärung im
    # AKM-Optimierer-Chat: gelockert für Stufe 1a/1b, strenger Defaultwert für
    # den finalen Präzisions-Solve). Werte übernommen aus dem validierten,
    # NICHT übermäßig aggressiven Setup der Kältemaschine.
    opt_solver_tol: float = 1.0e-6
    opt_max_nfev: int = 150

    # DE-Tuning
    de_popsize: int = 12
    de_maxiter: int = 60
    de_workers: int = 1

    # Early Stopping -- konservativ eingestellt (siehe Lernerfahrung aus dem
    # AKM-Optimierer: zu kurze patience/zu hohe min_improvement kann echte,
    # spätere Verbesserungen abschneiden).
    de_patience: Optional[int] = 25
    de_min_improvement: float = 0.01  # kW/K

    # Penalty für nicht-konvergente/unplausible Punkte -- bewusst moderat
    # skaliert (nicht 1e4), damit DE's eigenes tol-Konvergenzkriterium nicht
    # von einem einzigen Ausreißer dominiert wird.
    penalty_base: float = 200.0
    penalty_residual_weight: float = 20.0

    # Konvergenzplot
    make_convergence_plot: bool = True
    convergence_plot_path: str = "stage1_convergence_AHT.png"


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
    """Hält den letzten konvergierten Primärvektor für Warmstarts."""

    def __init__(self) -> None:
        self.z: Optional[np.ndarray] = None

    def get(self, inputs: AWTInputs) -> np.ndarray:
        z = self.z if self.z is not None else awt_initial_guess(inputs)
        return _clip_to_bounds(z, inputs)

    def update(self, z: np.ndarray) -> None:
        self.z = np.asarray(z, dtype=float).copy()

    def reset(self) -> None:
        self.z = None


def build_awt_inputs(
    theta: np.ndarray, config: DesignPointConfig, *, fast: bool
) -> AWTInputs:
    """Baut die AWT-Inputs. fast=True -> gelockerte Solver-Toleranzen (Stufe 1)."""
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
    return AWTInputs(**kwargs)


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
    inputs = build_awt_inputs(theta, config, fast=True)
    x0 = cache.get(inputs)

    t0 = time.perf_counter()
    result = solve_awt(inputs, x0=x0)
    dt = time.perf_counter() - t0

    feasible = _is_feasible(result)
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
    ax.set_title("AWT -- Stufe 1: Konvergenzverlauf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Konvergenzplot gespeichert: {path}")


def optimize_design_point(
    config: DesignPointConfig, *, seed: int = 42, verbose: bool = True
) -> tuple[np.ndarray, "object", EvalStats]:
    cache = WarmStartCache()
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
        print("Finaler Präzisions-Solve (strenge Toleranzen) ...")

    t0 = time.perf_counter()
    inputs_opt = build_awt_inputs(theta_opt, config, fast=False)
    result_opt = solve_awt(inputs_opt, x0=cache.get(inputs_opt))
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
            print("  WARNUNG: >10% der Aufrufe erreichen das nfev-Limit -- opt_max_nfev prüfen.")

    if config.make_convergence_plot:
        plot_convergence(stats, config.convergence_plot_path)

    return theta_opt, result_opt, stats

def print_design_point_summary(theta: np.ndarray, result) -> None:
    print("=" * 90)
    print("AWT Designpoint-Optimierung -- Ergebnis")
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
    print("Externe Massenströme am Designpunkt [kg/s] (ANNAHME zu Feldnamen, ggf. prüfen)")
    labels = {
        "m11_kg_s": "Absorber (Nutzwärme)",
        "m13_kg_s": "Desorber (Quelle)",
        "m15_kg_s": "Verdampfer (Quelle)",
        "m17_kg_s": "Kondensator (Rückkühlung)",
    }
    for key, label in labels.items():
        if key in result.diagnostics:
            print(f"  {key:10s} [{label:24s}]: {result.diagnostics[key]:10.6f}")
        else:
            print(f"  {key:10s}: nicht in diagnostics gefunden")
    m6 = result.diagnostics["m6_kg_s"]
    if m6 is not None:
        print(f"  {'m6':10s} [interner Lösungsstrom  ]: {m6:10.6f}")
    else:
        print("  Interner Lösungsmassenstrom nicht gefunden -- bitte Feldnamen im")
        print("  AHT_Pinch_Point-Quelltext prüfen und mir mitteilen.")
    print()
    print("KPIs")
    for k, v in result.kpis.items():
        try:
            print(f"  {k:10s}: {float(v):10.4f}")
        except (TypeError, ValueError):
            print(f"  {k:10s}: {v}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = DesignPointConfig()

    t_total_start = time.perf_counter()
    theta_opt, design_result, stats = optimize_design_point(config)
    print_design_point_summary(theta_opt, design_result)
    t_total = time.perf_counter() - t_total_start

    print()
    print(f"Gesamtdauer: {t_total/60:.2f} min")