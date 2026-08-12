"""Methode der Charakteristischen Gleichungen für den AWT.

Diese Datei stellt die Schnittstelle zwischen der vorhandenen,
ausführlichen AWT-Simulation und dem vereinfachten Ansatz der
charakteristischen Gleichungen her.

Implementiert ist hier die herkömmliche Methode der charakteristischen
Gleichungen. Die dafür benötigten Parameter werden aus einem gültig
ausgewerteten ``AWTResult`` der Detailsimulation extrahiert.

Wesentliche Funktionen
----------------------
- ``import_simulation_values(result)``:
    Extrahiert alle für die charakteristischen Gleichungen benötigten Größen
    aus einem Simulationsresultat.
- ``solve_detailed_and_import(inputs, x0=None)``:
    Führt die Detailsimulation aus und liefert sowohl das Simulationsresultat
    als auch die importierten Größen für die charakteristischen Gleichungen.
- ``calculate_conventional_characteristic_equations(imported, ...)``:
    Berechnet die herkömmliche Methode der charakteristischen Gleichungen.
- ``run_conventional_from_detailed_simulation(inputs, x0=None, ...)``:
    Komfortfunktion für den kompletten Ablauf.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from simulation_JW import AWTInputs, AWTResult


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImportedSimulationValues:
    """Aus der Detailsimulation extrahierte Größen für die Char.-Gleichungen.

    Alle Temperaturen werden in Kelvin geführt, Enthalpien in kJ/kg,
    Wärmeströme in kW, UA in kW/K und Massenströme in kg/s.
    """

    cp_w: float

    # Externe Ein- und Auslasstemperaturen
    t_evap_in: float
    t_cond_in: float
    t_abs_in: float
    t_des_in: float
    t_evap_out: float
    t_cond_out: float
    t_abs_out: float
    t_des_out: float

    # Enthalpien
    h_ref_evap_out: float
    h_ref_evap_in: float
    h_ref_cond_in: float
    h_ref_cond_out: float
    h_ref_abs_in: float
    h_sol_abs_out: float
    h_ref_des_out: float
    h_sol_des_in: float
    h_sol_abs_in: float
    h_sol_des_out: float

    # Interner Lösungsmassenstrom
    m_dot_p: float

    # UA-Werte
    UA_evap: float
    UA_cond: float
    UA_abs: float
    UA_des: float
    UA_shex: float

    # Externe Massenströme
    m_ext_evap: float
    m_ext_cond: float
    m_ext_abs: float
    m_ext_des: float

    # Simulationswerte zum Vergleich
    Q_sim_evap: float
    Q_sim_cond: float
    Q_sim_abs: float
    Q_sim_des: float

    T_sim_evap_mean: float
    T_sim_cond_mean: float
    T_sim_abs_mean: float
    T_sim_des_mean: float

    def as_legacy_dict(self) -> dict[str, float]:
        """Liefert die ursprünglichen Variablennamen der angefangenen Datei.

        Dadurch können spätere Formelerweiterungen mit denselben Bezeichnern
        weitergeführt werden.
        """
        return asdict(self)


@dataclass(frozen=True)
class ConventionalCharacteristicResult:
    """Ergebnis der herkömmlichen Methode der charakteristischen Gleichungen."""

    B_duering: float
    R_abs: float
    R_des: float
    imported: ImportedSimulationValues

    NTU_evap: float
    NTU_cond: float
    NTU_abs: float
    NTU_des: float

    z_evap: float
    z_cond: float
    z_abs: float
    z_des: float

    t_evap: float
    t_cond: float
    t_abs: float
    t_des: float
    delta_delta_t: float

    K_evap: float
    K_abs: float
    K_des: float
    K_cond: float

    Ks_abs: float
    Ks_des: float

    s_cond: float
    s_evap: float
    s_abs: float
    s_des: float

    delta_delta_t_min_cond: float
    delta_delta_t_min_evap: float
    delta_delta_t_min_abs: float
    delta_delta_t_min_des: float

    Q_dot_cond: float
    Q_dot_evap: float
    Q_dot_abs: float
    Q_dot_des: float

    T_evap: float
    T_cond: float
    T_abs: float
    T_des: float

    Q_dot_cond_error_abs: float
    Q_dot_evap_error_abs: float
    Q_dot_abs_error_abs: float
    Q_dot_des_error_abs: float

    Q_dot_cond_error_rel: float
    Q_dot_evap_error_rel: float
    Q_dot_abs_error_rel: float
    Q_dot_des_error_rel: float

    T_evap_error_abs: float
    T_cond_error_abs: float
    T_abs_error_abs: float
    T_des_error_abs: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

class CharacteristicEquationError(RuntimeError):
    """Fehler in der Schnittstelle oder Berechnung der char. Gleichungen."""



def _require_state_value(result: Any, state_id: str, key: str) -> float:
    try:
        value = result.states[state_id][key]
    except KeyError as exc:
        raise CharacteristicEquationError(
            f"Im Simulationsresultat fehlt states['{state_id}']['{key}']."
        ) from exc
    return float(value)



def _require_heat_flow(result: Any, key: str) -> float:
    try:
        value = result.heat_flows_kW[key]
    except KeyError as exc:
        raise CharacteristicEquationError(
            f"Im Simulationsresultat fehlt heat_flows_kW['{key}']."
        ) from exc
    return float(value)



def _require_valid_result(result: Any) -> None:
    solve_info = getattr(result, "solve_info", None)
    if solve_info is None:
        raise CharacteristicEquationError(
            "Das übergebene Objekt besitzt kein Attribut 'solve_info' und ist daher "
            "kein gültiges AWT-Simulationsresultat."
        )

    final_point_evaluable = bool(getattr(solve_info, "final_point_evaluable", False))
    if not final_point_evaluable:
        final_error = getattr(solve_info, "final_evaluation_error", "unbekannt")
        raise CharacteristicEquationError(
            "Die Detailsimulation liefert keinen physikalisch auswertbaren Endpunkt. "
            f"Import für charakteristische Gleichungen nicht möglich. Grund: {final_error}"
        )

    if not getattr(result, "states", None):
        raise CharacteristicEquationError(
            "Das Simulationsresultat enthält keine Zustandsdaten in 'states'."
        )



def _safe_rel_error(estimate: float, reference: float) -> float:
    if abs(reference) <= 1.0e-12:
        return math.nan
    return (estimate - reference) / reference



def _safe_inverse(value: float, label: str) -> float:
    if abs(value) <= 1.0e-12:
        raise CharacteristicEquationError(f"{label} ist numerisch zu klein bzw. null.")
    return 1.0 / value



def _safe_capacity_rate(m_dot: float, cp: float, label: str) -> float:
    c_rate = m_dot * cp
    if c_rate <= 0.0:
        raise CharacteristicEquationError(
            f"Wärmekapazitätsstrom für {label} muss positiv sein, erhalten {c_rate:.6e}."
        )
    return c_rate



def _z_factor(argument: float) -> float:
    """Temperaturdifferenzfaktor in numerisch stabiler Form.

    Aus
        z = 2 * (exp(a) - 1) / (a * (exp(a) + 1))
    wird stabil geschrieben als
        z = 2 * tanh(a / 2) / a

    Für a -> 0 gilt der Grenzwert z -> 1.
    """
    if abs(argument) <= 1.0e-12:
        return 1.0
    return 2.0 * math.tanh(argument / 2.0) / argument



def _mean_temperature(T_in: float, T_out: float) -> float:
    return 0.5 * (float(T_in) + float(T_out))


# ---------------------------------------------------------------------------
# Schnittstelle zur Detailsimulation
# ---------------------------------------------------------------------------


def import_simulation_values(result: "AWTResult") -> ImportedSimulationValues:
    """Extrahiert alle benötigten Größen aus der Detailsimulation.

    Erwartet ein erfolgreich ausgewertetes ``AWTResult`` aus ``simulation_JW``.
    """
    _require_valid_result(result)

    inputs = result.inputs

    # Externe Temperaturen
    t_evap_in = _require_state_value(result, "15", "T_K")
    t_cond_in = _require_state_value(result, "17", "T_K")
    t_abs_in = _require_state_value(result, "11", "T_K")
    t_des_in = _require_state_value(result, "13", "T_K")

    t_evap_out = _require_state_value(result, "16", "T_K")
    t_cond_out = _require_state_value(result, "18", "T_K")
    t_abs_out = _require_state_value(result, "12", "T_K")
    t_des_out = _require_state_value(result, "14", "T_K")

    # Zustandsenthalpien
    h1 = _require_state_value(result, "1", "h_kJ_kg")
    h3 = _require_state_value(result, "3", "h_kJ_kg")
    h4 = _require_state_value(result, "4", "h_kJ_kg")
    h6 = _require_state_value(result, "6", "h_kJ_kg")
    h7 = _require_state_value(result, "7", "h_kJ_kg")
    h8 = _require_state_value(result, "8", "h_kJ_kg")
    h9 = _require_state_value(result, "9", "h_kJ_kg")
    h10 = _require_state_value(result, "10", "h_kJ_kg")

    # Massenströme
    m6 = _require_state_value(result, "6", "m_kg_s")
    m11 = _require_state_value(result, "11", "m_kg_s")
    m13 = _require_state_value(result, "13", "m_kg_s")
    m15 = _require_state_value(result, "15", "m_kg_s")
    m17 = _require_state_value(result, "17", "m_kg_s")

    # Vergleichswerte der Detailsimulation
    Q_sim_evap = _require_heat_flow(result, "Q_evap")
    Q_sim_cond = _require_heat_flow(result, "Q_cond")
    Q_sim_abs = _require_heat_flow(result, "Q_abs")
    Q_sim_des = _require_heat_flow(result, "Q_des")

    # Interne mittlere Temperaturen der Detailsimulation für groben Vergleich
    T_sim_evap_mean = _require_state_value(result, "10", "T_K")
    T_sim_cond_mean = _require_state_value(result, "8", "T_K")
    T_sim_abs_mean = _mean_temperature(
        _require_state_value(result, "20", "T_K"),
        _require_state_value(result, "3", "T_K"),
    )
    T_sim_des_mean = _mean_temperature(
        _require_state_value(result, "1", "T_K"),
        _require_state_value(result, "6", "T_K"),
    )

    return ImportedSimulationValues(
        cp_w=float(inputs.cp_w_kJkgK),
        t_evap_in=t_evap_in,
        t_cond_in=t_cond_in,
        t_abs_in=t_abs_in,
        t_des_in=t_des_in,
        t_evap_out=t_evap_out,
        t_cond_out=t_cond_out,
        t_abs_out=t_abs_out,
        t_des_out=t_des_out,
        h_ref_evap_out=h10,
        h_ref_evap_in=h9,
        h_ref_cond_in=h7,
        h_ref_cond_out=h8,
        h_ref_abs_in=h10,
        h_sol_abs_out=h3,
        h_ref_des_out=h7,
        h_sol_des_in=h1,
        h_sol_abs_in=h4,
        h_sol_des_out=h6,
        m_dot_p=m6,
        UA_evap=float(inputs.UA_evap),
        UA_cond=float(inputs.UA_cond),
        UA_abs=float(inputs.UA_abs),
        UA_des=float(inputs.UA_des),
        UA_shex=float(inputs.UA_shex),
        m_ext_evap=m15,
        m_ext_cond=m17,
        m_ext_abs=m11,
        m_ext_des=m13,
        Q_sim_evap=Q_sim_evap,
        Q_sim_cond=Q_sim_cond,
        Q_sim_abs=Q_sim_abs,
        Q_sim_des=Q_sim_des,
        T_sim_evap_mean=T_sim_evap_mean,
        T_sim_cond_mean=T_sim_cond_mean,
        T_sim_abs_mean=T_sim_abs_mean,
        T_sim_des_mean=T_sim_des_mean,
    )



def solve_detailed_and_import(
    inputs: "AWTInputs",
    x0: "np.ndarray | None" = None,
) -> tuple["AWTResult", ImportedSimulationValues]:
    """Führt die Detailsimulation aus und importiert die benötigten Größen."""
    from simulation_JW import initial_guess, solve_awt

    if x0 is None:
        x0 = initial_guess(inputs)

    result = solve_awt(inputs, x0=x0)
    imported = import_simulation_values(result)
    return result, imported


# ---------------------------------------------------------------------------
# Herkömmliche Methode der charakteristischen Gleichungen
# ---------------------------------------------------------------------------


def calculate_conventional_characteristic_equations(
    imported: ImportedSimulationValues,
    *,
    B_duering: float = 1.15,
    R_abs: float = 0.0,
    R_des: float = 0.0,
) -> ConventionalCharacteristicResult:
    """Berechnet die herkömmliche Methode der charakteristischen Gleichungen."""
    cp_w = float(imported.cp_w)

    # Externe mittlere Temperaturen
    t_evap = _mean_temperature(imported.t_evap_in, imported.t_evap_out)
    t_cond = _mean_temperature(imported.t_cond_in, imported.t_cond_out)
    t_abs = _mean_temperature(imported.t_abs_in, imported.t_abs_out)
    t_des = _mean_temperature(imported.t_des_in, imported.t_des_out)

    # Temperaturdifferenzfaktoren
    NTU_evap = imported.UA_evap / _safe_capacity_rate(imported.m_ext_evap, cp_w, "Verdampfer")
    NTU_cond = imported.UA_cond / _safe_capacity_rate(imported.m_ext_cond, cp_w, "Kondensator")
    NTU_abs = imported.UA_abs / _safe_capacity_rate(imported.m_ext_abs, cp_w, "Absorber")
    NTU_des = imported.UA_des / _safe_capacity_rate(imported.m_ext_des, cp_w, "Desorber")

    z_evap = _z_factor(NTU_evap)
    z_cond = _z_factor(NTU_cond)
    z_abs = _z_factor((R_abs - 1.0) * NTU_abs)
    z_des = _z_factor((R_des - 1.0) * NTU_des)

    # Charakteristische Temperaturdifferenz
    delta_delta_t = B_duering * (t_evap - t_cond) - (t_abs - t_des)

    # Enthalpiekoeffizienten
    delta_h_ref_cond = imported.h_ref_cond_in - imported.h_ref_cond_out
    inv_delta_h_ref_cond = _safe_inverse(delta_h_ref_cond, "h_ref_cond_in - h_ref_cond_out")

    K_evap = (imported.h_ref_evap_out - imported.h_ref_evap_in) * inv_delta_h_ref_cond
    K_abs = (imported.h_ref_abs_in - imported.h_sol_abs_out) * inv_delta_h_ref_cond
    K_des = (imported.h_ref_des_out - imported.h_sol_des_in) * inv_delta_h_ref_cond
    K_cond = 1.0

    # Verlustkoeffizienten
    Ks_abs = imported.m_dot_p * (imported.h_sol_abs_in - imported.h_sol_abs_out)
    Ks_des = imported.m_dot_p * (imported.h_sol_des_out - imported.h_sol_des_in)

    # Apparateparameter
    term_cond = (
        B_duering
        * (
            K_evap / (imported.UA_evap * z_evap)
            + K_cond / (imported.UA_cond * z_cond)
        )
        + (
            K_abs / (imported.UA_abs * z_abs)
            + K_des / (imported.UA_des * z_des)
        )
    )
    s_cond = _safe_inverse(term_cond, "Steigungsparameter-Nenner des Kondensators")
    s_evap = K_evap * s_cond
    s_abs = K_abs * s_cond
    s_des = K_des * s_cond

    delta_delta_t_min_cond = Ks_abs / (imported.UA_abs * z_abs) + Ks_des / (imported.UA_des * z_des)
    delta_delta_t_min_evap = delta_delta_t_min_cond
    delta_delta_t_min_abs = delta_delta_t_min_cond - Ks_abs / (K_abs * s_cond)
    delta_delta_t_min_des = delta_delta_t_min_cond - Ks_des / (K_des * s_cond)

    # Charakteristische Gleichungen
    Q_dot_cond = s_cond * (delta_delta_t - delta_delta_t_min_cond)
    Q_dot_evap = s_evap * (delta_delta_t - delta_delta_t_min_evap)
    Q_dot_abs = s_abs * (delta_delta_t - delta_delta_t_min_abs)
    Q_dot_des = s_des * (delta_delta_t - delta_delta_t_min_des)

    # Interne mittlere Temperaturen
    T_evap = t_evap - K_evap * Q_dot_cond / (imported.UA_evap * z_evap)
    T_cond = t_cond + Q_dot_cond / (imported.UA_cond * z_cond)
    T_abs = t_abs + (K_abs * Q_dot_cond + Ks_abs) / (imported.UA_abs * z_abs)
    T_des = t_des - (K_des * Q_dot_cond + Ks_des) / (imported.UA_des * z_des)

    return ConventionalCharacteristicResult(
        B_duering=float(B_duering),
        R_abs=float(R_abs),
        R_des=float(R_des),
        imported=imported,
        NTU_evap=NTU_evap,
        NTU_cond=NTU_cond,
        NTU_abs=NTU_abs,
        NTU_des=NTU_des,
        z_evap=z_evap,
        z_cond=z_cond,
        z_abs=z_abs,
        z_des=z_des,
        t_evap=t_evap,
        t_cond=t_cond,
        t_abs=t_abs,
        t_des=t_des,
        delta_delta_t=delta_delta_t,
        K_evap=K_evap,
        K_abs=K_abs,
        K_des=K_des,
        K_cond=K_cond,
        Ks_abs=Ks_abs,
        Ks_des=Ks_des,
        s_cond=s_cond,
        s_evap=s_evap,
        s_abs=s_abs,
        s_des=s_des,
        delta_delta_t_min_cond=delta_delta_t_min_cond,
        delta_delta_t_min_evap=delta_delta_t_min_evap,
        delta_delta_t_min_abs=delta_delta_t_min_abs,
        delta_delta_t_min_des=delta_delta_t_min_des,
        Q_dot_cond=Q_dot_cond,
        Q_dot_evap=Q_dot_evap,
        Q_dot_abs=Q_dot_abs,
        Q_dot_des=Q_dot_des,
        T_evap=T_evap,
        T_cond=T_cond,
        T_abs=T_abs,
        T_des=T_des,
        Q_dot_cond_error_abs=Q_dot_cond - imported.Q_sim_cond,
        Q_dot_evap_error_abs=Q_dot_evap - imported.Q_sim_evap,
        Q_dot_abs_error_abs=Q_dot_abs - imported.Q_sim_abs,
        Q_dot_des_error_abs=Q_dot_des - imported.Q_sim_des,
        Q_dot_cond_error_rel=_safe_rel_error(Q_dot_cond, imported.Q_sim_cond),
        Q_dot_evap_error_rel=_safe_rel_error(Q_dot_evap, imported.Q_sim_evap),
        Q_dot_abs_error_rel=_safe_rel_error(Q_dot_abs, imported.Q_sim_abs),
        Q_dot_des_error_rel=_safe_rel_error(Q_dot_des, imported.Q_sim_des),
        T_evap_error_abs=T_evap - imported.T_sim_evap_mean,
        T_cond_error_abs=T_cond - imported.T_sim_cond_mean,
        T_abs_error_abs=T_abs - imported.T_sim_abs_mean,
        T_des_error_abs=T_des - imported.T_sim_des_mean,
    )



def run_conventional_from_detailed_simulation(
    inputs: "AWTInputs",
    x0: "np.ndarray | None" = None,
    *,
    B_duering: float = 1.15,
    R_abs: float = 0.0,
    R_des: float = 0.0,
) -> tuple["AWTResult", ImportedSimulationValues, ConventionalCharacteristicResult]:
    """Kompletter Ablauf: Detailsimulation -> Import -> char. Gleichungen."""
    result, imported = solve_detailed_and_import(inputs, x0=x0)
    conventional = calculate_conventional_characteristic_equations(
        imported,
        B_duering=B_duering,
        R_abs=R_abs,
        R_des=R_des,
    )
    return result, imported, conventional


# ---------------------------------------------------------------------------
# Ausgabehilfe
# ---------------------------------------------------------------------------


def _fmt_rel(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{100.0 * value:.3f} %"



def print_conventional_summary(conventional: ConventionalCharacteristicResult) -> None:
    imported = conventional.imported

    print("=" * 110)
    print("AWT – Herkömmliche Methode der charakteristischen Gleichungen")
    print("=" * 110)

    print("Importierte Simulationswerte")
    print(f"  cp_w                 : {imported.cp_w:14.6f} kJ/kg/K")
    print(f"  m_dot_p             : {imported.m_dot_p:14.6f} kg/s")
    print(f"  UA_evap             : {imported.UA_evap:14.6f} kW/K")
    print(f"  UA_cond             : {imported.UA_cond:14.6f} kW/K")
    print(f"  UA_abs              : {imported.UA_abs:14.6f} kW/K")
    print(f"  UA_des              : {imported.UA_des:14.6f} kW/K")
    print()

    print("Apparateparameter")
    print(f"  NTU_evap            : {conventional.NTU_evap:14.6f} -")
    print(f"  NTU_cond            : {conventional.NTU_cond:14.6f} -")
    print(f"  NTU_abs             : {conventional.NTU_abs:14.6f} -")
    print(f"  NTU_des             : {conventional.NTU_des:14.6f} -")
    print(f"  z_evap              : {conventional.z_evap:14.6f} -")
    print(f"  z_cond              : {conventional.z_cond:14.6f} -")
    print(f"  z_abs               : {conventional.z_abs:14.6f} -")
    print(f"  z_des               : {conventional.z_des:14.6f} -")
    print()

    print("Charakteristische Koeffizienten")
    print(f"  delta_delta_t       : {conventional.delta_delta_t:14.6f} K")
    print(f"  K_evap              : {conventional.K_evap:14.6f} -")
    print(f"  K_abs               : {conventional.K_abs:14.6f} -")
    print(f"  K_des               : {conventional.K_des:14.6f} -")
    print(f"  K_cond              : {conventional.K_cond:14.6f} -")
    print(f"  Ks_abs              : {conventional.Ks_abs:14.6f} kW")
    print(f"  Ks_des              : {conventional.Ks_des:14.6f} kW")
    print(f"  s_cond              : {conventional.s_cond:14.6f} kW/K")
    print(f"  s_evap              : {conventional.s_evap:14.6f} kW/K")
    print(f"  s_abs               : {conventional.s_abs:14.6f} kW/K")
    print(f"  s_des               : {conventional.s_des:14.6f} kW/K")
    print()

    print("Berechnete Wärmeströme")
    print(f"  Q_dot_cond          : {conventional.Q_dot_cond:14.6f} kW")
    print(f"  Q_dot_evap          : {conventional.Q_dot_evap:14.6f} kW")
    print(f"  Q_dot_abs           : {conventional.Q_dot_abs:14.6f} kW")
    print(f"  Q_dot_des           : {conventional.Q_dot_des:14.6f} kW")
    print()

    print("Vergleich mit Detailsimulation")
    print(
        f"  Kondensator         : Δabs = {conventional.Q_dot_cond_error_abs:12.6f} kW | "
        f"Δrel = {_fmt_rel(conventional.Q_dot_cond_error_rel)}"
    )
    print(
        f"  Verdampfer          : Δabs = {conventional.Q_dot_evap_error_abs:12.6f} kW | "
        f"Δrel = {_fmt_rel(conventional.Q_dot_evap_error_rel)}"
    )
    print(
        f"  Absorber            : Δabs = {conventional.Q_dot_abs_error_abs:12.6f} kW | "
        f"Δrel = {_fmt_rel(conventional.Q_dot_abs_error_rel)}"
    )
    print(
        f"  Desorber            : Δabs = {conventional.Q_dot_des_error_abs:12.6f} kW | "
        f"Δrel = {_fmt_rel(conventional.Q_dot_des_error_rel)}"
    )
    print()

    print("Interne mittlere Temperaturen")
    print(f"  T_evap              : {conventional.T_evap:14.6f} K")
    print(f"  T_cond              : {conventional.T_cond:14.6f} K")
    print(f"  T_abs               : {conventional.T_abs:14.6f} K")
    print(f"  T_des               : {conventional.T_des:14.6f} K")
    print()

    print("Temperaturvergleich zur Detailsimulation")
    print(f"  T_evap - T_sim      : {conventional.T_evap_error_abs:14.6f} K")
    print(f"  T_cond - T_sim      : {conventional.T_cond_error_abs:14.6f} K")
    print(f"  T_abs  - T_sim      : {conventional.T_abs_error_abs:14.6f} K")
    print(f"  T_des  - T_sim      : {conventional.T_des_error_abs:14.6f} K")
    print("=" * 110)


# ---------------------------------------------------------------------------
# Demo-Ausführung
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    try:
        from main_JW import build_example_inputs
    except ImportError as exc:
        raise CharacteristicEquationError(
            "Für die Demo-Ausführung konnte build_example_inputs() aus main_JW nicht geladen werden."
        ) from exc

    inputs = build_example_inputs()
    x0 = np.array(
    [
            305.45,    # T8  [K] 328.17, 305.45
            322.88,    # T10 [K] 374.24, 322,88
            0.149,       # x3  [-] 0.23312, 0.149
            0.163,       # x6  [-] 0.26933, 0.163
            0.155,      # x20 [-] 0.26367, 0.155
            329.97,    # T2  [K] 394.08, 329.97
            341.17,    # T4  [K] 422.82, 341.17
            0.1,       # beta [-] 0.142857, 0.1
        ],
    dtype=float,
    )
    _, _, conventional = run_conventional_from_detailed_simulation(inputs, x0=x0)
    print_conventional_summary(conventional)


if __name__ == "__main__":
    _run_demo()


__all__ = [
    "CharacteristicEquationError",
    "ImportedSimulationValues",
    "ConventionalCharacteristicResult",
    "import_simulation_values",
    "solve_detailed_and_import",
    "calculate_conventional_characteristic_equations",
    "run_conventional_from_detailed_simulation",
    "print_conventional_summary",
]
