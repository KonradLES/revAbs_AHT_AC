"""AWT-Simulation mit 7 primären Unbekannten.

Die Absorber-Spezifikation ist variabel:
- absorber_spec_mode = "m11": m11_spec wird vorgegeben, T12 wird berechnet
- absorber_spec_mode = "T12": T12_spec_C wird vorgegeben, m11 wird berechnet

Die Kreislaufskalierung ist variabel:
- cycle_scale_spec_mode = "m6": m6_spec wird vorgegeben
- cycle_scale_spec_mode = "Qeva": Qeva_spec_kW wird vorgegeben, m6 wird berechnet

Modellannahmen
--------------
- Arbeitsstoffpaar: H2O/LiBr
- stationärer Betrieb
- keine Druckverluste in Apparaten und Leitungen
- isenthalpe Lösungsdrossel mit lokaler Flash-Berechnung
- adiabate Vorabsorption vor dem Absorber wird explizit abgebildet
- externe Fluide werden mit konstantem cp_w beschrieben

Primäre Solvervariablen
-----------------------
z = [T8, T10, x4, x1, T3, T5]

Interne Einheiten
-----------------
- Temperatur: K
- Druck: Pa
- Massenstrom: kg/s
- spezifische Enthalpie: kJ/kg
- Wärmestrom / Leistung: kW (= kJ/s)
- UA: kW/K

Strategie für unphysikalische Zwischenzustände
----------------------------------------------
Der Solver (trf) besucht während der Iteration zwangsläufig Punkte, an denen
Temperaturdifferenzen in Wärmeübertragern negativ werden.  Die strenge
Endauswertung wirft dort eine ModelEvaluationError – das ist korrekt für die
physikalische Bewertung des Endpunkts.

Für den Solver-Pfad verwendet diese Datei denselben Modellkern in einer
robusten Variante:
  - Identische Gleichungsstruktur wie die strenge Endauswertung
  - counterflow_lmtd_soft statt counterflow_lmtd: gibt bei ΔT ≤ 0 den Wert
    min(ΔT1, ΔT2) zurück statt eine Exception zu werfen
  - Keine raises für negative Wärmeströme, negative Massenströme etc.
  - Residuen Q - UA·LMTD_soft sind groß und korrekt vorzeichenbehaftet
    → Solver erhält echtes Gradientensignal zurück in die physikalische Region

Diese robuste Variante hat dieselben Nullstellen wie die strenge Auswertung
(LMTD_soft = LMTD wenn beide ΔT > 0), konvergiert also zur korrekten
physikalischen Lösung.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    import CoolProp.CoolProp as CP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CoolProp ist nicht installiert. Installation z. B. mit `pip install CoolProp`."
    ) from exc

import Thermodynamic_Properties.libr_props as lp

PRIMARY_VARIABLE_NAMES = ["T8", "T10", "x4", "x1", "T3", "T5"]
RESIDUAL_NAMES = [
    "R1_SHEX_energy",
    "R2_SHEX_UA",
    "R4_desorber_UA",
    "R5_condenser_UA",
    "R6_evaporator_UA",
    "R7_absorber_UA",
]

PRIMARY_TEMPERATURE_INDICES = (0, 1, 4, 5)


def kelvin_to_celsius(T_K: float) -> float:
    return float(T_K) - 273.15


def celsius_to_kelvin(T_C: float) -> float:
    return float(T_C) + 273.15


def primary_temperatures_C_to_K(
    z_user: np.ndarray | list[float] | tuple[float, ...]
) -> np.ndarray:
    """Konvertiert die Temperatur-Komponenten des primären Vektors von °C nach K."""
    z_internal = np.asarray(z_user, dtype=float).copy()
    z_internal[list(PRIMARY_TEMPERATURE_INDICES)] += 273.15
    return z_internal


def primary_temperatures_K_to_C(
    z_internal: np.ndarray | list[float] | tuple[float, ...]
) -> np.ndarray:
    """Konvertiert die Temperatur-Komponenten des primären Vektors von K nach °C."""
    z_user = np.asarray(z_internal, dtype=float).copy()
    z_user[list(PRIMARY_TEMPERATURE_INDICES)] -= 273.15
    return z_user


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AKMInputs:
    # Externe Einlasstemperaturen [°C]
    T_11_C: float
    T_13_C: float | None
    T_15_C: float | None
    T_17_C: float

    # Externe Massenströme [kg/s]
    m_11: float
    m_13: float
    m_15: float

    # UA-Werte [kW/K]
    UA_cond: float
    UA_evap: float
    UA_abs: float
    UA_des: float
    UA_shex: float | None = None
    Effectiveness_shex: float | None = None

    _: KW_ONLY

    # SHEX-Spezifikationsmodus:
    # - "UA":  UA_shex wird verwendet (Default)
    # - "NTU": shex_effectiveness wird verwendet
    shex_model: str = "UA"

    # Externe Wärmesenken/-quellen von Desorber und Verdampfer:
    # - "parallel": Standardfall mit separater T_13- und T_15-Vorgabe
    # - "series_absorber_to_condenser": intern gilt T15 = T14
    # - "series_condenser_to_absorber": intern gilt T13 = T16
    absorber_condenser_routing_mode: str = "parallel"

    # Spezifikation der Kreislaufskalierung:
    # - "m1": m1_spec wird vorgegeben
    # - "Qevap": Qevap_spec_kW wird vorgegeben, m1 wird berechnet
    cycle_scale_spec_mode: str = "m1"
    m1_spec: float | None = None
    Qevap_spec_kW: float | None = None

    # Spezifikation des externen Absorberstroms:
    # - "m17": m17_spec wird vorgegeben, T18 wird berechnet
    # - "T18": T18_spec_C wird vorgegeben, m17 wird berechnet
    evaporator_spec_mode: str = "m17"
    m17_spec: float | None = None
    T18_spec_C: float | None = None

    # Externe Fluide: Wasser
    cp_w_kJkgK: float = 4.2

    # Desorberaustritt des Kältemitteldampfes
    # Default: gesättigter Dampf auf low-pressure-Niveau
    desorber_vapor_superheat_K: float = 0.0

    # Solver
    solver_tol: float = 1.0e-9
    max_nfev: int = 5000
    penalty_level: float = 1.0e6

    def __post_init__(self) -> None:
        if self.absorber_condenser_routing_mode not in {
            "parallel",
            "series_absorber_to_condenser",
            "series_condenser_to_absorber",
        }:
            raise ValueError(
                "absorber_condenser_routing_mode muss 'parallel', "
                "'series_absorber_to_condenser' oder 'series_condenser_to_absorber' sein."
            )

        if self.absorber_condenser_routing_mode == "parallel":
            if self.T_13_C is None:
                raise ValueError(
                    "Bei absorber_condenser_routing_mode='parallel' muss T_13_C vorgegeben werden."
                )
            if self.T_15_C is None:
                raise ValueError(
                    "Bei absorber_condenser_routing_mode='parallel' muss T_15_C vorgegeben werden."
                )
        elif self.absorber_condenser_routing_mode == "series_absorber_to_condenser":
            if self.T_13_C is None:
                raise ValueError(
                    "Bei absorber_condenser_routing_mode='series_absorber_to_condenser' muss "
                    "T_13_C vorgegeben werden."
                )
            if self.T_15_C is not None:
                raise ValueError(
                    "Bei absorber_condenser_routing_mode='series_absorber_to_condenser' darf T_15_C "
                    "nicht gesetzt sein; es gilt intern T15 = T14."
                )
        else:
            if self.T_15_C is None:
                raise ValueError(
                    "Bei absorber_condenser_routing_mode='series_condenser_to_absorber' muss "
                    "T_15_C vorgegeben werden; es gilt intern T13 = T16."
                )
            if self.T_13_C is not None:
                raise ValueError(
                    "Bei absorber_condenser_routing_mode='series_condenser_to_absorber' darf T_13_C "
                    "nicht gesetzt sein; es gilt intern T13 = T16."
                )

        if self.cycle_scale_spec_mode not in {"m1", "Qeva"}:
            raise ValueError("cycle_scale_spec_mode muss 'm1' oder 'Qeva' sein.")
        if self.cycle_scale_spec_mode == "m1":
            if self.m1_spec is None:
                raise ValueError("Bei cycle_scale_spec_mode='m1' muss m1_spec vorgegeben werden.")
            if self.Qevap_spec_kW is not None:
                raise ValueError(
                    "Bei cycle_scale_spec_mode='m1' darf Qeva_spec_kW nicht gesetzt sein."
                )
            if self.m1_spec <= 0.0:
                raise ValueError("Bei cycle_scale_spec_mode='m1' muss m1_spec > 0 gelten.")
        else:
            if self.Qevap_spec_kW is None:
                raise ValueError(
                    "Bei cycle_scale_spec_mode='Qeva' muss Qeva_spec_kW vorgegeben werden."
                )
            if self.m1_spec is not None:
                raise ValueError(
                    "Bei cycle_scale_spec_mode='Qeva' darf m1_spec nicht gesetzt sein."
                )
            if self.Qevap_spec_kW <= 0.0:
                raise ValueError("Bei cycle_scale_spec_mode='Qeva' muss Qeva_spec_kW > 0 gelten.")

        if self.evaporator_spec_mode not in {"m17", "T18"}:
            raise ValueError("evaporator_spec_mode muss 'm17' oder 'T18' sein.")
        if self.evaporator_spec_mode == "m17":
            if self.m17_spec is None:
                raise ValueError("Bei evaporator_spec_mode='m17' muss m17_spec vorgegeben werden.")
            if self.T18_spec_C is not None:
                raise ValueError(
                    "Bei evaporator_spec_mode='m17' darf T18_spec_C nicht gesetzt sein."
                )
            if self.m17_spec <= 0.0:
                raise ValueError("Bei evaporator_spec_mode='m17' muss m17_spec > 0 gelten.")
        else:
            if self.T18_spec_C is None:
                raise ValueError("Bei evaporator_spec_mode='T18' muss T18_spec_C vorgegeben werden.")
            if self.m17_spec is not None:
                raise ValueError(
                    "Bei evaporator_spec_mode='T18' darf m17_spec nicht gesetzt sein."
                )
            if self.T18_spec_C >= self.T_17_C:
                raise ValueError(
                    "Bei evaporator_spec_mode='T18' muss T18_spec_C < T_17_C gelten."
                )
        if self.shex_model not in {"UA", "NTU"}:
            raise ValueError("shex_model muss 'UA' oder 'NTU' sein.")
        if self.shex_model == "UA":
            if self.UA_shex is None:
                raise ValueError("Bei shex_model='UA' muss UA_shex vorgegeben werden.")
            if self.UA_shex <= 0.0:
                raise ValueError("Bei shex_model='UA' muss UA > 0 gelten.")
        else:
            if self.Effectiveness_shex is None:
                raise ValueError("Bei shex_model='NTU' muss Effectiveness_shex vorgegeben werden.")
            if self.UA_shex is not None:
                raise ValueError(
                    "Bei Effectiveness_shex='NTU' darf UA_shex nicht gesetzt sein."
                )
            
    @property
    def T_17(self) -> float:
        return celsius_to_kelvin(self.T_17_C)

    @property
    def T18_spec(self) -> float:
        if self.T18_spec_C is None:
            raise AttributeError("T18_spec_C ist für diese Spezifikation nicht gesetzt.")
        return celsius_to_kelvin(self.T18_spec_C)

    @property
    def T_13(self) -> float:
        if self.T_13_C is None:
            raise AttributeError(
                "T_13_C ist für die gewählte Routing-Variante nicht gesetzt."
            )
        return celsius_to_kelvin(self.T_13_C)

    @property
    def T_15(self) -> float:
        if self.T_15_C is None:
            raise AttributeError(
                "T_15_C ist für die gewählte Routing-Variante nicht gesetzt."
            )
        return celsius_to_kelvin(self.T_15_C)

    @property
    def T_11(self) -> float:
        return celsius_to_kelvin(self.T_11_C)

    @property
    def uses_serial_absorber_to_condenser_routing(self) -> bool:
        return self.absorber_condenser_routing_mode == "series_absorber_to_condenser"

    @property
    def uses_serial_condenser_to_absorber_routing(self) -> bool:
        return self.absorber_condenser_routing_mode == "series_condenser_to_absorber"

    @property
    def uses_any_serial_absorber_condenser_routing(self) -> bool:
        return self.absorber_condenser_routing_mode in {
            "series_absorber_to_condenser",
            "series_condenser_to_absorber",
        }

    @property
    def condenser_temperature_reference(self) -> float:
        """Referenztemperatur für Startwerte und Schranken des Kondensators."""
        if self.uses_serial_absorber_to_condenser_routing:
            return self.T_13
        return self.T_15

    @property
    def absorber_temperature_reference(self) -> float:
        """Referenztemperatur für Startwerte und Schranken des Absorbers."""
        if self.uses_serial_condenser_to_absorber_routing:
            return self.T_15
        return self.T_13


@dataclass(frozen=True)
class SolveInfo:
    success: bool
    status: int
    message: str
    cost: float
    scaled_residual_norm: float
    raw_residual_norm: float | None
    nfev: int
    final_point_evaluable: bool
    final_evaluation_error: str | None


@dataclass(frozen=True)
class ModelEvaluation:
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    pump_work_W: Dict[str, float]
    lmtd_K: Dict[str, float]
    compositions: Dict[str, float]
    flash_outputs: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    checks: Dict[str, bool]
    validity_messages: List[str]


@dataclass(frozen=True)
class AWTResult:
    inputs: AKMInputs
    solve_info: SolveInfo
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    pump_work_W: Dict[str, float]
    lmtd_K: Dict[str, float]
    compositions: Dict[str, float]
    flash_outputs: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    checks: Dict[str, bool]
    validity_messages: List[str]

@dataclass(frozen=True)
class ModelTrace:
    primary_variables: Dict[str, float]
    values: Dict[str, float]
    stage: str
    success: bool
    error_type: str | None
    error_message: str | None

class ModelEvaluationError(RuntimeError):
    """Interner Fehler bei der Modellbewertung."""


# ---------------------------------------------------------------------------
# Wasser-Stofffunktionen (CoolProp-Wrapper)
# ---------------------------------------------------------------------------

def water_h_kjkg_PT(P_pa: float, T_K: float) -> float:
    return CP.PropsSI("H", "P", P_pa, "T", T_K, "Water") / 1000.0


def water_h_kjkg_PQ(P_pa: float, Q: float) -> float:
    return CP.PropsSI("H", "P", P_pa, "Q", Q, "Water") / 1000.0


def water_T_K_PH(P_pa: float, h_kjkg: float) -> float:
    return CP.PropsSI("T", "P", P_pa, "H", h_kjkg * 1000.0, "Water")


def water_p_sat_from_T(T_K: float, Q: float) -> float:
    return CP.PropsSI("P", "T", T_K, "Q", Q, "Water")


def water_T_sat_from_p(P_pa: float, Q: float) -> float:
    return CP.PropsSI("T", "P", P_pa, "Q", Q, "Water")


def water_rho_kgm3_PQ(P_pa: float, Q: float) -> float:
    return CP.PropsSI("D", "P", P_pa, "Q", Q, "Water")


# ---------------------------------------------------------------------------
# Allgemeine Hilfsfunktionen
# ---------------------------------------------------------------------------

def lmtd(delta_T_1: float, delta_T_2: float) -> float:
    """Strenge LMTD: wirft ModelEvaluationError für ΔT ≤ 0."""
    if delta_T_1 <= 0.0 or delta_T_2 <= 0.0:
        raise ModelEvaluationError(
            f"LMTD undefiniert, weil delta_T_1={delta_T_1:.6f} K oder"
            f" delta_T_2={delta_T_2:.6f} K nicht positiv ist."
        )
    if math.isclose(delta_T_1, delta_T_2, rel_tol=1.0e-10, abs_tol=1.0e-10):
        return 0.5 * (delta_T_1 + delta_T_2)
    return (delta_T_1 - delta_T_2) / math.log(delta_T_1 / delta_T_2)


def lmtd_soft(delta_T_1: float, delta_T_2: float) -> float:
    """Robuste LMTD für den Solver-Pfad.

    Bei ΔT > 0: identisch mit lmtd().
    Bei ΔT ≤ 0: gibt min(ΔT1, ΔT2) zurück (negativ / null).

    Dadurch bleibt das Residuum Q - UA·LMTD_soft überall definiert und
    kontinuierlich.  Bei negativen Temperaturdifferenzen wird LMTD_soft
    negativ → Q - UA·(negativ) = Q + |UA·LMTD| ist groß und positiv →
    ||R||² steigt → der Solver erhält das richtige Gradientensignal zurück
    in die physikalisch gültige Region.

    Dieselben Nullstellen wie lmtd(): solange die Lösung physikalisch ist
    (ΔT > 0), sind lmtd() = lmtd_soft() → keine Verschiebung der Lösung.
    """
    if delta_T_1 <= 0.0 or delta_T_2 <= 0.0:
        return min(delta_T_1, delta_T_2)
    if math.isclose(delta_T_1, delta_T_2, rel_tol=1.0e-10, abs_tol=1.0e-10):
        return 0.5 * (delta_T_1 + delta_T_2)
    return (delta_T_1 - delta_T_2) / math.log(delta_T_1 / delta_T_2)


def counterflow_lmtd(hot_in: float, hot_out: float, cold_in: float, cold_out: float) -> float:
    return lmtd(hot_in - cold_out, hot_out - cold_in)


def counterflow_lmtd_soft(hot_in: float, hot_out: float, cold_in: float, cold_out: float) -> float:
    return lmtd_soft(hot_in - cold_out, hot_out - cold_in)


def heating_outlet_temperature(T_in: float, Q_kW: float, m_kg_s: float, cp_kJkgK: float) -> float:
    if m_kg_s <= 0.0 or cp_kJkgK <= 0.0:
        raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    return T_in + Q_kW / (m_kg_s * cp_kJkgK)


def cooling_outlet_temperature(T_in: float, Q_kW: float, m_kg_s: float, cp_kJkgK: float) -> float:
    if m_kg_s <= 0.0 or cp_kJkgK <= 0.0:
        raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    return T_in - Q_kW / (m_kg_s * cp_kJkgK)

def water_throttle_state(p_out_pa: float, h_in_kJkg: float) -> Dict[str, float]:
    """Isenthalpe Drossel für reines Kältemittel (Wasser), Kondensator -> Verdampfer.
 
    Im Gegensatz zur Lösungsdrossel (LiBr/H2O, siehe lp.flash_valve_state_5_to_6)
    ist hier keine Mischungsrechnung nötig: es handelt sich um reines Wasser,
    daher genügt eine einfache Dampfgehalts-Berechnung bei p_out_pa.
    """
    h_f = water_h_kjkg_PQ(p_out_pa, Q=0.0)
    h_g = water_h_kjkg_PQ(p_out_pa, Q=1.0)
    T_sat = water_T_sat_from_p(p_out_pa, Q=0.0)
 
    if h_g <= h_f:
        # Entartungsfall (numerisch), sollte praktisch nicht auftreten
        return {"T9_K": T_sat, "q9": 0.0, "h9_kJ_kg": h_in_kJkg}
 
    if h_in_kJkg <= h_f:
        q = 0.0
        T = T_sat
    elif h_in_kJkg >= h_g:
        q = 1.0
        T = water_T_K_PH(p_out_pa, h_in_kJkg)
    else:
        q = (h_in_kJkg - h_f) / (h_g - h_f)
        T = T_sat
 
    return {"T9_K": T, "q9": q, "h9_kJ_kg": h_in_kJkg}

def _penalty_vector(size: int, level: float) -> np.ndarray:
    return np.full(size, level, dtype=float)


def _residual_scales(m1: float) -> np.ndarray:
    """Skalierung der sieben energetischen Residuen [kW]."""
    return np.array(
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        dtype=float,
    )


def _state_dict(
    T_K: float,
    p_Pa: float | None = None,
    m_kg_s: float | None = None,
    h_kJ_kg: float | None = None,
    x_LiBr_mol: float | None = None,
    w_LiBr: float | None = None,
) -> Dict[str, float]:
    state: Dict[str, float] = {"T_K": float(T_K)}
    if p_Pa is not None:
        state["p_Pa"] = float(p_Pa)
    if m_kg_s is not None:
        state["m_kg_s"] = float(m_kg_s)
    if h_kJ_kg is not None:
        state["h_kJ_kg"] = float(h_kJ_kg)
    if x_LiBr_mol is not None:
        state["x_LiBr_mol"] = float(x_LiBr_mol)
    if w_LiBr is not None:
        state["w_LiBr"] = float(w_LiBr)
    return state


def _calculate_kpis(
    *,
    Q_des: float,
    Q_evap: float,
    T12: float,
    T_hs_ref: float,
    m1: float,
    m7: float,
) -> Dict[str, float]:
    denominator_cop = Q_des
    cop = float("nan") if abs(denominator_cop) <= 1.0e-12 else Q_evap / denominator_cop
    fr = float("nan") if abs(m7) <= 1.0e-12 else m1 / m7

    return {
        "COP": cop,
        "GTL_K": T12 - T_hs_ref,
        "FR": fr,
    }


def _resolve_cycle_scale(
    inputs: AKMInputs, *, w4: float, w1: float, h9: float, h10: float, strict: bool
) -> float:
    """Löst die Kreislaufskalierung auf den gepumpten Lösungsmassenstrom m1 auf."""
    if inputs.cycle_scale_spec_mode == "m1":
        m1 = float(inputs.m1_spec)  # durch __post_init__ abgesichert
        if strict and m1 <= 0.0:
            raise ModelEvaluationError("Gepumpter Lösungsmassenstrom m1 muss positiv sein.")
        return m1

    w4_balance = w4 if strict else max(w4, 1.0e-9)
    ratio = w1 / w4_balance
    denominator = (1 - ratio) * (h10 - h9)

    if strict:
        if abs(denominator) <= 1.0e-12:
            raise ModelEvaluationError(
                "Kreislaufskalierung aus Q_eva nicht möglich, weil der Nenner nahezu null ist."
            )
        m1 = float(inputs.Qevap_spec_kW) / denominator
        if m1 <= 0.0:
            raise ModelEvaluationError(
                f"Berechneter Lösungsmassenstrom m1 nicht positiv: m1={m1:.6f} kg/s."
            )
        return m1

    denominator_safe = denominator
    if abs(denominator_safe) <= 1.0e-12:
        denominator_safe = 1.0e-12 if denominator_safe >= 0.0 else -1.0e-12
    return float(inputs.Qevap_spec_kW) / denominator_safe


def _resolve_evaporator_external_stream(
    inputs: AKMInputs, Q_evap: float, *, strict: bool
) -> tuple[float, float]:
    """Löst die Verdampfer-Spezifikation auf interne Arbeitsgrößen auf.

    Rückgabe
    --------
    (m17, T18)
    """
    if inputs.evaporator_spec_mode == "m17":
        m17 = float(inputs.m17_spec)  # durch __post_init__ abgesichert
        if strict:
            T18 = cooling_outlet_temperature(inputs.T_17, Q_evap, m17, inputs.cp_w_kJkgK)
        else:
            T18 = inputs.T_17 - Q_evap / (m17 * inputs.cp_w_kJkgK)
        return m17, T18

    T18 = inputs.T18_spec
    delta_T = inputs.T_17 - T18
    if strict:
        if delta_T <= 0.0:
            raise ModelEvaluationError(
                "Für evaporator_spec_mode='T18' muss T17 > T18 gelten."
            )
        if inputs.cp_w_kJkgK <= 0.0:
            raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    m17 = Q_evap / (inputs.cp_w_kJkgK * delta_T)
    return m17, T18


def _resolve_condenser_external_inlet_temperature(inputs: AKMInputs, T14: float | None = None) -> float:
    """Löst die externe Kondensatoreinlasstemperatur auf.

    - parallel: T15 wird aus den Inputs gelesen
    - series_absorber_to_condenser: T15 entspricht dem externen Absorberaustritt T14
    - series_condenser_to_absorber: T15 bleibt externer Input
    """
    if inputs.uses_serial_absorber_to_condenser_routing:
        if T14 is None:
            raise ModelEvaluationError("Für series_absorber_to_condenser muss T14 bekannt sein.")
        return T14
    return inputs.T_15


def _resolve_absorber_external_inlet_temperature(inputs: AKMInputs, T16: float | None = None) -> float:
    """Löst die externe Absorbereinlasstemperatur auf.

    - parallel: T13 wird aus den Inputs gelesen
    - series_absorber_to_condenser: T13 bleibt externer Input
    - series_condenser_to_absorber: T13 entspricht dem externen Kondensatoraustritt T16
    """
    if inputs.uses_serial_condenser_to_absorber_routing:
        if T16 is None:
            raise ModelEvaluationError("Für series_condenser_to_absorber muss T16 bekannt sein.")
        return T16
    return inputs.T_13


# ---------------------------------------------------------------------------
# Solver-Hilfsfunktionen
# ---------------------------------------------------------------------------

def initial_guess(inputs: AKMInputs) -> np.ndarray:
    """Heuristische Startwerte für den 6-dimensionalen Solvervektor."""
    T_cond_ref = inputs.condenser_temperature_reference
    T_abs_ref = inputs.absorber_temperature_reference
    return np.array(
        [
            T_cond_ref + 15.0,      # T8
            inputs.T_17 - 8.0,      # T10
            0.22,                   # x4
            0.243,                  # x1
            T_abs_ref + 12.0,       # T3
            inputs.T_11 - 47.0,     # T5
        ],
        dtype=float,
    )


def bounds(inputs: AKMInputs) -> Tuple[np.ndarray, np.ndarray]:
    T_cond_ref = inputs.condenser_temperature_reference
    T_abs_ref = inputs.absorber_temperature_reference
    lower = np.array(
        [
            inputs.T_17 + 1.0,      # T8
            274.15,                 # T10
            0.05,                   # x4
            0.08,                   # x1
            inputs.T_17 + 1.0,      # T3
            inputs.T_17 + 1.0,      # T5
        ],
        dtype=float,
    )
    upper = np.array(
        [
            min(inputs.T_11 - 1.0, 420.0),   # T8
            min(inputs.T_17 + 0.5, 500.0),   # T10
            0.34,                           # x3
            0.39,                           # x6
            500.0,                          # T2
            500.0,                          # T4
        ],
        dtype=float,
    )
    return lower, upper



# ---------------------------------------------------------------------------
# Gemeinsamer Modellkern (streng für Endauswertung, robust für Solver-Pfad)
# ---------------------------------------------------------------------------

class _SoftResidualVector(RuntimeError):
    """Interne Ausnahme: robuster Residuenvektor steht bereits fest."""

    def __init__(self, residuals_scaled: np.ndarray):
        super().__init__("Soft residual vector ready.")
        self.residuals_scaled = np.asarray(residuals_scaled, dtype=float)


def _counterflow_lmtd_mode(
    *, strict: bool, hot_in: float, hot_out: float, cold_in: float, cold_out: float
) -> float:
    if strict:
        return counterflow_lmtd(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out)
    return counterflow_lmtd_soft(hot_in=hot_in, hot_out=hot_out, cold_in=cold_in, cold_out=cold_out)


def _scaled_residual_array(model: ModelEvaluation) -> np.ndarray:
    return np.array([model.residuals_scaled[name] for name in RESIDUAL_NAMES], dtype=float)


def _evaluate_model_common(z: np.ndarray, inputs: AKMInputs, *, strict: bool) -> ModelEvaluation:
    """Gemeinsamer Modellkern für strikte Endauswertung und robusten Solver-Pfad.

    strict=True:
        - identisches Verhalten wie die bisherige evaluate_model()-Funktion
        - wirft ModelEvaluationError für unphysikalische Zustände

    strict=False:
        - identische Gleichungsstruktur
        - robuste Varianten nur dort, wo sie für den Solver-Pfad nötig sind
        - keine Raises für negative Wärmeströme / Massenströme
        - counterflow_lmtd_soft statt counterflow_lmtd
        - Fallbacks für T5 und T1
        - bei fundamentaler Druckverletzung p_high <= p_low direkter Residuenvektor
    """
    T8, T10, x4, x1, T3, T5 = map(float, z)

    # ------------------------------------------------------------------
    # 1) Druckniveaus des Kältemittels
    # ------------------------------------------------------------------
    p_low = water_p_sat_from_T(T10, Q=1.0)
    p_high = water_p_sat_from_T(T8, Q=0.0)
    if p_high <= p_low:
        if strict:
            raise ModelEvaluationError(f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa).")
        pen = (p_low - p_high + 100.0) / 100.0
        raise _SoftResidualVector(np.full(len(RESIDUAL_NAMES), pen, dtype=float))

    # ------------------------------------------------------------------
    # 2) Gesättigte Lösungszustände, Konzentrationen und frühe Stoffgrößen
    # ------------------------------------------------------------------
    T1 = lp.T_sat_solution_from_p_x(p_low, x1)
    T4 = lp.T_sat_solution_from_p_x(p_high, x4)

    w1 = lp.w_libr_from_x(x1)
    w4 = lp.w_libr_from_x(x4)

    if strict and not (w4 > w1 > 0.0):
        raise ModelEvaluationError(
            f"Konzentrationshierarchie verletzt: w4={w4:.6f}, w1={w1:.6f}. Erwartet wird w4 > w1."
        )

    h3 = lp.h_solution_mass_kjkg(T3, x1)
    h4 = lp.h_solution_mass_kjkg(T4, x4)
    T7 = lp.T_sat_solution_from_p_x(p_high, x1) + inputs.desorber_vapor_superheat_K
    h7 = water_h_kjkg_PT(p_high, T7)
    h8 = water_h_kjkg_PQ(p_high, Q=0.0)
    h10 = water_h_kjkg_PQ(p_low, Q=1.0)

    # ------------------------------------------------------------------
    # 9) Drossel 8 -> 9
    # ------------------------------------------------------------------
    h9 = h8
    refrigerant_throttle = water_throttle_state(p_low, h9)
    T9 = refrigerant_throttle["T9_K"]
    q9 = refrigerant_throttle["q9"]

    # ------------------------------------------------------------------
    # 3) Kreislaufskalierung und Massenströme
    # ------------------------------------------------------------------
    m1 = _resolve_cycle_scale(inputs, w4=w4, w1=w1, h9=h9, h10=h10, strict=strict)
    m2 = m3 = m1 
    w4_for_balance = w4 if strict else max(w4, 1.0e-9)
    m4 = m3 * w1 / w4_for_balance
    m5 = m6 = m4 
    m7 = m8 = m9 = m10 = m1 - m4

    if strict and m7 <= 0.0:
        raise ModelEvaluationError(f"Kältemittelmassenstrom nicht positiv: m7={m7:.6f} kg/s.")

    if strict:
        if m10 <= 0.0:
            raise ModelEvaluationError(f"m10 nicht positiv: m10={m10:.6f} kg/s.")

    # ------------------------------------------------------------------
    # 4) Lösungsenthalpien und Lösungspumpe 1 -> 2
    # ------------------------------------------------------------------
    h5 = lp.h_solution_mass_kjkg(T5, x4)
    h1 = lp.h_solution_mass_kjkg(T1, x1)

    rho1 = lp.rho_solution_mass(T1, x1)
    v1 = 1.0 / rho1
    W_sol_pump = m1 * v1 * (p_high - p_low) / 1000.0  # kW
    h2 = h1 + W_sol_pump / m1
    if strict:
        T2 = lp.T_from_h_x_mass(h2, x1)
    else:
        try:
            T2 = lp.T_from_h_x_mass(h2, x1)
        except Exception:
            T2 = T1

    # ------------------------------------------------------------------
    # 5) Lösungswärmeübertrager (SHEX): 3 -> 2 und 5 -> 4
    # ------------------------------------------------------------------
    Q_shex_hot = m4 * (h4 - h5)
    Q_shex_cold = m1 * (h3 - h2)
    if strict and Q_shex_hot <= 0.0:
        raise ModelEvaluationError(f"Q_shex_hot nicht positiv: {Q_shex_hot:.6f} kW.")
    if strict and Q_shex_cold <= 0.0:
        raise ModelEvaluationError(f"Q_shex_cold nicht positiv: {Q_shex_cold:.6f} kW.")
    Q_shex = Q_shex_hot

    # LMTD wird unabhängig vom Spezifikationsmodus berechnet, da T2..T5 immer bekannt sind
    lmtd_shex = _counterflow_lmtd_mode(
        strict=strict, hot_in=T4, hot_out=T5, cold_in=T2, cold_out=T3
    )

    # SHEX: Wärmeübertragungsresiduum (R2) je nach Spezifikationsmodus
    if inputs.shex_model == "UA":
        R2_shex = Q_shex - inputs.UA_shex * lmtd_shex
        UA_shex_calc = inputs.UA_shex
    else:
        # Effektivitäts-NTU-Methode
        C23 = (h2 - h3) / (T2 - T3) if abs(T2 - T3) > 1.0e-12 else float("nan")
        C45 = (h4 - h5) / (T4 - T5) if abs(T4 - T5) > 1.0e-12 else float("nan")
        C2_3 = m1 * C23
        C4_5 = m4 * C45
        C_min = min(C2_3, C4_5)
        Q_shex_max = C_min * (T4 - T2)
        R2_shex = Q_shex - inputs.Effectiveness_shex * Q_shex_max
        # UA aus LMTD zurückgerechnet (nur informativ, geht nicht in die Residuen ein)
        if strict and lmtd_shex <= 0.0:
            raise ModelEvaluationError(
                f"LMTD_shex nicht positiv, UA-Rückrechnung nicht möglich: {lmtd_shex:.6f} K."
            )
        UA_shex_calc = Q_shex / lmtd_shex if lmtd_shex > 0.0 else float("nan")

    # Pinch-Residuum SHEX: kleinster Temperaturabstand = dT_min_shex
    dT_shex_hot_end  = T4 - T3   # heiß ein  / kalt aus
    dT_shex_cold_end = T5 - T2   # heiß aus  / kalt ein

    # ------------------------------------------------------------------
    # 6) Drossel 5 -> 6 (isenthalp, T6 aus Flash-Drossel)
    # ------------------------------------------------------------------
    h6 = h5
    if strict:
        flash = lp.flash_valve_state_5_to_6(
            p_out_pa=p_low,
            h5_kJkg=h5,
            m5_kg_s=m5,
            x5_libr_mol=x4,
        )
        T6 = flash["T6_K"]
    else:
        try:
            flash = lp.flash_valve_state_5_to_6(
                p_out_pa=p_low,
                h5_kJkg=h5,
                m5_kg_s=m5,
                x5_libr_mol=x4,
            )
            T6 = flash["T6_K"]
        except Exception:
            T6 = water_T_sat_from_p(p_low, Q=0.0)
            flash = {
                "T6_K": T6,
                "x6_LiBr_mol": x4,
                "w6_LiBr": w4,
                "m6_sol_kg_s": m4,
                "m6_flash_kg_s": 0.0,
                "h6_sol_kJ_kg": h6,
                "h6_flash_kJ_kg": float("nan"),
                "h6_mix_kJ_kg": h6,
                "flash_fraction": 0.0,
            }
    flash_outputs = {key: float(value) for key, value in flash.items()}

    # ------------------------------------------------------------------
    # 7) Kältemitteldampfpfad 7 sowie externe Heißseite von Desorber/Verdampfer
    # ------------------------------------------------------------------
    Q_des = - m1 * h3 + m7 * h7 + m4 * h4
    if strict and Q_des <= 0.0:
        raise ModelEvaluationError(f"Desorberwärmestrom nicht positiv: Q_des={Q_des:.6f} kW.")

    if strict:
        T12 = cooling_outlet_temperature(inputs.T_11, Q_des, inputs.m_11, inputs.cp_w_kJkgK)
    else:
        T12 = inputs.T_11 - Q_des / (inputs.m_11 * inputs.cp_w_kJkgK)
    
    lmtd_des = _counterflow_lmtd_mode(
        strict=strict, hot_in=inputs.T_11, hot_out=T12, cold_in=T7, cold_out=T4
    )

    # Pinch Desorber: min beider Enden
    dT_des_hot_end  = inputs.T_11 - T4          # heiß ein / kalt aus
    dT_des_cold_end = T12  - T7  # heiß aus / kalt ein

    # ------------------------------------------------------------------
    # 10) Verdampfer 9 -> 10 und gekoppelte externe Heißseite
    # ------------------------------------------------------------------
    Q_evap = m9 * (h10 - h9)
    if strict and Q_evap <= 0.0:
        raise ModelEvaluationError(f"Verdampferwärmestrom nicht positiv: Q_evap={Q_evap:.6f} kW.")
    m17, T18 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=strict)

    # if strict and T10 <= 273.15:
    #     raise ModelEvaluationError(f"Verdampfertemperatur T10 ngeativ -> Kätltemittel gefriert: T10={T10:.6f} K.")
    

    lmtd_evap = _counterflow_lmtd_mode(
        strict=strict, hot_in=inputs.T_17, hot_out=T18, cold_in=T9, cold_out=T10
    )

    # Pinch Verdampfer: min beider Enden (Lage hängt vom Betriebspunkt ab)
    dT_evap_hot_end  = inputs.T_17 - T10   # heiß ein / kalt aus
    dT_evap_cold_end = T18 - T9   # heiß aus / kalt ein

    # ------------------------------------------------------------------
    # 12) Absorber (globale Energiebilanz, lokale LMTD mit Zustand 20)
    # ------------------------------------------------------------------
    Q_abs = m10 * h10 + m4 * h6 - m1 * h1
    if strict and Q_abs <= 0.0:
        raise ModelEvaluationError(f"Absorberwärmestrom nicht positiv: Q_abs={Q_abs:.6f} kW.")

   # ------------------------------------------------------------------
    # 8) Kondensator 7 -> 8
    # ------------------------------------------------------------------
    Q_cond = m7 * (h7 - h8)
    if strict and Q_cond <= 0.0:
        raise ModelEvaluationError(f"Kondensatorwärmestrom nicht positiv: Q_cond={Q_cond:.6f} kW.")
    if inputs.uses_serial_condenser_to_absorber_routing:
        T15_in = _resolve_condenser_external_inlet_temperature(inputs)
        if strict:
            T16 = heating_outlet_temperature(T15_in, Q_cond, inputs.m_15, inputs.cp_w_kJkgK)
        else:
            T16 = T15_in + Q_cond / (inputs.m_15 * inputs.cp_w_kJkgK)
        lmtd_cond = _counterflow_lmtd_mode(
            strict=strict, hot_in=T8, hot_out=T8, cold_in=T15_in, cold_out=T16
        )
        T13_in = _resolve_absorber_external_inlet_temperature(inputs, T16)
        if strict:
            T14 = heating_outlet_temperature(T13_in, Q_abs, inputs.m_13, inputs.cp_w_kJkgK)
        else:
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        lmtd_abs = _counterflow_lmtd_mode(
            strict=strict, hot_in=T6, hot_out=T1, cold_in=T13_in, cold_out=T14
        )
    else:
        T13_in = _resolve_absorber_external_inlet_temperature(inputs)
        if strict:
            T14 = heating_outlet_temperature(T13_in, Q_abs, inputs.m_13, inputs.cp_w_kJkgK)
        else:
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        lmtd_abs = _counterflow_lmtd_mode(
            strict=strict, hot_in=T6, hot_out=T1, cold_in=T13_in, cold_out=T14
        )
        T15_in = _resolve_condenser_external_inlet_temperature(inputs, T14)
        if strict:
            T16 = heating_outlet_temperature(T15_in, Q_cond, inputs.m_15, inputs.cp_w_kJkgK)
        else:
            T16 = T15_in + Q_cond / (inputs.m_15 * inputs.cp_w_kJkgK)
        lmtd_cond = _counterflow_lmtd_mode(
            strict=strict, hot_in=T8, hot_out=T8, cold_in=T15_in, cold_out=T16
        )

    # Pinch Absorber: min beider Enden (Lage hängt vom Betriebspunkt ab)
    dT_abs_hot_end  = T6 - T14   # heiß ein / kalt aus
    dT_abs_cold_end = T1    - T13_in   # heiß aus / kalt ein

    # Pinch Kondensator: min beider Enden (Lage hängt vom Betriebspunkt ab)
    dT_cond_hot_end  = T8 - T16    # heiß ein / kalt aus
    dT_cond_cold_end = T8 - T15_in   # heiß aus / kalt ein

    # ------------------------------------------------------------------
    # 13) Residuen des 7x7-Systems
    # ------------------------------------------------------------------
    residuals_raw_array = np.array(
        [
            Q_shex_hot - Q_shex_cold,
            R2_shex,
            Q_des - inputs.UA_des * lmtd_des,
            Q_cond - inputs.UA_cond * lmtd_cond,
            Q_evap - inputs.UA_evap * lmtd_evap,
            Q_abs - inputs.UA_abs * lmtd_abs,
        ],
        dtype=float,
    )
    scales = _residual_scales(m1)
    residuals_scaled_array = residuals_raw_array / scales

    residuals_raw = dict(zip(RESIDUAL_NAMES, residuals_raw_array.tolist()))
    residuals_scaled = dict(zip(RESIDUAL_NAMES, residuals_scaled_array.tolist()))

    # ------------------------------------------------------------------
    # 14) Zustandsvalidierung und Plausibilitätschecks
    # ------------------------------------------------------------------
    validity_messages: List[str] = []
    crystallization_safe_all = True
    for label, T_state, w_state in [
        ("1 Flüssigphase nach Drossel", flash["T6_K"], flash["w6_LiBr"]),
        ("1", T1, w1), ("2", T2, w1), ("3", T3, w1), ("4", T4, w4)
    ]:
        validity = lp.validate_solution_state(T_state, w_state, label=f"Zustand {label}")
        validity_messages.append(validity.message)
        crystallization_safe_all = crystallization_safe_all and validity.crystallization_safe

    checks = {
        "p_high_gt_p_low": p_high > p_low,
        "w4_gt_w1": w4 > w1,
        "m7_positive": m7 > 0.0,
        "absorber_condenser_temperature_coupling_ok": (
            (abs(T15_in - T14) <= 1.0e-12)
            if inputs.uses_serial_absorber_to_condenser_routing
            else ((abs(T13_in - T16) <= 1.0e-12) if inputs.uses_serial_condenser_to_absorber_routing else True)
        ),
        "crystallization_safe_all_checked_states": crystallization_safe_all,
    }

    diagnostics = {
        "p_low_Pa": p_low,
        "p_high_Pa": p_high,
        "pressure_ratio_high_over_low": p_high / p_low,
        "T18_K": T18,
        "m6_kg_s": m6,
        "m17_kg_s": m17,
        "T13_K": T13_in,
        "T14_K": T14,
        "T15_K": T15_in,
        "T16_K": T16,
        "T12_K": T12,
        "deltaT_shex_1_K": dT_shex_hot_end,
        "deltaT_shex_2_K": dT_shex_cold_end,
        "deltaT_des_1_K": dT_des_hot_end,
        "deltaT_des_2_K": dT_des_cold_end,
        "deltaT_cond_1_K": dT_cond_hot_end,
        "deltaT_cond_2_K": dT_cond_cold_end,
        "deltaT_evap_1_K": dT_evap_hot_end,
        "deltaT_evap_2_K": dT_evap_cold_end,
        "deltaT_abs_1_K": dT_abs_hot_end,
        "deltaT_abs_2_K": dT_abs_cold_end,
    }

    # ------------------------------------------------------------------
    # 15) Zustandsdictionary
    # ------------------------------------------------------------------
    states = {
        "1":  _state_dict(T1,          p_Pa=p_low,  m_kg_s=m1,  h_kJ_kg=h1,  x_LiBr_mol=x1,  w_LiBr=w1),
        "2":  _state_dict(T2,          p_Pa=p_high, m_kg_s=m2,  h_kJ_kg=h2,  x_LiBr_mol=x1,  w_LiBr=w1),
        "3":  _state_dict(T3,          p_Pa=p_high, m_kg_s=m3,  h_kJ_kg=h3,  x_LiBr_mol=x1,  w_LiBr=w1),
        "4":  _state_dict(T4,          p_Pa=p_high, m_kg_s=m4,  h_kJ_kg=h4,  x_LiBr_mol=x4,  w_LiBr=w4),
        "5":  _state_dict(T5,          p_Pa=p_high, m_kg_s=m5,  h_kJ_kg=h5,  x_LiBr_mol=x4,  w_LiBr=w4),
        "6":  _state_dict(T6,          p_Pa=p_low,  m_kg_s=m6,  h_kJ_kg=h6,  x_LiBr_mol=x4,  w_LiBr=w4),
        "7":  _state_dict(T7,          p_Pa=p_high,  m_kg_s=m7,  h_kJ_kg=h7,  x_LiBr_mol=0.0, w_LiBr=0.0),
        "8":  _state_dict(T8,          p_Pa=p_high,  m_kg_s=m8,  h_kJ_kg=h8,  x_LiBr_mol=0.0, w_LiBr=0.0),
        "9":  _state_dict(T9,          p_Pa=p_low, m_kg_s=m9,  h_kJ_kg=h9,  x_LiBr_mol=0.0, w_LiBr=0.0),
        "10": _state_dict(T10,         p_Pa=p_low, m_kg_s=m10, h_kJ_kg=h10, x_LiBr_mol=0.0, w_LiBr=0.0),
        "11": _state_dict(inputs.T_11, m_kg_s=inputs.m_11),
        "12": _state_dict(T12,         m_kg_s=inputs.m_11),
        "13": _state_dict(T13_in,       m_kg_s=inputs.m_13),
        "14": _state_dict(T14,         m_kg_s=inputs.m_13),
        "15": _state_dict(T15_in,       m_kg_s=inputs.m_15),
        "16": _state_dict(T16,         m_kg_s=inputs.m_15),
        "17": _state_dict(inputs.T_17, m_kg_s=m17),
        "18": _state_dict(T18,         m_kg_s=m17),
    }

    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, [T8, T10, x4, x1, T3, T5]))
    if inputs.uses_serial_absorber_to_condenser_routing:
        T_hs_ref = inputs.T_13
    else:
        T_hs_ref = T15_in
    kpis = _calculate_kpis(
        Q_des=Q_des,
        Q_evap=Q_evap,
        T12=T12,
        T_hs_ref=T_hs_ref,
        m1=m1,
        m7=m7,
    )

    return ModelEvaluation(
        primary_variables=primary_variables,
        states=states,
        heat_flows_kW={
            "Q_shex": Q_shex,
            "Q_des": Q_des,
            "Q_cond": Q_cond,
            "Q_evap": Q_evap,
            "Q_abs": Q_abs,
        },
        kpis=kpis,
        pump_work_W={
            "W_sol_pump": W_sol_pump*1000,
        },
        lmtd_K={
            "LMTD_shex": lmtd_shex,
            "LMTD_des": lmtd_des,
            "LMTD_cond": lmtd_cond,
            "LMTD_evap": lmtd_evap,
            "LMTD_abs": lmtd_abs,
            "UA_shex_calc": UA_shex_calc,
        },
        compositions={
            "x1_LiBr_mol": x1,
            "x4_LiBr_mol": x4,
            "w1_LiBr": w1,
            "w4_LiBr": w4,
        },
        flash_outputs=flash_outputs,
        residuals_raw=residuals_raw,
        residuals_scaled=residuals_scaled,
        diagnostics=diagnostics,
        checks=checks,
        validity_messages=validity_messages,
    )


def evaluate_model(z: np.ndarray, inputs: AKMInputs) -> ModelEvaluation:
    """Berechnet alle Zustände, Apparategrößen und Residuen für einen Variablenvektor.

    Diese öffentliche Variante ist die strenge Endauswertung und wirft
    ModelEvaluationError für unphysikalische Zustände.
    """
    return _evaluate_model_common(z, inputs, strict=True)


# ---------------------------------------------------------------------------
# Solver-Interface
# ---------------------------------------------------------------------------

def residual_vector(z: np.ndarray, inputs: AKMInputs) -> np.ndarray:
    """Residuenvektor für least_squares.

    Schneller Pfad: strenge evaluate_model()-Auswertung.
    Fallback: derselbe Modellkern in robuster Solver-Variante (strict=False).

    Dadurch benutzt der Solver dieselbe Gleichungsstruktur wie die
    Endauswertung; nur die für den Solver-Pfad nötigen Robustifizierungen
    unterscheiden sich noch.
    """
    try:
        model = evaluate_model(z, inputs)
        return _scaled_residual_array(model)
    except ModelEvaluationError:
        try:
            model = _evaluate_model_common(z, inputs, strict=False)
            return _scaled_residual_array(model)
        except _SoftResidualVector as exc:
            return exc.residuals_scaled
        except Exception:
            return _penalty_vector(len(RESIDUAL_NAMES), inputs.penalty_level)
    except Exception:
        return _penalty_vector(len(RESIDUAL_NAMES), inputs.penalty_level)


def try_evaluate_model(
    z: np.ndarray, inputs: AKMInputs
) -> tuple[ModelEvaluation | None, str | None]:
    try:
        model = evaluate_model(z, inputs)
        return model, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def solve_awt(inputs: AKMInputs, x0: np.ndarray | None = None) -> AWTResult:
    if x0 is None:
        x0 = initial_guess(inputs)

    lower, upper = bounds(inputs)
    lsq = least_squares(
        fun=lambda zz: residual_vector(zz, inputs),
        x0=np.asarray(x0, dtype=float),
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=inputs.solver_tol,
        xtol=inputs.solver_tol,
        gtol=inputs.solver_tol,
        max_nfev=inputs.max_nfev,
        verbose=0,
    )

    scaled_residual_norm = float(np.linalg.norm(lsq.fun, ord=2))
    z_final = np.asarray(lsq.x, dtype=float)
    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, map(float, z_final)))

    model, final_evaluation_error = try_evaluate_model(z_final, inputs)
    final_point_evaluable = model is not None

    if final_point_evaluable:
        raw_residual_norm = float(
            np.linalg.norm(
                np.array([model.residuals_raw[name] for name in RESIDUAL_NAMES], dtype=float),
                ord=2,
            )
        )
    else:
        raw_residual_norm = None

    solve_info = SolveInfo(
        success=bool(lsq.success),
        status=int(lsq.status),
        message=str(lsq.message),
        cost=float(lsq.cost),
        scaled_residual_norm=scaled_residual_norm,
        raw_residual_norm=raw_residual_norm,
        nfev=int(lsq.nfev),
        final_point_evaluable=final_point_evaluable,
        final_evaluation_error=final_evaluation_error,
    )

    if model is None:
        return AWTResult(
            inputs=inputs,
            solve_info=solve_info,
            primary_variables=primary_variables,
            states={},
            heat_flows_kW={},
            kpis={},
            pump_work_W={},
            lmtd_K={},
            compositions={},
            flash_outputs={},
            residuals_raw={},
            residuals_scaled={},
            diagnostics={},
            checks={},
            validity_messages=[],
        )

    return AWTResult(
        inputs=inputs,
        solve_info=solve_info,
        primary_variables=model.primary_variables,
        states=model.states,
        heat_flows_kW=model.heat_flows_kW,
        kpis=model.kpis,
        pump_work_W=model.pump_work_W,
        lmtd_K=model.lmtd_K,
        compositions=model.compositions,
        flash_outputs=model.flash_outputs,
        residuals_raw=model.residuals_raw,
        residuals_scaled=model.residuals_scaled,
        diagnostics=model.diagnostics,
        checks=model.checks,
        validity_messages=model.validity_messages,
    )


# ---------------------------------------------------------------------------
# Debugging-Hilfe: Trace für Startwertanalyse
# ---------------------------------------------------------------------------

def trace_model(z: np.ndarray, inputs: AKMInputs) -> ModelTrace:
    """Wertet das Modell schrittweise aus und gibt alle Zwischenergebnisse zurück.
    Nützlich zur Diagnose von Startwertproblemen."""
    T8, T10, x4, x1, T3, T5 = map(float, z)

    values: Dict[str, float] = {}
    primary_variables = dict(
        zip(PRIMARY_VARIABLE_NAMES, [T8, T10, x4, x1, T3, T5])
    )
    stage = "initial"

    try:
        stage = "pressure_levels"
        p_low = water_p_sat_from_T(T10, Q=1.0)
        p_high = water_p_sat_from_T(T8, Q=0.0)
        values["p_low_Pa"] = p_low
        values["p_high_Pa"] = p_high

        if p_high <= p_low:
            raise ModelEvaluationError(
                f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa)."
            )

        stage = "solution_saturation_states"
        T4 = lp.T_sat_solution_from_p_x(p_high, x4)
        T1 = lp.T_sat_solution_from_p_x(p_low, x1)
        values["T4_K"] = T4
        values["T1_K"] = T1

        w4 = lp.w_libr_from_x(x4)
        w1 = lp.w_libr_from_x(x1)
        values["w4_LiBr"] = w4
        values["w1_LiBr"] = w1

        if not (w4 > w1 > 0.0):
            raise ModelEvaluationError(
                f"Konzentrationshierarchie verletzt: w4={w4:.6f}, w1={w1:.6f}."
            )

        stage = "cycle_scale"
        h3 = lp.h_solution_mass_kjkg(T3, x1)
        h4 = lp.h_solution_mass_kjkg(T4, x4)
        T7 = lp.T_sat_solution_from_p_x(p_high, x1) + inputs.desorber_vapor_superheat_K
        h7 = water_h_kjkg_PT(p_high, T7)
        h8 = water_h_kjkg_PQ(p_high, Q=0.0)
        h10 = water_h_kjkg_PQ(p_low, Q=1.0)
        h9 = h8
        refrigerant_throttle = water_throttle_state(p_low, h9)
        T9 = refrigerant_throttle["T9_K"]
        m1 = _resolve_cycle_scale(inputs, w4=w4, w1=w1, h9=h9, h10=h10, strict=True)
        values["h3_kJ_kg"] = h3
        values["h4_kJ_kg"] = h4
        values["h7_kJ_kg"] = h7
        values["h8_kJ_kg"] = h8
        values["h10_kJ_kg"] = h10
        values["m1_kg_s"] = m1
        values["T7_K"] = T7


        stage = "mass_flows"
        m3 = m1
        m4 = m3 * w1 / w4
        m7 = m1 - m4
        m10 = m7
        values["m3_kg_s"] = m3
        values["m7_kg_s"] = m7

        if m7 <= 0.0:
            raise ModelEvaluationError(f"m7={m7:.6f} kg/s nicht positiv.")

        stage = "solution_pump"
        h5 = lp.h_solution_mass_kjkg(T5, x4)
        h1 = lp.h_solution_mass_kjkg(T1, x1)
        rho1 = lp.rho_solution_mass(T1, x1)
        v1 = 1.0 / rho1
        W_sol_pump = m1 * v1 * (p_high - p_low) / 1000.0  # kW
        h2 = h1 + W_sol_pump / m1
        T2 = lp.T_from_h_x_mass(h2, x1)
        values.update({
            "h1_kJ_kg": h1,
            "h2_kJ_kg": h2,
            "W_sol_pump_kW": W_sol_pump,
            "h5_kJ_kg": h5,
            "T2_K": T2,
        })

        stage = "shex"
        Q_shex_hot = m4 * (h4 - h5)
        Q_shex_cold = m3 * (h3 - h2)
        values["Q_shex_hot_kW"] = Q_shex_hot
        values["Q_shex_cold_kW"] = Q_shex_cold
        values["deltaT_shex_1_K"] = T4 - T3
        values["deltaT_shex_2_K"] = T5 - T2

        if Q_shex_hot <= 0.0:
            raise ModelEvaluationError(f"Q_shex_hot={Q_shex_hot:.4f} kW nicht positiv.")
        if Q_shex_cold <= 0.0:
            raise ModelEvaluationError(f"Q_shex_cold={Q_shex_cold:.4f} kW nicht positiv.")

        lmtd_shex = counterflow_lmtd(hot_in=T4, hot_out=T5, cold_in=T2, cold_out=T3)
        values["LMTD_shex_K"] = lmtd_shex

        stage = "throttle_solution"
        h6 = h5
        flash = lp.flash_valve_state_5_to_6(
            p_out_pa=p_low,
            h5_kJkg=h5,
            m5_kg_s=m4,
            x5_libr_mol=x4,
        )
        T6 = flash["T6_K"]
        values["h6_kJ_kg"] = h6
        values["T6_K"] = T6
        for key, value in flash.items():
            values[f"flash_{key}"] = float(value)

        stage = "hot_side_coupling"
        Q_abs = m10 * h10 + m4 * h6 - m1 * h1
        values["Q_abs_kW"] = Q_abs
        if Q_abs <= 0.0:
            raise ModelEvaluationError(f"Q_abs={Q_abs:.4f} kW nicht positiv.")
        
        stage = "evaporator"
        Q_evap = m7 * (h10 - h9)
        if Q_evap <= 0.0:
            raise ModelEvaluationError(f"Q_evap={Q_evap:.4f} kW nicht positiv.")
        m17, T18 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=True)
        lmtd_evap = _counterflow_lmtd_mode(
            hot_in=inputs.T_17, hot_out=T18, cold_in=T9, cold_out=T10
        )

        values["Q_evap_kW"] = Q_evap
        values["m17_kg_s"] = m17
        values["T18_K"] = T18
        values["deltaT_abs_1_K"] = T18 - T9
        values["deltaT_abs_2_K"] = inputs.T_17 - T10
        values["LMTD_evap_K"] = lmtd_evap

        stage = "condenser"
        if inputs.uses_serial_condenser_to_absorber_routing:
            T15_in = _resolve_condenser_external_inlet_temperature(inputs)
        else:
            T13_in = _resolve_absorber_external_inlet_temperature(inputs)
            T14_seed = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
            T15_in = _resolve_condenser_external_inlet_temperature(inputs, T14_seed)

        Q_cond = m7 * (h7 - h8)
        T16 = T15_in + Q_cond / (inputs.m_15 * inputs.cp_w_kJkgK)
        values["Q_cond_kW"] = Q_cond
        values["T16_K"] = T16
        values["T15_K"] = T15_in
        values["deltaT_cond_1_K"] =T8 - T16
        values["deltaT_cond_2_K"] = T8 - T15_in

        if Q_cond <= 0.0:
            raise ModelEvaluationError(f"Q_cond={Q_cond:.4f} kW nicht positiv.")

        lmtd_cond = counterflow_lmtd(hot_in=T8, hot_out=T8, cold_in=T15_in, cold_out=T16)
        values["LMTD_cond_K"] = lmtd_cond

        stage = "throttle_refrigerant"
        values["h9_kJ_kg"] = h9
        values["T9_K"] = T9

        stage = "absorber"
        if inputs.uses_serial_condenser_to_absorber_routing:
            T13_in = _resolve_absorber_external_inlet_temperature(inputs, T16)
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        else:
            T13_in = _resolve_absorber_external_inlet_temperature(inputs)
            T14 = T13_in + Q_abs / (inputs.m_13 * inputs.cp_w_kJkgK)
        values["T13_K"] = T13_in
        values["T14_K"] = T14
        values["deltaT_abs_1_K"] = T6 - T14
        values["deltaT_abs_2_K"] = T1 - T13_in

        lmtd_abs = counterflow_lmtd(
            hot_in=T6, hot_out=T1, cold_in=T13_in, cold_out=T14
        )
        values["LMTD_abs_K"] = lmtd_abs

        stage = "desorber"
        Q_des = m4 * h4 + m7 * h7 - m1 * h3
        if Q_des <= 0.0:
            raise ModelEvaluationError(f"Q_des={Q_des:.4f} kW nicht positiv.")#
        values["Q_des_kW"] = Q_des
        T12 = inputs.T_11 - Q_des / (inputs.m_11 * inputs.cp_w_kJkgK)

        values["T12_K"] = T12
        values["deltaT_des_1_K"] = inputs.T_11 - T4
        values["deltaT_des_2_K"] = T12 - T7

        lmtd_des = counterflow_lmtd(
            hot_in=inputs.T_11, hot_out=T12, cold_in=T7, cold_out=T4
        )
        values["LMTD_des_K"] = lmtd_des

        return ModelTrace(
            primary_variables=primary_variables,
            values=values,
            stage=stage,
            success=True,
            error_type=None,
            error_message=None,
        )

    except Exception as exc:
        return ModelTrace(
            primary_variables=primary_variables,
            values=values,
            stage=stage,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Ausgabehilfen
# ---------------------------------------------------------------------------

def _is_absolute_temperature_key(key: str) -> bool:
    return key.startswith("T")


def _display_key(key: str) -> str:
    if key.endswith("_K") and _is_absolute_temperature_key(key):
        return f"{key[:-2]}_C"
    return key


def _display_value_and_unit(key: str, value: float) -> tuple[float, str]:
    if _is_absolute_temperature_key(key):
        return kelvin_to_celsius(value), "°C"
    if key.endswith("_Pa"):
        return value, "Pa"
    if key.endswith("_kg_s"):
        return value, "kg/s"
    if key.endswith("_kJ_kg"):
        return value, "kJ/kg"
    if key.endswith("_kW"):
        return value, "kW"
    if key.endswith("_K"):
        return value, "K"
    return value, "-"


def _format_state_line(state_id: str, state: Dict[str, float]) -> str:
    def maybe_temperature() -> str:
        if "T_K" not in state:
            return "-"
        return f"{kelvin_to_celsius(state['T_K']):.3f} °C"

    def maybe(key: str, fmt: str) -> str:
        return fmt.format(state[key]) if key in state else "-"

    return (
        f"{state_id:>3s} | "
        f"T = {maybe_temperature()} | "
        f"p = {maybe('p_Pa', '{:.3e}')} Pa | "
        f"m = {maybe('m_kg_s', '{:.6f}')} kg/s | "
        f"h = {maybe('h_kJ_kg', '{:.6f}')} kJ/kg | "
        f"x = {maybe('x_LiBr_mol', '{:.6f}')} | "
        f"w = {maybe('w_LiBr', '{:.6f}')}"
    )


def print_trace(trace: ModelTrace) -> None:
    print("=" * 110)
    print("Startwert-Trace / Modellpunkt-Trace")
    print("=" * 110)

    print("Primäre Variablen")
    for key, value in trace.primary_variables.items():
        if _is_absolute_temperature_key(key):
            display_value, unit = kelvin_to_celsius(value), "°C"
        else:
            display_value, unit = value, "-"
        print(f"  {key:12s}: {display_value:14.6f} {unit}")
    print()

    print(f"Auswertungsstatus : {trace.success}")
    print(f"Letzte Stufe      : {trace.stage}")
    if trace.error_type is not None:
        print(f"Fehlertyp         : {trace.error_type}")
    if trace.error_message is not None:
        print(f"Fehlermeldung     : {trace.error_message}")
    print()

    print("Berechnete Größen bis zum Abbruch")
    for key, value in trace.values.items():
        display_key = _display_key(key)
        display_value, unit = _display_value_and_unit(key, value)
        print(f"  {display_key:28s}: {display_value:14.6f} {unit}")
    print("=" * 110)


def print_summary(result: AWTResult) -> None:
    print("=" * 110)
    print("AWT-Simulation – Ergebnisübersicht (7 primäre Unbekannte)")
    print("=" * 110)

    print("Solver-Informationen")
    print(f"  Erfolg                 : {result.solve_info.success}")
    print(f"  Status                 : {result.solve_info.status}")
    print(f"  Nachricht              : {result.solve_info.message}")
    print(f"  Funktionsauswertungen  : {result.solve_info.nfev}")
    print(f"  least_squares cost     : {result.solve_info.cost:.6e}")
    print(f"  Norm skalierter Residuen: {result.solve_info.scaled_residual_norm:.6e}")
    if result.solve_info.raw_residual_norm is None:
        print("  Norm Rohresiduen       : n/a (Endpunkt nicht physikalisch auswertbar)")
    else:
        print(f"  Norm Rohresiduen       : {result.solve_info.raw_residual_norm:.6e}")
    print(f"  Endpunkt physikalisch auswertbar: {result.solve_info.final_point_evaluable}")
    if result.solve_info.final_evaluation_error is not None:
        print(f"  Endpunkt-Auswertungsfehler     : {result.solve_info.final_evaluation_error}")
    print()

    if not result.solve_info.final_point_evaluable:
        print("Keine physikalisch auswertbare Modelllösung vorhanden.")
        print("=" * 110)
        return

    print("Primäre Solvervariablen")
    for name in PRIMARY_VARIABLE_NAMES:
        if _is_absolute_temperature_key(name):
            display_value, unit = kelvin_to_celsius(result.primary_variables[name]), "°C"
        else:
            display_value, unit = result.primary_variables[name], "-"
        print(f"  {name:8s}: {display_value:12.6f} {unit}")
    print()

    print("Wärmeströme [kW]")
    for key, value in result.heat_flows_kW.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("KPIs")
    for key, value in result.kpis.items():
        unit = "K" if key.endswith("_K") else "-"
        print(f"  {key:12s}: {value:12.6f} {unit}")
    print()

    print("Pumpenarbeiten [kW]")
    for key, value in result.pump_work_W.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("LMTD [K]")
    for key, value in result.lmtd_K.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("Flash-Drossel 5 -> 6 (lokale Unterrechnung; T6 wird im Desorber verwendet)")
    for key, value in result.flash_outputs.items():
        display_key = _display_key(key)
        display_value, unit = _display_value_and_unit(key, value)
        print(f"  {display_key:28s}: {display_value:14.6f} {unit}")
    print()

    print("Residuen")
    for name in RESIDUAL_NAMES:
        raw = result.residuals_raw[name]
        scaled = result.residuals_scaled[name]
        print(f"  {name:18s}: raw = {raw:14.6e} | scaled = {scaled:14.6e}")
    print()

    print("Diagnostik")
    for key, value in result.diagnostics.items():
        display_key = _display_key(key)
        display_value, unit = _display_value_and_unit(key, value)
        print(f"  {display_key:28s}: {display_value:14.6f} {unit}")
    print()

    print("Plausibilitätschecks")
    for key, value in result.checks.items():
        print(f"  {key:35s}: {value}")
    print()

    print("Zustände")
    for state_id in sorted(result.states, key=lambda s: (len(s), s)):
        print(_format_state_line(state_id, result.states[state_id]))
    print()

    print("Validitätsmeldungen")
    for msg in result.validity_messages:
        print(f"  - {msg}")

    print("=" * 110)


__all__ = [
    "AKMInputs",
    "AWTResult",
    "SolveInfo",
    "ModelTrace",
    "PRIMARY_VARIABLE_NAMES",
    "RESIDUAL_NAMES",
    "PRIMARY_TEMPERATURE_INDICES",
    "kelvin_to_celsius",
    "celsius_to_kelvin",
    "primary_temperatures_C_to_K",
    "primary_temperatures_K_to_C",
    "bounds",
    "initial_guess",
    "evaluate_model",
    "trace_model",
    "residual_vector",
    "solve_awt",
    "print_trace",
    "print_summary",
]
