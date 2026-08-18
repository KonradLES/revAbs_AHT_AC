"""AWT-Simulation mit 7 primären Unbekannten.

Die Absorber-Spezifikation ist variabel:
- absorber_spec_mode = "m11": m11_spec wird vorgegeben, T12 wird berechnet
- absorber_spec_mode = "T12": T12_spec_C wird vorgegeben, m11 wird berechnet

Die Kreislaufskalierung ist variabel:
- cycle_scale_spec_mode = "m6": m6_spec wird vorgegeben
- cycle_scale_spec_mode = "Qabs": Qabs_spec_kW wird vorgegeben, m6 wird berechnet

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
z = [T8, T10, x3, x6, x20, T2, T4]

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

PRIMARY_VARIABLE_NAMES = ["T8", "T10", "x3", "x6", "x20", "T2", "T4"]
RESIDUAL_NAMES = [
    "R1_SHEX_energy",
    "R2_SHEX_pinch",
    "R3_preabs_energy",
    "R4_desorber_pinch",
    "R5_condenser_pinch",
    "R6_evaporator_pinch",
    "R7_absorber_pinch",
]

PRIMARY_TEMPERATURE_INDICES = (0, 1, 5, 6)


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
class AWTInputs:
    # Externe Einlasstemperaturen [°C]
    T_11_C: float
    T_13_C: float | None
    T_15_C: float | None
    T_17_C: float

    # Pinch-Temperaturdifferenzen [K]
    dT_min_shex: float
    dT_min_des:  float
    dT_min_cond: float
    dT_min_evap: float
    dT_min_abs:  float

    _: KW_ONLY

    # Externe Wärmesenken/-quellen von Desorber und Verdampfer:
    # - "parallel": Standardfall mit separater T_13- und T_15-Vorgabe
    # - "series_desorber_to_evaporator": intern gilt T15 = T14
    # - "series_evaporator_to_desorber": intern gilt T13 = T16
    desorber_evaporator_routing_mode: str = "parallel"

    # Spezifikation der Kreislaufskalierung:
    # - "m6": m6_spec wird vorgegeben
    # - "Qabs": Qabs_spec_kW wird vorgegeben, m6 wird berechnet
    cycle_scale_spec_mode: str = "m6"
    m6_spec: float | None = None
    Qabs_spec_kW: float | None = None

    # Spezifikation des externen Absorberstroms:
    # - "m11": m11_spec wird vorgegeben, T12 wird berechnet
    # - "T12": T12_spec_C wird vorgegeben, m11 wird berechnet
    absorber_spec_mode: str = "m11"
    m11_spec: float | None = None
    T12_spec_C: float | None = None

    # Spezifikation des externen Desorberstroms:
    # - "m13": m13_spec wird vorgegeben, T14 wird berechnet
    # - "T14": T14_spec_C wird vorgegeben, m13 wird berechnet
    desorber_spec_mode: str = "m13"
    m13_spec: float | None = None
    T14_spec_C: float | None = None

    # Spezifikation des externen Verdampferstroms:
    # - "m15": m15_spec wird vorgegeben, T16 wird berechnet
    # - "T16": T16_spec_C wird vorgegeben, m15 wird berechnet
    evaporator_spec_mode: str = "m15"
    m15_spec: float | None = None
    T16_spec_C: float | None = None

    # Spezifikation des externen Kondensatorstroms:
    # - "m17": m17_spec wird vorgegeben, T18 wird berechnet
    # - "T18": T18_spec_C wird vorgegeben, m17 wird berechnet
    condenser_spec_mode: str = "m17"
    m17_spec: float | None = None
    T18_spec_C: float | None = None

    # Externe Fluide: Wasser
    cp_w_kJkgK: float = 4.18

    # Desorberaustritt des Kältemitteldampfes
    # Default: gesättigter Dampf auf low-pressure-Niveau
    desorber_vapor_superheat_K: float = 0.0

    # Solver
    solver_tol: float = 1.0e-9
    max_nfev: int = 5000
    penalty_level: float = 1.0e6

    def __post_init__(self) -> None:
        # Pinch-Werte müssen positiv sein
        for name, val in [
            ("dT_min_shex", self.dT_min_shex),
            ("dT_min_des",  self.dT_min_des),
            ("dT_min_cond", self.dT_min_cond),
            ("dT_min_evap", self.dT_min_evap),
            ("dT_min_abs",  self.dT_min_abs),
        ]:
            if val <= 0.0:
                raise ValueError(f"{name} muss positiv sein, ist aber {val}.")
        
        if self.desorber_evaporator_routing_mode not in {
            "parallel",
            "series_desorber_to_evaporator",
            "series_evaporator_to_desorber",
        }:
            raise ValueError(
                "desorber_evaporator_routing_mode muss 'parallel', "
                "'series_desorber_to_evaporator' oder 'series_evaporator_to_desorber' sein."
            )

        if self.desorber_evaporator_routing_mode == "parallel":
            if self.T_13_C is None:
                raise ValueError(
                    "Bei desorber_evaporator_routing_mode='parallel' muss T_13_C vorgegeben werden."
                )
            if self.T_15_C is None:
                raise ValueError(
                    "Bei desorber_evaporator_routing_mode='parallel' muss T_15_C vorgegeben werden."
                )
        elif self.desorber_evaporator_routing_mode == "series_desorber_to_evaporator":
            if self.T_13_C is None:
                raise ValueError(
                    "Bei desorber_evaporator_routing_mode='series_desorber_to_evaporator' muss "
                    "T_13_C vorgegeben werden."
                )
            if self.T_15_C is not None:
                raise ValueError(
                    "Bei desorber_evaporator_routing_mode='series_desorber_to_evaporator' darf T_15_C "
                    "nicht gesetzt sein; es gilt intern T15 = T14."
                )
        else:
            if self.T_15_C is None:
                raise ValueError(
                    "Bei desorber_evaporator_routing_mode='series_evaporator_to_desorber' muss "
                    "T_15_C vorgegeben werden; es gilt intern T13 = T16."
                )
            if self.T_13_C is not None:
                raise ValueError(
                    "Bei desorber_evaporator_routing_mode='series_evaporator_to_desorber' darf T_13_C "
                    "nicht gesetzt sein; es gilt intern T13 = T16."
                )

        if self.cycle_scale_spec_mode not in {"m6", "Qabs"}:
            raise ValueError("cycle_scale_spec_mode muss 'm6' oder 'Qabs' sein.")
        if self.cycle_scale_spec_mode == "m6":
            if self.m6_spec is None:
                raise ValueError("Bei cycle_scale_spec_mode='m6' muss m6_spec vorgegeben werden.")
            if self.Qabs_spec_kW is not None:
                raise ValueError(
                    "Bei cycle_scale_spec_mode='m6' darf Qabs_spec_kW nicht gesetzt sein."
                )
            if self.m6_spec <= 0.0:
                raise ValueError("Bei cycle_scale_spec_mode='m6' muss m6_spec > 0 gelten.")
        else:
            if self.Qabs_spec_kW is None:
                raise ValueError(
                    "Bei cycle_scale_spec_mode='Qabs' muss Qabs_spec_kW vorgegeben werden."
                )
            if self.m6_spec is not None:
                raise ValueError(
                    "Bei cycle_scale_spec_mode='Qabs' darf m6_spec nicht gesetzt sein."
                )
            if self.Qabs_spec_kW <= 0.0:
                raise ValueError("Bei cycle_scale_spec_mode='Qabs' muss Qabs_spec_kW > 0 gelten.")

        if self.absorber_spec_mode not in {"m11", "T12"}:
            raise ValueError("absorber_spec_mode muss 'm11' oder 'T12' sein.")
        if self.absorber_spec_mode == "m11":
            if self.m11_spec is None:
                raise ValueError("Bei absorber_spec_mode='m11' muss m11_spec vorgegeben werden.")
            if self.T12_spec_C is not None:
                raise ValueError(
                    "Bei absorber_spec_mode='m11' darf T12_spec_C nicht gesetzt sein."
                )
            if self.m11_spec <= 0.0:
                raise ValueError("Bei absorber_spec_mode='m11' muss m11_spec > 0 gelten.")
        else:
            if self.T12_spec_C is None:
                raise ValueError("Bei absorber_spec_mode='T12' muss T12_spec_C vorgegeben werden.")
            if self.m11_spec is not None:
                raise ValueError(
                    "Bei absorber_spec_mode='T12' darf m11_spec nicht gesetzt sein."
                )
            if self.T12_spec_C <= self.T_11_C:
                raise ValueError(
                    "Bei absorber_spec_mode='T12' muss T12_spec_C > T_11_C gelten."
                )
            
        if self.desorber_spec_mode not in {"m13", "T14"}:
            raise ValueError("desorber_spec_mode muss 'm13' oder 'T14' sein.")
        if self.desorber_spec_mode == "m13":
            if self.m13_spec is None:
                raise ValueError("Bei desorber_spec_mode='m13' muss m13_spec vorgegeben werden.")
            if self.T14_spec_C is not None:
                raise ValueError(
                    "Bei desorber_spec_mode='m13' darf T14_spec_C nicht gesetzt sein."
                )
            if self.m13_spec <= 0.0:
                raise ValueError("Bei desorber_spec_mode='m13' muss m13_spec > 0 gelten.")
        else:
            if self.T14_spec_C is None:
                raise ValueError("Bei desorber_spec_mode='T14' muss T14_spec_C vorgegeben werden.")
            if self.m13_spec is not None:
                raise ValueError(
                    "Bei desorber_spec_mode='T14' darf m13_spec nicht gesetzt sein."
                )
            if self.T14_spec_C >= self.T_13_C:
                raise ValueError(
                    "Bei desorber_spec_mode='T14' muss T14_spec_C < T_13_C gelten."
                )

        if self.evaporator_spec_mode not in {"m15", "T16"}:
            raise ValueError("evaporator_spec_mode muss 'm15' oder 'T16' sein.")
        if self.evaporator_spec_mode == "m15":
            if self.m15_spec is None:
                raise ValueError("Bei evaporator_spec_mode='m15' muss m15_spec vorgegeben werden.")
            if self.T16_spec_C is not None:
                raise ValueError(
                    "Bei evaporator_spec_mode='m15' darf T16_spec_C nicht gesetzt sein."
                )
            if self.m15_spec <= 0.0:
                raise ValueError("Bei evaporator_spec_mode='m15' muss m15_spec > 0 gelten.")
        else:
            if self.T16_spec_C is None:
                raise ValueError("Bei evaporator_spec_mode='T16' muss T16_spec_C vorgegeben werden.")
            if self.m15_spec is not None:
                raise ValueError(
                    "Bei evaporator_spec_mode='T16' darf m15_spec nicht gesetzt sein."
                )
            if self.T16_spec_C >= self.T_15_C:
                raise ValueError(
                    "Bei evaporator_spec_mode='T16' muss T16_spec_C < T_15_C gelten."
                )
                                    
        if self.condenser_spec_mode not in {"m17", "T18"}:
            raise ValueError("condenser_spec_mode muss 'm17' oder 'T18' sein.")
        if self.condenser_spec_mode == "m17":
            if self.m17_spec is None:
                raise ValueError("Bei condenser_spec_mode='m17' muss m17_spec vorgegeben werden.")
            if self.T18_spec_C is not None:
                raise ValueError(
                    "Bei condenser_spec_mode='m17' darf T18_spec_C nicht gesetzt sein."
                )
            if self.m17_spec <= 0.0:
                raise ValueError("Bei condenser_spec_mode='m17' muss m17_spec > 0 gelten.")
        else:
            if self.T18_spec_C is None:
                raise ValueError("Bei condenser_spec_mode='T18' muss T18_spec_C vorgegeben werden.")
            if self.m17_spec is not None:
                raise ValueError(
                    "Bei condenser_spec_mode='T18' darf m17_spec nicht gesetzt sein."
                )
            if self.T18_spec_C <= self.T_17_C:
                raise ValueError(
                    "Bei condenser_spec_mode='T18' muss T18_spec_C > T_17_C gelten."
                )

    @property
    def T_11(self) -> float:
        return celsius_to_kelvin(self.T_11_C)

    @property
    def T12_spec(self) -> float:
        if self.T12_spec_C is None:
            raise AttributeError("T12_spec_C ist für diese Spezifikation nicht gesetzt.")
        return celsius_to_kelvin(self.T12_spec_C)

    @property
    def T_13(self) -> float:
        if self.T_13_C is None:
            raise AttributeError(
                "T_13_C ist für die gewählte Routing-Variante nicht gesetzt."
            )
        return celsius_to_kelvin(self.T_13_C)

    @property
    def T14_spec(self) -> float:
        if self.T14_spec_C is None:
            raise AttributeError("T14_spec_C ist für diese Spezifikation nicht gesetzt.")
        return celsius_to_kelvin(self.T14_spec_C)
    
    @property
    def T_15(self) -> float:
        if self.T_15_C is None:
            raise AttributeError(
                "T_15_C ist für die gewählte Routing-Variante nicht gesetzt."
            )
        return celsius_to_kelvin(self.T_15_C)

    @property
    def T16_spec(self) -> float:
        if self.T16_spec_C is None:
            raise AttributeError("T16_spec_C ist für diese Spezifikation nicht gesetzt.")
        return celsius_to_kelvin(self.T16_spec_C)

    @property
    def T_17(self) -> float:
        return celsius_to_kelvin(self.T_17_C)
    
    @property
    def T18_spec(self) -> float:
        if self.T18_spec_C is None:
            raise AttributeError("T18_spec_C ist für diese Spezifikation nicht gesetzt.")
        return celsius_to_kelvin(self.T18_spec_C)
    
    @property
    def uses_serial_desorber_to_evaporator_routing(self) -> bool:
        return self.desorber_evaporator_routing_mode == "series_desorber_to_evaporator"

    @property
    def uses_serial_evaporator_to_desorber_routing(self) -> bool:
        return self.desorber_evaporator_routing_mode == "series_evaporator_to_desorber"

    @property
    def uses_any_serial_desorber_evaporator_routing(self) -> bool:
        return self.desorber_evaporator_routing_mode in {
            "series_desorber_to_evaporator",
            "series_evaporator_to_desorber",
        }

    @property
    def evaporator_temperature_reference(self) -> float:
        """Referenztemperatur für Startwerte und Schranken des Verdampfers."""
        if self.uses_serial_desorber_to_evaporator_routing:
            return self.T_13
        return self.T_15

    @property
    def desorber_temperature_reference(self) -> float:
        """Referenztemperatur für Startwerte und Schranken des Desorbers."""
        if self.uses_serial_evaporator_to_desorber_routing:
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
    UA_conversion: Dict[str, float] 
    pinch_temperatures_K: Dict[str, float] 
    compositions: Dict[str, float]
    flash_outputs: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    exergy_kW: Dict[str, float]
    checks: Dict[str, bool]
    validity_messages: List[str]


@dataclass(frozen=True)
class AWTResult:
    inputs: AWTInputs
    solve_info: SolveInfo
    primary_variables: Dict[str, float]
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    pump_work_W: Dict[str, float]
    UA_conversion: Dict[str, float] 
    pinch_temperatures_K: Dict[str, float] 
    compositions: Dict[str, float]
    flash_outputs: Dict[str, float]
    residuals_raw: Dict[str, float]
    residuals_scaled: Dict[str, float]
    diagnostics: Dict[str, float]
    exergy_kW: Dict[str, float]
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


def water_s_kjkgK_PT(P_pa: float, T_K: float) -> float:
    return CP.PropsSI("S", "P", P_pa, "T", T_K, "Water") / 1000.0


def water_h_kjkg_PQ(P_pa: float, Q: float) -> float:
    return CP.PropsSI("H", "P", P_pa, "Q", Q, "Water") / 1000.0


def water_s_kjkgK_PQ(P_pa: float, Q: float) -> float:
    return CP.PropsSI("S", "P", P_pa, "Q", Q, "Water") / 1000.0

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
def smooth_min(a: float, b: float, k: float = 50.0) -> float:
    """Glatte Approximation von min(a, b), numerisch stabil.

    Für k -> unendlich konvergiert smooth_min gegen min(a, b) exakt.
    Bleibt überall stetig differenzierbar, auch bei a == b.
    """
    lo = min(a, b)
    diff = abs(a - b)
    return lo - math.log1p(math.exp(-k * diff)) / k

def lmtd(delta_T_1: float, delta_T_2: float) -> float:
    """Strenge LMTD: wirft ModelEvaluationError für ΔT <= 0."""
    if delta_T_1 <= 0.0 or delta_T_2 <= 0.0:
        raise ModelEvaluationError(
            f"LMTD undefiniert, weil delta_T_1={delta_T_1:.6f} K "
            f"oder delta_T_2={delta_T_2:.6f} K nicht positiv ist."
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


def _penalty_vector(size: int, level: float) -> np.ndarray:
    return np.full(size, level, dtype=float)


def _residual_scales(m6: float) -> np.ndarray:
    return np.array(
        [
            100.0,  # R1_SHEX_energy, kW
            1.0,    # R2_SHEX_pinch, K
            100.0,  # R3_preabs_energy, kW
            1.0,    # R4_desorber_pinch, K
            1.0,    # R5_condenser_pinch, K
            1.0,    # R6_evaporator_pinch, K
            1.0,    # R7_absorber_pinch, K
        ],
        dtype=float,
    )

def _state_dict(
    T_K: float,
    p_Pa: float | None = None,
    m_kg_s: float | None = None,
    h_kJ_kg: float | None = None,
    x_LiBr_mol: float | None = None,
    w_LiBr: float | None = None,
    s: float | None = None,
    e: float | None = None,
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
    if s is not None:
        state["s"] = float(s)
    if e is not None:
        state["e"] = float(e)
    return state


def _calculate_kpis(
    *, Q_abs: float, Q_evap: float, Q_des: float, T12: float, T15: float, m6: float, m7: float
) -> Dict[str, float]:
    denominator_cop = Q_evap + Q_des
    cop = float("nan") if abs(denominator_cop) <= 1.0e-12 else Q_abs / denominator_cop
    fr = float("nan") if abs(m7) <= 1.0e-12 else m6 / m7

    return {
        "COP": cop,
        "GTL_K": T12 - T15,
        "FR": fr,
    }


def _resolve_cycle_scale(
    inputs: AWTInputs, *, w3: float, w6: float, h3: float, h4: float, h10: float, strict: bool
) -> float:
    """Löst die Kreislaufskalierung auf den gepumpten Lösungsmassenstrom m6 auf."""
    if inputs.cycle_scale_spec_mode == "m6":
        m6 = float(inputs.m6_spec)  # durch __post_init__ abgesichert
        if strict and m6 <= 0.0:
            raise ModelEvaluationError("Gepumpter Lösungsmassenstrom m6 muss positiv sein.")
        return m6

    w3_balance = w3 if strict else max(w3, 1.0e-9)
    ratio = w6 / w3_balance
    denominator = (ratio - 1.0) * h10 + h4 - ratio * h3

    if strict:
        if abs(denominator) <= 1.0e-12:
            raise ModelEvaluationError(
                "Kreislaufskalierung aus Q_abs nicht möglich, weil der Nenner nahezu null ist."
            )
        m6 = float(inputs.Qabs_spec_kW) / denominator
        if m6 <= 0.0:
            raise ModelEvaluationError(
                f"Berechneter Lösungsmassenstrom m6 nicht positiv: m6={m6:.6f} kg/s."
            )
        return m6

    denominator_safe = denominator
    if abs(denominator_safe) <= 1.0e-12:
        denominator_safe = 1.0e-12 if denominator_safe >= 0.0 else -1.0e-12
    return float(inputs.Qabs_spec_kW) / denominator_safe

def _resolve_absorber_external_stream(
    inputs: AWTInputs, Q_abs: float, *, strict: bool
) -> tuple[float, float]:
    """Löst die Absorber-Spezifikation auf interne Arbeitsgrößen auf.

    Rückgabe
    --------
    (m11, T12)
    """
    if inputs.absorber_spec_mode == "m11":
        m11 = float(inputs.m11_spec)  # durch __post_init__ abgesichert
        if strict:
            T12 = heating_outlet_temperature(inputs.T_11, Q_abs, m11, inputs.cp_w_kJkgK)
        else:
            T12 = inputs.T_11 + Q_abs / (m11 * inputs.cp_w_kJkgK)
        return m11, T12

    T12 = inputs.T12_spec
    delta_T = T12 - inputs.T_11
    if strict:
        if delta_T <= 0.0:
            raise ModelEvaluationError(
                "Für absorber_spec_mode='T12' muss T12 > T11 gelten."
            )
        if inputs.cp_w_kJkgK <= 0.0:
            raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    m11 = Q_abs / (inputs.cp_w_kJkgK * delta_T)
    return m11, T12

def _resolve_desorber_external_stream(
    inputs: AWTInputs, Q_des: float, *, strict: bool, T13: float 
) -> tuple[float, float]:
    """Löst die Desorber-Spezifikation auf interne Arbeitsgrößen auf.

    Rückgabe
    --------
    (m13, T14)
    """
    if inputs.desorber_spec_mode == "m13":
        m13 = float(inputs.m13_spec)  # durch __post_init__ abgesichert
        if strict:
            T14 = cooling_outlet_temperature(T13, Q_des, m13, inputs.cp_w_kJkgK)
        else:
            T14 = T13 - Q_des / (m13 * inputs.cp_w_kJkgK)
        return m13, T14

    T14 = inputs.T14_spec
    delta_T = T13 - T14
    if strict:
        if delta_T <= 0.0:
            raise ModelEvaluationError(
                "Für desorber_spec_mode='T14' muss T13 > T14 gelten."
            )
        if inputs.cp_w_kJkgK <= 0.0:
            raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    m13 = Q_des / (inputs.cp_w_kJkgK * delta_T)
    return m13, T14

def _resolve_evaporator_external_stream(
    inputs: AWTInputs, Q_evap: float, *, strict: bool, T15: float
) -> tuple[float, float]:
    """Löst die Verdampfer-Spezifikation auf interne Arbeitsgrößen auf.

    Rückgabe
    --------
    (m15, T16)
    """
    if inputs.evaporator_spec_mode == "m15":
        m15 = float(inputs.m15_spec)  # durch __post_init__ abgesichert
        if strict:
            T16 = cooling_outlet_temperature(T15, Q_evap, m15, inputs.cp_w_kJkgK)
        else:
            T16 = T15 - Q_evap / (m15 * inputs.cp_w_kJkgK)
        return m15, T16

    T16 = inputs.T16_spec
    delta_T = T15 - T16
    if strict:
        if delta_T <= 0.0:
            raise ModelEvaluationError(
                "Für evaporator_spec_mode='T16' muss T15 > T16 gelten."
            )
        if inputs.cp_w_kJkgK <= 0.0:
            raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    m15 = Q_evap / (inputs.cp_w_kJkgK * delta_T)
    return m15, T16

def _resolve_condenser_external_stream(
    inputs: AWTInputs, Q_cond: float, *, strict: bool
) -> tuple[float, float]:
    """Löst die Kondensator-Spezifikation auf interne Arbeitsgrößen auf.

    Rückgabe
    --------
    (m17, T18)
    """
    if inputs.condenser_spec_mode == "m17":
        m17 = float(inputs.m17_spec)  # durch __post_init__ abgesichert
        if strict:
            T18 = heating_outlet_temperature(inputs.T_17, Q_cond, m17, inputs.cp_w_kJkgK)
        else:
            T18 = inputs.T_17 + Q_cond / (m17 * inputs.cp_w_kJkgK)
        return m17, T18

    T18 = inputs.T18_spec
    delta_T = T18 - inputs.T_17
    if strict:
        if delta_T <= 0.0:
            raise ModelEvaluationError(
                "Für condenser_spec_mode='T18' muss T18 > T17 gelten."
            )
        if inputs.cp_w_kJkgK <= 0.0:
            raise ModelEvaluationError("Externer Wärmekapazitätsstrom muss positiv sein.")
    m17 = Q_cond / (inputs.cp_w_kJkgK * delta_T)
    return m17, T18


def _resolve_evaporator_external_inlet_temperature(inputs: AWTInputs, T14: float | None = None) -> float:
    """Löst die externe Verdampfereinlasstemperatur auf.

    - parallel: T15 wird aus den Inputs gelesen
    - series_desorber_to_evaporator: T15 entspricht dem externen Desorberaustritt T14
    - series_evaporator_to_desorber: T15 bleibt externer Input
    """
    if inputs.uses_serial_desorber_to_evaporator_routing:
        if T14 is None:
            raise ModelEvaluationError("Für series_desorber_to_evaporator muss T14 bekannt sein.")
        return T14
    return inputs.T_15


def _resolve_desorber_external_inlet_temperature(inputs: AWTInputs, T16: float | None = None) -> float:
    """Löst die externe Desorbereinlasstemperatur auf.

    - parallel: T13 wird aus den Inputs gelesen
    - series_desorber_to_evaporator: T13 bleibt externer Input
    - series_evaporator_to_desorber: T13 entspricht dem externen Verdampferaustritt T16
    """
    if inputs.uses_serial_evaporator_to_desorber_routing:
        if T16 is None:
            raise ModelEvaluationError("Für series_evaporator_to_desorber muss T16 bekannt sein.")
        return T16
    return inputs.T_13


# ---------------------------------------------------------------------------
# Solver-Hilfsfunktionen
# ---------------------------------------------------------------------------

def initial_guess(inputs: AWTInputs) -> np.ndarray:
    """Heuristische Startwerte für den 8-dimensionalen Solvervektor."""
    T_evap_ref = inputs.evaporator_temperature_reference
    T_des_ref = inputs.desorber_temperature_reference
    return np.array(
        [
            inputs.T_17 + 8.0,      # T8
            T_evap_ref - 8.0,       # T10
            0.23,                   # x3
            0.30,                   # x6
            0.27,                   # x20
            inputs.T_11 + 12.0,     # T2
            T_des_ref - 18.0,       # T4
        ],
        dtype=float,
    )


def bounds(inputs: AWTInputs) -> Tuple[np.ndarray, np.ndarray]:
    T_evap_ref = inputs.evaporator_temperature_reference
    T_des_ref = inputs.desorber_temperature_reference
    lower = np.array(
        [
            inputs.T_17 + 0.5,  # T8
            inputs.T_17 + 5.0,  # T10
            0.05,               # x3
            0.08,               # x6
            0.05,               # x20
            inputs.T_17 + 1.0,  # T2
            inputs.T_17 + 1.0,  # T4
        ],
        dtype=float,
    )
    upper = np.array(
        [
            min(T_des_ref - 1.0, 420.0),   # T8
            min(T_evap_ref - 0.5, 500.0),   # T10
            0.34,                           # x3
            0.39,                           # x6
            0.39,                           # x20
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


def _evaluate_model_common(z: np.ndarray, inputs: AWTInputs, *, strict: bool) -> ModelEvaluation:
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
    T8, T10, x3, x6, x20, T2, T4 = map(float, z)

    # ------------------------------------------------------------------
    # 1) Druckniveaus des Kältemittels
    # ------------------------------------------------------------------
    p_low = water_p_sat_from_T(T8, Q=0.0)
    p_high = water_p_sat_from_T(T10, Q=1.0)
    if p_high <= p_low:
        if strict:
            raise ModelEvaluationError(f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa).")
        pen = (p_low - p_high + 100.0) / 100.0
        raise _SoftResidualVector(np.full(len(RESIDUAL_NAMES), pen, dtype=float))

    # ------------------------------------------------------------------
    # 2) Gesättigte Lösungszustände, Konzentrationen und frühe Stoffgrößen
    # ------------------------------------------------------------------
    T3 = lp.T_sat_solution_from_p_x(p_high, x3)
    T6 = lp.T_sat_solution_from_p_x(p_low, x6)
    T20 = lp.T_sat_solution_from_p_x(p_high, x20)

    w3 = lp.w_libr_from_x(x3)
    w6 = lp.w_libr_from_x(x6)
    w20 = lp.w_libr_from_x(x20)

    if strict and not (w6 > w3 > 0.0):
        raise ModelEvaluationError(
            f"Konzentrationshierarchie verletzt: w6={w6:.6f}, w3={w3:.6f}. Erwartet wird w6 > w3."
        )
    if strict and not (w6 > w20 > w3):
        raise ModelEvaluationError(
            f"Vorabsorptionszustand unplausibel: w6={w6:.6f}, w20={w20:.6f}, w3={w3:.6f}."
            f" Erwartet wird w6 > w20 > w3."
        )

    h3 = lp.h_solution_mass_kjkg(T3, x3)
    h4 = lp.h_solution_mass_kjkg(T4, x6)
    h10 = water_h_kjkg_PQ(p_high, Q=1.0)

    # ------------------------------------------------------------------
    # 3) Kreislaufskalierung und Massenströme
    # ------------------------------------------------------------------
    m6 = _resolve_cycle_scale(inputs, w3=w3, w6=w6, h3=h3, h4=h4, h10=h10, strict=strict)
    m5 = m4 = m6
    w3_for_balance = w3 if strict else max(w3, 1.0e-9)
    m3 = m4 * w6 / w3_for_balance
    m2 = m1 = m3
    m7 = m8 = m9 = m10 = m3 - m6

    if strict and m7 <= 0.0:
        raise ModelEvaluationError(f"Kältemittelmassenstrom nicht positiv: m7={m7:.6f} kg/s.")

    if strict:
        if w20 <= 0.0:
            raise ModelEvaluationError(f"w20 nicht positiv: w20={w20:.6f}.")
        if m10 <= 0.0:
            raise ModelEvaluationError(f"m10 nicht positiv: m10={m10:.6f} kg/s.")


    w20_safe = w20 if strict else max(w20, 1.0e-12)
    if strict:
        m10_safe = m10
    else:
        m10_safe = m10 if abs(m10) > 1.0e-12 else (1.0e-12 if m10 >= 0.0 else -1.0e-12)

    # LiBr-Bilanz der adiabaten Vorabsorption wird algebraisch erfüllt:
    #     m4 * w6 = m20 * w20
    # beta ist dadurch eine abgeleitete Größe und keine primäre Solvervariable mehr.
    m20 = m4 * w6 / w20_safe
    m19 = m20 - m4
    beta = m19 / m10_safe
    m21 = m10 - m19

    if strict and not (0.0 <= beta <= 1.0):
        raise ModelEvaluationError(
            f"Berechneter Vorabsorptionsanteil beta außerhalb [0,1]: beta={beta:.6f}."
        )

    if strict and m21 < 0.0:
        raise ModelEvaluationError(f"m21 negativ: {m21:.6f} kg/s.")

    # ------------------------------------------------------------------
    # 4) Lösungsenthalpien und Lösungspumpe 6 -> 5
    # ------------------------------------------------------------------
    h2 = lp.h_solution_mass_kjkg(T2, x3)
    h6 = lp.h_solution_mass_kjkg(T6, x6)

    rho6 = lp.rho_solution_mass(T6, x6)
    v6 = 1.0 / rho6
    W_sol_pump = m6 * v6 * (p_high - p_low) / 1000.0  # kW
    h5 = h6 + W_sol_pump / m6
    if strict:
        T5 = lp.T_from_h_x_mass(h5, x6)
    else:
        try:
            T5 = lp.T_from_h_x_mass(h5, x6)
        except Exception:
            T5 = T6

    # ------------------------------------------------------------------
    # 5) Lösungswärmeübertrager (SHEX): 3 -> 2 und 5 -> 4
    # ------------------------------------------------------------------
    Q_shex_hot = m3 * (h3 - h2)
    Q_shex_cold = m4 * (h4 - h5)
    if strict and Q_shex_hot <= 0.0:
        raise ModelEvaluationError(f"Q_shex_hot nicht positiv: {Q_shex_hot:.6f} kW.")
    if strict and Q_shex_cold <= 0.0:
        raise ModelEvaluationError(f"Q_shex_cold nicht positiv: {Q_shex_cold:.6f} kW.")
    Q_shex = Q_shex_hot
    # Pinch-Residuum SHEX: kleinster Temperaturabstand = dT_min_shex
    dT_shex_hot_end  = T3 - T4   # heiß ein  / kalt aus
    dT_shex_cold_end = T2 - T5   # heiß aus  / kalt ein
    pinch_shex = smooth_min(dT_shex_hot_end, dT_shex_cold_end, k=50.0)

    lmtd_shex = _counterflow_lmtd_mode(
        strict=strict, hot_in=T3, hot_out=T2, cold_in=T5, cold_out=T4
    )

    # ------------------------------------------------------------------
    # 6) Drossel 2 -> 1 (isenthalp, T1 aus Flash-Drossel)
    # ------------------------------------------------------------------
    h1 = h2
    if strict:
        flash = lp.flash_valve_state_2_to_1(
            p_out_pa=p_low,
            h2_kJkg=h2,
            m2_kg_s=m2,
            x2_libr_mol=x3,
        )
        T1 = flash["T1_K"]
    else:
        try:
            flash = lp.flash_valve_state_2_to_1(
                p_out_pa=p_low,
                h2_kJkg=h2,
                m2_kg_s=m2,
                x2_libr_mol=x3,
            )
            T1 = flash["T1_K"]
        except Exception:
            T1 = water_T_sat_from_p(p_low, Q=0.0)
            flash = {
                "T1_K": T1,
                "x1_LiBr_mol": x3,
                "w1_LiBr": w3,
                "m1_sol_kg_s": m2,
                "m1_flash_kg_s": 0.0,
                "h1_sol_kJ_kg": h1,
                "h1_flash_kJ_kg": float("nan"),
                "h1_mix_kJ_kg": h1,
                "flash_fraction": 0.0,
            }
    flash_outputs = {key: float(value) for key, value in flash.items()}

    # ------------------------------------------------------------------
    # 7) Kältemitteldampfpfad 7 sowie externe Heißseite von Desorber/Verdampfer
    # ------------------------------------------------------------------
    T7 = T1 + inputs.desorber_vapor_superheat_K
    h7 = water_h_kjkg_PT(p_low, T7)

    Q_des = m6 * h6 + m7 * h7 - m1 * h1
    if strict and Q_des <= 0.0:
        raise ModelEvaluationError(f"Desorberwärmestrom nicht positiv: Q_des={Q_des:.6f} kW.")

    # ------------------------------------------------------------------
    # 8) Kondensator 7 -> 8
    # ------------------------------------------------------------------
    h8 = water_h_kjkg_PQ(p_low, Q=0.0)

    Q_cond = m7 * (h7 - h8)
    if strict and Q_cond <= 0.0:
        raise ModelEvaluationError(f"Kondensatorwärmestrom nicht positiv: Q_cond={Q_cond:.6f} kW.")
    
    m17, T18 = _resolve_condenser_external_stream(inputs, Q_cond, strict=strict)
    
    # Pinch Kondensator: min beider Enden (Lage hängt vom Betriebspunkt ab)
    dT_cond_hot_end  = T7 - T18   # heiß ein / kalt aus
    dT_cond_cold_end = T8 - inputs.T_17   # heiß aus / kalt ein
    pinch_cond = smooth_min(dT_cond_hot_end, dT_cond_cold_end, k=50.0)

    lmtd_cond = _counterflow_lmtd_mode(
        strict=strict, hot_in=T8, hot_out=T8, cold_in=inputs.T_17, cold_out=T18
    )

    # ------------------------------------------------------------------
    # 9) Kältemittelpumpe 8 -> 9
    # ------------------------------------------------------------------
    rho8 = water_rho_kgm3_PQ(p_low, Q=0.0)
    v8 = 1.0 / rho8
    W_ref_pump = m8 * v8 * (p_high - p_low) / 1000.0
    if strict:
        h9 = h8 + W_ref_pump / m8
    else:
        h9 = h8 + (W_ref_pump / m8 if abs(m8) > 1.0e-12 else 0.0)
    T9 = water_T_K_PH(p_high, h9)

    # ------------------------------------------------------------------
    # 10) Verdampfer 9 -> 10 und gekoppelte externe Heißseite
    # ------------------------------------------------------------------
    Q_evap = m9 * (h10 - h9)
    if strict and Q_evap <= 0.0:
        raise ModelEvaluationError(f"Verdampferwärmestrom nicht positiv: Q_evap={Q_evap:.6f} kW.")

    # Externe Temperaturen Desorber / Verdampfer (Routing-Logik identisch)
    if inputs.uses_serial_evaporator_to_desorber_routing:
        T15_in = _resolve_evaporator_external_inlet_temperature(inputs)
        if strict:
            m15, T16 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=strict, T15=T15_in)
        else:
            m15, T16 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=False, T15=T15_in)
        lmtd_evap = _counterflow_lmtd_mode(
            strict=strict, hot_in=T15_in, hot_out=T16, cold_in=T10, cold_out=T10
        )
        T13_in = _resolve_desorber_external_inlet_temperature(inputs, T16)
        if strict:
            m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=strict, T13=T13_in)
        else:
            m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=False, T13=T13_in)
        lmtd_des = _counterflow_lmtd_mode(
            strict=strict, hot_in=T13_in, hot_out=T14, cold_in=T1, cold_out=T6
        )
    else:
        T13_in = _resolve_desorber_external_inlet_temperature(inputs)
        if strict:
            m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=strict, T13=T13_in)
        else:
            m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=False, T13=T13_in)
        lmtd_des = _counterflow_lmtd_mode(
            strict=strict, hot_in=T13_in, hot_out=T14, cold_in=T1, cold_out=T6
        )
        T15_in = _resolve_evaporator_external_inlet_temperature(inputs, T14)
        if strict:
            m15, T16 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=strict, T15=T15_in)
        else:
            m15, T16 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=False, T15=T15_in)
        lmtd_evap = _counterflow_lmtd_mode(
            strict=strict, hot_in=T15_in, hot_out=T16, cold_in=T10, cold_out=T10
        )
    
    # Pinch Desorber: min beider Enden (Lage hängt vom Betriebspunkt ab)
    dT_des_hot_end  = T13_in - T6   # heiß ein / kalt aus
    dT_des_cold_end = T14    - T1   # heiß aus / kalt ein
    pinch_des  = smooth_min(dT_des_hot_end,  dT_des_cold_end,  k=50.0)

    # Pinch Verdampfer: min beider Enden (Lage hängt vom Betriebspunkt ab)
    dT_evap_hot_end  = T16 - T10    # heiß ein / kalt aus
    dT_evap_cold_end = T15_in - T10   # heiß aus / kalt ein
    pinch_evap = smooth_min(dT_evap_hot_end, dT_evap_cold_end, k=50.0)

    # ------------------------------------------------------------------
    # 11) Adiabate Vorabsorption 4 + 19 -> 20
    # ------------------------------------------------------------------
    h20 = lp.h_solution_mass_kjkg(T20, x20)

    # ------------------------------------------------------------------
    # 12) Absorber (globale Energiebilanz, lokale LMTD mit Zustand 20)
    # ------------------------------------------------------------------
    Q_abs = m10 * h10 + m4 * h4 - m3 * h3
    if strict and Q_abs <= 0.0:
        raise ModelEvaluationError(f"Absorberwärmestrom nicht positiv: Q_abs={Q_abs:.6f} kW.")
    m11, T12 = _resolve_absorber_external_stream(inputs, Q_abs, strict=strict)

    # Pinch Absorber: min beider Enden
    dT_abs_hot_end  = T20 - T12          # heiß ein / kalt aus
    dT_abs_cold_end = T3  - inputs.T_11  # heiß aus / kalt ein
    pinch_abs  = smooth_min(dT_abs_hot_end,  dT_abs_cold_end,  k=50.0)
    
    lmtd_abs = _counterflow_lmtd_mode(
        strict=strict, hot_in=T20, hot_out=T3, cold_in=inputs.T_11, cold_out=T12
    )

    # ------------------------------------------------------------------
    #Exergy Bilanzierung / Refenrenzwerte
    # ------------------------------------------------------------------
    def T_mean_entropy(T_in, T_out):
        if abs(T_out - T_in) < 1e-12:
            return 0.5 * (T_in + T_out)
        return (T_out - T_in) / math.log(T_out / T_in)

    T_Abs_ext = T_mean_entropy(inputs.T_11, T12)
    T_Des_ext = T_mean_entropy(T13_in, T14)
    T_Con_ext = T_mean_entropy(inputs.T_17, T18)
    T_Eva_ext = T_mean_entropy(T15_in, T16)

    T_0_K: float = 273.15
    # T_0_K: float = T_Con_ext
    p_0_Pa: float = 101325.0
    x0: float = lp.x_from_w_libr(0.5)
    h_0_w_kJkg: float = water_h_kjkg_PT(p_0_Pa, T_0_K)
    h_0_kJkg: float = lp.h_solution_mass_kjkg(T_0_K, x0)
    s_0_w_kJkgK: float = water_s_kjkgK_PT(p_0_Pa, T_0_K) # 7,8,9,10
    s_0_kJkgK: float = lp.s_solution_mass_kjkgK(T_0_K, x0) # 7,8,9,10

    # ------------------------------------------------------------------
    # Entropische Kennzahlen 
    # ------------------------------------------------------------------
    COP_th = Q_abs / (Q_des + Q_evap)
    
    ECOP = Q_abs*(1.0 - T_0_K / T_Abs_ext) / (Q_des * (1.0 - T_0_K / T_Des_ext) + Q_evap * (1.0 - T_0_K / T_Eva_ext))

    COP_rev = (Q_des * (1 / T_Con_ext - 1 / T_Des_ext) + Q_evap * (1 / T_Con_ext - 1 / T_Eva_ext)) / ((Q_abs + Q_evap) * (1 / T_Con_ext - 1 / T_Abs_ext))

    Zeta = COP_th / COP_rev

    # ------------------------------------------------------------------
    # Exergiebilanz über gesamten Kreislauf
    # ------------------------------------------------------------------
    s1= flash["flash_fraction"] * water_s_kjkgK_PT(p_low, T1) + (1.0 - flash["flash_fraction"]) * lp.s_solution_mass_kjkgK(T1, flash["x1_LiBr_mol"])
    s1_w= water_s_kjkgK_PT(p_low, T1)
    s1_s= lp.s_solution_mass_kjkgK(T1, flash["x1_LiBr_mol"])
    e1_w= (flash["h1_flash_kJ_kg"] - h_0_w_kJkg) - T_0_K * (s1_w - s_0_w_kJkgK)
    e1_s= (flash["h1_sol_kJ_kg"] - h_0_kJkg) - T_0_K * (s1_s - s_0_kJkgK)
    e1 = flash["flash_fraction"] * e1_w + (1 - flash["flash_fraction"]) * e1_s

    s2 = lp.s_solution_mass_kjkgK(T2, x3)
    e2=(h2 - h_0_kJkg) - T_0_K * (s2 - s_0_kJkgK)

    s3 = lp.s_solution_mass_kjkgK(T3, x3)
    e3 = (h3 - h_0_kJkg) - T_0_K * (s3 - s_0_kJkgK)

    s4 = lp.s_solution_mass_kjkgK(T4, x6)
    e4 = (h4 - h_0_kJkg) - T_0_K * (s4 - s_0_kJkgK)

    s6 = lp.s_solution_mass_kjkgK(T6, x6)
    e6=(h6 - h_0_kJkg) - T_0_K * (s6 - s_0_kJkgK)

    s5 = s6 # Isentrope Pumpenverdichtung
    e5=(h5 - h_0_kJkg) - T_0_K * (s5 - s_0_kJkgK)

    s7 = water_s_kjkgK_PT(p_low, T7) # Konsistent mit Enthalpie berechnung
    e7=(h7 - h_0_w_kJkg) - T_0_K * (s7 - s_0_w_kJkgK)

    s8 = water_s_kjkgK_PQ(p_low, Q=0.0) # Konsistent mit Enthalpie berechnung
    e8=(h8 - h_0_w_kJkg) - T_0_K * (s8 - s_0_w_kJkgK)

    s9 = s8 # Isentrope Pumpenverdichtung
    e9=(h9 - h_0_w_kJkg) - T_0_K * (s9 - s_0_w_kJkgK)

    s10 = water_s_kjkgK_PQ(p_high, Q=1.0) # Konsistent mit Enthalpie berechnung
    e10 = (h10 - h_0_w_kJkg) - T_0_K * (s10 - s_0_w_kJkgK)

    E_abs = m10 * e10 + m4 * e4 - m3 * e3 - Q_abs * (1.0 - T_0_K / T_Abs_ext)
    if strict and E_abs <= 0.0:
        raise ModelEvaluationError(f"Exergie im Absorber nicht positiv: E_abs={E_abs:.6f} kW.")
    
    E_evap = m9 * e9 - m10 * e10 + Q_evap * (1.0 - T_0_K / T_Eva_ext)
    if strict and E_evap <= 0.0:    
        raise ModelEvaluationError(f"Exergie im Verdampfer nicht positiv: E_evap={E_evap:.6f} kW.")
    
    E_cond = m7 * e7 - m8 * e8 - Q_cond * (1.0 - T_0_K / T_Con_ext)
    if strict and E_cond <= 0.0:    
        raise ModelEvaluationError(f"Exergie im Kondensator nicht positiv: E_cond={E_cond:.6f} kW.")
    
    E_des = (flash["m1_sol_kg_s"] * e1_s + flash["m1_flash_kg_s"] * e1_w - m7 * e7 - m6 * e6 + Q_des * (1.0 - T_0_K / T_Des_ext))
    if strict and E_des <= 0.0:
        raise ModelEvaluationError(f"Exergie im Desorber nicht positiv: E_des={E_des:.6f} kW.")
    
    E_SHEX = m3 * e3 + m5 * e5 - m4 * e4 - m2 * e2
    if strict and E_SHEX <= 0.0:
        raise ModelEvaluationError(f"Exergie im SHEX nicht positiv: E_SHEX={E_SHEX:.6f} kW.")
    
    E_sol_pump = m6 * e6 - m5 * e5 + W_sol_pump
    if strict and E_sol_pump < -1.0:
        raise ModelEvaluationError(f"Exergie der Lösungspumpe negativ: E_sol_pump={E_sol_pump:.6f} kW.")

    E_throttle = (m2 * e2 - flash["m1_sol_kg_s"] * e1_s - flash["m1_flash_kg_s"] * e1_w)
    if strict and E_throttle < 0.0:
        raise ModelEvaluationError(f"Exergie der Drossel negativ: E_throttle={E_throttle:.6f} kW.")
    
    E_ref_pump = m8 * e8 - m9 * e9 + W_ref_pump
    if strict and E_ref_pump < -1.0:
        raise ModelEvaluationError(f"Exergie der Kältemittelpumpe negativ: E_ref_pump={E_ref_pump:.6f} kW.")
    
    E_total = E_abs + E_evap + E_cond + E_des + E_SHEX + E_sol_pump + E_throttle + E_ref_pump
    perc_E_abs = E_abs / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_evap = E_evap / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_cond = E_cond / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_des = E_des / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_SHEX = E_SHEX / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_sol_pump = E_sol_pump / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_throttle = E_throttle / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    perc_E_ref_pump = E_ref_pump / E_total * 100.0 if abs(E_total) > 1.0e-12 else float("nan")
    
    # ------------------------------------------------------------------
    # 13) Residuen des 8x8-Systems
    # ------------------------------------------------------------------
    residuals_raw_array = np.array(
        [
            Q_shex_hot - Q_shex_cold,                      
            pinch_shex - inputs.dT_min_shex,                
            m4 * h4 + m19 * h10 - m20 * h20,                          
            pinch_des  - inputs.dT_min_des,                 
            pinch_cond - inputs.dT_min_cond,                
            pinch_evap - inputs.dT_min_evap,                
            pinch_abs  - inputs.dT_min_abs,                
        ],
        dtype=float,
    )
    scales = _residual_scales(m6)
    residuals_scaled_array = residuals_raw_array / scales

    residuals_raw = dict(zip(RESIDUAL_NAMES, residuals_raw_array.tolist()))
    residuals_scaled = dict(zip(RESIDUAL_NAMES, residuals_scaled_array.tolist()))

    # ------------------------------------------------------------------
    # 14) Zustandsvalidierung und Plausibilitätschecks
    # ------------------------------------------------------------------
    validity_messages: List[str] = []
    crystallization_safe_all = True
    for label, T_state, w_state in [
        ("3", T3, w3), ("4", T4, w6), ("5", T5, w6), ("6", T6, w6), ("20", T20, w20)
    ]:
        validity = lp.validate_solution_state(T_state, w_state, label=f"Zustand {label}")
        validity_messages.append(validity.message)
        crystallization_safe_all = crystallization_safe_all and validity.crystallization_safe

    checks = {
        "p_high_gt_p_low": p_high > p_low,
        "w6_gt_w3": w6 > w3,
        "w6_gt_w20_gt_w3": w6 > w20 > w3,
        "beta_in_0_1": 0.0 <= beta <= 1.0,
        "m7_positive": m7 > 0.0,
        "m21_nonnegative": m21 >= 0.0,
        "desorber_evaporator_temperature_coupling_ok": (
            (abs(T15_in - T14) <= 1.0e-12)
            if inputs.uses_serial_desorber_to_evaporator_routing
            else ((abs(T13_in - T16) <= 1.0e-12) if inputs.uses_serial_evaporator_to_desorber_routing else True)
        ),
        "crystallization_safe_all_checked_states": crystallization_safe_all,
    }

    diagnostics = {
        "p_low_Pa": p_low,
        "p_high_Pa": p_high,
        "pressure_ratio_high_over_low": p_high / p_low,
        "T12_K": T12,
        "T13_K": T13_in,
        "T14_K": T14,
        "T15_K": T15_in,
        "T16_K": T16,
        "T18_K": T18,
        "m6_kg_s": m6,
        "m11_kg_s": m11,
        "m13_kg_s": m13,
        "m15_kg_s": m15,
        "m17_kg_s": m17,
        "deltaT_shex_1_K": T3 - T4,
        "deltaT_shex_2_K": T2 - T5,
        "deltaT_des_1_K": T13_in - T6,
        "deltaT_des_2_K": T14 - T1,
        "deltaT_cond_1_K": T7 - T18,
        "deltaT_cond_2_K": T8 - inputs.T_17,
        "deltaT_evap_1_K": T15_in - T10,
        "deltaT_evap_2_K": T16 - T10,
        "deltaT_abs_1_K": T20 - T12,
        "deltaT_abs_2_K": T3 - inputs.T_11,
    }

    # ------------------------------------------------------------------
    # 15) Zustandsdictionary
    # ------------------------------------------------------------------
    states = {
        "1":  _state_dict(T1,          p_Pa=p_low,  m_kg_s=m1,  h_kJ_kg=h1,  x_LiBr_mol=x3,  w_LiBr=w3, s=s1, e=e1),
        "2":  _state_dict(T2,          p_Pa=p_high, m_kg_s=m2,  h_kJ_kg=h2,  x_LiBr_mol=x3,  w_LiBr=w3, s=s2, e=e2),
        "3":  _state_dict(T3,          p_Pa=p_high, m_kg_s=m3,  h_kJ_kg=h3,  x_LiBr_mol=x3,  w_LiBr=w3, s=s3, e=e3),
        "4":  _state_dict(T4,          p_Pa=p_high, m_kg_s=m4,  h_kJ_kg=h4,  x_LiBr_mol=x6,  w_LiBr=w6, s=s4, e=e4),
        "5":  _state_dict(T5,          p_Pa=p_high, m_kg_s=m5,  h_kJ_kg=h5,  x_LiBr_mol=x6,  w_LiBr=w6, s=s5, e=e5),
        "6":  _state_dict(T6,          p_Pa=p_low,  m_kg_s=m6,  h_kJ_kg=h6,  x_LiBr_mol=x6,  w_LiBr=w6, s=s6, e=e6),
        "7":  _state_dict(T7,          p_Pa=p_low,  m_kg_s=m7,  h_kJ_kg=h7,  x_LiBr_mol=0.0, w_LiBr=0.0, s=s7, e=e7),
        "8":  _state_dict(T8,          p_Pa=p_low,  m_kg_s=m8,  h_kJ_kg=h8,  x_LiBr_mol=0.0, w_LiBr=0.0, s=s8, e=e8),
        "9":  _state_dict(T9,          p_Pa=p_high, m_kg_s=m9,  h_kJ_kg=h9,  x_LiBr_mol=0.0, w_LiBr=0.0, s=s9, e=e9),
        "10": _state_dict(T10,         p_Pa=p_high, m_kg_s=m10, h_kJ_kg=h10, x_LiBr_mol=0.0, w_LiBr=0.0, s=s10, e=e10),
        "19": _state_dict(T10,         p_Pa=p_high, m_kg_s=m19, h_kJ_kg=h10, x_LiBr_mol=0.0, w_LiBr=0.0, s=s10, e=e10),
        "20": _state_dict(T20,         p_Pa=p_high, m_kg_s=m20, h_kJ_kg=h20, x_LiBr_mol=x20, w_LiBr=w20, s=0.0, e=0.0),
        "21": _state_dict(T10,         p_Pa=p_high, m_kg_s=m21, h_kJ_kg=h10, x_LiBr_mol=0.0, w_LiBr=0.0, s=s10, e=e10),
        "11": _state_dict(inputs.T_11, m_kg_s=m11),
        "12": _state_dict(T12,         m_kg_s=m11),
        "13": _state_dict(T13_in,       m_kg_s=m13),
        "14": _state_dict(T14,         m_kg_s=m13),
        "15": _state_dict(T15_in,       m_kg_s=m15),
        "16": _state_dict(T16,         m_kg_s=m15),
        "17": _state_dict(inputs.T_17, m_kg_s=m17),
        "18": _state_dict(T18,         m_kg_s=m17),
    }

    primary_variables = dict(zip(PRIMARY_VARIABLE_NAMES, [T8, T10, x3, x6, x20, T2, T4]))
    kpis = _calculate_kpis(
        Q_abs=Q_abs,
        Q_evap=Q_evap,
        Q_des=Q_des,
        T12=T12,
        T15=T15_in,
        m6=m6,
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
            "W_ref_pump": W_ref_pump*1000,
        },
        UA_conversion =  {
            "LMTD_shex": lmtd_shex,
            "LMTD_des": lmtd_des,
            "LMTD_cond": lmtd_cond,
            "LMTD_evap": lmtd_evap,
            "LMTD_abs": lmtd_abs,
            "UA_shex": Q_shex / lmtd_shex,
            "UA_des": Q_des / lmtd_des,
            "UA_cond": Q_cond / lmtd_cond,
            "UA_evap": Q_evap / lmtd_evap,
            "UA_abs": Q_abs / lmtd_abs,
        },
        pinch_temperatures_K = {
            "pinch_shex_K": pinch_shex,
            "pinch_des_K": pinch_des,
            "pinch_cond_K": pinch_cond,
            "pinch_evap_K": pinch_evap,
            "pinch_abs_K": pinch_abs,

            "dT_shex_hot_end_K": dT_shex_hot_end,
            "dT_shex_cold_end_K": dT_shex_cold_end,

            "dT_des_hot_end_K": dT_des_hot_end,
            "dT_des_cold_end_K": dT_des_cold_end,

            "dT_cond_hot_end_K": dT_cond_hot_end,
            "dT_cond_cold_end_K": dT_cond_cold_end,

            "dT_evap_hot_end_K": dT_evap_hot_end,
            "dT_evap_cold_end_K": dT_evap_cold_end,

            "dT_abs_hot_end_K": dT_abs_hot_end,
            "dT_abs_cold_end_K": dT_abs_cold_end,
        },
        compositions={
            "x3_LiBr_mol": x3,
            "x6_LiBr_mol": x6,
            "x20_LiBr_mol": x20,
            "w3_LiBr": w3,
            "w6_LiBr": w6,
            "w20_LiBr": w20,
            "beta_m19_over_m10": beta,
        },
        exergy_kW={
            "E_abs": f"{E_abs:.3f} kW ({perc_E_abs:.1f} %)",
            "E_des": f"{E_des:.3f} kW ({perc_E_des:.1f} %)",
            "E_cond": f"{E_cond:.3f} kW ({perc_E_cond:.1f} %)",
            "E_evap": f"{E_evap:.3f} kW ({perc_E_evap:.1f} %)",
            "E_SHEX": f"{E_SHEX:.3f} kW ({perc_E_SHEX:.1f} %)",
            "E_throttle": f"{E_throttle:.3f} kW ({perc_E_throttle:.1f} %)",
            "E_sol_pump": f"{E_sol_pump:.3f} kW ({perc_E_sol_pump:.1f} %)",
            "E_ref_pump": f"{E_ref_pump:.3f} kW ({perc_E_ref_pump:.1f} %)",
            "Exergy_efficiency": ECOP,
            "COP_rev": COP_rev,
            "Zeta": Zeta,
            "T_Abs_ext_K": T_Abs_ext - 273.15,
            "T_Des_ext_K": T_Des_ext - 273.15,
            "T_Con_ext_K": T_Con_ext - 273.15,
            "T_Eva_ext_K": T_Eva_ext - 273.15,
            "h_0_kJkg": h_0_kJkg,
            "h_0_w_kJkg": h_0_w_kJkg,
            "s_0_kJkgK": s_0_kJkgK,
            "s_0_w_kJkgK": s_0_w_kJkgK
        },
        flash_outputs=flash_outputs,
        residuals_raw=residuals_raw,
        residuals_scaled=residuals_scaled,
        diagnostics=diagnostics,
        checks=checks,
        validity_messages=validity_messages,
    )


def evaluate_model(z: np.ndarray, inputs: AWTInputs) -> ModelEvaluation:
    """Berechnet alle Zustände, Apparategrößen und Residuen für einen Variablenvektor.

    Diese öffentliche Variante ist die strenge Endauswertung und wirft
    ModelEvaluationError für unphysikalische Zustände.
    """
    return _evaluate_model_common(z, inputs, strict=True)


# ---------------------------------------------------------------------------
# Solver-Interface
# ---------------------------------------------------------------------------

def residual_vector(z: np.ndarray, inputs: AWTInputs) -> np.ndarray:
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
    z: np.ndarray, inputs: AWTInputs
) -> tuple[ModelEvaluation | None, str | None]:
    try:
        model = evaluate_model(z, inputs)
        return model, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def solve_awt(inputs: AWTInputs, x0: np.ndarray | None = None) -> AWTResult:
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
            UA_conversion={},
            pinch_temperatures_K={},
            compositions={},
            flash_outputs={},
            residuals_raw={},
            residuals_scaled={},
            diagnostics={},
            exergy_kW={},
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
        UA_conversion=model.UA_conversion,
        pinch_temperatures_K=model.pinch_temperatures_K,
        compositions=model.compositions,
        flash_outputs=model.flash_outputs,
        residuals_raw=model.residuals_raw,
        residuals_scaled=model.residuals_scaled,
        diagnostics=model.diagnostics,
        exergy_kW=model.exergy_kW,
        checks=model.checks,
        validity_messages=model.validity_messages,
    )


# ---------------------------------------------------------------------------
# Debugging-Hilfe: Trace für Startwertanalyse
# ---------------------------------------------------------------------------

def trace_model(z: np.ndarray, inputs: AWTInputs) -> ModelTrace:
    """Wertet das Modell schrittweise aus und gibt alle Zwischenergebnisse zurück.
    Nützlich zur Diagnose von Startwertproblemen."""
    T8, T10, x3, x6, x20, T2, T4 = map(float, z)

    values: Dict[str, float] = {}
    primary_variables = dict(
        zip(PRIMARY_VARIABLE_NAMES, [T8, T10, x3, x6, x20, T2, T4])
    )
    stage = "initial"

    try:
        stage = "pressure_levels"
        p_low = water_p_sat_from_T(T8, Q=0.0)
        p_high = water_p_sat_from_T(T10, Q=1.0)
        values["p_low_Pa"] = p_low
        values["p_high_Pa"] = p_high

        if p_high <= p_low:
            raise ModelEvaluationError(
                f"p_high <= p_low ({p_high:.3e} <= {p_low:.3e} Pa)."
            )

        stage = "solution_saturation_states"
        T3 = lp.T_sat_solution_from_p_x(p_high, x3)
        T6 = lp.T_sat_solution_from_p_x(p_low, x6)
        T20 = lp.T_sat_solution_from_p_x(p_high, x20)
        values["T3_K"] = T3
        values["T6_K"] = T6
        values["T20_K"] = T20

        w3 = lp.w_libr_from_x(x3)
        w6 = lp.w_libr_from_x(x6)
        w20 = lp.w_libr_from_x(x20)
        values["w3_LiBr"] = w3
        values["w6_LiBr"] = w6
        values["w20_LiBr"] = w20

        if not (w6 > w3 > 0.0):
            raise ModelEvaluationError(
                f"Konzentrationshierarchie verletzt: w6={w6:.6f}, w3={w3:.6f}."
            )
        if not (w6 > w20 > w3):
            raise ModelEvaluationError(
                f"Vorabsorptionszustand unplausibel: w6={w6:.6f}, w20={w20:.6f}, w3={w3:.6f}."
            )

        stage = "cycle_scale"
        h3 = lp.h_solution_mass_kjkg(T3, x3)
        h4 = lp.h_solution_mass_kjkg(T4, x6)
        h10 = water_h_kjkg_PQ(p_high, Q=1.0)
        m6 = _resolve_cycle_scale(inputs, w3=w3, w6=w6, h3=h3, h4=h4, h10=h10, strict=True)
        values["h3_kJ_kg"] = h3
        values["h4_kJ_kg"] = h4
        values["h10_kJ_kg"] = h10
        values["m6_kg_s"] = m6

        stage = "mass_flows"
        m4 = m6
        m3 = m4 * w6 / w3
        m7 = m3 - m6
        m10 = m7
        m20 = m4 * w6 / w20
        m19 = m20 - m4
        beta = m19 / m10
        m21 = m10 - m19
        values["m3_kg_s"] = m3
        values["m7_kg_s"] = m7
        values["m19_kg_s"] = m19
        values["m20_kg_s"] = m20
        values["m21_kg_s"] = m21
        values["beta_m19_over_m10"] = beta
        values["preabs_LiBr_residual_kg_s"] = m4 * w6 - m20 * w20

        if m7 <= 0.0:
            raise ModelEvaluationError(f"m7={m7:.6f} kg/s nicht positiv.")
        if not (0.0 <= beta <= 1.0):
            raise ModelEvaluationError(f"beta={beta:.6f} liegt außerhalb [0,1].")
        if m21 < 0.0:
            raise ModelEvaluationError(f"m21={m21:.6f} kg/s negativ.")

        stage = "solution_pump"
        h2 = lp.h_solution_mass_kjkg(T2, x3)
        h6 = lp.h_solution_mass_kjkg(T6, x6)
        rho6 = lp.rho_solution_mass(T6, x6)
        W_sol_pump = m6 / rho6 * (p_high - p_low) / 1000.0
        h5 = h6 + W_sol_pump / m6
        T5 = lp.T_from_h_x_mass(h5, x6)
        values.update({
            "h2_kJ_kg": h2,
            "h6_kJ_kg": h6,
            "W_sol_pump_kW": W_sol_pump,
            "h5_kJ_kg": h5,
            "T5_K": T5,
        })

        stage = "shex"
        Q_shex_hot = m3 * (h3 - h2)
        Q_shex_cold = m4 * (h4 - h5)
        values["Q_shex_hot_kW"] = Q_shex_hot
        values["Q_shex_cold_kW"] = Q_shex_cold
        values["deltaT_shex_1_K"] = T3 - T4
        values["deltaT_shex_2_K"] = T2 - T5

        if Q_shex_hot <= 0.0:
            raise ModelEvaluationError(f"Q_shex_hot={Q_shex_hot:.4f} kW nicht positiv.")
        if Q_shex_cold <= 0.0:
            raise ModelEvaluationError(f"Q_shex_cold={Q_shex_cold:.4f} kW nicht positiv.")

        stage = "throttle"
        h1 = h2
        flash = lp.flash_valve_state_2_to_1(
            p_out_pa=p_low, h2_kJkg=h2, m2_kg_s=m3, x2_libr_mol=x3
        )
        T1 = flash["T1_K"]
        values["h1_kJ_kg"] = h1
        values["T1_K"] = T1
        for key, value in flash.items():
            values[f"flash_{key}"] = float(value)

        stage = "hot_side_coupling"
        T7 = T1 + inputs.desorber_vapor_superheat_K
        h7 = water_h_kjkg_PT(p_low, T7)
        Q_des = m6 * h6 + m7 * h7 - m3 * h1
        values["T7_K"] = T7
        values["h7_kJ_kg"] = h7
        values["Q_des_kW"] = Q_des

        if Q_des <= 0.0:
            raise ModelEvaluationError(f"Q_des={Q_des:.4f} kW nicht positiv.")

        stage = "condenser"
        h8 = water_h_kjkg_PQ(p_low, Q=0.0)
        Q_cond = m7 * (h7 - h8)
        m17, T18 = _resolve_condenser_external_stream(inputs, Q_cond, strict=True)
        values["h8_kJ_kg"] = h8
        values["Q_cond_kW"] = Q_cond
        values["m17_kg_s"] = m17
        values["T18_K"] = T18
        values["deltaT_cond_1_K"] = T7 - T18
        values["deltaT_cond_2_K"] = T8 - inputs.T_17

        if Q_cond <= 0.0:
            raise ModelEvaluationError(f"Q_cond={Q_cond:.4f} kW nicht positiv.")
        
        stage = "refrigerant_pump"
        rho8 = water_rho_kgm3_PQ(p_low, Q=0.0)
        W_ref_pump = m7 / rho8 * (p_high - p_low) / 1000.0
        h9 = h8 + W_ref_pump / m7
        T9 = water_T_K_PH(p_high, h9)
        values["W_ref_pump_kW"] = W_ref_pump
        values["h9_kJ_kg"] = h9
        values["T9_K"] = T9

        stage = "evaporator"
        if inputs.uses_serial_evaporator_to_desorber_routing:
            T15_in = _resolve_evaporator_external_inlet_temperature(inputs)
        else:
            T13_in = _resolve_desorber_external_inlet_temperature(inputs)
            m13_seed, T14_seed = _resolve_desorber_external_stream(inputs, Q_des, strict=True, T13=T13_in)
            T15_in = _resolve_evaporator_external_inlet_temperature(inputs, T14_seed)

        Q_evap = m7 * (h10 - h9)
        m15, T16 = _resolve_evaporator_external_stream(inputs, Q_evap, strict=True, T15=T15_in)
        values["Q_evap_kW"] = Q_evap
        values["T15_K"] = T15_in
        values["T16_K"] = T16
        values["m15_kg_s"] = m15
        values["deltaT_evap_1_K"] = T15_in - T10
        values["deltaT_evap_2_K"] = T16 - T10

        if Q_evap <= 0.0:
            raise ModelEvaluationError(f"Q_evap={Q_evap:.4f} kW nicht positiv.")

        stage = "desorber"
        if inputs.uses_serial_evaporator_to_desorber_routing:
            T13_in = _resolve_desorber_external_inlet_temperature(inputs, T16)
            m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=True, T13=T13_in)
        else:
            T13_in = _resolve_desorber_external_inlet_temperature(inputs)
            m13, T14 = _resolve_desorber_external_stream(inputs, Q_des, strict=True, T13=T13_in)
        values["T13_K"] = T13_in
        values["T14_K"] = T14
        values["m13_kg_s"] = m13
        values["deltaT_des_1_K"] = T13_in - T6
        values["deltaT_des_2_K"] = T14 - T1

        stage = "pre_absorption"
        h20 = lp.h_solution_mass_kjkg(T20, x20)
        values["h20_kJ_kg"] = h20

        stage = "absorber"
        Q_abs = m10 * h10 + m4 * h4 - m3 * h3
        if Q_abs <= 0.0:
            raise ModelEvaluationError(f"Q_abs={Q_abs:.4f} kW nicht positiv.")
        m11, T12 = _resolve_absorber_external_stream(inputs, Q_abs, strict=True)
        values["Q_abs_kW"] = Q_abs
        values["m11_kg_s"] = m11
        values["T12_K"] = T12
        values["deltaT_abs_1_K"] = T20 - T12
        values["deltaT_abs_2_K"] = T3 - inputs.T_11

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
        f"w = {maybe('w_LiBr', '{:.6f}')} | "
        f"s = {maybe('s', '{:.6f}')} kJ/kg/K | "
        f"e = {maybe('e', '{:.6f}')} kJ/kg"
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
    print("AWT-Simulation – Ergebnisübersicht (8 primäre Unbekannte)")
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

    print("UA Umrechnung + LMTD [kW/K] [K]")
    for key, value in result.UA_conversion.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("Pinch-Temperaturen [K]")
    for key, value in result.pinch_temperatures_K.items():
        print(f"  {key:12s}: {value:12.6f}")
    print()

    print("Flash-Drossel 2 -> 1 (nur Ausgabe, nicht für weitere Bilanzierung verwendet)")
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

    print("Exergiebezogene Diagnostik [kW]")
    for key, value in result.exergy_kW.items():
        print(f"  {key:35s}: {value}")
    print()

    print("Validitätsmeldungen")
    for msg in result.validity_messages:
        print(f"  - {msg}")

    print("=" * 110)


__all__ = [
    "AWTInputs",
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
