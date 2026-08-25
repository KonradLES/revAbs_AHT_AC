#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dühring-Diagramme für wässrige LiBr-Lösungen, inkl. AWT-Betriebspunkt.

Dieses Modul enthält unverändert die Basisfunktionalität zur Erzeugung von
Dühring-Diagrammen (Isosteren nach Pátek & Klomfar, Kristallisationsgrenze
nach Albers/Boryta) sowie zusätzlich `plot_duehring_operating_point()`, die
den von `solve_awt()` berechneten Betriebspunkt (Zustände 1,2,3,4,5,6,20)
als Kreisprozess-Polygon in das Diagramm einzeichnet.

Positionierung des Betriebspunkts
-----------------------------------
Auf der Dühring-Achse ist die y-Koordinate eines Lösungszustands die
Tautemperatur des reinen Wassers bei seinem Druck. Diese entspricht im
Modell exakt den Zuständen 8 (p_low) bzw. 10 (p_high), da beide per
Definition auf der reinen Wassersättigungslinie liegen (Q=0 bzw. Q=1) und
diese Drücke gerade definieren. Eine zusätzliche Umrechnung ist daher nicht
nötig:
    - Zustände bei p_low  (1, 6):                y = T(Zustand 8)  [°C]
    - Zustände bei p_high (2, 3, 4, 5, 20):       y = T(Zustand 10) [°C]
Die x-Koordinate ist jeweils die Lösungstemperatur T(Zustand n) [°C].

Prozessreihenfolge des eingezeichneten Kreises
-------------------------------------------------
6 (Desorberaustritt, p_low) -> 5 (nach Lösungspumpe, p_high) ->
4 (nach SHEX-Vorwärmung, p_high) -> 20 (nach adiabater Vorabsorption,
p_high) -> 3 (Absorberaustritt, p_high) -> 2 (nach SHEX-Abkühlung, p_high)
-> 1 (nach Drossel, p_low) -> zurück zu 6 (Desorption).

Hinweis zum Darstellungsbereich
-----------------------------------
Die Standardgrenzen (X_AXIS_MAX_C = 160 °C, P_RIGHT_MAX_MBAR = 2000 mbar)
stammen aus der Vorlage. Falls dein Betriebspunkt (insbesondere bei hohen
Werten für T_11 bzw. hohen Drücken) außerhalb dieses Bereichs liegt, diese
beiden Konstanten am Kopf der Datei anpassen.

Aufruf als eigenständiges Skript
-----------------------------------
    python AHT_Duehring_Plot.py --variant both --output-dir plots

Nutzung im Hauptskript
-----------------------
    from Postprocessing.AHT_Duehring_Plot import plot_duehring_operating_point

    if ENABLE_DUEHRING_PLOT:
        plot_duehring_operating_point(result, save_path="Postprocessing/Plots/AWT_Duehring_Diagramm.png")
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
except ImportError as exc:  # pragma: no cover - nur bei fehlenden Paketen
    raise SystemExit(
        "Fehlendes Python-Paket. Installiere die Abhängigkeiten mit:\n"
        "    python -m pip install numpy matplotlib"
    ) from exc

from Models.AHT_Pinch_Point import AWTResult, kelvin_to_celsius
import Thermodynamic_Properties.libr_props as lp


# =============================================================================
# Konstanten und Diagrammkonfiguration
# =============================================================================

M_LIBR = 0.08685       # kg/mol, entsprechend der verwendeten Implementierung
M_H2O = 0.018015268    # kg/mol
T_CRIT_H2O = 647.096   # K
P_CRIT_H2O = 22.064e6  # Pa
T0_C = 273.15          # K -> °C

X_AXIS_MIN_C = 0.0
X_AXIS_MAX_C = 160.0
X_AXIS_MAJOR_C = 20.0

# Der obere Rand der linken y-Achse wird aus p = 2000 mbar für reines Wasser
# bestimmt. Dadurch stimmen linke Temperatur- und rechte Druckachse exakt überein.
P_RIGHT_MAX_MBAR = 2000.0

# Dichte der Isosterenschar. Alle 10 Prozentpunkte: schwarz; dazwischen grau.
MAJOR_COMPOSITION_STEP_PERCENT = 10.0
MINOR_COMPOSITION_STEP_PERCENT = 2.5

# Albers/Boryta: gültiger Bereich des Polynoms T_cr(w)
W_CRYST_MIN = 0.57
W_CRYST_MAX = 0.70

# Obergrenze der gezeichneten Konzentrationen. Sie entspricht zugleich dem
# oberen Gültigkeitsende der verwendeten Kristallisationskorrelation.
W_PLOT_MAX = W_CRYST_MAX
X_PLOT_MAX = None  # wird nach Definition der Umrechnungsfunktion gesetzt

# Rechter Druckmaßstab, orientiert an den beigefügten Referenzabbildungen.
PRESSURE_TICKS_MBAR = np.array(
    [7, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000, 1500, 2000],
    dtype=float,
)

# Pátek/Klomfar, Gleichung (1), Tabelle 4: Druckkorrelation
PAT_A = np.array(
    [-2.41303e2, 1.91750e7, -1.75521e8, 3.25430e7,
      3.92571e2, -2.12626e3, 1.85127e8, 1.91216e3],
    dtype=float,
)
PAT_M = np.array([3, 4, 4, 8, 1, 1, 4, 6], dtype=float)
PAT_N = np.array([0, 5, 6, 3, 0, 2, 6, 0], dtype=float)
PAT_T = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# Pátek/Klomfar, Gleichung (28), Tabelle 11: Dampfdruck von reinem Wasser
WATER_ALPHA = np.array(
    [-7.85951783, 1.84408259, -11.7866497,
      22.6807411, -15.9618719, 1.80122502],
    dtype=float,
)
WATER_BETA = np.array([1.0, 1.5, 3.0, 3.5, 4.0, 7.5], dtype=float)

# Albers (2019), Gleichung (2.52), Tabelle 2.8: T_cr als Funktion von w
ALBERS_T_COEFF = np.array(
    [
        42.90198341384762,
        34.67510890651030,
        31.30778644395644,
        2.99859601946791,
        -19.36781324384540,
        -4.88856108511827,
        4.61433775768846,
        1.80636830673333,
    ],
    dtype=float,
)


# =============================================================================
# Konzentrationsumrechnung
# =============================================================================

def mass_fraction_from_mole_fraction(x_libr: float | np.ndarray) -> float | np.ndarray:
    """LiBr-Massenanteil w aus LiBr-Molanteil x."""
    x = np.asarray(x_libr, dtype=float)
    denominator = x * M_LIBR + (1.0 - x) * M_H2O
    w = x * M_LIBR / denominator
    return float(w) if w.ndim == 0 else w


def mole_fraction_from_mass_fraction(w_libr: float | np.ndarray) -> float | np.ndarray:
    """LiBr-Molanteil x aus LiBr-Massenanteil w."""
    w = np.asarray(w_libr, dtype=float)
    denominator = M_LIBR - w * (M_LIBR - M_H2O)
    x = w * M_H2O / denominator
    return float(x) if x.ndim == 0 else x


X_PLOT_MAX = float(mole_fraction_from_mass_fraction(W_PLOT_MAX))


# =============================================================================
# Pátek-/Klomfar-Korrelation und Dühring-Transformation
# =============================================================================

def _patek_duehring_coefficients(x_libr_mol: float) -> tuple[float, float]:
    """Gibt A(x) [K] und B(x) [K] aus Albers Gl. (2.40) zurück.

    Mit theta = T_H2O bei gleichem Druck gilt:
        theta = T_Lsg - A - B * T_Lsg / T_crit
    und damit die analytische Dühring-Gerade:
        T_Lsg = T_crit/(T_crit-B) * theta + T_crit*A/(T_crit-B)
    """
    x = float(x_libr_mol)
    if not (0.0 <= x < 0.4):
        raise ValueError(f"LiBr-Molanteil x={x:.8f} liegt außerhalb 0 <= x < 0.4.")

    terms = PAT_A * x**PAT_M * (0.4 - x) ** PAT_N
    a_term = float(np.sum(terms[PAT_T == 0.0]))
    b_term = float(np.sum(terms[PAT_T == 1.0]))
    return a_term, b_term


def solution_boiling_temperature_c(
    water_dew_temperature_c: float | np.ndarray,
    x_libr_mol: float,
) -> float | np.ndarray:
    """Lösungssiedetemperatur [°C] für Tautemperatur von reinem Wasser [°C]."""
    theta_k = np.asarray(water_dew_temperature_c, dtype=float) + T0_C
    a_term, b_term = _patek_duehring_coefficients(x_libr_mol)
    denominator = T_CRIT_H2O - b_term
    t_solution_k = (T_CRIT_H2O / denominator) * theta_k + (
        T_CRIT_H2O * a_term / denominator
    )
    result = t_solution_k - T0_C
    return float(result) if result.ndim == 0 else result


def water_dew_temperature_c_from_solution(
    solution_temperature_c: float | np.ndarray,
    x_libr_mol: float,
) -> float | np.ndarray:
    """Inverse Dühring-Beziehung: T_H2O [°C] aus T_Lsg [°C] und x."""
    t_solution_k = np.asarray(solution_temperature_c, dtype=float) + T0_C
    a_term, b_term = _patek_duehring_coefficients(x_libr_mol)
    theta_k = t_solution_k * (1.0 - b_term / T_CRIT_H2O) - a_term
    result = theta_k - T0_C
    return float(result) if result.ndim == 0 else result


def water_saturation_pressure_pa(temperature_k: float | np.ndarray) -> float | np.ndarray:
    """Sättigungsdruck von reinem Wasser nach Pátek/Klomfar Gl. (28) [Pa]."""
    t = np.asarray(temperature_k, dtype=float)
    if np.any((t <= 0.0) | (t >= T_CRIT_H2O)):
        raise ValueError("Wassertemperatur muss zwischen 0 K und T_crit liegen.")

    tau = 1.0 - t / T_CRIT_H2O
    exponent = np.zeros_like(t, dtype=float)
    for alpha, beta in zip(WATER_ALPHA, WATER_BETA):
        exponent += alpha * tau**beta
    p = P_CRIT_H2O * np.exp((T_CRIT_H2O / t) * exponent)
    return float(p) if p.ndim == 0 else p


def water_saturation_temperature_c_from_pressure_mbar(p_mbar: float) -> float:
    """Inverse der Wasser-Dampfdruckgleichung per robuster Bisektion [°C]."""
    target_pa = float(p_mbar) * 100.0
    if target_pa <= 0.0:
        raise ValueError("Der Druck muss positiv sein.")

    lo_k = 250.0
    hi_k = T_CRIT_H2O - 1.0e-8
    p_lo = float(water_saturation_pressure_pa(lo_k))
    p_hi = float(water_saturation_pressure_pa(hi_k))
    if not (p_lo <= target_pa <= p_hi):
        raise ValueError(f"p={p_mbar:g} mbar liegt außerhalb des invertierbaren Bereichs.")

    for _ in range(120):
        mid_k = 0.5 * (lo_k + hi_k)
        p_mid = float(water_saturation_pressure_pa(mid_k))
        if p_mid < target_pa:
            lo_k = mid_k
        else:
            hi_k = mid_k
    return 0.5 * (lo_k + hi_k) - T0_C


# =============================================================================
# Kristallisationsgrenze nach Albers/Boryta
# =============================================================================

def crystallization_temperature_c_from_mass_fraction(
    w_libr: float | np.ndarray,
) -> float | np.ndarray:
    """Kristallisationstemperatur T_cr(w) [°C], Albers Gl. (2.52).

    Gültigkeitsbereich: 0.57 < w < 0.70 kg/kg.
    """
    w = np.asarray(w_libr, dtype=float)
    if np.any((w < W_CRYST_MIN) | (w > W_CRYST_MAX)):
        raise ValueError(
            f"T_cr(w) ist hier nur für {W_CRYST_MIN:.2f} <= w <= {W_CRYST_MAX:.2f} gültig."
        )
    w_b = (w - 0.64794) / 0.044858
    t_cr = np.zeros_like(w_b, dtype=float)
    for power, coefficient in enumerate(ALBERS_T_COEFF):
        t_cr += coefficient * w_b**power
    return float(t_cr) if t_cr.ndim == 0 else t_cr


def crystallization_water_dew_temperature_c(w_libr: float | np.ndarray) -> float | np.ndarray:
    """y-Koordinate der Kristallisationsgrenze im Dühring-Diagramm [°C]."""
    w = np.asarray(w_libr, dtype=float)
    t_solution_cr = crystallization_temperature_c_from_mass_fraction(w)
    x_mol = mole_fraction_from_mass_fraction(w)

    result = np.empty_like(w, dtype=float)
    for idx in np.ndindex(w.shape):
        result[idx] = water_dew_temperature_c_from_solution(
            float(t_solution_cr[idx]), float(x_mol[idx])
        )
    return float(result) if result.ndim == 0 else result


# =============================================================================
# Plausibilitätsprüfungen
# =============================================================================

def run_self_checks() -> None:
    """Prüft die Implementierung gegen Referenzwerte aus Pátek/Klomfar Tabelle 9."""
    reference_points = [
        # (x_LiBr mol/mol, T/K, p/Pa)
        (0.05, 300.0, 3025.1805),
        (0.05, 450.0, 835097.47),
        (0.10, 300.0, 2286.4858),
        (0.10, 450.0, 647702.12),
        (0.30, 350.0, 2237.3986),
        (0.40 - 1.0e-12, 450.0, 43075.149),
    ]

    for x_mol, t_k, p_reference in reference_points:
        theta_c = water_dew_temperature_c_from_solution(t_k - T0_C, x_mol)
        p_calculated = float(water_saturation_pressure_pa(theta_c + T0_C))
        relative_error = abs(p_calculated / p_reference - 1.0)
        if relative_error > 3.0e-5:
            raise RuntimeError(
                "Pátek-Selbstprüfung fehlgeschlagen: "
                f"x={x_mol:.8f}, T={t_k:.3f} K, "
                f"p_calc={p_calculated:.6f} Pa, p_ref={p_reference:.6f} Pa, "
                f"rel. Fehler={relative_error:.3e}."
            )

    # Der obere Albers-Gültigkeitsrand liegt gemäß Bericht ungefähr bei 101 °C.
    if not math.isclose(
        float(crystallization_temperature_c_from_mass_fraction(0.70)),
        100.9689444,
        rel_tol=0.0,
        abs_tol=2.0e-6,
    ):
        raise RuntimeError("Albers-Kristallisationskorrelation liefert unerwartete Werte.")


# =============================================================================
# Plot-Hilfsfunktionen
# =============================================================================

def _is_major_percent(value_percent: float) -> bool:
    remainder = math.fmod(value_percent, MAJOR_COMPOSITION_STEP_PERCENT)
    return min(abs(remainder), abs(remainder - MAJOR_COMPOSITION_STEP_PERCENT)) < 1.0e-8


def _composition_grid(variant: Literal["mole", "mass"]) -> np.ndarray:
    step = MINOR_COMPOSITION_STEP_PERCENT / 100.0
    maximum = X_PLOT_MAX if variant == "mole" else W_PLOT_MAX
    count = int(math.floor(maximum / step + 1.0e-10))
    values = step * np.arange(count + 1, dtype=float)
    if maximum - values[-1] > 1.0e-10:
        values = np.append(values, maximum)
    return values


def _top_axis_compositions(variant: Literal["mole", "mass"]) -> np.ndarray:
    # Bei der Molanteil-Variante sind 5-Prozent-Ticks zweckmäßig, da bei der
    # vorgegebenen x-Achse nur die Schnittpunkte bis etwa 15 mol-% sichtbar sind.
    step_percent = 5.0 if variant == "mole" else 10.0
    maximum = X_PLOT_MAX if variant == "mole" else W_PLOT_MAX
    start = 0.0 if variant == "mole" else 0.20
    return np.arange(start, maximum + 0.5 * step_percent / 100.0, step_percent / 100.0)


def _line_rotation_degrees(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """Berechnet die visuelle Rotation einer Linie in Bildschirmkoordinaten."""
    if len(x_data) < 2:
        return 0.0
    p0 = ax.transData.transform((x_data[0], y_data[0]))
    p1 = ax.transData.transform((x_data[-1], y_data[-1]))
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def create_duehring_figure(variant: Literal["mole", "mass"]) -> Tuple[plt.Figure, plt.Axes]:
    """Erzeugt eine Diagrammvariante und gibt (Figure, primäre Achse) zurück."""
    if variant not in {"mole", "mass"}:
        raise ValueError("variant muss 'mole' oder 'mass' sein.")

    y_axis_max_c = water_saturation_temperature_c_from_pressure_mbar(P_RIGHT_MAX_MBAR)
    y_values_full = np.linspace(0.0, y_axis_max_c, 900)

    fig, ax = plt.subplots(figsize=(10.8, 7.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.875, bottom=0.115, top=0.865)

    ax.set_xlim(X_AXIS_MIN_C, X_AXIS_MAX_C)
    ax.set_ylim(0.0, y_axis_max_c)
    ax.set_xlabel(
        r"Siedetemperatur $T_{\mathrm{H_2O/LiBr}}^{\mathrm{LV}}$ [°C]",
        fontsize=12,
    )
    ax.set_ylabel(
        r"Tautemperatur $T_{\mathrm{H_2O}}^{\mathrm{LV}}$ [°C]",
        fontsize=12,
    )

    ax.xaxis.set_major_locator(MultipleLocator(X_AXIS_MAJOR_C))
    ax.xaxis.set_minor_locator(MultipleLocator(X_AXIS_MAJOR_C / 2.0))
    ax.yaxis.set_major_locator(MultipleLocator(20.0))
    ax.yaxis.set_minor_locator(MultipleLocator(10.0))
    ax.grid(which="major", linewidth=0.8, alpha=0.55)
    ax.grid(which="minor", linewidth=0.55, linestyle=":", alpha=0.55)
    ax.tick_params(direction="in", which="both", top=False, right=False, labelsize=10)

    # -------------------------------------------------------------------------
    # Isosteren
    # -------------------------------------------------------------------------
    for composition in _composition_grid(variant):
        if variant == "mole":
            x_mol = float(composition)
            w_mass = float(mass_fraction_from_mole_fraction(x_mol))
            composition_percent = 100.0 * x_mol
        else:
            w_mass = float(composition)
            x_mol = float(mole_fraction_from_mass_fraction(w_mass))
            composition_percent = 100.0 * w_mass

        # Oberhalb des Albers/Boryta-Bereichs werden keine Isosteren gezeichnet,
        # damit die Kristallisationsgrenze nicht unzulässig extrapoliert wird.
        if w_mass > W_CRYST_MAX + 1.0e-12:
            continue

        y_min_c = 0.0
        if w_mass >= W_CRYST_MIN - 1.0e-12:
            w_for_cryst = min(max(w_mass, W_CRYST_MIN), W_CRYST_MAX)
            t_cr_c = float(crystallization_temperature_c_from_mass_fraction(w_for_cryst))
            y_cr_c = float(water_dew_temperature_c_from_solution(t_cr_c, x_mol))
            y_min_c = max(0.0, y_cr_c)

        if y_min_c >= y_axis_max_c:
            continue

        mask_y = y_values_full >= y_min_c
        y_line = y_values_full[mask_y]
        x_line = np.asarray(solution_boiling_temperature_c(y_line, x_mol))
        mask_plot = (
            (x_line >= X_AXIS_MIN_C)
            & (x_line <= X_AXIS_MAX_C)
            & np.isfinite(x_line)
        )
        if np.count_nonzero(mask_plot) < 2:
            continue

        major = _is_major_percent(composition_percent)
        ax.plot(
            x_line[mask_plot],
            y_line[mask_plot],
            color="black" if major else "0.38",
            linewidth=1.65 if major else 0.65,
            alpha=1.0 if major else 0.78,
            solid_capstyle="round",
            zorder=3 if major else 2,
        )

    # -------------------------------------------------------------------------
    # Kristallisationsgrenze
    # -------------------------------------------------------------------------
    w_crystal = np.linspace(W_CRYST_MIN, W_CRYST_MAX, 500)
    t_solution_crystal = np.asarray(crystallization_temperature_c_from_mass_fraction(w_crystal))
    t_water_crystal = np.asarray(crystallization_water_dew_temperature_c(w_crystal))
    crystal_mask = (
        (t_solution_crystal >= X_AXIS_MIN_C)
        & (t_solution_crystal <= X_AXIS_MAX_C)
        & (t_water_crystal >= 0.0)
        & (t_water_crystal <= y_axis_max_c)
    )
    crystal_x = t_solution_crystal[crystal_mask]
    crystal_y = t_water_crystal[crystal_mask]
    ax.plot(
        crystal_x,
        crystal_y,
        color="firebrick",
        linewidth=4.6,
        solid_capstyle="round",
        zorder=6,
    )
    ax.plot(
        crystal_x,
        crystal_y,
        color="darkred",
        linewidth=1.35,
        solid_capstyle="round",
        zorder=7,
    )

    # Beschriftung der Wasserlinie
    x_text = 101.0
    y_text = 101.0
    rotation_water = _line_rotation_degrees(
        ax,
        np.array([70.0, 110.0]),
        np.array([70.0, 110.0]),
    )
    ax.text(
        x_text,
        y_text,
        "Wasser",
        rotation=rotation_water,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
        fontsize=10,
        zorder=10,
    )

    if len(crystal_x) > 8:
        idx = max(2, int(0.42 * (len(crystal_x) - 1)))
        i0 = max(0, idx - 4)
        i1 = min(len(crystal_x) - 1, idx + 4)
        rotation_crystal = _line_rotation_degrees(
            ax,
            np.array([crystal_x[i0], crystal_x[i1]]),
            np.array([crystal_y[i0], crystal_y[i1]]),
        )
        ax.annotate(
            "Kristallisationsgrenze",
            xy=(crystal_x[idx], crystal_y[idx]),
            xytext=(0.0, -8.0),
            textcoords="offset points",
            rotation=rotation_crystal,
            rotation_mode="anchor",
            ha="left",
            va="top",
            fontsize=9,
            color="darkred",
            zorder=10,
        )

    # -------------------------------------------------------------------------
    # Rechte Druckachse
    # -------------------------------------------------------------------------
    pressure_positions = np.array(
        [water_saturation_temperature_c_from_pressure_mbar(p) for p in PRESSURE_TICKS_MBAR]
    )
    pressure_mask = (pressure_positions >= 0.0) & (pressure_positions <= y_axis_max_c + 1.0e-8)

    # Zusätzliche horizontale Hilfslinien für die diskreten Druckniveaus.
    # Sie ergänzen das reguläre Temperaturgitter, ohne die Isosteren zu überdecken.
    for y_pressure in pressure_positions[pressure_mask]:
        ax.axhline(
            y=y_pressure,
            color="0.78",
            linewidth=0.65,
            linestyle=(0, (4.0, 3.0)),
            zorder=0.8,
        )

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(pressure_positions[pressure_mask])
    ax_right.set_yticklabels([f"{p:g}" for p in PRESSURE_TICKS_MBAR[pressure_mask]])
    ax_right.set_ylabel(r"Gleichgewichtsdruck $p^{\mathrm{LV}}$ [mbar]", fontsize=12)
    ax_right.tick_params(direction="in", which="major", labelsize=10)

    # -------------------------------------------------------------------------
    # Obere Konzentrationsachse: Schnittpunkte mit dem oberen Diagrammrand
    # -------------------------------------------------------------------------
    top_compositions = _top_axis_compositions(variant)
    top_positions: list[float] = []
    top_labels: list[str] = []
    for composition in top_compositions:
        x_mol = float(composition) if variant == "mole" else float(
            mole_fraction_from_mass_fraction(composition)
        )
        position = float(solution_boiling_temperature_c(y_axis_max_c, x_mol))
        if X_AXIS_MIN_C - 1.0e-9 <= position <= X_AXIS_MAX_C + 1.0e-9:
            top_positions.append(position)
            top_labels.append(f"{100.0 * composition:g}")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(top_positions)
    ax_top.set_xticklabels(top_labels, rotation=28, ha="left", rotation_mode="anchor")
    composition_name = r"Molanteil $x^{\mathrm{LiBr}}$" if variant == "mole" else r"Massenanteil $w^{\mathrm{LiBr}}$"
    ax_top.set_xlabel(f"LiBr-{composition_name} der Isosteren [%]", labelpad=0, fontsize=10.5)
    # Die Achsenbezeichnung wird bewusst nahe an der oberen Achse und links von
    # den Konzentrationsticks positioniert, damit sie nicht wie ein Diagrammtitel wirkt.
    ax_top.xaxis.set_label_coords(0.5, 1.025)
    ax_top.tick_params(direction="in", which="major", pad=1, labelsize=9)

    # Vollständige gemeinsame Legende: Quellen-/Varianteninformation und
    # Erklärung aller verwendeten Linienarten in einem einzigen Feld.
    # Vereinfachte Legende
    legend_handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.65,
            label="Isosteren in 10%-Schritten (Pátek & Klomfar)",
        ),
        Line2D(
            [0], [0],
            color="0.38",
            linewidth=0.65,
            label="Isosteren in 2.5%-Schritten (Pátek & Klomfar)",
        ),
        Line2D(
            [0], [0],
            color="darkred",
            linewidth=2.2,
            label="Kristallisationsgrenze (Albers/Boryta)",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=8.5,
    )

    return fig, ax


def save_figure(
    fig: plt.Figure,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    """Speichert eine Figure in allen gewünschten Formaten."""
    written: list[Path] = []
    for extension in formats:
        ext = extension.lower().lstrip(".")
        if ext not in {"png", "pdf", "svg"}:
            raise ValueError(f"Nicht unterstütztes Ausgabeformat: {extension}")
        output_path = output_base.with_suffix(f".{ext}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        written.append(output_path)
    return written


# =============================================================================
# AWT-Betriebspunkt-Überlagerung
# =============================================================================
#
# Dargestellt wird der AWT-Prozess im Dühring-Diagramm.
#
# Die x-Koordinate eines Lösungszustands ist NICHT die tatsächliche
# Prozess-/Austrittstemperatur aus dem Modell, sondern die
# Gleichgewichtstemperatur der LiBr-Lösung bei dem jeweiligen Druck und
# der jeweiligen Lösungskonzentration:
#
#     T_eq = T_sat_solution_from_p_x(p, x_LiBr)
#
# Dadurch werden die beiden Konzentrationslinien (Isosteren) konsistent
# dargestellt:
#
#     schwache Lösung:
#         1 -> 3   mit x_1 = x_3
#
#     starke Lösung:
#         6 -> 20  mit x_6 = x_20
#
# Die y-Koordinate ist die Tautemperatur von reinem Wasser beim jeweiligen
# Druck:
#
#     p_low  -> T8
#     p_high -> T10
#
# Damit entsprechen die vier Lösungspunkte:
#
#     1  = T_eq(p_low,  x3)
#     3  = T_eq(p_high, x3)
#     6  = T_eq(p_low,  x6)
#     20 = T_eq(p_high, x6)
#
# Die Darstellung ist damit unabhängig von den tatsächlichen
# Lösungstemperaturen T1, T3, T6 und T20 des Prozessmodells und zeigt
# ausschließlich die Gleichgewichtslage im Dühring-Diagramm.


# Geschlossener Prozesszug
_HEXAGON_STATE_ORDER: Tuple[str, ...] = (
    "6",
    "20",
    "3",
    "10",
    "8",
    "1",
    "6",
)

# Zusätzliche Linien:
# 1 -> 3 : Isostere der schwachen Lösung
# 6 -> 20: Isostere der starken Lösung
_DIAGONAL_STATE_PAIR_WEAK: Tuple[str, str] = ("1", "3")
_DIAGONAL_STATE_PAIR_STRONG: Tuple[str, str] = ("6", "20")


# Zustände bei p_low bzw. p_high.
# Diese Information bestimmt die y-Koordinate im Dühring-Diagramm.
_LOW_PRESSURE_STATES = {"1", "6", "8"}
_HIGH_PRESSURE_STATES = {"3", "20", "10"}


def _operating_point_positions(
    result: AWTResult,
) -> dict[str, Tuple[float, float]]:
    """Bestimmt die Positionen des AWT-Betriebspunkts im Dühring-Diagramm.

    Die x-Koordinaten der Lösungspunkte werden ausschließlich aus
    Druck und LiBr-Konzentration über die Gleichgewichtstemperatur
    bestimmt:

        T_eq = T_sat_solution_from_p_x(p, x)

    Dadurch liegen 1 und 3 exakt auf der Isostere der schwachen Lösung
    (x3) und 6 und 20 exakt auf der Isostere der starken Lösung (x6).

    Die y-Koordinate entspricht der Tautemperatur von reinem Wasser
    beim jeweiligen Druck.
    """

    s = result.states

    # ------------------------------------------------------------------
    # Druckniveaus aus dem Modell
    # ------------------------------------------------------------------
    p_low = s["8"]["p_Pa"]
    p_high = s["10"]["p_Pa"]

    # ------------------------------------------------------------------
    # Konzentrationen der beiden Lösungslinien
    #
    # Schwache Lösung:
    #     x_1 = x_3
    #
    # Starke Lösung:
    #     x_6 = x_20
    # ------------------------------------------------------------------
    x_weak = s["3"]["x_LiBr_mol"]
    x_strong = s["6"]["x_LiBr_mol"]

    # ------------------------------------------------------------------
    # Tautemperaturen von reinem Wasser bei den beiden Druckniveaus
    #
    # Diese bilden die y-Koordinaten im Dühring-Diagramm.
    # ------------------------------------------------------------------
    T8_C = kelvin_to_celsius(s["8"]["T_K"])
    T10_C = kelvin_to_celsius(s["10"]["T_K"])

    # ------------------------------------------------------------------
    # Gleichgewichtstemperaturen der Lösung
    #
    # WICHTIG:
    # Hier werden bewusst NICHT s["1"]["T_K"], s["3"]["T_K"],
    # s["6"]["T_K"] oder s["20"]["T_K"] verwendet.
    #
    # Stattdessen wird für jeden Schnittpunkt die Gleichgewichtstemperatur
    # aus Druck und Konzentration bestimmt.
    #
    # Die gleiche Funktion wird auch im Hauptmodell verwendet:
    #
    #     T3 = lp.T_sat_solution_from_p_x(p_high, x3)
    #     T6 = lp.T_sat_solution_from_p_x(p_low, x6)
    # ------------------------------------------------------------------

    # Schwache Lösung / Isostere x_weak = x3
    T1_eq_K = lp.T_sat_solution_from_p_x(p_low, x_weak)
    T3_eq_K = lp.T_sat_solution_from_p_x(p_high, x_weak)

    # Starke Lösung / Isostere x_strong = x6
    T6_eq_K = lp.T_sat_solution_from_p_x(p_low, x_strong)
    T20_eq_K = lp.T_sat_solution_from_p_x(p_high, x_strong)

    # ------------------------------------------------------------------
    # Umrechnung nach °C
    # ------------------------------------------------------------------
    T1_eq_C = kelvin_to_celsius(T1_eq_K)
    T3_eq_C = kelvin_to_celsius(T3_eq_K)
    T6_eq_C = kelvin_to_celsius(T6_eq_K)
    T20_eq_C = kelvin_to_celsius(T20_eq_K)

    # ------------------------------------------------------------------
    # x-Koordinate:
    #     Gleichgewichtstemperatur der Lösung
    #
    # y-Koordinate:
    #     Tautemperatur von reinem Wasser bei gleichem Druck
    # ------------------------------------------------------------------
    x_by_state = {
        "1": T1_eq_C,
        "3": T3_eq_C,
        "6": T6_eq_C,
        "20": T20_eq_C,

        # Reine Wasserzustände liegen auf der Wasserlinie.
        "8": T8_C,
        "10": T10_C,
    }

    y_by_state = {
        "1": T8_C,
        "6": T8_C,
        "8": T8_C,

        "3": T10_C,
        "20": T10_C,
        "10": T10_C,
    }

    state_ids = set(_HEXAGON_STATE_ORDER)

    return {
        sid: (x_by_state[sid], y_by_state[sid])
        for sid in state_ids
    }


def plot_duehring_operating_point(
    result: AWTResult,
    *,
    variant: Literal["mole", "mass"] = "mass",
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    run_checks: bool = False,
):
    """Zeichnet den AWT-Betriebspunkt im Dühring-Diagramm.

    Die Lösungspunkte werden über ihre Gleichgewichtstemperaturen aus
    Druck und LiBr-Konzentration bestimmt. Dadurch liegen die Verbindungen

        1 -> 3

    und

        6 -> 20

    exakt auf den entsprechenden Isosteren.

    Parameters
    ----------
    result:
        Ergebnisobjekt von `solve_awt()`.

    variant:
        "mass" für Massenanteil bzw. "mole" für Molanteil der
        dargestellten Isosteren.

    show:
        Öffnet ein interaktives Fenster, falls True.

    save_path:
        Optionaler Pfad zum Speichern der Grafik.

    dpi:
        Auflösung beim Speichern.

    run_checks:
        Führt vorab `run_self_checks()` aus.
    """

    if not result.solve_info.final_point_evaluable:
        raise ValueError(
            "Dühring-Diagramm kann nicht erzeugt werden: Endpunkt ist nicht "
            "physikalisch auswertbar "
            "(result.solve_info.final_point_evaluable=False)."
        )

    if run_checks:
        run_self_checks()

    fig, ax = create_duehring_figure(variant)

    positions = _operating_point_positions(result)

    # ------------------------------------------------------------------
    # Sechseck / Prozesspfad
    #
    # 6 -> 20 -> 3 -> 10 -> 8 -> 1 -> 6
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Darstellungseinstellungen für den AWT-Betriebspunkt
    # ------------------------------------------------------------------
    # Deutlich kontrastreichere Farbe als tab:orange.
    cycle_color = "tab:blue"

    # Anzeige-Namen der Zustände.
    #
    # Die internen State-IDs bleiben unverändert:
    #     "1"  -> "1*"
    #     "20" -> "4*"
    #
    # Dadurch muss an der eigentlichen Modell-/State-Logik nichts geändert
    # werden.
    state_labels = {
        "1": "1*",
        "3": "3",
        "6": "6",
        "20": "4*",
        "8": "8",
        "10": "10",
    }

    # ------------------------------------------------------------------
    # Sechseck / Prozesspfad
    #
    # 6 -> 20 -> 3 -> 10 -> 8 -> 1 -> 6
    # ------------------------------------------------------------------
    xs_hex = [positions[sid][0] for sid in _HEXAGON_STATE_ORDER]
    ys_hex = [positions[sid][1] for sid in _HEXAGON_STATE_ORDER]

    (cycle_handle,) = ax.plot(
        xs_hex,
        ys_hex,
        "o-",
        color=cycle_color,
        linewidth=2.4,
        markersize=6.5,
        zorder=12,
        label="AWT-Betriebspunkt",
    )

    # ------------------------------------------------------------------
    # Isostere der schwachen Lösung: 1 -> 3
    # ------------------------------------------------------------------
    x_weak = [
        positions[sid][0]
        for sid in _DIAGONAL_STATE_PAIR_WEAK
    ]
    y_weak = [
        positions[sid][1]
        for sid in _DIAGONAL_STATE_PAIR_WEAK
    ]

    ax.plot(
        x_weak,
        y_weak,
        "--",
        color=cycle_color,
        linewidth=1.5,
        zorder=11,
    )

    # ------------------------------------------------------------------
    # Isostere der starken Lösung: 6 -> 20
    # ------------------------------------------------------------------
    x_strong = [
        positions[sid][0]
        for sid in _DIAGONAL_STATE_PAIR_STRONG
    ]
    y_strong = [
        positions[sid][1]
        for sid in _DIAGONAL_STATE_PAIR_STRONG
    ]

    ax.plot(
        x_strong,
        y_strong,
        "--",
        color=cycle_color,
        linewidth=1.5,
        zorder=11,
    )

    # ------------------------------------------------------------------
    # Zustandsbeschriftungen
    # ------------------------------------------------------------------
    label_offset = {
        "1": (6, -12),
        "6": (6, -12),
        "8": (6, -12),
        "3": (6, 6),
        "20": (6, 6),
        "10": (6, 6),
    }

    for sid, (x, y) in positions.items():
        ax.annotate(
            state_labels.get(sid, sid),
            xy=(x, y),
            xytext=label_offset.get(sid, (6, 6)),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=cycle_color,
            zorder=13,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.90,
            ),
        )
    # ------------------------------------------------------------------
    # Referenzlinien für p_low und p_high
    # ------------------------------------------------------------------
    T8_C = kelvin_to_celsius(result.states["8"]["T_K"])
    T10_C = kelvin_to_celsius(result.states["10"]["T_K"])

    ax.axhline(
        T8_C,
        color=cycle_color,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
        zorder=5,
    )

    ax.axhline(
        T10_C,
        color=cycle_color,
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
        zorder=5,
    )
    # ------------------------------------------------------------------
    # Legende
    # ------------------------------------------------------------------
    existing_legend = ax.get_legend()

    if existing_legend is not None:
        handles = list(existing_legend.legend_handles) + [cycle_handle]
        labels_ = [
            t.get_text()
            for t in existing_legend.get_texts()
        ] + ["AWT-Betriebspunkt"]
    else:
        handles = [cycle_handle]
        labels_ = ["AWT-Betriebspunkt"]

    ax.legend(
        handles=handles,
        labels=labels_,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=8.5,
    )

    fig.suptitle(
        "AWT – Dühring-Diagramm mit Betriebspunkt",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # ------------------------------------------------------------------
    # Speichern
    # ------------------------------------------------------------------
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig