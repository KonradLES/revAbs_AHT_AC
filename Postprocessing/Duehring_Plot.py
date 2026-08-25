#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Erzeugt zwei Dühring-Diagramme für wässrige LiBr-Lösungen.

Varianten
---------
1. Isosteren mit konstantem LiBr-Molanteil
2. Isosteren mit konstantem LiBr-Massenanteil

Die Siedelinien werden aus der Dampfdruckkorrelation von Pátek und Klomfar
(2006) analytisch in Dühring-Form ausgewertet. Die Kristallisationsgrenze wird
mit dem Regressionspolynom von Albers (2019) auf Basis der Messdaten von
Boryta (1970) berechnet. Isosteren werden an dieser Grenze abgeschnitten.

Benötigte Pakete
----------------
- numpy
- matplotlib

Aufruf
------
    python duehring_diagramm.py

Optionen, zum Beispiel:
    python duehring_diagramm.py --output-dir plots --formats png pdf svg --dpi 300
    python duehring_diagramm.py --variant mole --show

Standardmäßig werden vier Dateien erzeugt:
- duehring_diagramm_molanteil.png
- duehring_diagramm_molanteil.pdf
- duehring_diagramm_massenanteil.png
- duehring_diagramm_massenanteil.pdf

Hinweis zur oberen Achse
------------------------
Die Konzentration ist keine globale Funktion der unteren Temperaturachse.
Wie in üblichen Dühring-Darstellungen werden die Ticks der oberen Achse daher
an den Schnittpunkten der jeweiligen Isosteren mit dem oberen Diagrammrand
positioniert.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Literal

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


def create_duehring_figure(variant: Literal["mole", "mass"]) -> plt.Figure:
    """Erzeugt eine Diagrammvariante und gibt die Matplotlib-Figure zurück."""
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
    ax_top.set_xlabel(f"LiBr-{composition_name} der Isosteren [%]", labelpad=0, fontsize=10.5) #ax_top.set_xlabel(f"LiBr-{composition_name} der Isosteren [%]", labelpad=0, fontsize=10.5)
    # Die Achsenbezeichnung wird bewusst nahe an der oberen Achse und links von
    # den Konzentrationsticks positioniert, damit sie nicht wie ein Diagrammtitel wirkt.
    ax_top.xaxis.set_label_coords(0.5, 1.025)
    ax_top.tick_params(direction="in", which="major", pad=1, labelsize=9)

    # Vollständige gemeinsame Legende: Quellen-/Varianteninformation und
    # Erklärung aller verwendeten Linienarten in einem einzigen Feld.
    variant_text = (
        "konstanter LiBr-Molanteil"
        if variant == "mole"
        else "konstanter LiBr-Massenanteil"
    )
    
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

    return fig


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
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        written.append(output_path)
    return written


# =============================================================================
# Kommandozeile
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt Dühring-Diagramme für H2O/LiBr nach Pátek/Klomfar und Albers/Boryta."
    )
    parser.add_argument(
        "--variant",
        choices=("both", "mole", "mass"),
        default="both",
        help="Zu erzeugende Variante (Standard: both).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Ausgabeverzeichnis (Standard: aktuelles Verzeichnis).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        help="Ausgabeformate aus png, pdf, svg (Standard: png pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Auflösung der PNG-Dateien (Standard: 300 dpi).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Diagramme zusätzlich interaktiv anzeigen.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Interne Referenzwertprüfungen überspringen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_checks:
        run_self_checks()

    variants: tuple[Literal["mole", "mass"], ...]
    if args.variant == "both":
        variants = ("mole", "mass")
    elif args.variant == "mole":
        variants = ("mole",)
    else:
        variants = ("mass",)

    all_written: list[Path] = []
    figures: list[plt.Figure] = []
    for variant in variants:
        fig = create_duehring_figure(variant)
        figures.append(fig)
        suffix = "molanteil" if variant == "mole" else "massenanteil"
        output_base = args.output_dir / f"duehring_diagramm_{suffix}"
        all_written.extend(save_figure(fig, output_base, args.formats, args.dpi))

    for path in all_written:
        print(path.resolve())

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
