#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duehring diagrams for aqueous LiBr solutions, incl. AHT operating point.

This module contains, unchanged, the base functionality for producing
Duehring diagrams (isosteres per Patek & Klomfar, crystallization limit
per Albers/Boryta), plus `plot_duehring_operating_point()`, which draws
the operating point computed by `solve_aht()` (states 1,2,3,4,5,6,20) as a
cycle polygon into the diagram.

Positioning the operating point
-----------------------------------
On the Duehring axis, the y-coordinate of a solution state is the dew
temperature of pure water at its pressure. In the model this corresponds
exactly to states 8 (p_low) and 10 (p_high), since both lie by definition
on the pure-water saturation line (Q=0 and Q=1 respectively) and these
pressures define exactly that. No further conversion is needed:
    - States at p_low  (1, 6):                y = T(state 8)  [°C]
    - States at p_high (2, 3, 4, 5, 20):       y = T(state 10) [°C]
The x-coordinate is each state's solution temperature T(state n) [°C].

Process order of the drawn cycle
-------------------------------------
6 (desorber outlet, p_low) -> 5 (after solution pump, p_high) ->
4 (after SHEX preheating, p_high) -> 20 (after adiabatic pre-absorption,
p_high) -> 3 (absorber outlet, p_high) -> 2 (after SHEX cooling, p_high)
-> 1 (after throttle, p_low) -> back to 6 (desorption).

Note on the plot range
-----------------------------------
The default bounds (X_AXIS_MAX_C = 160 °C, P_RIGHT_MAX_MBAR = 2000 mbar)
come from the template. If your operating point (particularly at high
T_11 or high pressures) falls outside this range, adjust these two
constants at the top of the file.

Usage from the main script
-----------------------
    from Postprocessing.AHT_Duehring_Plot import plot_duehring_operating_point

    if ENABLE_DUEHRING_PLOT:
        plot_duehring_operating_point(result, save_path="Postprocessing/Plots/AHT_Duehring_Diagramm.png")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
except ImportError as exc:  # pragma: no cover - only if packages are missing
    raise SystemExit(
        "Missing Python package. Install the dependencies with:\n"
        "    python -m pip install numpy matplotlib"
    ) from exc

from Models.AHT_Pinch_Point import AHTResult, kelvin_to_celsius
import Thermodynamic_Properties.libr_props as lp


# =============================================================================
# Constants and plot configuration
# =============================================================================

M_LIBR = 0.08685       # kg/mol, matching the implementation used here
M_H2O = 0.018015268    # kg/mol
T_CRIT_H2O = 647.096   # K
P_CRIT_H2O = 22.064e6  # Pa
T0_C = 273.15          # K -> °C

X_AXIS_MIN_C = 0.0
X_AXIS_MAX_C = 160.0
X_AXIS_MAJOR_C = 20.0

# The top edge of the left y-axis is derived from p = 2000 mbar for pure
# water, so the left temperature axis and right pressure axis line up exactly.
P_RIGHT_MAX_MBAR = 2000.0

# Density of the isostere family. Every 10 percentage points: black; in between: gray.
MAJOR_COMPOSITION_STEP_PERCENT = 10.0
MINOR_COMPOSITION_STEP_PERCENT = 2.5

# Albers/Boryta: valid range of the T_cr(w) polynomial
W_CRYST_MIN = 0.57
W_CRYST_MAX = 0.70

# Upper bound of the plotted concentrations. Also the upper validity limit
# of the crystallization correlation used here.
W_PLOT_MAX = W_CRYST_MAX
X_PLOT_MAX = None  # set after the conversion function is defined

# Right pressure scale, matching the accompanying reference figures.
PRESSURE_TICKS_MBAR = np.array(
    [7, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000, 1500, 2000],
    dtype=float,
)

# Patek/Klomfar, Eq. (1), Table 4: pressure correlation
PAT_A = np.array(
    [-2.41303e2, 1.91750e7, -1.75521e8, 3.25430e7,
      3.92571e2, -2.12626e3, 1.85127e8, 1.91216e3],
    dtype=float,
)
PAT_M = np.array([3, 4, 4, 8, 1, 1, 4, 6], dtype=float)
PAT_N = np.array([0, 5, 6, 3, 0, 2, 6, 0], dtype=float)
PAT_T = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# Patek/Klomfar, Eq. (28), Table 11: vapor pressure of pure water
WATER_ALPHA = np.array(
    [-7.85951783, 1.84408259, -11.7866497,
      22.6807411, -15.9618719, 1.80122502],
    dtype=float,
)
WATER_BETA = np.array([1.0, 1.5, 3.0, 3.5, 4.0, 7.5], dtype=float)

# Albers (2019), Eq. (2.52), Table 2.8: T_cr as a function of w
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
# Concentration conversion
# =============================================================================

def mass_fraction_from_mole_fraction(x_libr: float | np.ndarray) -> float | np.ndarray:
    """LiBr mass fraction w from LiBr mole fraction x."""
    x = np.asarray(x_libr, dtype=float)
    denominator = x * M_LIBR + (1.0 - x) * M_H2O
    w = x * M_LIBR / denominator
    return float(w) if w.ndim == 0 else w


def mole_fraction_from_mass_fraction(w_libr: float | np.ndarray) -> float | np.ndarray:
    """LiBr mole fraction x from LiBr mass fraction w."""
    w = np.asarray(w_libr, dtype=float)
    denominator = M_LIBR - w * (M_LIBR - M_H2O)
    x = w * M_H2O / denominator
    return float(x) if x.ndim == 0 else x


X_PLOT_MAX = float(mole_fraction_from_mass_fraction(W_PLOT_MAX))


# =============================================================================
# Patek/Klomfar correlation and Duehring transformation
# =============================================================================

def _patek_duehring_coefficients(x_libr_mol: float) -> tuple[float, float]:
    """Returns A(x) [K] and B(x) [K] from Albers Eq. (2.40).

    With theta = T_H2O at the same pressure:
        theta = T_sol - A - B * T_sol / T_crit
    which gives the analytical Duehring line:
        T_sol = T_crit/(T_crit-B) * theta + T_crit*A/(T_crit-B)
    """
    x = float(x_libr_mol)
    if not (0.0 <= x < 0.4):
        raise ValueError(f"LiBr mole fraction x={x:.8f} is outside 0 <= x < 0.4.")

    terms = PAT_A * x**PAT_M * (0.4 - x) ** PAT_N
    a_term = float(np.sum(terms[PAT_T == 0.0]))
    b_term = float(np.sum(terms[PAT_T == 1.0]))
    return a_term, b_term


def solution_boiling_temperature_c(
    water_dew_temperature_c: float | np.ndarray,
    x_libr_mol: float,
) -> float | np.ndarray:
    """Solution boiling temperature [°C] for a given pure-water dew
    temperature [°C]."""
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
    """Inverse Duehring relation: T_H2O [°C] from T_sol [°C] and x."""
    t_solution_k = np.asarray(solution_temperature_c, dtype=float) + T0_C
    a_term, b_term = _patek_duehring_coefficients(x_libr_mol)
    theta_k = t_solution_k * (1.0 - b_term / T_CRIT_H2O) - a_term
    result = theta_k - T0_C
    return float(result) if result.ndim == 0 else result


def water_saturation_pressure_pa(temperature_k: float | np.ndarray) -> float | np.ndarray:
    """Saturation pressure of pure water per Patek/Klomfar Eq. (28) [Pa]."""
    t = np.asarray(temperature_k, dtype=float)
    if np.any((t <= 0.0) | (t >= T_CRIT_H2O)):
        raise ValueError("Water temperature must lie between 0 K and T_crit.")

    tau = 1.0 - t / T_CRIT_H2O
    exponent = np.zeros_like(t, dtype=float)
    for alpha, beta in zip(WATER_ALPHA, WATER_BETA):
        exponent += alpha * tau**beta
    p = P_CRIT_H2O * np.exp((T_CRIT_H2O / t) * exponent)
    return float(p) if p.ndim == 0 else p


def water_saturation_temperature_c_from_pressure_mbar(p_mbar: float) -> float:
    """Inverse of the water vapor-pressure equation via robust bisection [°C]."""
    target_pa = float(p_mbar) * 100.0
    if target_pa <= 0.0:
        raise ValueError("The pressure must be positive.")

    lo_k = 250.0
    hi_k = T_CRIT_H2O - 1.0e-8
    p_lo = float(water_saturation_pressure_pa(lo_k))
    p_hi = float(water_saturation_pressure_pa(hi_k))
    if not (p_lo <= target_pa <= p_hi):
        raise ValueError(f"p={p_mbar:g} mbar is outside the invertible range.")

    for _ in range(120):
        mid_k = 0.5 * (lo_k + hi_k)
        p_mid = float(water_saturation_pressure_pa(mid_k))
        if p_mid < target_pa:
            lo_k = mid_k
        else:
            hi_k = mid_k
    return 0.5 * (lo_k + hi_k) - T0_C


# =============================================================================
# Crystallization limit per Albers/Boryta
# =============================================================================

def crystallization_temperature_c_from_mass_fraction(
    w_libr: float | np.ndarray,
) -> float | np.ndarray:
    """Crystallization temperature T_cr(w) [°C], Albers Eq. (2.52).

    Valid range: 0.57 < w < 0.70 kg/kg.
    """
    w = np.asarray(w_libr, dtype=float)
    if np.any((w < W_CRYST_MIN) | (w > W_CRYST_MAX)):
        raise ValueError(
            f"T_cr(w) is only valid here for {W_CRYST_MIN:.2f} <= w <= {W_CRYST_MAX:.2f}."
        )
    w_b = (w - 0.64794) / 0.044858
    t_cr = np.zeros_like(w_b, dtype=float)
    for power, coefficient in enumerate(ALBERS_T_COEFF):
        t_cr += coefficient * w_b**power
    return float(t_cr) if t_cr.ndim == 0 else t_cr


def crystallization_water_dew_temperature_c(w_libr: float | np.ndarray) -> float | np.ndarray:
    """y-coordinate of the crystallization limit in the Duehring diagram [°C]."""
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
# Plausibility checks
# =============================================================================

def run_self_checks() -> None:
    """Checks the implementation against reference values from Patek/Klomfar Table 9."""
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
                "Patek self-check failed: "
                f"x={x_mol:.8f}, T={t_k:.3f} K, "
                f"p_calc={p_calculated:.6f} Pa, p_ref={p_reference:.6f} Pa, "
                f"rel. error={relative_error:.3e}."
            )

    # The upper Albers validity edge is, per the report, roughly at 101 °C.
    if not math.isclose(
        float(crystallization_temperature_c_from_mass_fraction(0.70)),
        100.9689444,
        rel_tol=0.0,
        abs_tol=2.0e-6,
    ):
        raise RuntimeError("Albers crystallization correlation returns unexpected values.")


# =============================================================================
# Plot helper functions
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
    # For the mole-fraction variant, 5-percent ticks are appropriate, since
    # with the given x-axis only the intersections up to about 15 mol-% are visible.
    step_percent = 5.0 if variant == "mole" else 10.0
    maximum = X_PLOT_MAX if variant == "mole" else W_PLOT_MAX
    start = 0.0 if variant == "mole" else 0.20
    return np.arange(start, maximum + 0.5 * step_percent / 100.0, step_percent / 100.0)


def _line_rotation_degrees(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """Computes the visual rotation of a line in screen coordinates."""
    if len(x_data) < 2:
        return 0.0
    p0 = ax.transData.transform((x_data[0], y_data[0]))
    p1 = ax.transData.transform((x_data[-1], y_data[-1]))
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def create_duehring_figure(variant: Literal["mole", "mass"]) -> Tuple[plt.Figure, plt.Axes]:
    """Produces one plot variant and returns (Figure, primary axis)."""
    if variant not in {"mole", "mass"}:
        raise ValueError("variant must be 'mole' or 'mass'.")

    y_axis_max_c = water_saturation_temperature_c_from_pressure_mbar(P_RIGHT_MAX_MBAR)
    y_values_full = np.linspace(0.0, y_axis_max_c, 900)

    fig, ax = plt.subplots(figsize=(10.8, 7.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.875, bottom=0.115, top=0.865)

    ax.set_xlim(X_AXIS_MIN_C, X_AXIS_MAX_C)
    ax.set_ylim(0.0, y_axis_max_c)
    ax.set_xlabel(
        r"Boiling temperature $T_{\mathrm{H_2O/LiBr}}^{\mathrm{LV}}$ [°C]",
        fontsize=12,
    )
    ax.set_ylabel(
        r"Dew temperature $T_{\mathrm{H_2O}}^{\mathrm{LV}}$ [°C]",
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
    # Isosteres
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

        # No isosteres are drawn above the Albers/Boryta range, so the
        # crystallization limit is never extrapolated beyond its validity.
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
    # Crystallization limit
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

    # Water-line label
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
        "Water",
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
            "Crystallization limit",
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
    # Right pressure axis
    # -------------------------------------------------------------------------
    pressure_positions = np.array(
        [water_saturation_temperature_c_from_pressure_mbar(p) for p in PRESSURE_TICKS_MBAR]
    )
    pressure_mask = (pressure_positions >= 0.0) & (pressure_positions <= y_axis_max_c + 1.0e-8)

    # Additional horizontal guide lines for the discrete pressure levels.
    # These complement the regular temperature grid without obscuring the isosteres.
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
    ax_right.set_ylabel(r"Equilibrium pressure $p^{\mathrm{LV}}$ [mbar]", fontsize=12)
    ax_right.tick_params(direction="in", which="major", labelsize=10)

    # -------------------------------------------------------------------------
    # Top concentration axis: intersections with the top plot edge
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
    composition_name = r"mole fraction $x^{\mathrm{LiBr}}$" if variant == "mole" else r"mass fraction $w^{\mathrm{LiBr}}$"
    ax_top.set_xlabel(f"LiBr {composition_name} of the isosteres [%]", labelpad=0, fontsize=10.5)
    # The axis label is deliberately positioned close to the top axis and
    # left of the concentration ticks, so it doesn't read like a plot title.
    ax_top.xaxis.set_label_coords(0.5, 1.025)
    ax_top.tick_params(direction="in", which="major", pad=1, labelsize=9)

    # Single combined legend explaining all line styles used.
    legend_handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.65,
            label="Isosteres in 10% steps (Patek & Klomfar)",
        ),
        Line2D(
            [0], [0],
            color="0.38",
            linewidth=0.65,
            label="Isosteres in 2.5% steps (Patek & Klomfar)",
        ),
        Line2D(
            [0], [0],
            color="darkred",
            linewidth=2.2,
            label="Crystallization limit (Albers/Boryta)",
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
    """Saves a figure in all requested formats."""
    written: list[Path] = []
    for extension in formats:
        ext = extension.lower().lstrip(".")
        if ext not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported output format: {extension}")
        output_path = output_base.with_suffix(f".{ext}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        written.append(output_path)
    return written


# =============================================================================
# AHT operating-point overlay
# =============================================================================
#
# Draws the AHT process in the Duehring diagram.
#
# The x-coordinate of a solution state is NOT the actual process/outlet
# temperature from the model, but the equilibrium temperature of the LiBr
# solution at the respective pressure and solution concentration:
#
#     T_eq = T_sat_solution_from_p_x(p, x_LiBr)
#
# This keeps the two concentration lines (isosteres) consistent:
#
#     weak solution:
#         1 -> 3   with x_1 = x_3
#
#     strong solution:
#         6 -> 20  with x_6 = x_20
#
# The y-coordinate is the dew temperature of pure water at the respective
# pressure:
#
#     p_low  -> T8
#     p_high -> T10
#
# So the four solution points correspond to:
#
#     1  = T_eq(p_low,  x3)
#     3  = T_eq(p_high, x3)
#     6  = T_eq(p_low,  x6)
#     20 = T_eq(p_high, x6)
#
# This representation is therefore independent of the actual solution
# temperatures T1, T3, T6, T20 from the process model and shows only the
# equilibrium position in the Duehring diagram.


# Closed process loop
_HEXAGON_STATE_ORDER: Tuple[str, ...] = (
    "6",
    "20",
    "3",
    "10",
    "8",
    "1",
    "6",
)

# Additional lines:
# 1 -> 3 : isostere of the weak solution
# 6 -> 20: isostere of the strong solution
_DIAGONAL_STATE_PAIR_WEAK: Tuple[str, str] = ("1", "3")
_DIAGONAL_STATE_PAIR_STRONG: Tuple[str, str] = ("6", "20")


# States at p_low and p_high. This determines the y-coordinate in the
# Duehring diagram.
_LOW_PRESSURE_STATES = {"1", "6", "8"}
_HIGH_PRESSURE_STATES = {"3", "20", "10"}


def _operating_point_positions(
    result: AHTResult,
) -> dict[str, Tuple[float, float]]:
    """Determines the positions of the AHT operating point in the Duehring
    diagram.

    The x-coordinates of the solution points are determined exclusively
    from pressure and LiBr concentration via the equilibrium temperature:

        T_eq = T_sat_solution_from_p_x(p, x)

    This places 1 and 3 exactly on the weak-solution isostere (x3), and 6
    and 20 exactly on the strong-solution isostere (x6).

    The y-coordinate is the dew temperature of pure water at the
    respective pressure.
    """

    s = result.states

    # ------------------------------------------------------------------
    # Pressure levels from the model
    # ------------------------------------------------------------------
    p_low = s["8"]["p_Pa"]
    p_high = s["10"]["p_Pa"]

    # ------------------------------------------------------------------
    # Concentrations of the two solution lines
    #
    # Weak solution:
    #     x_1 = x_3
    #
    # Strong solution:
    #     x_6 = x_20
    # ------------------------------------------------------------------
    x_weak = s["3"]["x_LiBr_mol"]
    x_strong = s["6"]["x_LiBr_mol"]

    # ------------------------------------------------------------------
    # Dew temperatures of pure water at the two pressure levels.
    # These form the y-coordinates in the Duehring diagram.
    # ------------------------------------------------------------------
    T8_C = kelvin_to_celsius(s["8"]["T_K"])
    T10_C = kelvin_to_celsius(s["10"]["T_K"])

    # ------------------------------------------------------------------
    # Equilibrium temperatures of the solution.
    #
    # IMPORTANT: s["1"]["T_K"], s["3"]["T_K"], s["6"]["T_K"], and
    # s["20"]["T_K"] are deliberately NOT used here. Instead, the
    # equilibrium temperature is determined from pressure and
    # concentration for each intersection point.
    #
    # The same function is also used in the main model:
    #
    #     T3 = lp.T_sat_solution_from_p_x(p_high, x3)
    #     T6 = lp.T_sat_solution_from_p_x(p_low, x6)
    # ------------------------------------------------------------------

    # Weak solution / isostere x_weak = x3
    T1_eq_K = lp.T_sat_solution_from_p_x(p_low, x_weak)
    T3_eq_K = lp.T_sat_solution_from_p_x(p_high, x_weak)

    # Strong solution / isostere x_strong = x6
    T6_eq_K = lp.T_sat_solution_from_p_x(p_low, x_strong)
    T20_eq_K = lp.T_sat_solution_from_p_x(p_high, x_strong)

    # ------------------------------------------------------------------
    # Convert to °C
    # ------------------------------------------------------------------
    T1_eq_C = kelvin_to_celsius(T1_eq_K)
    T3_eq_C = kelvin_to_celsius(T3_eq_K)
    T6_eq_C = kelvin_to_celsius(T6_eq_K)
    T20_eq_C = kelvin_to_celsius(T20_eq_K)

    # ------------------------------------------------------------------
    # x-coordinate: equilibrium temperature of the solution
    # y-coordinate: dew temperature of pure water at the same pressure
    # ------------------------------------------------------------------
    x_by_state = {
        "1": T1_eq_C,
        "3": T3_eq_C,
        "6": T6_eq_C,
        "20": T20_eq_C,

        # Pure-water states lie on the water line.
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
    result: AHTResult,
    *,
    variant: Literal["mole", "mass"] = "mass",
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    run_checks: bool = False,
):
    """Draws the AHT operating point in the Duehring diagram.

    The solution points are determined via their equilibrium temperatures
    from pressure and LiBr concentration. This places the connections

        1 -> 3

    and

        6 -> 20

    exactly on the corresponding isosteres.

    Parameters
    ----------
    result:
        Result object from `solve_aht()`.

    variant:
        "mass" for mass fraction or "mole" for mole fraction of the
        plotted isosteres.

    show:
        Opens an interactive window if True.

    save_path:
        Optional path to save the figure under.

    dpi:
        Resolution when saving.

    run_checks:
        Runs `run_self_checks()` beforehand.
    """

    if not result.solve_info.final_point_evaluable:
        raise ValueError(
            "Cannot produce the Duehring diagram: the end point is not "
            "physically evaluable "
            "(result.solve_info.final_point_evaluable=False)."
        )

    if run_checks:
        run_self_checks()

    fig, ax = create_duehring_figure(variant)

    positions = _operating_point_positions(result)

    # ------------------------------------------------------------------
    # Display settings for the AHT operating point
    # ------------------------------------------------------------------
    # Noticeably higher-contrast color than tab:orange.
    cycle_color = "tab:blue"

    # Display names of the states.
    #
    # The internal state IDs remain unchanged:
    #     "1"  -> "1*"
    #     "20" -> "4*"
    #
    # This avoids any need to change the actual model/state logic.
    state_labels = {
        "1": "1*",
        "3": "3",
        "6": "6",
        "20": "4*",
        "8": "8",
        "10": "10",
    }

    # ------------------------------------------------------------------
    # Hexagon / process path
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
        label="AHT operating point",
    )

    # ------------------------------------------------------------------
    # Isostere of the weak solution: 1 -> 3
    # ------------------------------------------------------------------
    x_weak = [positions[sid][0] for sid in _DIAGONAL_STATE_PAIR_WEAK]
    y_weak = [positions[sid][1] for sid in _DIAGONAL_STATE_PAIR_WEAK]

    ax.plot(
        x_weak,
        y_weak,
        "--",
        color=cycle_color,
        linewidth=1.5,
        zorder=11,
    )

    # ------------------------------------------------------------------
    # Isostere of the strong solution: 6 -> 20
    # ------------------------------------------------------------------
    x_strong = [positions[sid][0] for sid in _DIAGONAL_STATE_PAIR_STRONG]
    y_strong = [positions[sid][1] for sid in _DIAGONAL_STATE_PAIR_STRONG]

    ax.plot(
        x_strong,
        y_strong,
        "--",
        color=cycle_color,
        linewidth=1.5,
        zorder=11,
    )

    # ------------------------------------------------------------------
    # State labels
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
    # Reference lines for p_low and p_high
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
    # Legend
    # ------------------------------------------------------------------
    existing_legend = ax.get_legend()

    if existing_legend is not None:
        handles = list(existing_legend.legend_handles) + [cycle_handle]
        labels_ = [
            t.get_text()
            for t in existing_legend.get_texts()
        ] + ["AHT operating point"]
    else:
        handles = [cycle_handle]
        labels_ = ["AHT operating point"]

    ax.legend(
        handles=handles,
        labels=labels_,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=8.5,
    )

    # ------------------------------------------------------------------
    # Save
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


# =============================================================================
# Multiple AHT operating points in the same Duehring diagram (one color each)
# =============================================================================
#
# Like plot_duehring_operating_point(), but for a set of results (e.g. one
# operating point per examined waste-heat temperature from
# Design_Point/AHT_feasibility_sweep.py) -- each process is drawn as its
# own hexagon in its own color, so the operating ranges of different
# waste-heat temperatures can be compared in the same Duehring diagram.
# Individual state text labels (1*, 3, 6, ...) are omitted here -- with
# several overlaid hexagons that would just become unreadable. The
# color-to-waste-heat-temperature mapping is in the legend.

def plot_duehring_multi_operating_points(
    entries: Iterable[Tuple[float, AHTResult]],
    *,
    variant: Literal["mole", "mass"] = "mass",
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    run_checks: bool = False,
    cmap_name: str = "coolwarm",
    title: str = "AHT - Duehring diagram, multiple waste-heat temperatures",
):
    """Draws multiple AHT operating points (one (T_waste_C, AHTResult) pair
    each from `entries`) as differently colored hexagons in a shared
    Duehring diagram.

    Parameters
    ----------
    entries:
        Iterable of (T_waste_C, result) pairs. result must come from
        solve_aht() (see plot_duehring_operating_point()). T_waste_C is
        used ONLY for the color mapping (cold->warm gradient) and the
        legend label, not re-derived from `result` (T_waste is an
        external input, not an internal model state).
    cmap_name:
        Matplotlib colormap for the color mapping by T_waste_C.
        "coolwarm" maps low waste-heat temperatures to blue and high ones
        to red.
    """
    entries = sorted(entries, key=lambda e: e[0])
    if not entries:
        raise ValueError("plot_duehring_multi_operating_points: entries is empty.")

    for T_waste_C, result in entries:
        if not result.solve_info.final_point_evaluable:
            raise ValueError(
                f"Cannot produce the Duehring diagram: the end point at "
                f"T_waste={T_waste_C:.2f} °C is not physically evaluable "
                "(result.solve_info.final_point_evaluable=False)."
            )

    if run_checks:
        run_self_checks()

    fig, ax = create_duehring_figure(variant)

    T_values = [e[0] for e in entries]
    T_lo, T_hi = min(T_values), max(T_values)
    T_span = (T_hi - T_lo) or 1.0
    cmap = plt.get_cmap(cmap_name)

    process_handles: list[Line2D] = []
    process_labels: list[str] = []

    for T_waste_C, result in entries:
        color = cmap((T_waste_C - T_lo) / T_span)
        positions = _operating_point_positions(result)

        xs_hex = [positions[sid][0] for sid in _HEXAGON_STATE_ORDER]
        ys_hex = [positions[sid][1] for sid in _HEXAGON_STATE_ORDER]
        (cycle_handle,) = ax.plot(
            xs_hex, ys_hex, "o-", color=color, linewidth=2.0, markersize=4.5,
            zorder=12, label=f"T_waste = {T_waste_C:.0f} °C",
        )

        for pair in (_DIAGONAL_STATE_PAIR_WEAK, _DIAGONAL_STATE_PAIR_STRONG):
            x_pair = [positions[sid][0] for sid in pair]
            y_pair = [positions[sid][1] for sid in pair]
            ax.plot(x_pair, y_pair, "--", color=color, linewidth=1.1, alpha=0.85, zorder=11)

        process_handles.append(cycle_handle)
        process_labels.append(f"T_waste = {T_waste_C:.0f} °C")

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        handles = list(existing_legend.legend_handles) + process_handles
        labels_ = [t.get_text() for t in existing_legend.get_texts()] + process_labels
    else:
        handles, labels_ = process_handles, process_labels

    ax.legend(
        handles=handles, labels=labels_,
        loc="upper left", frameon=True, framealpha=0.94, fontsize=8.5,
    )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig
