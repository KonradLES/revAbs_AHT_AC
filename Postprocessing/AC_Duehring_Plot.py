#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dühring-Diagramm-Überlagerung für AC-Betriebspunkte.

Analogon zu Postprocessing/AHT_Duehring_Plot.py, aber für die
Absorptionskältemaschine. Die generische Diagrammgrundlage (Isosteren nach
Pátek & Klomfar, Kristallisationsgrenze nach Albers/Boryta, Achsen/Gitter)
wird UNVERÄNDERT von dort importiert (create_duehring_figure) -- nur die
Positionierung des Betriebspunkts im Diagramm ist AC-spezifisch, weil die
AC eine andere Zuordnung von Zustandsnummern zu Druckniveaus hat als der AHT
(siehe Docstring von AC_feasibility_sweep.py, Abschnitt "Rollentausch").

Sechseck-Geometrie (Analogon zum AHT-Sechseck, siehe AHT_Duehring_Plot.py)
----------------------------------------------------------------------------
Bei der AC sind Desorber + Kondensator auf der HOHEN Druckseite, Absorber +
Verdampfer auf der NIEDRIGEN -- genau umgekehrt zum AHT. Das vertauscht auch,
welche state-IDs die "reinen Wasser"-Eckpunkte des Sechsecks liefern:
    AHT: Zustand 8 -> p_low,  Zustand 10 -> p_high
    AC: Zustand 8 -> p_high, Zustand 10 -> p_low
(Beides sind bei beiden Modellen per Definition Sättigungszustände von reinem
Wasser -- Kondensatoraustritt Q=0 bzw. Verdampferaustritt Q=1 -- daher direkt
als y-Koordinate im Dühring-Diagramm nutzbar, ohne Umrechnung.)

Die beiden Lösungs-Isosteren der AC:
    - Schwache Lösung (x1 = x3): Zustand 1 (p_low, Absorberaustritt) und
      Zustand 3 (p_high, nach SHEX-Vorwärmung, Desorbereintritt)
    - Starke Lösung (x4 = x6): Zustand 4 (p_high, Desorberaustritt) und
      Zustand 6 (p_low, nach SHEX-Abkühlung + Drossel, Absorbereintritt)

Sechseck-Reihenfolge (analog zur AHT-Struktur: starke Isostere rauf, schwache
Isostere bei Hochdruck, Wasser-Ecke Hochdruck, Wasser-Ecke Niederdruck,
schwache Isostere bei Niederdruck, zurück zur starken Isostere):
    6 (stark, p_low) -> 4 (stark, p_high) -> 3 (schwach, p_high) ->
    8 (Wasser, p_high) -> 10 (Wasser, p_low) -> 1 (schwach, p_low) -> 6

Nutzung
-------
    from Postprocessing.AC_Duehring_Plot import plot_duehring_multi_operating_points
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Postprocessing.AHT_Duehring_Plot import create_duehring_figure, run_self_checks
from Models.AC_Pinch_Point import AHTResult, kelvin_to_celsius
import Thermodynamic_Properties.libr_props as lp


# Geschlossener Prozesszug (siehe Modul-Docstring)
_HEXAGON_STATE_ORDER: Tuple[str, ...] = (
    "6",
    "4",
    "3",
    "8",
    "10",
    "1",
    "6",
)

# Zusätzliche Linien:
# 1 -> 3 : Isostere der schwachen Lösung
# 4 -> 6 : Isostere der starken Lösung
_DIAGONAL_STATE_PAIR_WEAK: Tuple[str, str] = ("1", "3")
_DIAGONAL_STATE_PAIR_STRONG: Tuple[str, str] = ("4", "6")


def _operating_point_positions(result: AHTResult) -> dict[str, Tuple[float, float]]:
    """Bestimmt die Positionen des AC-Betriebspunkts im Dühring-Diagramm.

    Wie beim AHT-Analogon: x-Koordinaten der Lösungspunkte kommen
    ausschliesslich aus Druck + LiBr-Konzentration über die
    Gleichgewichtstemperatur T_eq = T_sat_solution_from_p_x(p, x) -- damit
    liegen 1/3 exakt auf der Isostere der schwachen, 4/6 exakt auf der
    Isostere der starken Lösung. y-Koordinate = Tautemperatur von reinem
    Wasser beim jeweiligen Druck."""

    s = result.states

    p_high = s["8"]["p_Pa"]
    p_low = s["10"]["p_Pa"]

    x_weak = s["3"]["x_LiBr_mol"]
    x_strong = s["4"]["x_LiBr_mol"]

    T8_C = kelvin_to_celsius(s["8"]["T_K"])
    T10_C = kelvin_to_celsius(s["10"]["T_K"])

    T1_eq_C = kelvin_to_celsius(lp.T_sat_solution_from_p_x(p_low, x_weak))
    T3_eq_C = kelvin_to_celsius(lp.T_sat_solution_from_p_x(p_high, x_weak))
    T4_eq_C = kelvin_to_celsius(lp.T_sat_solution_from_p_x(p_high, x_strong))
    T6_eq_C = kelvin_to_celsius(lp.T_sat_solution_from_p_x(p_low, x_strong))

    x_by_state = {
        "1": T1_eq_C,
        "3": T3_eq_C,
        "4": T4_eq_C,
        "6": T6_eq_C,
        "8": T8_C,
        "10": T10_C,
    }
    y_by_state = {
        "1": T10_C,
        "6": T10_C,
        "10": T10_C,
        "3": T8_C,
        "4": T8_C,
        "8": T8_C,
    }

    state_ids = set(_HEXAGON_STATE_ORDER)
    return {sid: (x_by_state[sid], y_by_state[sid]) for sid in state_ids}


def plot_duehring_multi_operating_points(
    entries: Iterable[Tuple[float, AHTResult]],
    *,
    variant: Literal["mole", "mass"] = "mass",
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    run_checks: bool = False,
    cmap_name: str = "coolwarm",
    title: str = "AC – Dühring-Diagramm, mehrere Rückkühltemperaturen",
):
    """Zeichnet mehrere AC-Betriebspunkte (je ein (T_rueck_C, AHTResult)-Paar
    aus `entries`) als farblich unterschiedene Sechsecke in ein gemeinsames
    Dühring-Diagramm. Analogon zu
    AHT_Duehring_Plot.plot_duehring_multi_operating_points().

    Parameters
    ----------
    entries:
        Iterable von (T_rueck_C, result)-Paaren. result muss von solve_aht()
        (Models.AC_Pinch_Point) stammen. T_rueck_C wird NUR für die
        Farbzuordnung (Verlauf kalt->warm) und die Legendenbeschriftung
        verwendet.
    cmap_name:
        Matplotlib-Colormap für die Farbzuordnung nach T_rueck_C.
        "coolwarm" bildet niedrige Rückkühltemperaturen blau und hohe rot ab.
    """
    entries = sorted(entries, key=lambda e: e[0])
    if not entries:
        raise ValueError("plot_duehring_multi_operating_points: entries ist leer.")

    for T_rueck_C, result in entries:
        if not result.solve_info.final_point_evaluable:
            raise ValueError(
                f"Dühring-Diagramm kann nicht erzeugt werden: Endpunkt bei "
                f"T_rueck={T_rueck_C:.2f} °C ist nicht physikalisch auswertbar "
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

    for T_rueck_C, result in entries:
        color = cmap((T_rueck_C - T_lo) / T_span)
        positions = _operating_point_positions(result)

        xs_hex = [positions[sid][0] for sid in _HEXAGON_STATE_ORDER]
        ys_hex = [positions[sid][1] for sid in _HEXAGON_STATE_ORDER]
        (cycle_handle,) = ax.plot(
            xs_hex, ys_hex, "o-", color=color, linewidth=2.0, markersize=4.5,
            zorder=12, label=f"T_rueck = {T_rueck_C:.0f} °C",
        )

        for pair in (_DIAGONAL_STATE_PAIR_WEAK, _DIAGONAL_STATE_PAIR_STRONG):
            x_pair = [positions[sid][0] for sid in pair]
            y_pair = [positions[sid][1] for sid in pair]
            ax.plot(x_pair, y_pair, "--", color=color, linewidth=1.1, alpha=0.85, zorder=11)

        process_handles.append(cycle_handle)
        process_labels.append(f"T_rueck = {T_rueck_C:.0f} °C")

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

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


__all__ = ["plot_duehring_multi_operating_points"]
