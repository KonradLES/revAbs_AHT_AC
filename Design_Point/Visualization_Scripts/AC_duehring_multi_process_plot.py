"""Mehrere AC-Betriebspunkte (je eine Rückkühltemperatur) in einem
Dühring-Diagramm.

Nutzt denselben Sweep wie AC_feasibility_sweep.py (Warmstart-verkettete
Homotopie über T_rueck, siehe dort für die vollständige Erklärung), zeichnet
aber statt einer Tabelle/eines T11-Fenster-Plots für eine Auswahl der
untersuchten Rückkühltemperaturen den zugehörigen AC-Kreisprozess -- jeweils
bei T11_min (die pro Rückkühltemperatur minimal nötige
Generatoreintrittstemperatur, siehe Modul-Docstring von
AC_feasibility_sweep.py) -- als eigenes Sechseck in einer eigenen Farbe in
DASSELBE Dühring-Diagramm (Basisdarstellung wie in
Postprocessing/AC_Duehring_Plot.py).

Farbverlauf: kalt (blau) = niedrige Rückkühltemperatur, warm (rot) = hohe
Rückkühltemperatur (Colormap "coolwarm").

select_and_plot_duehring() ist die wiederverwendbare Kernfunktion -- sie
nimmt eine bereits berechnete Punkteliste entgegen (z.B. direkt aus
AC_feasibility_sweep.py, ohne den Sweep ein zweites Mal zu rechnen). Das
eigene __main__ hier führt den Sweep nur für den eigenständigen Aufruf
dieses Skripts aus.

Aufruf als eigenständiges Skript
--------------------------------
    python Design_Point/Visualization_Scripts/AC_duehring_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

# Liegt in Design_Point/Visualization_Scripts/ -- drei Ebenen bis zum Repo-Root
# (Design_Point/Visualization_Scripts -> Design_Point -> Repo-Root).
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from Postprocessing.AC_Duehring_Plot import plot_duehring_multi_operating_points

if TYPE_CHECKING:
    from Design_Point.AC_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------
# Dieselbe Bedeutung wie in AC_feasibility_sweep.py: die Homotopie startet
# beim niedrigsten Wert (T_RUECK_START_C) und wandert von dort aufwärts. Nur
# für den EIGENSTÄNDIGEN Aufruf dieses Skripts relevant (siehe __main__).
T_RUECK_START_C = 15.0
T_RUECK_END_C = 35.0
T_RUECK_STEP_C = 2.5

# Nicht jede untersuchte Rückkühltemperatur wird eingezeichnet, sonst wird
# das Diagramm mit zu vielen überlagerten Sechsecken unleserlich. every_nth=2
# entspricht "jede zweite" (bei 2.5K-Rasterschritten und 9 Punkten -> 5
# eingezeichnete Sechsecke).
PLOT_EVERY_NTH = 2

PLOT_SAVE_PATH = "Design_Point/Plots/AC_duehring_multi_process.png"
DUEHRING_VARIANT = "mass"  # "mass" oder "mole"


def select_and_plot_duehring(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    variant: str = DUEHRING_VARIANT,
    save_path: Optional[str] = PLOT_SAVE_PATH,
    show: bool = False,
    title: str = "AC – Dühring-Diagramm: Betriebspunkte bei T11_min je Rückkühltemperatur",
):
    """Wählt aus einer bereits berechneten Sweep-Punkteliste (siehe
    AC_feasibility_sweep.sweep_min_generator_temperature_homotopy()) jeden
    `every_nth`-ten feasiblen Punkt aus und zeichnet dessen Betriebspunkt bei
    T11_min als eigenes Sechseck in ein gemeinsames Dühring-Diagramm.

    Rechnet NICHTS neu -- nutzt die in `points[i].result` bereits enthaltenen
    (streng nachgerechneten, bei T11_min ausgewerteten) AHTResult-Objekte.
    Gibt None zurück (und überspringt mit Hinweis), wenn kein feasibler
    Punkt vorhanden ist.
    """
    feasible_points = [p for p in points if p.feasible and p.result is not None]
    selected = feasible_points[::every_nth]

    if not selected:
        print(
            "select_and_plot_duehring: kein feasibler Betriebspunkt vorhanden -- "
            "Dühring-Mehrfach-Plot wird übersprungen."
        )
        return None

    print(
        f"\n{len(selected)} von {len(feasible_points)} feasiblen Punkten werden "
        f"im Dühring-Diagramm eingezeichnet (jeder {every_nth}. Punkt):"
    )
    for p in selected:
        print(f"  T_rueck={p.T_reject_C:6.2f} °C -> T11_min={p.T11_min_C:6.2f} °C")

    entries = [(p.T_reject_C, p.result) for p in selected]

    fig = plot_duehring_multi_operating_points(
        entries, variant=variant, save_path=save_path, show=show, title=title,
    )
    if save_path is not None:
        print(f"Dühring-Mehrfach-Plot gespeichert: {save_path}")
    return fig


if __name__ == "__main__":
    from Design_Point.AC_feasibility_sweep import (
        FeasibilitySweepConfig,
        sweep_min_generator_temperature_homotopy,
    )

    config = FeasibilitySweepConfig()

    T_RUECK_RANGE_C = list(
        np.arange(T_RUECK_START_C, T_RUECK_END_C + 0.5 * T_RUECK_STEP_C, T_RUECK_STEP_C)
    )

    print(f"Sweep über T_rueck = {T_RUECK_RANGE_C[0]:.0f} .. {T_RUECK_RANGE_C[-1]:.0f} °C ...")
    points = sweep_min_generator_temperature_homotopy(T_RUECK_RANGE_C, config)

    select_and_plot_duehring(points)
