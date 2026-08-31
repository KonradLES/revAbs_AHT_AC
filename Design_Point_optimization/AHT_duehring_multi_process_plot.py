"""Mehrere AWT-Betriebspunkte (je Abwärmetemperatur) in einem Dühring-Diagramm.

Nutzt denselben Sweep wie AHT_feasibility_sweep.py (Warmstart-verkettete
Homotopie über T_waste, siehe dort für die vollständige Erklärung), zeichnet
aber statt einer Tabelle/eines GTL-Fenster-Plots für eine Auswahl der
untersuchten Abwärmetemperaturen den zugehörigen AWT-Kreisprozess -- jeweils
am oberen Fensterrand (T12_max, siehe Modul-Docstring von
AHT_feasibility_sweep.py: "welche Nutztemperatur ist bei dieser
Abwärmetemperatur maximal erreichbar") -- als eigenes Sechseck in einer
eigenen Farbe in DASSELBE Dühring-Diagramm (Basisdarstellung wie in
Postprocessing/AHT_Duehring_Plot.py, dort auch für einen EINZELNEN
Betriebspunkt genutzt).

Farbverlauf: kalt (blau) = niedrige Abwärmetemperatur, warm (rot) = hohe
Abwärmetemperatur (Colormap "coolwarm").

select_and_plot_duehring() ist die wiederverwendbare Kernfunktion -- sie
nimmt eine bereits berechnete Punkteliste entgegen (z.B. direkt aus
AHT_feasibility_sweep.py, ohne den Sweep ein zweites Mal zu rechnen). Das
eigene __main__ hier führt den Sweep nur für den eigenständigen Aufruf
dieses Skripts aus.

Aufruf als eigenständiges Skript
--------------------------------
    python Design_Point_optimization/AHT_duehring_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from Postprocessing.AHT_Duehring_Plot import plot_duehring_multi_operating_points

if TYPE_CHECKING:
    from Design_Point_optimization.AHT_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------
# Dieselbe Bedeutung wie in AHT_feasibility_sweep.py: die Homotopie startet
# beim höchsten Wert (T_WASTE_START_C) kalt und wandert von dort abwärts.
# Nur für den EIGENSTÄNDIGEN Aufruf dieses Skripts relevant (siehe __main__).
T_WASTE_START_C = 85.0
T_WASTE_END_C = 40.0
T_WASTE_STEP_C = 5.0

# Nicht jede untersuchte Abwärmetemperatur wird eingezeichnet, sonst wird das
# Diagramm mit zu vielen überlagerten Sechsecken unleserlich. every_nth=2
# entspricht "jede zweite" (z.B. bei 5K-Rasterschritten -> 10K-Abstand im
# Diagramm: 85, 75, 65, ...).
PLOT_EVERY_NTH = 2

PLOT_SAVE_PATH = "Design_Point_optimization/duehring_multi_process.png"
DUEHRING_VARIANT = "mass"  # "mass" oder "mole"


def select_and_plot_duehring(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    variant: str = DUEHRING_VARIANT,
    save_path: Optional[str] = PLOT_SAVE_PATH,
    show: bool = False,
    title: str = "AWT – Dühring-Diagramm: Betriebspunkte bei T12_max je Abwärmetemperatur",
):
    """Wählt aus einer bereits berechneten Sweep-Punkteliste (siehe
    AHT_feasibility_sweep.sweep_relative_lift_window_homotopy()) jeden
    `every_nth`-ten feasiblen Punkt aus und zeichnet dessen Betriebspunkt bei
    T12_max als eigenes Sechseck in ein gemeinsames Dühring-Diagramm.

    Rechnet NICHTS neu -- nutzt die in `points[i].result` bereits enthaltenen
    (streng nachgerechneten) AWTResult-Objekte. Gibt None zurück (und
    überspringt mit Hinweis), wenn kein feasibler Punkt vorhanden ist.
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
        print(f"  T_waste={p.T_waste_C:6.2f} °C -> T12_max={p.T12_max_C:6.2f} °C (GTL_max={p.GTL_max_K:.2f} K)")

    entries = [(p.T_waste_C, p.result) for p in selected]

    fig = plot_duehring_multi_operating_points(
        entries, variant=variant, save_path=save_path, show=show, title=title,
    )
    if save_path is not None:
        print(f"Dühring-Mehrfach-Plot gespeichert: {save_path}")
    return fig


if __name__ == "__main__":
    from Design_Point_optimization.AHT_feasibility_sweep import (
        FeasibilitySweepConfig,
        sweep_relative_lift_window_homotopy,
    )

    config = FeasibilitySweepConfig()

    T_WASTE_RANGE_C = list(
        np.arange(T_WASTE_START_C, T_WASTE_END_C - 0.5 * T_WASTE_STEP_C, -T_WASTE_STEP_C)
    )

    print(f"Sweep über T_waste = {T_WASTE_RANGE_C[0]:.0f} .. {T_WASTE_RANGE_C[-1]:.0f} °C ...")
    points = sweep_relative_lift_window_homotopy(T_WASTE_RANGE_C, config)

    select_and_plot_duehring(points)
