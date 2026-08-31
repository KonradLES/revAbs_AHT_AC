"""Q-T-Diagramme (Pinch-Analyse) für mehrere Abwärmetemperaturen als PDF.

Nutzt denselben Sweep wie AHT_feasibility_sweep.py / AHT_duehring_multi_process_plot.py
(Warmstart-verkettete Homotopie über T_waste) und erzeugt für dieselbe Auswahl
an Abwärmetemperaturen -- jeweils am oberen Fensterrand T12_max -- eine Seite
mit den Q-T-Diagrammen aller fünf Wärmeübertrager (SHEX, Desorber, Kondensator,
Verdampfer, Absorber), im selben Format wie Postprocessing/AHT_QT_Plot.py
(genutzt in AHT_main_Pinch_Point.py für einen einzelnen Betriebspunkt). Die
Funktion selbst wird dafür NICHT verändert, nur pro Punkt aufgerufen und die
zurückgegebene Figure als eigene Seite in ein gemeinsames PDF geschrieben.

select_and_plot_qt_pdf() ist die wiederverwendbare Kernfunktion -- sie nimmt
eine bereits berechnete Punkteliste entgegen (z.B. direkt aus
AHT_feasibility_sweep.py, ohne den Sweep ein zweites Mal zu rechnen). Das
eigene __main__ hier führt den Sweep nur für den eigenständigen Aufruf dieses
Skripts aus.

Aufruf als eigenständiges Skript
--------------------------------
    python Design_Point_optimization/AHT_qt_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Postprocessing.AHT_QT_Plot import plot_qt_diagrams

if TYPE_CHECKING:
    from Design_Point_optimization.AHT_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------
# Dieselben Werte wie in AHT_duehring_multi_process_plot.py, damit beide
# Auswertungen dieselben Abwärmetemperaturen zeigen. Nur für den
# EIGENSTÄNDIGEN Aufruf dieses Skripts relevant (siehe __main__).
T_WASTE_START_C = 85.0
T_WASTE_END_C = 40.0
T_WASTE_STEP_C = 5.0
PLOT_EVERY_NTH = 2  # "jede zweite" untersuchte Abwärmetemperatur

PDF_SAVE_PATH = "Design_Point_optimization/qt_multi_process.pdf"


def select_and_plot_qt_pdf(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    save_path: Optional[str] = PDF_SAVE_PATH,
):
    """Wählt aus einer bereits berechneten Sweep-Punkteliste (siehe
    AHT_feasibility_sweep.sweep_relative_lift_window_homotopy()) jeden
    `every_nth`-ten feasiblen Punkt aus und schreibt für jeden eine Seite mit
    den Q-T-Diagrammen (bei T12_max) in ein gemeinsames PDF.

    Rechnet NICHTS neu -- nutzt die in `points[i].result` bereits enthaltenen
    (streng nachgerechneten) AWTResult-Objekte. Überspringt mit Hinweis, wenn
    kein feasibler Punkt vorhanden ist.
    """
    feasible_points = [p for p in points if p.feasible and p.result is not None]
    selected = feasible_points[::every_nth]

    if not selected:
        print(
            "select_and_plot_qt_pdf: kein feasibler Betriebspunkt vorhanden -- "
            "QT-PDF wird übersprungen."
        )
        return

    # Absteigende T_waste-Reihenfolge fürs PDF (höchste zuerst), analog zur
    # Legendenreihenfolge im Dühring-Mehrfach-Plot.
    selected = sorted(selected, key=lambda p: p.T_waste_C, reverse=True)

    print(f"\n{len(selected)} von {len(feasible_points)} feasiblen Punkten werden als PDF-Seiten erzeugt:")
    for p in selected:
        print(f"  T_waste={p.T_waste_C:6.2f} °C -> T12_max={p.T12_max_C:6.2f} °C (GTL_max={p.GTL_max_K:.2f} K)")

    if save_path is None:
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(save_path) as pdf:
        for p in selected:
            fig = plot_qt_diagrams(p.result, show=False, save_path=None)
            fig.suptitle(
                f"AWT – Q-T-Diagramme (Pinch-Analyse) — T_waste = {p.T_waste_C:.0f} °C, "
                f"T12_max = {p.T12_max_C:.2f} °C",
                fontsize=14,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"QT-PDF gespeichert: {save_path} ({len(selected)} Seiten)")


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

    select_and_plot_qt_pdf(points)
