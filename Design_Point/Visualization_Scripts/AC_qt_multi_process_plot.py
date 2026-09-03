"""Q-T-Diagramme (Pinch-Analyse) für mehrere Rückkühltemperaturen als PDF.

Nutzt denselben Sweep wie AC_feasibility_sweep.py / AC_duehring_multi_process_plot.py
(Warmstart-verkettete Homotopie über T_rueck) und erzeugt für dieselbe
Auswahl an Rückkühltemperaturen -- jeweils bei T11_min -- eine Seite mit den
Q-T-Diagrammen aller fünf Wärmeübertrager (SHEX, Desorber, Kondensator,
Verdampfer, Absorber), im selben Format wie Postprocessing/AC_QT_Plot.py. Die
Funktion selbst wird dafür NICHT verändert, nur pro Punkt aufgerufen und die
zurückgegebene Figure als eigene Seite in ein gemeinsames PDF geschrieben.

select_and_plot_qt_pdf() ist die wiederverwendbare Kernfunktion -- sie nimmt
eine bereits berechnete Punkteliste entgegen (z.B. direkt aus
AC_feasibility_sweep.py, ohne den Sweep ein zweites Mal zu rechnen). Das
eigene __main__ hier führt den Sweep nur für den eigenständigen Aufruf dieses
Skripts aus.

Aufruf als eigenständiges Skript
--------------------------------
    python Design_Point/Visualization_Scripts/AC_qt_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

# Liegt in Design_Point/Visualization_Scripts/ -- drei Ebenen bis zum Repo-Root
# (Design_Point/Visualization_Scripts -> Design_Point -> Repo-Root).
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Postprocessing.AC_QT_Plot import plot_qt_diagrams

if TYPE_CHECKING:
    from Design_Point.AC_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Konfiguration -- HIER ANPASSEN
# ---------------------------------------------------------------------------
# Dieselben Werte wie in AC_duehring_multi_process_plot.py, damit beide
# Auswertungen dieselben Rückkühltemperaturen zeigen. Nur für den
# EIGENSTÄNDIGEN Aufruf dieses Skripts relevant (siehe __main__).
T_RUECK_START_C = 15.0
T_RUECK_END_C = 35.0
T_RUECK_STEP_C = 2.5
PLOT_EVERY_NTH = 2  # "jede zweite" untersuchte Rückkühltemperatur

PDF_SAVE_PATH = "Design_Point/Plots/AC_qt_multi_process.pdf"


def select_and_plot_qt_pdf(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    save_path: Optional[str] = PDF_SAVE_PATH,
):
    """Wählt aus einer bereits berechneten Sweep-Punkteliste (siehe
    AC_feasibility_sweep.sweep_min_generator_temperature_homotopy()) jeden
    `every_nth`-ten feasiblen Punkt aus und schreibt für jeden eine Seite mit
    den Q-T-Diagrammen (bei T11_min) in ein gemeinsames PDF.

    Rechnet NICHTS neu -- nutzt die in `points[i].result` bereits enthaltenen
    (streng nachgerechneten, bei T11_min ausgewerteten) AHTResult-Objekte.
    Überspringt mit Hinweis, wenn kein feasibler Punkt vorhanden ist.
    """
    feasible_points = [p for p in points if p.feasible and p.result is not None]
    selected = feasible_points[::every_nth]

    if not selected:
        print(
            "select_and_plot_qt_pdf: kein feasibler Betriebspunkt vorhanden -- "
            "QT-PDF wird übersprungen."
        )
        return

    # Aufsteigende T_rueck-Reihenfolge fürs PDF, analog zur
    # Legendenreihenfolge im Dühring-Mehrfach-Plot.
    selected = sorted(selected, key=lambda p: p.T_reject_C)

    print(f"\n{len(selected)} von {len(feasible_points)} feasiblen Punkten werden als PDF-Seiten erzeugt:")
    for p in selected:
        print(f"  T_rueck={p.T_reject_C:6.2f} °C -> T11_min={p.T11_min_C:6.2f} °C")

    if save_path is None:
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(save_path) as pdf:
        for p in selected:
            fig = plot_qt_diagrams(p.result, show=False, save_path=None)
            fig.suptitle(
                f"AC – Q-T-Diagramme (Pinch-Analyse) — T_rueck = {p.T_reject_C:.0f} °C, "
                f"T11_min = {p.T11_min_C:.2f} °C",
                fontsize=14,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"QT-PDF gespeichert: {save_path} ({len(selected)} Seiten)")


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

    select_and_plot_qt_pdf(points)
