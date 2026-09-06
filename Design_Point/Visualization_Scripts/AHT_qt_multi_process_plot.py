"""Q-T diagrams (pinch analysis) for several waste-heat temperatures, as one PDF.

Reuses the same sweep as AHT_feasibility_sweep.py / AHT_duehring_multi_process_plot.py
(warm-started homotopy over T_waste) and, for the same selection of waste-heat
temperatures (each at the upper window edge T12_max), renders one page per
point with the Q-T diagrams of all five heat exchangers (SHEX, desorber,
condenser, evaporator, absorber), in the same layout as
Postprocessing/AHT_QT_Plot.py (used in AHT_main_Pinch_Point.py for a single
operating point). The plotting function itself is unchanged; it's just
called once per point, and each returned figure becomes a page in a shared
PDF.

select_and_plot_qt_pdf() is the reusable core: it takes an already computed
list of points (e.g. straight from AHT_feasibility_sweep.py, without
re-running the sweep). The __main__ block below only runs the sweep when
this script is invoked standalone.

Standalone usage
-----------------
    python Design_Point/Visualization_Scripts/AHT_qt_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

# Lives in Design_Point/Visualization_Scripts/ -- three levels up to the repo root
# (Design_Point/Visualization_Scripts -> Design_Point -> repo root).
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Postprocessing.AHT_QT_Plot import plot_qt_diagrams

if TYPE_CHECKING:
    from Design_Point.AHT_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Configuration -- ADJUST HERE
# ---------------------------------------------------------------------------
# Same values as in AHT_duehring_multi_process_plot.py, so both plots show
# the same waste-heat temperatures. Only relevant for the STANDALONE
# invocation of this script (see __main__).
T_WASTE_START_C = 85.0
T_WASTE_END_C = 40.0
T_WASTE_STEP_C = 5.0
PLOT_EVERY_NTH = 2  # "every second" waste-heat temperature examined

PDF_SAVE_PATH = "Design_Point/Plots/qt_multi_process.pdf"


def select_and_plot_qt_pdf(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    save_path: Optional[str] = PDF_SAVE_PATH,
):
    """Picks every `every_nth`-th feasible point from an already computed sweep
    (see AHT_feasibility_sweep.sweep_relative_lift_window_homotopy()) and
    writes one page per point with the Q-T diagrams (at T12_max) into a
    shared PDF.

    Computes NOTHING new -- reuses the AHTResult objects already stored in
    `points[i].result` (strictly re-evaluated). Skips with a message if no
    feasible point is available.
    """
    feasible_points = [p for p in points if p.feasible and p.result is not None]
    selected = feasible_points[::every_nth]

    if not selected:
        print(
            "select_and_plot_qt_pdf: no feasible operating point available -- "
            "skipping QT PDF."
        )
        return

    # Descending T_waste order for the PDF (highest first), matching the
    # legend order in the Duehring multi-process plot.
    selected = sorted(selected, key=lambda p: p.T_waste_C, reverse=True)

    print(f"\nRendering {len(selected)} of {len(feasible_points)} feasible points as PDF pages:")
    for p in selected:
        print(f"  T_waste={p.T_waste_C:6.2f} °C -> T12_max={p.T12_max_C:6.2f} °C (GTL_max={p.GTL_max_K:.2f} K)")

    if save_path is None:
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(save_path) as pdf:
        for p in selected:
            fig = plot_qt_diagrams(p.result, show=False, save_path=None)
            fig.suptitle(
                f"AHT - Q-T diagrams (pinch analysis) - T_waste = {p.T_waste_C:.0f} °C, "
                f"T12_max = {p.T12_max_C:.2f} °C",
                fontsize=14,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"QT PDF saved: {save_path} ({len(selected)} pages)")


if __name__ == "__main__":
    from Design_Point.AHT_feasibility_sweep import (
        FeasibilitySweepConfig,
        sweep_relative_lift_window_homotopy,
    )

    config = FeasibilitySweepConfig()

    T_WASTE_RANGE_C = list(
        np.arange(T_WASTE_START_C, T_WASTE_END_C - 0.5 * T_WASTE_STEP_C, -T_WASTE_STEP_C)
    )

    print(f"Sweeping T_waste = {T_WASTE_RANGE_C[0]:.0f} .. {T_WASTE_RANGE_C[-1]:.0f} °C ...")
    points = sweep_relative_lift_window_homotopy(T_WASTE_RANGE_C, config)

    select_and_plot_qt_pdf(points)
