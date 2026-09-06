"""Q-T diagrams (pinch analysis) for several reject temperatures, as one PDF.

Reuses the same sweep as AC_feasibility_sweep.py / AC_duehring_multi_process_plot.py
(warm-started homotopy over T_reject) and, for the same selection of reject
temperatures (each at T11_min), renders one page per point with the Q-T
diagrams of all five heat exchangers (SHEX, desorber, condenser, evaporator,
absorber), in the same layout as Postprocessing/AC_QT_Plot.py. The plotting
function itself is unchanged; it's just called once per point, and each
returned figure becomes a page in a shared PDF.

select_and_plot_qt_pdf() is the reusable core: it takes an already computed
list of points (e.g. straight from AC_feasibility_sweep.py, without
re-running the sweep). The __main__ block below only runs the sweep when
this script is invoked standalone.

Standalone usage
-----------------
    python Design_Point/Visualization_Scripts/AC_qt_multi_process_plot.py
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

from Postprocessing.AC_QT_Plot import plot_qt_diagrams

if TYPE_CHECKING:
    from Design_Point.AC_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Configuration -- ADJUST HERE
# ---------------------------------------------------------------------------
# Same values as in AC_duehring_multi_process_plot.py, so both plots show the
# same reject temperatures. Only relevant for the STANDALONE invocation of
# this script (see __main__).
T_RUECK_START_C = 15.0
T_RUECK_END_C = 35.0
T_RUECK_STEP_C = 2.5
PLOT_EVERY_NTH = 2  # "every second" reject temperature examined

PDF_SAVE_PATH = "Design_Point/Plots/AC_qt_multi_process.pdf"


def select_and_plot_qt_pdf(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    save_path: Optional[str] = PDF_SAVE_PATH,
):
    """Picks every `every_nth`-th feasible point from an already computed sweep
    (see AC_feasibility_sweep.sweep_min_generator_temperature_homotopy()) and
    writes one page per point with the Q-T diagrams (at T11_min) into a
    shared PDF.

    Computes NOTHING new -- reuses the ACResult objects already stored in
    `points[i].result` (strictly re-evaluated at T11_min). Skips with a
    message if no feasible point is available.
    """
    feasible_points = [p for p in points if p.feasible and p.result is not None]
    selected = feasible_points[::every_nth]

    if not selected:
        print(
            "select_and_plot_qt_pdf: no feasible operating point available -- "
            "skipping QT PDF."
        )
        return

    # Ascending T_reject order for the PDF, matching the legend order in the
    # Duehring multi-process plot.
    selected = sorted(selected, key=lambda p: p.T_reject_C)

    print(f"\nRendering {len(selected)} of {len(feasible_points)} feasible points as PDF pages:")
    for p in selected:
        print(f"  T_reject={p.T_reject_C:6.2f} °C -> T11_min={p.T11_min_C:6.2f} °C")

    if save_path is None:
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(save_path) as pdf:
        for p in selected:
            fig = plot_qt_diagrams(p.result, show=False, save_path=None)
            fig.suptitle(
                f"AC - Q-T diagrams (pinch analysis) - T_reject = {p.T_reject_C:.0f} °C, "
                f"T11_min = {p.T11_min_C:.2f} °C",
                fontsize=14,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"QT PDF saved: {save_path} ({len(selected)} pages)")


if __name__ == "__main__":
    from Design_Point.AC_feasibility_sweep import (
        FeasibilitySweepConfig,
        sweep_min_generator_temperature_homotopy,
    )

    config = FeasibilitySweepConfig()

    T_RUECK_RANGE_C = list(
        np.arange(T_RUECK_START_C, T_RUECK_END_C + 0.5 * T_RUECK_STEP_C, T_RUECK_STEP_C)
    )

    print(f"Sweeping T_reject = {T_RUECK_RANGE_C[0]:.0f} .. {T_RUECK_RANGE_C[-1]:.0f} °C ...")
    points = sweep_min_generator_temperature_homotopy(T_RUECK_RANGE_C, config)

    select_and_plot_qt_pdf(points)
