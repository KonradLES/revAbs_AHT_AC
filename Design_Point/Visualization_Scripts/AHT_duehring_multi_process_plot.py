"""Several AHT operating points (one per waste-heat temperature) in one Duehring diagram.

Reuses the same sweep as AHT_feasibility_sweep.py (warm-started homotopy over
T_waste; see that module for the full explanation), but instead of a
table/GTL-window plot, draws the AHT cycle for a selection of the examined
waste-heat temperatures -- each at the upper window edge (T12_max: the
maximum useful temperature reachable at that waste-heat temperature, see
AHT_feasibility_sweep.py) -- as its own colored hexagon in the SAME Duehring
diagram (base layout as in Postprocessing/AHT_Duehring_Plot.py, also used
there for a SINGLE operating point).

Color gradient: cold (blue) = low waste-heat temperature, warm (red) = high
waste-heat temperature (colormap "coolwarm").

select_and_plot_duehring() is the reusable core: it takes an already computed
list of points (e.g. straight from AHT_feasibility_sweep.py, without
re-running the sweep). The __main__ block below only runs the sweep when
this script is invoked standalone.

Standalone usage
-----------------
    python Design_Point/Visualization_Scripts/AHT_duehring_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

# Lives in Design_Point/Visualization_Scripts/ -- three levels up to the repo root
# (Design_Point/Visualization_Scripts -> Design_Point -> repo root).
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from Postprocessing.AHT_Duehring_Plot import plot_duehring_multi_operating_points

if TYPE_CHECKING:
    from Design_Point.AHT_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Configuration -- ADJUST HERE
# ---------------------------------------------------------------------------
# Same meaning as in AHT_feasibility_sweep.py: the homotopy starts cold at
# the highest value (T_WASTE_START_C) and moves downward from there. Only
# relevant for the STANDALONE invocation of this script (see __main__).
T_WASTE_START_C = 85.0
T_WASTE_END_C = 40.0
T_WASTE_STEP_C = 5.0

# Not every examined waste-heat temperature is drawn, otherwise the diagram
# gets cluttered with too many overlapping hexagons. every_nth=2 means
# "every second" (e.g. with a 5 K grid step -> 10 K spacing in the diagram:
# 85, 75, 65, ...).
PLOT_EVERY_NTH = 2

PLOT_SAVE_PATH = "Design_Point/Plots/duehring_multi_process.png"
DUEHRING_VARIANT = "mass"  # "mass" or "mole"


def select_and_plot_duehring(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    variant: str = DUEHRING_VARIANT,
    save_path: Optional[str] = PLOT_SAVE_PATH,
    show: bool = False,
    title: str = "AHT - Duehring diagram: operating points at T12_max per waste-heat temperature",
):
    """Picks every `every_nth`-th feasible point from an already computed sweep
    (see AHT_feasibility_sweep.sweep_relative_lift_window_homotopy()) and
    draws its operating point at T12_max as its own hexagon in a shared
    Duehring diagram.

    Computes NOTHING new -- reuses the AHTResult objects already stored in
    `points[i].result` (strictly re-evaluated). Returns None (and skips with
    a message) if no feasible point is available.
    """
    feasible_points = [p for p in points if p.feasible and p.result is not None]
    selected = feasible_points[::every_nth]

    if not selected:
        print(
            "select_and_plot_duehring: no feasible operating point available -- "
            "skipping Duehring multi-process plot."
        )
        return None

    print(
        f"\nDrawing {len(selected)} of {len(feasible_points)} feasible points "
        f"in the Duehring diagram (every {every_nth}. point):"
    )
    for p in selected:
        print(f"  T_waste={p.T_waste_C:6.2f} °C -> T12_max={p.T12_max_C:6.2f} °C (GTL_max={p.GTL_max_K:.2f} K)")

    entries = [(p.T_waste_C, p.result) for p in selected]

    fig = plot_duehring_multi_operating_points(
        entries, variant=variant, save_path=save_path, show=show, title=title,
    )
    if save_path is not None:
        print(f"Duehring multi-process plot saved: {save_path}")
    return fig


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

    select_and_plot_duehring(points)
