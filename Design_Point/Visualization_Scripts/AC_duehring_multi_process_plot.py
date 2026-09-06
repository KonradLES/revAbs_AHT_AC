"""Several AC operating points (one per reject temperature) in one Duehring diagram.

Reuses the same sweep as AC_feasibility_sweep.py (warm-started homotopy over
T_reject; see that module for the full explanation), but instead of a
table/T11-window plot, draws the AC cycle for a selection of the examined
reject temperatures -- each at T11_min (the minimum generator inlet
temperature needed at that reject temperature, see AC_feasibility_sweep.py) --
as its own colored hexagon in the SAME Duehring diagram (base layout as in
Postprocessing/AC_Duehring_Plot.py).

Color gradient: cold (blue) = low reject temperature, warm (red) = high
reject temperature (colormap "coolwarm").

select_and_plot_duehring() is the reusable core: it takes an already computed
list of points (e.g. straight from AC_feasibility_sweep.py, without
re-running the sweep). The __main__ block below only runs the sweep when
this script is invoked standalone.

Standalone usage
-----------------
    python Design_Point/Visualization_Scripts/AC_duehring_multi_process_plot.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Optional, Sequence

# Lives in Design_Point/Visualization_Scripts/ -- three levels up to the repo root
# (Design_Point/Visualization_Scripts -> Design_Point -> repo root).
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from Postprocessing.AC_Duehring_Plot import plot_duehring_multi_operating_points

if TYPE_CHECKING:
    from Design_Point.AC_feasibility_sweep import FeasibilityPoint

# ---------------------------------------------------------------------------
# Configuration -- ADJUST HERE
# ---------------------------------------------------------------------------
# Same meaning as in AC_feasibility_sweep.py: the homotopy starts at the
# lowest value (T_RUECK_START_C) and moves upward from there. Only relevant
# for the STANDALONE invocation of this script (see __main__).
T_RUECK_START_C = 15.0
T_RUECK_END_C = 35.0
T_RUECK_STEP_C = 2.5

# Not every examined reject temperature is drawn, otherwise the diagram gets
# cluttered with too many overlapping hexagons. every_nth=2 means "every
# second" (with a 2.5 K grid step and 9 points -> 5 hexagons drawn).
PLOT_EVERY_NTH = 2

PLOT_SAVE_PATH = "Design_Point/Plots/AC_duehring_multi_process.png"
DUEHRING_VARIANT = "mass"  # "mass" or "mole"


def select_and_plot_duehring(
    points: Sequence["FeasibilityPoint"],
    *,
    every_nth: int = PLOT_EVERY_NTH,
    variant: str = DUEHRING_VARIANT,
    save_path: Optional[str] = PLOT_SAVE_PATH,
    show: bool = False,
    title: str = "AC - Duehring diagram: operating points at T11_min per reject temperature",
):
    """Picks every `every_nth`-th feasible point from an already computed sweep
    (see AC_feasibility_sweep.sweep_min_generator_temperature_homotopy()) and
    draws its operating point at T11_min as its own hexagon in a shared
    Duehring diagram.

    Computes NOTHING new -- reuses the ACResult objects already stored in
    `points[i].result` (strictly re-evaluated at T11_min). Returns None (and
    skips with a message) if no feasible point is available.
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
        print(f"  T_reject={p.T_reject_C:6.2f} °C -> T11_min={p.T11_min_C:6.2f} °C")

    entries = [(p.T_reject_C, p.result) for p in selected]

    fig = plot_duehring_multi_operating_points(
        entries, variant=variant, save_path=save_path, show=show, title=title,
    )
    if save_path is not None:
        print(f"Duehring multi-process plot saved: {save_path}")
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

    print(f"Sweeping T_reject = {T_RUECK_RANGE_C[0]:.0f} .. {T_RUECK_RANGE_C[-1]:.0f} °C ...")
    points = sweep_min_generator_temperature_homotopy(T_RUECK_RANGE_C, config)

    select_and_plot_duehring(points)
