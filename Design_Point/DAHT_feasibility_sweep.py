"""Fast pinch feasibility map for the DAHT (double-lift AHT) -- simulation
ONLY, no optimization. Analog of AHT_feasibility_sweep.py for the
Models/DAHT_Pinch_Point.py model, adapted after this module's own
development history showed the AHT script's exact-boundary approach
doesn't transfer cleanly -- see "Why a grid scan, not an exact boundary
search" below.

Difference from AHT_feasibility_sweep.py
------------------------------------------
- DAHTInputs has no routing-mode choice: EL's external inlet is always the
  desorber's own external outlet (T15 = T14, fixed series routing), see
  Models/DAHT_Pinch_Point.py's module docstring.
- DAHTInputs has SEVEN pinch values instead of five (SL, desorber,
  condenser, EL, AL/EH internal coupling, AH, SH), because of the two
  solution economizers and the internal AL->EH heat coupling that the
  double-lift topology adds.
- cycle_scale_spec_mode is deliberately fixed to "m_gen" with an arbitrary
  representative m_gen_spec here (NOT "Qwaste"). The T12 feasibility
  question is scale-invariant in principle -- all heat flows scale
  linearly with m_gen -- but "Qwaste" mode pays for an extra full nested
  model evaluation per residual call (see Models/DAHT_Pinch_Point.py's
  _resolve_cycle_scale), which this script avoids since it already calls
  solve_daht_robust() many times. Use DAHT_main_Pinch_Point.py separately
  once a specific (T_waste, T12) point of interest is picked, to get an
  absolute-duty design point.

Why a grid scan, not an exact boundary search
------------------------------------------------
AHT_feasibility_sweep.py finds the EXACT feasible T12 window per T_waste
via a warm-start-chained continuation walk + bisection. The first version
of this script mirrored that approach directly, and it did not hold up for
DAHT during development:

  - The plain unconstrained solve_daht() can converge to a mathematically
    valid but unphysical "collapsed pressure" root (T10_EL pulled together
    with T10_EH) -- addressed with a gap-parametrized fallback, but that
    fallback's own margins can themselves be too tight to reach a genuinely
    feasible point a few K further along the chain.
  - Worse: the continuation walk's step-by-step boundary search was
    reproducibly INCONSISTENT even after fixing the above -- at one
    waste-heat temperature it stopped ~30 K short of a target directly
    proven reachable (GTL=40 K solved cleanly, pinches exactly on target,
    from a fresh cold start in ~11 s), while at another temperature it
    accepted a "boundary" that turned out to be a degenerate near-zero-
    throughput solution (Q_AH = 0.11 kW, COP = 0.0019 -- pinches and
    residuals nominally satisfied, but essentially no useful heat flowing;
    see _is_valid_solution()'s min_useful_fraction check below, added
    specifically because of this).
  - solve_daht_robust() itself, in contrast, reliably solves any single,
    reasonable (T_waste, T12) point directly from a fresh
    Models.DAHT_Pinch_Point.initial_guess() in seconds to tens of seconds
    -- proven repeatedly during development, including jumps of 30-40 K
    that the continuation walk itself could not track.

Given that, this script tests a fixed T12 GRID per T_waste directly with
solve_daht_robust() from a fresh guess each time, instead of searching for
the exact window edge. This trades boundary precision (you get "feasible
at T12=X" for grid points X, not the exact cutoff between two of them) for
much better reliability and simplicity -- each grid point is an
independent, already-proven-robust solve, with no continuation chain that
can drift off course or get stuck.

Standalone usage
-----------------
    python Design_Point/DAHT_feasibility_sweep.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from Models.DAHT_Pinch_Point import (
    DAHTInputs,
    DAHTResult,
    initial_guess,
    solve_daht_robust,
)

RESIDUAL_TOL = 0.5  # looser than AHT's 1e-6: DAHT's residual scaling mixes
                     # raw-K pinch residuals with /100-scaled kW energy
                     # residuals (see _residual_scales in DAHT_Pinch_Point.py).
                     # A genuinely good solve (all 7 pinches on target) can
                     # still show a scaled norm around 0.1-0.3 driven by one
                     # energy-balance residual not fully settled -- this is
                     # a backstop against genuinely bad points (typically >>1),
                     # not the primary validity gate (see below).

MIN_USEFUL_FRACTION = 0.02  # Q_AH / (Q_des + Q_EL) must exceed this, and
                             # mB / mA likewise -- rejects the degenerate
                             # near-zero-throughput root described above
                             # (found empirically: a spurious point can
                             # satisfy every pinch/energy residual to
                             # within tolerance while carrying essentially
                             # no useful heat flow).

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DAHTFeasibilitySweepConfig:
    # --- Operating-point boundary conditions -------------------------------
    T_17_C: float = 15.0     # reject cooling, cold inlet
    m_gen_spec_probe: float = 0.3   # arbitrary fixed scale -- feasibility
                                     # does not depend on its value, see
                                     # module docstring.

    # --- Minimum-lift requirement (T11 = T_waste + min_lift_offset_C) ------
    min_lift_offset_C: float = 2.0

    # --- Pinch values (design assumption, not an optimization target) ------
    dT_min_SL: float = 3.0
    dT_min_des: float = 3.0
    dT_min_cond: float = 3.0
    dT_min_EL: float = 3.0
    dT_min_AL_EH: float = 3.0
    dT_min_AH: float = 3.0
    dT_min_SH: float = 3.0

    # --- External approach values (design assumption) -----------------------
    dT_approach_des_C: float = 4.0    # T14 = T_waste - this
    dT_approach_evap_C: float = 4.0   # T16 = T14 - this (series: T15=T14)
    dT_approach_cond_C: float = 3.0   # T18 = T17 + this

    cp_w_kJkgK: float = 4.18
    desorber_vapor_superheat_K: float = 0.0

    # --- Grid-scan parameters ------------------------------------------------
    GTL_grid_K: Sequence[float] = (5.0, 15.0, 25.0, 35.0, 45.0)
    # T12 candidates are T_waste + min_lift_offset_C + each of these -- i.e.
    # "GTL above the minimum-lift sink temperature", not above T_waste
    # itself, so every candidate automatically respects T12 > T11.
    # Kept deliberately coarse by default (5 points, not 11): failed
    # candidates are expensive (solve_daht_robust() exhausts its full
    # fallback chain before giving up, ~130s/point observed for a mixed
    # feasible/infeasible grid) -- widen this once a rough range of
    # interest is known.


# ---------------------------------------------------------------------------
# Search grid for the sweep
# ---------------------------------------------------------------------------
T_WASTE_RANGE_C = [85.0, 70.0, 55.0]

plot_name = "DAHT_feasibility_grid"


@dataclass(frozen=True)
class DAHTGridPoint:
    T_waste_C: float
    T12_C: float
    GTL_K: float
    feasible: bool
    COP: float = float("nan")
    max_pinch_deviation_K: float = float("nan")
    message: str = ""
    result: Optional[DAHTResult] = None


# ---------------------------------------------------------------------------
# Solve helper functions
# ---------------------------------------------------------------------------

def _build_inputs(
    T_waste_C: float, T12_spec_C: float, T11_C: float, config: DAHTFeasibilitySweepConfig,
) -> DAHTInputs:
    """Series routing is fixed (T15 = T14), so T14 is derived from
    T_waste_C and T16 chains off T14 -- not independently settable."""
    T14_spec_C = T_waste_C - config.dT_approach_des_C
    T16_spec_C = T14_spec_C - config.dT_approach_evap_C

    return DAHTInputs(
        T_11_C=T11_C,
        T_13_C=T_waste_C,
        T_17_C=config.T_17_C,
        dT_min_SL=config.dT_min_SL,
        dT_min_des=config.dT_min_des,
        dT_min_cond=config.dT_min_cond,
        dT_min_EL=config.dT_min_EL,
        dT_min_AL_EH=config.dT_min_AL_EH,
        dT_min_AH=config.dT_min_AH,
        dT_min_SH=config.dT_min_SH,
        cycle_scale_spec_mode="m_gen",
        m_gen_spec=config.m_gen_spec_probe,
        absorber_spec_mode="T12",
        T12_spec_C=T12_spec_C,
        desorber_spec_mode="T14",
        T14_spec_C=T14_spec_C,
        evaporator_spec_mode="T16",
        T16_spec_C=T16_spec_C,
        condenser_spec_mode="T18",
        T18_spec_C=config.T_17_C + config.dT_approach_cond_C,
        cp_w_kJkgK=config.cp_w_kJkgK,
        desorber_vapor_superheat_K=config.desorber_vapor_superheat_K,
    )


def _is_valid_solution(result: DAHTResult, inputs: DAHTInputs) -> tuple[bool, str]:
    info = result.solve_info
    if not info.success or not info.final_point_evaluable:
        return False, "not evaluable / did not converge"
    if info.scaled_residual_norm > RESIDUAL_TOL:
        return False, f"residual norm too large ({info.scaled_residual_norm:.3f})"
    if not result.checks or not all(result.checks.values()):
        failed = [k for k, v in result.checks.items() if not v]
        return False, f"failed checks: {failed}"

    Q_useful = result.heat_flows_kW["Q_AH"]
    Q_drive = result.heat_flows_kW["Q_des"] + result.heat_flows_kW["Q_EL"]
    if Q_drive <= 1.0e-9 or Q_useful / Q_drive < MIN_USEFUL_FRACTION:
        return False, f"degenerate near-zero throughput (Q_AH/Q_drive={Q_useful / max(Q_drive, 1e-9):.4f})"

    mA = result.mass_flows_kg_s["mA"]
    mB = result.mass_flows_kg_s["mB"]
    if mA <= 1.0e-9 or mB / mA < MIN_USEFUL_FRACTION:
        return False, f"degenerate near-zero EH branch flow (mB/mA={mB / max(mA, 1e-9):.4f})"

    return True, "OK"


def _max_pinch_deviation_K(result: DAHTResult, inputs: DAHTInputs) -> float:
    targets = {
        "pinch_SL_K": inputs.dT_min_SL, "pinch_SH_K": inputs.dT_min_SH,
        "pinch_des_K": inputs.dT_min_des, "pinch_cond_K": inputs.dT_min_cond,
        "pinch_EL_K": inputs.dT_min_EL, "pinch_AH_K": inputs.dT_min_AH,
        "pinch_AL_EH_K": inputs.dT_min_AL_EH,
    }
    return max(abs(result.pinch_temperatures_K[k] - v) for k, v in targets.items())


def solve_grid_point(
    T_waste_C: float, T12_C: float, T11_C: float, config: DAHTFeasibilitySweepConfig,
) -> DAHTGridPoint:
    """Solves ONE (T_waste, T12) point independently, from a fresh
    Models.DAHT_Pinch_Point.initial_guess(), via the proven-robust
    solve_daht_robust(). No warm-start chaining between grid points -- see
    module docstring for why that was dropped."""
    try:
        inputs = _build_inputs(T_waste_C, T12_C, T11_C, config)
    except ValueError as exc:
        return DAHTGridPoint(
            T_waste_C=T_waste_C, T12_C=T12_C, GTL_K=T12_C - T_waste_C,
            feasible=False, message=f"invalid inputs: {exc}",
        )

    try:
        x0 = initial_guess(inputs)
        result = solve_daht_robust(inputs, x0=x0)
    except Exception as exc:
        return DAHTGridPoint(
            T_waste_C=T_waste_C, T12_C=T12_C, GTL_K=T12_C - T_waste_C,
            feasible=False, message=f"solve raised: {type(exc).__name__}: {exc}",
        )

    ok, message = _is_valid_solution(result, inputs)
    if not ok:
        return DAHTGridPoint(
            T_waste_C=T_waste_C, T12_C=T12_C, GTL_K=T12_C - T_waste_C,
            feasible=False, message=message, result=result,
        )

    return DAHTGridPoint(
        T_waste_C=T_waste_C, T12_C=T12_C, GTL_K=T12_C - T_waste_C,
        feasible=True, COP=result.kpis["COP"],
        max_pinch_deviation_K=_max_pinch_deviation_K(result, inputs),
        message="OK", result=result,
    )


def grid_scan_feasibility(
    T_waste_values_C: Sequence[float],
    config: DAHTFeasibilitySweepConfig,
) -> List[DAHTGridPoint]:
    """For each T_waste, tests config.GTL_grid_K candidates
    (T12 = T_waste + min_lift_offset_C + GTL) independently. Prints each
    point as it's solved (this can take a while -- each point is a full
    solve_daht_robust() call, up to a couple dozen seconds worst case)."""
    points: List[DAHTGridPoint] = []
    total = len(T_waste_values_C) * len(config.GTL_grid_K)
    i = 0
    for T_waste_C in T_waste_values_C:
        T11_C = T_waste_C + config.min_lift_offset_C
        print(f"--- T_waste = {T_waste_C:.2f} degC (T11 = {T11_C:.2f} degC) ---")
        for gtl in config.GTL_grid_K:
            i += 1
            T12_C = T11_C + gtl
            point = solve_grid_point(T_waste_C, T12_C, T11_C, config)
            points.append(point)
            status = "OK " if point.feasible else "FAIL"
            extra = (
                f"COP={point.COP:.4f}, max|pinch-target|={point.max_pinch_deviation_K:.3f} K"
                if point.feasible else point.message
            )
            print(
                f"  [{i}/{total}] T12={T12_C:6.2f} degC (GTL={point.GTL_K:6.2f} K) "
                f"[{status}] {extra}"
            )
    return points


# ---------------------------------------------------------------------------
# Output: table + plot
# ---------------------------------------------------------------------------

def print_grid_table(points: Sequence[DAHTGridPoint]) -> None:
    print("=" * 100)
    print(f"{'T_waste[C]':>10} {'T12[C]':>8} {'GTL[K]':>7} {'Status':>7} {'COP':>8} {'maxdPinch[K]':>13}")
    print("-" * 100)
    for p in points:
        status = "OK" if p.feasible else "FAIL"
        cop_str = f"{p.COP:8.4f}" if p.feasible else f"{'--':>8}"
        dev_str = f"{p.max_pinch_deviation_K:13.3f}" if p.feasible else f"{'--':>13}"
        print(f"{p.T_waste_C:10.2f} {p.T12_C:8.2f} {p.GTL_K:7.2f} {status:>7} {cop_str} {dev_str}")
    print("=" * 100)


def plot_feasibility_grid(
    points: Sequence[DAHTGridPoint],
    *,
    save_path: Optional[str] = f"Design_Point/Plots/{plot_name}.png",
    show: bool = True,
):
    """Scatter of GTL vs. T_waste, colored by feasibility, with a COP
    colorbar for the feasible points (each an independently solved grid
    point -- no interpolation, no implied boundary between grid points)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ok_points = [p for p in points if p.feasible]
    fail_points = [p for p in points if not p.feasible]

    if ok_points:
        x = np.array([p.T_waste_C for p in ok_points])
        y = np.array([p.GTL_K for p in ok_points])
        c = np.array([p.COP for p in ok_points])
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=70, edgecolors="0.2", linewidths=0.5, label="feasible")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("COP")

    if fail_points:
        xf = np.array([p.T_waste_C for p in fail_points])
        yf = np.array([p.GTL_K for p in fail_points])
        ax.scatter(xf, yf, marker="x", color="tab:red", s=60, linewidths=2, label="not feasible")

    ax.set_xlabel("Waste-heat temperature T13 = T15 [°C]")
    ax.set_ylabel("GTL [K]  (= T12 - T13)")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    config = DAHTFeasibilitySweepConfig()

    print(f"Grid scan: T11 = T_waste + {config.min_lift_offset_C:.0f} K, T17 = {config.T_17_C:.1f} degC")
    print(f"GTL candidates (above T11): {list(config.GTL_grid_K)} K")
    points = grid_scan_feasibility(T_WASTE_RANGE_C, config)
    print()
    print_grid_table(points)
    plot_feasibility_grid(points)
