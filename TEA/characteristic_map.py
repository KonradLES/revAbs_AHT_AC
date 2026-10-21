"""Shared infrastructure for normalized characteristic maps (Kennfelder) of
the reversible AC/AHT machine at a FIXED UA geometry.

Both ac_characteristic_map.py and aht_characteristic_map.py:
  - run a model's solve_*(inputs, x0=...) on a 2D raster
    (T_ambient_C, load_fraction)
  - warm-start every point from an already-solved neighbor (left, then
    above), falling back to a cold start if both fail
  - leave infeasible / non-converged points as NaN -- the Machbarkeitsmaske
  - store results NORMALIZED to the design duty, so one map (computed once
    at the design UA) can be reused for every candidate plant size in the
    annual dispatch/size sweep (fixed UA + fixed temperature levels =>
    duty scales linearly with size, see chat discussion)

This module only holds the generic raster/storage/interpolation logic; the
model-specific input-building and field extraction lives in the two
characteristic_map scripts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class CharacteristicMap:
    """A 2D normalized characteristic map: axes (T_ambient_C, load_fraction).

    Each entry in `fields` is a (len(T_ambient_C), len(load_fraction))
    array, NaN wherever the point is infeasible / did not converge.
    """

    T_ambient_C: np.ndarray
    load_fraction: np.ndarray
    fields: Dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def interpolator(self, field_name: str, *, fill_value: float = float("nan")) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.T_ambient_C, self.load_fraction),
            self.fields[field_name],
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            T_ambient_C=self.T_ambient_C,
            load_fraction=self.load_fraction,
            meta=np.array([repr(self.meta)]),
            **{f"field__{name}": arr for name, arr in self.fields.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "CharacteristicMap":
        data = np.load(Path(path), allow_pickle=False)
        fields = {
            key[len("field__"):]: data[key]
            for key in data.files
            if key.startswith("field__")
        }
        meta = ast.literal_eval(str(data["meta"][0]))
        return cls(
            T_ambient_C=data["T_ambient_C"],
            load_fraction=data["load_fraction"],
            fields=fields,
            meta=meta,
        )


SolveOutcome = Optional[Tuple[np.ndarray, Dict[str, float]]]


def sweep_raster(
    T_ambient_values_C: np.ndarray,
    load_fraction_values: np.ndarray,
    *,
    solve_point: Callable[[float, float, np.ndarray], SolveOutcome],
    cold_start: Callable[[float, float], np.ndarray],
    field_names: List[str],
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Generic 2D raster sweep with neighbor warm-starting.

    `solve_point(T_ambient_C, load_fraction, x0)` returns
    `(x_solution, {field_name: value, ...})` on success, or None if the
    point is infeasible / did not converge.

    `cold_start(T_ambient_C, load_fraction)` builds a fresh initial guess
    (model.initial_guess()-based); used for the very first row's first
    point and as a fallback whenever both warm-start candidates fail.

    Per grid point, tries in order: the previously processed load_fraction
    in this row, then the point directly above (same load_fraction, the
    previous T_ambient row), then a cold start. load_fraction is walked
    HIGH -> LOW within each row: model.initial_guess() is tuned around the
    nominal/full-load state, so full load converges cold most reliably,
    and part load is then reached by continuation from there.

    NOTE on coverage: an EARLIER version of this function anchored the
    T_ambient sweep at the design/nominal temperature and expanded outward
    in both directions (on the theory that the design condition converges
    most reliably cold). Tested empirically against this plain
    ascending-T-order version on the AC map: the anchored version did
    WORSE (31/234 feasible points vs. 115/234 here) -- the anchor row
    itself has no row-to-row warm start to fall back on, and apparently
    converges less reliably here than the T_ambient_values_C[0] row does,
    so every other row inherits a worse starting chain. Coverage is
    whatever this specific model's initial_guess()/bounds() basin happens
    to support from a monotonic march; it is NOT yet a full feasibility
    map (see AC_feasibility_sweep.py's bisection/homotopy approach for a
    more thorough -- and more expensive -- alternative if you need the
    remaining gaps filled in).
    """
    n_T, n_L = len(T_ambient_values_C), len(load_fraction_values)
    grids = {name: np.full((n_T, n_L), np.nan) for name in field_names}
    x_solutions: List[List[Optional[np.ndarray]]] = [[None] * n_L for _ in range(n_T)]

    for i, T_amb in enumerate(T_ambient_values_C):
        prev_j: Optional[int] = None
        for j in range(n_L - 1, -1, -1):
            load = load_fraction_values[j]
            candidates: List[np.ndarray] = []
            if prev_j is not None and x_solutions[i][prev_j] is not None:
                candidates.append(x_solutions[i][prev_j])
            if i > 0 and x_solutions[i - 1][j] is not None:
                candidates.append(x_solutions[i - 1][j])
            candidates.append(cold_start(T_amb, load))
            prev_j = j

            outcome: SolveOutcome = None
            for x0 in candidates:
                outcome = solve_point(T_amb, load, x0)
                if outcome is not None:
                    break

            if outcome is None:
                if verbose:
                    print(f"  [T={T_amb:6.2f} C, load={load:5.2f}] -> INFEASIBLE")
                continue

            x_sol, values = outcome
            x_solutions[i][j] = x_sol
            for name in field_names:
                grids[name][i, j] = values[name]
            if verbose:
                print(f"  [T={T_amb:6.2f} C, load={load:5.2f}] -> OK  " + ", ".join(
                    f"{k}={v:.4f}" for k, v in values.items()
                ))

    return grids
