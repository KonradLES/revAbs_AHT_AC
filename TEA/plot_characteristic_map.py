"""Visualizes a saved CharacteristicMap (ac_ or aht_characteristic_map.npz):
where is the process feasible, and how does COP vary across the grid.

Pure plotting of already-computed data -- the feasibility mask is already
baked into the NaN pattern of every field, so this costs no additional
solver time, just a matplotlib render.

Standalone usage
-----------------
    python TEA/plot_characteristic_map.py TEA/maps/ac_characteristic_map.npz
    python TEA/plot_characteristic_map.py TEA/maps/aht_characteristic_map.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from TEA.characteristic_map import CharacteristicMap


def plot_map(cmap: CharacteristicMap, *, field: str = "COP", save_path: Path | None = None, show: bool = True):
    data = cmap.fields[field]
    feasible = ~np.isnan(data)

    fig, (ax_cop, ax_mask) = plt.subplots(1, 2, figsize=(13, 5.5))

    extent = [
        cmap.load_fraction.min(), cmap.load_fraction.max(),
        cmap.T_ambient_C.min(), cmap.T_ambient_C.max(),
    ]

    # --- Left: the field itself (e.g. COP), infeasible points left blank ---
    im = ax_cop.imshow(
        data, origin="lower", aspect="auto", extent=extent,
        cmap="viridis", interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_cop, label=field)
    ax_cop.set_title(f"{field} ({cmap.meta.get('mode', '?')} mode)")
    ax_cop.set_xlabel("load_fraction")
    ax_cop.set_ylabel("T_ambient [°C]")

    # --- Right: feasibility mask, with a traced boundary ---
    ax_mask.imshow(
        feasible.astype(float), origin="lower", aspect="auto", extent=extent,
        cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest",
    )
    ax_mask.contour(
        cmap.load_fraction, cmap.T_ambient_C, feasible.astype(float),
        levels=[0.5], colors="black", linewidths=1.5,
    )
    ax_mask.set_title("feasibility (green = OK, red = infeasible)")
    ax_mask.set_xlabel("load_fraction")
    ax_mask.set_ylabel("T_ambient [°C]")

    n_feasible = int(feasible.sum())
    n_total = feasible.size
    fig.suptitle(
        f"{cmap.meta.get('design_point_source', '')} -- "
        f"{n_feasible}/{n_total} grid points feasible ({100 * n_feasible / n_total:.0f}%)",
        fontsize=9,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if show:
        plt.show()
    return fig, (ax_cop, ax_mask)


if __name__ == "__main__":
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("TEA/maps/ac_characteristic_map.npz")
    cmap = CharacteristicMap.load(npz_path)
    save_path = npz_path.with_suffix(".png")
    plot_map(cmap, field="COP", save_path=save_path, show=False)
