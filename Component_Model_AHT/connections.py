"""State-point ("connection") data holder for the component-based AHT model.

A Connection is a thin, mutable bag of thermodynamic properties at one
numbered state point of the cycle, using the same state numbering as
Models/AHT_Pinch_Point.py (1-10, 19-21 internal; 11-18 external streams).
Components read the connections they need and write the values they
compute back into them; network.py owns the actual unknown vector and
residual assembly, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Connection:
    """One thermodynamic state point."""

    name: str
    T_K: float | None = None
    p_Pa: float | None = None
    m_kg_s: float | None = None
    h_kJ_kg: float | None = None
    x_LiBr_mol: float | None = None
    w_LiBr: float | None = None
