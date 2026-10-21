"""Skeleton: hourly annual dispatch + size sweep, reversible AC/AHT vs.
pure AC.

This is intentionally runnable END-TO-END today, even though the real cost
equations don't exist yet -- every function under "ECONOMICS (PLACEHOLDER)"
below returns an explicitly fake, clearly flagged number so the whole
pipeline (profile -> hourly dispatch -> size sweep -> objective) can be
exercised and plotted now. Swap those functions out once the real CAPEX/
OPEX/price assumptions are ready; nothing else in this file needs to
change -- `size_sweep()` just calls whatever `economic_objective()` returns.

Pipeline (see chat discussion for the full rationale)
------------------------------------------------------
1. Load an hourly year profile: cooling demand, heat demand, ambient
   temperature (`load_profile_csv`, or `synthetic_demo_profile` while you
   don't have real data wired up yet).
2. Load the two normalized characteristic maps (TEA/ac_characteristic_map.py,
   TEA/aht_characteristic_map.py) as interpolators.
3. For a given size factor s (SAME s scales both modes -- it's one
   physical reversible machine), run an hourly dispatch:
     - AC_eligible months -> look up the AC map at the hour's reject
       temperature and the demanded load fraction, serve
       min(demand, s * Q_evap_design * achievable_fraction); residual
       (unserved) cooling is flagged for costing via a reference
       compression chiller.
     - AHT_eligible months -> same idea with the AHT map and the heat
       demand.
     - Transition months: TODO, see `SwitchoverConfig` -- currently a
       hard month-based switch, not a load- or temperature-based one.
     - Below `min_load_fraction`: machine OFF for that hour (no part-load
       floor modeling yet).
4. Aggregate the year into a handful of energy KPIs (full-load hours,
   served/residual energy) and feed them into the (placeholder) economic
   objective.
5. Sweep s, compare the "reversible" scenario against an "AC-only"
   reference (AHT branch disabled) -- this is the actual answer to "lohnt
   sich der Trafo".

Standalone usage
-----------------
    python TEA/annual_dispatch_sweep_skeleton.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from TEA.ac_characteristic_map import DEFAULT_MAP_PATH as AC_MAP_PATH
from TEA.aht_characteristic_map import DEFAULT_MAP_PATH as AHT_MAP_PATH
from TEA.characteristic_map import CharacteristicMap

HOURS_PER_YEAR = 8760


# ---------------------------------------------------------------------------
# 1) Load profile
# ---------------------------------------------------------------------------

@dataclass
class LoadProfile:
    cooling_demand_kW: np.ndarray   # (8760,)
    heat_demand_kW: np.ndarray      # (8760,) -- total district heat demand (AHT covers only a slice of this)
    T_ambient_C: np.ndarray         # (8760,) -- drives both the AC reject side and the AHT condenser side


def load_profile_csv(path: Path) -> LoadProfile:
    """Expects columns: hour, cooling_demand_kW, heat_demand_kW, T_ambient_C
    (one row per hour, 8760 rows). TODO: point this at your real Lastgang
    once it's available; adjust column names/units here if they differ.
    """
    import csv

    cooling, heat, t_amb = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cooling.append(float(row["cooling_demand_kW"]))
            heat.append(float(row["heat_demand_kW"]))
            t_amb.append(float(row["T_ambient_C"]))
    return LoadProfile(
        cooling_demand_kW=np.asarray(cooling, dtype=float),
        heat_demand_kW=np.asarray(heat, dtype=float),
        T_ambient_C=np.asarray(t_amb, dtype=float),
    )


def synthetic_demo_profile(*, seed: int = 0) -> LoadProfile:
    """Placeholder annual profile (smooth seasonal sine + noise) so the
    pipeline is runnable before a real Lastgang is wired up. NOT a
    substitute for real data -- replace with `load_profile_csv(...)`.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(HOURS_PER_YEAR)
    day_of_year = hours / 24.0

    # Seasonal ambient temperature: warm in summer, cold in winter.
    T_ambient_C = 10.0 - 12.0 * np.cos(2 * np.pi * day_of_year / 365.0) + rng.normal(0, 2.0, HOURS_PER_YEAR)

    # Cooling demand: peaks in summer, ~0 in winter, with a daily cycle.
    seasonal_cooling = np.clip(-np.cos(2 * np.pi * day_of_year / 365.0), 0.0, None)
    daily_cooling = 0.6 + 0.4 * np.clip(np.sin(2 * np.pi * (hours % 24 - 6) / 24.0), 0.0, None)
    cooling_demand_kW = 100.0 * seasonal_cooling * daily_cooling + rng.normal(0, 2.0, HOURS_PER_YEAR).clip(min=0)

    # District heat demand: peaks in winter.
    seasonal_heat = np.clip(np.cos(2 * np.pi * day_of_year / 365.0), 0.0, None)
    heat_demand_kW = 300.0 * seasonal_heat + rng.normal(0, 5.0, HOURS_PER_YEAR).clip(min=0)

    return LoadProfile(cooling_demand_kW=cooling_demand_kW, heat_demand_kW=heat_demand_kW, T_ambient_C=T_ambient_C)


# ---------------------------------------------------------------------------
# 2) Switchover rule (AC months vs. AHT months)
# ---------------------------------------------------------------------------

@dataclass
class SwitchoverConfig:
    # TODO: this is a placeholder hard month switch. The chat discussion
    # flags the transition months as a real design decision (the machine
    # can only run ONE mode at a time) -- revisit with a load- or
    # temperature-based rule once you have both profiles to compare.
    ac_months: frozenset = field(default_factory=lambda: frozenset({4, 5, 6, 7, 8, 9}))

    def is_ac_month(self, hour_index: int) -> bool:
        month = (hour_index // (HOURS_PER_YEAR // 12)) % 12 + 1
        return month in self.ac_months


# ---------------------------------------------------------------------------
# 3) Hourly dispatch
# ---------------------------------------------------------------------------

@dataclass
class DispatchSummary:
    cooling_served_kWh: float
    cooling_residual_kWh: float
    heat_served_by_aht_kWh: float
    ac_full_load_hours: float
    aht_full_load_hours: float
    ac_electricity_proxy_kWh: float  # TODO: replace with real pump/aux power once available
    aht_electricity_proxy_kWh: float


def dispatch_year(
    profile: LoadProfile,
    *,
    ac_map: CharacteristicMap,
    aht_map: Optional[CharacteristicMap],
    s: float,
    Q_evap_design_kW: float,
    Qabs_reference_kW: float,
    switchover: SwitchoverConfig = SwitchoverConfig(),
    min_load_fraction_ac: float = 0.15,
    min_load_fraction_aht: float = 0.15,
) -> DispatchSummary:
    """Runs one size factor s through the full year. `aht_map=None`
    reproduces the "pure AC" reference scenario (AHT branch disabled,
    winter heat demand entirely unserved by this plant).
    """
    Q_evap_cap_kW = s * Q_evap_design_kW
    Qabs_cap_kW = s * Qabs_reference_kW

    ac_cop_interp = ac_map.interpolator("COP")
    aht_cop_interp = aht_map.interpolator("COP") if aht_map is not None else None

    cooling_served = 0.0
    cooling_residual = 0.0
    heat_served_by_aht = 0.0
    ac_hours_at_full_load_equiv = 0.0
    aht_hours_at_full_load_equiv = 0.0
    ac_elec_proxy = 0.0
    aht_elec_proxy = 0.0

    for h in range(HOURS_PER_YEAR):
        T_amb = profile.T_ambient_C[h]

        if switchover.is_ac_month(h):
            demand = profile.cooling_demand_kW[h]
            load_fraction = np.clip(demand / Q_evap_cap_kW, 0.0, 1.0) if Q_evap_cap_kW > 0 else 0.0
            if load_fraction < min_load_fraction_ac:
                cooling_residual += demand
                continue
            cop = float(ac_cop_interp([[T_amb, load_fraction]])[0])
            served = min(demand, Q_evap_cap_kW)  # TODO: cross-check against geothermal m11 flow availability
            cooling_served += served
            cooling_residual += max(0.0, demand - served)
            ac_hours_at_full_load_equiv += served / Q_evap_design_kW if Q_evap_design_kW > 0 else 0.0
            if np.isfinite(cop) and cop > 0:
                ac_elec_proxy += served / cop  # placeholder proxy, NOT the real auxiliary power draw

        elif aht_map is not None:
            # TODO: replace this placeholder demand mapping -- the AHT is
            # meant to cover only a SLICE of the heat demand (lowering the
            # geothermal injection temperature further), not the whole
            # district heat load. Decide the actual target slice before
            # trusting these numbers.
            demand_slice = min(profile.heat_demand_kW[h], Qabs_cap_kW)
            load_fraction = np.clip(demand_slice / Qabs_cap_kW, 0.0, 1.0) if Qabs_cap_kW > 0 else 0.0
            if load_fraction < min_load_fraction_aht:
                continue
            cop = float(aht_cop_interp([[T_amb, load_fraction]])[0])
            served = min(demand_slice, Qabs_cap_kW)
            heat_served_by_aht += served
            aht_hours_at_full_load_equiv += served / Qabs_reference_kW if Qabs_reference_kW > 0 else 0.0
            if np.isfinite(cop) and cop > 0:
                aht_elec_proxy += served / cop

    return DispatchSummary(
        cooling_served_kWh=cooling_served,
        cooling_residual_kWh=cooling_residual,
        heat_served_by_aht_kWh=heat_served_by_aht,
        ac_full_load_hours=ac_hours_at_full_load_equiv,
        aht_full_load_hours=aht_hours_at_full_load_equiv,
        ac_electricity_proxy_kWh=ac_elec_proxy,
        aht_electricity_proxy_kWh=aht_elec_proxy,
    )


# ---------------------------------------------------------------------------
# 4) ECONOMICS (PLACEHOLDER) -- replace everything in this section
# ---------------------------------------------------------------------------

@dataclass
class EconomicAssumptions:
    """Every number here is a PLACEHOLDER so size_sweep() produces a
    plottable curve today. None of this should be quoted -- replace with
    your real cost/price model.
    """
    capex_per_kW_eur: float = 400.0        # PLACEHOLDER
    capex_scaling_exponent: float = 0.75   # PLACEHOLDER (economy-of-scale degression)
    opex_fraction_of_capex: float = 0.02   # PLACEHOLDER, per year
    electricity_price_eur_per_kWh: float = 0.20   # PLACEHOLDER
    avoided_cooling_cost_eur_per_kWh: float = 0.08  # PLACEHOLDER (vs. a reference compression chiller)
    avoided_cooling_residual_cost_eur_per_kWh: float = 0.12  # PLACEHOLDER (decentral compression chiller, less efficient)
    heat_value_eur_per_kWh: float = 0.06   # PLACEHOLDER (value of the extra geothermal heat the AHT unlocks)
    discount_rate: float = 0.05
    lifetime_years: int = 20


def capex_eur(s: float, Q_evap_design_kW: float, econ: EconomicAssumptions) -> float:
    """PLACEHOLDER CAPEX scaling law (six-tenths rule), anchored at
    capex_per_kW_eur * Q_evap_design_kW for s=1. TODO: replace with a real
    cost function (e.g. from vendor quotes or a scaling study)."""
    return econ.capex_per_kW_eur * Q_evap_design_kW * s ** econ.capex_scaling_exponent


def annual_cashflow_eur(summary: DispatchSummary, capex: float, econ: EconomicAssumptions) -> float:
    """PLACEHOLDER annual net cashflow. TODO: replace revenue/cost terms
    with your real tariff/avoided-cost structure."""
    revenue = (
        summary.cooling_served_kWh * econ.avoided_cooling_cost_eur_per_kWh
        + summary.heat_served_by_aht_kWh * econ.heat_value_eur_per_kWh
    )
    avoided_residual_cost = summary.cooling_residual_kWh * 0.0  # TODO: cost the residual to a decentral chiller
    electricity_cost = (
        (summary.ac_electricity_proxy_kWh + summary.aht_electricity_proxy_kWh)
        * econ.electricity_price_eur_per_kWh
    )
    opex = econ.opex_fraction_of_capex * capex
    return revenue + avoided_residual_cost - electricity_cost - opex


def npv_eur(summary: DispatchSummary, s: float, Q_evap_design_kW: float, econ: EconomicAssumptions) -> float:
    capex = capex_eur(s, Q_evap_design_kW, econ)
    annual_cf = annual_cashflow_eur(summary, capex, econ)
    discount_factors = 1.0 / (1.0 + econ.discount_rate) ** np.arange(1, econ.lifetime_years + 1)
    return -capex + annual_cf * discount_factors.sum()


# ---------------------------------------------------------------------------
# 5) Size sweep: reversible AC+AHT vs. pure AC
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    s: float
    reversible_npv_eur: float
    ac_only_npv_eur: float
    reversible_summary: DispatchSummary
    ac_only_summary: DispatchSummary


def size_sweep(
    s_values: np.ndarray,
    profile: LoadProfile,
    ac_map: CharacteristicMap,
    aht_map: CharacteristicMap,
    *,
    Q_evap_design_kW: float,
    Qabs_reference_kW: float,
    econ: EconomicAssumptions = EconomicAssumptions(),
    switchover: SwitchoverConfig = SwitchoverConfig(),
) -> list[SweepPoint]:
    points = []
    for s in s_values:
        reversible_summary = dispatch_year(
            profile, ac_map=ac_map, aht_map=aht_map, s=s,
            Q_evap_design_kW=Q_evap_design_kW, Qabs_reference_kW=Qabs_reference_kW,
            switchover=switchover,
        )
        ac_only_summary = dispatch_year(
            profile, ac_map=ac_map, aht_map=None, s=s,
            Q_evap_design_kW=Q_evap_design_kW, Qabs_reference_kW=Qabs_reference_kW,
            switchover=switchover,
        )
        points.append(SweepPoint(
            s=s,
            reversible_npv_eur=npv_eur(reversible_summary, s, Q_evap_design_kW, econ),
            ac_only_npv_eur=npv_eur(ac_only_summary, s, Q_evap_design_kW, econ),
            reversible_summary=reversible_summary,
            ac_only_summary=ac_only_summary,
        ))
    return points


def print_sweep_table(points: list[SweepPoint]) -> None:
    print("=" * 100)
    print("WARNING: all NPV numbers use PLACEHOLDER economics (EconomicAssumptions) -- do not quote them.")
    print("=" * 100)
    print(f"{'s':>6} {'NPV reversible [EUR]':>22} {'NPV AC-only [EUR]':>20} {'AC FLH':>8} {'AHT FLH':>8}")
    for p in points:
        print(
            f"{p.s:6.2f} {p.reversible_npv_eur:22,.0f} {p.ac_only_npv_eur:20,.0f} "
            f"{p.reversible_summary.ac_full_load_hours:8.0f} {p.reversible_summary.aht_full_load_hours:8.0f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    if not AC_MAP_PATH.exists() or not AHT_MAP_PATH.exists():
        raise SystemExit(
            "Characteristic maps not found -- run TEA/design_point_export.py, then "
            "TEA/ac_characteristic_map.py and TEA/aht_characteristic_map.py first."
        )

    ac_map = CharacteristicMap.load(AC_MAP_PATH)
    aht_map = CharacteristicMap.load(AHT_MAP_PATH)

    # TODO: swap this for load_profile_csv(Path("your_lastgang.csv"))
    profile = synthetic_demo_profile()

    s_values = np.arange(0.5, 2.01, 0.25)
    points = size_sweep(
        s_values, profile, ac_map, aht_map,
        Q_evap_design_kW=ac_map.meta["Q_evap_design_kW"],
        Qabs_reference_kW=aht_map.meta["Qabs_reference_kW"],
    )
    print_sweep_table(points)
