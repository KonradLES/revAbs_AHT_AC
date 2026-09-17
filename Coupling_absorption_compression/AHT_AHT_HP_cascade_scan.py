"""Rough feasibility probe: does stacking a SECOND single-lift AHT stage
behind the first one meaningfully extend the achievable lift, compared to
topping up with a compression HP right away (Configuration 1 in
AHT_HP_cascade_scan.py)? This is a cheap way to bound the operating range a
real double-lift (DL) AHT could offer -- WITHOUT building one -- to help
decide whether a full DL-AHT model (see Design_Point/AHT_DL_duehring_screening.py
for the real double-lift physics) is worth the effort.

Configuration
-------------
    DC -> AHT#1 -> AHT#2 -> DH                      (if AHT#2 alone reaches T_demand_supply_C)
    DC -> AHT#1 -> AHT#2 -> HP -> DH   (fallback)    (if it doesn't -- HP tops up the residual lift)

Two single-lift AHTs in series are NOT the same as a genuine double-lift
AHT (no shared internal solution circuit, no shared crystallization limit
across both stages, twice the SHEX/pump overhead) -- this script only
estimates the extended RANGE, not a physically faithful DL-AHT design
point. Kept deliberately separate from AHT_HP_cascade_scan.py since this
scenario is exploratory and may not be needed going forward.

Reuses ScenarioConfig and all AHT/HP solve/continuation helpers from
AHT_HP_cascade_scan.py unchanged -- see that module's docstring for the
scale-fixing mechanisms (`Qdes_eva` for AHT, `_solve_hp_for_Q_evap`'s
COP-invariance trick for HP) and the "simplified captive-loop coupling"
design choice (fixed glide dT_mid_C, duty-only handover between stages).

Sweep structure
----------------
One free variable is swept explicitly: T_mid1, the handover temperature
between AHT#1's absorber and AHT#2's desorber/evaporator (same role as
T_mid in the single-AHT cascades). For each T_mid1:
    1. AHT#1 is solved first (fully pinned by the fixed DC input, same
       causal order as Configuration 1) via the shared continuation-walk
       driver (_advance_aht_leg), with the same "past the confirmed end of
       the feasible window" skip-ahead optimization.
    2. AHT#2 first tries to reach the DH target (T_demand_supply_C)
       DIRECTLY, exactly like solve_aht_alone() does for a single AHT
       (T11 fixed at T_DH_return_C, i.e. serving the DH network on its
       own glide). If that succeeds, no HP is needed at this T_mid1 --
       "2 AHT alone" wins outright.
    3. If not, a SECOND free variable (T_mid2, AHT#2's own achievable
       absorber outlet, on a captive-loop glide -- same dT_mid_C
       convention Configuration 1 uses to feed its HP) is found via a
       bounded one-shot anchor+bisection search (_find_aht_max_T12), not a
       nested grid sweep (kept cheap: this only runs in the fallback
       case, once per T_mid1, not once per (T_mid1, T_mid2) pair). An HP
       then tops up the residual lift from that best achievable T_mid2 up
       to T_demand_supply_C, via the same _solve_hp_for_Q_evap()
       COP-invariance trick.

Standalone usage
-----------------
    python Coupling_absorption_compression/AHT_AHT_HP_cascade_scan.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from Models.AHT_Pinch_Point import PRIMARY_VARIABLE_NAMES
from Compression_Models.HeatPump_Pinch_Point import HeatPumpModel
from Coupling_absorption_compression.AHT_HP_cascade_scan import (
    ScenarioConfig,
    T_MID_MARGIN_C,
    T_MID_STEP_C,
    _aht_duehring_T12_max,
    _build_aht_inputs,
    _solve_aht,
    _is_valid_aht_solution,
    _locate_aht_point,
    _advance_aht_leg,
    _solve_hp_for_Q_evap,
    solve_aht_alone,
    solve_hp_alone,
    sweep_aht_then_hp,
)


plot_name = "Coupling_absorption_compression/Plots/aht_aht_hp_vs_aht_hp"


def _find_aht_max_T12(build_inputs_fn, T_anchor_guess_C: float, T_hi_bisect_C: float, *, span_C: float = 25.0):
    """One-shot search (NOT a per-point sweep) for the maximum T12 up to
    T_hi_bisect_C that build_inputs_fn(T12) can reach -- anchor via
    _locate_aht_point() near T_anchor_guess_C, then bisect upward toward
    T_hi_bisect_C, exploiting the same "the feasible window is a single
    contiguous interval" property already relied on throughout this
    repo's AHT tooling. Used only as a fallback when AHT#2 can't reach
    the DH target directly in a single solve.

    T_anchor_guess_C should be the Duehring ceiling (a closed-form,
    optimistic upper bound -- see _aht_duehring_T12_max) rather than
    T_hi_bisect_C itself: the real achievable max sits close to, but
    below, that ceiling, so anchoring there instead of at the (typically
    much higher) DH target avoids a long, mostly-wasted fan-out walk.

    Returns (T12_max_C, result_at_T12_max), or (None, None) if nothing in
    the range is feasible at all.
    """
    T_anchor, x0_anchor, result_anchor = _locate_aht_point(
        build_inputs_fn, T_anchor_guess_C, span_C=span_C, step_C=1.0,
    )
    if T_anchor is None:
        return None, None
    if T_anchor >= T_hi_bisect_C - 1.0e-6:
        return T_anchor, result_anchor

    lo, x0_lo, result_lo = T_anchor, x0_anchor, result_anchor
    hi = T_hi_bisect_C
    for _ in range(25):
        if hi - lo < 0.25:
            break
        mid = 0.5 * (lo + hi)
        try:
            result = _solve_aht(build_inputs_fn(mid), x0=x0_lo)
        except Exception:
            result = None
        if result is not None and _is_valid_aht_solution(result):
            lo, result_lo = mid, result
            x0_lo = np.array([result.primary_variables[n] for n in PRIMARY_VARIABLE_NAMES])
        else:
            hi = mid
    return lo, result_lo


# ---------------------------------------------------------------------------
# Configuration: DC -> AHT#1 -> AHT#2 (-> HP if needed) -> DH
# ---------------------------------------------------------------------------

def sweep_aht_aht_then_hp(config: ScenarioConfig, T_mid1_values_C):
    hp_model = HeatPumpModel(refrigerant=config.refrigerant)
    hp_model.nw.iterinfo = False

    # AHT#1: T13=T15=T_waste_heat_in_C is constant throughout this sweep.
    T12_max_aht1 = _aht_duehring_T12_max(
        config.T_waste_heat_in_C, config.T_waste_heat_in_C, config.T_reject_in_C, config,
    )

    records = []
    aht1_state = {"T": None, "x0": None}
    found_any_feasible = False
    past_window = False
    for T_mid1_C in T_mid1_values_C:
        if past_window:
            records.append({
                "T_mid1_C": T_mid1_C, "feasible": False,
                "reason": "AHT#1: skipped -- past the confirmed end of the feasible window",
            })
            continue

        if T12_max_aht1 is None or T_mid1_C > T12_max_aht1:
            records.append({
                "T_mid1_C": T_mid1_C, "feasible": False,
                "reason": f"AHT#1: T_mid1 > Duehring ceiling T12_max={T12_max_aht1}",
            })
            aht1_state["T"], aht1_state["x0"] = None, None
            continue

        GTL1_est = T_mid1_C - config.T_waste_heat_in_C
        if GTL1_est < config.min_GTL_K:
            records.append({"T_mid1_C": T_mid1_C, "feasible": False, "reason": f"AHT#1: GTL={GTL1_est:.1f} K < min_GTL_K"})
            aht1_state["T"], aht1_state["x0"] = None, None
            continue

        def build_fn1(T):
            return _build_aht_inputs(
                T11_C=T - config.dT_mid_C, T12_C=T,
                T_ext_hot_C=config.T_waste_heat_in_C, T_ext_cold_C=config.T_DC_out_C,
                T17_C=config.T_reject_in_C, T18_C=config.T_reject_out_C,
                Qdes_eva_kW=config.Q_DC_kW, config=config, fast=True,
            )

        aht1_result, aht1_reason = _advance_aht_leg(build_fn1, T_mid1_C, aht1_state)
        if aht1_result is None:
            records.append({"T_mid1_C": T_mid1_C, "feasible": False, "reason": aht1_reason})
            if found_any_feasible:
                past_window = True
            continue
        found_any_feasible = True
        P_AHT1_kW = aht1_result.pump_work_kW["W_AHT_total"]
        Q_captive1_kW = aht1_result.heat_flows_kW["Q_abs"]

        # AHT#2 is fed by AHT#1's absorber output (T13=T15=T_mid1_C).
        T14_aht2 = T_mid1_C - config.dT_mid_C

        def build_fn2_direct(T12, _Qdes_eva_kW=Q_captive1_kW, _T_hot=T_mid1_C, _T_cold=T14_aht2):
            # Serves the DH network directly, same convention as
            # solve_aht_alone(): T11 fixed at the DH return temperature.
            return _build_aht_inputs(
                T11_C=config.T_DH_return_C, T12_C=T12,
                T_ext_hot_C=_T_hot, T_ext_cold_C=_T_cold,
                T17_C=config.T_reject_in_C, T18_C=config.T_reject_out_C,
                Qdes_eva_kW=_Qdes_eva_kW, config=config, fast=True,
            )

        def build_fn2_captive(T, _Qdes_eva_kW=Q_captive1_kW, _T_hot=T_mid1_C, _T_cold=T14_aht2):
            # Feeds a captive loop into an HP top-up instead, same
            # dT_mid_C-glide convention Configuration 1 uses for its AHT
            # -> HP handover.
            return _build_aht_inputs(
                T11_C=T - config.dT_mid_C, T12_C=T,
                T_ext_hot_C=_T_hot, T_ext_cold_C=_T_cold,
                T17_C=config.T_reject_in_C, T18_C=config.T_reject_out_C,
                Qdes_eva_kW=_Qdes_eva_kW, config=config, fast=True,
            )

        T12_max_aht2 = _aht_duehring_T12_max(T_mid1_C, T_mid1_C, config.T_reject_in_C, config)
        if T12_max_aht2 is None:
            records.append({"T_mid1_C": T_mid1_C, "feasible": False, "reason": "AHT#2: Duehring ceiling not computable"})
            continue

        two_aht_result = None
        if config.T_demand_supply_C <= T12_max_aht2:
            try:
                candidate = _solve_aht(build_fn2_direct(config.T_demand_supply_C))
            except Exception:
                candidate = None
            if candidate is not None and _is_valid_aht_solution(candidate):
                two_aht_result = candidate

        if two_aht_result is not None:
            P_AHT2_kW = two_aht_result.pump_work_kW["W_AHT_total"]
            Q_delivered_kW = two_aht_result.heat_flows_kW["Q_abs"]
            P_total_kW = P_AHT1_kW + P_AHT2_kW
            records.append({
                "T_mid1_C": T_mid1_C, "feasible": True, "hp_needed": False,
                "T_mid2_C": config.T_demand_supply_C,
                "Q_delivered_kW": Q_delivered_kW, "P_total_kW": P_total_kW,
                "specific_power_kWel_per_kWth": P_total_kW / Q_delivered_kW,
                "AHT1_COP": aht1_result.kpis.get("COP", float("nan")),
                "AHT2_COP": two_aht_result.kpis.get("COP", float("nan")),
            })
            continue

        # AHT#2 alone can't reach the DH target in one hop -- find its own
        # best achievable absorber outlet, then top up the rest with an HP.
        T_mid2_max_C, result2 = _find_aht_max_T12(
            build_fn2_captive, min(T12_max_aht2, config.T_demand_supply_C), config.T_demand_supply_C,
        )
        if T_mid2_max_C is None:
            records.append({"T_mid1_C": T_mid1_C, "feasible": False, "reason": "AHT#2: no valid solution anywhere in range"})
            continue

        P_AHT2_kW = result2.pump_work_kW["W_AHT_total"]
        Q_captive2_kW = result2.heat_flows_kW["Q_abs"]

        hp_result, hp_reason = _solve_hp_for_Q_evap(
            hp_model,
            T_source_in_C=T_mid2_max_C, T_source_out_C=T_mid2_max_C - config.dT_mid_C,
            T_sink_in_C=config.T_DH_return_C, T_sink_out_C=config.T_demand_supply_C,
            Q_evap_target_kW=Q_captive2_kW, config=config,
        )
        if hp_result is None:
            records.append({"T_mid1_C": T_mid1_C, "feasible": False, "reason": f"HP top-up: {hp_reason}"})
            continue
        P_HP_el_kW = hp_result.heat_flows_kW["P_compressor"] / config.eta_mech_motor
        Q_delivered_kW = abs(hp_result.heat_flows_kW["Q_cond"])

        P_total_kW = P_AHT1_kW + P_AHT2_kW + P_HP_el_kW
        records.append({
            "T_mid1_C": T_mid1_C, "feasible": True, "hp_needed": True,
            "T_mid2_C": T_mid2_max_C,
            "Q_delivered_kW": Q_delivered_kW, "P_total_kW": P_total_kW,
            "specific_power_kWel_per_kWth": P_total_kW / Q_delivered_kW,
            "AHT1_COP": aht1_result.kpis.get("COP", float("nan")),
            "AHT2_COP": result2.kpis.get("COP", float("nan")),
            "HP_COP": hp_result.kpis["COP"],
        })

    return records


# ---------------------------------------------------------------------------
# Plot + reporting
# ---------------------------------------------------------------------------

def plot_comparison(records_aht_aht_hp, records_aht_hp, aht_alone: dict, hp_alone: dict, config: ScenarioConfig):
    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(13.5, 6.0))

    feasible_ref = [r for r in records_aht_hp if r["feasible"]]
    if feasible_ref:
        T_mid = [r["T_mid_C"] for r in feasible_ref]
        ax_p.plot(T_mid, [r["P_total_kW"] for r in feasible_ref], "o-", color="tab:blue", label="DC -> AHT -> HP -> DH")
        ax_q.plot(T_mid, [r["Q_delivered_kW"] for r in feasible_ref], "o-", color="tab:blue", label="DC -> AHT -> HP -> DH")

    two_aht_only = [r for r in records_aht_aht_hp if r["feasible"] and not r["hp_needed"]]
    two_aht_hp = [r for r in records_aht_aht_hp if r["feasible"] and r["hp_needed"]]
    if two_aht_only:
        T_mid1 = [r["T_mid1_C"] for r in two_aht_only]
        ax_p.plot(T_mid1, [r["P_total_kW"] for r in two_aht_only], "o-", color="tab:purple", label="DC -> AHT -> AHT -> DH")
        ax_q.plot(T_mid1, [r["Q_delivered_kW"] for r in two_aht_only], "o-", color="tab:purple", label="DC -> AHT -> AHT -> DH")
    if two_aht_hp:
        T_mid1 = [r["T_mid1_C"] for r in two_aht_hp]
        ax_p.plot(T_mid1, [r["P_total_kW"] for r in two_aht_hp], "^--", color="tab:purple", label="DC -> AHT -> AHT -> HP -> DH")
        ax_q.plot(T_mid1, [r["Q_delivered_kW"] for r in two_aht_hp], "^--", color="tab:purple", label="DC -> AHT -> AHT -> HP -> DH")

    for key, label_fmt in (
        (aht_alone, lambda r: f"DC -> AHT -> DH (COP={r['AHT_COP']:.2f})"),
        (hp_alone, lambda r: f"DC -> HP -> DH (COP={r['HP_COP']:.2f})"),
    ):
        if not key["feasible"]:
            continue
        color = "tab:orange" if key is aht_alone else "tab:red"
        ax_p.axhline(key["P_el_kW"], color=color, linestyle=":", label=label_fmt(key))
        ax_q.axhline(key["Q_delivered_kW"], color=color, linestyle=":", label=label_fmt(key))

    ax_q.axhline(config.Q_DC_kW, color="0.4", linestyle=":", label=f"DC waste-heat input (fixed, {config.Q_DC_kW:.0f} kW)")

    ax_p.set_xlabel("Handover temperature T_mid1 [°C]")
    ax_p.set_ylabel("Electrical power demand P_el [kW]")
    ax_p.legend(fontsize=8)
    ax_p.grid(alpha=0.3)

    ax_q.set_xlabel("Handover temperature T_mid1 [°C]")
    ax_q.set_ylabel("Delivered DH heat Q_delivered [kW]")
    ax_q.legend(fontsize=8)
    ax_q.grid(alpha=0.3)

    fig.tight_layout()

    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{plot_name}.svg")
    fig.savefig(f"{plot_name}.png", dpi=150)
    print(f"Saved plot to {plot_name}.svg / .png")


def _print_sweep_table(name: str, records) -> None:
    print(f"\n{name}:")
    n_feasible = sum(1 for r in records if r["feasible"])
    print(f"  Feasible T_mid1 points: {n_feasible} / {len(records)}")
    for r in records:
        if r["feasible"]:
            stage = "2 AHT alone" if not r["hp_needed"] else "2 AHT + HP top-up"
            hp_cop = f"  HP_COP={r['HP_COP']:.2f}" if r["hp_needed"] else ""
            print(
                f"    T_mid1={r['T_mid1_C']:5.1f} C  T_mid2={r['T_mid2_C']:5.1f} C  [{stage}]  "
                f"P_total={r['P_total_kW']:6.2f} kW  Q_delivered={r['Q_delivered_kW']:7.1f} kW  "
                f"spec.power={r['specific_power_kWel_per_kWth']:.4f}  "
                f"AHT1_COP={r['AHT1_COP']:.2f}  AHT2_COP={r['AHT2_COP']:.2f}{hp_cop}"
            )
        else:
            print(f"    T_mid1={r['T_mid1_C']:5.1f} C  infeasible ({r['reason']})")
    if n_feasible:
        best = min((r for r in records if r["feasible"]), key=lambda r: r["specific_power_kWel_per_kWth"])
        stage = "2 AHT alone" if not best["hp_needed"] else "2 AHT + HP top-up"
        print(
            f"  Best (lowest spec.power): T_mid1={best['T_mid1_C']:.1f} C [{stage}], "
            f"P_total={best['P_total_kW']:.2f} kW, Q_delivered={best['Q_delivered_kW']:.1f} kW, "
            f"spec.power={best['specific_power_kWel_per_kWth']:.4f}"
        )


if __name__ == "__main__":
    config = ScenarioConfig()

    print("=" * 70)
    print(
        f"DC waste heat {config.T_waste_heat_in_C:.1f} -> {config.T_DC_out_C:.1f} C @ "
        f"{config.Q_DC_kW:.0f} kW (fixed input), "
        f"DH network {config.T_DH_return_C:.1f} -> {config.T_demand_supply_C:.1f} C"
    )
    print("DC -> AHT -> AHT (-> HP if needed) -> DH -- rough DL-AHT range estimate")
    print("=" * 70)

    print("\nReference (single AHT stage, Configuration 2 of AHT_HP_cascade_scan.py):")
    aht_alone = solve_aht_alone(config)
    if aht_alone["feasible"]:
        print(f"  FEASIBLE -- P_el={aht_alone['P_el_kW']:.2f} kW, Q_delivered={aht_alone['Q_delivered_kW']:.1f} kW")
    else:
        print(f"  INFEASIBLE ({aht_alone['reason']})")

    print("\nReference (HP alone, Configuration 3):")
    hp_alone = solve_hp_alone(config)
    if hp_alone["feasible"]:
        print(f"  FEASIBLE -- P_el={hp_alone['P_el_kW']:.2f} kW, Q_delivered={hp_alone['Q_delivered_kW']:.1f} kW")
    else:
        print(f"  INFEASIBLE ({hp_alone['reason']})")

    T_mid1_grid = list(np.arange(
        config.T_waste_heat_in_C + T_MID_MARGIN_C,
        config.T_demand_supply_C - T_MID_MARGIN_C + 1.0e-9,
        T_MID_STEP_C,
    ))

    print("\nReference (single AHT + HP top-up, Configuration 1):")
    records_aht_hp = sweep_aht_then_hp(config, T_mid1_grid)
    _n = sum(1 for r in records_aht_hp if r["feasible"])
    print(f"  Feasible T_mid points: {_n} / {len(records_aht_hp)}")

    records_aht_aht_hp = sweep_aht_aht_then_hp(config, T_mid1_grid)
    _print_sweep_table("DC -> AHT -> AHT (-> HP if needed) -> DH", records_aht_aht_hp)

    plot_comparison(records_aht_aht_hp, records_aht_hp, aht_alone, hp_alone, config)
