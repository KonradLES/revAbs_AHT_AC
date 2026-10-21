"""Validates the component-based network against Models/AHT_Pinch_Point.py.

Runs both models on the same AHTInputs and compares every shared state
(T, m, h) and heat flow. This is a parallel, additional model -- it is
only trustworthy to the extent this comparison passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from Models.AHT_Pinch_Point import AHTInputs
from Models.AHT_Pinch_Point import solve_aht as solve_reference
from Models.AHT_Pinch_Point import primary_temperatures_C_to_K

from Component_Model_AHT import network as net

REFERENCE_STATE_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "12", "14", "16", "18"]


def build_inputs(*, cycle_scale_spec_mode: str, m6_spec=None, Qabs_spec_kW=None) -> AHTInputs:
    return AHTInputs(
        T_11_C=135.0,
        T_13_C=120.0,
        T_15_C=120.0,
        T_17_C=30.0,
        dT_min_shex=4.282178,
        dT_min_des=6.257224,
        dT_min_cond=14.224404,
        dT_min_evap=8.618245,
        dT_min_abs=17.836021,
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        absorber_spec_mode="T12",
        T12_spec_C=146.02,
        desorber_spec_mode="T14",
        T14_spec_C=108.92,
        evaporator_spec_mode="T16",
        T16_spec_C=108.80,
        condenser_spec_mode="T18",
        T18_spec_C=41.26,
        desorber_evaporator_routing_mode="parallel",
        cycle_scale_spec_mode=cycle_scale_spec_mode,
        m6_spec=m6_spec,
        Qabs_spec_kW=Qabs_spec_kW,
    )


def compare(inputs: AHTInputs, x0_7: np.ndarray, label: str) -> bool:
    print(f"\n=== {label} ===")

    ref = solve_reference(inputs, x0=x0_7)
    if not ref.solve_info.success:
        print("Reference model FAILED to converge:", ref.solve_info.message)
        return False

    comp_x0 = net.initial_guess(inputs)
    comp_result, comp_ev = net.solve(inputs, x0=comp_x0)
    if not comp_result.success:
        print("Component model FAILED to converge:", comp_result.message)
        return False

    print(f"reference nfev={ref.solve_info.nfev}  component nfev={comp_result.nfev}")

    ok = True
    max_rel_err = 0.0
    for key in REFERENCE_STATE_KEYS:
        ref_state = ref.states[key]
        comp_state = comp_ev.states.get(key)
        if comp_state is None:
            continue
        for field in ("T_K", "m_kg_s", "h_kJ_kg"):
            if field not in ref_state or field not in comp_state:
                continue
            a, b = ref_state[field], comp_state[field]
            rel_err = abs(a - b) / max(abs(a), 1.0e-9)
            max_rel_err = max(max_rel_err, rel_err)
            status = "OK" if rel_err < 1.0e-6 else "MISMATCH"
            if status == "MISMATCH":
                ok = False
                print(f"  state {key:>3s} {field:10s} ref={a: .6f}  comp={b: .6f}  rel_err={rel_err:.3e}  {status}")

    for key in ("Q_shex", "Q_des", "Q_cond", "Q_evap", "Q_abs"):
        a, b = ref.heat_flows_kW[key], comp_ev.heat_flows_kW[key]
        rel_err = abs(a - b) / max(abs(a), 1.0e-9)
        max_rel_err = max(max_rel_err, rel_err)
        status = "OK" if rel_err < 1.0e-6 else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(f"  {key:10s} ref={a: .6f}  comp={b: .6f}  rel_err={rel_err:.3e}  {status}")

    m6_ref = ref.diagnostics["m6_kg_s"]
    rel_err = abs(m6_ref - comp_ev.m6) / max(abs(m6_ref), 1.0e-9)
    max_rel_err = max(max_rel_err, rel_err)
    print(f"  {'m6':10s} ref={m6_ref: .6f}  comp={comp_ev.m6: .6f}  rel_err={rel_err:.3e}")

    print(f"  max relative error across all checked quantities: {max_rel_err:.3e}")
    print("  RESULT:", "MATCH" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    x0 = primary_temperatures_C_to_K(
        np.array([55, 101, 0.23, 0.27, 0.26, 121, 150], dtype=float)
    )

    all_ok = True
    all_ok &= compare(
        build_inputs(cycle_scale_spec_mode="m6", m6_spec=1.0), x0, "cycle_scale_spec_mode='m6'"
    )
    all_ok &= compare(
        build_inputs(cycle_scale_spec_mode="Qabs", Qabs_spec_kW=184.4), x0, "cycle_scale_spec_mode='Qabs'"
    )

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
