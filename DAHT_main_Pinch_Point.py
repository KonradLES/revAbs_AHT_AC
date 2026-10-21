"""Entry point for the double-lift AHT (DAHT) simulation, 10 primary unknowns.

Topology: see Models/DAHT_Pinch_Point.py's module docstring -- verified
directly against Lubis et al. 2017 (Figs. 3 and 5, Table 4).

Absorber (AH) specification is explicitly selectable:
- ABSORBER_SPEC_MODE = "m11" -> give m11_spec, T12 is computed
- ABSORBER_SPEC_MODE = "T12" -> give T12_spec_C, m11 is computed

Desorber specification is explicitly selectable:
- DESORBER_SPEC_MODE = "m13" -> give m13_spec, T14 is computed
- DESORBER_SPEC_MODE = "T14" -> give T14_spec_C, m13 is computed

Evaporator (EL) specification is explicitly selectable:
- EVAPORATOR_SPEC_MODE = "m15" -> give m15_spec, T16 is computed
- EVAPORATOR_SPEC_MODE = "T16" -> give T16_spec_C, m15 is computed

Condenser specification is explicitly selectable:
- CONDENSER_SPEC_MODE = "m17" -> give m17_spec, T18 is computed
- CONDENSER_SPEC_MODE = "T18" -> give T18_spec_C, m17 is computed

Cycle scaling is explicitly selectable:
- CYCLE_SCALE_SPEC_MODE = "m_gen"   -> give m_gen_spec directly
- CYCLE_SCALE_SPEC_MODE = "QAH"     -> give QAH_spec_kW (the useful output), m_gen is computed
- CYCLE_SCALE_SPEC_MODE = "Qwaste"  -> give Qwaste_spec_kW (= Q_des + Q_EL, the total waste-heat
  input), m_gen is computed. This is normally the most natural way to size a design point: you
  know how much waste heat is available, not what solution flow rate to run.

T-spec modes (fixed temperature approaches, derived mass flows) give much better-conditioned,
realistically-sized design points than m-spec modes with guessed flow rates -- an arbitrarily
chosen mass flow can leave a stream barely changing temperature (a "starved", low-utilization
design), which was found during development to depress the COP into unrealistic territory
(<0.05) even though the model itself was solving correctly. T-spec + "Qwaste" scaling is the
recommended combination and is what's used below by default.
"""

from __future__ import annotations

from Models.DAHT_Pinch_Point import (
    DAHTInputs,
    initial_guess,
    print_summary,
    solve_daht_robust,
)

# ----------------------------------------------------------------------------

# CYCLE_SCALE_SPEC_MODE = "m_gen"
# CYCLE_SCALE_SPEC_MODE = "QAH"
CYCLE_SCALE_SPEC_MODE = "Qwaste"

# ABSORBER_SPEC_MODE = "m11"
ABSORBER_SPEC_MODE = "T12"

# DESORBER_SPEC_MODE = "m13"
DESORBER_SPEC_MODE = "T14"

# EVAPORATOR_SPEC_MODE = "m15"
EVAPORATOR_SPEC_MODE = "T16"

# CONDENSER_SPEC_MODE = "m17"
CONDENSER_SPEC_MODE = "T18"


def build_example_inputs() -> DAHTInputs:
    common_kwargs = dict(
        T_11_C=85.0,   # AH heat-sink inlet (e.g. boiler feed water)
        T_13_C=60.0,   # waste-heat source inlet, routed in series into EL (matches Lubis et al.)
        T_17_C=15.0,   # condenser cooling-water inlet
        dT_min_SL=3.0,
        dT_min_des=3.0,
        dT_min_cond=3.0,
        dT_min_EL=3.0,
        dT_min_AL_EH=3.0,
        dT_min_AH=3.0,
        dT_min_SH=3.0,
        cp_w_kJkgK=4.18,
        desorber_vapor_superheat_K=0.0,
        absorber_spec_mode=ABSORBER_SPEC_MODE,
        desorber_spec_mode=DESORBER_SPEC_MODE,
        evaporator_spec_mode=EVAPORATOR_SPEC_MODE,
        condenser_spec_mode=CONDENSER_SPEC_MODE,
        cycle_scale_spec_mode=CYCLE_SCALE_SPEC_MODE,
    )

    spec_kwargs: dict[str, float] = {}

    if CYCLE_SCALE_SPEC_MODE == "m_gen":
        spec_kwargs["m_gen_spec"] = 0.11
    elif CYCLE_SCALE_SPEC_MODE == "QAH":
        spec_kwargs["QAH_spec_kW"] = 31.4
    elif CYCLE_SCALE_SPEC_MODE == "Qwaste":
        spec_kwargs["Qwaste_spec_kW"] = 100.0  # total waste heat into the desorber + EL
    else:
        raise ValueError("CYCLE_SCALE_SPEC_MODE must be 'm_gen', 'QAH', or 'Qwaste'.")

    if ABSORBER_SPEC_MODE == "m11":
        spec_kwargs["m11_spec"] = 0.07
    elif ABSORBER_SPEC_MODE == "T12":
        spec_kwargs["T12_spec_C"] = 100.0  # useful output temperature -- 52 K lift above T13
    else:
        raise ValueError("ABSORBER_SPEC_MODE must be 'm11' or 'T12'.")

    if DESORBER_SPEC_MODE == "m13":
        spec_kwargs["m13_spec"] = 2.0
    elif DESORBER_SPEC_MODE == "T14":
        spec_kwargs["T14_spec_C"] = 55.0  # T13 - 8 K approach
    else:
        raise ValueError("DESORBER_SPEC_MODE must be 'm13' or 'T14'.")

    if EVAPORATOR_SPEC_MODE == "m15":
        spec_kwargs["m15_spec"] = 1.0
    elif EVAPORATOR_SPEC_MODE == "T16":
        spec_kwargs["T16_spec_C"] = 52.0  # T15(=T14) - 8 K approach
    else:
        raise ValueError("EVAPORATOR_SPEC_MODE must be 'm15' or 'T16'.")

    if CONDENSER_SPEC_MODE == "m17":
        spec_kwargs["m17_spec"] = 2.0
    elif CONDENSER_SPEC_MODE == "T18":
        spec_kwargs["T18_spec_C"] = 22.0  # T17 + 8 K approach
    else:
        raise ValueError("CONDENSER_SPEC_MODE must be 'm17' or 'T18'.")

    return DAHTInputs(**common_kwargs, **spec_kwargs)


if __name__ == "__main__":
    inputs = build_example_inputs()

    x0 = initial_guess(inputs)

    # solve_daht_robust() tries the fast unconstrained solver first, and
    # automatically falls back to the slower, ordering-safe gap-parametrized
    # solver if the first attempt lands on a "shadow" root with the
    # concentrations in the wrong order -- see Models/DAHT_Pinch_Point.py.
    result = solve_daht_robust(inputs, x0=x0)
    print_summary(result)
