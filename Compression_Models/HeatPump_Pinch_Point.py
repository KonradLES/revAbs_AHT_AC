"""Water/water compression heat pump, built on TESPy.

Companion model to the H2O/LiBr absorption heat transformer (AHT) in
revAbs_AHT_AC (Models/AHT_Pinch_Point.py). Mirrors that model's interface
and design philosophy so both plants can eventually be parameterized and
swept the same way:

- pinch-point (minimum approach temperature) specification per heat
  exchanger instead of a fixed duty or fixed refrigerant temperature
- external water streams can each be specified either by mass flow
  (outlet temperature is computed) or by outlet temperature (mass flow
  is computed) -- same "m" / "T" toggle as the AHT's absorber/desorber/
  evaporator/condenser streams
- overall plant size ("cycle scale") is fixed either via the useful heat
  duty (condenser) or directly via the refrigerant mass flow
- a temperature_lift KPI defined the same way as the AHT's GTL
  (useful-heat outlet minus source inlet), for apples-to-apples
  comparison in a combined plot later

Unlike the AHT model, this one does NOT implement its own nonlinear
solver: TESPy's Network already provides that, so the "similar
complexity" here shows up as the richness of the specification options
and KPIs, not as hand-rolled residual equations.

Model assumptions
------------------
- refrigerant: subcritical vapor-compression cycle, default R134a
- steady-state operation
- no pressure losses in piping or in the heat exchangers on the
  refrigerant side (pr = 1); optional pressure ratios on the water
  sides only
- isenthalpic expansion valve
- external fluid: water, evaluated with the network's real fluid
  property backend (not a constant cp_w approximation, unlike the AHT
  model's external-stream treatment)

Plant layout and connection labels
-----------------------------------
Refrigerant loop (cycle closer 'cc'):
    cc -(0)-> evaporator(in2)  -(1)-> compressor -(2)-> condenser(in1)
    -(3)-> valve -(4)-> cc

External source loop (heat is taken FROM this stream, e.g. ambient/waste
water):
    source_in -(11)-> evaporator(in1) -(12)-> source_out

External sink loop (heat is delivered TO this stream, e.g. heating/process
water):
    sink_in -(21)-> condenser(in2) -(22)-> sink_out

Pinch specification: TESPy's `ttd_min`, i.e. min(ttd_u, ttd_l), NOT a fixed
`ttd_l`. For a modest external glide (sink/source outlet - inlet spread
much smaller than the lift), the cold-end approach (ttd_l: refrigerant
outlet vs. the coldest external temperature) is normally the tighter one,
which is what earlier versions of this model hardcoded. But for a large
external glide -- e.g. a condenser heating district-heating water across
30-40 K in one pass, as happens in the AHT+HP cascade coupling script --
the warm-end approach (ttd_u: refrigerant inlet vs. the hottest external
temperature) can become the tighter constraint instead, or the fixed-ttd_l
formulation can become infeasible even though a valid design with a
higher condensing pressure exists. `ttd_min` lets the solver pick
whichever end actually binds, at the cost of only checking the two
terminal points -- like the AHT model's own pinch treatment, this still
does not check the internal desuperheat/subcooling transition point
pinch (see the AHT model's `dT_cond_sat` / `dT_evap_sat` candidates for
what a fully rigorous treatment looks like); for that level of fidelity
here, TESPy's `MovingBoundaryHeatExchanger` or sectioned heat exchanger
components would be the next step.
"""

from __future__ import annotations

from dataclasses import dataclass, KW_ONLY
from typing import Dict, Optional

from tespy.networks import Network
from tespy.components import (
    CycleCloser, Compressor, Valve, HeatExchanger, Source, Sink,
)
from tespy.connections import Connection


# ---------------------------------------------------------------------------
# Inputs / results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeatPumpInputs:
    # External inlet temperatures [degC]
    T_source_in_C: float
    T_sink_in_C: float

    # Pinch (minimum approach temperature) values [K]
    dT_min_evap: float
    dT_min_cond: float

    _: KW_ONLY

    refrigerant: str = "R134a"
    eta_s: float = 0.85

    # Pressure ratios, water sides only (refrigerant side is pr = 1)
    pr_source: float = 1.0
    pr_sink: float = 1.0

    # Refrigerant-side state at the phase boundary
    superheat_K: float = 0.0    # superheat at evaporator outlet (compressor inlet)
    subcooling_K: float = 0.0   # subcooling at condenser outlet (valve inlet)

    # Cycle scaling specification:
    # - "Q_cond": Q_cond_kW is given, refrigerant mass flow is computed
    # - "m_refrigerant": m_refrigerant is given, Q_cond is computed
    scale_spec_mode: str = "Q_cond"
    Q_cond_kW: Optional[float] = None
    m_refrigerant: Optional[float] = None

    # External source stream specification:
    # - "m": m_source is given, T_source_out_C is computed
    # - "T": T_source_out_C is given, m_source is computed
    source_spec_mode: str = "m"
    m_source: Optional[float] = None
    T_source_out_C: Optional[float] = None

    # External sink stream specification:
    # - "m": m_sink is given, T_sink_out_C is computed
    # - "T": T_sink_out_C is given, m_sink is computed
    sink_spec_mode: str = "T"
    m_sink: Optional[float] = None
    T_sink_out_C: Optional[float] = None

    def __post_init__(self) -> None:
        if self.dT_min_evap <= 0.0:
            raise ValueError(f"dT_min_evap must be positive, but is {self.dT_min_evap}.")
        if self.dT_min_cond <= 0.0:
            raise ValueError(f"dT_min_cond must be positive, but is {self.dT_min_cond}.")
        if self.superheat_K < 0.0:
            raise ValueError(f"superheat_K must be >= 0, but is {self.superheat_K}.")
        if self.subcooling_K < 0.0:
            raise ValueError(f"subcooling_K must be >= 0, but is {self.subcooling_K}.")

        if self.scale_spec_mode not in {"Q_cond", "m_refrigerant"}:
            raise ValueError("scale_spec_mode must be 'Q_cond' or 'm_refrigerant'.")
        if self.scale_spec_mode == "Q_cond":
            if self.Q_cond_kW is None:
                raise ValueError("For scale_spec_mode='Q_cond', Q_cond_kW must be given.")
            if self.m_refrigerant is not None:
                raise ValueError("For scale_spec_mode='Q_cond', m_refrigerant must not be set.")
            if self.Q_cond_kW <= 0.0:
                raise ValueError("Q_cond_kW must be positive.")
        else:
            if self.m_refrigerant is None:
                raise ValueError("For scale_spec_mode='m_refrigerant', m_refrigerant must be given.")
            if self.Q_cond_kW is not None:
                raise ValueError("For scale_spec_mode='m_refrigerant', Q_cond_kW must not be set.")
            if self.m_refrigerant <= 0.0:
                raise ValueError("m_refrigerant must be positive.")

        if self.source_spec_mode not in {"m", "T"}:
            raise ValueError("source_spec_mode must be 'm' or 'T'.")
        if self.source_spec_mode == "m":
            if self.m_source is None:
                raise ValueError("For source_spec_mode='m', m_source must be given.")
            if self.T_source_out_C is not None:
                raise ValueError("For source_spec_mode='m', T_source_out_C must not be set.")
            if self.m_source <= 0.0:
                raise ValueError("m_source must be positive.")
        else:
            if self.T_source_out_C is None:
                raise ValueError("For source_spec_mode='T', T_source_out_C must be given.")
            if self.m_source is not None:
                raise ValueError("For source_spec_mode='T', m_source must not be set.")
            if self.T_source_out_C >= self.T_source_in_C:
                raise ValueError("For source_spec_mode='T', T_source_out_C < T_source_in_C must hold.")

        if self.sink_spec_mode not in {"m", "T"}:
            raise ValueError("sink_spec_mode must be 'm' or 'T'.")
        if self.sink_spec_mode == "m":
            if self.m_sink is None:
                raise ValueError("For sink_spec_mode='m', m_sink must be given.")
            if self.T_sink_out_C is not None:
                raise ValueError("For sink_spec_mode='m', T_sink_out_C must not be set.")
            if self.m_sink <= 0.0:
                raise ValueError("m_sink must be positive.")
        else:
            if self.T_sink_out_C is None:
                raise ValueError("For sink_spec_mode='T', T_sink_out_C must be given.")
            if self.m_sink is not None:
                raise ValueError("For sink_spec_mode='T', m_sink must not be set.")
            if self.T_sink_out_C <= self.T_sink_in_C:
                raise ValueError("For sink_spec_mode='T', T_sink_out_C > T_sink_in_C must hold.")


@dataclass(frozen=True)
class HeatPumpResult:
    inputs: HeatPumpInputs
    converged: bool
    states: Dict[str, Dict[str, float]]
    heat_flows_kW: Dict[str, float]
    kpis: Dict[str, float]
    pinch_temperatures_K: Dict[str, float]


class HeatPumpEvaluationError(RuntimeError):
    """Raised when the TESPy network does not converge to a valid design point."""


# ---------------------------------------------------------------------------
# Network builder (built once per refrigerant, then re-solved for each point)
# ---------------------------------------------------------------------------

class HeatPumpModel:
    """Reusable TESPy network for the water/water heat pump.

    Build once, then call `solve(inputs)` repeatedly (e.g. in a parameter
    sweep) -- this reuses the previous converged state as the initial
    guess for the next solve, which is both faster and more robust than
    rebuilding the network from scratch every time.
    """

    def __init__(self, refrigerant: str = "R134a") -> None:
        self.refrigerant = refrigerant
        self._solved_once = False
        self._build_network()

    def _build_network(self) -> None:
        nw = Network()
        nw.units.set_defaults(
            temperature="degC", pressure="bar", pressure_difference="bar",
            enthalpy="kJ/kg", heat="kW", power="kW", mass_flow="kg/s",
        )

        cc = CycleCloser("cycle closer")
        ev = HeatExchanger("evaporator")
        cp = Compressor("compressor")
        co = HeatExchanger("condenser")
        va = Valve("expansion valve")

        source_in = Source("source in")
        source_out = Sink("source out")
        sink_in = Source("sink in")
        sink_out = Sink("sink out")

        # refrigerant loop
        c0 = Connection(cc, "out1", ev, "in2", label="0")
        c1 = Connection(ev, "out2", cp, "in1", label="1")
        c2 = Connection(cp, "out1", co, "in1", label="2")
        c3 = Connection(co, "out1", va, "in1", label="3")
        c4 = Connection(va, "out1", cc, "in1", label="4")

        # external source loop (evaporator hot side)
        c11 = Connection(source_in, "out1", ev, "in1", label="11")
        c12 = Connection(ev, "out1", source_out, "in1", label="12")

        # external sink loop (condenser cold side)
        c21 = Connection(sink_in, "out1", co, "in2", label="21")
        c22 = Connection(co, "out2", sink_out, "in1", label="22")

        nw.add_conns(c0, c1, c2, c3, c4, c11, c12, c21, c22)

        c0.set_attr(fluid={self.refrigerant: 1})
        # absolute pressure level of the water loops: only the ratio across
        # the heat exchanger is fixed via pr1/pr2, so the loops need one
        # absolute pressure each; 3 bar keeps liquid water comfortably
        # below saturation up to ~130 degC
        c11.set_attr(fluid={"water": 1}, p=3)
        c21.set_attr(fluid={"water": 1}, p=3)

        # refrigerant side: no pressure losses
        ev.set_attr(pr2=1)
        co.set_attr(pr1=1)

        self.nw = nw
        self.components = {"cc": cc, "ev": ev, "cp": cp, "co": co, "va": va}
        self.conns = {
            "0": c0, "1": c1, "2": c2, "3": c3, "4": c4,
            "11": c11, "12": c12, "21": c21, "22": c22,
        }

    def solve(self, inputs: HeatPumpInputs) -> HeatPumpResult:
        if inputs.refrigerant != self.refrigerant:
            raise ValueError(
                f"This HeatPumpModel instance is built for {self.refrigerant!r}, "
                f"but inputs specify {inputs.refrigerant!r}. Create a new instance."
            )

        ev, cp, co = self.components["ev"], self.components["cp"], self.components["co"]
        c0, c1, c2, c3, c4 = (self.conns[k] for k in ("0", "1", "2", "3", "4"))
        c11, c12, c21, c22 = (self.conns[k] for k in ("11", "12", "21", "22"))

        cp.set_attr(eta_s=inputs.eta_s)
        ev.set_attr(pr1=inputs.pr_source, ttd_min=inputs.dT_min_evap)
        co.set_attr(pr2=inputs.pr_sink, ttd_min=inputs.dT_min_cond)

        if inputs.superheat_K > 0.0:
            c1.set_attr(td_dew=inputs.superheat_K, x=None)
        else:
            c1.set_attr(x=1, td_dew=None)

        if inputs.subcooling_K > 0.0:
            c3.set_attr(td_bubble=inputs.subcooling_K, x=None)
        else:
            c3.set_attr(x=0, td_bubble=None)

        if inputs.scale_spec_mode == "Q_cond":
            co.set_attr(Q=-inputs.Q_cond_kW)
            c0.set_attr(m=None)
        else:
            co.set_attr(Q=None)
            c0.set_attr(m=inputs.m_refrigerant)

        c11.set_attr(T=inputs.T_source_in_C)
        if inputs.source_spec_mode == "m":
            c11.set_attr(m=inputs.m_source)
            c12.set_attr(T=None)
        else:
            c11.set_attr(m=None)
            c12.set_attr(T=inputs.T_source_out_C)

        c21.set_attr(T=inputs.T_sink_in_C)
        if inputs.sink_spec_mode == "m":
            c21.set_attr(m=inputs.m_sink)
            c22.set_attr(T=None)
        else:
            c21.set_attr(m=None)
            c22.set_attr(T=inputs.T_sink_out_C)

        self.nw.solve(mode="design", init_only=False)
        converged = self.nw.converged
        self._solved_once = True

        if not converged:
            raise HeatPumpEvaluationError(
                "TESPy network did not converge for the given HeatPumpInputs."
            )

        states = {
            label: {
                "T_C": conn.T.val, "p_bar": conn.p.val, "h_kJ_kg": conn.h.val,
                "m_kg_s": conn.m.val,
            }
            for label, conn in self.conns.items()
        }

        Q_evap = ev.Q.val
        Q_cond = co.Q.val
        P_comp = cp.P.val

        cop = abs(Q_cond) / P_comp if P_comp else float("nan")

        T_source_in = c11.T.val
        T_sink_out = c22.T.val
        T_evap_sat = c0.T.val
        T_cond_sat = c3.T.val

        kpis = {
            "COP": cop,
            "T_lift_external_K": T_sink_out - T_source_in,
            "T_lift_internal_K": T_cond_sat - T_evap_sat,
            "pressure_ratio": c2.p.val / c1.p.val,
        }

        pinch_temperatures_K = {
            "ttd_min_evap": ev.ttd_min.val,
            "ttd_min_cond": co.ttd_min.val,
        }

        return HeatPumpResult(
            inputs=inputs,
            converged=converged,
            states=states,
            heat_flows_kW={"Q_evap": Q_evap, "Q_cond": Q_cond, "P_compressor": P_comp},
            kpis=kpis,
            pinch_temperatures_K=pinch_temperatures_K,
        )


def solve_heat_pump(inputs: HeatPumpInputs, model: Optional[HeatPumpModel] = None) -> HeatPumpResult:
    """Convenience one-shot entry point, analogous to AHT_Pinch_Point.solve_aht().

    For repeated calls (e.g. a parameter sweep), build a HeatPumpModel once
    and call its `.solve()` method directly instead -- this reuses the
    converged state between calls.
    """
    if model is None:
        model = HeatPumpModel(refrigerant=inputs.refrigerant)
    return model.solve(inputs)


if __name__ == "__main__":
    inputs = HeatPumpInputs(
        T_source_in_C=55.0,
        T_sink_in_C=90.0,
        dT_min_evap=5.0,
        dT_min_cond=5.0,
        scale_spec_mode="Q_cond",
        Q_cond_kW=500.0,
        source_spec_mode="T",
        T_source_out_C=45.0,
        sink_spec_mode="T",
        T_sink_out_C=100.0,
    )
    result = solve_heat_pump(inputs)
    print("Converged:", result.converged)
    print("Heat flows [kW]:", result.heat_flows_kW)
    print("KPIs:", result.kpis)
    print("Pinches [K]:", result.pinch_temperatures_K)
