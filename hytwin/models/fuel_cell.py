"""
PEM Fuel Cell Model
===================
Proton Exchange Membrane fuel cell with:
  - Polarisation curve (activation, ohmic, concentration losses)
  - Faradaic efficiency and hydrogen utilisation ratio
  - Thermal model (heat generation, cooling requirement)
  - Partial-load and ramp-rate constraints
  - Degradation due to voltage cycling

Physical references
-------------------
Larminie, J. & Dicks, A. (2003). Fuel Cell Systems Explained. Wiley.
Yousfi-Steiner, N. et al. (2009). J. Power Sources 194(1), 130-145.
US DOE Fuel Cell Technical Targets.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict

import numpy as np

from .base_model import BaseModel, ModelState

# Physical constants
F = 96_485.33          # C/mol
R_GAS = 8.314          # J/(mol·K)
M_H2 = 2.01568e-3      # kg/mol
HHV_H2 = 141.86e6      # J/kg
LHV_H2 = 119.96e6      # J/kg
N_ELECTRONS = 2


class FuelCellModel(BaseModel):
    """
    PEM fuel cell stack model.

    Parameters (``params`` dict)
    ----------------------------
    rated_power_kw : float      — nominal electrical output [kW]
    n_cells : int               — number of cells in stack
    cell_area_cm2 : float       — active area [cm²], default 300
    membrane_resistance_ohm_cm2 : float — ASR [Ω·cm²], default 0.12
    i_exchange_A_cm2 : float    — exchange current density [A/cm²], default 1e-3
    alpha_c : float             — cathodic charge transfer coeff, default 0.4
    temperature_c : float       — operating temperature [°C], default 65
    h2_utilisation : float      — fraction of H2 consumed (stoich~0.8), default 0.8
    min_load_fraction : float   — minimum stable load, default 0.10
    ramp_rate_kw_s : float      — max ramp rate [kW/s]
    degradation_rate : float    — power drop per kh [1/kh], default 2e-5
    """

    def __init__(self, component_id: str, params: Dict[str, Any]) -> None:
        super().__init__(component_id, params)
        self._rated_kw = float(params["rated_power_kw"])
        self._n_cells = int(params.get("n_cells", 80))
        self._cell_area = float(params.get("cell_area_cm2", 300.0))
        self._R_mem = float(params.get("membrane_resistance_ohm_cm2", 0.12))
        self._i0 = float(params.get("i_exchange_A_cm2", 1e-3))
        self._alpha = float(params.get("alpha_c", 0.4))
        self._T_op = float(params.get("temperature_c", 65.0)) + 273.15
        self._h2_util = float(params.get("h2_utilisation", 0.80))
        self._min_load = float(params.get("min_load_fraction", 0.10))
        self._ramp_rate = float(params.get("ramp_rate_kw_s", 1e9))
        self._degrad_rate = float(params.get("degradation_rate", 2e-5))

        # Internal state
        self._power_kw: float = 0.0
        self._h2_consumed_kg: float = 0.0
        self._operating_hours: float = 0.0
        self._energy_produced_kwh: float = 0.0

    # ------------------------------------------------------------------
    # Electrochemistry (fuel cell polarisation curve)
    # ------------------------------------------------------------------

    def _open_circuit_voltage(self) -> float:
        """
        OCV [V] — Nernst equation simplified for H2/O2 at operating T.
        V_oc ≈ 1.229 - 0.9e-3*(T-298) + (RT/4F)*ln(P_H2²*P_O2)
        Assume atmospheric pressures → log term ≈ 0.
        """
        return 1.229 - 9.0e-4 * (self._T_op - 298.15)

    def _activation_loss(self, i_density: float) -> float:
        """Tafel-equation activation loss at cathode [V]."""
        if i_density <= 0:
            return 0.0
        return (R_GAS * self._T_op) / (self._alpha * F) * math.log(i_density / self._i0)

    def _ohmic_loss(self, i_density: float) -> float:
        return i_density * self._R_mem

    def _concentration_loss(self, i_density: float) -> float:
        i_lim = 1.8  # A/cm²
        if i_density >= i_lim:
            return 0.4
        if i_density <= 0:
            return 0.0
        return -(R_GAS * self._T_op) / (4 * F) * math.log(max(1e-9, 1.0 - i_density / i_lim))

    def _cell_voltage(self, i_density: float) -> float:
        V_oc = self._open_circuit_voltage()
        return max(0.0, V_oc - self._activation_loss(i_density)
                   - self._ohmic_loss(i_density)
                   - self._concentration_loss(i_density))

    # ------------------------------------------------------------------
    # Power setpoint → H2 consumption
    # ------------------------------------------------------------------

    def _power_to_h2_demand(self, p_ac_kw: float, dt: float) -> tuple[float, float]:
        """
        Given demanded output power, compute H2 consumed [kg] and actual
        electrical power delivered [kW].
        Returns (h2_kg, actual_power_kw).
        """
        if p_ac_kw <= 0:
            return 0.0, 0.0

        P_W = p_ac_kw * 1000.0
        V_oc = self._open_circuit_voltage()
        I_est = P_W / (self._n_cells * (V_oc - 0.3))
        i_est = max(1e-6, I_est / self._cell_area)

        for _ in range(25):
            V_cell = self._cell_voltage(i_est)
            I = i_est * self._cell_area
            P_calc = self._n_cells * V_cell * I
            err = P_calc - P_W
            dV_di = -(self._R_mem + (R_GAS * self._T_op) / (self._alpha * F * (i_est + 1e-12)))
            dP_di = self._n_cells * self._cell_area * (V_cell + i_est * dV_di)
            if abs(dP_di) < 1e-12:
                break
            i_est = max(1e-6, i_est - err / dP_di)
            if abs(err) / (P_W + 1e-9) < 1e-6:
                break

        I_final = i_est * self._cell_area
        V_final = self._cell_voltage(i_est)
        actual_p_kw = self._n_cells * V_final * I_final / 1000.0

        degrad = max(0.0, 1.0 - self._degrad_rate * self._operating_hours / 1000)
        actual_p_kw *= degrad

        # H2 consumed: Faraday's law + utilisation
        h2_kg_s = (I_final * self._n_cells) / (N_ELECTRONS * F) * M_H2 / self._h2_util
        h2_kg = h2_kg_s * dt

        return h2_kg, actual_p_kw

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        context keys:
          power_setpoint_kw : float — requested AC output
          h2_available_kg   : float — H2 available from tank
        """
        setpoint = float(context.get("power_setpoint_kw", 0.0))
        h2_avail = float(context.get("h2_available_kg", 1e9))
        ts = context.get("timestamp", datetime.utcnow())
        # context["temperature_c"] is the AIR temperature from the weather: the stack runs at the
        # regulated operating temperature of its parameters (docs/02_physical_models.md)

        # Ramp rate
        delta_max = self._ramp_rate * dt
        new_power = float(np.clip(
            setpoint,
            self._power_kw - delta_max,
            self._power_kw + delta_max,
        ))
        new_power = float(np.clip(new_power, 0.0, self._rated_kw))
        if 0 < new_power < self._min_load * self._rated_kw:
            new_power = 0.0

        h2_demand, actual_kw = self._power_to_h2_demand(new_power, dt)

        # Cannot exceed H2 availability
        if h2_demand > h2_avail:
            ratio = h2_avail / (h2_demand + 1e-12)
            h2_demand = h2_avail
            actual_kw *= ratio

        self._power_kw = actual_kw
        self._h2_consumed_kg += h2_demand
        self._energy_produced_kwh += actual_kw * dt / 3600.0
        if actual_kw > 0:
            self._operating_hours += dt / 3600.0

        # Efficiency (LHV)
        p_chem_kw = h2_demand * LHV_H2 / (dt + 1e-12) / 1000.0
        eta_lhv = actual_kw / (p_chem_kw + 1e-12) if p_chem_kw > 0 else 0.0

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "power_kw": actual_kw,
                "h2_consumed_kg_step": h2_demand,
                "h2_consumed_kg_total": self._h2_consumed_kg,
                "efficiency_lhv": min(1.0, eta_lhv),
                "energy_produced_kwh": self._energy_produced_kwh,
                "h2_flow_kg_h": h2_demand * 3600.0 / dt if dt > 0 else 0.0,
                "operating_hours": self._operating_hours,
            },
        )
        return self._state

    def reset(self) -> None:
        self._power_kw = 0.0
        self._h2_consumed_kg = 0.0
        self._operating_hours = 0.0
        self._energy_produced_kwh = 0.0
        self._state = None
