"""
PEM Electrolyzer Model
======================
Proton Exchange Membrane electrolyzer with:
  - Faradaic efficiency (current efficiency)
  - Butler-Volmer activation overpotential
  - Ohmic (membrane) overpotential
  - Concentration overpotential
  - Thermal management (temperature rise / cooling power)
  - Degradation / ageing effects
  - Partial-load operation and ramp-rate limits

Physical references
-------------------
Olivier, P. et al. (2017). Int. J. Hydrogen Energy 42(3), 1609-1625.
Awasthi, A. et al. (2011). Int. J. Hydrogen Energy 36(22), 14779-14786.
IRENA (2020). Green Hydrogen Cost Reduction.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict

import numpy as np

from .base_model import BaseModel, ModelState

# Physical constants
F = 96_485.33          # C/mol  — Faraday constant
R_GAS = 8.314          # J/(mol·K) — universal gas constant
M_H2 = 2.01568e-3      # kg/mol — molar mass H2
HHV_H2 = 141.86e6      # J/kg   — higher heating value H2
LHV_H2 = 119.96e6      # J/kg   — lower heating value H2
N_ELECTRONS = 2        # electrons per H2 molecule in electrolysis


class ElectrolyzerModel(BaseModel):
    """
    PEM electrolyzer stack model.

    Parameters (``params`` dict)
    ----------------------------
    rated_power_kw : float      — nominal DC input [kW]
    n_cells : int               — number of electrolysis cells
    cell_area_cm2 : float       — active electrode area [cm²], default 300
    membrane_resistance_ohm_cm2 : float — ASR [Ω·cm²], default 0.17
    i_exchange_A_cm2 : float    — exchange current density [A/cm²], default 1e-3
    alpha_a : float             — anodic charge transfer coeff, default 0.5
    temperature_c : float       — operating temperature [°C], default 60
    min_load_fraction : float   — minimum stable load (0-1), default 0.05
    ramp_rate_kw_s : float      — max ramp rate [kW/s], default no limit
    degradation_rate : float    — efficiency drop per kh [1/kh], default 1e-5
    water_flow_l_h_kw : float   — water consumption [L/h per kW], default 0.9
    """

    def __init__(self, component_id: str, params: Dict[str, Any]) -> None:
        super().__init__(component_id, params)
        self._rated_kw = float(params["rated_power_kw"])
        self._n_cells = int(params.get("n_cells", 100))
        self._cell_area = float(params.get("cell_area_cm2", 300.0))       # cm²
        self._R_mem = float(params.get("membrane_resistance_ohm_cm2", 0.17))
        self._i0 = float(params.get("i_exchange_A_cm2", 1e-3))
        self._alpha = float(params.get("alpha_a", 0.5))
        self._T_op = float(params.get("temperature_c", 60.0)) + 273.15    # K
        self._min_load = float(params.get("min_load_fraction", 0.05))
        self._ramp_rate = float(params.get("ramp_rate_kw_s", 1e9))
        self._degrad_rate = float(params.get("degradation_rate", 1e-5))
        self._water_rate = float(params.get("water_flow_l_h_kw", 0.9))

        # Internal state
        self._power_kw: float = 0.0
        self._h2_produced_kg: float = 0.0
        self._operating_hours: float = 0.0
        self._energy_consumed_kwh: float = 0.0

    # ------------------------------------------------------------------
    # Electrochemistry
    # ------------------------------------------------------------------

    def _reversible_voltage(self) -> float:
        """
        Nernst / reversible cell voltage [V] at operating temperature.
        V_rev = 1.229 - 9.0e-4 * (T - 298) (approximation)
        """
        return 1.229 - 9.0e-4 * (self._T_op - 298.15)

    def _activation_overpotential(self, i_density: float) -> float:
        """Butler-Volmer activation overpotential [V]."""
        if i_density <= 0:
            return 0.0
        # Simplified Butler-Volmer: η_act = (RT/αF) * ln(i/i0)
        V_act = (R_GAS * self._T_op) / (self._alpha * F) * math.log(i_density / self._i0)
        return max(0.0, V_act)

    def _ohmic_overpotential(self, i_density: float) -> float:
        """Ohmic overpotential through membrane [V]."""
        return i_density * self._R_mem  # i [A/cm²] * R [Ω·cm²] = V

    def _concentration_overpotential(self, i_density: float) -> float:
        """Mass-transport concentration overpotential [V] — simplified."""
        i_lim = 2.0  # limiting current density A/cm²
        if i_density >= i_lim:
            return 0.5
        if i_density <= 0:
            return 0.0
        factor = max(1e-6, 1.0 - i_density / i_lim)
        return -(R_GAS * self._T_op) / (2 * F) * math.log(factor)

    def _cell_voltage(self, i_density: float) -> float:
        """Total cell voltage [V]."""
        V_rev = self._reversible_voltage()
        V_act = self._activation_overpotential(i_density)
        V_ohm = self._ohmic_overpotential(i_density)
        V_conc = self._concentration_overpotential(i_density)
        return V_rev + V_act + V_ohm + V_conc

    def _faradaic_efficiency(self, i_density: float) -> float:
        """
        Faradaic efficiency: fraction of charge producing H2 (vs parasitic).
        η_F ≈ 1 - exp(-a * i + b)  empirical fit.
        """
        # Empirical: at high current almost 100 %, drops at low current
        a, b = 4.0, 0.5
        return min(1.0, max(0.0, 1.0 - math.exp(-a * i_density + b)))

    # ------------------------------------------------------------------
    # Power → H2 production
    # ------------------------------------------------------------------

    def _power_to_h2(self, p_dc_kw: float, dt: float) -> float:
        """
        Convert DC input power to H2 mass produced [kg] in time step dt [s].

        Strategy: iterate to find current density consistent with
        voltage × current × n_cells = p_dc_kw.
        """
        if p_dc_kw <= 0:
            return 0.0

        P_W = p_dc_kw * 1000.0
        # Stack current: P = N * V_cell * I → estimate via Ohmic model first
        # I_total = P / (N_cells * V_rev)  as starting point
        V_rev = self._reversible_voltage()
        I_est = P_W / (self._n_cells * (V_rev + 0.3))  # rough init [A]
        i_est = I_est / self._cell_area  # A/cm²

        # Newton iterations (converge quickly)
        for _ in range(25):
            V_cell = self._cell_voltage(i_est)
            I = i_est * self._cell_area
            P_calc = self._n_cells * V_cell * I
            error = P_calc - P_W
            # dP/di ≈ n_cells * (V_cell + i * dV/di) * A
            dV_di = self._R_mem + (R_GAS * self._T_op) / (self._alpha * F * i_est + 1e-12)
            dP_di = self._n_cells * self._cell_area * (V_cell + i_est * dV_di)
            if abs(dP_di) < 1e-12:
                break
            i_new = i_est - error / dP_di
            i_est = max(1e-6, i_new)
            if abs(error / (P_W + 1e-9)) < 1e-6:
                break

        I_final = i_est * self._cell_area  # A per cell
        eta_F = self._faradaic_efficiency(i_est)
        # Degradation factor
        degrad = max(0.0, 1.0 - self._degrad_rate * self._operating_hours / 1000)

        # Faraday: ṁ_H2 [kg/s] = (I * N_cells) / (N_e * F) * M_H2 * η_F
        h2_rate_kg_s = (I_final * self._n_cells) / (N_ELECTRONS * F) * M_H2 * eta_F * degrad
        return h2_rate_kg_s * dt  # kg in this step

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def step(self, dt: float, context: Dict[str, Any]) -> ModelState:
        """
        Parameters
        ----------
        context keys:
          power_setpoint_kw : float — power setpoint from controller
          temperature_c : float     — optional temperature override
        """
        setpoint = float(context.get("power_setpoint_kw", 0.0))
        if "temperature_c" in context:
            self._T_op = float(context["temperature_c"]) + 273.15
        ts = context.get("timestamp", datetime.utcnow())

        # Ramp-rate limiting
        delta_max = self._ramp_rate * dt
        new_power = np.clip(setpoint, self._power_kw - delta_max, self._power_kw + delta_max)
        new_power = float(np.clip(new_power, 0.0, self._rated_kw))

        # Minimum load threshold
        if 0 < new_power < self._min_load * self._rated_kw:
            new_power = 0.0

        h2_kg = self._power_to_h2(new_power, dt)

        # Stack efficiency: HHV-based
        e_input_j = new_power * 1000.0 * dt  # J
        if e_input_j > 0 and h2_kg > 0:
            eta_hhv = h2_kg * HHV_H2 / e_input_j  # dimensionless
        else:
            eta_hhv = 0.0

        self._power_kw = new_power
        self._h2_produced_kg += h2_kg
        self._energy_consumed_kwh += new_power * dt / 3600.0
        if new_power > 0:
            self._operating_hours += dt / 3600.0

        water_consumed_l = self._water_rate * new_power * dt / 3600.0

        self._state = ModelState(
            timestamp=ts,
            component_id=self.component_id,
            values={
                "power_kw": new_power,
                "h2_kg_step": h2_kg,
                "h2_kg_total": self._h2_produced_kg,
                "efficiency_hhv": eta_hhv,
                "energy_consumed_kwh": self._energy_consumed_kwh,
                "h2_flow_kg_h": h2_kg * 3600.0 / dt if dt > 0 else 0.0,
                "water_consumed_l": water_consumed_l,
                "operating_hours": self._operating_hours,
            },
        )
        return self._state

    def reset(self) -> None:
        self._power_kw = 0.0
        self._h2_produced_kg = 0.0
        self._operating_hours = 0.0
        self._energy_consumed_kwh = 0.0
        self._state = None
