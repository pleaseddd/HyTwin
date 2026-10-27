"""
test_models.py
==============
Unit tests for all HyTwin physics models.
Run: pytest tests/test_models.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hytwin.models.wind_turbine import WindTurbineModel
from hytwin.models.photovoltaic import PhotovoltaicModel
from hytwin.models.electrolyzer import ElectrolyzerModel
from hytwin.models.fuel_cell import FuelCellModel
from hytwin.models.hydrogen_tank import HydrogenTankModel
from hytwin.models.energy_load import EnergyLoadModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def wind_params():
    return dict(
        rotor_diameter_m=77.0,
        hub_height_m=80.0,
        rated_power_kw=500.0,
        v_cut_in=3.0,
        v_rated=12.0,
        v_cut_out=25.0,
        efficiency_gen=0.95,
        altitude_m=50.0,
        turbulence_intensity=0.0,   # deterministic
    )


@pytest.fixture
def pv_params():
    return dict(
        n_panels=100,
        panel_area_m2=1.96,
        eta_stc=0.20,
        temp_coeff_pmax=-0.004,
        noct_c=45.0,
        rated_power_kw=40.0,
        soiling_loss=0.0,
        degradation_per_year=0.0,
        tilt_deg=0.0,       # horizontal
        azimuth_deg=180.0,
    )


@pytest.fixture
def el_params():
    return dict(
        rated_power_kw=100.0,
        n_cells=50,
        cell_area_cm2=300.0,
        membrane_resistance_ohm_cm2=0.16,
        temperature_c=65.0,
        min_load_fraction=0.05,
        ramp_rate_kw_s=100.0,   # no ramp limiting in tests
    )


@pytest.fixture
def fc_params():
    # 420 cells × 200 cm² @ ~0.6 V / 1 A·cm⁻² → ~50 kW deliverable
    return dict(
        rated_power_kw=50.0,
        n_cells=420,
        cell_area_cm2=200.0,
        membrane_resistance_ohm_cm2=0.12,
        temperature_c=65.0,
        h2_utilisation=0.80,
        min_load_fraction=0.10,
        ramp_rate_kw_s=100.0,
    )


@pytest.fixture
def tank_params():
    return dict(
        volume_m3=5.0,
        max_pressure_bar=700.0,
        min_pressure_bar=10.0,
        initial_soc=0.50,
        temperature_c=20.0,
        max_charge_rate_kg_s=0.10,
        max_discharge_rate_kg_s=0.10,
        boiloff_rate_per_day=0.0,
    )


# ===========================================================================
# Wind turbine
# ===========================================================================
class TestWindTurbineModel:

    def test_no_power_below_cut_in(self, wind_params):
        wt = WindTurbineModel("wt", wind_params)
        state = wt.step(600.0, context={"wind_speed_ms": 0.0})
        assert state.values["power_kw"] == pytest.approx(0.0)

    def test_no_power_above_cut_out(self, wind_params):
        wt = WindTurbineModel("wt", wind_params)
        state = wt.step(600.0, context={"wind_speed_ms": 30.0})
        assert state.values["power_kw"] == pytest.approx(0.0)
        assert not state.values["available"]

    def test_rated_power_at_v_rated(self, wind_params):
        wt = WindTurbineModel("wt", wind_params)
        state = wt.step(600.0, context={"wind_speed_ms": 12.0})
        assert state.values["power_kw"] == pytest.approx(wind_params["rated_power_kw"], rel=0.05)

    def test_power_increases_with_wind(self, wind_params):
        wt = WindTurbineModel("wt", wind_params)
        p4 = wt.step(600.0, context={"wind_speed_ms": 4.0}).values["power_kw"]
        wt.reset()
        p8 = wt.step(600.0, context={"wind_speed_ms": 8.0}).values["power_kw"]
        assert p8 > p4 > 0.0

    def test_energy_accumulates(self, wind_params):
        wt = WindTurbineModel("wt", wind_params)
        for _ in range(10):
            wt.step(600.0, context={"wind_speed_ms": 8.0})
        assert wt.state.values["energy_kwh_total"] > 0.0

    def test_reset_clears_state(self, wind_params):
        wt = WindTurbineModel("wt", wind_params)
        wt.step(600.0, context={"wind_speed_ms": 8.0})
        wt.reset()
        # After reset state may be None or have zeroed energy
        if wt.state is not None:
            assert wt.state.values.get("energy_kwh_total", 0.0) == pytest.approx(0.0)


# ===========================================================================
# Photovoltaic
# ===========================================================================
class TestPhotovoltaicModel:

    def test_no_power_at_night(self, pv_params):
        pv = PhotovoltaicModel("pv", pv_params)
        ctx = {"ghi_wm2": 0.0, "temperature_c": 20.0,
               "sun_zenith_deg": 95.0, "sun_azimuth_deg": 180.0,
               "dni_wm2": 0.0, "dhi_wm2": 0.0}
        state = pv.step(600.0, context=ctx)
        assert state.values["power_ac_kw"] == pytest.approx(0.0, abs=0.01)

    def test_power_positive_at_noon(self, pv_params):
        pv = PhotovoltaicModel("pv", pv_params)
        ctx = {"ghi_wm2": 900.0, "temperature_c": 25.0,
               "sun_zenith_deg": 20.0, "sun_azimuth_deg": 180.0,
               "dni_wm2": 800.0, "dhi_wm2": 100.0}
        state = pv.step(600.0, context=ctx)
        assert state.values["power_ac_kw"] > 0.0

    def test_power_bounded_by_rated(self, pv_params):
        pv = PhotovoltaicModel("pv", pv_params)
        ctx = {"ghi_wm2": 1200.0, "temperature_c": 15.0,
               "sun_zenith_deg": 10.0, "sun_azimuth_deg": 180.0,
               "dni_wm2": 1100.0, "dhi_wm2": 100.0}
        state = pv.step(600.0, context=ctx)
        assert state.values["power_ac_kw"] <= pv_params["rated_power_kw"] * 1.05

    def test_high_temp_reduces_power(self, pv_params):
        pv_hot  = PhotovoltaicModel("pv_hot",  pv_params)
        pv_cold = PhotovoltaicModel("pv_cold", pv_params)
        ctx_hot  = {"ghi_wm2": 800.0, "temperature_c": 40.0, "sun_zenith_deg": 30.0,
                    "sun_azimuth_deg": 180.0, "dni_wm2": 700.0, "dhi_wm2": 100.0}
        ctx_cold = {**ctx_hot, "temperature_c": 10.0}
        p_hot  = pv_hot.step(600.0, context=ctx_hot).values["power_ac_kw"]
        p_cold = pv_cold.step(600.0, context=ctx_cold).values["power_ac_kw"]
        assert p_cold > p_hot


# ===========================================================================
# Electrolyzer
# ===========================================================================
class TestElectrolyzerModel:

    def test_no_h2_when_off(self, el_params):
        el = ElectrolyzerModel("el", el_params)
        state = el.step(600.0, context={"power_setpoint_kw": 0.0})
        assert state.values["h2_kg_step"] == pytest.approx(0.0, abs=1e-6)

    def test_h2_production_positive_when_on(self, el_params):
        el = ElectrolyzerModel("el", el_params)
        state = el.step(600.0, context={"power_setpoint_kw": el_params["rated_power_kw"]})
        assert state.values["h2_kg_step"] > 0.0

    def test_faraday_law_rough_check(self, el_params):
        """At 100% load for 1 h (3600 s), verify H2 production is in physical range."""
        el = ElectrolyzerModel("el", el_params)
        total_h2 = 0.0
        for _ in range(6):      # 6 × 600 s = 3600 s = 1 h
            s = el.step(600.0, context={"power_setpoint_kw": el_params["rated_power_kw"]})
            total_h2 += s.values["h2_kg_step"]
        # Rough Faraday bound: 33 kWh/kg HHV → 100 kW × 1 h / 33 kWh/kg ≈ 3 kg
        # Model uses LHV efficiency (~50–60%) so actual yield ≈ 0.45–0.65 × HHV bound
        expected_kg = el_params["rated_power_kw"] * 1.0 / 33.0   # HHV approx
        assert 0.3 * expected_kg < total_h2 < 2.0 * expected_kg

    def test_efficiency_reasonable(self, el_params):
        el = ElectrolyzerModel("el", el_params)
        state = el.step(600.0, context={"power_setpoint_kw": el_params["rated_power_kw"]})
        eff = state.values["efficiency_hhv"]
        assert 0.4 < eff < 0.9, f"Efficiency out of range: {eff}"

    def test_power_respects_rated(self, el_params):
        el = ElectrolyzerModel("el", el_params)
        state = el.step(600.0, context={"power_setpoint_kw": 1_000_000.0})
        assert state.values["power_kw"] <= el_params["rated_power_kw"] * 1.01

    def test_min_load_clamps(self, el_params):
        el = ElectrolyzerModel("el", el_params)
        min_kw = el_params["rated_power_kw"] * el_params["min_load_fraction"]
        # Asking slightly below min-load → either full off or at min load
        state = el.step(600.0, context={"power_setpoint_kw": min_kw * 0.3})
        assert state.values["power_kw"] == pytest.approx(0.0, abs=0.5) or \
               state.values["power_kw"] >= min_kw - 0.1


# ===========================================================================
# Fuel cell
# ===========================================================================
class TestFuelCellModel:

    def test_no_output_when_off(self, fc_params):
        fc = FuelCellModel("fc", fc_params)
        state = fc.step(600.0, context={"power_setpoint_kw": 0.0,
                                         "h2_available_kg": 100.0})
        assert state.values["power_kw"] == pytest.approx(0.0, abs=0.01)

    def test_power_at_rated_load(self, fc_params):
        fc = FuelCellModel("fc", fc_params)
        state = fc.step(600.0, context={"power_setpoint_kw": fc_params["rated_power_kw"],
                                         "h2_available_kg": 100.0})
        assert state.values["power_kw"] > 0.0

    def test_h2_consumed_positive(self, fc_params):
        fc = FuelCellModel("fc", fc_params)
        state = fc.step(600.0, context={"power_setpoint_kw": fc_params["rated_power_kw"],
                                         "h2_available_kg": 100.0})
        assert state.values["h2_consumed_kg_step"] > 0.0

    def test_no_h2_no_power(self, fc_params):
        fc = FuelCellModel("fc", fc_params)
        state = fc.step(600.0, context={"power_setpoint_kw": fc_params["rated_power_kw"],
                                         "h2_available_kg": 0.0})
        assert state.values["power_kw"] == pytest.approx(0.0, abs=0.01)

    def test_efficiency_reasonable(self, fc_params):
        fc = FuelCellModel("fc", fc_params)
        state = fc.step(600.0, context={"power_setpoint_kw": fc_params["rated_power_kw"] * 0.5,
                                         "h2_available_kg": 100.0})
        eff = state.values.get("efficiency_lhv", 0.0)
        assert 0.3 < eff < 0.85, f"Efficiency out of range: {eff}"


# ===========================================================================
# Hydrogen tank
# ===========================================================================
class TestHydrogenTankModel:

    def test_initial_soc(self, tank_params):
        tk = HydrogenTankModel("tk", tank_params)
        # Trigger first step to populate state
        state = tk.step(0.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 0.0})
        assert state.values["soc"] == pytest.approx(
            tank_params["initial_soc"], rel=0.10)

    def test_filling_increases_soc(self, tank_params):
        tk = HydrogenTankModel("tk", tank_params)
        s0 = tk.step(0.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 0.0})
        soc0 = s0.values["soc"]
        s1 = tk.step(600.0, context={"h2_charge_kg": 5.0, "h2_discharge_kg": 0.0})
        assert s1.values["soc"] > soc0

    def test_discharging_decreases_soc(self, tank_params):
        tk = HydrogenTankModel("tk", tank_params)
        s0 = tk.step(0.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 0.0})
        soc0 = s0.values["soc"]
        s1 = tk.step(600.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 5.0})
        assert s1.values["soc"] < soc0

    def test_pressure_bounded(self, tank_params):
        tk = HydrogenTankModel("tk", tank_params)
        for _ in range(20):
            s = tk.step(600.0, context={"h2_charge_kg": 50.0, "h2_discharge_kg": 0.0})
        assert s.values["pressure_bar"] <= tank_params["max_pressure_bar"] + 1.0
        assert s.values["soc"] <= 1.0

    def test_soc_bounded_below(self, tank_params):
        tk = HydrogenTankModel("tk", tank_params)
        for _ in range(50):
            s = tk.step(600.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 50.0})
        assert s.values["soc"] >= 0.0

    def test_vdw_pressure_physical(self, tank_params):
        """Pressure at ~50% SOC should be physically reasonable."""
        tk = HydrogenTankModel("tk", tank_params)
        s = tk.step(0.0, context={"h2_charge_kg": 0.0, "h2_discharge_kg": 0.0})
        p = s.values["pressure_bar"]
        assert 0 < p < tank_params["max_pressure_bar"]


# ===========================================================================
# Energy load
# ===========================================================================
class TestEnergyLoadModel:

    def _make_load(self, profile="residential"):
        return EnergyLoadModel("load", {
            "base_load_kw": 100.0,
            "profile_type": profile,
            "noise_std_fraction": 0.0,   # deterministic
            "seasonal_amplitude": 0.0,
            "demand_response_factor": 0.20,
        })

    def test_load_positive(self):
        from datetime import datetime
        load = self._make_load()
        ctx = {"timestamp": datetime(2024, 6, 15, 12, 0), "demand_response": 0.0}
        state = load.step(600.0, context=ctx)
        assert state.values["load_kw"] > 0.0

    def test_demand_response_reduces_load(self):
        from datetime import datetime
        load = self._make_load()
        ctx_normal = {"timestamp": datetime(2024, 6, 15, 18, 0), "demand_response": 0.0}
        ctx_dr     = {"timestamp": datetime(2024, 6, 15, 18, 0), "demand_response": 1.0}
        p_normal = load.step(600.0, context=ctx_normal).values["load_kw"]
        load.reset()
        p_dr = load.step(600.0, context=ctx_dr).values["load_kw"]
        assert p_dr < p_normal

    def test_profile_types_run(self):
        from datetime import datetime
        for profile in ("residential", "industrial", "commercial"):
            load = EnergyLoadModel("load", {
                "base_load_kw": 200.0,
                "profile_type": profile,
                "noise_std_fraction": 0.0,
                "seasonal_amplitude": 0.0,
                "demand_response_factor": 0.0,
            })
            ctx = {"timestamp": datetime(2024, 6, 15, 10, 0), "demand_response": 0.0}
            state = load.step(600.0, context=ctx)
            assert state.values["load_kw"] > 0.0, f"Profile {profile!r} returned zero load"

    def _load_at(self, profile, ts, **overrides):
        params = {"base_load_kw": 1000.0, "profile_type": profile, "noise_std_fraction": 0.0}
        params.update(overrides)
        m = EnergyLoadModel("load", params)
        return m.step(600.0, context={"timestamp": ts}).values["load_kw"]

    def test_industrial_drops_sharply_on_sunday(self):
        from datetime import datetime
        weekday = self._load_at("industrial", datetime(2024, 6, 17, 12, 0))  # Monday
        sunday = self._load_at("industrial", datetime(2024, 6, 23, 12, 0))
        assert sunday < 0.5 * weekday

    def test_commercial_drops_moderately_on_sunday(self):
        from datetime import datetime
        weekday = self._load_at("commercial", datetime(2024, 6, 17, 12, 0))
        sunday = self._load_at("commercial", datetime(2024, 6, 23, 12, 0))
        assert sunday < weekday  # reduced, but not as drastically as industrial

    def test_residential_only_mildly_affected_by_weekend(self):
        from datetime import datetime
        weekday = self._load_at("residential", datetime(2024, 6, 17, 12, 0))
        sunday = self._load_at("residential", datetime(2024, 6, 23, 12, 0))
        assert 0.7 * weekday < sunday <= weekday

    def test_national_holiday_treated_as_sunday_for_industrial(self):
        """Ferragosto 2024 (a Thursday) must depress industrial load like a Sunday,
        independent of the August-shutdown window (isolated here by disabling it)."""
        from datetime import datetime
        thursday_normal = self._load_at("industrial", datetime(2024, 6, 13, 12, 0), august_shutdown=False)
        ferragosto = self._load_at("industrial", datetime(2024, 8, 15, 12, 0), august_shutdown=False)
        assert ferragosto < 0.5 * thursday_normal

    def test_august_shutdown_depresses_industrial_output(self):
        from datetime import datetime
        june = self._load_at("industrial", datetime(2024, 6, 17, 12, 0))
        august = self._load_at("industrial", datetime(2024, 8, 6, 12, 0))  # a Tuesday, inside the shutdown window
        assert august < 0.5 * june

    def test_august_shutdown_disabled_by_default_for_non_industrial(self):
        from datetime import datetime
        june = self._load_at("commercial", datetime(2024, 6, 18, 12, 0))  # Tuesday
        august = self._load_at("commercial", datetime(2024, 8, 6, 12, 0))  # Tuesday
        # No shutdown factor for commercial — only the (smaller) seasonal/AC swing applies.
        assert august > 0.6 * june

    def test_weekday_sensitive_flag_disables_calendar_effects(self):
        """On the SAME calendar date (so seasonal/diurnal terms are identical),
        disabling weekday_sensitive must remove the Sunday reduction entirely."""
        from datetime import datetime
        sunday = datetime(2024, 6, 23, 12, 0)
        reduced = self._load_at("industrial", sunday, weekday_sensitive=True)
        full = self._load_at("industrial", sunday, weekday_sensitive=False)
        assert reduced < full
