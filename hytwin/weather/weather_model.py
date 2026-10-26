"""
Weather Model
=============
Stochastic weather generator for simulation, producing:
  · Wind speed and direction (Weibull-based, autocorrelated)
  · Solar irradiance (GHI / DHI / DNI) using clear-sky + cloud perturbation
  · Temperature (AR(1) process with seasonal trend)
  · Relative humidity

The model can operate in two modes:
  1. **Synthetic** (default) — generates physically plausible random weather
  2. **Replay** — replays a pre-loaded Pandas DataFrame of historical data

Physical references
-------------------
Manwell, J.F. et al. (2009). Wind Energy Explained. Wiley.
Ineichen, P. & Perez, R. (2002). Solar Energy 73(3).
Holton, J.R. (2004). Introduction to Dynamic Meteorology. Elsevier.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

from ..core.rng import resolve_rng, spawn_one


def _norm_cdf(x: float) -> float:
    """Standard normal CDF Φ(x), via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------
# Solar geometry helpers
# ------------------------------------------------------------------

def _solar_declination(day_of_year: int) -> float:
    """Solar declination δ [degrees] — Spencer (1971)."""
    B = 2 * math.pi * (day_of_year - 1) / 365.0
    return (
        180.0 / math.pi
        * (
            0.006918
            - 0.399912 * math.cos(B)
            + 0.070257 * math.sin(B)
            - 0.006758 * math.cos(2 * B)
            + 0.000907 * math.sin(2 * B)
            - 0.002697 * math.cos(3 * B)
            + 0.00148 * math.sin(3 * B)
        )
    )


def _equation_of_time(day_of_year: int) -> float:
    """Equation of time [minutes] — Wooff & Iqbal."""
    B = 2 * math.pi * (day_of_year - 1) / 365.0
    return (
        229.18
        * (
            0.000075
            + 0.001868 * math.cos(B)
            - 0.032077 * math.sin(B)
            - 0.014615 * math.cos(2 * B)
            - 0.04089 * math.sin(2 * B)
        )
    )


def solar_position(dt: datetime, latitude_deg: float, longitude_deg: float) -> Dict[str, float]:
    """
    Compute solar zenith and azimuth angles.

    Returns dict with keys: zenith_deg, azimuth_deg, elevation_deg, hour_angle_deg.
    """
    doy = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    decl = _solar_declination(doy)
    eot = _equation_of_time(doy)

    # Local Standard Time Meridian (15° per hour)
    lstm = round(longitude_deg / 15.0) * 15.0
    tc = 4 * (longitude_deg - lstm) + eot   # minutes
    local_solar_time = hour_utc + tc / 60.0
    hour_angle = 15.0 * (local_solar_time - 12.0)  # degrees

    lat_r = math.radians(latitude_deg)
    decl_r = math.radians(decl)
    ha_r = math.radians(hour_angle)

    cos_z = (
        math.sin(lat_r) * math.sin(decl_r)
        + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r)
    )
    cos_z = max(-1.0, min(1.0, cos_z))
    zenith_deg = math.degrees(math.acos(cos_z))
    elevation_deg = 90.0 - zenith_deg

    # Azimuth (from North, clockwise)
    sin_z = math.sin(math.radians(zenith_deg))
    if sin_z < 1e-9:
        azimuth_deg = 0.0
    else:
        cos_az = (
            (math.sin(lat_r) * cos_z - math.sin(decl_r))
            / (math.cos(lat_r) * sin_z + 1e-12)
        )
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth_deg = math.degrees(math.acos(cos_az))
        if hour_angle > 0:
            azimuth_deg = 360.0 - azimuth_deg

    return {
        "zenith_deg": zenith_deg,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "hour_angle_deg": hour_angle,
        "declination_deg": decl,
    }


def _clear_sky_ghi(zenith_deg: float, altitude_m: float = 0.0) -> float:
    """
    Simplified Ineichen clear-sky GHI [W/m²].
    Uses a simplified Linke turbidity approach.
    """
    if zenith_deg >= 90.0:
        return 0.0
    cos_z = math.cos(math.radians(zenith_deg))
    # Extra-terrestrial irradiance (average)
    Gsc = 1361.0  # W/m²
    altitude_factor = math.exp(-altitude_m / 8500.0)   # atmospheric thickness
    TL = 2.5   # Linke turbidity (clear sky ~2-3)
    AM = 1.0 / (cos_z + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364))  # Kasten
    tau = math.exp(-0.8662 * TL * AM * altitude_factor * 0.0296)
    ghi_clear = Gsc * cos_z * tau
    return max(0.0, ghi_clear)


# ------------------------------------------------------------------
# Stochastic wind model
# ------------------------------------------------------------------

class WindModel:
    """
    Autocorrelated Weibull wind speed model.

    Parameters
    ----------
    weibull_k : float      — Weibull shape parameter (typical 1.5-2.5)
    weibull_c : float      — Weibull scale parameter [m/s] (related to mean)
    autocorr : float       — lag-1 autocorrelation coefficient (0.85-0.97)
    wind_ref_height_m : float — reference anemometer height
    """

    def __init__(
        self,
        weibull_k: float = 2.0,
        weibull_c: float = 7.0,
        autocorr: float = 0.92,
        wind_ref_height_m: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._k = weibull_k
        self._c = weibull_c
        self._autocorr = autocorr
        self._ref_h = wind_ref_height_m
        self._rng = resolve_rng(rng)
        # Latent standard-normal AR(1) state ("Gaussian copula"): x is kept
        # N(0,1)-distributed by construction, and each step's wind speed is
        # obtained by mapping x through the normal CDF then the *inverse*
        # Weibull(k,c) CDF. This is what makes the long-run marginal
        # distribution of wind_speed_ms actually equal Weibull(k,c) — i.e.
        # its long-run mean equals c*Gamma(1+1/k), as configured — while
        # still producing a smooth, autocorrelated time series.
        self._x: float = float(self._rng.normal())
        self._v: float = self._weibull_quantile(_norm_cdf(self._x))
        self._direction: float = float(self._rng.uniform(0, 360))

    def _weibull_quantile(self, u: float) -> float:
        u = min(max(u, 1e-9), 1.0 - 1e-9)
        return self._c * (-math.log(1.0 - u)) ** (1.0 / self._k)

    def set_scale(self, weibull_c: float) -> None:
        """
        Retarget the Weibull scale parameter the AR(1) process reverts to.

        Called every step by :class:`WeatherModel` with a seasonally/diurnally
        modulated value so wind climatology varies smoothly through the year
        (e.g. windier in winter) instead of being a stationary process — the
        AR(1) update below is expressed in *standardised* space (``z``), so
        retargeting ``c`` before each step does not introduce a discontinuity:
        the physical wind speed drifts smoothly toward the new climatology.
        """
        self._c = max(0.5, float(weibull_c))

    def step(self) -> Dict[str, float]:
        """Generate next wind state."""
        # AR(1) in latent normal space, then map through the Weibull(k,c)
        # inverse CDF — see the constructor docstring for why this is what
        # keeps the long-run mean wind speed matching c*Gamma(1+1/k).
        sigma = math.sqrt(1 - self._autocorr ** 2)
        self._x = self._autocorr * self._x + sigma * self._rng.normal()
        self._v = self._weibull_quantile(_norm_cdf(self._x))

        # Wind direction — slow drift
        self._direction = (self._direction + self._rng.normal(0, 5.0)) % 360.0

        return {
            "wind_speed_ms": self._v,
            "wind_direction_deg": self._direction,
            "wind_ref_height_m": self._ref_h,
        }


# ------------------------------------------------------------------
# Main weather generator
# ------------------------------------------------------------------

class WeatherModel:
    """
    Integrated stochastic weather generator.

    Beyond a stationary per-step AR(1)/Weibull process, this model layers in
    the climatological structure that makes a year-long Italian simulation
    look realistic instead of statistically flat:

      * **Seasonal wind climatology** — the Weibull scale the AR(1) wind
        process reverts to is modulated by a winter-peaking cosine
        (``wind_seasonal_amplitude``), matching the stronger autumn/winter
        wind regime typical of Italy (frontal systems).
      * **Diurnal wind climatology** — a smaller afternoon boost
        (``wind_diurnal_amplitude``), approximating thermal/sea-breeze effects.
      * **Seasonal cloud climatology** — the cloud-cover AR(1) reversion
        target is winter-peaking / summer-trough (``cloud_seasonal_amplitude``)
        instead of a constant mean.
      * **Synoptic coupling** (optional ``synoptic`` argument to :meth:`step`)
        — a single shared disturbance index (supplied by
        :class:`~hytwin.weather.weather_field.WeatherField` for multi-site
        runs, spatially correlated across sites) nudges wind speed UP and
        cloud cover UP together (frontal/stormy conditions) or both DOWN
        together (anticyclonic calm), instead of treating wind and cloud as
        independent processes.

    Parameters
    ----------
    latitude_deg : float
    longitude_deg : float
    altitude_m : float
    weibull_k, weibull_c : Weibull wind parameters (annual-mean scale)
    autocorr_wind : float
    wind_seasonal_amplitude : float — fractional winter-peak wind swing, default 0.15
    wind_diurnal_amplitude : float  — fractional afternoon wind boost, default 0.08
    cloud_cover_mean : float  — annual-mean cloud cover fraction [0..1]
    cloud_seasonal_amplitude : float — fractional winter-peak cloud swing, default 0.30
    temp_mean_c : float       — annual mean temperature [°C]
    temp_amplitude_c : float  — seasonal amplitude [°C]
    """

    def __init__(
        self,
        latitude_deg: float = 40.5,
        longitude_deg: float = 14.8,
        altitude_m: float = 50.0,
        weibull_k: float = 2.0,
        weibull_c: float = 7.0,
        autocorr_wind: float = 0.92,
        wind_seasonal_amplitude: float = 0.15,
        wind_diurnal_amplitude: float = 0.08,
        cloud_cover_mean: float = 0.35,
        cloud_seasonal_amplitude: float = 0.30,
        temp_mean_c: float = 15.0,
        temp_amplitude_c: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._lat = latitude_deg
        self._lon = longitude_deg
        self._alt = altitude_m
        self._cloud_mean = cloud_cover_mean
        self._cloud_seasonal_amp = cloud_seasonal_amplitude
        self._T_mean = temp_mean_c
        self._T_amp = temp_amplitude_c
        self._rng = resolve_rng(rng)

        self._weibull_c_base = weibull_c
        self._wind_seasonal_amp = wind_seasonal_amplitude
        self._wind_diurnal_amp = wind_diurnal_amplitude
        self._wind = WindModel(weibull_k, weibull_c, autocorr_wind, rng=spawn_one(rng))

        # Cloud cover — AR(1) beta-distributed
        self._cloud: float = cloud_cover_mean
        self._cloud_autocorr: float = 0.90

        # Temperature AR(1)
        self._T: float = temp_mean_c
        self._T_autocorr: float = 0.97

        # Cloud and temperature start from the climatology of the first step's
        # date, not the annual mean (a January run would otherwise need ~1 day
        # to cool down).
        self._at_start: bool = True

    def step(self, timestamp: datetime, synoptic: float = 0.0) -> Dict[str, float]:
        """
        Generate weather for one time step.

        Parameters
        ----------
        timestamp : datetime
        synoptic : float
            Shared disturbance index (standardised, typically in [-3, 3]),
            supplied by :class:`WeatherField` for multi-site runs where it is
            spatially correlated across nearby sites — represents a frontal
            system (positive: windier *and* cloudier) or an anticyclone
            (negative: calmer *and* clearer) moving through the region.
            Defaults to 0 (no synoptic influence) for standalone use.

        Returns a dict with all weather variables needed by the models.
        """
        doy = timestamp.timetuple().tm_yday
        hour = timestamp.hour + timestamp.minute / 60.0

        # --- Wind: seasonal (winter-peaking) + diurnal (afternoon) climatology ---
        seasonal_wind_factor = 1.0 + self._wind_seasonal_amp * math.cos(2 * math.pi * (doy - 15) / 365)
        diurnal_wind_factor = 1.0 + self._wind_diurnal_amp * math.sin(math.pi * (hour - 9) / 12)
        c_eff = self._weibull_c_base * seasonal_wind_factor * diurnal_wind_factor
        self._wind.set_scale(c_eff)
        wind_state = self._wind.step()
        # Synoptic coupling: a frontal system (synoptic>0) boosts wind speed;
        # an anticyclone (synoptic<0) calms it.
        wind_synoptic_mult = float(np.clip(1.0 + 0.4 * synoptic, 0.35, 2.2))
        wind_state = dict(wind_state)
        wind_state["wind_speed_ms"] = max(0.0, wind_state["wind_speed_ms"] * wind_synoptic_mult)

        # --- Cloud cover: proper mean-reverting AR(1) on logit-transformed
        # cloud fraction — x_t = mu + phi*(x_{t-1}-mu) + eps_t, reverting to a
        # SEASONALLY-MODULATED target `mu` (winter-peaking) rather than to a
        # fixed constant, so each site's configured `cloud_cover_mean` stays
        # meaningful over a long run instead of washing out to ~0.5.
        seasonal_cloud_mean = float(np.clip(
            self._cloud_mean * (1.0 + self._cloud_seasonal_amp * math.cos(2 * math.pi * (doy - 15) / 365)),
            0.02, 0.97,
        ))
        if self._at_start:
            self._cloud = seasonal_cloud_mean
        logit_target = math.log(seasonal_cloud_mean / (1 - seasonal_cloud_mean))
        cloud_noise_std = math.sqrt(1 - self._cloud_autocorr ** 2) * 0.15
        logit_cloud = math.log(self._cloud / (1 - self._cloud + 1e-9) + 1e-9)
        logit_cloud = (
            logit_target
            + self._cloud_autocorr * (logit_cloud - logit_target)
            + 0.8 * synoptic  # frontal system -> more cloud; anticyclone -> clearer
            + self._rng.normal(0, cloud_noise_std)
        )
        self._cloud = 1.0 / (1.0 + math.exp(-logit_cloud))
        self._cloud = float(np.clip(self._cloud, 0.0, 1.0))

        # --- Solar position ---
        sol = solar_position(timestamp, self._lat, self._lon)
        zenith = sol["zenith_deg"]

        # --- Irradiance ---
        ghi_clear = _clear_sky_ghi(zenith, self._alt)
        # Cloud attenuation: simplified Kasten model
        cloud_attenuation = 1.0 - 0.75 * self._cloud ** 3.4
        ghi = ghi_clear * cloud_attenuation
        # Diffuse fraction (erbs correlation)
        Kt = ghi / (ghi_clear + 1e-9) if ghi_clear > 5 else 0.0
        if Kt <= 0.22:
            dhi_frac = 1.0 - 0.09 * Kt
        elif Kt <= 0.8:
            dhi_frac = 0.9511 - 0.1604 * Kt + 4.388 * Kt**2 - 16.638 * Kt**3 + 12.336 * Kt**4
        else:
            dhi_frac = 0.165
        dhi = ghi * dhi_frac
        dni = max(0.0, (ghi - dhi) / (math.cos(math.radians(zenith)) + 1e-9)) if zenith < 89 else 0.0
        # Add irradiance measurement noise
        ghi = max(0.0, ghi + self._rng.normal(0, 5.0))
        dhi = max(0.0, dhi + self._rng.normal(0, 2.0))

        # --- Temperature ---
        # Note the minus sign: unlike wind/cloud (winter-peaking, so they use
        # +cos(2π(doy-15)/365) which is maximal at day 15), temperature is
        # SUMMER-peaking — coldest in mid-January, hottest in mid-July.
        seasonal_T = self._T_mean - self._T_amp * math.cos(2 * math.pi * (doy - 15) / 365)
        diurnal_T = 5.0 * math.sin(math.pi * (timestamp.hour - 6) / 12)
        if self._at_start:
            self._T = seasonal_T + diurnal_T
            self._at_start = False
        T_noise_std = math.sqrt(1 - self._T_autocorr ** 2) * 1.5
        self._T = self._T_autocorr * self._T + (1 - self._T_autocorr) * (seasonal_T + diurnal_T) + self._rng.normal(0, T_noise_std)

        # --- Humidity ---
        rh = float(np.clip(0.50 + 0.20 * self._cloud + self._rng.normal(0, 0.05), 0.1, 1.0))

        return {
            "timestamp": timestamp,
            "wind_speed_ms": wind_state["wind_speed_ms"],
            "wind_direction_deg": wind_state["wind_direction_deg"],
            "wind_ref_height_m": wind_state["wind_ref_height_m"],
            "ghi_wm2": ghi,
            "dhi_wm2": dhi,
            "dni_wm2": dni,
            "cloud_cover": self._cloud,
            "temperature_c": self._T,
            "relative_humidity": rh,
            "sun_zenith_deg": zenith,
            "sun_azimuth_deg": sol["azimuth_deg"],
            "sun_elevation_deg": sol["elevation_deg"],
            "altitude_m": self._alt,
            "latitude_deg": self._lat,
            "longitude_deg": self._lon,
        }

    def reset(self) -> None:
        self._cloud = self._cloud_mean
        self._T = self._T_mean
        self._at_start = True
        # Rebuild from the ORIGINAL base scale (self._wind._c drifts away from
        # it each step via set_scale()'s seasonal/diurnal modulation) and
        # preserve the existing generator (whether a real per-instance
        # Generator or the legacy global-RNG proxy) so a reset doesn't
        # silently break the RNG isolation a NetworkTwin(seed=...) set up.
        self._wind = WindModel(
            self._wind._k, self._weibull_c_base, self._wind._autocorr, rng=self._wind._rng,
        )

    def forecast(
        self,
        timestamp: datetime,
        n_steps: int,
        dt_seconds: float,
    ) -> list:
        """
        Return noise-free expected-value weather forecasts without
        advancing or mutating internal state.

        Uses AR(1) mean reversion for wind and cloud, deterministic
        solar geometry for GHI. No stochastic draws are made.

        Parameters
        ----------
        timestamp : datetime
            Current simulation timestamp (forecasts start at t + dt).
        n_steps : int
            Number of future steps to forecast.
        dt_seconds : float
            Time step between forecast points [s].

        Returns
        -------
        list of dict
            Each dict contains:
            ``wind_speed_ms``, ``cloud_cover``, ``ghi_wm2``,
            ``ghi_clear_wm2``, ``solar_elevation_deg``.
        """
        if n_steps <= 0:
            return []

        # Snapshot current AR state (read-only)
        v_curr = self._wind._v
        c_w = self._wind._c
        k_w = self._wind._k
        autocorr_w = self._wind._autocorr
        z_curr = (v_curr / (c_w + 1e-9)) ** k_w  # Weibull z-transform

        cloud_curr = self._cloud
        cloud_autocorr = self._cloud_autocorr
        cloud_mean = self._cloud_mean

        results = []
        for i in range(1, n_steps + 1):
            ft = timestamp + timedelta(seconds=i * dt_seconds)

            # Wind AR(1) expected value in Weibull z-space
            # E[z(t+k)] = autocorr^k * z(t) + (1 - autocorr^k) * 1.0
            # (normalised Weibull mean is 1.0 in power-k space)
            alpha_w = autocorr_w ** i
            z_fcast = alpha_w * z_curr + (1.0 - alpha_w) * 1.0
            v_fcast = c_w * max(z_fcast, 1e-9) ** (1.0 / k_w)

            # Cloud AR(1) mean reversion
            alpha_c = cloud_autocorr ** i
            cloud_fcast = float(np.clip(
                alpha_c * cloud_curr + (1.0 - alpha_c) * cloud_mean,
                0.0, 1.0,
            ))

            # Solar geometry (fully deterministic)
            sol = solar_position(ft, self._lat, self._lon)
            ghi_clear = _clear_sky_ghi(sol["zenith_deg"], self._alt)
            cloud_atten = 1.0 - 0.75 * cloud_fcast ** 3.4
            ghi = max(0.0, ghi_clear * cloud_atten)

            results.append({
                "wind_speed_ms": v_fcast,
                "cloud_cover": cloud_fcast,
                "ghi_wm2": ghi,
                "ghi_clear_wm2": ghi_clear,
                "solar_elevation_deg": sol["elevation_deg"],
            })

        return results
