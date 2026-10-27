"""
Network Topology
================
Structural description of a **multi-site H2 energy network**: a set of
geolocated *sites* (each one an independent H2 plant, i.e. what a single
``GridTwin`` models today) connected by *links* — electric lines and/or
hydrogen pipelines.

This module is intentionally **pure/structural**: it only describes the
topology (who is where, what connects to what, with which parameters).
The physics of the links lives in ``hytwin/models/`` and the orchestration
lives in ``hytwin/network/network_twin.py`` (added in a later phase).

Design notes
------------
* Every site keeps *exactly* the same config schema used today under the
  YAML ``grid:`` / ``weather:`` / ``energy_cost:`` / ``sensors:`` keys, so a
  single ``SiteSpec`` can be handed to a ``GridTwin`` unchanged.
* Site choice is **fully data-driven**: coordinates, sizes and links all come
  from the YAML.  If real plant data becomes available, only the YAML changes —
  no code.  Link lengths are auto-computed from coordinates (great-circle)
  unless explicitly overridden, so moving a site updates distances for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Geographic helpers
# ------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points [km]."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ------------------------------------------------------------------
# Link kinds
# ------------------------------------------------------------------

class LinkType(str, Enum):
    """Type of inter-site interconnection."""
    ELECTRIC_LINE = "electric_line"
    H2_PIPELINE = "h2_pipeline"

    @classmethod
    def from_str(cls, value: str) -> "LinkType":
        v = str(value).strip().lower()
        for member in cls:
            if member.value == v:
                return member
        raise ValueError(
            f"Unknown link type {value!r}. "
            f"Valid: {[m.value for m in cls]}"
        )


# ------------------------------------------------------------------
# Location
# ------------------------------------------------------------------

@dataclass
class Location:
    """Geographic location of a site."""
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "Location":
        cfg = cfg or {}
        return cls(
            name=str(cfg.get("name", "")),
            lat=float(cfg.get("lat", cfg.get("latitude_deg", 0.0))),
            lon=float(cfg.get("lon", cfg.get("longitude_deg", 0.0))),
            alt_m=float(cfg.get("alt_m", cfg.get("altitude_m", 0.0))),
        )


# ------------------------------------------------------------------
# Site
# ------------------------------------------------------------------

@dataclass
class SiteSpec:
    """
    Full specification of one network site (an independent H2 plant).

    Attributes
    ----------
    id : str
    location : Location
    grid_config : dict
        Component specs — identical schema to today's YAML ``grid:`` block.
    weather_params : dict
        kwargs for ``WeatherModel`` for this site's local climate.
    energy_cost : dict
        Local market-zone price model config (optional).
    sensor_config : list
        Virtual sensor specs for this site (optional).
    """
    id: str
    location: Location = field(default_factory=Location)
    grid_config: Dict[str, Any] = field(default_factory=dict)
    weather_params: Dict[str, Any] = field(default_factory=dict)
    energy_cost: Dict[str, Any] = field(default_factory=dict)
    sensor_config: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SiteSpec":
        if "id" not in cfg:
            raise ValueError("Each site must declare an 'id'.")
        location = Location.from_config(cfg.get("location"))

        # Merge location geodata into weather params so WeatherModel gets
        # the right lat/lon/alt even if the weather block omits them.
        weather_params = dict(cfg.get("weather", {}))
        weather_params.setdefault("latitude_deg", location.lat)
        weather_params.setdefault("longitude_deg", location.lon)
        weather_params.setdefault("altitude_m", location.alt_m)

        # The site's grid config may carry energy_cost inline (backward-compat
        # with how Scenario.from_yaml forwards it into grid_config today).
        grid_config = dict(cfg.get("grid", {}))
        energy_cost = dict(cfg.get("energy_cost", {}))
        if energy_cost and "energy_cost" not in grid_config:
            grid_config["energy_cost"] = energy_cost

        return cls(
            id=str(cfg["id"]),
            location=location,
            grid_config=grid_config,
            weather_params=weather_params,
            energy_cost=energy_cost,
            sensor_config=list(cfg.get("sensors", [])),
        )


# ------------------------------------------------------------------
# Link
# ------------------------------------------------------------------

@dataclass
class LinkSpec:
    """
    Specification of one interconnection between two sites.

    ``length_km`` is taken from ``params['length_km']`` if present, otherwise
    computed as the great-circle distance between the two endpoints once the
    topology is assembled (see ``NetworkTopology._finalize_links``).
    """
    id: str
    link_type: LinkType
    from_site: str
    to_site: str
    params: Dict[str, Any] = field(default_factory=dict)
    length_km: Optional[float] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LinkSpec":
        for req in ("id", "type", "from", "to"):
            if req not in cfg:
                raise ValueError(f"Link is missing required key {req!r}: {cfg}")
        params = dict(cfg.get("params", {}))
        length = params.get("length_km")
        return cls(
            id=str(cfg["id"]),
            link_type=LinkType.from_str(cfg["type"]),
            from_site=str(cfg["from"]),
            to_site=str(cfg["to"]),
            params=params,
            length_km=float(length) if length is not None else None,
        )


# ------------------------------------------------------------------
# Topology
# ------------------------------------------------------------------

@dataclass
class NetworkTopology:
    """
    A geolocated multi-site network: sites + links.

    Use :meth:`from_config` to build from a parsed YAML ``network:`` block, or
    :meth:`single_site` to wrap a legacy single-plant config (backward-compat).
    """
    sites: Dict[str, SiteSpec] = field(default_factory=dict)
    links: Dict[str, LinkSpec] = field(default_factory=dict)
    # NetworkRewardConfig overrides for this network (``rl_reward`` YAML block).
    rl_reward: Dict[str, float] = field(default_factory=dict)

    # -- construction --------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "NetworkTopology":
        """Build from a ``network`` dict: {sites: [...], links: [...]}."""
        sites: Dict[str, SiteSpec] = {}
        for s_cfg in cfg.get("sites", []):
            site = SiteSpec.from_config(s_cfg)
            if site.id in sites:
                raise ValueError(f"Duplicate site id {site.id!r}")
            sites[site.id] = site

        links: Dict[str, LinkSpec] = {}
        for l_cfg in cfg.get("links", []):
            link = LinkSpec.from_config(l_cfg)
            if link.id in links:
                raise ValueError(f"Duplicate link id {link.id!r}")
            links[link.id] = link

        topo = cls(sites=sites, links=links,
                   rl_reward={k: float(v) for k, v in (cfg.get("rl_reward") or {}).items()})
        topo.validate()
        topo._finalize_links()
        return topo

    @classmethod
    def single_site(
        cls,
        grid_config: Dict[str, Any],
        weather_params: Dict[str, Any],
        sensor_config: Optional[List[Dict[str, Any]]] = None,
        site_id: str = "site1",
        location: Optional[Location] = None,
    ) -> "NetworkTopology":
        """Wrap a legacy single-plant config as a 1-site, 0-link network."""
        if location is None:
            location = Location(
                name=site_id,
                lat=float(weather_params.get("latitude_deg", 0.0)),
                lon=float(weather_params.get("longitude_deg", 0.0)),
                alt_m=float(weather_params.get("altitude_m", 0.0)),
            )
        site = SiteSpec(
            id=site_id,
            location=location,
            grid_config=dict(grid_config),
            weather_params=dict(weather_params),
            energy_cost=dict(grid_config.get("energy_cost", {})),
            sensor_config=list(sensor_config or []),
        )
        return cls(sites={site_id: site}, links={})

    # -- integrity -----------------------------------------------------

    def validate(self) -> None:
        """Raise if the topology references unknown sites or is empty."""
        if not self.sites:
            raise ValueError("Network topology has no sites.")
        for link in self.links.values():
            if link.from_site not in self.sites:
                raise ValueError(
                    f"Link {link.id!r} references unknown from-site "
                    f"{link.from_site!r}"
                )
            if link.to_site not in self.sites:
                raise ValueError(
                    f"Link {link.id!r} references unknown to-site "
                    f"{link.to_site!r}"
                )
            if link.from_site == link.to_site:
                raise ValueError(f"Link {link.id!r} connects a site to itself.")

    def _finalize_links(self) -> None:
        """Fill in missing link lengths from great-circle site distances."""
        for link in self.links.values():
            if link.length_km is None:
                a = self.sites[link.from_site].location
                b = self.sites[link.to_site].location
                link.length_km = haversine_km(a.lat, a.lon, b.lat, b.lon)
                # expose it back into params for downstream physics models
                link.params.setdefault("length_km", link.length_km)

    # -- queries -------------------------------------------------------

    @property
    def site_ids(self) -> List[str]:
        return list(self.sites.keys())

    @property
    def is_multi_site(self) -> bool:
        return len(self.sites) > 1

    def links_of(self, site_id: str) -> List[LinkSpec]:
        """All links incident to a site (as from- or to-endpoint)."""
        return [
            l for l in self.links.values()
            if l.from_site == site_id or l.to_site == site_id
        ]

    def neighbors(self, site_id: str) -> List[str]:
        """Site ids directly connected to ``site_id`` by any link."""
        out: List[str] = []
        for l in self.links.values():
            if l.from_site == site_id and l.to_site not in out:
                out.append(l.to_site)
            elif l.to_site == site_id and l.from_site not in out:
                out.append(l.from_site)
        return out

    def links_by_type(self, link_type: LinkType) -> List[LinkSpec]:
        return [l for l in self.links.values() if l.link_type == link_type]

    def __repr__(self) -> str:
        return (
            f"NetworkTopology(sites={self.site_ids}, "
            f"links={[ (l.id, l.link_type.value) for l in self.links.values() ]})"
        )
