"""
Network State
=============
Hierarchical state of a multi-site H2 network for one time step:

  * :class:`NodeState` — per-site balance, storage, economics (wraps the
    site's underlying :class:`~hytwin.digital_twin.grid_twin.GridState`).
  * :class:`LinkFlow`  — flow / loss / utilisation on one interconnection.
  * :class:`NetworkState` — the two collections above plus aggregated
    system-level KPIs (cost, CO₂, self-sufficiency, reliability).

These are plain dataclasses so they serialise easily for the recorder / the
dashboard, mirroring how ``GridState`` is used today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class NodeState:
    """Per-site state after network dispatch."""
    site_id: str
    timestamp: datetime

    # Electric balance [kW].  renewable / fuel cell / load / electrolyzer are the twin's
    # measured (sensor-fused) view that controllers observe; ``physical`` has the actual
    # values, which generation_kw / demand_kw, the dispatch and the costs are based on.
    renewable_kw: float = 0.0
    fuel_cell_kw: float = 0.0
    load_kw: float = 0.0
    electrolyzer_kw: float = 0.0
    compressor_load_kw: float = 0.0          # H2 pipeline compression at this site
    generation_kw: float = 0.0               # renewable + fuel cell
    demand_kw: float = 0.0                   # load + electrolyzer + compressor

    # Inter-node electric exchange [kW]
    link_import_kw: float = 0.0              # power delivered in from neighbours
    link_export_kw: float = 0.0              # power injected out to neighbours

    # National grid slack [kW]
    grid_import_kw: float = 0.0
    grid_export_kw: float = 0.0
    curtailed_kw: float = 0.0
    unmet_kw: float = 0.0
    grid_available: bool = True

    # Hydrogen [kg / kg-h]
    h2_soc: float = 0.0
    h2_storage_kg: float = 0.0
    h2_pipeline_in_kg_h: float = 0.0         # delivered in from pipelines
    h2_pipeline_out_kg_h: float = 0.0        # injected out to pipelines

    # Economics
    price_eur_kwh: float = 0.0
    step_cost_eur: float = 0.0
    step_co2_kg: float = 0.0

    # Underlying single-site grid state (for dashboard / drill-down reuse)
    grid_state: Any = None
    physical: Dict[str, float] = field(default_factory=dict)


@dataclass
class LinkFlow:
    """Flow state of one interconnection this step."""
    link_id: str
    link_type: str
    from_site: str
    to_site: str
    flow: float = 0.0            # source-side (kW for electric, kg/h for H2)
    delivered: float = 0.0       # destination-side after loss / delay
    loss: float = 0.0
    utilization: float = 0.0
    available: bool = True


@dataclass
class NetworkState:
    """Aggregated network state snapshot."""
    timestamp: datetime
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    links: Dict[str, LinkFlow] = field(default_factory=dict)

    # System-level KPIs
    total_load_kw: float = 0.0
    total_renewable_kw: float = 0.0
    total_generation_kw: float = 0.0
    total_grid_import_kw: float = 0.0
    total_grid_export_kw: float = 0.0
    total_curtailed_kw: float = 0.0
    unmet_demand_kw: float = 0.0

    total_cost_eur_step: float = 0.0
    cumulative_cost_eur: float = 0.0
    total_co2_kg_step: float = 0.0

    inter_node_power_kw: float = 0.0         # sum of electric link deliveries
    inter_node_h2_kg_h: float = 0.0          # sum of pipeline deliveries

    network_renewable_fraction: float = 0.0
    network_self_sufficiency: float = 0.0
    reliability_index: float = 1.0           # 1 - unmet/load this step
    avg_h2_soc: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Flat JSON-friendly dict of system KPIs (nodes/links summarised)."""
        return {
            "ts": self.timestamp.isoformat(),
            "total_load_kw": round(self.total_load_kw, 1),
            "total_renewable_kw": round(self.total_renewable_kw, 1),
            "total_generation_kw": round(self.total_generation_kw, 1),
            "total_grid_import_kw": round(self.total_grid_import_kw, 1),
            "total_grid_export_kw": round(self.total_grid_export_kw, 1),
            "total_curtailed_kw": round(self.total_curtailed_kw, 1),
            "unmet_demand_kw": round(self.unmet_demand_kw, 2),
            "total_cost_eur_step": round(self.total_cost_eur_step, 4),
            "cumulative_cost_eur": round(self.cumulative_cost_eur, 2),
            "total_co2_kg_step": round(self.total_co2_kg_step, 4),
            "inter_node_power_kw": round(self.inter_node_power_kw, 1),
            "inter_node_h2_kg_h": round(self.inter_node_h2_kg_h, 3),
            "network_renewable_fraction": round(self.network_renewable_fraction, 4),
            "network_self_sufficiency": round(self.network_self_sufficiency, 4),
            "reliability_index": round(self.reliability_index, 4),
            "avg_h2_soc": round(self.avg_h2_soc, 4),
            "nodes": {sid: {
                "load_kw": round(n.load_kw, 1),
                "renewable_kw": round(n.renewable_kw, 1),
                "fuel_cell_kw": round(n.fuel_cell_kw, 1),
                "electrolyzer_kw": round(n.electrolyzer_kw, 1),
                "generation_kw": round(n.generation_kw, 1),
                "grid_import_kw": round(n.grid_import_kw, 1),
                "grid_export_kw": round(n.grid_export_kw, 1),
                "curtailed_kw": round(n.curtailed_kw, 1),
                "link_import_kw": round(n.link_import_kw, 1),
                "link_export_kw": round(n.link_export_kw, 1),
                "unmet_kw": round(n.unmet_kw, 2),
                "h2_soc": round(n.h2_soc, 4),
                "h2_storage_kg": round(n.h2_storage_kg, 1),
                "h2_pipeline_in_kg_h": round(n.h2_pipeline_in_kg_h, 2),
                "h2_pipeline_out_kg_h": round(n.h2_pipeline_out_kg_h, 2),
                "price_eur_kwh": round(n.price_eur_kwh, 4),
                "step_cost_eur": round(n.step_cost_eur, 4),
                "grid_available": n.grid_available,
            } for sid, n in self.nodes.items()},
            "links": {lid: {
                "type": l.link_type,
                "from": l.from_site,
                "to": l.to_site,
                "flow": round(l.flow, 2),
                "delivered": round(l.delivered, 2),
                "utilization": round(l.utilization, 3),
            } for lid, l in self.links.items()},
        }
