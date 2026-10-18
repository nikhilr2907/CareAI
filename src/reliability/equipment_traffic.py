import math
import random
from typing import TYPE_CHECKING

from ..environment.graph.edge import GraphEdge

if TYPE_CHECKING:
    from ..environment.graph.graph_state import GraphState


class EquipmentTrafficManager:
    """Equipment arrivals/departures per edge as a non-homogeneous Poisson process.

    Plain (homogeneous) Poisson would spread meal trolleys/linen rounds uniformly across
    24 hours; real hospital logistics cluster around meal times and shift changes, so the
    rate itself varies with time of day (a sum of Gaussian bumps per category). Categories
    with no real diurnal pattern (drug trolleys, wheelchairs) default to a flat multiplier.
    """

    # category -> hospital-wide default arrivals/hour, used when an edge sets no override
    _DEFAULT_BASE_RATE = {
        "drug_trolley": 2.0,
        "wheelchair": 1.5,
        "cleaning_cart": 0.5,
        "linen_cart": 0.8,
        "meal_trolley": 0.3,
        "imaging_unit": 0.1,
        "pallet_jack": 0.2,
        "bariatric_bed": 0.05,
    }

    # category -> dwell time range (s) once arrived, before it moves on
    _DWELL_S = {
        "drug_trolley": (30.0, 90.0),
        "wheelchair": (60.0, 180.0),
        "cleaning_cart": (300.0, 600.0),
        "linen_cart": (120.0, 300.0),
        "meal_trolley": (180.0, 360.0),
        "imaging_unit": (300.0, 600.0),
        "pallet_jack": (180.0, 400.0),
        "bariatric_bed": (300.0, 600.0),
    }

    # category -> [(peak_hour, peak_multiplier, spread_hours), ...]; omitted categories stay flat.
    _TIME_PROFILES: dict[str, list[tuple[float, float, float]]] = {
        "meal_trolley": [(8.0, 5.0, 1.0), (12.0, 5.0, 1.0), (18.0, 5.0, 1.0)],
        "linen_cart": [(7.0, 3.0, 1.5)],
        "cleaning_cart": [(2.0, 3.0, 2.0)],
    }

    def __init__(self, enable: bool = True):
        self.enable = enable
        self._active: list[tuple[int, str, float]] = []  # (edge_index, category, end_time)

    def _time_multiplier(self, category: str, current_time: float) -> float:
        bumps = self._TIME_PROFILES.get(category)
        if not bumps:
            return 1.0
        hour = (current_time % 86400.0) / 3600.0
        total = 0.0
        for peak_hour, peak_mult, spread in bumps:
            diff = min(abs(hour - peak_hour), 24.0 - abs(hour - peak_hour))
            total += peak_mult * math.exp(-0.5 * (diff / spread) ** 2)
        return max(0.1, total)

    def update(self, graph_state: "GraphState", current_time: float, dt: float) -> None:
        if not self.enable:
            return

        still_active = []
        for edge_index, category, end_time in self._active:
            if current_time >= end_time:
                edge = graph_state.edges[edge_index]
                edge.equipment_counts[category] = max(0, edge.equipment_counts.get(category, 0) - 1)
            else:
                still_active.append((edge_index, category, end_time))
        self._active = still_active

        for edge_index, edge in enumerate(graph_state.edges):
            for category, default_rate in self._DEFAULT_BASE_RATE.items():
                rate = edge.equipment_base_rate.get(category, default_rate)
                rate *= self._time_multiplier(category, current_time)
                if random.random() < rate * dt / 3600.0:
                    edge.equipment_counts[category] = edge.equipment_counts.get(category, 0) + 1
                    dwell = random.uniform(*self._DWELL_S[category])
                    self._active.append((edge_index, category, current_time + dwell))


assert set(EquipmentTrafficManager._DEFAULT_BASE_RATE) == set(GraphEdge._EQUIPMENT_SPECS)
