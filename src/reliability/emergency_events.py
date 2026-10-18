import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment.graph.graph_state import GraphState


class EmergencyEventManager:
    """Rare, independent patient-movement emergencies (crash call / rapid transfer) as a Poisson process.

    Each edge has its own emergency_rate_per_hour; emergencies are memoryless and don't cluster,
    which matches how code-blue call frequency is typically modeled in hospital operations research.
    While active, effective_speed() floors to force other traffic to yield rather than negotiate.
    """

    def __init__(self, enable: bool = True, duration_s_range: tuple[float, float] = (120.0, 300.0)):
        self.enable = enable
        self.duration_s_range = duration_s_range
        self._active_until: dict[int, float] = {}

    def update(self, graph_state: "GraphState", current_time: float, dt: float) -> None:
        if not self.enable:
            return

        still_active = {}
        for edge_index, end_time in self._active_until.items():
            if current_time >= end_time:
                graph_state.edges[edge_index].emergency_active = False
            else:
                still_active[edge_index] = end_time
        self._active_until = still_active

        for edge_index, edge in enumerate(graph_state.edges):
            if edge_index in self._active_until or edge.emergency_rate_per_hour <= 0:
                continue
            trigger_prob = edge.emergency_rate_per_hour * dt / 3600.0
            if random.random() < trigger_prob:
                edge.emergency_active = True
                self._active_until[edge_index] = current_time + random.uniform(*self.duration_s_range)
