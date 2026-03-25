"""Route performance metric: latency by origin→destination quadrant pairs."""

from collections import defaultdict
from .base import BaseMetric


class RoutePerformanceMetric(BaseMetric):
    """
    Compute latency statistics grouped by origin→destination quadrant pairs.

    Reveals which routes through the hospital are slow.
    """

    def compute(self) -> dict:
        """
        Compute latency stats per route (from_quadrant → to_quadrant).

        Returns:
            {
                'NW→NE': {'mean': float, 'p95': float, 'count': int},
                'NW→SE': {...},
                'NW→SW': {...},
                'NE→NW': {...},
                'NE→SW': {...},
                'NE→SE': {...},
                'SE→NW': {...},
                'SE→NE': {...},
                'SE→SW': {...},
                'SW→NW': {...},
                'SW→NE': {...},
                'SW→SE': {...}
            }
        """
        if not self._validate():
            return {}

        route_latencies = defaultdict(list)

        # Group by (from_quadrant, to_quadrant) pair
        for entry in self.task_logger.cost_data:
            from_idx = entry.get('from_location_idx')
            to_idx = entry.get('to_location_idx')
            latency = entry.get('latency')

            if None not in (from_idx, to_idx, latency):
                from_quad = self._get_quadrant(from_idx)
                to_quad = self._get_quadrant(to_idx)
                route_key = (from_quad, to_quad)
                route_latencies[route_key].append(latency)

        # Compute stats per route with cardinal labels
        result = {}
        for (from_q, to_q), latencies in route_latencies.items():
            route_label = f"{self.helpers.quadrant_label(from_q)}→{self.helpers.quadrant_label(to_q)}"
            result[route_label] = self._stats(latencies)

        return result
