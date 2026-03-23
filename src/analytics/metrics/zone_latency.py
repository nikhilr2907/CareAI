"""Zone latency metric: average/p95/p99/std latency per spatial quadrant."""

from collections import defaultdict
from .base import BaseMetric


class ZoneLatencyMetric(BaseMetric):
    """
    Compute latency statistics grouped by destination quadrant.

    Reveals which areas of the hospital are slow.
    """

    def compute(self) -> dict:
        """
        Compute latency stats per quadrant.

        Returns:
            {
                'SW': {'mean': float, 'p95': float, 'p99': float, 'std': float, 'count': int},
                'SE': {...},
                'NW': {...},
                'NE': {...},
            }
        """
        if not self._validate():
            return {}

        quadrant_latencies = defaultdict(list)

        # Group task latencies by destination quadrant
        for entry in self.task_logger.cost_data:
            to_idx = entry.get('to_location_idx')
            latency = entry.get('latency')

            if to_idx is not None and latency is not None:
                quadrant = self._get_quadrant(to_idx)
                quadrant_latencies[quadrant].append(latency)

        # Compute stats per quadrant, label with cardinal directions
        result = {}
        for q in range(4):
            latencies = quadrant_latencies.get(q, [])
            result[self.helpers.quadrant_label(q)] = self._stats(latencies)

        return result
