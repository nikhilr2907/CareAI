"""Task completion distribution metric: full latency distribution per quadrant."""

from collections import defaultdict
import numpy as np
from .base import BaseMetric


class TaskCompletionDistributionMetric(BaseMetric):
    """
    Compute full latency distribution (p50, p95, p99, std, min, max) per quadrant.

    Shows variance in task completion times—high variance indicates unpredictable congestion.
    """

    def compute(self) -> dict:
        """
        Compute latency distribution per quadrant.

        Returns:
            {
                'NW': {
                    'p50': float, 'p95': float, 'p99': float, 'std': float,
                    'min': float, 'max': float, 'mean': float, 'count': int
                },
                'NE': {...},
                'SE': {...},
                'SW': {...}
            }
        """
        if not self._validate():
            return {}

        quadrant_latencies = defaultdict(list)

        # Group by destination quadrant
        for entry in self.task_logger.cost_data:
            to_idx = entry.get('to_location_idx')
            latency = entry.get('latency')

            if to_idx is not None and latency is not None:
                quadrant = self._get_quadrant(to_idx)
                quadrant_latencies[quadrant].append(latency)

        # Compute full distribution per quadrant
        result = {}
        for q in range(4):
            latencies = quadrant_latencies.get(q, [])
            if not latencies:
                continue

            arr = np.array(latencies, dtype=float)
            stats = self._stats(latencies)
            # Add distribution extremes and median
            stats['p50'] = float(np.percentile(arr, 50))
            stats['min'] = float(arr.min())
            stats['max'] = float(arr.max())

            result[self.helpers.quadrant_label(q)] = stats

        return result
