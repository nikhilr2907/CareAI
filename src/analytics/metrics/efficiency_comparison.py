"""Efficiency comparison metric: % latency loss vs fastest quadrant."""

from .base import BaseMetric
from .zone_latency import ZoneLatencyMetric


class EfficiencyComparisonMetric(BaseMetric):
    """
    Compute operational efficiency loss by comparing each quadrant to the fastest.

    Shows which areas are bottlenecks: high % loss indicates operational drag.
    """

    def compute(self) -> dict:
        """
        Compute efficiency loss (% slower vs fastest quadrant).

        Returns:
            {
                'SW': {
                    'percent_loss': float,       # % slower than fastest
                    'absolute_loss_s': float,    # seconds slower
                    'mean_latency_s': float,     # actual mean latency
                    'is_baseline': bool,         # True if fastest
                },
                'SE': {...},
                ...
            }
        """
        if not self._validate():
            return {}

        # Reuse zone latency computation
        zone_stats = ZoneLatencyMetric(
            self.task_logger, self.nodes, self.helpers
        ).compute()

        if not zone_stats:
            return {}

        # Extract mean latencies for zones with data
        means = {
            q: s['mean']
            for q, s in zone_stats.items()
            if s.get('count', 0) > 0
        }

        if not means:
            return {}

        # Find fastest quadrant
        fastest = min(means.values())
        if fastest == 0:
            return {}

        # Compute efficiency loss for each quadrant
        result = {}
        for quadrant_label, mean in means.items():
            pct_loss = ((mean - fastest) / fastest) * 100
            result[quadrant_label] = {
                'percent_loss': round(pct_loss, 1),
                'absolute_loss_s': round(mean - fastest, 2),
                'mean_latency_s': round(mean, 2),
                'is_baseline': pct_loss == 0.0,
            }

        return result
