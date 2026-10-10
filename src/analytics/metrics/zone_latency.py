from collections import defaultdict

import numpy as np

from .helpers import MetricHelpers


class ZoneLatencyMetric:
    """Latency statistics grouped by spatial quadrant, viewed either by destination (compute_by_zone) or by origin→destination pair (compute_by_route)."""

    def __init__(self, task_logger, nodes, helpers: MetricHelpers):
        self.task_logger = task_logger
        self.nodes = nodes
        self.helpers = helpers

    def compute(self) -> dict:
        """Return both zone and route views."""
        return {
            "by_zone": self.compute_by_zone(),
            "by_route": self.compute_by_route(),
        }

    def compute_by_zone(self) -> dict:
        """Latency stats per destination quadrant, plus efficiency comparison against the fastest zone."""
        if not self._validate():
            return {}

        quadrant_latencies = defaultdict(list)
        for entry in self.task_logger.cost_data:
            to_idx = entry.get("to_location_idx")
            latency = entry.get("latency")
            if to_idx is not None and latency is not None:
                quadrant_latencies[self._get_quadrant(to_idx)].append(latency)

        result = {}
        for q in range(4):
            label = self.helpers.quadrant_label(q)
            result[label] = self._stats(quadrant_latencies.get(q, []))

        means = {label: s["mean"] for label, s in result.items() if s["count"] > 0}
        if means:
            fastest = min(means.values())
            if fastest > 0:
                for label, mean in means.items():
                    result[label]["percent_loss"] = round(((mean - fastest) / fastest) * 100, 1)
                    result[label]["absolute_loss_s"] = round(mean - fastest, 2)
                    result[label]["is_baseline"] = mean == fastest

        return result

    def compute_by_route(self) -> dict:
        """Latency stats per origin→destination quadrant pair."""
        if not self._validate():
            return {}

        route_latencies = defaultdict(list)
        for entry in self.task_logger.cost_data:
            from_idx = entry.get("from_location_idx")
            to_idx = entry.get("to_location_idx")
            latency = entry.get("latency")
            if None not in (from_idx, to_idx, latency):
                route_latencies[(self._get_quadrant(from_idx), self._get_quadrant(to_idx))].append(
                    latency
                )

        return {
            f"{self.helpers.quadrant_label(fq)}→{self.helpers.quadrant_label(tq)}": self._stats(
                lats
            )
            for (fq, tq), lats in route_latencies.items()
        }

    def _validate(self) -> bool:
        return bool(self.task_logger and self.task_logger.cost_data and self.nodes)

    def _get_quadrant(self, location_idx) -> int:
        return self.helpers.get_location_quadrant(location_idx)

    @staticmethod
    def _stats(values: list) -> dict:
        if not values:
            return {
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        arr = np.array(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": len(values),
        }
