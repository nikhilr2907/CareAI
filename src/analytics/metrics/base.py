"""Base class for all flow metrics."""

from abc import ABC, abstractmethod
import numpy as np


class BaseMetric(ABC):
    """
    Abstract base for flow metrics.

    Enforces uniform interface and provides shared utilities for stats computation.
    """

    def __init__(self, task_logger, nodes, helpers):
        """
        Args:
            task_logger: TaskLogger instance with cost_data
            nodes: List of HospitalNode objects
            helpers: MetricHelpers instance
        """
        self.task_logger = task_logger
        self.nodes = nodes
        self.helpers = helpers

    @abstractmethod
    def compute(self) -> dict:
        """
        Compute the metric.

        Must be implemented by subclasses.

        Returns:
            Dict with metric results, or {} if data unavailable
        """
        pass

    def _validate(self) -> bool:
        """Check if minimum data available to compute metric."""
        return bool(
            self.task_logger
            and self.task_logger.cost_data
            and self.nodes
        )

    def _get_quadrant(self, location_idx) -> int:
        """Map location_index to quadrant (delegates to helpers)."""
        return self.helpers.get_location_quadrant(location_idx)

    def _stats(self, values: list) -> dict:
        """
        Compute mean/p95/p99/std safely.

        Args:
            values: List of numeric values

        Returns:
            Dict with mean, p95, p99, std, count (zeros if no data)
        """
        if not values:
            return {
                'mean': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'std': 0.0,
                'count': 0,
            }

        arr = np.array(values, dtype=float)
        return {
            'mean': float(arr.mean()),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'std': float(arr.std()),
            'count': len(values),
        }
