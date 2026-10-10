from collections.abc import Sequence
from typing import Protocol


class SpatialNode(Protocol):
    @property
    def center_x(self) -> float: ...

    @property
    def center_y(self) -> float: ...


class MetricHelpers:
    """Maps location indices to spatial quadrants on 2x2 grid."""

    GRID_SIZE = 2  # 2x2 = 4 quadrants

    def __init__(self, nodes: Sequence[SpatialNode]) -> None:
        self.nodes = nodes
        self._bounds: dict[str, float] | None = None

    def get_location_quadrant(self, location_idx: int | None) -> int:
        """Map a location index to its spatial quadrant: 0=SW, 1=SE, 2=NW, 3=NE (0 if invalid)."""
        if (
            not self.nodes
            or location_idx is None
            or location_idx < 0
            or location_idx >= len(self.nodes)
        ):
            return 0

        node = self.nodes[location_idx]
        bounds = self._get_bounds()

        x_mid = (bounds["x_min"] + bounds["x_max"]) / 2
        y_mid = (bounds["y_min"] + bounds["y_max"]) / 2

        col = 0 if node.center_x < x_mid else 1
        row = 0 if node.center_y < y_mid else 1

        return row * 2 + col

    def _get_bounds(self) -> dict[str, float]:
        """Lazily compute floor extent from node centers."""
        if self._bounds is None:
            xs = [n.center_x for n in self.nodes]
            ys = [n.center_y for n in self.nodes]
            self._bounds = {
                "x_min": min(xs) - 5,
                "x_max": max(xs) + 5,
                "y_min": min(ys) - 5,
                "y_max": max(ys) + 5,
            }
        return self._bounds

    def quadrant_label(self, q: int) -> str:
        """Human-readable label for quadrant."""
        labels = {0: "SW", 1: "SE", 2: "NW", 3: "NE"}
        return labels.get(q, f"Q{q}")
