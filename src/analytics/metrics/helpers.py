"""Shared spatial utilities for flow metrics."""


class MetricHelpers:
    """Maps location indices to spatial quadrants on 2x2 grid."""

    GRID_SIZE = 2  # 2x2 = 4 quadrants

    def __init__(self, nodes):
        """
        Args:
            nodes: List of HospitalNode objects
        """
        self.nodes = nodes
        self._bounds = None

    def get_location_quadrant(self, location_idx: int) -> int:
        """
        Map location_index → quadrant 0-3.

        Grid layout:
            2 (NW) | 3 (NE)
            -------|-------
            0 (SW) | 1 (SE)

        Args:
            location_idx: Index into nodes list

        Returns:
            Quadrant ID 0-3, or 0 if index invalid
        """
        if (not self.nodes or location_idx is None or
            location_idx < 0 or location_idx >= len(self.nodes)):
            return 0

        node = self.nodes[location_idx]
        bounds = self._get_bounds()

        x_mid = (bounds['x_min'] + bounds['x_max']) / 2
        y_mid = (bounds['y_min'] + bounds['y_max']) / 2

        # Determine column (0=left/W, 1=right/E)
        col = 0 if node.center_x < x_mid else 1
        # Determine row (0=bottom/S, 1=top/N)
        row = 0 if node.center_y < y_mid else 1

        # Map to quadrant: row*2 + col
        return row * 2 + col

    def _get_bounds(self) -> dict:
        """Lazily compute floor extent from node centers."""
        if self._bounds is None:
            xs = [n.center_x for n in self.nodes]
            ys = [n.center_y for n in self.nodes]
            self._bounds = {
                'x_min': min(xs) - 5,
                'x_max': max(xs) + 5,
                'y_min': min(ys) - 5,
                'y_max': max(ys) + 5,
            }
        return self._bounds

    def quadrant_label(self, q: int) -> str:
        """Human-readable label for quadrant."""
        labels = {0: 'SW', 1: 'SE', 2: 'NW', 3: 'NE'}
        return labels.get(q, f'Q{q}')
