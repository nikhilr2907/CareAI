from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class GraphEdge:
    from_node: str  # Node ID this corridor connects from
    to_node: str    # Node ID this corridor connects to
    edge_id: Optional[str] = None
    mode: Optional[str] = None
    floor_delta: int = 0
    travel_time_model: Optional[str] = None
    constraints: str = ""

    # Corridor Geometry
    distance_m: float = 0.0  # Physical length of corridor (computed from entry/exit points)
    corridor_width: float = 1.9  # Physical width of corridor (meters) [cite: 407]

    # Entry/Exit Points (optional - for detailed corridor geometry)
    entry_point: Optional[Tuple[float, float]] = None  # (x,y) where corridor exits from_node boundary
    exit_point: Optional[Tuple[float, float]] = None   # (x,y) where corridor enters to_node boundary

    # Speed Constraints
    max_v_ms: float = 0.5  # Max speed (1 m/s from specs) [cite: 583]

    # Dynamic States (The 'State' part)
    clutter_level: float = 0.0  # 0.0 (clear) to 1.0 (blocked)
    active_robot_ids: list = field(default_factory=list)  # Robot IDs currently traversing this corridor
    people_count: int = 0  # Estimated number of people in corridor
    approaching_robot_count: int = 0  # Robots with this edge in their planned path
    active_robot_progress: dict = field(default_factory=dict)  # robot_id -> (progress, from_idx, to_idx)

    # DEPRECATED: Use corridor_width instead
    width_m: float = None  # Will be set to corridor_width if not provided

    def __post_init__(self):
        """Initialize backwards compatibility and compute entry/exit points if needed."""
        # Backwards compatibility
        if self.width_m is None:
            self.width_m = self.corridor_width
        elif self.corridor_width == 1.9:  # Default value not overridden
            self.corridor_width = self.width_m

    @property
    def current_weight(self) -> float:
        """
        Heuristic fallback cost for path planning.
        Used only when LearnedEdgeCostModel has insufficient data.
        Once the learned model is active, Dijkstra uses model predictions instead.
        """
        base_cost = self.distance_m / self.max_v_ms
        congestion_penalty = len(self.active_robot_ids) * 1.5
        opposite_dir_penalty = self._count_opposing_robots() * 2.0
        people_penalty = self.people_count * 0.5
        clutter_penalty = self.clutter_level * 10.0
        return base_cost + congestion_penalty + opposite_dir_penalty + people_penalty + clutter_penalty

    def _count_opposing_robots(self) -> int:
        """Count pairs of robots traveling in opposite directions on this edge."""
        if len(self.active_robot_progress) < 2:
            return 0
        directions = {}
        for rid, (prog, fidx, tidx) in self.active_robot_progress.items():
            key = (fidx, tidx)
            directions.setdefault(key, 0)
            directions[key] += 1
        # If there are robots going both ways, count the minority direction
        if len(directions) >= 2:
            counts = list(directions.values())
            return min(counts)
        return 0

    def set_people_count(self, count: int):
        """Set estimated number of people in this corridor."""
        self.people_count = max(0, int(count))

    def add_robot(self, robot_id: int):
        """Add a robot to this corridor (updates congestion)."""
        if robot_id not in self.active_robot_ids:
            self.active_robot_ids.append(robot_id)

    def remove_robot(self, robot_id: int):
        """Remove a robot from this corridor."""
        if robot_id in self.active_robot_ids:
            self.active_robot_ids.remove(robot_id)

    def is_point_on_corridor(self, x: float, y: float, tolerance: float = 1.0) -> bool:
        """
        Check if a point (x, y) is on this corridor.
        Uses distance to line segment between entry and exit points.

        Args:
            x, y: Point coordinates
            tolerance: Max perpendicular distance from corridor centerline (meters)

        Returns:
            True if point is within tolerance of corridor path
        """
        if self.entry_point is None or self.exit_point is None:
            return False

        # Vector from entry to exit
        ex, ey = self.exit_point
        sx, sy = self.entry_point

        # Vector from entry to point
        px, py = x - sx, y - sy
        dx, dy = ex - sx, ey - sy

        # Project point onto corridor line
        corridor_length_sq = dx * dx + dy * dy
        if corridor_length_sq == 0:
            return False

        t = max(0.0, min(1.0, (px * dx + py * dy) / corridor_length_sq))

        # Closest point on corridor
        closest_x = sx + t * dx
        closest_y = sy + t * dy

        # Distance from point to corridor
        dist = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5

        return dist <= tolerance

    def compute_progress(self, x: float, y: float) -> float:
        """
        Calculate how far along the corridor a point is (0.0 = entry, 1.0 = exit).

        Args:
            x, y: Current position

        Returns:
            Progress value between 0.0 and 1.0
        """
        if self.entry_point is None or self.exit_point is None:
            return 0.0

        sx, sy = self.entry_point
        ex, ey = self.exit_point

        # Vector from entry to exit
        dx, dy = ex - sx, ey - sy

        # Vector from entry to point
        px, py = x - sx, y - sy

        # Project onto corridor vector
        corridor_length_sq = dx * dx + dy * dy
        if corridor_length_sq == 0:
            return 0.0

        dot_product = px * dx + py * dy
        progress = dot_product / corridor_length_sq

        return max(0.0, min(1.0, progress))
    
