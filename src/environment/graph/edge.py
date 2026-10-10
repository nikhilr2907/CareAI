from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GraphEdge:
    from_node: str  # Node ID this corridor connects from
    to_node: str  # Node ID this corridor connects to
    edge_id: Optional[str] = None
    mode: Optional[str] = None
    floor_delta: int = 0
    travel_time_model: Optional[str] = None
    constraints: str = ""

    distance_m: float = 0.0  # Physical length of corridor (computed from entry/exit points)
    corridor_width: float = 1.9  # Physical width of corridor (meters) [cite: 407]

    entry_point: Optional[Tuple[float, float]] = (
        None  # (x,y) where corridor exits from_node boundary
    )
    exit_point: Optional[Tuple[float, float]] = None  # (x,y) where corridor enters to_node boundary

    max_v_ms: float = 0.5  # Max speed (1 m/s from specs) [cite: 583]

    clutter_level: float = 0.0  # 0.0 (clear) to 1.0 (blocked)
    active_robot_ids: list = field(
        default_factory=list
    )  # Robot IDs currently traversing this corridor
    has_patient_bed: bool = False  # True if a bed is currently passing [cite: 421]
    people_count: int = 0  # Estimated number of people in corridor
    approaching_robot_count: int = 0  # Robots with this edge in their planned path
    active_robot_progress: dict = field(
        default_factory=dict
    )  # robot_id -> (progress, from_idx, to_idx)

    # DEPRECATED: Use corridor_width instead
    width_m: float = None  # Will be set to corridor_width if not provided

    def __post_init__(self):
        """Initialize backwards compatibility between width_m and corridor_width."""
        if self.width_m is None:
            self.width_m = self.corridor_width
        elif self.corridor_width == 1.9:  # Default value not overridden
            self.corridor_width = self.width_m

    @property
    def current_weight(self) -> float:
        """Heuristic fallback cost for path planning, used only until the learned edge cost model has enough data."""
        base_cost = self.distance_m / self.max_v_ms
        congestion_penalty = len(self.active_robot_ids) * 1.5
        opposite_dir_penalty = self._count_opposing_robots() * 2.0
        people_penalty = self.people_count * 0.5
        clutter_penalty = self.clutter_level * 10.0
        return (
            base_cost + congestion_penalty + opposite_dir_penalty + people_penalty + clutter_penalty
        )

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
        self.people_count = max(0, int(count))

    def add_robot(self, robot_id: int):
        """Add a robot to this corridor; updates congestion as a side effect."""
        if robot_id not in self.active_robot_ids:
            self.active_robot_ids.append(robot_id)

    def remove_robot(self, robot_id: int):
        if robot_id in self.active_robot_ids:
            self.active_robot_ids.remove(robot_id)

    def is_point_on_corridor(self, x: float, y: float, tolerance: float = 1.0) -> bool:
        """Check if (x, y) is within tolerance of the line segment between entry and exit points."""
        if self.entry_point is None or self.exit_point is None:
            return False

        ex, ey = self.exit_point
        sx, sy = self.entry_point

        px, py = x - sx, y - sy
        dx, dy = ex - sx, ey - sy

        corridor_length_sq = dx * dx + dy * dy
        if corridor_length_sq == 0:
            return False

        t = max(0.0, min(1.0, (px * dx + py * dy) / corridor_length_sq))

        closest_x = sx + t * dx
        closest_y = sy + t * dy

        dist = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5

        return dist <= tolerance

    def compute_progress(self, x: float, y: float) -> float:
        """Calculate how far along the corridor (x, y) is, from 0.0 (entry) to 1.0 (exit)."""
        if self.entry_point is None or self.exit_point is None:
            return 0.0

        sx, sy = self.entry_point
        ex, ey = self.exit_point

        dx, dy = ex - sx, ey - sy

        px, py = x - sx, y - sy

        corridor_length_sq = dx * dx + dy * dy
        if corridor_length_sq == 0:
            return 0.0

        dot_product = px * dx + py * dy
        progress = dot_product / corridor_length_sq

        return max(0.0, min(1.0, progress))
