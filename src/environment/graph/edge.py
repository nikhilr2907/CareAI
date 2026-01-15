from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class HospitalEdge:
    from_node: str  # Node ID this corridor connects from
    to_node: str    # Node ID this corridor connects to

    # Corridor Geometry
    distance_m: float  # Physical length of corridor (computed from entry/exit points)
    corridor_width: float = 1.9  # Physical width of corridor (meters) [cite: 407]

    # Entry/Exit Points (optional - for detailed corridor geometry)
    entry_point: Optional[Tuple[float, float]] = None  # (x,y) where corridor exits from_node boundary
    exit_point: Optional[Tuple[float, float]] = None   # (x,y) where corridor enters to_node boundary

    # Speed Constraints
    max_v_ms: float = 1.0  # Max speed (1 m/s from specs) [cite: 583]

    # Dynamic States (The 'State' part)
    clutter_level: float = 0.0  # 0.0 (clear) to 1.0 (blocked)
    active_robot_ids: list = field(default_factory=list)  # Robot IDs currently traversing this corridor
    has_patient_bed: bool = False  # True if a bed is currently passing [cite: 421]
    people_count: int = 0  # Estimated number of people in corridor

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
        Calculates dynamic cost for path planning.
        Accounts for:
        - Base travel time
        - Congestion from other robots
        - Static clutter
        - Patient bed obstacles
        """
        base_cost = self.distance_m / self.max_v_ms
        congestion_penalty = len(self.active_robot_ids) * 1.5
        people_penalty = self.people_count * 0.5
        clutter_penalty = self.clutter_level * 10
        bed_penalty = 15.0 if self.has_patient_bed else 0.0
        return base_cost + congestion_penalty + people_penalty + clutter_penalty + bed_penalty

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
    
