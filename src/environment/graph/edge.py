import math
import random
import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

# travel_time_model strings look like:
# "lift_wait_min~LogNormal(mu=1.6,sigma=0.5); ride_s=25*floor_delta+15"
_LIFT_WAIT_PATTERN = re.compile(r"LogNormal\(mu=([\d.]+)\s*,\s*sigma=([\d.]+)\)")
_LIFT_RIDE_PATTERN = re.compile(r"ride_s\s*=\s*([\d.]+)\s*\*\s*floor_delta\s*\+\s*([\d.]+)")
_DEFAULT_LIFT_WAIT_MU = 1.6
_DEFAULT_LIFT_WAIT_SIGMA = 0.5
_DEFAULT_RIDE_COEF = 25.0
_DEFAULT_RIDE_INTERCEPT = 15.0

# Ablation flag: hides equipment/emergency state from the heuristic/model inputs only.
INCLUDE_EQUIPMENT_EMERGENCY_FEATURES = True


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
    active_robot_ids: list[int] = field(
        default_factory=list
    )  # Robot IDs currently traversing this corridor
    has_patient_bed: bool = False  # True if a bed is currently passing [cite: 421]
    people_count: int = 0  # Estimated number of people in corridor
    approaching_robot_count: int = 0  # Robots with this edge in their planned path
    active_robot_progress: dict[int, tuple[float, int | None, int | None]] = field(
        default_factory=dict
    )  # robot_id -> (progress, from_idx, to_idx)

    # DEPRECATED: Use corridor_width instead
    width_m: float | None = None  # Will be set to corridor_width if not provided

    congestion_model: str = "density"  # "legacy" (capacity=2) or "density" (see effective_speed)

    equipment_counts: dict[str, int] = field(default_factory=dict)  # category -> count, see _EQUIPMENT_SPECS
    equipment_base_rate: dict[str, float] = field(
        default_factory=dict
    )  # category -> Poisson arrivals/hour on this edge; falls back to a hospital-wide default if unset

    emergency_active: bool = False  # rare patient-movement emergency in progress -- other traffic yields
    emergency_rate_per_hour: float = 0.01  # Poisson rate for this edge specifically

    def __post_init__(self) -> None:
        """Initialize backwards compatibility between width_m and corridor_width."""
        if self.width_m is None:
            self.width_m = self.corridor_width
        elif self.corridor_width == 1.9:  # Default value not overridden
            self.corridor_width = self.width_m

    @property
    def current_weight(self) -> float:
        """Heuristic fallback cost for path planning, used only until the learned edge cost model has enough data."""
        if self.is_lift():
            return self.expected_transit_duration_s()

        if self.emergency_active and INCLUDE_EQUIPMENT_EMERGENCY_FEATURES:
            return self.distance_m / (self.max_v_ms * self._MIN_SPEED_FRACTION)

        base_cost = self.distance_m / self.max_v_ms
        congestion_penalty = len(self.active_robot_ids) * 1.5
        opposite_dir_penalty = self._count_opposing_robots() * 2.0
        people_penalty = self.people_count * 0.5
        clutter_penalty = self.clutter_level * 10.0
        equipment_penalty = (
            self.equipment_equivalents() * 1.5 if INCLUDE_EQUIPMENT_EMERGENCY_FEATURES else 0.0
        )
        bed_penalty = self._BED_EQUIVALENCE * 1.5 if self.has_patient_bed else 0.0
        return (
            base_cost
            + congestion_penalty
            + opposite_dir_penalty
            + people_penalty
            + clutter_penalty
            + equipment_penalty
            + bed_penalty
        )

    def _count_opposing_robots(self) -> int:
        """Count pairs of robots traveling in opposite directions on this edge."""
        if len(self.active_robot_progress) < 2:
            return 0
        directions: dict[tuple[int | None, int | None], int] = {}
        for rid, (prog, fidx, tidx) in self.active_robot_progress.items():
            key = (fidx, tidx)
            directions.setdefault(key, 0)
            directions[key] += 1
        # If there are robots going both ways, count the minority direction
        if len(directions) >= 2:
            counts = list(directions.values())
            return min(counts)
        return 0

    def set_people_count(self, count: int) -> None:
        self.people_count = max(0, int(count))

    def set_equipment_count(self, category: str, count: int) -> None:
        if category not in self._EQUIPMENT_SPECS:
            raise ValueError(f"Unknown equipment category: {category}")
        self.equipment_counts[category] = max(0, int(count))

    _PERSON_EQUIVALENCE = 1.8  # a person's disruption to flow vs. one robot
    _BED_WIDTH_M = 1.0  # bed + porter's clearance; eaten out of corridor_width directly
    _BED_EQUIVALENCE = 4.0  # bed + porter vs. one robot -- wide, slow, can't be routed around
    _JAM_DENSITY_PER_M2 = 1.5  # occupant-equivalents per m^2 at which flow stops
    _MIN_SPEED_FRACTION = 0.05

    # category -> (width_m eaten from the corridor, occupant-equivalent weight vs. one robot)
    _EQUIPMENT_SPECS: ClassVar[dict[str, tuple[float, float]]] = {
        # Tier 1 -- narrow, person-scale disruption
        "drug_trolley": (0.5, 1.5),
        "wheelchair": (0.65, 1.5),
        "cleaning_cart": (0.6, 1.5),
        # Tier 2 -- wide, exceeds a person, approaches bed-scale
        "linen_cart": (0.9, 2.5),
        "meal_trolley": (0.8, 2.5),
        "imaging_unit": (0.85, 3.0),
        "pallet_jack": (0.9, 3.0),
        # Tier 3 -- wide and long, exceeds a standard bed
        "bariatric_bed": (1.3, 5.5),
    }

    def _equipment_width_m(self) -> float:
        return sum(
            self._EQUIPMENT_SPECS[category][0]
            for category, count in self.equipment_counts.items()
            if count > 0
        )

    def equipment_equivalents(self) -> float:
        return sum(
            self._EQUIPMENT_SPECS[category][1] * count
            for category, count in self.equipment_counts.items()
        )

    def effective_speed(self) -> float:
        """Speed a robot can travel at on this edge right now, given who/what is on it."""
        if self.congestion_model == "legacy":
            capacity = 2
            num_active = max(1, len(self.active_robot_ids))
            if num_active > capacity:
                return self.max_v_ms * max(0.2, capacity / num_active)
            return self.max_v_ms

        if self.emergency_active:
            # Right-of-way: other traffic yields rather than negotiating space.
            return self.max_v_ms * self._MIN_SPEED_FRACTION

        occupant_equivalents = (
            len(self.active_robot_ids)
            + self.people_count * self._PERSON_EQUIVALENCE
            + self.equipment_equivalents()
        )
        if self.has_patient_bed:
            occupant_equivalents += self._BED_EQUIVALENCE
        speed_fraction = max(
            self._MIN_SPEED_FRACTION, 1.0 - min(occupant_equivalents / self.jam_density(), 1.0)
        )
        return self.max_v_ms * speed_fraction

    def jam_density(self) -> float:
        """Occupant-equivalents this corridor's area can hold before effective_speed bottoms out."""
        bed_width = self._BED_WIDTH_M if self.has_patient_bed else 0.0
        eaten_width = bed_width + self._equipment_width_m()
        effective_width = max(0.3, (self.corridor_width - eaten_width) * (1.0 - min(self.clutter_level, 0.95)))
        area_m2 = effective_width * max(self.distance_m, 0.5)
        return area_m2 * self._JAM_DENSITY_PER_M2

    def congestion_factor(self) -> float:
        """How congested this edge is right now, in [0, 1] -- the same signal effective_speed acts on."""
        if self.max_v_ms <= 0:
            return 0.0
        return 1.0 - (self.effective_speed() / self.max_v_ms)

    def is_lift(self) -> bool:
        return self.mode in ("lift", "elevator")

    def _lift_params(self) -> tuple[float, float, float, float]:
        """Parse travel_time_model into (wait_mu, wait_sigma, ride_coef, ride_intercept), or defaults."""
        mu, sigma = _DEFAULT_LIFT_WAIT_MU, _DEFAULT_LIFT_WAIT_SIGMA
        coef, intercept = _DEFAULT_RIDE_COEF, _DEFAULT_RIDE_INTERCEPT
        if self.travel_time_model:
            wait_match = _LIFT_WAIT_PATTERN.search(self.travel_time_model)
            if wait_match:
                mu, sigma = float(wait_match.group(1)), float(wait_match.group(2))
            ride_match = _LIFT_RIDE_PATTERN.search(self.travel_time_model)
            if ride_match:
                coef, intercept = float(ride_match.group(1)), float(ride_match.group(2))
        return mu, sigma, coef, intercept

    def sample_transit_duration_s(self) -> float:
        """Total time to ride this lift edge: wait-for-lift (LogNormal, minutes) + ride (linear in floor_delta)."""
        mu, sigma, coef, intercept = self._lift_params()
        wait_s = random.lognormvariate(mu, sigma) * 60.0
        ride_s = coef * abs(self.floor_delta) + intercept
        return wait_s + ride_s

    def expected_transit_duration_s(self) -> float:
        """Closed-form expected transit time, for path-cost planning (not a sampled draw)."""
        mu, sigma, coef, intercept = self._lift_params()
        expected_wait_s = math.exp(mu + sigma**2 / 2.0) * 60.0
        ride_s = coef * abs(self.floor_delta) + intercept
        return expected_wait_s + ride_s

    def add_robot(self, robot_id: int) -> None:
        """Add a robot to this corridor; updates congestion as a side effect."""
        if robot_id not in self.active_robot_ids:
            self.active_robot_ids.append(robot_id)

    def remove_robot(self, robot_id: int) -> None:
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

        dist: float = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5

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
