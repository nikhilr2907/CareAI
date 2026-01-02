from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class HospitalNode:
    # 1. Basic Identification
    node_id: str  # e.g., "Ward_10817" or "Hub_Area_A" [cite: 89, 243]
    node_type: str  # e.g., 'storage', 'corridor', 'recovery', 'hub' [cite: 216, 259, 268]

    # 2. Region Geometry (Bounding Box)
    # CRITICAL: Nodes are spatial REGIONS, not points
    center_x: float  # Center of region (meters)
    center_y: float  # Center of region (meters)
    width: float     # Region width (meters) - e.g., 5.0m for a ward
    height: float    # Region height (meters) - e.g., 4.0m for a ward

    # Physical Constraints
    clearance_m: float = 0.9  # Available space for turning (0.8m - 1.0m ideal) [cite: 504, 505]

    # 3. Storage & Interaction Metadata (Verticality)
    max_reach_height: float = 1.35  # Max height robot can reach (1350mm) [cite: 570, 580]
    unit_height: float = 2.1  # Physical height of the tall modular cabinets (2100mm) [cite: 562, 579]
    has_wash_basin: bool = False  # Flag for restricted parking near clinical sinks [cite: 267, 336]

    # 4. Sensory & Visual Cues (Digital Twin features)
    visibility_zones: dict = field(default_factory=lambda: {
        "lower": [0.5, 1.1], # >= 500mm and < 1100mm [cite: 507, 546]
        "upper": [1.1, 1.5]  # > 1100mm and < 1500mm [cite: 509, 514]
    })
    is_cluttered: bool = False  # Dynamic flag for nursing terminals/trolleys

    # 5. Operational State - Region Occupancy
    current_robot_ids: List[int] = field(default_factory=list)  # Which robots are currently in this region
    max_capacity: int = 1  # DEPRECATED: use area-based calculation instead

    # 6. Inventory Management (for just-in-time delivery)
    stock_level: float = 100.0  # Current inventory items at this location
    consumption_rate: float = 0.0  # Items consumed per hour (0 for non-ward nodes)
    buffer_time: float = 2.0  # Desired hours of buffer before stockout
    max_stock: float = 200.0  # Maximum stock capacity

    def __post_init__(self):
        """Initialize backwards compatibility fields."""
        # For backwards compatibility: x, y point to center
        if not hasattr(self, 'x') or self.x is None:
            self.x = self.center_x
        if not hasattr(self, 'y') or self.y is None:
            self.y = self.center_y

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of this region.
        Returns: (x_min, y_min, x_max, y_max)
        """
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.center_x - half_w,
            self.center_y - half_h,
            self.center_x + half_w,
            self.center_y + half_h
        )

    @property
    def area(self) -> float:
        """Area of this region in square meters."""
        return self.width * self.height

    @property
    def occupancy_count(self) -> int:
        """Number of robots currently in this region."""
        return len(self.current_robot_ids)

    @property
    def occupancy_ratio(self) -> float:
        """
        How crowded this region is (robots per square meter).
        Normalized by assuming ~10 sq meters per robot as comfortable density.
        """
        return self.occupancy_count / max(1.0, self.area / 10.0)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) is inside this region's bounds."""
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def add_robot(self, robot_id: int):
        """Add a robot to this region."""
        if robot_id not in self.current_robot_ids:
            self.current_robot_ids.append(robot_id)

    def remove_robot(self, robot_id: int):
        """Remove a robot from this region."""
        if robot_id in self.current_robot_ids:
            self.current_robot_ids.remove(robot_id)

    @property
    def time_to_stockout(self) -> float:
        """Calculate hours until stock runs out at current consumption rate."""
        if self.consumption_rate <= 0:
            return float('inf')
        return self.stock_level / self.consumption_rate

    @property
    def needs_restock(self) -> bool:
        """Check if location needs restocking based on buffer time."""
        return self.time_to_stockout < self.buffer_time

    @property
    def is_stockout(self) -> bool:
        """Check if location has run out of stock."""
        return self.stock_level <= 0

    @property
    def urgency_level(self) -> int:
        """
        Calculate urgency level (1-5) based on time to stockout.
        5 = critical (< 0.5 hours), 1 = low urgency (> 4 hours)
        """
        tts = self.time_to_stockout
        if tts < 0.5:
            return 5  # Critical
        elif tts < 1.0:
            return 4  # High
        elif tts < 2.0:
            return 3  # Medium
        elif tts < 4.0:
            return 2  # Low-Medium
        else:
            return 1  # Low

    def consume_stock(self, time_delta: float):
        """Reduce stock level based on consumption rate and time elapsed."""
        consumed = self.consumption_rate * time_delta
        self.stock_level = max(0.0, self.stock_level - consumed)

    def restock(self, amount: float):
        """Add stock to this location (called when delivery completes)."""
        self.stock_level = min(self.max_stock, self.stock_level + amount)

