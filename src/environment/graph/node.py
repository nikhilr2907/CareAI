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
    consumption_rate: float = 0.0  # BASE consumption rate (items/hour) - may vary with time
    buffer_time: float = 2.0  # Desired hours of buffer before stockout
    max_stock: float = 200.0  # Maximum stock capacity

    # Variable consumption (optional)
    _variable_consumption_enabled: bool = False
    _consumption_pattern_id: Optional[str] = None  # Reference to pattern in consumption system

    # 7. Category-Based Inventory (Enhanced for SKU tracking)
    # Maps category_name → {stock_level, max_stock, num_skus, consumption_rate}
    category_inventory: dict = field(default_factory=dict)
    # Example: {'IV_Therapy': {'stock': 50, 'max': 100, 'num_skus': 12, 'rate': 5.0}}

    # Operational location and shelf info (from extended configs)
    location_id: Optional[str] = None  # e.g., "L00001"
    shelf_ids: List[str] = field(default_factory=list)  # e.g., ["SH000001", "SH000002"]
    floor: int = 1  # Floor number in hospital

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

    def consume_stock(self, time_delta: float, current_rate: Optional[float] = None):
        """
        Reduce stock level based on consumption rate and time elapsed.

        Args:
            time_delta: Time elapsed in hours
            current_rate: Optional override rate (e.g., from variable consumption system)
                         If None, uses self.consumption_rate
        """
        rate = current_rate if current_rate is not None else self.consumption_rate
        consumed = rate * time_delta
        self.stock_level = max(0.0, self.stock_level - consumed)

    def restock(self, amount: float):
        """Add stock to this location (called when delivery completes)."""
        self.stock_level = min(self.max_stock, self.stock_level + amount)

    def enable_variable_consumption(self, pattern_id: Optional[str] = None):
        """
        Enable variable consumption for this node.

        Args:
            pattern_id: ID to use in consumption system (defaults to node_id)
        """
        self._variable_consumption_enabled = True
        self._consumption_pattern_id = pattern_id or self.node_id

    def disable_variable_consumption(self):
        """Disable variable consumption (use fixed rate)."""
        self._variable_consumption_enabled = False

    def get_current_rate(
        self,
        consumption_system=None,
        current_time: float = 0.0,
        sim_start_day: int = 0,
        day_type=None
    ) -> float:
        """
        Get current consumption rate (variable or fixed).

        Args:
            consumption_system: VariableConsumptionSystem instance (if using variable rates)
            current_time: Current simulation time (seconds)
            sim_start_day: Day simulation started (0=Monday)
            day_type: DayType enum (if using variable rates)

        Returns:
            Current consumption rate (items/hour)
        """
        if self._variable_consumption_enabled and consumption_system is not None:
            return consumption_system.get_current_rate(
                node_id=self._consumption_pattern_id,
                current_time=current_time,
                sim_start_day=sim_start_day,
                day_type=day_type,
                node_type=self.node_type
            )
        else:
            return self.consumption_rate

    # ========================================================================
    # Category-Based Inventory Methods
    # ========================================================================

    def get_category_stock_level(self, category: str) -> float:
        """Get current stock level for a category."""
        if category in self.category_inventory:
            return self.category_inventory[category].get('stock', 0.0)
        return 0.0

    def get_category_max_stock(self, category: str) -> float:
        """Get max stock capacity for a category."""
        if category in self.category_inventory:
            return self.category_inventory[category].get('max', 0.0)
        return 0.0

    def get_category_num_skus(self, category: str) -> int:
        """Get number of distinct SKUs in a category."""
        if category in self.category_inventory:
            return self.category_inventory[category].get('num_skus', 0)
        return 0

    def get_total_num_skus(self) -> int:
        """Get total number of distinct SKUs across all categories."""
        return sum(cat.get('num_skus', 0) for cat in self.category_inventory.values())

    def get_category_consumption_rate(self, category: str) -> float:
        """Get consumption rate for a category."""
        if category in self.category_inventory:
            return self.category_inventory[category].get('rate', 0.0)
        return 0.0

    def get_category_time_to_stockout(self, category: str) -> float:
        """Calculate time to stockout for a specific category."""
        stock = self.get_category_stock_level(category)
        rate = self.get_category_consumption_rate(category)
        if rate <= 0:
            return float('inf')
        return stock / rate

    def get_all_categories(self) -> List[str]:
        """Get list of all category names at this node."""
        return list(self.category_inventory.keys())

