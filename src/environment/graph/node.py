from dataclasses import dataclass, field

@dataclass
class HospitalNode:
    # 1. Basic Identification
    node_id: str  # e.g., "Ward_10817" or "Hub_Area_A" [cite: 89, 243]
    node_type: str  # e.g., 'storage', 'corridor', 'recovery', 'hub' [cite: 216, 259, 268]

    # 2. Geometric Metadata (From Floorplan/Specs)
    x: float  # Cartesian coordinates for distance calculation
    y: float
    width_m: float  # Corridor/Door width (e.g., 1000mm for doors) [cite: 407]
    clearance_m: float  # Available space for turning (0.8m - 1.0m ideal) [cite: 504, 505]

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

    # 5. Operational State
    current_occupancy: int = 0
    max_capacity: int = 1  # How many robots can fit in this specific zone

    # 6. Inventory Management (for just-in-time delivery)
    stock_level: float = 100.0  # Current inventory items at this location
    consumption_rate: float = 0.0  # Items consumed per hour (0 for non-ward nodes)
    buffer_time: float = 2.0  # Desired hours of buffer before stockout
    max_stock: float = 200.0  # Maximum stock capacity

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

