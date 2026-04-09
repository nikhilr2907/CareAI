from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

@dataclass
class GraphNode:
    # 1. Basic Identification
    node_id: str
    node_type: str  # 'storage', 'recovery', 'hub'

    # 2. Region Geometry (Bounding Box)
    center_x: float
    center_y: float
    width: float
    height: float

    # 3. Operational State
    current_robot_ids: List[int] = field(default_factory=list)

    # 4. Inventory
    stock_level: float = 0.0
    consumption_rate: float = 0.0
    max_stock: float = 0.0
    category_inventory: dict = field(default_factory=dict)
    sku_inventory: Dict[str, Dict] = field(default_factory=dict)

    # 5. Location metadata
    floor: int = 0
    consumption_enabled: bool = False

    # 6. ILC demand scaling
    foot_traffic_weight: float = 0.0   # 0–1; replaces served_beds
    location_tag: Optional[str] = None  # e.g. "seating_area", "far_side", "home_base"

    def __post_init__(self):
        self.x = self.center_x
        self.y = self.center_y

    # ── Geometry ────────────────────────────────────────────────────────────────

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.center_x - half_w, self.center_y - half_h,
                self.center_x + half_w, self.center_y + half_h)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float) -> bool:
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    # ── Occupancy ───────────────────────────────────────────────────────────────

    @property
    def occupancy_count(self) -> int:
        return len(self.current_robot_ids)

    @property
    def occupancy_ratio(self) -> float:
        return self.occupancy_count / max(1.0, self.area / 10.0)

    def add_robot(self, robot_id: int):
        if robot_id not in self.current_robot_ids:
            self.current_robot_ids.append(robot_id)

    def remove_robot(self, robot_id: int):
        if robot_id in self.current_robot_ids:
            self.current_robot_ids.remove(robot_id)

    # ── Inventory state ─────────────────────────────────────────────────────────

    @property
    def time_to_stockout(self) -> float:
        if self.sku_inventory:
            rate = self.consumption_rate
            if rate <= 0:
                return float('inf')
            return self.stock_level / rate
        if self.consumption_rate <= 0:
            return float('inf')
        return self.stock_level / self.consumption_rate

    @property
    def needs_restock(self) -> bool:
        if self.sku_inventory:
            for data in self.sku_inventory.values():
                if float(data.get('stock', 0)) <= float(data.get('reorder', 0)):
                    return True
            return False
        return self.time_to_stockout < 2.0

    @property
    def is_stockout(self) -> bool:
        if self.sku_inventory:
            return any(d.get('stock', 0.0) <= 0 for d in self.sku_inventory.values())
        return self.stock_level <= 0

    @property
    def urgency_level(self) -> int:
        tts = self.time_to_stockout
        if tts < 0.5:   return 5
        if tts < 1.0:   return 4
        if tts < 2.0:   return 3
        if tts < 4.0:   return 2
        return 1

    # ── SKU operations ──────────────────────────────────────────────────────────

    def get_sku_stock(self, sku_id: str) -> float:
        return float(self.sku_inventory.get(sku_id, {}).get('stock', 0.0))

    def get_sku_reorder_point(self, sku_id: str) -> float:
        return float(self.sku_inventory.get(sku_id, {}).get('reorder', 0.0))

    def get_sku_par_level(self, sku_id: str) -> float:
        return float(self.sku_inventory.get(sku_id, {}).get('par', 0.0))

    def get_sku_max_level(self, sku_id: str) -> float:
        return float(self.sku_inventory.get(sku_id, {}).get('max', 0.0))

    def consume_sku(self, sku_id: str, amount: float):
        if sku_id in self.sku_inventory:
            current = float(self.sku_inventory[sku_id].get('stock', 0.0))
            self.sku_inventory[sku_id]['stock'] = max(0.0, current - amount)

    def restock_sku(self, sku_id: str, amount: float):
        if sku_id in self.sku_inventory:
            current = float(self.sku_inventory[sku_id].get('stock', 0.0))
            max_level = float(self.sku_inventory[sku_id].get('max', current + amount))
            self.sku_inventory[sku_id]['stock'] = min(max_level, current + amount)

    def recalc_category_inventory(self):
        if not self.sku_inventory:
            return
        aggregates = {}
        for sku_id, data in self.sku_inventory.items():
            category = data.get('category')
            if category is None:
                continue
            if category not in aggregates:
                aggregates[category] = {'stock': 0.0, 'max': 0.0, 'num_skus': 0, 'rate': 0.0}
            aggregates[category]['stock'] += float(data.get('stock', 0.0))
            aggregates[category]['max']   += float(data.get('max', 0.0))
        for category, agg in aggregates.items():
            agg['num_skus'] = sum(
                1 for d in self.sku_inventory.values() if d.get('category') == category
            )
        self.category_inventory = aggregates

    # ── Category queries ────────────────────────────────────────────────────────

    def get_category_stock_level(self, category: str) -> float:
        return self.category_inventory.get(category, {}).get('stock', 0.0)

    def get_category_max_stock(self, category: str) -> float:
        return self.category_inventory.get(category, {}).get('max', 0.0)

    def get_category_num_skus(self, category: str) -> int:
        return self.category_inventory.get(category, {}).get('num_skus', 0)

    def get_total_num_skus(self) -> int:
        return sum(cat.get('num_skus', 0) for cat in self.category_inventory.values())

    def get_category_consumption_rate(self, category: str) -> float:
        return self.category_inventory.get(category, {}).get('rate', 0.0)

    def get_category_time_to_stockout(self, category: str) -> float:
        stock = self.get_category_stock_level(category)
        rate  = self.get_category_consumption_rate(category)
        if rate <= 0:
            return float('inf')
        return stock / rate

    def get_all_categories(self) -> List[str]:
        return list(self.category_inventory.keys())
