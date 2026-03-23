"""Parser for the mock hospital config format."""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CategoryInventory:
    """Category inventory for a node."""
    category_name: str
    stock_level: float
    max_stock: float
    num_distinct_skus: int
    items: List[str] = field(default_factory=list)
    consumption_rate: float = 0.0


@dataclass
class ItemInfo:
    """SKU metadata."""
    sku_id: str
    name: str
    category: str
    family: str
    substitute_group_id: Optional[str] = None
    demand_class: str = "medium"
    units_per_pack: int = 1
    storage_requirements: List[str] = field(default_factory=list)
    weight_g: float = 0.0
    volume_ml: float = 0.0


@dataclass
class OperationalLocation:
    """Operational location data for a node."""
    node_id: str
    location_ids: List[str] = field(default_factory=list)
    shelf_ids: List[str] = field(default_factory=list)
    storage_tiers_present: List[str] = field(default_factory=list)


@dataclass
class MockNodeConfig:
    """Mock-format node configuration."""
    node_id: int
    node_type: str
    name: str
    pos: tuple
    size: tuple
    construction: Dict = field(default_factory=dict)
    category_inventory: Dict[str, CategoryInventory] = field(default_factory=dict)
    floor: int = 0
    operational_location: Optional[OperationalLocation] = None


class MockConfigParser:
    """Parse the mock hospital config format."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)

        self.name = self.config_data.get('name', 'unnamed')
        self.metadata = self.config_data.get('metadata', {})

        self.item_database: Dict[str, ItemInfo] = {}
        if 'item_database' in self.config_data:
            for sku_id, item_data in self.config_data['item_database'].items():
                self.item_database[sku_id] = ItemInfo(
                    sku_id=sku_id,
                    name=item_data.get('name', ''),
                    category=item_data.get('category', 'general'),
                    family=item_data.get('family', 'general'),
                    substitute_group_id=item_data.get('substitute_group_id'),
                    demand_class=item_data.get('demand_class', 'medium'),
                    units_per_pack=item_data.get('units_per_pack', 1),
                    storage_requirements=item_data.get('storage_requirements', []),
                    weight_g=item_data.get('weight_g', 0.0),
                    volume_ml=item_data.get('volume_ml', 0.0)
                )

        self.nodes: List[MockNodeConfig] = []
        for node_data in self.config_data.get('nodes', []):
            self.nodes.append(self._parse_node(node_data))

        self.edges = self.config_data.get('edges', [])
        self.all_categories = self._extract_all_categories()
        self.location_to_node = self._build_location_mapping()
        self.shelf_to_node = self._build_shelf_mapping()

    def _parse_node(self, node_data: Dict) -> MockNodeConfig:
        construction = node_data.get('construction', {})
        category_inventory = {}
        if 'inventory' in node_data and 'categories' in node_data['inventory']:
            for cat_name, cat_data in node_data['inventory']['categories'].items():
                category_inventory[cat_name] = CategoryInventory(
                    category_name=cat_name,
                    stock_level=cat_data.get('stock_level', 0.0),
                    max_stock=cat_data.get('max_stock', 100.0),
                    num_distinct_skus=cat_data.get('num_distinct_skus', 0),
                    items=cat_data.get('items', []),
                    consumption_rate=cat_data.get('consumption_rate', 0.0)
                )

        operational_location = None
        if 'operational_locations' in node_data:
            op_loc_data = node_data['operational_locations']
            operational_location = OperationalLocation(
                node_id=op_loc_data.get('node_id', f"NODE_{node_data['id']}"),
                location_ids=op_loc_data.get('location_ids', []),
                shelf_ids=op_loc_data.get('shelf_ids', []),
                storage_tiers_present=op_loc_data.get('storage_tiers_present', [])
            )

        return MockNodeConfig(
            node_id=node_data['id'],
            node_type=node_data['type'],
            name=node_data.get('name', f"Node_{node_data['id']}"),
            pos=tuple(node_data['pos']),
            size=tuple(node_data['size']),
            construction=construction,
            category_inventory=category_inventory,
            floor=node_data.get('floor', 0),
            operational_location=operational_location
        )

    def _extract_all_categories(self) -> List[str]:
        category_set = set()
        for node in self.nodes:
            category_set.update(node.category_inventory.keys())
        return sorted(list(category_set))

    def _build_location_mapping(self) -> Dict[str, int]:
        mapping = {}
        for idx, node in enumerate(self.nodes):
            if node.operational_location:
                for loc_id in node.operational_location.location_ids:
                    mapping[loc_id] = idx
        return mapping

    def _build_shelf_mapping(self) -> Dict[str, int]:
        mapping = {}
        for idx, node in enumerate(self.nodes):
            if node.operational_location:
                for shelf_id in node.operational_location.shelf_ids:
                    mapping[shelf_id] = idx
        return mapping

    def get_node_by_id(self, node_id: int) -> Optional[MockNodeConfig]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> List[MockNodeConfig]:
        return [node for node in self.nodes if node.node_type == node_type]

    def get_item_info(self, sku_id: str) -> Optional[ItemInfo]:
        return self.item_database.get(sku_id)

    def get_category_skus(self, category: str) -> List[str]:
        skus = set()
        for node in self.nodes:
            if category in node.category_inventory:
                skus.update(node.category_inventory[category].items)
        return sorted(list(skus))

    def get_node_by_location_id(self, location_id: str) -> Optional[int]:
        return self.location_to_node.get(location_id)

    def get_node_by_shelf_id(self, shelf_id: str) -> Optional[int]:
        return self.shelf_to_node.get(shelf_id)

    def get_total_stock_by_category(self, category: str) -> float:
        total = 0.0
        for node in self.nodes:
            if category in node.category_inventory:
                total += node.category_inventory[category].stock_level
        return total

    def get_total_skus(self) -> int:
        return len(self.item_database)

    def summarize(self):
        print(f"\n{'='*70}")
        print(f"Config: {self.name}")
        print(f"Version: {self.metadata.get('version', 'N/A')}")
        print(f"{'='*70}")

        print(f"\nMetadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")

        print(f"\nNodes ({len(self.nodes)}):")
        for node in self.nodes:
            print(f"\n  [{node.node_id}] {node.name} ({node.node_type})")
            print(f"    Position: {node.pos}, Size: {node.size}, Floor: {node.floor}")

            if node.operational_location:
                print(f"    Location IDs: {node.operational_location.location_ids}")
                print(f"    Shelf IDs: {node.operational_location.shelf_ids}")

            if node.category_inventory:
                print(f"    Categories ({len(node.category_inventory)}):")
                for cat_name, cat in node.category_inventory.items():
                    print(f"      - {cat_name}: {cat.stock_level}/{cat.max_stock} "
                          f"({cat.num_distinct_skus} SKUs)")

        print(f"\nItem Database: {len(self.item_database)} SKUs")
        print(f"Categories: {len(self.all_categories)} - {', '.join(self.all_categories)}")
        print(f"Edges: {len(self.edges)}")
        print(f"{'='*70}\n")
