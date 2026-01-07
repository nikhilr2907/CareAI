"""
Parser for mock_hospital_config.json format with full SKU, category, and location details.

This parser handles the augmented config format that includes:
- Category-based inventory with SKUs
- Operational locations and shelf IDs
- Floor information
- Item database with properties
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CategoryInventory:
    """Inventory information for a single category at a node."""
    category_name: str
    stock_level: float
    max_stock: float
    num_distinct_skus: int
    items: List[str] = field(default_factory=list)  # List of SKU IDs
    consumption_rate: float = 0.0  # Will be derived or specified


@dataclass
class ItemInfo:
    """Detailed information about a specific SKU."""
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
    """Operational location information for a node."""
    node_id: str
    location_ids: List[str] = field(default_factory=list)
    shelf_ids: List[str] = field(default_factory=list)
    storage_tiers_present: List[str] = field(default_factory=list)


@dataclass
class MockNodeConfig:
    """Node configuration from mock_hospital_config.json."""
    node_id: int
    node_type: str
    name: str
    pos: tuple  # (x, y)
    size: tuple  # (width, height)

    # Construction details
    construction: Dict = field(default_factory=dict)

    # Category-based inventory
    category_inventory: Dict[str, CategoryInventory] = field(default_factory=dict)

    # Operational location info
    floor: int = 0
    operational_location: Optional[OperationalLocation] = None


class MockConfigParser:
    """
    Parser for mock_hospital_config.json format.

    This format includes:
    - Nodes with category-based inventory (iv_therapy, ppe, wound_care, etc.)
    - SKU-level item tracking
    - Operational locations and shelf IDs
    - Floor information
    - Item database with detailed properties
    """

    def __init__(self, config_path: str):
        """
        Load config from JSON file.

        Args:
            config_path: Path to mock_hospital_config.json
        """
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)

        self.name = self.config_data.get('name', 'unnamed')
        self.metadata = self.config_data.get('metadata', {})

        # Parse item database
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

        # Parse nodes
        self.nodes: List[MockNodeConfig] = []
        for node_data in self.config_data.get('nodes', []):
            self.nodes.append(self._parse_node(node_data))

        # Parse edges
        self.edges = self.config_data.get('edges', [])

        # Build category list (all unique categories across nodes)
        self.all_categories = self._extract_all_categories()

        # Build location/shelf ID mappings
        self.location_to_node = self._build_location_mapping()
        self.shelf_to_node = self._build_shelf_mapping()

    def _parse_node(self, node_data: Dict) -> MockNodeConfig:
        """Parse a single node from config."""

        # Parse construction
        construction = node_data.get('construction', {})

        # Parse inventory categories
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

        # Parse operational location
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
        """Extract all unique category names across all nodes."""
        category_set = set()
        for node in self.nodes:
            category_set.update(node.category_inventory.keys())
        return sorted(list(category_set))

    def _build_location_mapping(self) -> Dict[str, int]:
        """Build mapping from location_id to node index."""
        mapping = {}
        for idx, node in enumerate(self.nodes):
            if node.operational_location:
                for loc_id in node.operational_location.location_ids:
                    mapping[loc_id] = idx
        return mapping

    def _build_shelf_mapping(self) -> Dict[str, int]:
        """Build mapping from shelf_id to node index."""
        mapping = {}
        for idx, node in enumerate(self.nodes):
            if node.operational_location:
                for shelf_id in node.operational_location.shelf_ids:
                    mapping[shelf_id] = idx
        return mapping

    def get_node_by_id(self, node_id: int) -> Optional[MockNodeConfig]:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> List[MockNodeConfig]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.node_type == node_type]

    def get_item_info(self, sku_id: str) -> Optional[ItemInfo]:
        """Get item details from database."""
        return self.item_database.get(sku_id)

    def get_category_skus(self, category: str) -> List[str]:
        """Get all SKU IDs for a specific category."""
        skus = set()
        for node in self.nodes:
            if category in node.category_inventory:
                skus.update(node.category_inventory[category].items)
        return sorted(list(skus))

    def get_node_by_location_id(self, location_id: str) -> Optional[int]:
        """Get node index by location_id."""
        return self.location_to_node.get(location_id)

    def get_node_by_shelf_id(self, shelf_id: str) -> Optional[int]:
        """Get node index by shelf_id."""
        return self.shelf_to_node.get(shelf_id)

    def get_total_stock_by_category(self, category: str) -> float:
        """Get total stock across all nodes for a category."""
        total = 0.0
        for node in self.nodes:
            if category in node.category_inventory:
                total += node.category_inventory[category].stock_level
        return total

    def get_total_skus(self) -> int:
        """Get total number of unique SKUs in the system."""
        return len(self.item_database)

    def summarize(self):
        """Print summary of config."""
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


# Example usage
if __name__ == '__main__':
    import os
    import sys

    config_path = 'examples/mock_hospital_config.json'

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Run from project root directory")
        sys.exit(1)

    print("Loading mock hospital configuration...")
    parser = MockConfigParser(config_path)

    # Summarize
    parser.summarize()

    # Example queries
    print("\nExample Queries:")
    print("-" * 70)

    # Get all SKUs in IV therapy
    print("\n1. SKUs in IV Therapy category:")
    iv_skus = parser.get_category_skus('iv_therapy')
    print(f"   Total: {len(iv_skus)} SKUs")
    print(f"   First 5: {iv_skus[:5]}")

    # Get node by location
    print("\n2. Node for location L00001:")
    node_idx = parser.get_node_by_location_id('L00001')
    if node_idx is not None:
        node = parser.nodes[node_idx]
        print(f"   Node {node.node_id}: {node.name}")

    # Get total stock for PPE
    print("\n3. Total PPE stock across all nodes:")
    ppe_stock = parser.get_total_stock_by_category('ppe')
    print(f"   {ppe_stock:.0f} items")

    # Sample SKU details
    print("\n4. Sample SKU details (SKU00000001):")
    sku_info = parser.get_item_info('SKU00000001')
    if sku_info:
        print(f"   Name: {sku_info.name}")
        print(f"   Category: {sku_info.category}")
        print(f"   Family: {sku_info.family}")
        print(f"   Demand Class: {sku_info.demand_class}")
