"""
Extended hospital configuration parser.

Handles detailed configs with:
- Item categories and inventory details
- Equipment and construction info
- Patient specialization
- Item properties (size, weight, temperature, controlled substance)
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ItemInfo:
    """Detailed information about an item/medicine."""
    item_id: str
    category: str
    unit: str  # 'tablet', 'vial', 'bag', 'box'
    size: str  # 'small', 'medium', 'large'
    weight_g: float
    temperature_sensitive: bool = False
    controlled_substance: bool = False
    requires_signature: bool = False
    expiration_sensitive: bool = False


@dataclass
class InventoryCategory:
    """Inventory category with specific items."""
    category_name: str
    stock_level: float
    max_stock: float
    consumption_rate: float
    buffer_time: float = 2.0
    items: List[str] = field(default_factory=list)
    priority: int = 1  # 1=low, 5=high


@dataclass
class NodeConstruction:
    """Physical construction details of a node."""
    equipment: List[str] = field(default_factory=list)
    num_beds: int = 0
    shelving: Dict = field(default_factory=dict)
    environmental: Dict = field(default_factory=dict)
    bedside_storage: Dict = field(default_factory=dict)


@dataclass
class ExtendedNodeConfig:
    """Extended node configuration with detailed inventory."""
    node_id: str
    node_type: str
    name: str
    pos: tuple
    size: tuple

    # Construction details
    construction: NodeConstruction = None

    # Inventory by category
    inventory_categories: Dict[str, InventoryCategory] = field(default_factory=dict)

    # Storage organization
    storage_zones: Dict[str, List[str]] = field(default_factory=dict)

    # Patient/specialization info
    patient_info: Dict = field(default_factory=dict)

    # Priority items for this node
    priority_items: List[str] = field(default_factory=list)


class ExtendedConfigParser:
    """Parse extended JSON configs with detailed node information."""

    def __init__(self, config_path: str):
        """
        Load extended config from JSON file.

        Args:
            config_path: Path to JSON config file
        """
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)

        self.name = self.config_data.get('name', 'unnamed')
        self.metadata = self.config_data.get('metadata', {})

        # Parse item database
        self.item_database: Dict[str, ItemInfo] = {}
        if 'item_database' in self.config_data:
            for item_id, item_data in self.config_data['item_database'].items():
                self.item_database[item_id] = ItemInfo(
                    item_id=item_id,
                    **item_data
                )

        # Parse nodes
        self.nodes: List[ExtendedNodeConfig] = []
        for node_data in self.config_data.get('nodes', []):
            self.nodes.append(self._parse_node(node_data))

        # Parse edges
        self.edges = self.config_data.get('edges', [])

        # Task type mappings
        self.task_type_mappings = self.config_data.get('task_type_item_mapping', {})

    def _parse_node(self, node_data: Dict) -> ExtendedNodeConfig:
        """Parse a single node from config."""

        # Parse construction
        construction = None
        if 'construction' in node_data:
            const_data = node_data['construction']
            construction = NodeConstruction(
                equipment=const_data.get('equipment', []),
                num_beds=const_data.get('num_beds', 0),
                shelving=const_data.get('shelving', {}),
                environmental=const_data.get('environmental', {}),
                bedside_storage=const_data.get('bedside_storage', {})
            )

        # Parse inventory categories
        inventory_categories = {}
        if 'inventory' in node_data and 'categories' in node_data['inventory']:
            for cat_name, cat_data in node_data['inventory']['categories'].items():
                inventory_categories[cat_name] = InventoryCategory(
                    category_name=cat_name,
                    stock_level=cat_data.get('stock_level', 0.0),
                    max_stock=cat_data.get('max_stock', 100.0),
                    consumption_rate=cat_data.get('consumption_rate', 0.0),
                    buffer_time=cat_data.get('buffer_time', 2.0),
                    items=cat_data.get('items', [])
                )

        # Storage zones
        storage_zones = {}
        if 'inventory' in node_data and 'storage_zones' in node_data['inventory']:
            storage_zones = node_data['inventory']['storage_zones']

        # Priority items
        priority_items = []
        if 'inventory' in node_data:
            priority_items = node_data['inventory'].get('priority_items', [])

        return ExtendedNodeConfig(
            node_id=str(node_data['id']),
            node_type=node_data['type'],
            name=node_data.get('name', f"Node_{node_data['id']}"),
            pos=tuple(node_data['pos']),
            size=tuple(node_data['size']),
            construction=construction,
            inventory_categories=inventory_categories,
            storage_zones=storage_zones,
            patient_info=node_data.get('patient_info', {}),
            priority_items=priority_items
        )

    def get_node_by_id(self, node_id: str) -> Optional[ExtendedNodeConfig]:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> List[ExtendedNodeConfig]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.node_type == node_type]

    def get_item_info(self, item_id: str) -> Optional[ItemInfo]:
        """Get item details from database."""
        return self.item_database.get(item_id)

    def get_nodes_with_item(self, item_id: str) -> List[ExtendedNodeConfig]:
        """Find all nodes that stock a specific item."""
        nodes_with_item = []
        for node in self.nodes:
            for category in node.inventory_categories.values():
                if item_id in category.items:
                    nodes_with_item.append(node)
                    break
        return nodes_with_item

    def get_category_consumption(self, node_id: str, category_name: str) -> float:
        """Get consumption rate for a specific category at a node."""
        node = self.get_node_by_id(node_id)
        if node and category_name in node.inventory_categories:
            return node.inventory_categories[category_name].consumption_rate
        return 0.0

    def get_total_consumption(self, node_id: str) -> float:
        """Get total consumption rate across all categories."""
        node = self.get_node_by_id(node_id)
        if not node:
            return 0.0

        total = sum(
            cat.consumption_rate
            for cat in node.inventory_categories.values()
        )
        return total

    def get_equipment_list(self, node_id: str) -> List[str]:
        """Get list of equipment at a node."""
        node = self.get_node_by_id(node_id)
        if node and node.construction:
            return node.construction.equipment
        return []

    def get_controlled_substances(self) -> List[str]:
        """Get list of all controlled substance IDs."""
        return [
            item_id
            for item_id, item in self.item_database.items()
            if item.controlled_substance
        ]

    def needs_temperature_control(self, item_id: str) -> bool:
        """Check if item requires temperature control."""
        item = self.get_item_info(item_id)
        return item.temperature_sensitive if item else False

    def summarize(self):
        """Print summary of config."""
        print(f"\n{'='*60}")
        print(f"Config: {self.name}")
        print(f"{'='*60}")

        print(f"\nMetadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")

        print(f"\nNodes ({len(self.nodes)}):")
        for node in self.nodes:
            print(f"\n  {node.node_id}: {node.name} ({node.node_type})")
            print(f"    Position: {node.pos}, Size: {node.size}")

            if node.construction:
                print(f"    Equipment: {len(node.construction.equipment)} items")
                if node.construction.num_beds > 0:
                    print(f"    Beds: {node.construction.num_beds}")

            if node.inventory_categories:
                print(f"    Inventory categories: {len(node.inventory_categories)}")
                for cat_name, cat in node.inventory_categories.items():
                    print(f"      - {cat_name}: {cat.stock_level}/{cat.max_stock} "
                          f"(consumption: {cat.consumption_rate}/hr)")

            if node.patient_info:
                spec = node.patient_info.get('specialization', 'N/A')
                print(f"    Specialization: {spec}")

        print(f"\nItem Database: {len(self.item_database)} items")
        controlled = self.get_controlled_substances()
        if controlled:
            print(f"  Controlled substances: {len(controlled)}")

        print(f"\nEdges: {len(self.edges)}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    import os
    import sys

    # Try to load example config
    config_path = 'examples/extended_config_example.json'

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Run from project root directory")
        sys.exit(1)

    print("Loading extended hospital configuration...")
    parser = ExtendedConfigParser(config_path)

    # Summarize
    parser.summarize()

    # Example queries
    print("\nExample Queries:")
    print("-" * 60)

    # Find nodes with cardiac medications
    print("\n1. Nodes with cardiac medications:")
    nodes = parser.get_nodes_with_item('aspirin')
    for node in nodes:
        print(f"   - {node.name}")

    # Get controlled substances
    print("\n2. Controlled substances:")
    controlled = parser.get_controlled_substances()
    for item_id in controlled:
        item = parser.get_item_info(item_id)
        print(f"   - {item_id} ({item.category})")

    # Check equipment
    print("\n3. Equipment in Cardiology Ward:")
    equipment = parser.get_equipment_list('1')
    for eq in equipment:
        print(f"   - {eq}")

    # Total consumption
    print("\n4. Total consumption rates:")
    for node in parser.nodes:
        if node.node_type == 'recovery':
            total = parser.get_total_consumption(node.node_id)
            print(f"   - {node.name}: {total:.1f} items/hour")
