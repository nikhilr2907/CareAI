"""Parser for the extended hospital config format."""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ItemInfo:
    """Item metadata."""
    item_id: str
    category: str
    unit: str
    size: str
    weight_g: float
    temperature_sensitive: bool = False
    controlled_substance: bool = False
    requires_signature: bool = False
    expiration_sensitive: bool = False


@dataclass
class InventoryCategory:
    """Inventory category data."""
    category_name: str
    stock_level: float
    max_stock: float
    consumption_rate: float
    buffer_time: float = 2.0
    items: List[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class NodeConstruction:
    """Node construction data."""
    equipment: List[str] = field(default_factory=list)
    num_beds: int = 0
    shelving: Dict = field(default_factory=dict)
    environmental: Dict = field(default_factory=dict)
    bedside_storage: Dict = field(default_factory=dict)


@dataclass
class ExtendedNodeConfig:
    """Extended-format node configuration."""
    node_id: str
    node_type: str
    name: str
    pos: tuple
    size: tuple

    construction: NodeConstruction = None
    inventory_categories: Dict[str, InventoryCategory] = field(default_factory=dict)
    storage_zones: Dict[str, List[str]] = field(default_factory=dict)
    patient_info: Dict = field(default_factory=dict)
    priority_items: List[str] = field(default_factory=list)


class ExtendedConfigParser:
    """Parse the extended hospital config format."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)

        self.name = self.config_data.get('name', 'unnamed')
        self.metadata = self.config_data.get('metadata', {})

        self.item_database: Dict[str, ItemInfo] = {}
        if 'item_database' in self.config_data:
            for item_id, item_data in self.config_data['item_database'].items():
                self.item_database[item_id] = ItemInfo(
                    item_id=item_id,
                    **item_data
                )

        self.nodes: List[ExtendedNodeConfig] = []
        for node_data in self.config_data.get('nodes', []):
            self.nodes.append(self._parse_node(node_data))

        self.edges = self.config_data.get('edges', [])
        self.task_type_mappings = self.config_data.get('task_type_item_mapping', {})

    def _parse_node(self, node_data: Dict) -> ExtendedNodeConfig:
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

        storage_zones = {}
        if 'inventory' in node_data and 'storage_zones' in node_data['inventory']:
            storage_zones = node_data['inventory']['storage_zones']

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
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> List[ExtendedNodeConfig]:
        return [node for node in self.nodes if node.node_type == node_type]

    def get_item_info(self, item_id: str) -> Optional[ItemInfo]:
        return self.item_database.get(item_id)

    def get_nodes_with_item(self, item_id: str) -> List[ExtendedNodeConfig]:
        nodes_with_item = []
        for node in self.nodes:
            for category in node.inventory_categories.values():
                if item_id in category.items:
                    nodes_with_item.append(node)
                    break
        return nodes_with_item

    def get_category_consumption(self, node_id: str, category_name: str) -> float:
        node = self.get_node_by_id(node_id)
        if node and category_name in node.inventory_categories:
            return node.inventory_categories[category_name].consumption_rate
        return 0.0

    def get_total_consumption(self, node_id: str) -> float:
        node = self.get_node_by_id(node_id)
        if not node:
            return 0.0

        total = sum(
            cat.consumption_rate
            for cat in node.inventory_categories.values()
        )
        return total

    def get_equipment_list(self, node_id: str) -> List[str]:
        node = self.get_node_by_id(node_id)
        if node and node.construction:
            return node.construction.equipment
        return []

    def get_controlled_substances(self) -> List[str]:
        return [
            item_id
            for item_id, item in self.item_database.items()
            if item.controlled_substance
        ]

    def needs_temperature_control(self, item_id: str) -> bool:
        item = self.get_item_info(item_id)
        return item.temperature_sensitive if item else False

    def summarize(self):
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
