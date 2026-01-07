"""
Config loader that converts various config formats to HospitalNode/Edge format.

Supports:
- Mock config format (mock_hospital_config.json with SKUs, categories, locations)
- Extended config format (extended_config_example.json with categories)
- Legacy config format (simple nodes with type/pos/size)
"""
from pathlib import Path
from typing import List, Tuple
import json

from .node import HospitalNode
from .edge import HospitalEdge
from .mock_config_parser import MockConfigParser
from .extended_config_parser import ExtendedConfigParser


def load_config_from_file(config_path: str) -> Tuple[List[HospitalNode], List[Tuple[int, int]]]:
    """
    Load hospital config from JSON file and convert to nodes + edges.

    Automatically detects config format and parses accordingly.

    Args:
        config_path: Path to JSON config file

    Returns:
        (nodes, edge_pairs) where:
        - nodes: List of HospitalNode objects
        - edge_pairs: List of (from_node_idx, to_node_idx) tuples

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load raw JSON to detect format
    with open(config_path, 'r') as f:
        raw_data = json.load(f)

    # Detect config format
    version = raw_data.get('metadata', {}).get('version', '')

    if 'consumables' in version or 'item_database' in raw_data:
        # Mock format with full SKU tracking
        print(f"Detected mock config format (version: {version})")
        return _load_mock_config(config_path)

    elif 'item_database' in raw_data or 'extended' in version:
        # Extended format with categories but simpler structure
        print(f"Detected extended config format (version: {version})")
        return _load_extended_config(config_path)

    else:
        # Legacy simple format
        print("Detected legacy simple config format")
        return _load_simple_config(config_path)


def _load_mock_config(config_path: str) -> Tuple[List[HospitalNode], List[Tuple[int, int]]]:
    """Load mock config format with full SKU/category tracking."""
    parser = MockConfigParser(config_path)

    nodes = []
    for mock_node in parser.nodes:
        # Convert category inventory from CategoryInventory dataclass to dict format
        category_inventory = {}
        for cat_name, cat_data in mock_node.category_inventory.items():
            category_inventory[cat_name] = {
                'stock': cat_data.stock_level,
                'max': cat_data.max_stock,
                'num_skus': cat_data.num_distinct_skus,
                'rate': cat_data.consumption_rate
            }

        # Extract operational location info
        location_id = None
        shelf_ids = []
        if mock_node.operational_location:
            location_id = mock_node.operational_location.location_ids[0] if mock_node.operational_location.location_ids else None
            shelf_ids = mock_node.operational_location.shelf_ids

        # Calculate aggregate stock/consumption from categories
        total_stock = sum(cat.stock_level for cat in mock_node.category_inventory.values())
        total_max_stock = sum(cat.max_stock for cat in mock_node.category_inventory.values())
        total_consumption = sum(cat.consumption_rate for cat in mock_node.category_inventory.values())

        node = HospitalNode(
            node_id=f"node_{mock_node.node_id}",
            node_type=mock_node.node_type,
            center_x=mock_node.pos[0],
            center_y=mock_node.pos[1],
            width=mock_node.size[0],
            height=mock_node.size[1],

            # Physical properties (use defaults for now)
            clearance_m=0.9,
            max_reach_height=1.35,
            unit_height=2.1,
            has_wash_basin=False,
            is_cluttered=False,

            # Aggregate inventory
            stock_level=total_stock,
            max_stock=total_max_stock,
            consumption_rate=total_consumption,
            buffer_time=2.0,

            # Category-based inventory
            category_inventory=category_inventory,

            # Operational location
            location_id=location_id,
            shelf_ids=shelf_ids,
            floor=mock_node.floor
        )
        nodes.append(node)

    # Edge pairs
    edge_pairs = parser.edges

    return nodes, edge_pairs


def _load_extended_config(config_path: str) -> Tuple[List[HospitalNode], List[Tuple[int, int]]]:
    """Load extended config format with categories."""
    parser = ExtendedConfigParser(config_path)

    nodes = []
    for ext_node in parser.nodes:
        # Convert category inventory
        category_inventory = {}
        for cat_name, cat_data in ext_node.inventory_categories.items():
            category_inventory[cat_name] = {
                'stock': cat_data.stock_level,
                'max': cat_data.max_stock,
                'num_skus': len(cat_data.items),  # Count of items
                'rate': cat_data.consumption_rate
            }

        # Calculate aggregate
        total_stock = sum(cat.stock_level for cat in ext_node.inventory_categories.values())
        total_max_stock = sum(cat.max_stock for cat in ext_node.inventory_categories.values())
        total_consumption = sum(cat.consumption_rate for cat in ext_node.inventory_categories.values())

        node = HospitalNode(
            node_id=ext_node.node_id,
            node_type=ext_node.node_type,
            center_x=ext_node.pos[0],
            center_y=ext_node.pos[1],
            width=ext_node.size[0],
            height=ext_node.size[1],

            clearance_m=0.9,
            max_reach_height=1.35,
            unit_height=2.1,
            has_wash_basin=False,
            is_cluttered=False,

            stock_level=total_stock,
            max_stock=total_max_stock,
            consumption_rate=total_consumption,
            buffer_time=2.0,

            category_inventory=category_inventory,
            floor=0  # Extended format doesn't have floor info
        )
        nodes.append(node)

    edge_pairs = parser.edges

    return nodes, edge_pairs


def _load_simple_config(config_path: str) -> Tuple[List[HospitalNode], List[Tuple[int, int]]]:
    """Load simple legacy config format."""
    with open(config_path, 'r') as f:
        data = json.load(f)

    nodes = []
    for node_data in data.get('nodes', []):
        # Simple format: just type, pos, size
        node_type = node_data['type']

        # Set inventory based on type
        if node_type == 'recovery':
            stock_level = 100.0
            consumption_rate = 10.0
            max_stock = 200.0
        elif node_type == 'storage':
            stock_level = 1000.0
            consumption_rate = 0.0
            max_stock = 1000.0
        else:
            stock_level = 0.0
            consumption_rate = 0.0
            max_stock = 0.0

        node = HospitalNode(
            node_id=f"node_{node_data['id']}",
            node_type=node_type,
            center_x=node_data['pos'][0],
            center_y=node_data['pos'][1],
            width=node_data['size'][0],
            height=node_data['size'][1],

            clearance_m=0.9,
            max_reach_height=1.35,
            unit_height=2.1,
            has_wash_basin=False,
            is_cluttered=False,

            stock_level=stock_level,
            max_stock=max_stock,
            consumption_rate=consumption_rate,
            buffer_time=2.0
        )
        nodes.append(node)

    edge_pairs = data.get('edges', [])

    return nodes, edge_pairs


def get_num_robots_from_config(config_path: str) -> int:
    """
    Get recommended number of robots from config metadata.

    Returns default of 5 if not specified.
    """
    with open(config_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    return metadata.get('num_robots', 5)


def list_available_configs(configs_dir: str = "configs") -> List[str]:
    """
    List all available config files in the configs directory.

    Returns:
        List of config file paths
    """
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return []

    config_files = list(configs_path.glob("*.json"))
    return [str(f) for f in sorted(config_files)]
