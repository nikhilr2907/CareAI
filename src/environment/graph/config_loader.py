"""Load hospital configs into ``GraphNode`` objects."""
from pathlib import Path
from typing import List, Tuple, Dict
import json
import logging

from .node import GraphNode
from .mock_config_parser import MockConfigParser
from .extended_config_parser import ExtendedConfigParser

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Tuple[List[GraphNode], List[Tuple[int, int]], Dict]:
    """Load a config file and return nodes, edges, and metadata."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_data = json.load(f)

    version = raw_data.get('metadata', {}).get('version', '')

    if '3.0' in version and 'item_database' in raw_data and 'shelves' in raw_data:
        logger.info(f"Detected v3 config format (version: {version})")
        return _load_v3_config(raw_data)

    if 'consumables' in version or 'item_database' in raw_data:
        logger.info(f"Detected mock config format (version: {version})")
        nodes, edges = _load_mock_config(config_path)
        return nodes, edges, {}

    if 'item_database' in raw_data or 'extended' in version:
        logger.info(f"Detected extended config format (version: {version})")
        nodes, edges = _load_extended_config(config_path)
        return nodes, edges, {}

    logger.info("Detected legacy simple config format")
    nodes, edges = _load_simple_config(config_path)
    return nodes, edges, {}


def _normalize_category(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("&", "and")
    key = key.replace("/", " ")
    key = key.replace("-", " ")
    key = " ".join(key.split())
    key = key.replace(" ", "_")
    return key


def _canonical_category_order(raw_data: Dict) -> List[str]:
    multipliers = raw_data.get("metadata", {}).get("demand_profiles", {}).get("department_category_multipliers", {})
    if multipliers:
        for dept, cats in multipliers.items():
            if dept in ("Stores", "buffer"):
                continue
            if isinstance(cats, dict):
                return [_normalize_category(c) for c in cats.keys() if c != "*"]
    # Fallback from item_database
    order = []
    for sku_data in raw_data.get("item_database", {}).values():
        cat = _normalize_category(sku_data.get("category", ""))
        if cat and cat not in order:
            order.append(cat)
    return order


def _build_category_multipliers(raw_data: Dict) -> Dict[str, Dict[str, float]]:
    raw = raw_data.get("metadata", {}).get("demand_profiles", {}).get("department_category_multipliers", {})
    multipliers = {}
    for dept, cats in raw.items():
        if not isinstance(cats, dict):
            continue
        dept_key = dept
        multipliers[dept_key] = {}
        for cat_name, val in cats.items():
            if cat_name == "*":
                multipliers[dept_key]["*"] = float(val)
            else:
                multipliers[dept_key][_normalize_category(cat_name)] = float(val)
    return multipliers


def _load_v3_config(raw_data: Dict) -> Tuple[List[GraphNode], List[Tuple[int, int]], Dict]:
    """Load a v3 config."""
    item_database = raw_data.get("item_database", {})
    sku_to_category = {}
    for sku_id, sku_data in item_database.items():
        sku_to_category[sku_id] = _normalize_category(sku_data.get("category", ""))

    category_order = _canonical_category_order(raw_data)
    demand_profiles = raw_data.get("metadata", {}).get("demand_profiles", {})
    category_multipliers = _build_category_multipliers(raw_data)
    demand_profiles["department_category_multipliers_normalized"] = category_multipliers

    location_context = {}
    for loc in raw_data.get("locations", []):
        location_context[loc.get("location_id")] = {
            "department_tag": loc.get("department_tag"),
            "storage_tier": loc.get("storage_tier"),
            "demand_context": loc.get("demand_context", {}),
            "has_storage": loc.get("has_storage", True)
        }

    shelves_by_location: Dict[str, List[Dict]] = {}
    for shelf in raw_data.get("shelves", []):
        loc_id = shelf.get("location_id")
        if not loc_id:
            continue
        shelves_by_location.setdefault(loc_id, []).append(shelf)

    nodes = []
    for node_data in raw_data.get("nodes", []):
        inv = node_data.get("inventory", {}).get("categories", {})
        category_inventory = {}
        for cat_key, cat_data in inv.items():
            norm = _normalize_category(cat_key)
            category_inventory[norm] = {
                "stock": float(cat_data.get("stock_level", 0.0)),
                "max": float(cat_data.get("max_stock", 0.0)),
                "num_skus": int(cat_data.get("num_distinct_skus", 0)),
                "rate": 0.0,
            }

        operational = node_data.get("operational_locations", {})
        location_ids = operational.get("location_ids", []) or []
        shelf_ids = operational.get("shelf_ids", []) or []

        served_beds = 0
        department_tag = None
        consumption_enabled = False
        for loc_id in location_ids:
            ctx = location_context.get(loc_id, {})
            storage_tier = ctx.get("storage_tier")
            if storage_tier == "point_of_use":
                consumption_enabled = True
                demand_ctx = ctx.get("demand_context", {})
                served_beds += int(demand_ctx.get("served_beds", 0))
                if department_tag is None:
                    department_tag = demand_ctx.get("department_tag", ctx.get("department_tag"))

        node = GraphNode(
            node_id=f"node_{node_data['id']}",
            node_type=node_data["type"],
            center_x=node_data["pos"][0],
            center_y=node_data["pos"][1],
            width=node_data["size"][0],
            height=node_data["size"][1],
            clearance_m=0.9,
            max_reach_height=1.35,
            unit_height=2.1,
            has_wash_basin=False,
            is_cluttered=False,
            stock_level=sum(v["stock"] for v in category_inventory.values()),
            max_stock=sum(v["max"] for v in category_inventory.values()),
            consumption_rate=0.0,
            buffer_time=2.0,
            category_inventory=category_inventory,
            location_id=location_ids[0] if location_ids else None,
            location_ids=location_ids,
            shelf_ids=shelf_ids,
            floor=node_data.get("floor", 0),
            department_tag=department_tag,
            served_beds=served_beds,
            consumption_enabled=consumption_enabled
        )
        node.display_name = node_data.get("name", node.node_id)

        sku_inventory = {}
        for loc_id in location_ids:
            for shelf in shelves_by_location.get(loc_id, []):
                for bin_data in shelf.get("sku_bins", []):
                    sku_id = bin_data.get("sku_id")
                    if not sku_id:
                        continue
                    category = sku_to_category.get(sku_id, "")
                    entry = sku_inventory.setdefault(sku_id, {
                        "stock": 0.0,
                        "par": 0.0,
                        "reorder": 0.0,
                        "max": 0.0,
                        "category": category,
                        "area_multiplier": 1.0
                    })
                    entry["par"] += float(bin_data.get("par_level_units", 0.0))
                    entry["reorder"] += float(bin_data.get("reorder_point_units", 0.0))
                    entry["max"] += float(bin_data.get("max_units", 0.0))
                    entry["area_multiplier"] = float(bin_data.get("area_sku_multiplier", entry["area_multiplier"]))

        for cat, cat_data in category_inventory.items():
            skus_in_cat = [sku for sku, data in sku_inventory.items() if data.get("category") == cat]
            if not skus_in_cat:
                continue
            total_par = sum(sku_inventory[sku].get("par", 0.0) for sku in skus_in_cat)
            for sku in skus_in_cat:
                if total_par > 0:
                    share = sku_inventory[sku].get("par", 0.0) / total_par
                else:
                    share = 1.0 / max(len(skus_in_cat), 1)
                alloc = cat_data.get("stock", 0.0) * share
                max_level = sku_inventory[sku].get("max", alloc)
                sku_inventory[sku]["stock"] = min(max_level, alloc)

        node.sku_inventory = sku_inventory
        node.recalc_category_inventory()
        nodes.append(node)

    edge_pairs = []
    if raw_data.get("edges_detailed"):
        for edge_data in raw_data.get("edges_detailed", []):
            edge_pairs.append((edge_data.get("from"), edge_data.get("to")))
    else:
        edge_pairs = raw_data.get("edges", [])

    department_order = [k for k in category_multipliers.keys() if k not in ("Stores", "buffer")]
    meta = {
        "sku_database": item_database,
        "demand_profiles": demand_profiles,
        "category_order": category_order,
        "department_order": department_order,
        "edges_detailed": raw_data.get("edges_detailed", []),
        "locations": raw_data.get("locations", []),
    }

    return nodes, edge_pairs, meta


def _load_mock_config(config_path: str) -> Tuple[List[GraphNode], List[Tuple[int, int]]]:
    """Load a mock config."""
    parser = MockConfigParser(config_path)

    nodes = []
    for mock_node in parser.nodes:
        category_inventory = {}
        for cat_name, cat_data in mock_node.category_inventory.items():
            category_inventory[cat_name] = {
                'stock': cat_data.stock_level,
                'max': cat_data.max_stock,
                'num_skus': cat_data.num_distinct_skus,
                'rate': cat_data.consumption_rate
            }

        location_id = None
        shelf_ids = []
        if mock_node.operational_location:
            location_id = mock_node.operational_location.location_ids[0] if mock_node.operational_location.location_ids else None
            shelf_ids = mock_node.operational_location.shelf_ids

        total_stock = sum(cat.stock_level for cat in mock_node.category_inventory.values())
        total_max_stock = sum(cat.max_stock for cat in mock_node.category_inventory.values())
        total_consumption = sum(cat.consumption_rate for cat in mock_node.category_inventory.values())

        node = GraphNode(
            node_id=f"node_{mock_node.node_id}",
            node_type=mock_node.node_type,
            center_x=mock_node.pos[0],
            center_y=mock_node.pos[1],
            width=mock_node.size[0],
            height=mock_node.size[1],

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
            location_id=location_id,
            shelf_ids=shelf_ids,
            floor=mock_node.floor
        )
        node.display_name = mock_node.name
        nodes.append(node)

    edge_pairs = parser.edges

    return nodes, edge_pairs


def _load_extended_config(config_path: str) -> Tuple[List[GraphNode], List[Tuple[int, int]]]:
    """Load an extended config."""
    parser = ExtendedConfigParser(config_path)

    nodes = []
    for ext_node in parser.nodes:
        category_inventory = {}
        for cat_name, cat_data in ext_node.inventory_categories.items():
            category_inventory[cat_name] = {
                'stock': cat_data.stock_level,
                'max': cat_data.max_stock,
                'num_skus': len(cat_data.items),
                'rate': cat_data.consumption_rate
            }

        total_stock = sum(cat.stock_level for cat in ext_node.inventory_categories.values())
        total_max_stock = sum(cat.max_stock for cat in ext_node.inventory_categories.values())
        total_consumption = sum(cat.consumption_rate for cat in ext_node.inventory_categories.values())

        node = GraphNode(
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
            floor=0
        )
        node.display_name = ext_node.name
        nodes.append(node)

    edge_pairs = parser.edges

    return nodes, edge_pairs


def _load_simple_config(config_path: str) -> Tuple[List[GraphNode], List[Tuple[int, int]]]:
    """Load a legacy config."""
    with open(config_path, 'r') as f:
        data = json.load(f)

    nodes = []
    for node_data in data.get('nodes', []):
        node_type = node_data['type']

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

        node = GraphNode(
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
        node.display_name = node_data.get('name', node.node_id)
        nodes.append(node)

    edge_pairs = data.get('edges', [])

    return nodes, edge_pairs


def get_num_robots_from_config(config_path: str) -> int:
    """Return the configured robot count or a default."""
    with open(config_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    return metadata.get('num_robots', 5)


def list_available_configs(configs_dir: str = "configs") -> List[str]:
    """List JSON config files in a directory."""
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return []

    config_files = list(configs_path.glob("*.json"))
    return [str(f) for f in sorted(config_files)]
