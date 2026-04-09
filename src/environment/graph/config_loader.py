"""Load ILC pilot configs into GraphNode objects."""
from pathlib import Path
from typing import List, Tuple, Dict
import json
import logging

from .node import GraphNode

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Tuple[List[GraphNode], List[Tuple[int, int]], Dict]:
    """Load an ILC pilot config file and return (nodes, edge_pairs, meta)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_data = json.load(f)

    env_type = raw_data.get('metadata', {}).get('environment_type', '')
    if env_type != 'ilc_open_space' and 'school_schedule' not in raw_data:
        raise ValueError(
            f"Expected an ILC pilot config (environment_type: ilc_open_space) "
            f"but got environment_type='{env_type}'. "
            f"Hospital configs are no longer supported."
        )

    return _load_ilc_config(raw_data)


def _normalize_category(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("&", "and").replace("/", " ").replace("-", " ")
    key = " ".join(key.split()).replace(" ", "_")
    return key


def _load_ilc_config(raw_data: Dict) -> Tuple[List[GraphNode], List[Tuple[int, int]], Dict]:
    """Load an ILC pilot config (environment_type: ilc_open_space).

    - sku_database is a list; normalised to {sku_id: entry} dict.
    - Inventory field names: initial_stock/par_level/reorder_point/max_capacity
      → mapped to internal: stock/par/reorder/max.
    - Demand scaled by foot_traffic_weight (not served_beds).
    - school_schedule passed through to GraphState.
    """
    # Normalise sku_database list → dict keyed by sku_id
    sku_db_raw = raw_data.get("sku_database", [])
    sku_database: Dict = {}
    if isinstance(sku_db_raw, list):
        for entry in sku_db_raw:
            sid = entry.get("sku_id")
            if sid:
                sku_database[sid] = entry
    else:
        sku_database = dict(sku_db_raw)

    # Category order from sku_index_by_category
    sku_index = raw_data.get("sku_index_by_category", {})
    category_order = [_normalize_category(c) for c in sku_index.keys()]

    # Location tags in config order (used as categorical embedding vocab)
    location_tag_order = [
        n.get("location_tag") for n in raw_data.get("nodes", [])
        if n.get("location_tag")
    ]

    nodes: List[GraphNode] = []
    for node_data in raw_data.get("nodes", []):
        raw_inventory = node_data.get("inventory", {})

        sku_inventory: Dict = {}
        for sku_id, inv in raw_inventory.items():
            if sku_id.startswith("_"):
                continue
            category = _normalize_category(
                sku_database.get(sku_id, {}).get("category", "")
            )
            sku_inventory[sku_id] = {
                "stock":   float(inv.get("initial_stock",  inv.get("stock",   0.0))),
                "par":     float(inv.get("par_level",      inv.get("par",     0.0))),
                "reorder": float(inv.get("reorder_point",  inv.get("reorder", 0.0))),
                "max":     float(inv.get("max_capacity",   inv.get("max",     0.0))),
                "category": category,
            }

        total_stock = sum(v["stock"] for v in sku_inventory.values())
        total_max   = sum(v["max"]   for v in sku_inventory.values())

        node = GraphNode(
            node_id=f"node_{node_data['id']}",
            node_type=node_data["type"],
            center_x=float(node_data["pos"][0]),
            center_y=float(node_data["pos"][1]),
            width=float(node_data["size"][0]),
            height=float(node_data["size"][1]),
            floor=int(node_data.get("floor", 0)),
            consumption_enabled=bool(node_data.get("consumption_enabled", False)),
            foot_traffic_weight=float(node_data.get("foot_traffic_weight", 0.0)),
            location_tag=node_data.get("location_tag"),
            stock_level=total_stock,
            max_stock=total_max,
            consumption_rate=0.0,
            sku_inventory=sku_inventory,
        )
        node.display_name = node_data.get("name", node.node_id)
        node.recalc_category_inventory()
        nodes.append(node)

    edge_pairs: List[Tuple[int, int]] = [
        (int(e["from"]), int(e["to"])) for e in raw_data.get("edges", [])
    ]

    meta = {
        "sku_database":      sku_database,
        "demand_profiles":   raw_data.get("demand_profiles", {}),
        "category_order":    category_order,
        "location_tag_order": location_tag_order,
        "school_schedule":   raw_data.get("school_schedule"),
        "edges_detailed":    raw_data.get("edges", []),
    }

    return nodes, edge_pairs, meta


def get_num_robots_from_config(config_path: str) -> int:
    """Return the configured robot count or a default of 1 (ILC uses 1 robot)."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    return data.get('metadata', {}).get('num_robots', 1)


def list_available_configs(configs_dir: str = "configs") -> List[str]:
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return []
    return [str(f) for f in sorted(configs_path.glob("*.json"))]
