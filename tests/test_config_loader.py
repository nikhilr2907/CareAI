from pathlib import Path

from src.environment.graph.config_loader import (
    list_available_configs,
    load_config_from_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_list_available_configs_returns_repo_configs():
    configs = list_available_configs(str(ROOT / "configs"))

    assert any(path.endswith("hospital_3nodes_consumables.json") for path in configs)
    assert any(path.endswith("revised_hospital_config_v3.json") for path in configs)


def test_load_config_from_repo_v3_config_returns_nodes_edges_and_metadata():
    config_path = ROOT / "configs" / "revised_hospital_config_v3_consumption_reduced.json"

    nodes, edges, meta = load_config_from_file(str(config_path))

    assert nodes
    assert edges
    assert "sku_database" in meta
    assert "category_order" in meta
    assert "locations" in meta
