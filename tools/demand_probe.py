"""Check whether an environment config generates restocking demand.

Builds the environment from a config on the current checkout, steps it for a
number of simulated hours, and reports SKUs loaded, consumption, stock change
and tasks. A config that loads no SKUs or never consumes stock cannot produce
meaningful training.

Usage (from the repository root):
    python tools/demand_probe.py configs/hospital_3nodes_consumables.json --hours 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.training_utils import create_env_from_config_file  # noqa: E402


def total_sku_stock(graph_state) -> float:
    return sum(float(d.get("stock", 0.0)) for n in graph_state.nodes for d in n.sku_inventory.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    built = create_env_from_config_file(args.config)
    env = built[0] if isinstance(built, tuple) else built
    env.reset()
    gs = env.graph_state

    print(f"config: {args.config}")
    print(f"SKUs per node: {[len(n.sku_inventory) for n in gs.nodes]}")
    print(f"SKU database entries: {len(gs.sku_database or {})}, "
          f"with consumption_model: {sum(1 for v in (gs.sku_database or {}).values() if v.get('consumption_model'))}")

    sku_start, node_start = total_sku_stock(gs), sum(n.stock_level for n in gs.nodes)
    for _ in range(int(args.hours * 3600 / env.timestep_seconds)):
        env.step()
    sku_end, node_end = total_sku_stock(gs), sum(n.stock_level for n in gs.nodes)

    print(f"after {args.hours:g} simulated hours:")
    print(f"  SKU stock  {sku_start:.0f} -> {sku_end:.0f}  (consumed {sku_start - sku_end:.0f})")
    print(f"  node stock {node_start:.0f} -> {node_end:.0f}")
    print(f"  tasks created {env.next_task_id}, completed {len(env.completed_tasks)}, pending {len(env.pending_tasks)}")


if __name__ == "__main__":
    main()
