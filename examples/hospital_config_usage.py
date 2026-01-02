"""
Examples of using hospital configuration system.

Shows different ways to create and configure hospital environments.
"""
import sys
sys.path.append('..')

from src.environment.graph.graph_state import GraphState
from src.environment.graph.hospital_config import (
    HospitalConfig,
    create_small_hospital,
    create_large_hospital
)
from src.environment.gapo_env import GAPOTaskAssignmentEnv


def example_1_default_config():
    """Example 1: Use default configuration (current hardcoded layout)."""
    print("=" * 60)
    print("Example 1: Default Configuration")
    print("=" * 60)

    # Method A: Let GraphState use default
    graph = GraphState()

    # Method B: Explicit default config
    config = HospitalConfig()  # Gets default config
    graph = GraphState(config=config)

    print(f"Config name: {graph.config.name}")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    print(f"Node types: {graph.config.metadata}")


def example_2_random_grid():
    """Example 2: Generate random grid-based hospital."""
    print("\n" + "=" * 60)
    print("Example 2: Random Grid Hospital")
    print("=" * 60)

    # Generate a 3x3 grid with randomized node types
    config_dict = HospitalConfig.generate_random_grid(
        rows=3,
        cols=3,
        spacing=10.0,
        storage_ratio=0.3,  # 30% storage nodes
        recovery_ratio=0.4  # 40% recovery/ward nodes
    )

    config = HospitalConfig(config_dict)
    graph = GraphState(config=config)

    print(f"Config name: {config.name}")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Layout: {config.metadata['layout']} ({config.metadata['rows']}x{config.metadata['cols']})")

    # Show node type distribution
    type_counts = {}
    for node in graph.nodes:
        type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
    print(f"Node types: {type_counts}")


def example_3_predefined_configs():
    """Example 3: Use predefined small/large hospitals."""
    print("\n" + "=" * 60)
    print("Example 3: Predefined Configurations")
    print("=" * 60)

    # Small hospital (6 nodes, for quick testing)
    small_config = create_small_hospital()
    small_graph = GraphState(config=small_config)
    print(f"\nSmall hospital: {len(small_graph.nodes)} nodes")

    # Large hospital (20 nodes, for realistic training)
    large_config = create_large_hospital()
    large_graph = GraphState(config=large_config)
    print(f"Large hospital: {len(large_graph.nodes)} nodes")


def example_4_save_load_config():
    """Example 4: Save and load configurations from files."""
    print("\n" + "=" * 60)
    print("Example 4: Save/Load Configurations")
    print("=" * 60)

    # Generate a random config
    config_dict = HospitalConfig.generate_random_grid(rows=4, cols=3)
    config = HospitalConfig(config_dict)

    # Save to file
    config.to_file('hospital_4x3.json')
    print(f"Saved config to: hospital_4x3.json")

    # Load from file
    loaded_config = HospitalConfig.from_file('hospital_4x3.json')
    graph = GraphState(config=loaded_config)
    print(f"Loaded config: {loaded_config.name}")
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")


def example_5_curriculum_training():
    """Example 5: Curriculum learning with increasing complexity."""
    print("\n" + "=" * 60)
    print("Example 5: Curriculum Training")
    print("=" * 60)

    curriculum = [
        ('Easy', 2, 3),    # 2x3 = 6 nodes
        ('Medium', 3, 3),  # 3x3 = 9 nodes
        ('Hard', 3, 4),    # 3x4 = 12 nodes
        ('Expert', 4, 5),  # 4x5 = 20 nodes
    ]

    for stage_name, rows, cols in curriculum:
        config_dict = HospitalConfig.generate_random_grid(
            rows=rows,
            cols=cols,
            storage_ratio=0.25,
            recovery_ratio=0.25
        )
        config = HospitalConfig(config_dict)

        # Could create environment with this config
        # env = GAPOTaskAssignmentEnv(...)
        # env.graph_state = GraphState(config=config)

        num_nodes = rows * cols
        print(f"{stage_name:8s}: {rows}x{cols} = {num_nodes:2d} nodes")


def example_6_domain_randomization():
    """Example 6: Domain randomization for training."""
    print("\n" + "=" * 60)
    print("Example 6: Domain Randomization")
    print("=" * 60)

    print("Generating 5 random hospital layouts:")

    for i in range(5):
        # Randomize grid size
        rows = np.random.randint(2, 5)
        cols = np.random.randint(3, 6)

        # Randomize node type ratios
        storage_ratio = np.random.uniform(0.2, 0.35)
        recovery_ratio = np.random.uniform(0.2, 0.4)

        config_dict = HospitalConfig.generate_random_grid(
            rows=rows,
            cols=cols,
            spacing=np.random.uniform(8.0, 12.0),
            storage_ratio=storage_ratio,
            recovery_ratio=recovery_ratio
        )

        config = HospitalConfig(config_dict)
        graph = GraphState(config=config)

        print(f"  Layout {i+1}: {rows}x{cols} = {len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges")


if __name__ == '__main__':
    import numpy as np

    example_1_default_config()
    example_2_random_grid()
    example_3_predefined_configs()
    example_4_save_load_config()
    example_5_curriculum_training()
    example_6_domain_randomization()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
