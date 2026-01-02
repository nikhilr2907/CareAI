"""
Demonstrate loading hospital configurations from JSON files.
"""
from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.hospital_config import HospitalConfig


def test_load_config_file(config_path, num_robots=3):
    """Load a config file and create environment."""
    print("=" * 60)
    print(f"Loading: {config_path}")
    print("=" * 60)

    # Load config from file
    config = HospitalConfig.from_file(config_path)

    print(f"Config name: {config.name}")
    print(f"Number of nodes: {len(config.nodes)}")
    print(f"Number of edges: {len(config.edges)}")

    if config.metadata:
        print(f"Metadata:")
        for key, value in config.metadata.items():
            print(f"  - {key}: {value}")

    # Create environment with this config
    env = GAPOTaskAssignmentEnv(
        num_robots=num_robots,
        num_nodes=len(config.nodes),
        hospital_config=config
    )

    # Reset and verify
    state = env.reset()

    print(f"\nEnvironment created successfully!")
    print(f"  - Nodes in graph: {len(env.graph_state.nodes)}")
    print(f"  - Edges in graph: {len(env.graph_state.edges)}")
    print(f"  - Robots: {num_robots}")
    print(f"  - Initial pending tasks: {len(env.pending_tasks)}")

    # Show node breakdown
    storage_count = sum(1 for n in env.graph_state.nodes if n.node_type == 'storage')
    recovery_count = sum(1 for n in env.graph_state.nodes if n.node_type == 'recovery')
    hallway_count = sum(1 for n in env.graph_state.nodes if n.node_type == 'hallway')

    print(f"\nNode types:")
    print(f"  - Storage: {storage_count}")
    print(f"  - Recovery: {recovery_count}")
    print(f"  - Hallway: {hallway_count}")

    # Run a few steps to verify it works
    print(f"\nRunning simulation test...")
    for i in range(20):
        state, reward, done, info = env.step(Δt=1.0)

    print(f"  - Simulation ran for 20 timesteps")
    print(f"  - Current time: {env.current_time:.1f}s")
    print(f"  - Pending tasks: {len(env.pending_tasks)}")
    print(f"  - Completed tasks: {len(env.completed_tasks)}")
    print(f"  - Cumulative reward: {env.cumulative_reward:.2f}")

    print()


def test_save_and_load():
    """Test saving current environment config to file."""
    print("=" * 60)
    print("Test: Save and Load")
    print("=" * 60)

    # Create environment with default config
    env = GAPOTaskAssignmentEnv(num_robots=5, num_nodes=10)
    env.reset()

    # Save the config
    output_path = "configs/saved_default.json"
    env.graph_state.config.to_file(output_path)
    print(f"Saved config to: {output_path}")

    # Load it back
    loaded_config = HospitalConfig.from_file(output_path)
    print(f"Loaded config: {loaded_config.name}")
    print(f"  - Nodes: {len(loaded_config.nodes)}")
    print(f"  - Edges: {len(loaded_config.edges)}")

    # Create new environment with loaded config
    env2 = GAPOTaskAssignmentEnv(
        num_robots=5,
        num_nodes=10,
        hospital_config=loaded_config
    )
    env2.reset()

    print(f"Created new environment from saved config")
    print(f"  - Nodes: {len(env2.graph_state.nodes)}")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing Config File Loading")
    print("=" * 60 + "\n")

    # Test each config file
    test_load_config_file("configs/small_hospital.json", num_robots=2)
    test_load_config_file("configs/grid_3x3.json", num_robots=4)
    test_load_config_file("configs/large_hospital.json", num_robots=8)

    # Test save/load
    test_save_and_load()

    print("=" * 60)
    print("All config file tests passed!")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Create custom JSON configs in the configs/ folder")
    print("  2. Load them with: HospitalConfig.from_file('path/to/config.json')")
    print("  3. Pass to environment: GAPOTaskAssignmentEnv(hospital_config=config)")
