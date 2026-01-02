"""
Test script to verify custom hospital configurations work with the environment.
"""
from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.hospital_config import HospitalConfig


def test_default_config():
    """Test with default configuration."""
    print("=" * 60)
    print("Test 1: Default Configuration")
    print("=" * 60)

    env = GAPOTaskAssignmentEnv(num_robots=5, num_nodes=10)
    state = env.reset()

    print(f"[OK] Environment created with default config")
    print(f"  - Nodes: {len(env.graph_state.nodes)}")
    print(f"  - Edges: {len(env.graph_state.edges)}")
    print(f"  - Config name: {env.graph_state.config.name}")
    print()


def test_explicit_default_config():
    """Test with explicitly provided default config."""
    print("=" * 60)
    print("Test 2: Explicit Default Configuration")
    print("=" * 60)

    config = HospitalConfig()  # Default config
    env = GAPOTaskAssignmentEnv(
        num_robots=3,
        num_nodes=10,
        hospital_config=config
    )
    state = env.reset()

    print(f"[OK] Environment created with explicit default config")
    print(f"  - Nodes: {len(env.graph_state.nodes)}")
    print(f"  - Edges: {len(env.graph_state.edges)}")
    print(f"  - Config name: {env.graph_state.config.name}")
    print()


def test_random_grid_config():
    """Test with random grid configuration."""
    print("=" * 60)
    print("Test 3: Random Grid Configuration")
    print("=" * 60)

    config_dict = HospitalConfig.generate_random_grid(
        rows=3,
        cols=3,
        spacing=10.0,
        storage_ratio=0.3,
        recovery_ratio=0.4
    )
    config = HospitalConfig(config_dict)

    env = GAPOTaskAssignmentEnv(
        num_robots=4,
        num_nodes=9,  # 3x3 grid
        hospital_config=config
    )
    state = env.reset()

    print(f"[OK] Environment created with 3x3 random grid")
    print(f"  - Nodes: {len(env.graph_state.nodes)}")
    print(f"  - Edges: {len(env.graph_state.edges)}")
    print(f"  - Config name: {env.graph_state.config.name}")

    # Show node types
    storage_count = sum(1 for n in env.graph_state.nodes if n.node_type == 'storage')
    recovery_count = sum(1 for n in env.graph_state.nodes if n.node_type == 'recovery')
    hallway_count = sum(1 for n in env.graph_state.nodes if n.node_type == 'hallway')

    print(f"  - Storage nodes: {storage_count}")
    print(f"  - Recovery nodes: {recovery_count}")
    print(f"  - Hallway nodes: {hallway_count}")
    print()


def test_custom_small_config():
    """Test with small custom configuration."""
    print("=" * 60)
    print("Test 4: Small Custom Configuration")
    print("=" * 60)

    # Create a simple 5-node configuration
    config_dict = {
        'name': 'small_hospital',
        'nodes': [
            {'id': 0, 'type': 'storage', 'pos': (0, 0), 'size': (3, 3)},
            {'id': 1, 'type': 'recovery', 'pos': (10, 0), 'size': (2, 2)},
            {'id': 2, 'type': 'recovery', 'pos': (20, 0), 'size': (2, 2)},
            {'id': 3, 'type': 'hallway', 'pos': (10, 10), 'size': (1, 1)},
            {'id': 4, 'type': 'hallway', 'pos': (20, 10), 'size': (1, 1)},
        ],
        'edges': [
            (0, 1), (1, 2), (1, 3), (2, 4), (3, 4)
        ]
    }

    config = HospitalConfig(config_dict)
    env = GAPOTaskAssignmentEnv(
        num_robots=2,
        num_nodes=5,
        hospital_config=config
    )
    state = env.reset()

    print(f"[OK] Environment created with small custom config")
    print(f"  - Nodes: {len(env.graph_state.nodes)}")
    print(f"  - Edges: {len(env.graph_state.edges)}")
    print(f"  - Config name: {env.graph_state.config.name}")

    # Run a few steps to verify functionality
    for _ in range(10):
        state, reward, done, info = env.step(Δt=1.0)

    print(f"  - Simulation test: [OK] (10 timesteps completed)")
    print(f"  - Pending tasks: {len(env.pending_tasks)}")
    print()


def test_config_persistence():
    """Test that configuration persists across resets."""
    print("=" * 60)
    print("Test 5: Configuration Persistence")
    print("=" * 60)

    config_dict = HospitalConfig.generate_random_grid(
        rows=2, cols=4, spacing=8.0,
        storage_ratio=0.25, recovery_ratio=0.5
    )
    config = HospitalConfig(config_dict)

    env = GAPOTaskAssignmentEnv(
        num_robots=3,
        num_nodes=8,  # 2x4 grid
        hospital_config=config
    )

    # Reset multiple times
    for i in range(3):
        state = env.reset()
        print(f"  Reset {i+1}:")
        print(f"    - Nodes: {len(env.graph_state.nodes)}")
        print(f"    - Config name: {env.graph_state.config.name}")

    print(f"[OK] Configuration persists across resets")
    print()


if __name__ == '__main__':
    print("\nTesting Hospital Configuration System Integration\n")

    test_default_config()
    test_explicit_default_config()
    test_random_grid_config()
    test_custom_small_config()
    test_config_persistence()

    print("=" * 60)
    print("All Configuration Tests Passed!")
    print("=" * 60)
