"""
Hospital graph configuration system.

Supports:
- Loading from config files
- Random generation
- Domain randomization
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import json


class HospitalConfig:
    """Configuration for hospital graph layout."""

    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Initialize hospital configuration.

        Args:
            config_dict: Dictionary with configuration, or None for default
        """
        if config_dict is None:
            config_dict = self.get_default_config()

        self.nodes = config_dict.get('nodes', [])
        self.edges = config_dict.get('edges', [])
        self.name = config_dict.get('name', 'default')
        self.metadata = config_dict.get('metadata', {})

    @staticmethod
    def get_default_config() -> Dict:
        """Get default 10-node configuration (current hardcoded layout)."""
        return {
            'name': 'default_10node',
            'metadata': {
                'num_nodes': 10,
                'num_storage': 3,
                'num_recovery': 2,
                'num_corridors': 3,
                'num_hubs': 2
            },
            'nodes': [
                # Bottom row
                {'id': 0, 'type': 'storage', 'pos': (2.5, 2.5), 'size': (4.0, 4.0)},
                {'id': 1, 'type': 'corridor', 'pos': (10.0, 2.5), 'size': (3.0, 4.0)},
                {'id': 2, 'type': 'recovery', 'pos': (20.0, 2.5), 'size': (5.0, 4.0)},
                {'id': 3, 'type': 'hub', 'pos': (30.0, 2.5), 'size': (4.0, 4.0)},
                # Top row
                {'id': 4, 'type': 'storage', 'pos': (2.5, 12.5), 'size': (4.0, 4.0)},
                {'id': 5, 'type': 'corridor', 'pos': (10.0, 12.5), 'size': (3.0, 4.0)},
                {'id': 6, 'type': 'recovery', 'pos': (20.0, 12.5), 'size': (5.0, 4.0)},
                {'id': 7, 'type': 'hub', 'pos': (30.0, 12.5), 'size': (4.0, 4.0)},
                # Middle
                {'id': 8, 'type': 'corridor', 'pos': (6.0, 7.5), 'size': (2.0, 8.0)},
                {'id': 9, 'type': 'storage', 'pos': (25.0, 7.5), 'size': (3.0, 6.0)},
            ],
            'edges': [
                # Horizontal
                (0, 1), (1, 2), (2, 3),
                (4, 5), (5, 6), (6, 7),
                # Vertical
                (0, 4), (1, 5), (2, 6), (3, 7),
                # Cross
                (0, 8), (1, 8), (4, 8), (5, 8),
                (2, 9), (3, 9), (6, 9), (7, 9),
                # Extra
                (8, 9), (1, 6)
            ]
        }

    @staticmethod
    def generate_random_grid(
        rows: int = 3,
        cols: int = 4,
        spacing: float = 10.0,
        storage_ratio: float = 0.25,
        recovery_ratio: float = 0.25
    ) -> Dict:
        """
        Generate random grid-based hospital layout.

        Args:
            rows: Number of rows
            cols: Number of columns
            spacing: Distance between nodes
            storage_ratio: Fraction of nodes that are storage
            recovery_ratio: Fraction of nodes that are recovery (wards)

        Returns:
            Configuration dictionary
        """
        num_nodes = rows * cols
        num_storage = int(num_nodes * storage_ratio)
        num_recovery = int(num_nodes * recovery_ratio)
        num_hubs = max(1, num_nodes // 10)  # ~10% hubs
        num_corridors = num_nodes - num_storage - num_recovery - num_hubs

        # Generate node types (randomize positions)
        node_types = (
            ['storage'] * num_storage +
            ['recovery'] * num_recovery +
            ['hub'] * num_hubs +
            ['corridor'] * num_corridors
        )
        np.random.shuffle(node_types)

        # Generate grid positions
        nodes = []
        node_idx = 0
        for row in range(rows):
            for col in range(cols):
                if node_idx >= num_nodes:
                    break

                # Random jitter for realism
                jitter_x = np.random.uniform(-spacing * 0.1, spacing * 0.1)
                jitter_y = np.random.uniform(-spacing * 0.1, spacing * 0.1)

                pos = (
                    col * spacing + spacing / 2 + jitter_x,
                    row * spacing + spacing / 2 + jitter_y
                )

                # Random size based on type
                if node_types[node_idx] == 'storage':
                    size = (np.random.uniform(3.5, 4.5), np.random.uniform(3.5, 4.5))
                elif node_types[node_idx] == 'recovery':
                    size = (np.random.uniform(4.5, 6.0), np.random.uniform(3.5, 5.0))
                elif node_types[node_idx] == 'corridor':
                    size = (np.random.uniform(2.0, 3.5), np.random.uniform(3.0, 8.0))
                else:  # hub
                    size = (np.random.uniform(4.0, 5.0), np.random.uniform(4.0, 5.0))

                nodes.append({
                    'id': node_idx,
                    'type': node_types[node_idx],
                    'pos': pos,
                    'size': size
                })
                node_idx += 1

        # Generate edges (grid connectivity + some random)
        edges = []

        # Grid connections (horizontal + vertical)
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if idx >= num_nodes:
                    break

                # Right neighbor
                if col < cols - 1 and idx + 1 < num_nodes:
                    edges.append((idx, idx + 1))

                # Bottom neighbor
                if row < rows - 1 and idx + cols < num_nodes:
                    edges.append((idx, idx + cols))

        # Add some random diagonal/long-range connections
        num_random_edges = max(2, num_nodes // 5)
        for _ in range(num_random_edges):
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)
            if i != j and (i, j) not in edges and (j, i) not in edges:
                edges.append((i, j))

        return {
            'name': f'random_grid_{rows}x{cols}',
            'metadata': {
                'num_nodes': num_nodes,
                'num_storage': num_storage,
                'num_recovery': num_recovery,
                'num_corridors': num_corridors,
                'num_hubs': num_hubs,
                'layout': 'grid',
                'rows': rows,
                'cols': cols
            },
            'nodes': nodes,
            'edges': edges
        }

    @staticmethod
    def from_file(filepath: str) -> 'HospitalConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return HospitalConfig(config_dict)

    def to_file(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            'name': self.name,
            'metadata': self.metadata,
            'nodes': self.nodes,
            'edges': self.edges
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_node_params(self, node_config: Dict) -> Dict:
        """
        Get HospitalNode constructor parameters from config.

        Args:
            node_config: Node configuration dict

        Returns:
            Dict of parameters for HospitalNode
        """
        node_type = node_config['type']
        pos = node_config['pos']
        size = node_config['size']

        # Set inventory parameters based on type
        if node_type == 'recovery':
            stock_level = np.random.uniform(50.0, 150.0)
            consumption_rate = np.random.uniform(5.0, 15.0)
            buffer_time = 2.0
            max_stock = 200.0
        elif node_type == 'storage':
            stock_level = 1000.0
            consumption_rate = 0.0
            buffer_time = 999.0
            max_stock = 1000.0
        else:
            stock_level = 0.0
            consumption_rate = 0.0
            buffer_time = 999.0
            max_stock = 0.0

        return {
            'node_id': f"node_{node_config['id']}",
            'node_type': node_type,
            'center_x': pos[0],
            'center_y': pos[1],
            'width': size[0],
            'height': size[1],
            'clearance_m': 0.9,
            'max_reach_height': 1.35,
            'unit_height': 2.1,
            'has_wash_basin': False,
            'is_cluttered': np.random.random() < 0.2,
            'stock_level': stock_level,
            'consumption_rate': consumption_rate,
            'buffer_time': buffer_time,
            'max_stock': max_stock
        }


# Example configurations
def create_small_hospital() -> HospitalConfig:
    """Create a small 6-node hospital for quick testing."""
    config = {
        'name': 'small_6node',
        'metadata': {'num_nodes': 6},
        'nodes': [
            {'id': 0, 'type': 'storage', 'pos': (5, 5), 'size': (4, 4)},
            {'id': 1, 'type': 'corridor', 'pos': (15, 5), 'size': (3, 4)},
            {'id': 2, 'type': 'recovery', 'pos': (25, 5), 'size': (5, 4)},
            {'id': 3, 'type': 'storage', 'pos': (5, 15), 'size': (4, 4)},
            {'id': 4, 'type': 'corridor', 'pos': (15, 15), 'size': (3, 4)},
            {'id': 5, 'type': 'recovery', 'pos': (25, 15), 'size': (5, 4)},
        ],
        'edges': [
            (0, 1), (1, 2),
            (3, 4), (4, 5),
            (0, 3), (1, 4), (2, 5),
            (1, 5)  # Diagonal
        ]
    }
    return HospitalConfig(config)


def create_large_hospital() -> HospitalConfig:
    """Create a large 20-node hospital."""
    return HospitalConfig(HospitalConfig.generate_random_grid(
        rows=4, cols=5, spacing=10.0
    ))
