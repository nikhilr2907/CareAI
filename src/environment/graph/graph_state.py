from dataclasses import dataclass, field
from typing import List
import numpy as np
from .node import HospitalNode
from .edge import HospitalEdge


@dataclass
class GraphState:
    """Represents the hospital graph structure (nodes + edges)."""
    nodes: List[HospitalNode] = field(default_factory=list)
    edges: List[HospitalEdge] = field(default_factory=list)

    def __post_init__(self):
        """Create default hospital graph if none provided."""
        if not self.nodes:
            self._create_default_nodes()
        if not self.edges:
            self._create_default_edges()

    def _create_default_nodes(self):
        """Create 10 default hospital nodes."""
        # Create a simple grid-like hospital layout
        positions = [
            (0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0),
            (0.0, 10.0), (10.0, 10.0), (20.0, 10.0), (30.0, 10.0),
            (5.0, 5.0), (25.0, 5.0)
        ]

        node_types = ['storage', 'corridor', 'recovery', 'hub', 'storage',
                     'corridor', 'recovery', 'hub', 'corridor', 'storage']

        for i, ((x, y), node_type) in enumerate(zip(positions, node_types)):
            # Set inventory parameters based on node type
            # Recovery nodes = wards that need supplies
            # Storage nodes = central storage (infinite supply)
            if node_type == 'recovery':
                stock_level = np.random.uniform(50.0, 150.0)
                consumption_rate = np.random.uniform(5.0, 15.0)  # items/hour
                buffer_time = 2.0
                max_stock = 200.0
            elif node_type == 'storage':
                stock_level = 1000.0  # Central storage has large capacity
                consumption_rate = 0.0  # Storage doesn't consume
                buffer_time = 999.0
                max_stock = 1000.0
            else:
                # Corridors, hubs don't have inventory
                stock_level = 0.0
                consumption_rate = 0.0
                buffer_time = 999.0
                max_stock = 0.0

            node = HospitalNode(
                node_id=f"node_{i}",
                node_type=node_type,
                x=x,
                y=y,
                width_m=1.9,
                clearance_m=0.9,
                max_reach_height=1.35,
                unit_height=2.1,
                has_wash_basin=False,
                is_cluttered=np.random.random() < 0.2,  # 20% chance of clutter
                current_occupancy=0,
                max_capacity=2,
                stock_level=stock_level,
                consumption_rate=consumption_rate,
                buffer_time=buffer_time,
                max_stock=max_stock
            )
            self.nodes.append(node)

    def _create_default_edges(self):
        """Create 20 default edges connecting the nodes."""
        # Define edge connections (from_idx, to_idx)
        connections = [
            # Horizontal connections
            (0, 1), (1, 2), (2, 3),
            (4, 5), (5, 6), (6, 7),
            # Vertical connections
            (0, 4), (1, 5), (2, 6), (3, 7),
            # Diagonal/cross connections
            (0, 8), (1, 8), (4, 8), (5, 8),
            (2, 9), (3, 9), (6, 9), (7, 9),
            # Extra connections
            (8, 9), (1, 6)
        ]

        for from_idx, to_idx in connections:
            from_node = self.nodes[from_idx]
            to_node = self.nodes[to_idx]

            # Calculate Euclidean distance
            distance = np.sqrt((from_node.x - to_node.x)**2 +
                             (from_node.y - to_node.y)**2)

            edge = HospitalEdge(
                from_node=from_node.node_id,
                to_node=to_node.node_id,
                distance_m=distance,
                max_v_ms=1.0,
                width_m=1.9,
                clutter_level=np.random.random() * 0.3,  # Random clutter 0-0.3
                active_robot_ids=[],
                has_patient_bed=np.random.random() < 0.1  # 10% chance of bed
            )
            self.edges.append(edge)

    def get_node_features(self) -> np.ndarray:
        """Extract node features as numpy array (10 nodes × 5 features)."""
        features = []
        for node in self.nodes:
            node_features = [
                node.x,
                node.y,
                float(node.is_cluttered),
                node.current_occupancy / max(node.max_capacity, 1),  # occupancy ratio
                node.max_reach_height
            ]
            features.append(node_features)
        return np.array(features, dtype=np.float32)

    def get_edge_features(self) -> np.ndarray:
        """Extract edge features as numpy array (20 edges × 3 features)."""
        features = []
        for edge in self.edges:
            edge_features = [
                edge.current_weight,
                float(edge.has_patient_bed),
                edge.clutter_level
            ]
            features.append(edge_features)
        return np.array(features, dtype=np.float32)

    def get_node_position(self, node_id: str) -> tuple:
        """Get (x, y) position of a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return (node.x, node.y)
        return (0.0, 0.0)

    def get_node_by_index(self, idx: int) -> HospitalNode:
        """Get node by index."""
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return self.nodes[0]
