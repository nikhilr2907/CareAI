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
        """
        Create 10 default hospital nodes as REGIONS (not points).
        Each node has a bounding box with width and height.
        """
        # Define regions with (center_x, center_y, width, height, node_type)
        # Layout: Grid with regions having realistic room sizes
        regions = [
            # Bottom row (y ~= 2.5)
            (2.5, 2.5, 4.0, 4.0, 'storage'),      # node_0: Storage room (4×4 meters)
            (10.0, 2.5, 3.0, 4.0, 'corridor'),    # node_1: Corridor section
            (20.0, 2.5, 5.0, 4.0, 'recovery'),    # node_2: Ward/Recovery room
            (30.0, 2.5, 4.0, 4.0, 'hub'),         # node_3: Hub area

            # Top row (y ~= 12.5)
            (2.5, 12.5, 4.0, 4.0, 'storage'),     # node_4: Storage room
            (10.0, 12.5, 3.0, 4.0, 'corridor'),   # node_5: Corridor section
            (20.0, 12.5, 5.0, 4.0, 'recovery'),   # node_6: Ward/Recovery room
            (30.0, 12.5, 4.0, 4.0, 'hub'),        # node_7: Hub area

            # Middle nodes
            (6.0, 7.5, 2.0, 8.0, 'corridor'),     # node_8: Central corridor (narrow, tall)
            (25.0, 7.5, 3.0, 6.0, 'storage'),     # node_9: Storage
        ]

        for i, (center_x, center_y, width, height, node_type) in enumerate(regions):
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
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                clearance_m=0.9,
                max_reach_height=1.35,
                unit_height=2.1,
                has_wash_basin=False,
                is_cluttered=np.random.random() < 0.2,  # 20% chance of clutter
                stock_level=stock_level,
                consumption_rate=consumption_rate,
                buffer_time=buffer_time,
                max_stock=max_stock
            )
            self.nodes.append(node)

    def _create_default_edges(self):
        """
        Create 20 default edges connecting the nodes.
        Edges now include entry/exit points based on region boundaries.
        """
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

            # Compute entry and exit points (simplified: use region centers)
            # In a real system, these would be computed based on which edges of the
            # bounding boxes face each other
            entry_point = (from_node.center_x, from_node.center_y)
            exit_point = (to_node.center_x, to_node.center_y)

            # Calculate Euclidean distance between region centers
            distance = np.sqrt((from_node.center_x - to_node.center_x)**2 +
                             (from_node.center_y - to_node.center_y)**2)

            edge = HospitalEdge(
                from_node=from_node.node_id,
                to_node=to_node.node_id,
                distance_m=distance,
                corridor_width=1.9,
                entry_point=entry_point,
                exit_point=exit_point,
                max_v_ms=1.0,
                clutter_level=np.random.random() * 0.3,  # Random clutter 0-0.3
                active_robot_ids=[],
                has_patient_bed=np.random.random() < 0.1  # 10% chance of bed
            )
            self.edges.append(edge)

    def get_node_features(self) -> np.ndarray:
        """
        Extract node features as numpy array (10 nodes × 8 features).

        Features per node:
        - center_x, center_y: Region center coordinates
        - width, height: Region dimensions
        - stock_level: Current inventory
        - consumption_rate: Items consumed per hour
        - time_to_stockout: Hours until stockout
        - occupancy_count: Number of robots in this region
        """
        features = []
        for node in self.nodes:
            node_features = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.stock_level,
                node.consumption_rate,
                node.time_to_stockout,
                float(node.occupancy_count)  # Number of robots in region
            ]
            features.append(node_features)
        return np.array(features, dtype=np.float32)

    def get_edge_features(self) -> np.ndarray:
        """
        Extract edge features as numpy array (20 edges × 5 features).

        Features per edge:
        - distance_m: Physical corridor length
        - corridor_width: Corridor width (affects capacity)
        - current_weight: Dynamic cost (includes congestion)
        - has_patient_bed: Boolean (bed obstacle)
        - clutter_level: Static clutter (0-1)
        """
        features = []
        for edge in self.edges:
            edge_features = [
                edge.distance_m,
                edge.corridor_width,
                edge.current_weight,
                float(edge.has_patient_bed),
                edge.clutter_level
            ]
            features.append(edge_features)
        return np.array(features, dtype=np.float32)

    # ===========================================================================
    # ENHANCED FEATURE EXTRACTION (Complete node and edge attributes)
    # ===========================================================================

    def get_node_features_complete(self) -> tuple:
        """
        Extract COMPLETE node features including all available attributes.

        Returns:
            (continuous_features, categorical_features) where:
            - continuous_features: [num_nodes, 15] numpy array
            - categorical_features: [num_nodes, 1] numpy array (node_type_id)

        Continuous features per node (15 total):
        1. center_x: Region center X coordinate
        2. center_y: Region center Y coordinate
        3. width: Region width (meters)
        4. height: Region height (meters)
        5. area: Region area (width * height)
        6. clearance_m: Available turning space
        7. max_reach_height: Robot's max reach (1.35m)
        8. unit_height: Physical cabinet height (2.1m)
        9. has_wash_basin: Clinical sink restriction (0/1)
        10. is_cluttered: Dynamic clutter flag (0/1)
        11. stock_level: Current inventory items
        12. consumption_rate: Items consumed per hour
        13. time_to_stockout: Hours until stockout (inf if not consuming)
        14. occupancy_count: Number of robots in region
        15. urgency_level: Stockout urgency (1-5 scale)

        Categorical features per node (1 total):
        1. node_type_id: Integer ID for node_type
           - 0: 'storage'
           - 1: 'corridor'
           - 2: 'recovery'
           - 3: 'hub'
        """
        node_type_to_id = {
            'storage': 0,
            'corridor': 1,
            'recovery': 2,
            'hub': 3
        }

        continuous_features = []
        categorical_features = []

        for node in self.nodes:
            # Continuous features (15)
            node_continuous = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,  # Computed property
                node.clearance_m,
                node.max_reach_height,
                node.unit_height,
                float(node.has_wash_basin),
                float(node.is_cluttered),
                node.stock_level,
                node.consumption_rate,
                min(node.time_to_stockout, 999.0),  # Cap at 999 to avoid inf
                float(node.occupancy_count),
                float(node.urgency_level)  # 1-5 scale
            ]
            continuous_features.append(node_continuous)

            # Categorical feature (1)
            node_type_id = node_type_to_id.get(node.node_type, 0)
            categorical_features.append([node_type_id])

        return (
            np.array(continuous_features, dtype=np.float32),  # [num_nodes, 15]
            np.array(categorical_features, dtype=np.int64)    # [num_nodes, 1]
        )

    def get_edge_features_complete(self) -> tuple:
        """
        Extract COMPLETE edge features including topology and node connectivity.

        Returns:
            (continuous_features, node_indices) where:
            - continuous_features: [num_edges, 12] numpy array
            - node_indices: [num_edges, 2] numpy array (from_node_idx, to_node_idx)

        Continuous features per edge (12 total):
        1. distance_m: Physical corridor length
        2. corridor_width: Physical width (meters)
        3. max_v_ms: Max speed allowed (1.0 m/s)
        4. entry_point_x: Entry coordinate X
        5. entry_point_y: Entry coordinate Y
        6. exit_point_x: Exit coordinate X
        7. exit_point_y: Exit coordinate Y
        8. clutter_level: Static clutter (0-1)
        9. num_active_robots: Count of robots on corridor
        10. has_patient_bed: Bed obstacle flag (0/1)
        11. current_weight: Dynamic cost (base + congestion)
        12. base_cost: Base travel time (distance / max_v_ms)

        Node connectivity (2 indices):
        1. from_node_idx: Source node index (0-9)
        2. to_node_idx: Destination node index (0-9)

        Note: Node indices will be used by GNN encoder to retrieve node embeddings
        and augment edge features with from/to node embedding vectors.
        """
        # Create node_id to index mapping
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(self.nodes)}

        continuous_features = []
        node_indices = []

        for edge in self.edges:
            # Continuous features (12)
            entry_x = edge.entry_point[0] if edge.entry_point else 0.0
            entry_y = edge.entry_point[1] if edge.entry_point else 0.0
            exit_x = edge.exit_point[0] if edge.exit_point else 0.0
            exit_y = edge.exit_point[1] if edge.exit_point else 0.0

            base_cost = edge.distance_m / edge.max_v_ms

            edge_continuous = [
                edge.distance_m,
                edge.corridor_width,
                edge.max_v_ms,
                entry_x,
                entry_y,
                exit_x,
                exit_y,
                edge.clutter_level,
                float(len(edge.active_robot_ids)),  # Number of active robots
                float(edge.has_patient_bed),
                edge.current_weight,  # Dynamic weight (includes congestion)
                base_cost  # Base travel time without congestion
            ]
            continuous_features.append(edge_continuous)

            # Node connectivity (2 indices)
            from_idx = node_id_to_idx.get(edge.from_node, 0)
            to_idx = node_id_to_idx.get(edge.to_node, 0)
            node_indices.append([from_idx, to_idx])

        return (
            np.array(continuous_features, dtype=np.float32),  # [num_edges, 12]
            np.array(node_indices, dtype=np.int64)            # [num_edges, 2]
        )

    def get_node_position(self, node_id: str) -> tuple:
        """
        Get center (x, y) position of a node region by its ID.

        Note: This returns the center of the region.
        For full region bounds, use node.bounds property.
        """
        for node in self.nodes:
            if node.node_id == node_id:
                return (node.center_x, node.center_y)
        return (0.0, 0.0)

    def get_node_by_index(self, idx: int) -> HospitalNode:
        """Get node by index."""
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return self.nodes[0]
