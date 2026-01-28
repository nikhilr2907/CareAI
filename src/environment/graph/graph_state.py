from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .node import HospitalNode
from .edge import HospitalEdge
from .hospital_config import HospitalConfig


@dataclass
class GraphState:
    """Represents the hospital graph structure (nodes + edges)."""
    nodes: List[HospitalNode] = field(default_factory=list)
    edges: List[HospitalEdge] = field(default_factory=list)
    config: Optional[HospitalConfig] = None
    sku_database: Optional[dict] = None
    demand_profiles: Optional[dict] = None
    category_order: Optional[List[str]] = None
    department_order: Optional[List[str]] = None
    current_time: float = 0.0
    consumption_scale: float = 1.0

    def __post_init__(self):
        """Create hospital graph from config, or use default if none provided."""
        if not self.nodes:
            if self.config is None:
                # Use default configuration
                self.config = HospitalConfig()
            self._create_nodes_from_config()
        if not self.edges:
            self._create_edges_from_config()

    def _create_nodes_from_config(self):
        """Create nodes from configuration."""
        for node_config in self.config.nodes:
            params = self.config.get_node_params(node_config)
            node = HospitalNode(**params)
            self.nodes.append(node)

    def _create_edges_from_config(self):
        """Create edges from configuration."""
        for from_idx, to_idx in self.config.edges:
            from_node = self.nodes[from_idx]
            to_node = self.nodes[to_idx]

            # Entry/exit points (simplified: use centers)
            entry_point = (from_node.center_x, from_node.center_y)
            exit_point = (to_node.center_x, to_node.center_y)

            # Calculate distance
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
                clutter_level=np.random.random() * 0.3,
                active_robot_ids=[],
                has_patient_bed=np.random.random() < 0.1
            )
            self.edges.append(edge)

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

    def get_node_features_with_category_stats(self) -> tuple:
        """
        Extract node features with category-level inventory stats.

        Returns:
            (continuous_features, categorical_features)
        """
        node_type_to_id = {
            'storage': 0,
            'corridor': 1,
            'recovery': 2,
            'hub': 3
        }

        category_order = self.category_order or []
        department_order = self.department_order or []

        continuous_features = []
        categorical_features = []

        for node in self.nodes:
            dept_id = department_order.index(node.department_tag) if (node.department_tag in department_order) else -1
            base = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,
                node.clearance_m,
                node.max_reach_height,
                node.unit_height,
                float(node.has_wash_basin),
                float(node.is_cluttered),
                node.stock_level,
                node.consumption_rate,
                min(node.time_to_stockout, 999.0),
                float(node.occupancy_count),
                float(node.urgency_level),
                float(node.served_beds),
                float(dept_id),
                float(node.floor),
                float(len(node.location_ids)),
                float(len(node.shelf_ids)),
                float(node.get_total_num_skus()),
                float(len(node.category_inventory)),
                float(1.0 if node.consumption_enabled else 0.0),
            ]

            cat_feats = []
            for cat in category_order:
                stock = node.get_category_stock_level(cat)
                max_stock = node.get_category_max_stock(cat)
                rate = node.get_category_consumption_rate(cat)
                tts = node.get_category_time_to_stockout(cat)
                stock_ratio = (stock / max_stock) if max_stock > 0 else 0.0
                cat_feats.extend([stock_ratio, rate, min(tts, 999.0)])

            node_type_id = node_type_to_id.get(node.node_type, 0)
            categorical_features.append([node_type_id])
            continuous_features.append(base + cat_feats)

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(categorical_features, dtype=np.int64)
        )

    def get_edge_features_complete(self) -> tuple:
        """
        Extract COMPLETE edge features including topology and node connectivity.

        Returns:
            (continuous_features, node_indices) where:
            - continuous_features: [num_edges, 14] numpy array
            - node_indices: [num_edges, 2] numpy array (from_node_idx, to_node_idx)

        Continuous features per edge (14 total):
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
        13. floor_delta: Floor change for edge (signed)
        14. mode_id: Encoded travel mode (0=unknown, 1=walk, 2=lift, 3=stairs)

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

        mode_to_id = {
            None: 0,
            "walk": 1,
            "corridor": 1,
            "lift": 2,
            "elevator": 2,
            "stairs": 3
        }

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
                base_cost,  # Base travel time without congestion
                float(getattr(edge, "floor_delta", 0)),
                float(mode_to_id.get(getattr(edge, "mode", None), 0))
            ]
            continuous_features.append(edge_continuous)

            # Node connectivity (2 indices)
            from_idx = node_id_to_idx.get(edge.from_node, 0)
            to_idx = node_id_to_idx.get(edge.to_node, 0)
            node_indices.append([from_idx, to_idx])

        return (
            np.array(continuous_features, dtype=np.float32),  # [num_edges, 14]
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

    # ===========================================================================
    # ENHANCED FEATURE EXTRACTION WITH CATEGORY EMBEDDINGS
    # ===========================================================================

    def get_node_features_with_categories(
        self,
        category_names: Optional[List[str]] = None,
        max_categories: int = 10
    ) -> tuple:
        """
        Extract node features including category-based inventory with embedding indices.

        This is the ENHANCED version that includes per-category stock information
        and prepares category embedding indices for the neural network.

        Args:
            category_names: List of category names to extract (e.g., ['iv_therapy', 'ppe'])
                           If None, automatically discovers categories from nodes
            max_categories: Maximum number of categories to track (for fixed tensor size)

        Returns:
            tuple of:
            - node_continuous: [num_nodes, N] - continuous features (geometry, total stock, etc.)
            - node_categorical: [num_nodes, 1] - node_type_id
            - category_features: [num_nodes, max_categories, 4] - per-category features
              Each category: [stock_level, max_stock, num_skus, consumption_rate]
            - category_mask: [num_nodes, max_categories] - 1 if category exists, 0 if padding
            - category_ids: [num_nodes, max_categories] - category embedding indices
            - location_ids: [num_nodes] - location embedding indices (or -1 if none)
            - floor_ids: [num_nodes] - floor numbers

        Continuous node features (15 + 3 = 18 total):
        [Same as get_node_features_complete, plus:]
        16. total_num_skus: Total distinct SKUs at this node
        17. num_categories: Number of inventory categories
        18. num_shelves: Number of shelf IDs

        Category features per node (max_categories × 4):
        For each category slot:
        - stock_level: Current stock in this category
        - max_stock: Max capacity for this category
        - num_skus: Number of distinct SKUs in category
        - consumption_rate: Consumption rate for category
        """
        # Auto-discover categories if not provided
        if category_names is None:
            category_set = set()
            for node in self.nodes:
                category_set.update(node.get_all_categories())
            category_names = sorted(list(category_set))

        # Truncate to max_categories if needed
        if len(category_names) > max_categories:
            print(f"Warning: {len(category_names)} categories found, truncating to {max_categories}")
            category_names = category_names[:max_categories]

        # Build category name → ID mapping
        category_to_id = {cat: idx for idx, cat in enumerate(category_names)}

        node_type_to_id = {
            'storage': 0,
            'corridor': 1,
            'recovery': 2,
            'hub': 3
        }

        # Initialize arrays
        num_nodes = len(self.nodes)
        node_continuous = []
        node_categorical = []
        category_features = []
        category_mask = []
        category_ids = []
        location_ids = []
        floor_ids = []

        for node in self.nodes:
            # ===== Continuous features (18) =====
            node_cont = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,
                node.clearance_m,
                node.max_reach_height,
                node.unit_height,
                float(node.has_wash_basin),
                float(node.is_cluttered),
                node.stock_level,  # Total aggregate stock
                node.consumption_rate,  # Total aggregate consumption
                min(node.time_to_stockout, 999.0),
                float(node.occupancy_count),
                float(node.urgency_level),
                float(node.get_total_num_skus()),  # NEW
                float(len(node.category_inventory)),  # NEW: num categories
                float(len(node.shelf_ids))  # NEW: num shelves
            ]
            node_continuous.append(node_cont)

            # ===== Categorical feature (1) =====
            node_type_id = node_type_to_id.get(node.node_type, 0)
            node_categorical.append([node_type_id])

            # ===== Category features (max_categories × 4) =====
            node_cat_features = []
            node_cat_mask = []
            node_cat_ids = []

            for cat_name in category_names:
                if cat_name in node.category_inventory:
                    # Category exists at this node
                    cat_data = [
                        node.get_category_stock_level(cat_name),
                        node.get_category_max_stock(cat_name),
                        float(node.get_category_num_skus(cat_name)),
                        node.get_category_consumption_rate(cat_name)
                    ]
                    node_cat_features.append(cat_data)
                    node_cat_mask.append(1.0)
                    node_cat_ids.append(category_to_id[cat_name])
                else:
                    # Category doesn't exist - pad with zeros
                    node_cat_features.append([0.0, 0.0, 0.0, 0.0])
                    node_cat_mask.append(0.0)
                    node_cat_ids.append(-1)  # -1 = no category

            # Pad to max_categories if needed
            while len(node_cat_features) < max_categories:
                node_cat_features.append([0.0, 0.0, 0.0, 0.0])
                node_cat_mask.append(0.0)
                node_cat_ids.append(-1)

            category_features.append(node_cat_features)
            category_mask.append(node_cat_mask)
            category_ids.append(node_cat_ids)

            # ===== Location and floor IDs =====
            # Location ID: if node.location_id exists, map to index (or use -1)
            # For simplicity, we'll use a hash or assign sequential IDs
            # In practice, you'd maintain a global location_id → index mapping
            if node.location_id:
                # Simple hash to int (replace with proper mapping in production)
                loc_idx = hash(node.location_id) % 1000
            else:
                loc_idx = -1
            location_ids.append(loc_idx)

            floor_ids.append(node.floor)

        return (
            np.array(node_continuous, dtype=np.float32),  # [num_nodes, 18]
            np.array(node_categorical, dtype=np.int64),    # [num_nodes, 1]
            np.array(category_features, dtype=np.float32), # [num_nodes, max_categories, 4]
            np.array(category_mask, dtype=np.float32),     # [num_nodes, max_categories]
            np.array(category_ids, dtype=np.int64),        # [num_nodes, max_categories]
            np.array(location_ids, dtype=np.int64),        # [num_nodes]
            np.array(floor_ids, dtype=np.int64)            # [num_nodes]
        )

    def get_edge_features_enhanced(self) -> tuple:
        """
        Extract ENHANCED edge features including physical constraints.

        Returns:
            (continuous_features, categorical_features, node_indices) where:
            - continuous_features: [num_edges, 17] numpy array
            - categorical_features: [num_edges, 2] numpy array (bidirectional, obstacle_type)
            - node_indices: [num_edges, 2] numpy array (from_node_idx, to_node_idx)

        NEW continuous features (17 total, added 3 more):
        [Same 14 as get_edge_features_complete, plus:]
        15. congestion_factor: len(active_robots) / corridor_capacity
        16. corridor_capacity: width / 0.6 (assumes 60cm per robot)
        17. is_congested: 1.0 if congestion_factor > 0.8 else 0.0

        Categorical features per edge (2 total):
        1. is_bidirectional: 1 (all hospital corridors are bidirectional)
        2. obstacle_type: 0=none, 1=patient_bed, 2=clutter, 3=both
        """
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(self.nodes)}

        continuous_features = []
        categorical_features = []
        node_indices = []

        mode_to_id = {
            None: 0,
            "walk": 1,
            "corridor": 1,
            "lift": 2,
            "elevator": 2,
            "stairs": 3
        }

        for edge in self.edges:
            # ===== Continuous features (15) =====
            entry_x = edge.entry_point[0] if edge.entry_point else 0.0
            entry_y = edge.entry_point[1] if edge.entry_point else 0.0
            exit_x = edge.exit_point[0] if edge.exit_point else 0.0
            exit_y = edge.exit_point[1] if edge.exit_point else 0.0

            base_cost = edge.distance_m / edge.max_v_ms

            # NEW: Congestion metrics
            corridor_capacity = edge.corridor_width / 0.6  # Assume 60cm per robot
            num_active = len(edge.active_robot_ids)
            congestion_factor = num_active / max(1.0, corridor_capacity)
            is_congested = 1.0 if congestion_factor > 0.8 else 0.0

            edge_continuous = [
                edge.distance_m,
                edge.corridor_width,
                edge.max_v_ms,
                entry_x,
                entry_y,
                exit_x,
                exit_y,
                edge.clutter_level,
                float(num_active),
                float(edge.has_patient_bed),
                edge.current_weight,
                base_cost,
                float(getattr(edge, "floor_delta", 0)),
                float(mode_to_id.get(getattr(edge, "mode", None), 0)),
                congestion_factor,  # NEW
                corridor_capacity,  # NEW
                is_congested        # NEW
            ]
            continuous_features.append(edge_continuous)

            # ===== Categorical features (2) =====
            is_bidirectional = 1  # All hospital corridors are bidirectional

            # Obstacle type encoding
            if edge.has_patient_bed and edge.clutter_level > 0.5:
                obstacle_type = 3  # Both bed and clutter
            elif edge.has_patient_bed:
                obstacle_type = 1  # Patient bed
            elif edge.clutter_level > 0.5:
                obstacle_type = 2  # Clutter
            else:
                obstacle_type = 0  # No obstacles

            edge_categorical = [is_bidirectional, obstacle_type]
            categorical_features.append(edge_categorical)

            # ===== Node connectivity (2 indices) =====
            from_idx = node_id_to_idx.get(edge.from_node, 0)
            to_idx = node_id_to_idx.get(edge.to_node, 0)
            node_indices.append([from_idx, to_idx])

        return (
            np.array(continuous_features, dtype=np.float32),  # [num_edges, 17]
            np.array(categorical_features, dtype=np.int64),   # [num_edges, 2]
            np.array(node_indices, dtype=np.int64)            # [num_edges, 2]
        )
