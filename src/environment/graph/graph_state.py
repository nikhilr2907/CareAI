from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .node import HospitalNode
from .edge import HospitalEdge


@dataclass
class GraphState:
    """Represents the ILC pilot graph structure (nodes + edges)."""
    nodes: List[HospitalNode] = field(default_factory=list)
    edges: List[HospitalEdge] = field(default_factory=list)
    sku_database: Optional[dict] = None
    demand_profiles: Optional[dict] = None
    category_order: Optional[List[str]] = None
    location_tag_order: Optional[List[str]] = None
    current_time: float = 0.0
    consumption_scale: float = 1.0
    school_schedule: Optional[dict] = None

    def get_node_features(self) -> np.ndarray:
        """
        Extract node features as numpy array (10 nodes × 8 features).

        Features per node:
        - center_x, center_y: Region center coordinates
        - width, height: Region dimensions
        - stock_level: Current inventory
        - consumption_rate: Items consumed per hour
        - time_to_stockout: Hours until stockout
        - (deprecated) occupancy_count removed from this compact view
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
                node.time_to_stockout
            ]
            features.append(node_features)
        return np.array(features, dtype=np.float32)

    def get_edge_features(self) -> np.ndarray:
        """
        Extract edge features as numpy array (num_edges × 4 features).

        Features per edge:
        - distance_m: Physical corridor length
        - corridor_width: Corridor width (affects capacity)
        - current_weight: Dynamic cost (includes congestion)
        - clutter_level: Static clutter (0-1)
        """
        features = []
        for edge in self.edges:
            edge_features = [
                edge.distance_m,
                edge.corridor_width,
                edge.current_weight,
                edge.clutter_level
            ]
            features.append(edge_features)
        return np.array(features, dtype=np.float32)

    # ===========================================================================
    # ENHANCED FEATURE EXTRACTION (Complete node and edge attributes)
    # ===========================================================================

    def get_node_features_complete(self) -> tuple:
        """
        Extract core node features for models without category stats.

        Returns:
            (continuous_features, categorical_features) where:
            - continuous_features: [num_nodes, 8] numpy array
            - categorical_features: [num_nodes, 1] numpy array (node_type_id)

        Continuous features per node (8 total):
        1. center_x: Region center X coordinate
        2. center_y: Region center Y coordinate
        3. width: Region width (meters)
        4. height: Region height (meters)
        5. area: Region area (width * height)
        6. stock_level: Current inventory items
        7. consumption_rate: Items consumed per hour
        8. time_to_stockout: Hours until stockout (inf if not consuming)

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
            node_continuous = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,
                node.stock_level,
                node.consumption_rate,
                min(node.time_to_stockout, 999.0)
            ]
            continuous_features.append(node_continuous)

            node_type_id = node_type_to_id.get(node.node_type, 0)
            categorical_features.append([node_type_id])

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(categorical_features, dtype=np.int64)
        )

    def _get_school_period_id(self, time_seconds: float) -> int:
        """Return index (0-N) of the active school period, or -1 if outside schedule."""
        schedule = self.school_schedule
        if not schedule:
            return -1
        periods = schedule.get("periods", [])
        if not periods:
            return -1
        first_start = periods[0]["start_time"]
        hh, mm = first_start.split(":")
        episode_start_h = int(hh) + int(mm) / 60.0
        wall_h = (episode_start_h + (time_seconds % 86400) / 3600.0) % 24.0
        for idx, period in enumerate(periods):
            s_hh, s_mm = period["start_time"].split(":")
            e_hh, e_mm = period["end_time"].split(":")
            s = int(s_hh) + int(s_mm) / 60.0
            e = int(e_hh) + int(e_mm) / 60.0
            if s <= wall_h < e:
                return idx
        return -1

    def _get_day_type_id(self, time_seconds: float) -> int:
        """0 = weekday (Mon-Fri), 1 = weekend (Sat-Sun)."""
        day_of_week = int((time_seconds / 86400.0) % 7)
        return 1 if day_of_week >= 5 else 0

    def _get_period_demand_weight(self, time_seconds: float) -> float:
        """Fraction of daily demand in the current school period (0–1). 0 outside schedule."""
        schedule = self.school_schedule
        if not schedule:
            return 0.0
        periods = schedule.get("periods", [])
        if not periods:
            return 0.0
        first_start = periods[0]["start_time"]
        hh, mm = first_start.split(":")
        episode_start_h = int(hh) + int(mm) / 60.0
        wall_h = (episode_start_h + (time_seconds % 86400) / 3600.0) % 24.0
        for period in periods:
            s_hh, s_mm = period["start_time"].split(":")
            e_hh, e_mm = period["end_time"].split(":")
            s = int(s_hh) + int(s_mm) / 60.0
            e = int(e_hh) + int(e_mm) / 60.0
            if s <= wall_h < e:
                # Look up the weight from the first SKU that has a school_period_profile
                # as a proxy for the current demand intensity
                period_name = period["name"]
                for node in self.nodes:
                    if not node.sku_inventory:
                        continue
                    sku_db = self.sku_database or {}
                    for sku_id in node.sku_inventory:
                        profile = sku_db.get(sku_id, {}).get(
                            "consumption_model", {}
                        ).get("school_period_profile")
                        if profile and period_name in profile:
                            return float(profile[period_name])
                return 0.0
        return 0.0

    def get_node_features_with_category_stats(self) -> tuple:
        """
        Extract node features with category-level inventory stats.

        Categorical embeddings for:
        - node_type (storage, corridor, recovery, hub)
        - department_id (department tag index)
        - shift_period (night, morning, afternoon, evening)
        - day_type (weekday, weekend)

        Returns:
            (continuous_features, categorical_features)

        Continuous features (per node):
            - Geometry: center_x, center_y, width, height, area
            - Inventory: stock_level, consumption_rate, time_to_stockout
            - Context: foot_traffic_weight, num_skus, num_categories
            - Temporal: period_demand_weight
            - Per-category: stock_ratio, consumption_rate, time_to_stockout (N categories)

        Categorical features [num_nodes, 4]:
            - node_type_id: 0-3 (storage, corridor, recovery, hub)
            - loc_tag_id: index in location_tag_order list (-1 if unknown)
            - school_period_id: 0-N index into school_schedule periods
            - day_type_id: 0-1 (weekday, weekend)
        """
        node_type_to_id = {
            'storage': 0,
            'corridor': 1,
            'recovery': 2,
            'hub': 3
        }

        category_order = self.category_order or []

        school_period_id = self._get_school_period_id(self.current_time)
        day_type_id = self._get_day_type_id(self.current_time)
        period_demand_weight = self._get_period_demand_weight(self.current_time)

        location_tag_order = self.location_tag_order or []

        continuous_features = []
        categorical_features = []

        for node in self.nodes:
            loc_tag_id = location_tag_order.index(node.location_tag) if node.location_tag in location_tag_order else -1

            base = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,
                node.stock_level,
                node.consumption_rate,
                min(node.time_to_stockout, 999.0),
                float(node.foot_traffic_weight),
                float(node.floor),
                float(node.get_total_num_skus()),
                float(len(node.category_inventory)),
                float(1.0 if node.consumption_enabled else 0.0),
            ]

            temporal_features = [
                period_demand_weight,
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
            categorical_features.append([
                node_type_id,
                loc_tag_id,
                school_period_id,
                day_type_id
            ])

            continuous_features.append(base + temporal_features + cat_feats)

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(categorical_features, dtype=np.int64)
        )

    def get_edge_features_complete(self) -> tuple:
        """
        Extract COMPLETE edge features including topology and node connectivity.

        Returns:
            (continuous_features, node_indices) where:
            - continuous_features: [num_edges, 20] numpy array
            - node_indices: [num_edges, 2] numpy array (from_node_idx, to_node_idx)

        Continuous features per edge (21 total):
        1. distance_m: Physical corridor length
        2. corridor_width: Physical width (meters)
        3. max_v_ms: Max speed allowed (1.0 m/s)
        4. entry_point_x: Entry coordinate X
        5. entry_point_y: Entry coordinate Y
        6. exit_point_x: Exit coordinate X
        7. exit_point_y: Exit coordinate Y
        8. clutter_level: Static clutter (0-1)
        9. num_active_robots: Count of robots on corridor
        10. current_weight: Dynamic cost (base + congestion)
        11. base_cost: Base travel time (distance / max_v_ms)
        12. floor_delta: Floor change for edge (signed)
        13. mode_id: Encoded travel mode (0=unknown, 1=walk, 2=lift, 3=stairs)
        14. same_direction_robots: Robots moving along edge direction
        15. opposite_direction_robots: Robots moving opposite edge direction
        16. approaching_robots: Robots planning to enter this edge
        17. people_count: Estimated people in corridor
        18. congestion_factor: num_active / corridor_capacity
        19. corridor_capacity: width / 0.6 (assumes 60cm per robot)
        20. is_congested: 1.0 if congestion_factor > 0.8 else 0.0

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
        idx_to_id = {idx: node.node_id for idx, node in enumerate(self.nodes)}

        for edge in self.edges:
            # Continuous features (12)
            entry_x = edge.entry_point[0] if edge.entry_point else 0.0
            entry_y = edge.entry_point[1] if edge.entry_point else 0.0
            exit_x = edge.exit_point[0] if edge.exit_point else 0.0
            exit_y = edge.exit_point[1] if edge.exit_point else 0.0

            base_cost = edge.distance_m / edge.max_v_ms
            corridor_capacity = edge.corridor_width / 0.6
            num_active = float(len(edge.active_robot_ids))
            congestion_factor = num_active / max(1.0, corridor_capacity)
            is_congested = 1.0 if congestion_factor > 0.8 else 0.0

            same_dir = 0.0
            opposite_dir = 0.0
            for _, (_, from_idx, to_idx) in edge.active_robot_progress.items():
                from_id = idx_to_id.get(from_idx)
                to_id = idx_to_id.get(to_idx)
                if from_id == edge.from_node and to_id == edge.to_node:
                    same_dir += 1.0
                else:
                    opposite_dir += 1.0

            edge_continuous = [
                edge.distance_m,
                edge.corridor_width,
                edge.max_v_ms,
                entry_x,
                entry_y,
                exit_x,
                exit_y,
                edge.clutter_level,
                num_active,
                edge.current_weight,  # Dynamic weight (includes congestion)
                base_cost,  # Base travel time without congestion
                float(getattr(edge, "floor_delta", 0)),
                float(mode_to_id.get(getattr(edge, "mode", None), 0)),
                same_dir,
                opposite_dir,
                float(getattr(edge, "approaching_robot_count", 0)),
                float(edge.people_count),
                float(congestion_factor),
                float(corridor_capacity),
                float(is_congested),
            ]
            continuous_features.append(edge_continuous)

            # Node connectivity (2 indices)
            from_idx = node_id_to_idx.get(edge.from_node, 0)
            to_idx = node_id_to_idx.get(edge.to_node, 0)
            node_indices.append([from_idx, to_idx])

        return (
            np.array(continuous_features, dtype=np.float32),  # [num_edges, 20]
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
                                node.stock_level,  # Total aggregate stock
                node.consumption_rate,  # Total aggregate consumption
                min(node.time_to_stockout, 999.0),
                float(node.occupancy_count),
                float(node.urgency_level),
                float(node.get_total_num_skus()),
                float(len(node.category_inventory)),
                float(node.foot_traffic_weight)
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
            if node.location_tag:
                loc_idx = hash(node.location_tag) % 1000
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
            if edge.clutter_level > 0.5:
                obstacle_type = 1  # Clutter
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

    def get_node_sku_features(self) -> tuple:
        """
        Build per-node SKU feature tensor with global SKU order.

        Returns:
            (sku_features, sku_mask) where:
            - sku_features: [num_nodes, num_skus, sku_feat_dim]
            - sku_mask: [num_nodes, num_skus] (1 if SKU present at node)

        sku_feat_dim = num_categories + 10
        """
        sku_db = self.sku_database or {}
        sku_order = sorted(sku_db.keys())
        category_order = self.category_order or []
        cat_index = {c: i for i, c in enumerate(category_order)}
        num_cat = len(category_order)
        num_nodes = len(self.nodes)
        num_skus = len(sku_order)
        sku_feat_dim = num_cat + 10

        sku_features = np.zeros((num_nodes, num_skus, sku_feat_dim), dtype=np.float32)
        sku_mask = np.zeros((num_nodes, num_skus), dtype=np.float32)

        for ni, node in enumerate(self.nodes):
            for si, sku_id in enumerate(sku_order):
                sku_entry = node.sku_inventory.get(sku_id)
                if not sku_entry:
                    continue
                sku_mask[ni, si] = 1.0

                cat_key = sku_entry.get('category', '')
                if cat_key in cat_index:
                    sku_features[ni, si, cat_index[cat_key]] = 1.0

                stock = float(sku_entry.get('stock', 0.0))
                max_level = float(sku_entry.get('max', 0.0))
                reorder = float(sku_entry.get('reorder', 0.0))
                par = float(sku_entry.get('par', 0.0))
                area_mult = float(sku_entry.get('area_multiplier', 1.0))

                if max_level > 0:
                    stock_ratio = stock / max_level
                    reorder_ratio = reorder / max_level
                    par_ratio = par / max_level
                    max_ratio = 1.0
                else:
                    stock_ratio = reorder_ratio = par_ratio = max_ratio = 0.0

                sku_data = sku_db.get(sku_id, {})
                consumption = sku_data.get('consumption_model', {})
                daily_dist = consumption.get('daily_distribution', {})
                demand_mean = float(daily_dist.get('mean', 0.0))
                demand_k = float(daily_dist.get('dispersion_k', 0.0))
                intraday = consumption.get('intraday_profile', {}) or {}
                intraday_numeric = [float(v) for v in intraday.values() if isinstance(v, (int, float))]
                if intraday_numeric:
                    intraday_peak = float(max(intraday_numeric))
                    intraday_offpeak = float(min(intraday_numeric))
                else:
                    intraday_peak = 0.0
                    intraday_offpeak = 0.0
                weekday = consumption.get('weekday_multiplier', {}) or {}
                weekday_vals = [float(v) for v in weekday.values() if isinstance(v, (int, float))]
                if weekday_vals:
                    mean = sum(weekday_vals) / len(weekday_vals)
                    var = sum((v - mean) ** 2 for v in weekday_vals) / len(weekday_vals)
                    weekday_std = var ** 0.5
                else:
                    weekday_std = 0.0

                offset = num_cat
                sku_features[ni, si, offset + 0] = stock_ratio
                sku_features[ni, si, offset + 1] = reorder_ratio
                sku_features[ni, si, offset + 2] = par_ratio
                sku_features[ni, si, offset + 3] = max_ratio
                sku_features[ni, si, offset + 4] = demand_mean
                sku_features[ni, si, offset + 5] = demand_k
                sku_features[ni, si, offset + 6] = intraday_peak
                sku_features[ni, si, offset + 7] = intraday_offpeak
                sku_features[ni, si, offset + 8] = weekday_std
                sku_features[ni, si, offset + 9] = area_mult

        return sku_features, sku_mask

