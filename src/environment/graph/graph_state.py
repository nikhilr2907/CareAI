from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .edge import GraphEdge
from .node import GraphNode


@dataclass
class GraphState:
    """Represents the graph structure (nodes + edges)."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    sku_database: Optional[dict] = None
    demand_profiles: Optional[dict] = None
    category_order: Optional[List[str]] = None
    department_order: Optional[List[str]] = None
    current_time: float = 0.0
    consumption_scale: float = 1.0

    def get_node_features(self) -> np.ndarray:
        """Extract compact per-node features: center, dimensions, stock level, consumption rate, time to stockout."""
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
            ]
            features.append(node_features)
        return np.array(features, dtype=np.float32)

    def get_edge_features(self) -> np.ndarray:
        """Extract compact per-edge features: distance, width, dynamic weight, patient bed flag, clutter level."""
        features = []
        for edge in self.edges:
            edge_features = [
                edge.distance_m,
                edge.corridor_width,
                edge.current_weight,
                float(edge.has_patient_bed),
                edge.clutter_level,
            ]
            features.append(edge_features)
        return np.array(features, dtype=np.float32)

    def get_node_features_complete(self) -> tuple:
        """Extract core (continuous, categorical node_type_id) node features for models without category stats."""
        node_type_to_id = {"storage": 0, "corridor": 1, "recovery": 2, "hub": 3}

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
                min(node.time_to_stockout, 999.0),
            ]
            continuous_features.append(node_continuous)

            node_type_id = node_type_to_id.get(node.node_type, 0)
            categorical_features.append([node_type_id])

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(categorical_features, dtype=np.int64),
        )

    def _get_shift_period_id(self, time_seconds: float) -> int:
        """Compute the shift period id (0=night, 1=morning, 2=afternoon, 3=evening) from time of day."""
        hour_of_day = (time_seconds % 86400) / 3600.0

        if hour_of_day < 6.0:
            return 0  # night
        elif hour_of_day < 12.0:
            return 1  # morning
        elif hour_of_day < 18.0:
            return 2  # afternoon
        else:
            return 3  # evening

    def _get_day_type_id(self, time_seconds: float) -> int:
        """Compute the day type id (0=weekday, 1=weekend); assumes time_seconds starts at Monday midnight."""
        day_of_week = int((time_seconds / 86400.0) % 7)  # 0=Monday, 6=Sunday

        if day_of_week >= 5:  # Saturday or Sunday
            return 1  # weekend
        else:
            return 0  # weekday

    def _get_intraday_weight(self, time_seconds: float) -> float:
        """Compute the intraday consumption weight (0.068-0.259); must match task_generator._intraday_weight()."""
        hour_of_day = (time_seconds % 86400) / 3600.0

        # Demand profile: low at night, high in morning/afternoon
        if hour_of_day < 6.0:  # Night (00:00-06:00)
            return 0.068
        elif hour_of_day < 12.0:  # Morning (06:00-12:00)
            return 0.259
        elif hour_of_day < 18.0:  # Afternoon (12:00-18:00)
            return 0.239
        else:  # Evening (18:00-24:00)
            return 0.183

    def _get_weekday_multiplier(self, time_seconds: float) -> float:
        """Compute the weekday consumption multiplier (0.95-1.03); must match task_generator._weekday_multiplier()."""
        day_of_week = int((time_seconds / 86400.0) % 7)  # 0=Monday, 6=Sunday

        # Tuesday/Wednesday highest, weekend lowest
        weekday_multipliers = {
            0: 1.01,  # Monday
            1: 1.03,  # Tuesday
            2: 1.02,  # Wednesday
            3: 1.00,  # Thursday
            4: 0.98,  # Friday
            5: 0.95,  # Saturday
            6: 0.96,  # Sunday
        }

        return weekday_multipliers.get(day_of_week, 1.0)

    def get_node_features_with_category_stats(self) -> tuple:
        """Extract node features (geometry, inventory, temporal) plus per-category stats and categorical embeddings."""
        node_type_to_id = {"storage": 0, "corridor": 1, "recovery": 2, "hub": 3}

        category_order = self.category_order or []
        department_order = self.department_order or []

        shift_period_id = self._get_shift_period_id(self.current_time)
        day_type_id = self._get_day_type_id(self.current_time)
        intraday_weight = self._get_intraday_weight(self.current_time)
        weekday_multiplier = self._get_weekday_multiplier(self.current_time)

        continuous_features = []
        categorical_features = []

        for node in self.nodes:
            dept_id = (
                department_order.index(node.department_tag)
                if (node.department_tag in department_order)
                else -1
            )

            base = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,
                node.stock_level,
                node.consumption_rate,
                min(node.time_to_stockout, 999.0),
                float(node.served_beds),
                float(node.floor),
                float(len(node.location_ids)),
                float(len(node.shelf_ids)),
                float(node.get_total_num_skus()),
                float(len(node.category_inventory)),
                float(1.0 if node.consumption_enabled else 0.0),
            ]

            temporal_features = [
                intraday_weight,
                weekday_multiplier,
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
            categorical_features.append([node_type_id, dept_id, shift_period_id, day_type_id])

            continuous_features.append(base + temporal_features + cat_feats)

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(categorical_features, dtype=np.int64),
        )

    def get_edge_features_complete(self) -> tuple:
        """Extract full per-edge continuous features (geometry, congestion, mode) plus from/to node indices for the GNN."""
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(self.nodes)}

        continuous_features = []
        node_indices = []

        mode_to_id = {None: 0, "walk": 1, "corridor": 1, "lift": 2, "elevator": 2, "stairs": 3}
        idx_to_id = {idx: node.node_id for idx, node in enumerate(self.nodes)}

        for edge in self.edges:
            entry_x = edge.entry_point[0] if edge.entry_point else 0.0
            entry_y = edge.entry_point[1] if edge.entry_point else 0.0
            exit_x = edge.exit_point[0] if edge.exit_point else 0.0
            exit_y = edge.exit_point[1] if edge.exit_point else 0.0

            base_cost = edge.distance_m / edge.max_v_ms
            num_active = float(len(edge.active_robot_ids))
            corridor_capacity = edge.jam_density()
            congestion_factor = edge.congestion_factor()  # matches simulator physics exactly
            is_congested = 1.0 if congestion_factor > 0.8 else 0.0

            same_dir = 0.0
            opposite_dir = 0.0
            for _, (_, from_idx, to_idx) in edge.active_robot_progress.items():
                from_id = idx_to_id.get(from_idx) if from_idx is not None else None
                to_id = idx_to_id.get(to_idx) if to_idx is not None else None
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
                float(edge.has_patient_bed),
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

            from_idx = node_id_to_idx.get(edge.from_node, 0)
            to_idx = node_id_to_idx.get(edge.to_node, 0)
            node_indices.append([from_idx, to_idx])

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(node_indices, dtype=np.int64),
        )

    def get_node_position(self, node_id: str) -> tuple:
        """Get center (x, y) position of a node region by its ID; use node.bounds for full region bounds."""
        for node in self.nodes:
            if node.node_id == node_id:
                return (node.center_x, node.center_y)
        return (0.0, 0.0)

    def get_node_by_index(self, idx: int) -> GraphNode:
        if 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return self.nodes[0]

    def get_node_features_with_categories(
        self, category_names: Optional[List[str]] = None, max_categories: int = 10
    ) -> tuple:
        """Extract node features plus per-category stock stats and embedding indices (category/location/floor) for the network."""
        if category_names is None:
            category_set = set()
            for node in self.nodes:
                category_set.update(node.get_all_categories())
            category_names = sorted(list(category_set))

        if len(category_names) > max_categories:
            print(
                f"Warning: {len(category_names)} categories found, truncating to {max_categories}"
            )
            category_names = category_names[:max_categories]

        category_to_id = {cat: idx for idx, cat in enumerate(category_names)}

        node_type_to_id = {"storage": 0, "corridor": 1, "recovery": 2, "hub": 3}

        len(self.nodes)
        node_continuous = []
        node_categorical = []
        category_features = []
        category_mask = []
        category_ids = []
        location_ids = []
        floor_ids = []

        for node in self.nodes:
            node_cont = [
                node.center_x,
                node.center_y,
                node.width,
                node.height,
                node.area,
                node.stock_level,
                node.consumption_rate,
                min(node.time_to_stockout, 999.0),
                float(node.occupancy_count),
                float(node.urgency_level),
                float(node.get_total_num_skus()),
                float(len(node.category_inventory)),
                float(len(node.shelf_ids)),
            ]
            node_continuous.append(node_cont)

            node_type_id = node_type_to_id.get(node.node_type, 0)
            node_categorical.append([node_type_id])

            node_cat_features = []
            node_cat_mask = []
            node_cat_ids = []

            for cat_name in category_names:
                if cat_name in node.category_inventory:
                    cat_data = [
                        node.get_category_stock_level(cat_name),
                        node.get_category_max_stock(cat_name),
                        float(node.get_category_num_skus(cat_name)),
                        node.get_category_consumption_rate(cat_name),
                    ]
                    node_cat_features.append(cat_data)
                    node_cat_mask.append(1.0)
                    node_cat_ids.append(category_to_id[cat_name])
                else:
                    node_cat_features.append([0.0, 0.0, 0.0, 0.0])
                    node_cat_mask.append(0.0)
                    node_cat_ids.append(-1)  # -1 = no category

            while len(node_cat_features) < max_categories:
                node_cat_features.append([0.0, 0.0, 0.0, 0.0])
                node_cat_mask.append(0.0)
                node_cat_ids.append(-1)

            category_features.append(node_cat_features)
            category_mask.append(node_cat_mask)
            category_ids.append(node_cat_ids)

            # Placeholder: hashes location_id to an int rather than a real global mapping.
            if node.location_id:
                loc_idx = hash(node.location_id) % 1000
            else:
                loc_idx = -1
            location_ids.append(loc_idx)

            floor_ids.append(node.floor)

        return (
            np.array(node_continuous, dtype=np.float32),
            np.array(node_categorical, dtype=np.int64),
            np.array(category_features, dtype=np.float32),
            np.array(category_mask, dtype=np.float32),
            np.array(category_ids, dtype=np.int64),
            np.array(location_ids, dtype=np.int64),
            np.array(floor_ids, dtype=np.int64),
        )

    def get_edge_features_enhanced(self) -> tuple:
        """Extract edge features (continuous geometry/congestion, categorical bidirectional+obstacle_type) and node indices."""
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(self.nodes)}

        continuous_features = []
        categorical_features = []
        node_indices = []

        mode_to_id = {None: 0, "walk": 1, "corridor": 1, "lift": 2, "elevator": 2, "stairs": 3}

        for edge in self.edges:
            entry_x = edge.entry_point[0] if edge.entry_point else 0.0
            entry_y = edge.entry_point[1] if edge.entry_point else 0.0
            exit_x = edge.exit_point[0] if edge.exit_point else 0.0
            exit_y = edge.exit_point[1] if edge.exit_point else 0.0

            base_cost = edge.distance_m / edge.max_v_ms

            num_active = len(edge.active_robot_ids)
            corridor_capacity = edge.jam_density()
            congestion_factor = edge.congestion_factor()  # matches simulator physics exactly
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
                congestion_factor,
                corridor_capacity,
                is_congested,
            ]
            continuous_features.append(edge_continuous)

            is_bidirectional = 1  # All hospital corridors are bidirectional

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

            from_idx = node_id_to_idx.get(edge.from_node, 0)
            to_idx = node_id_to_idx.get(edge.to_node, 0)
            node_indices.append([from_idx, to_idx])

        return (
            np.array(continuous_features, dtype=np.float32),
            np.array(categorical_features, dtype=np.int64),
            np.array(node_indices, dtype=np.int64),
        )

    def get_node_sku_features(self) -> tuple:
        """Build the (sku_features, sku_mask) tensors over the global SKU order, one row per node per SKU."""
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

                cat_key = sku_entry.get("category", "")
                if cat_key in cat_index:
                    sku_features[ni, si, cat_index[cat_key]] = 1.0

                stock = float(sku_entry.get("stock", 0.0))
                max_level = float(sku_entry.get("max", 0.0))
                reorder = float(sku_entry.get("reorder", 0.0))
                par = float(sku_entry.get("par", 0.0))
                area_mult = float(sku_entry.get("area_multiplier", 1.0))

                if max_level > 0:
                    stock_ratio = stock / max_level
                    reorder_ratio = reorder / max_level
                    par_ratio = par / max_level
                    max_ratio = 1.0
                else:
                    stock_ratio = reorder_ratio = par_ratio = max_ratio = 0.0

                sku_data = sku_db.get(sku_id, {})
                consumption = sku_data.get("consumption_model", {})
                daily_dist = consumption.get("daily_distribution", {})
                demand_mean = float(daily_dist.get("mean", 0.0))
                demand_k = float(daily_dist.get("dispersion_k", 0.0))
                intraday = consumption.get("intraday_profile", {}) or {}
                if intraday:
                    weights = list(intraday.values())
                    intraday_peak = float(max(weights))
                    intraday_offpeak = float(min(weights))
                else:
                    intraday_peak = 0.0
                    intraday_offpeak = 0.0
                weekday = consumption.get("weekday_multiplier", {}) or {}
                if weekday:
                    vals = list(weekday.values())
                    mean = sum(vals) / max(len(vals), 1)
                    var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
                    weekday_std = var**0.5
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
