"""
Graph Neural Network encoders for GAPO (Graph Attention-Based Policy Optimization).

Implements:
- Hospital Graph Encoder (nodes + edges)
- Robot Fleet Encoder (dynamic proximity graph)
- Task Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


from torch_geometric.nn import GATConv, SAGEConv


class HospitalGraphEncoder(nn.Module):
    """
    Encodes hospital graph structure using Graph Attention Networks with TWO-PASS encoding.

    Pass 1: Encode nodes from continuous + categorical features
    Pass 2: Augment edges with from/to node embeddings, then refine node embeddings

    Input: Node features (continuous + categorical), edge features, edge-node connectivity
    Output: Node embeddings + edge embeddings + global graph embedding
    """

    def __init__(
        self,
        node_continuous_dim=8,
        num_node_types=4,
        num_departments=10,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=21,
        hidden_dim=64,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4
    ):
        """
        Initialize Hospital Graph Encoder with FINE-GRAINED categorical embeddings.

        Args:
            node_continuous_dim: Continuous node features (default 8)
                [center_x, center_y, width, height, area, clearance_m, max_reach_height,
                 unit_height, has_wash_basin, is_cluttered, stock_level, consumption_rate,
                 time_to_stockout, occupancy_count, urgency_level]
            num_node_types: Number of node type categories (default 4)
                [storage, corridor, recovery, hub]
            num_departments: Number of hospital departments (default 10)
                [e.g., ICU, Emergency, Surgery, etc.]
            num_shift_periods: Number of shift periods (default 4)
                [night, morning, afternoon, evening]
            num_day_types: Number of day types (default 2)
                [weekday, weekend]
            edge_feat_dim: Edge continuous features (default 12)
                [distance_m, corridor_width, max_v_ms, entry_x, entry_y, exit_x, exit_y,
                 clutter_level, num_active_robots, has_patient_bed, current_weight, base_cost]
            hidden_dim: Hidden embedding dimension (default 64)
            node_type_embedding_dim: Dimension for node_type embeddings (default 8)
            department_embedding_dim: Dimension for department embeddings (default 16)
            shift_embedding_dim: Dimension for shift period embeddings (default 4)
            day_type_embedding_dim: Dimension for day type embeddings (default 4)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.node_type_embedding_dim = node_type_embedding_dim
        self.department_embedding_dim = department_embedding_dim
        self.shift_embedding_dim = shift_embedding_dim
        self.day_type_embedding_dim = day_type_embedding_dim

        # Input normalization for continuous features
        self.node_input_norm = nn.LayerNorm(node_continuous_dim)
        self.edge_input_norm = nn.LayerNorm(edge_feat_dim)

        # Categorical embeddings for each feature
        self.node_type_embedding = nn.Embedding(num_node_types, node_type_embedding_dim)
        self.department_embedding = nn.Embedding(num_departments + 1, department_embedding_dim)  # +1 for -1 (unknown)
        self.shift_embedding = nn.Embedding(num_shift_periods, shift_embedding_dim)
        self.day_type_embedding = nn.Embedding(num_day_types, day_type_embedding_dim)

        # Total node feature dimension after concatenating continuous + all embedded categoricals
        total_embedding_dim = (
            node_type_embedding_dim +      # 8
            department_embedding_dim +      # 16
            shift_embedding_dim +           # 4
            day_type_embedding_dim          # 4
        )  # = 32

        node_input_dim = node_continuous_dim + total_embedding_dim  # 8 + 32 = 40 (plus SKU/category expansions if used)

        self.gat1 = GATConv(
            in_channels=node_input_dim,
            out_channels=hidden_dim // 4,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim
        )

        # Edge feature dimension after augmentation:
        # Original 12 + from_node_embedding (64) + to_node_embedding (64) = 140
        augmented_edge_dim = edge_feat_dim + hidden_dim + hidden_dim

        # PASS 2: Refine nodes with augmented edge features
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=augmented_edge_dim
        )

        # Edge feature encoder (for final edge embeddings)
        self.edge_encoder = nn.Sequential(
            nn.Linear(augmented_edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(
        self,
        node_continuous: torch.Tensor,
        node_categorical: torch.Tensor,
        edge_features: torch.Tensor,
        edge_node_indices: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-pass GNN encoding with node-embedded edges.

        Args:
            node_continuous: [num_nodes, 24] - continuous node features (UPDATED with temporal)
            node_categorical: [num_nodes, 4] - categorical IDs:
                [:, 0] = node_type_id (0-3)
                [:, 1] = department_id (0-9 or -1 for unknown)
                [:, 2] = shift_period_id (0-3)
                [:, 3] = day_type_id (0-1)
            edge_features: [num_edges, 21] - continuous edge features
            edge_node_indices: [num_edges, 2] - (from_node_idx, to_node_idx) for each edge
            edge_index: [2, num_edges] - graph connectivity (for GNN message passing)

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            edge_embeddings: [num_edges, hidden_dim]
            graph_embedding: [hidden_dim]
        """
        # Normalize continuous inputs to stabilize GNN attention
        node_continuous = self.node_input_norm(node_continuous)
        edge_features = self.edge_input_norm(edge_features)

        # Embed all categorical features
        node_type_embeds = self.node_type_embedding(node_categorical[:, 0])  # [num_nodes, 8]

        # Department embedding: map -1 (unknown) to last index
        dept_ids = node_categorical[:, 1].clone()
        dept_ids[dept_ids == -1] = self.department_embedding.num_embeddings - 1
        dept_embeds = self.department_embedding(dept_ids)  # [num_nodes, 16]

        shift_embeds = self.shift_embedding(node_categorical[:, 2])  # [num_nodes, 4]
        day_type_embeds = self.day_type_embedding(node_categorical[:, 3])  # [num_nodes, 4]

        # Concatenate continuous + all embedded categoricals
        node_features = torch.cat([
            node_continuous,      # [num_nodes, ?]
            node_type_embeds,     # [num_nodes, 8]
            dept_embeds,          # [num_nodes, 16]
            shift_embeds,         # [num_nodes, 4]
            day_type_embeds       # [num_nodes, 4]
        ], dim=-1)

        # Validate edge_node_indices format
        if edge_node_indices.dim() != 2:
            raise ValueError(f"edge_node_indices must be 2D, got {edge_node_indices.shape}")

        if edge_node_indices.shape[1] != 2:
            raise ValueError(f"edge_node_indices must be [num_edges, 2], got {edge_node_indices.shape}")

        # Validate edge_index format (should be [2, num_edges_bidirectional])
        if edge_index is None:
            raise ValueError("edge_index must be provided for GNN message passing")

        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, num_edges], got {edge_index.shape}")

        # PASS 1: Initial node encoding
        x = self.gat1(node_features, edge_index, edge_attr=edge_features)
        x = F.elu(x)
        node_embeddings_pass1 = x  # [num_nodes, 64]

        # Augment edges with from/to node embeddings
        # Validate edge_node_indices are in valid range
        num_nodes_total = node_embeddings_pass1.shape[0]
        if edge_node_indices.numel() > 0:
            max_idx = edge_node_indices.max().item()
            min_idx = edge_node_indices.min().item()
            if max_idx >= num_nodes_total or min_idx < 0:
                raise IndexError(
                    f"edge_node_indices out of bounds: range [{min_idx}, {max_idx}] "
                    f"but node_embeddings has only {num_nodes_total} nodes. "
                    f"edge_node_indices shape: {edge_node_indices.shape}, "
                    f"node_embeddings shape: {node_embeddings_pass1.shape}"
                )

        from_node_embeds = node_embeddings_pass1[edge_node_indices[:, 0]]  # [num_edges, 64]
        to_node_embeds = node_embeddings_pass1[edge_node_indices[:, 1]]    # [num_edges, 64]

        augmented_edge_features = torch.cat([
            edge_features,      # [num_edges, 21]
            from_node_embeds,   # [num_edges, 64]
            to_node_embeds      # [num_edges, 64]
        ], dim=-1)  # [num_edges, 140]

        # PASS 2: Refine nodes with augmented edges
        node_embeddings = self.gat2(
            node_embeddings_pass1,
            edge_index,
            edge_attr=augmented_edge_features
        )  # [num_nodes, 64]

        # Final edge embeddings
        edge_embeddings = self.edge_encoder(augmented_edge_features)  # [num_edges, 64]

        # Global graph embedding (mean pooling)
        graph_embedding = torch.mean(node_embeddings, dim=0)

        return node_embeddings, edge_embeddings, graph_embedding


class RobotFleetEncoder(nn.Module):
    """
    Encodes robot fleet using dynamic proximity graph.

    Input: Robot features, robot positions
    Output: Robot embeddings + fleet embedding
    """

    def __init__(self, robot_feat_dim=20, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input normalization for robot features
        self.robot_input_norm = nn.LayerNorm(robot_feat_dim)

        # GraphSAGE for variable number of robots
        self.sage1 = SAGEConv(
            in_channels=robot_feat_dim,
            out_channels=hidden_dim,
            aggr='mean'
        )

        self.sage2 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr='mean'
        )

    def forward(
        self,
        robot_features: torch.Tensor,
        robot_positions: Optional[torch.Tensor] = None,
        proximity_threshold: float = 10.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            robot_features: [num_robots, robot_feat_dim]
            robot_positions: [num_robots, 2] - (x, y) positions
            proximity_threshold: Distance threshold for proximity graph

        Returns:
            robot_embeddings: [num_robots, hidden_dim]
            fleet_embedding: [hidden_dim]
        """
        # Normalize robot features
        robot_features = self.robot_input_norm(robot_features)

        if robot_features.dim() == 3:
            batch_size, num_robots, feat_dim = robot_features.shape
            flat_features = robot_features.view(batch_size * num_robots, feat_dim)

            if robot_positions is not None:
                edge_indices = []
                for b in range(batch_size):
                    edge_index_b = self._build_proximity_graph(
                        robot_positions[b],
                        threshold=proximity_threshold
                    )
                    edge_index_b = edge_index_b + (b * num_robots)
                    edge_indices.append(edge_index_b)
                edge_index = torch.cat(edge_indices, dim=1)
            else:
                idx = torch.arange(
                    batch_size * num_robots,
                    device=robot_features.device
                )
                edge_index = torch.stack([idx, idx], dim=0)

            x = self.sage1(flat_features, edge_index)
            x = F.relu(x)
            robot_embeddings = self.sage2(x, edge_index)
            robot_embeddings = robot_embeddings.view(batch_size, num_robots, self.hidden_dim)

            fleet_embedding = torch.mean(robot_embeddings, dim=1)
            return robot_embeddings, fleet_embedding

        if robot_positions is not None:
            # Build proximity graph
            edge_index = self._build_proximity_graph(
                robot_positions,
                threshold=proximity_threshold
            )
        else:
            num_robots = robot_features.shape[0]
            edge_index = torch.arange(num_robots, device=robot_features.device)
            edge_index = torch.stack([edge_index, edge_index], dim=0)

        # GNN encoding

        x = self.sage1(robot_features, edge_index)
        x = F.relu(x)
        robot_embeddings = self.sage2(x, edge_index)

        # Global fleet embedding (mean pooling)
        fleet_embedding = torch.mean(robot_embeddings, dim=0)

        return robot_embeddings, fleet_embedding

    def _build_proximity_graph(
        self,
        positions: torch.Tensor,
        threshold: float = 10.0
    ) -> torch.Tensor:
        """
        Build edges between robots within threshold distance.

        Args:
            positions: [num_robots, 2]
            threshold: Distance threshold

        Returns:
            edge_index: [2, num_edges]
        """
        num_robots = positions.shape[0]
        edges = []

        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                dist = torch.norm(positions[i] - positions[j])
                if dist < threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # No connections - create self-loops
            edge_index = torch.tensor(
                [[i, i] for i in range(num_robots)],
                dtype=torch.long
            ).t().contiguous()

        return edge_index.to(positions.device)


class TaskEncoder(nn.Module):
    """
    Encodes task features into embedding space.

    Input: Task features (15 dims)
    Output: Task embedding
    """

    def __init__(self, task_feat_dim=15, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_features: [task_feat_dim] or [batch_size, task_feat_dim]

        Returns:
            task_embedding: [hidden_dim] or [batch_size, hidden_dim]
        """
        return self.encoder(task_features)


def test_encoders():
    """Test GNN encoders with dummy data."""
    print("Testing GAPO GNN Encoders...")

    # Test Hospital Graph Encoder (with complete features)
    print("\n1. Hospital Graph Encoder (Two-Pass with Fine-Grained Categorical Embeddings)")
    hospital_encoder = HospitalGraphEncoder(
        node_continuous_dim=24,  # UPDATED: includes temporal features
        num_node_types=4,
        num_departments=10,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=21,
        hidden_dim=64,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4
    )

    node_continuous = torch.randn(10, 24)  # 10 nodes, 24 continuous features (UPDATED)
    node_categorical = torch.randint(0, 4, (10, 4))  # 10 nodes, 4 categorical IDs (UPDATED)
    # node_categorical[:, 0] = node_type (0-3)
    # node_categorical[:, 1] = dept_id (0-9, or -1)
    # node_categorical[:, 2] = shift_period (0-3)
    # node_categorical[:, 3] = day_type (0-1)
    edge_features = torch.randn(20, 21)  # 20 edges, 21 continuous features
    edge_node_indices = torch.randint(0, 10, (20, 2))  # Edge-node connectivity
    edge_index = torch.randint(0, 10, (2, 20))  # Graph connectivity

    node_embeds, edge_embeds, graph_embed = hospital_encoder(
        node_continuous, node_categorical, edge_features, edge_node_indices, edge_index
    )

    print(f"  Node embeddings: {node_embeds.shape}")
    print(f"  Edge embeddings: {edge_embeds.shape}")
    print(f"  Graph embedding: {graph_embed.shape}")
    print(f"  Node input: 24 continuous + 8 node_type + 16 dept + 4 shift + 4 day_type = 56 dims")
    print(f"  Edge augmentation: 21 continuous + 64 from + 64 to = 149 dims")

    # Test Robot Fleet Encoder
    print("\n2. Robot Fleet Encoder")
    robot_encoder = RobotFleetEncoder(
        robot_feat_dim=20,
        hidden_dim=64
    )

    robot_features = torch.randn(5, 12)  # 5 robots
    robot_positions = torch.randn(5, 2)  # (x, y) positions

    robot_embeds, fleet_embed = robot_encoder(
        robot_features, robot_positions
    )

    print(f"  Robot embeddings: {robot_embeds.shape}")
    print(f"  Fleet embedding: {fleet_embed.shape}")

    # Test Task Encoder
    print("\n3. Task Encoder")
    task_encoder = TaskEncoder(task_feat_dim=15, hidden_dim=64)

    task_features = torch.randn(15)
    task_embed = task_encoder(task_features)

    print(f"  Task embedding: {task_embed.shape}")

    print("\nAll encoders working!")


