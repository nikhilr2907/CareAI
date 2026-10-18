from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GraphEncoder(nn.Module):
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
        day_type_embedding_dim=4,
    ):
        """Initialize the hospital graph encoder with fine-grained categorical embeddings for node type, department, shift, and day type."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.node_type_embedding_dim = node_type_embedding_dim
        self.department_embedding_dim = department_embedding_dim
        self.shift_embedding_dim = shift_embedding_dim
        self.day_type_embedding_dim = day_type_embedding_dim

        self.node_input_norm = nn.LayerNorm(node_continuous_dim)
        self.edge_input_norm = nn.LayerNorm(edge_feat_dim)

        self.node_type_embedding = nn.Embedding(num_node_types, node_type_embedding_dim)
        self.department_embedding = nn.Embedding(
            num_departments + 1, department_embedding_dim
        )  # +1 for -1 (unknown)
        self.shift_embedding = nn.Embedding(
            num_shift_periods + 1, shift_embedding_dim
        )  # +1 for outside-schedule slot
        self.day_type_embedding = nn.Embedding(num_day_types, day_type_embedding_dim)

        total_embedding_dim = (
            node_type_embedding_dim
            + department_embedding_dim
            + shift_embedding_dim
            + day_type_embedding_dim
        )

        # node_continuous_dim must already include any SKU/category expansion done upstream.
        node_input_dim = node_continuous_dim + total_embedding_dim

        self.gat1 = GATConv(
            in_channels=node_input_dim,
            out_channels=hidden_dim // 4,
            heads=4,
            concat=True,
            edge_dim=edge_feat_dim,
        )

        augmented_edge_dim = edge_feat_dim + hidden_dim + hidden_dim

        # PASS 2: refine nodes with augmented edge features
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=augmented_edge_dim,
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(augmented_edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        node_continuous: torch.Tensor,
        node_categorical: torch.Tensor,
        edge_features: torch.Tensor,
        edge_node_indices: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-pass GNN encoding that augments edges with endpoint node embeddings before refining nodes."""
        node_continuous = self.node_input_norm(node_continuous)
        edge_features = self.edge_input_norm(edge_features)

        node_type_embeds = self.node_type_embedding(node_categorical[:, 0])

        # Department -1 (unknown) maps to the last embedding index
        dept_ids = node_categorical[:, 1].clone()
        dept_ids[dept_ids == -1] = self.department_embedding.num_embeddings - 1
        dept_embeds = self.department_embedding(dept_ids)

        # Shift period: negative ids (outside any period) map to the last embedding index
        shift_ids = node_categorical[:, 2].clone()
        shift_ids[shift_ids < 0] = self.shift_embedding.num_embeddings - 1
        shift_embeds = self.shift_embedding(shift_ids)
        day_type_embeds = self.day_type_embedding(node_categorical[:, 3])

        node_features = torch.cat(
            [node_continuous, node_type_embeds, dept_embeds, shift_embeds, day_type_embeds], dim=-1
        )

        if edge_node_indices.dim() != 2:
            raise ValueError(f"edge_node_indices must be 2D, got {edge_node_indices.shape}")

        if edge_node_indices.shape[1] != 2:
            raise ValueError(
                f"edge_node_indices must be [num_edges, 2], got {edge_node_indices.shape}"
            )

        if edge_index is None:
            raise ValueError("edge_index must be provided for GNN message passing")

        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, num_edges], got {edge_index.shape}")

        # PASS 1: initial node encoding
        x = self.gat1(node_features, edge_index, edge_attr=edge_features)
        x = F.elu(x)
        node_embeddings_pass1 = x

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

        from_node_embeds = node_embeddings_pass1[edge_node_indices[:, 0]]
        to_node_embeds = node_embeddings_pass1[edge_node_indices[:, 1]]

        augmented_edge_features = torch.cat(
            [edge_features, from_node_embeds, to_node_embeds], dim=-1
        )

        # PASS 2: refine nodes with augmented edges
        node_embeddings = self.gat2(
            node_embeddings_pass1, edge_index, edge_attr=augmented_edge_features
        )

        edge_embeddings = self.edge_encoder(augmented_edge_features)

        # Residual-adds mean incoming-edge embeddings so nodes encode corridor traversal costs (unaffected if no incoming edges).
        num_nodes_final = node_embeddings.shape[0]
        edge_to_node = edge_node_indices[:, 1]
        edge_dim = edge_embeddings.shape[1]
        node_edge_agg = torch.zeros(
            num_nodes_final, edge_dim, device=edge_embeddings.device, dtype=edge_embeddings.dtype
        )
        edge_counts = torch.zeros(
            num_nodes_final, 1, device=edge_embeddings.device, dtype=edge_embeddings.dtype
        )
        idx = edge_to_node.unsqueeze(1).expand(-1, edge_dim)
        node_edge_agg.scatter_add_(0, idx, edge_embeddings)
        edge_counts.scatter_add_(
            0,
            edge_to_node.unsqueeze(1),
            torch.ones(
                edge_embeddings.shape[0],
                1,
                device=edge_embeddings.device,
                dtype=edge_embeddings.dtype,
            ),
        )
        node_edge_agg = node_edge_agg / edge_counts.clamp(min=1.0)
        node_embeddings = node_embeddings + node_edge_agg

        graph_embedding = torch.mean(node_embeddings, dim=0)

        return node_embeddings, edge_embeddings, graph_embedding


class RobotFleetEncoder(nn.Module):
    """Encodes the robot fleet using a dynamic proximity graph."""

    def __init__(self, robot_feat_dim=20, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.robot_input_norm = nn.LayerNorm(robot_feat_dim)

        # SAGEConv supports the variable number of robots across episodes
        self.sage1 = SAGEConv(in_channels=robot_feat_dim, out_channels=hidden_dim, aggr="mean")

        self.sage2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr="mean")

    def forward(
        self,
        robot_features: torch.Tensor,
        robot_positions: Optional[torch.Tensor] = None,
        proximity_threshold: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a proximity graph from robot positions and encode it with GraphSAGE."""
        robot_features = self.robot_input_norm(robot_features)

        if robot_features.dim() == 3:
            batch_size, num_robots, feat_dim = robot_features.shape
            flat_features = robot_features.view(batch_size * num_robots, feat_dim)

            if robot_positions is not None:
                # Vectorized: compute all pairwise distances across the entire batch at once
                dist_matrix = torch.cdist(robot_positions, robot_positions)

                row, col = torch.triu_indices(
                    num_robots, num_robots, offset=1, device=robot_features.device
                )

                dists = dist_matrix[:, row, col]
                mask = dists < proximity_threshold

                batch_idx, pair_idx = mask.nonzero(as_tuple=True)

                if batch_idx.numel() > 0:
                    batch_offsets = batch_idx * num_robots
                    src_global = row[pair_idx] + batch_offsets
                    dst_global = col[pair_idx] + batch_offsets
                    edge_index = torch.stack(
                        [torch.cat([src_global, dst_global]), torch.cat([dst_global, src_global])],
                        dim=0,
                    )
                else:
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=robot_features.device)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=robot_features.device)

            x = self.sage1(flat_features, edge_index)
            x = F.relu(x)
            robot_embeddings = self.sage2(x, edge_index)
            robot_embeddings = robot_embeddings.view(batch_size, num_robots, self.hidden_dim)

            fleet_embedding = torch.mean(robot_embeddings, dim=1)
            return robot_embeddings, fleet_embedding

        if robot_positions is not None:
            edge_index = self._build_proximity_graph(robot_positions, threshold=proximity_threshold)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=robot_features.device)

        x = self.sage1(robot_features, edge_index)
        x = F.relu(x)
        robot_embeddings = self.sage2(x, edge_index)

        fleet_embedding = torch.mean(robot_embeddings, dim=0)

        return robot_embeddings, fleet_embedding

    def _build_proximity_graph(
        self, positions: torch.Tensor, threshold: float = 10.0
    ) -> torch.Tensor:
        """Build edges between robots within threshold distance, vectorized via torch.cdist."""
        num_robots = positions.shape[0]

        dist_matrix = torch.cdist(positions, positions)
        row, col = torch.triu_indices(num_robots, num_robots, offset=1, device=positions.device)

        mask = dist_matrix[row, col] < threshold
        src = row[mask]
        dst = col[mask]

        if src.numel() == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=positions.device)

        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        return edge_index


class TaskEncoder(nn.Module):
    """Encodes task features into embedding space."""

    def __init__(self, task_feat_dim=15, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(task_features)


def test_encoders():
    """Test GNN encoders with dummy data."""
    print("Testing GAPO GNN Encoders...")

    print("\n1. Hospital Graph Encoder (Two-Pass with Fine-Grained Categorical Embeddings)")
    hospital_encoder = GraphEncoder(
        node_continuous_dim=24,
        num_node_types=4,
        num_departments=10,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=21,
        hidden_dim=64,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4,
    )

    node_continuous = torch.randn(10, 24)
    node_categorical = torch.randint(0, 4, (10, 4))
    edge_features = torch.randn(20, 21)
    edge_node_indices = torch.randint(0, 10, (20, 2))
    edge_index = torch.randint(0, 10, (2, 20))

    node_embeds, edge_embeds, graph_embed = hospital_encoder(
        node_continuous, node_categorical, edge_features, edge_node_indices, edge_index
    )

    print(f"  Node embeddings: {node_embeds.shape}")
    print(f"  Edge embeddings: {edge_embeds.shape}")
    print(f"  Graph embedding: {graph_embed.shape}")
    print("  Node input: 24 continuous + 8 node_type + 16 dept + 4 shift + 4 day_type = 56 dims")
    print("  Edge augmentation: 21 continuous + 64 from + 64 to = 149 dims")

    print("\n2. Robot Fleet Encoder")
    robot_encoder = RobotFleetEncoder(robot_feat_dim=20, hidden_dim=64)

    robot_features = torch.randn(5, 12)
    robot_positions = torch.randn(5, 2)

    robot_embeds, fleet_embed = robot_encoder(robot_features, robot_positions)

    print(f"  Robot embeddings: {robot_embeds.shape}")
    print(f"  Fleet embedding: {fleet_embed.shape}")

    print("\n3. Task Encoder")
    task_encoder = TaskEncoder(task_feat_dim=15, hidden_dim=64)

    task_features = torch.randn(15)
    task_embed = task_encoder(task_features)

    print(f"  Task embedding: {task_embed.shape}")

    print("\nAll encoders working!")
