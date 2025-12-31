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

try:
    from torch_geometric.nn import GATConv, SAGEConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("WARNING: torch_geometric not installed. Install with: pip install torch-geometric")
    print("Falling back to simple MLP encoders.")


class HospitalGraphEncoder(nn.Module):
    """
    Encodes hospital graph structure using Graph Attention Networks.

    Input: Node features, edge features, edge connectivity
    Output: Node embeddings + global graph embedding
    """

    def __init__(self, node_feat_dim=5, edge_feat_dim=3, hidden_dim=64, use_gnn=True):
        super().__init__()

        self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE
        self.hidden_dim = hidden_dim

        if self.use_gnn:
            # Graph Attention Network for nodes
            self.gat1 = GATConv(
                in_channels=node_feat_dim,
                out_channels=hidden_dim // 4,
                heads=4,
                concat=True,
                edge_dim=edge_feat_dim
            )

            self.gat2 = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=1,
                concat=False,
                edge_dim=edge_feat_dim
            )

            # Edge feature encoder
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # Fallback: Simple MLP encoder
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [num_nodes, node_feat_dim]
            edge_features: [num_edges, edge_feat_dim]
            edge_index: [2, num_edges] - connectivity (only if use_gnn=True)

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            edge_embeddings: [num_edges, hidden_dim]
            graph_embedding: [hidden_dim]
        """
        if self.use_gnn and edge_index is not None:
            # GNN encoding
            x = self.gat1(node_features, edge_index, edge_attr=edge_features)
            x = F.elu(x)
            node_embeddings = self.gat2(x, edge_index, edge_attr=edge_features)
        else:
            # MLP encoding (fallback)
            node_embeddings = self.node_encoder(node_features)

        # Encode edges
        edge_embeddings = self.edge_encoder(edge_features)

        # Global graph embedding (mean pooling)
        graph_embedding = torch.mean(node_embeddings, dim=0)

        return node_embeddings, edge_embeddings, graph_embedding


class RobotFleetEncoder(nn.Module):
    """
    Encodes robot fleet using dynamic proximity graph.

    Input: Robot features, robot positions
    Output: Robot embeddings + fleet embedding
    """

    def __init__(self, robot_feat_dim=12, hidden_dim=64, use_gnn=True):
        super().__init__()

        self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE
        self.hidden_dim = hidden_dim

        if self.use_gnn:
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
        else:
            # Fallback: Simple MLP
            self.robot_encoder = nn.Sequential(
                nn.Linear(robot_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
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
        if self.use_gnn and robot_positions is not None:
            # Build proximity graph
            proximity_edges = self._build_proximity_graph(
                robot_positions,
                threshold=proximity_threshold
            )

            # GNN encoding
            x = self.sage1(robot_features, proximity_edges)
            x = F.relu(x)
            robot_embeddings = self.sage2(x, proximity_edges)
        else:
            # MLP encoding (fallback)
            robot_embeddings = self.robot_encoder(robot_features)

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

    Input: Task features (12 dims)
    Output: Task embedding
    """

    def __init__(self, task_feat_dim=12, hidden_dim=64):
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

    # Test Hospital Graph Encoder
    print("\n1. Hospital Graph Encoder")
    hospital_encoder = HospitalGraphEncoder(
        node_feat_dim=5,
        edge_feat_dim=3,
        hidden_dim=64
    )

    node_features = torch.randn(10, 5)  # 10 nodes
    edge_features = torch.randn(20, 3)  # 20 edges
    edge_index = torch.randint(0, 10, (2, 20))  # Random connectivity

    node_embeds, edge_embeds, graph_embed = hospital_encoder(
        node_features, edge_features, edge_index
    )

    print(f"  Node embeddings: {node_embeds.shape}")
    print(f"  Edge embeddings: {edge_embeds.shape}")
    print(f"  Graph embedding: {graph_embed.shape}")

    # Test Robot Fleet Encoder
    print("\n2. Robot Fleet Encoder")
    robot_encoder = RobotFleetEncoder(
        robot_feat_dim=12,
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
    task_encoder = TaskEncoder(task_feat_dim=12, hidden_dim=64)

    task_features = torch.randn(12)
    task_embed = task_encoder(task_features)

    print(f"  Task embedding: {task_embed.shape}")

    print("\n✓ All encoders working!")


if __name__ == '__main__':
    test_encoders()
