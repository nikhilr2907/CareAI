"""
GAPO Attention Mechanisms.

Implements cross-attention between:
- Task ↔ Robots (which robots are suitable?)
- Task ↔ Nodes (which locations matter?)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TaskRobotAttention(nn.Module):
    """
    Cross-attention from task to robots.
    Learns which robots are most suitable for a given task.
    """

    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        task_embedding: torch.Tensor,
        robot_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_embedding: [1, embed_dim] or [batch_size, 1, embed_dim]
            robot_embeddings: [num_robots, embed_dim] or [batch_size, num_robots, embed_dim]
            attention_mask: [num_robots] - boolean mask for available robots

        Returns:
            robot_context: [embed_dim] - attended robot context
            attention_weights: [num_robots] - attention distribution
        """
        # Ensure task embedding has batch dimension
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).unsqueeze(0)
        elif task_embedding.dim() == 2:
            task_embedding = task_embedding.unsqueeze(1)

        # Ensure robot embeddings have batch dimension
        if robot_embeddings.dim() == 2:
            robot_embeddings = robot_embeddings.unsqueeze(0)

        # Prepare key_padding_mask: must be 2-D [batch_size, num_robots] for batched input
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask is [num_robots] boolean (True = available, False = unavailable)
            # key_padding_mask expects True = ignore, False = use
            # So we need to invert and add batch dimension
            if attention_mask.dim() == 1:
                key_padding_mask = (~attention_mask).unsqueeze(0)  # [1, num_robots]
            else:
                key_padding_mask = ~attention_mask  # Already batched

        # Apply attention
        attn_output, attn_weights = self.multihead_attn(
            query=task_embedding,      # [batch, 1, embed_dim]
            key=robot_embeddings,       # [batch, num_robots, embed_dim]
            value=robot_embeddings,
            key_padding_mask=key_padding_mask,  # [batch, num_robots] or None
            need_weights=True,
            average_attn_weights=True
        )

        # Normalize
        attn_output = self.layer_norm(attn_output)

        # Remove batch/sequence dimensions
        robot_context = attn_output.squeeze(1).squeeze(0)  # [embed_dim]
        attention_weights = attn_weights.squeeze(0).squeeze(0)  # [num_robots]

        return robot_context, attention_weights


class TaskNodeAttention(nn.Module):
    """
    Cross-attention from task to hospital nodes.
    Learns which locations are relevant for a given task.
    """

    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        task_embedding: torch.Tensor,
        node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_embedding: [1, embed_dim]
            node_embeddings: [num_nodes, embed_dim]

        Returns:
            node_context: [embed_dim] - attended node context
            attention_weights: [num_nodes] - attention distribution
        """
        # Ensure task embedding has batch dimension
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).unsqueeze(0)
        elif task_embedding.dim() == 2:
            task_embedding = task_embedding.unsqueeze(1)

        # Ensure node embeddings have batch dimension
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)

        # Apply attention
        attn_output, attn_weights = self.multihead_attn(
            query=task_embedding,
            key=node_embeddings,
            value=node_embeddings,
            need_weights=True,
            average_attn_weights=True
        )

        # Normalize
        attn_output = self.layer_norm(attn_output)

        # Remove batch/sequence dimensions
        node_context = attn_output.squeeze(1).squeeze(0)
        attention_weights = attn_weights.squeeze(0).squeeze(0)

        return node_context, attention_weights


class RobotScorer(nn.Module):
    """
    Scores each robot for a given task based on:
    - Robot's own features
    - Task requirements
    - Global context (from attention)
    """

    def __init__(self, embed_dim=64, hidden_dim=128):
        super().__init__()

        # Score individual robots
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),  # Robot + Task + RobotContext + NodeContext
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        robot_embeddings: torch.Tensor,
        task_embedding: torch.Tensor,
        robot_context: torch.Tensor,
        node_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            robot_embeddings: [num_robots, embed_dim]
            task_embedding: [embed_dim]
            robot_context: [embed_dim]
            node_context: [embed_dim]

        Returns:
            robot_scores: [num_robots]
        """
        num_robots = robot_embeddings.shape[0]

        # Expand task and contexts to match num_robots
        task_expanded = task_embedding.unsqueeze(0).expand(num_robots, -1)
        robot_context_expanded = robot_context.unsqueeze(0).expand(num_robots, -1)
        node_context_expanded = node_context.unsqueeze(0).expand(num_robots, -1)

        # Combine robot features with task context
        combined = torch.cat([
            robot_embeddings,           # Robot's own state
            task_expanded,              # What task needs
            robot_context_expanded,     # What other robots are doing
            node_context_expanded,      # Spatial context (relevant locations)
        ], dim=-1)  # [num_robots, embed_dim * 4]

        # Score each robot
        scores = self.scorer(combined).squeeze(-1)  # [num_robots]

        return scores


class GAPOAttentionModule(nn.Module):
    """
    Complete GAPO attention module combining all components.
    """

    def __init__(self, embed_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()

        self.task_robot_attn = TaskRobotAttention(embed_dim, num_heads)
        self.task_node_attn = TaskNodeAttention(embed_dim, num_heads)
        self.robot_scorer = RobotScorer(embed_dim, hidden_dim)

        # Learnable HOLD action score baseline
        self.hold_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        task_embedding: torch.Tensor,
        robot_embeddings: torch.Tensor,
        node_embeddings: torch.Tensor,
        robot_availability_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Complete GAPO attention forward pass.

        Args:
            task_embedding: [embed_dim]
            robot_embeddings: [num_robots, embed_dim]
            node_embeddings: [num_nodes, embed_dim]
            robot_availability_mask: [num_robots] - boolean mask

        Returns:
            action_logits: [num_robots + 1] - scores for each robot + HOLD
            attention_info: dict with attention weights and contexts
        """
        # Attention to robots
        robot_context, robot_attn_weights = self.task_robot_attn(
            task_embedding,
            robot_embeddings,
            robot_availability_mask
        )

        # Attention to nodes
        node_context, node_attn_weights = self.task_node_attn(
            task_embedding,
            node_embeddings
        )

        # Score each robot
        robot_scores = self.robot_scorer(
            robot_embeddings,
            task_embedding,
            robot_context,
            node_context
        )

        # Add HOLD action score
        action_logits = torch.cat([robot_scores, self.hold_bias], dim=0)

        # Collect attention info for analysis
        attention_info = {
            'robot_context': robot_context,
            'node_context': node_context,
            'robot_attn_weights': robot_attn_weights,
            'node_attn_weights': node_attn_weights
        }

        return action_logits, attention_info


def test_attention():
    """Test attention modules."""
    print("Testing GAPO Attention Modules...")

    embed_dim = 64
    num_robots = 5
    num_nodes = 10

    # Create dummy embeddings
    task_embed = torch.randn(embed_dim)
    robot_embeds = torch.randn(num_robots, embed_dim)
    node_embeds = torch.randn(num_nodes, embed_dim)

    # Test complete GAPO attention
    gapo_attn = GAPOAttentionModule(embed_dim=embed_dim)

    action_logits, attn_info = gapo_attn(
        task_embed,
        robot_embeds,
        node_embeds
    )

    print(f"\nAction logits shape: {action_logits.shape}")
    print(f"Action logits: {action_logits}")
    print(f"\nRobot attention weights: {attn_info['robot_attn_weights']}")
    print(f"Node attention weights: {attn_info['node_attn_weights']}")

    # Test with availability mask
    print("\n\nTesting with robot availability mask...")
    mask = torch.tensor([True, False, True, True, False])  # Robots 1 and 4 unavailable

    action_logits_masked, attn_info_masked = gapo_attn(
        task_embed,
        robot_embeds,
        node_embeds,
        robot_availability_mask=mask
    )

    print(f"Action logits (masked): {action_logits_masked}")
    print(f"Robot attention (masked): {attn_info_masked['robot_attn_weights']}")
    print("  ^ Notice robots 1 and 4 have near-zero attention!")

    print("\nAttention modules working!")


if __name__ == '__main__':
    test_attention()
