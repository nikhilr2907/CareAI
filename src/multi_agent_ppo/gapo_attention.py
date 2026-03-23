"""Attention modules used by GAPO."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TaskRobotAttention(nn.Module):
    """Attend from a task embedding to robot embeddings."""

    def __init__(self, embed_dim=64, num_heads=4, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        task_embedding: torch.Tensor,
        robot_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_batched = task_embedding.dim() > 1

        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).unsqueeze(0)
        elif task_embedding.dim() == 2:
            task_embedding = task_embedding.unsqueeze(1)

        if robot_embeddings.dim() == 2:
            robot_embeddings = robot_embeddings.unsqueeze(0)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                key_padding_mask = (~attention_mask).unsqueeze(0)
            else:
                key_padding_mask = ~attention_mask

        attn_output, attn_weights = self.multihead_attn(
            query=task_embedding,
            key=robot_embeddings,
            value=robot_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        attn_output = self.layer_norm(attn_output)
        robot_context = attn_output.squeeze(1)
        attention_weights = attn_weights.squeeze(1)

        if not input_batched:
            robot_context = robot_context.squeeze(0)
            attention_weights = attention_weights.squeeze(0)

        return robot_context, attention_weights


class TaskNodeAttention(nn.Module):
    """Attend from a task embedding to node embeddings."""

    def __init__(self, embed_dim=64, num_heads=4, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        task_embedding: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_batched = task_embedding.dim() > 1

        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).unsqueeze(0)
        elif task_embedding.dim() == 2:
            task_embedding = task_embedding.unsqueeze(1)

        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)

        attn_output, attn_weights = self.multihead_attn(
            query=task_embedding,
            key=node_embeddings,
            value=node_embeddings,
            need_weights=True,
            average_attn_weights=True,
        )

        attn_output = self.layer_norm(attn_output)
        node_context = attn_output.squeeze(1)
        attention_weights = attn_weights.squeeze(1)

        if not input_batched:
            node_context = node_context.squeeze(0)
            attention_weights = attention_weights.squeeze(0)

        return node_context, attention_weights


class RobotScorer(nn.Module):
    """Score robots from task and context features."""

    def __init__(self, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        robot_embeddings: torch.Tensor,
        task_embedding: torch.Tensor,
        robot_context: torch.Tensor,
        node_context: torch.Tensor,
    ) -> torch.Tensor:
        if robot_embeddings.dim() == 2:
            num_robots = robot_embeddings.shape[0]

            task_expanded = task_embedding.unsqueeze(0).expand(num_robots, -1)
            robot_context_expanded = robot_context.unsqueeze(0).expand(num_robots, -1)
            node_context_expanded = node_context.unsqueeze(0).expand(num_robots, -1)

            combined = torch.cat(
                [
                    robot_embeddings,
                    task_expanded,
                    robot_context_expanded,
                    node_context_expanded,
                ],
                dim=-1,
            )
            return self.scorer(combined).squeeze(-1)

        batch_size, num_robots, _ = robot_embeddings.shape
        task_expanded = task_embedding.unsqueeze(1).expand(batch_size, num_robots, -1)
        robot_context_expanded = robot_context.unsqueeze(1).expand(batch_size, num_robots, -1)
        node_context_expanded = node_context.unsqueeze(1).expand(batch_size, num_robots, -1)

        combined = torch.cat(
            [
                robot_embeddings,
                task_expanded,
                robot_context_expanded,
                node_context_expanded,
            ],
            dim=-1,
        )
        return self.scorer(combined).squeeze(-1)


class GAPOAttentionModule(nn.Module):
    """Combine robot attention, node attention, and robot scoring."""

    def __init__(self, embed_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.task_robot_attn = TaskRobotAttention(embed_dim, num_heads)
        self.task_node_attn = TaskNodeAttention(embed_dim, num_heads)
        self.robot_scorer = RobotScorer(embed_dim, hidden_dim)

    def forward(
        self,
        task_embedding: torch.Tensor,
        robot_embeddings: torch.Tensor,
        node_embeddings: torch.Tensor,
        robot_availability_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        robot_context, robot_attn_weights = self.task_robot_attn(
            task_embedding,
            robot_embeddings,
            robot_availability_mask,
        )
        node_context, node_attn_weights = self.task_node_attn(
            task_embedding,
            node_embeddings,
        )
        robot_scores = self.robot_scorer(
            robot_embeddings,
            task_embedding,
            robot_context,
            node_context,
        )

        attention_info = {
            "robot_context": robot_context,
            "node_context": node_context,
            "robot_attn_weights": robot_attn_weights,
            "node_attn_weights": node_attn_weights,
        }
        return robot_scores, attention_info
