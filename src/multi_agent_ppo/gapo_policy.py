"""
GAPO (Graph Attention-Based Policy Optimization) Policy Network.

Integrates:
- GNN encoders for hospital graph and robot fleet
- Attention mechanisms for task-robot-node
- De-biasing for autoregressive decisions
- Actor-Critic architecture for PPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .gapo_gnn_encoders import HospitalGraphEncoder, RobotFleetEncoder, TaskEncoder
from .gapo_attention import GAPOAttentionModule
from .debiasing import ComprehensiveDebiasing


class GAPOPolicyNetwork(nn.Module):
    """
    Complete GAPO policy network with GNN + Attention + De-biasing.
    """

    def __init__(
        self,
        node_continuous_dim=15,
        num_node_types=4,
        edge_feat_dim=14,
        node_type_embedding_dim=8,
        robot_feat_dim=12,
        task_feat_dim=15,
        queue_feat_dim=14,
        hidden_dim=64,
        num_attention_heads=4,
        use_debiasing=True,
        lambda_state_delta=0.1,
        lambda_consistency=0.05
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_debiasing = use_debiasing

        # ===== ENCODERS =====
        self.hospital_encoder = HospitalGraphEncoder(
            node_continuous_dim=node_continuous_dim,
            num_node_types=num_node_types,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            node_type_embedding_dim=node_type_embedding_dim
        )

        self.robot_encoder = RobotFleetEncoder(
            robot_feat_dim, hidden_dim
        )

        self.task_encoder = TaskEncoder(
            task_feat_dim, hidden_dim
        )

        self.task_scorer = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ===== ATTENTION =====
        self.attention_module = GAPOAttentionModule(
            hidden_dim, num_attention_heads, hidden_dim * 2
        )

        # ===== CRITIC (Value Network) =====
        # Takes global context to estimate state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3 + queue_feat_dim, 256),  # Graph + Fleet + Task + Queue
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ===== DE-BIASING =====
        if use_debiasing:
            self.debiaser = ComprehensiveDebiasing(
                lambda_state_delta=lambda_state_delta,
                lambda_consistency=lambda_consistency
            )
        else:
            self.debiaser = None

        # Episode tracking for de-biasing
        self.episode_states = []
        self.episode_actions = []
        self.episode_logits = []
        self.episode_task_features = []

    def forward(
        self,
        state_dict: Dict[str, torch.Tensor],
        return_attention: bool = False,
        robot_availability_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through GAPO network.

        Args:
            state_dict: Dictionary with:
                - 'task_features': [15]
                - 'node_continuous': [num_nodes, 15] - continuous node features
                - 'node_categorical': [num_nodes, 1] - node_type_id
                - 'edge_features': [num_edges, 14] - continuous edge features
                - 'edge_node_indices': [num_edges, 2] - (from_node_idx, to_node_idx)
                - 'edge_index': [2, num_edges] - graph connectivity for GNN
                - 'robot_features': [num_robots, 12]
                - 'robot_positions': [num_robots, 2] (optional)
            - 'queue_features': [14]
            return_attention: Whether to return attention weights
            robot_availability_mask: [num_robots] - boolean mask

        Returns:
            action_logits: [num_robots] - scores for each robot
            state_value: [1] - estimated state value
            attention_info: Optional dict with attention weights
        """
        # ===== ENCODE COMPONENTS =====

        # 1. Hospital graph (two-pass encoding with node embeddings)
        node_embeddings, edge_embeddings, graph_embedding = self.hospital_encoder(
            state_dict['node_continuous'],
            state_dict['node_categorical'],
            state_dict['edge_features'],
            state_dict['edge_node_indices'],
            state_dict.get('edge_index', None)
        )

        # 2. Robot fleet
        robot_embeddings, fleet_embedding = self.robot_encoder(
            state_dict['robot_features'],
            state_dict.get('robot_positions', None)
        )

        # 3. Task
        task_embedding = self.task_encoder(state_dict['task_features'])
        task_score = self.task_scorer(state_dict['task_features']).squeeze(-1)
        task_embedding = task_embedding * (1.0 + torch.tanh(task_score)).unsqueeze(-1)

        # ===== ATTENTION =====
        action_logits, attention_info = self.attention_module(
            task_embedding,
            robot_embeddings,
            node_embeddings,
            robot_availability_mask
        )

        # ===== VALUE ESTIMATION =====
        # Global state representation
        global_state = torch.cat([
            graph_embedding,
            fleet_embedding,
            task_embedding,
            state_dict['queue_features']
        ], dim=-1)

        state_value = self.critic(global_state)

        # ===== TRACKING FOR DE-BIASING =====
        if self.training and self.use_debiasing:
            # Store for de-biasing loss computation
            self.episode_logits.append(action_logits.detach().clone())
            self.episode_task_features.append(
                state_dict['task_features'].detach().clone()
            )

        if return_attention:
            return action_logits, state_value, attention_info
        else:
            return action_logits, state_value, None

    def score_tasks(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Score tasks for prioritization.

        Args:
            task_features: [num_tasks, task_feat_dim]

        Returns:
            scores: [num_tasks]
        """
        scores = self.task_scorer(task_features).squeeze(-1)
        return scores

    def select_action(
        self,
        state_dict: Dict[str, torch.Tensor],
        robot_availability_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action using current policy.

        Args:
            state_dict: State dictionary
            robot_availability_mask: Boolean mask for available robots
            deterministic: If True, select argmax; else sample

        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
        """
        with torch.no_grad():
            action_logits, _, _ = self.forward(
                state_dict,
                robot_availability_mask=robot_availability_mask
            )

        # Apply action mask (set unavailable actions to -inf)
        if robot_availability_mask is not None:
            if not robot_availability_mask.any():
                robot_availability_mask = torch.ones_like(robot_availability_mask, dtype=torch.bool)
            action_logits = action_logits.masked_fill(~robot_availability_mask, float('-inf'))

        # Sample or take argmax
        action_probs = F.softmax(action_logits, dim=-1)

        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

        # Compute log probability
        log_prob = F.log_softmax(action_logits, dim=-1)[action]

        return action, log_prob

    def evaluate_actions(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        robot_masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            state_dicts: List of state dictionaries
            actions: [batch_size] - actions taken
            robot_masks: List of robot availability masks

        Returns:
            log_probs: [batch_size] - log probabilities of actions
            state_values: [batch_size] - estimated state values
            entropy: [batch_size] - policy entropy
        """
        batch_size = len(state_dicts)
        if batch_size == 0:
            return (
                torch.tensor([], device=actions.device),
                torch.tensor([], device=actions.device),
                torch.tensor([], device=actions.device)
            )

        action_logits, state_values, _ = self._forward_batched(state_dicts, robot_masks)

        # Apply mask
        mask_list = []
        num_robots = action_logits.shape[1]
        if robot_masks is not None:
            for mask in robot_masks:
                if mask is None or not mask.any():
                    mask_list.append(torch.ones(num_robots, dtype=torch.bool, device=actions.device))
                else:
                    mask_list.append(mask.to(actions.device))
        else:
            mask_list = [torch.ones(num_robots, dtype=torch.bool, device=actions.device) for _ in range(batch_size)]

        mask_tensor = torch.stack(mask_list, dim=0)
        action_logits = action_logits.masked_fill(~mask_tensor, float('-inf'))

        # Compute log prob
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1).gather(
            1, actions.view(-1, 1)
        ).squeeze(1)

        # Compute entropy
        entropies = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)

        return log_probs, state_values.squeeze(-1), entropies

    def _forward_batched(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        robot_masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Batched forward pass for PPO evaluation.
        """
        batch_size = len(state_dicts)
        device = next(self.parameters()).device

        # Shapes from first sample
        num_nodes = state_dicts[0]['node_continuous'].shape[0]
        num_edges = state_dicts[0]['edge_features'].shape[0]
        num_robots = state_dicts[0]['robot_features'].shape[0]

        # Stack per-sample inputs
        node_continuous = torch.stack(
            [sd['node_continuous'] for sd in state_dicts], dim=0
        ).to(device)
        node_categorical = torch.stack(
            [sd['node_categorical'] for sd in state_dicts], dim=0
        ).to(device)
        edge_features = torch.stack(
            [sd['edge_features'] for sd in state_dicts], dim=0
        ).to(device)
        edge_node_indices = torch.stack(
            [sd['edge_node_indices'] for sd in state_dicts], dim=0
        ).to(device)
        edge_index = torch.stack(
            [sd['edge_index'] for sd in state_dicts], dim=0
        ).to(device)

        robot_features = torch.stack(
            [sd['robot_features'] for sd in state_dicts], dim=0
        ).to(device)
        robot_positions = None
        if 'robot_positions' in state_dicts[0]:
            robot_positions = torch.stack(
                [sd['robot_positions'] for sd in state_dicts], dim=0
            ).to(device)

        task_features = torch.stack(
            [sd['task_features'] for sd in state_dicts], dim=0
        ).to(device)
        queue_features = torch.stack(
            [sd['queue_features'] for sd in state_dicts], dim=0
        ).to(device)

        # Build batched graph inputs via disjoint union
        node_offsets = (torch.arange(batch_size, device=device) * num_nodes).view(-1, 1, 1)
        edge_node_indices = (edge_node_indices + node_offsets).view(-1, 2)

        edge_index = edge_index + node_offsets.view(-1, 1, 1)
        edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)

        node_continuous = node_continuous.view(batch_size * num_nodes, -1)
        node_categorical = node_categorical.view(batch_size * num_nodes, -1)
        edge_features = edge_features.view(batch_size * num_edges, -1)

        # Encode hospital graph
        node_embeddings, _, _ = self.hospital_encoder(
            node_continuous,
            node_categorical,
            edge_features,
            edge_node_indices,
            edge_index
        )
        node_embeddings = node_embeddings.view(batch_size, num_nodes, self.hidden_dim)
        graph_embedding = torch.mean(node_embeddings, dim=1)

        # Encode robots and tasks
        robot_embeddings, fleet_embedding = self.robot_encoder(
            robot_features,
            robot_positions
        )
        task_embedding = self.task_encoder(task_features)

        # Build mask tensor for attention
        mask_tensor = None
        if robot_masks is not None:
            mask_list = []
            for mask in robot_masks:
                if mask is None:
                    mask_list.append(torch.ones(num_robots, dtype=torch.bool, device=device))
                else:
                    mask_list.append(mask.to(device))
            mask_tensor = torch.stack(mask_list, dim=0)

        # Attention
        action_logits, attention_info = self.attention_module(
            task_embedding,
            robot_embeddings,
            node_embeddings,
            mask_tensor
        )

        # Value estimation
        global_state = torch.cat([
            graph_embedding,
            fleet_embedding,
            task_embedding,
            queue_features
        ], dim=-1)

        state_value = self.critic(global_state)

        return action_logits, state_value, attention_info

    def compute_debias_loss(self) -> Tuple[torch.Tensor, Dict]:
        """
        Compute de-biasing loss from current episode.

        Returns:
            debias_loss: Scalar de-biasing penalty
            loss_breakdown: Dictionary with loss components
        """
        if not self.use_debiasing or len(self.episode_logits) < 2:
            return torch.tensor(0.0), {}

        # Convert lists to format expected by debiaser
        # (We don't have explicit state sequence in this architecture,
        #  so we use embeddings as proxy)

        debias_loss, breakdown = self.debiaser.compute_total_debias_loss(
            state_sequence=[],  # Not used in current implementation
            action_sequence=self.episode_actions,
            action_logits_sequence=self.episode_logits,
            task_features_sequence=self.episode_task_features
        )

        return debias_loss, breakdown

    def reset_episode_tracking(self):
        """Reset episode tracking for de-biasing."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_logits = []
        self.episode_task_features = []

    def record_action(self, action: int):
        """Record action for de-biasing."""
        if self.training and self.use_debiasing:
            self.episode_actions.append(action)


def test_gapo_policy():
    """Test GAPO policy network."""
    print("Testing GAPO Policy Network...")

    # Create policy with enhanced features
    policy = GAPOPolicyNetwork(
        node_continuous_dim=15,
        num_node_types=4,
        edge_feat_dim=14,
        node_type_embedding_dim=8,
        robot_feat_dim=12,
        task_feat_dim=15,
        hidden_dim=64,
        use_debiasing=True
    )

    # Create dummy state (with enhanced features)
    num_nodes = 10
    num_edges = 20
    num_robots = 5

    state_dict = {
        'task_features': torch.randn(15),
        'node_continuous': torch.randn(num_nodes, 15),  # Enhanced: 15 continuous features
        'node_categorical': torch.randint(0, 4, (num_nodes, 1)),  # node_type_id (0-3)
        'edge_features': torch.randn(num_edges, 14),  # Enhanced: 14 continuous features
        'edge_node_indices': torch.randint(0, num_nodes, (num_edges, 2)),  # (from, to) indices
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),  # GNN connectivity
        'robot_features': torch.randn(num_robots, 12),
        'robot_positions': torch.randn(num_robots, 2),
        'queue_features': torch.randn(14)
    }

    # Test forward pass
    print("\n1. Forward pass")
    action_logits, state_value, attention_info = policy(
        state_dict,
        return_attention=True
    )

    print(f"  Action logits: {action_logits.shape}")
    print(f"  State value: {state_value.shape}")
    print(f"  Robot attention: {attention_info['robot_attn_weights']}")

    # Test action selection
    print("\n2. Action selection")
    action, log_prob = policy.select_action(state_dict)
    print(f"  Selected action: {action}")
    print(f"  Log probability: {log_prob:.4f}")

    # Test with availability mask
    print("\n3. Action selection with mask")
    mask = torch.tensor([True, False, True, True, False])
    action_masked, log_prob_masked = policy.select_action(
        state_dict,
        robot_availability_mask=mask
    )
    print(f"  Selected action (masked): {action_masked}")
    print(f"  Available robots: {torch.where(mask)[0].tolist()}")

    # Test de-biasing
    print("\n4. De-biasing")
    policy.train()

    for t in range(5):
        action_logits, _, _ = policy(state_dict)
        action = torch.argmax(action_logits).item()
        policy.record_action(action)

    debias_loss, breakdown = policy.compute_debias_loss()
    print(f"  De-biasing loss: {debias_loss.item():.4f}")
    print(f"  Breakdown: {breakdown}")

    print("\n✓ GAPO Policy Network working!")


if __name__ == '__main__':
    test_gapo_policy()
