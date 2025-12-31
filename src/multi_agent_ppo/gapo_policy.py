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
        node_feat_dim=5,
        edge_feat_dim=3,
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=64,
        num_attention_heads=4,
        use_gnn=True,
        use_debiasing=True,
        lambda_state_delta=0.1,
        lambda_consistency=0.05
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_debiasing = use_debiasing

        # ===== ENCODERS =====
        self.hospital_encoder = HospitalGraphEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, use_gnn
        )

        self.robot_encoder = RobotFleetEncoder(
            robot_feat_dim, hidden_dim, use_gnn
        )

        self.task_encoder = TaskEncoder(
            task_feat_dim, hidden_dim
        )

        # ===== ATTENTION =====
        self.attention_module = GAPOAttentionModule(
            hidden_dim, num_attention_heads, hidden_dim * 2
        )

        # ===== CRITIC (Value Network) =====
        # Takes global context to estimate state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 5, 256),  # Graph + Fleet + Task + Queue
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
                - 'task_features': [12]
                - 'node_features': [num_nodes, 5]
                - 'edge_features': [num_edges, 3]
                - 'edge_index': [2, num_edges] (optional, for GNN)
                - 'robot_features': [num_robots, 12]
                - 'robot_positions': [num_robots, 2] (optional)
                - 'queue_features': [5]
            return_attention: Whether to return attention weights
            robot_availability_mask: [num_robots] - boolean mask

        Returns:
            action_logits: [num_robots + 1] - scores for each robot + HOLD
            state_value: [1] - estimated state value
            attention_info: Optional dict with attention weights
        """
        # ===== ENCODE COMPONENTS =====

        # 1. Hospital graph
        node_embeddings, edge_embeddings, graph_embedding = self.hospital_encoder(
            state_dict['node_features'],
            state_dict['edge_features'],
            state_dict.get('edge_index', None)
        )

        # 2. Robot fleet
        robot_embeddings, fleet_embedding = self.robot_encoder(
            state_dict['robot_features'],
            state_dict.get('robot_positions', None)
        )

        # 3. Task
        task_embedding = self.task_encoder(state_dict['task_features'])

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
            # Extend mask for HOLD action (always available)
            extended_mask = torch.cat([
                robot_availability_mask,
                torch.tensor([True])
            ])

            action_logits = action_logits.masked_fill(~extended_mask, float('-inf'))

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

        log_probs = []
        state_values = []
        entropies = []

        for i in range(batch_size):
            mask = robot_masks[i] if robot_masks else None

            action_logits, state_value, _ = self.forward(
                state_dicts[i],
                robot_availability_mask=mask
            )

            # Apply mask
            if mask is not None:
                extended_mask = torch.cat([mask, torch.tensor([True])])
                action_logits = action_logits.masked_fill(~extended_mask, float('-inf'))

            # Compute log prob
            action_probs = F.softmax(action_logits, dim=-1)
            log_prob = F.log_softmax(action_logits, dim=-1)[actions[i]]

            # Compute entropy
            entropy = -(action_probs * log_prob).sum()

            log_probs.append(log_prob)
            state_values.append(state_value)
            entropies.append(entropy)

        log_probs = torch.stack(log_probs)
        state_values = torch.stack(state_values).squeeze(-1)
        entropies = torch.stack(entropies)

        return log_probs, state_values, entropies

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

    # Create policy
    policy = GAPOPolicyNetwork(
        node_feat_dim=5,
        edge_feat_dim=3,
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=64,
        use_gnn=True,
        use_debiasing=True
    )

    # Create dummy state
    num_nodes = 10
    num_edges = 20
    num_robots = 5

    state_dict = {
        'task_features': torch.randn(12),
        'node_features': torch.randn(num_nodes, 5),
        'edge_features': torch.randn(num_edges, 3),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'robot_features': torch.randn(num_robots, 12),
        'robot_positions': torch.randn(num_robots, 2),
        'queue_features': torch.randn(5)
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
