import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .gapo_gnn_encoders import ILCGraphEncoder, RobotFleetEncoder, TaskEncoder
from .gapo_attention import GAPOAttentionModule
from .debiasing import ComprehensiveDebiasing


class GAPOPolicyNetwork(nn.Module):
    def __init__(
        self,
        node_continuous_dim=8,
        num_node_types=4,
        num_location_tags=10,
        num_school_periods=4,
        num_day_types=2,
        edge_feat_dim=20,
        node_type_embedding_dim=8,
        location_tag_embedding_dim=16,
        school_period_embedding_dim=4,
        day_type_embedding_dim=4,
        robot_feat_dim=19,
        task_feat_dim=15,
        queue_feat_dim=16,
        sku_feat_dim=None,
        sku_embed_dim=16,
        hidden_dim=64,
        num_attention_heads=4,
        use_debiasing=True,
        lambda_state_delta=0.1,
        lambda_consistency=0.05
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_debiasing = use_debiasing
        self.sku_embed_dim = sku_embed_dim
        self.sku_feat_dim = sku_feat_dim

        # ===== ENCODERS =====
        self.graph_encoder = ILCGraphEncoder(
            node_continuous_dim=node_continuous_dim,
            num_node_types=num_node_types,
            num_location_tags=num_location_tags,
            num_school_periods=num_school_periods,
            num_day_types=num_day_types,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            node_type_embedding_dim=node_type_embedding_dim,
            location_tag_embedding_dim=location_tag_embedding_dim,
            school_period_embedding_dim=school_period_embedding_dim,
            day_type_embedding_dim=day_type_embedding_dim
        )

        self.robot_encoder = RobotFleetEncoder(
            robot_feat_dim, hidden_dim
        )

        self.task_encoder = TaskEncoder(
            task_feat_dim, hidden_dim
        )

        self.sku_encoder = None
        if sku_feat_dim is not None:
            self.sku_encoder = nn.Sequential(
                nn.Linear(sku_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, sku_embed_dim)
            )

        # ===== CONTEXT-AWARE TASK SCORER =====
        # Scores tasks for queue prioritization using task + graph + fleet context
        # Input: task_features [task_feat_dim] + graph_embedding [hidden_dim] + fleet_embedding [hidden_dim]
        scorer_input_dim = task_feat_dim + hidden_dim * 2
        self.task_scorer = nn.Sequential(
            nn.LayerNorm(scorer_input_dim),
            nn.Linear(scorer_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ===== INPUT NORMALIZATION =====
        self.task_input_norm = nn.LayerNorm(task_feat_dim)
        self.queue_input_norm = nn.LayerNorm(queue_feat_dim)

        # ===== ATTENTION =====
        self.attention_module = GAPOAttentionModule(
            hidden_dim, num_attention_heads, hidden_dim * 2
        )

        # ===== CRITIC (Value Network) =====
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3 + queue_feat_dim, 256),
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

    def _pool_sku_embeddings(self, node_sku_features, node_sku_mask):
        if self.sku_encoder is None or node_sku_features is None:
            return None
        num_nodes, num_skus, _ = node_sku_features.shape
        feats = node_sku_features.view(num_nodes * num_skus, -1)
        embeds = self.sku_encoder(feats).view(num_nodes, num_skus, self.sku_embed_dim)

        if node_sku_mask is None:
            weights = torch.ones((num_nodes, num_skus), device=embeds.device)
        else:
            weights = node_sku_mask.float()

        if self.sku_feat_dim is not None and self.sku_feat_dim > 0:
            cat_len = max(self.sku_feat_dim - 10, 0)
            if cat_len < self.sku_feat_dim:
                stock_ratio = node_sku_features[:, :, cat_len]
                weights = weights * (1.0 - stock_ratio).clamp(min=0.0)

        weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weights = weights / weights_sum
        pooled = (embeds * weights.unsqueeze(-1)).sum(dim=1)
        return pooled

    def forward(
        self,
        state_dict: Dict[str, torch.Tensor],
        return_attention: bool = False,
        robot_availability_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Run the policy forward pass."""
        if 'node_sku_features' in state_dict and state_dict['node_sku_features'] is not None:
            sku_mask = state_dict.get('node_sku_mask', None)
            pooled = self._pool_sku_embeddings(state_dict['node_sku_features'], sku_mask)
            if pooled is not None:
                state_dict = dict(state_dict)
                state_dict['node_continuous'] = torch.cat([state_dict['node_continuous'], pooled], dim=-1)

        node_embeddings, edge_embeddings, graph_embedding = self.graph_encoder(
            state_dict['node_continuous'],
            state_dict['node_categorical'],
            state_dict['edge_features'],
            state_dict['edge_node_indices'],
            state_dict.get('edge_index', None)
        )

        robot_embeddings, fleet_embedding = self.robot_encoder(
            state_dict['robot_features'],
            state_dict.get('robot_positions', None)
        )

        task_features_normed = self.task_input_norm(state_dict['task_features'])
        task_embedding = self.task_encoder(task_features_normed)

        action_logits, attention_info = self.attention_module(
            task_embedding,
            robot_embeddings,
            node_embeddings,
            robot_availability_mask
        )

        # ===== VALUE ESTIMATION =====
        # Global state representation
        queue_features_normed = self.queue_input_norm(state_dict['queue_features'])
        global_state = torch.cat([
            graph_embedding,
            fleet_embedding,
            task_embedding,
            queue_features_normed
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

    def encode_context(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph and fleet context for task scoring.
        Runs the hospital and robot encoders to produce context embeddings.

        Args:
            state_dict: State dictionary with graph and robot features

        Returns:
            graph_embedding: [hidden_dim]
            fleet_embedding: [hidden_dim]
        """
        # Optional SKU pooling
        if 'node_sku_features' in state_dict and state_dict['node_sku_features'] is not None:
            sku_mask = state_dict.get('node_sku_mask', None)
            pooled = self._pool_sku_embeddings(state_dict['node_sku_features'], sku_mask)
            if pooled is not None:
                state_dict = dict(state_dict)
                state_dict['node_continuous'] = torch.cat([state_dict['node_continuous'], pooled], dim=-1)

        node_embeddings, _, graph_embedding = self.graph_encoder(
            state_dict['node_continuous'],
            state_dict['node_categorical'],
            state_dict['edge_features'],
            state_dict['edge_node_indices'],
            state_dict.get('edge_index', None)
        )

        _, fleet_embedding = self.robot_encoder(
            state_dict['robot_features'],
            state_dict.get('robot_positions', None)
        )

        return graph_embedding, fleet_embedding

    def score_tasks(
        self,
        task_features: torch.Tensor,
        graph_embedding: torch.Tensor,
        fleet_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Score tasks for prioritization using task features + global context.

        Args:
            task_features: [num_tasks, task_feat_dim]
            graph_embedding: [hidden_dim] - hospital graph context
            fleet_embedding: [hidden_dim] - robot fleet context

        Returns:
            scores: [num_tasks]
        """
        num_tasks = task_features.shape[0]
        task_features_normed = self.task_input_norm(task_features)

        # Expand context to match num_tasks
        graph_expanded = graph_embedding.unsqueeze(0).expand(num_tasks, -1)
        fleet_expanded = fleet_embedding.unsqueeze(0).expand(num_tasks, -1)

        # Concatenate task features with context
        scorer_input = torch.cat([task_features_normed, graph_expanded, fleet_expanded], dim=-1)
        scores = self.task_scorer(scorer_input).squeeze(-1)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            raw_logits: [batch_size, num_robots] - pre-mask logits (for de-biasing)
            graph_emb: [batch_size, hidden_dim] - per-state hospital graph embeddings
            fleet_emb: [batch_size, hidden_dim] - per-state fleet embeddings
        """
        batch_size = len(state_dicts)
        device = actions.device
        if batch_size == 0:
            empty = torch.tensor([], device=device)
            empty2d = torch.zeros(0, self.hidden_dim, device=device)
            return empty, empty, empty, empty, empty2d, empty2d

        raw_action_logits, state_values, _, graph_emb_batch, fleet_emb_batch = self._forward_batched(state_dicts, robot_masks)

        # Apply mask
        mask_list = []
        num_robots = raw_action_logits.shape[1]
        if robot_masks is not None:
            for mask in robot_masks:
                if mask is None or not mask.any():
                    mask_list.append(torch.ones(num_robots, dtype=torch.bool, device=actions.device))
                else:
                    mask_list.append(mask.to(actions.device))
        else:
            mask_list = [torch.ones(num_robots, dtype=torch.bool, device=actions.device) for _ in range(batch_size)]

        mask_tensor = torch.stack(mask_list, dim=0)
        action_logits = raw_action_logits.masked_fill(~mask_tensor, float('-inf'))

        # Compute log prob
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1).gather(
            1, actions.view(-1, 1)
        ).squeeze(1)

        # Compute entropy
        entropies = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)

        # Return raw (pre-mask) logits for de-biasing so gradient flows cleanly,
        # plus per-state context embeddings for reuse in compute_ranking_loss.
        return log_probs, state_values.squeeze(-1), entropies, raw_action_logits, graph_emb_batch, fleet_emb_batch

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

        # Optional SKU pooling into node features
        if 'node_sku_features' in state_dicts[0] and state_dicts[0]['node_sku_features'] is not None:
            sku_feats = torch.stack([sd['node_sku_features'] for sd in state_dicts], dim=0).to(device)
            sku_mask = None
            if state_dicts[0].get('node_sku_mask') is not None:
                sku_mask = torch.stack([sd['node_sku_mask'] for sd in state_dicts], dim=0).to(device)
            # Pool per sample then concatenate
            pooled_list = []
            for i in range(batch_size):
                pooled = self._pool_sku_embeddings(sku_feats[i], sku_mask[i] if sku_mask is not None else None)
                pooled_list.append(pooled)
            pooled = torch.stack(pooled_list, dim=0)  # [batch, num_nodes, sku_embed_dim]
            node_continuous = torch.cat([node_continuous, pooled], dim=-1)

        # Build batched graph inputs via disjoint union
        node_offsets = (torch.arange(batch_size, device=device) * num_nodes).view(-1, 1, 1)

        # Validate edge_node_indices before offsetting
        if edge_node_indices.numel() > 0:
            max_before = edge_node_indices.max().item()
            min_before = edge_node_indices.min().item()
            if max_before >= num_nodes or min_before < 0:
                raise ValueError(
                    f"Invalid edge_node_indices BEFORE batching: range [{min_before}, {max_before}] "
                    f"but should be in [0, {num_nodes}). Check state_dict construction."
                )

        edge_node_indices = (edge_node_indices + node_offsets).view(-1, 2)

        # Validate after offsetting
        if edge_node_indices.numel() > 0:
            max_after = edge_node_indices.max().item()
            expected_max = batch_size * num_nodes - 1
            if max_after > expected_max:
                raise ValueError(
                    f"Invalid edge_node_indices AFTER batching: max={max_after} "
                    f"but should be <= {expected_max} (batch_size={batch_size}, num_nodes={num_nodes})"
                )

        edge_index = edge_index + node_offsets.view(-1, 1, 1)
        edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)

        node_continuous = node_continuous.view(batch_size * num_nodes, -1)
        node_categorical = node_categorical.view(batch_size * num_nodes, -1)
        edge_features = edge_features.view(batch_size * num_edges, -1)

        # Encode hospital graph
        node_embeddings, _, _ = self.graph_encoder(
            node_continuous,
            node_categorical,
            edge_features,
            edge_node_indices,
            edge_index
        )
        node_embeddings = node_embeddings.view(batch_size, num_nodes, self.hidden_dim)
        graph_embedding = torch.mean(node_embeddings, dim=1)

        # Encode robots and tasks (normalization applied inside encoders)
        robot_embeddings, fleet_embedding = self.robot_encoder(
            robot_features,
            robot_positions
        )
        task_features_normed = self.task_input_norm(task_features)
        task_embedding = self.task_encoder(task_features_normed)

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
        queue_features_normed = self.queue_input_norm(queue_features)
        global_state = torch.cat([
            graph_embedding,
            fleet_embedding,
            task_embedding,
            queue_features_normed
        ], dim=-1)

        state_value = self.critic(global_state)

        return action_logits, state_value, attention_info, graph_embedding, fleet_embedding

    def compute_debias_loss(
        self,
        fresh_logits: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute de-biasing loss from current episode.

        Args:
            fresh_logits: Live (gradient-connected) logits from the current PPO
                          update forward pass. When provided these are used instead
                          of the detached episode_logits stored during rollout, so
                          gradients can reach the GNN encoders.

        Returns:
            debias_loss: Scalar de-biasing penalty
            loss_breakdown: Dictionary with loss components
        """
        logits_to_use = fresh_logits if fresh_logits is not None else self.episode_logits
        if not self.use_debiasing or len(logits_to_use) < 2:
            return torch.tensor(0.0), {}

        # Cap sequence length to avoid T² OOM: _compute_task_similarity builds a [T,T]
        # similarity matrix and TemporalConsistencyDebiasing iterates O(T²) pairs.
        # With rollout_steps=1000 and max_assignments_per_step=10, uncapped T≈10000
        # would require ~400MB and ~50M Python iterations per K_epoch.
        _MAX_DEBIAS_SEQ = 256
        task_features = self.episode_task_features
        if len(logits_to_use) > _MAX_DEBIAS_SEQ:
            # Take a uniform stride subsample so the full rollout is represented
            step = len(logits_to_use) // _MAX_DEBIAS_SEQ
            indices = list(range(0, len(logits_to_use), step))[:_MAX_DEBIAS_SEQ]
            logits_to_use = [logits_to_use[i] for i in indices]
            if task_features and len(task_features) > _MAX_DEBIAS_SEQ:
                task_features = [task_features[i] for i in indices]

        debias_loss, breakdown = self.debiaser.compute_total_debias_loss(
            state_sequence=task_features,  # task features as state proxy
            action_sequence=self.episode_actions,
            action_logits_sequence=logits_to_use,
            task_features_sequence=task_features
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
