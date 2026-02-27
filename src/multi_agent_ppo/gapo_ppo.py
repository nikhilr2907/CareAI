"""
GAPO PPO Trainer with de-biasing.

Implements PPO training for GAPO policy network with:
- Graph-structured states
- De-biasing loss
- Attention visualization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

from .gapo_policy import GAPOPolicyNetwork
from .task_creation_actor import TaskCreationActor


class Memory:
    """Memory buffer for PPO."""

    def __init__(self):
        self.state_dicts = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.robot_masks = []
        # Track which memory indices belong to the same assignment step
        # Each entry is a list of memory indices for assignments made in that step
        self.step_assignment_groups = []

    def clear_memory(self):
        del self.state_dicts[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.robot_masks[:]
        del self.step_assignment_groups[:]


class GAPOPPO:
    """
    PPO trainer for GAPO policy network.
    """

    def __init__(
        self,
        node_continuous_dim=8,
        num_node_types=4,
        num_departments=10,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=21,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4,
        robot_feat_dim=19,
        task_feat_dim=15,
        queue_feat_dim=16,
        sku_feat_dim=None,
        sku_embed_dim=16,
        hidden_dim=64,
        num_attention_heads=4,
        lr=0.0003,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        use_debiasing=True,
        lambda_debias=0.1,
        lambda_gae=0.95,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        min_adv_std: float = 1e-3,
        normalize_returns=True,
        ranking_max_pairs_per_group: int = 64,
        ranking_min_adv_gap: float = 1e-4,
        task_creation_actor: Optional[TaskCreationActor] = None,
        lr_creation: float = 3e-4,
        lambda_creation_scorer: float = 1.0,
        lambda_creation_factorizer: float = 0.1,
        lambda_creation_kl: float = 0.01,
        device='cpu',
        logger: Optional[logging.Logger] = None
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambda_gae = lambda_gae
        self.use_debiasing = use_debiasing
        self.normalize_returns = normalize_returns
        self.device = torch.device(device)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.min_adv_std = min_adv_std
        self.ranking_max_pairs_per_group = int(max(1, ranking_max_pairs_per_group))
        self.ranking_min_adv_gap = float(max(0.0, ranking_min_adv_gap))

        # GAPO policy network
        self.policy = GAPOPolicyNetwork(
            node_continuous_dim=node_continuous_dim,
            num_node_types=num_node_types,
            num_departments=num_departments,
            num_shift_periods=num_shift_periods,
            num_day_types=num_day_types,
            edge_feat_dim=edge_feat_dim,
            node_type_embedding_dim=node_type_embedding_dim,
            department_embedding_dim=department_embedding_dim,
            shift_embedding_dim=shift_embedding_dim,
            day_type_embedding_dim=day_type_embedding_dim,
            robot_feat_dim=robot_feat_dim,
            task_feat_dim=task_feat_dim,
            queue_feat_dim=queue_feat_dim,
            sku_feat_dim=sku_feat_dim,
            sku_embed_dim=sku_embed_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            use_debiasing=use_debiasing,
            lambda_state_delta=lambda_debias
        ).to(self.device)

        # Old policy for PPO ratio
        self.policy_old = GAPOPolicyNetwork(
            node_continuous_dim=node_continuous_dim,
            num_node_types=num_node_types,
            num_departments=num_departments,
            num_shift_periods=num_shift_periods,
            num_day_types=num_day_types,
            edge_feat_dim=edge_feat_dim,
            node_type_embedding_dim=node_type_embedding_dim,
            department_embedding_dim=department_embedding_dim,
            shift_embedding_dim=shift_embedding_dim,
            day_type_embedding_dim=day_type_embedding_dim,
            robot_feat_dim=robot_feat_dim,
            task_feat_dim=task_feat_dim,
            queue_feat_dim=queue_feat_dim,
            sku_feat_dim=sku_feat_dim,
            sku_embed_dim=sku_embed_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            use_debiasing=False  # Don't need debiasing in old policy
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        # policy_old is used exclusively for rollout inference — keep it in eval mode
        # so any remaining dropout (e.g. critic head) is disabled during rollout.
        self.policy_old.eval()

        self.MseLoss = nn.MSELoss()

        # Optimizer (actor/critic param groups)
        actor_lr = actor_lr if actor_lr is not None else lr
        critic_lr = critic_lr if critic_lr is not None else lr
        actor_params = []
        critic_params = []
        for name, param in self.policy.named_parameters():
            if "critic" in name:
                critic_params.append(param)
            else:
                actor_params.append(param)
        self.optimizer = optim.Adam(
            [
                {"params": actor_params, "lr": actor_lr},
                {"params": critic_params, "lr": critic_lr},
            ]
        )

        # Task creation actor (#7/#8) — separate optimizer, separate loss signal
        self.task_creation_actor = task_creation_actor
        self.lambda_creation_scorer = lambda_creation_scorer
        self.lambda_creation_factorizer = lambda_creation_factorizer
        self.lambda_creation_kl = lambda_creation_kl
        if task_creation_actor is not None:
            self.creation_optimizer = optim.Adam(
                task_creation_actor.parameters(), lr=lr_creation
            )
        else:
            self.creation_optimizer = None

    def select_action(
        self,
        state_dict: Dict[str, np.ndarray],
        memory: Memory,
        robot_mask: np.ndarray = None
    ) -> int:
        """
        Select action using current policy.

        Args:
            state_dict: State dictionary from environment
            memory: Memory buffer to store experience
            robot_mask: Boolean mask for available robots

        Returns:
            action: Selected action
        """
        # Convert numpy arrays to tensors
        state_dict_tensor = self._state_dict_to_tensor(state_dict)

        # Convert mask
        if robot_mask is not None:
            robot_mask_tensor = torch.tensor(robot_mask, dtype=torch.bool).to(self.device)
        else:
            robot_mask_tensor = None

        # Select action
        with torch.no_grad():
            action, log_prob = self.policy_old.select_action(
                state_dict_tensor,
                robot_mask_tensor
            )

        # Store in memory
        memory.state_dicts.append(state_dict)
        memory.actions.append(action)
        memory.logprobs.append(log_prob.item())
        memory.robot_masks.append(robot_mask)

        # Record action for de-biasing
        self.policy.record_action(action)

        return action

    def select_action_greedy(
        self,
        state_dict: Dict[str, np.ndarray],
        robot_mask: np.ndarray = None
    ) -> int:
        """
        Select action greedily (for deployment/evaluation).

        Args:
            state_dict: State dictionary from environment
            robot_mask: Boolean mask for available robots

        Returns:
            action: Selected action (greedy)
        """
        # Convert numpy arrays to tensors
        state_dict_tensor = self._state_dict_to_tensor(state_dict)

        # Convert mask
        if robot_mask is not None:
            robot_mask_tensor = torch.tensor(robot_mask, dtype=torch.bool).to(self.device)
        else:
            robot_mask_tensor = None

        # Select action greedily
        with torch.no_grad():
            action, _ = self.policy.select_action(
                state_dict_tensor,
                robot_mask_tensor,
                deterministic=True
            )

        return action

    def score_and_rank_tasks(
        self,
        pending_tasks,
        state_dict: Dict[str, np.ndarray],
        current_time: float,
        temperature: float = 1.0
    ) -> List:
        """
        Score and rank pending tasks using context-aware scorer with stochastic sampling.

        Args:
            pending_tasks: List of Task objects
            state_dict: Current environment state (numpy)
            current_time: Current simulation time
            temperature: Sampling temperature (higher = more random, 0 = deterministic)

        Returns:
            ranked_tasks: Reordered task list
        """
        if not pending_tasks or len(pending_tasks) <= 1:
            return pending_tasks

        # Build candidate feature tensor.
        task_features = np.stack([t.get_features(current_time) for t in pending_tasks])
        task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(self.device)
        state_tensor = self._state_dict_to_tensor(state_dict)

        with torch.no_grad():
            graph_embedding, fleet_embedding = self.policy.encode_context(state_tensor)
            base_scores = self.policy.score_tasks(
                task_features_tensor, graph_embedding, fleet_embedding
            )
            if temperature > 0:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(base_scores).clamp(min=1e-8)))
                noisy_scores = base_scores + temperature * gumbel_noise
                _, indices = noisy_scores.sort(descending=True)
            else:
                _, indices = base_scores.sort(descending=True)

        indices_np = indices.detach().cpu().numpy()
        ranked = [pending_tasks[i] for i in indices_np]

        base_scores_np = base_scores.detach().cpu().numpy()
        for rank, task_idx in enumerate(indices_np):
            task = pending_tasks[task_idx]
            task.queue_position = rank
            task.learned_score = float(base_scores_np[task_idx])

        return ranked

    def compute_ranking_loss(
        self,
        memory: 'Memory',
        state_dict_tensors: List[Dict[str, torch.Tensor]],
        advantages: torch.Tensor,
        context_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Compute ranking loss: tasks that led to higher advantages should have higher scores.

        Uses a margin ranking loss on pairs of tasks within the same assignment step.

        Args:
            memory: Memory buffer with step_assignment_groups
            state_dict_tensors: Tensorized state dicts
            advantages: GAE advantages for each decision
            context_cache: Optional dict mapping memory index → (graph_emb, fleet_emb)
                           pre-computed by evaluate_actions to avoid redundant GNN passes.

        Returns:
            ranking_loss: Scalar loss
        """
        if not memory.step_assignment_groups:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0

        for group_indices in memory.step_assignment_groups:
            valid = [idx for idx in group_indices if 0 <= idx < len(advantages) and idx < len(state_dict_tensors)]
            if len(valid) < 2:
                continue

            # Compute each score ONCE per assignment in this group.
            # Use cached (graph_emb, fleet_emb) when available to skip encode_context.
            group_scores = []
            group_advs = []
            for idx in valid:
                sd = state_dict_tensors[idx]
                if context_cache is not None and idx in context_cache:
                    graph_emb, fleet_emb = context_cache[idx]
                else:
                    graph_emb, fleet_emb = self.policy.encode_context(sd)
                score = self.policy.score_tasks(
                    sd["task_features"].unsqueeze(0),
                    graph_emb,
                    fleet_emb
                ).squeeze(0)
                group_scores.append(score)
                group_advs.append(advantages[idx])

            scores = torch.stack(group_scores, dim=0)
            advs = torch.stack(group_advs, dim=0)

            pair_indices = self._sample_group_pairs(
                group_size=len(valid),
                max_pairs=self.ranking_max_pairs_per_group
            )
            for i, j in pair_indices:
                adv_delta = advs[i] - advs[j]
                if torch.abs(adv_delta).item() <= self.ranking_min_adv_gap:
                    continue
                target = torch.sign(adv_delta).view(1)
                margin_loss = F.margin_ranking_loss(
                    scores[i].view(1),
                    scores[j].view(1),
                    target,
                    margin=0.1
                )
                total_loss = total_loss + margin_loss
                num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / num_pairs

    def _sample_group_pairs(self, group_size: int, max_pairs: int) -> List[Tuple[int, int]]:
        """Enumerate or subsample pair indices for one assignment group."""
        if group_size < 2:
            return []

        all_pairs = [(i, j) for i in range(group_size) for j in range(i + 1, group_size)]
        if len(all_pairs) <= max_pairs:
            return all_pairs

        perm = torch.randperm(len(all_pairs), device=self.device)[:max_pairs].tolist()
        return [all_pairs[k] for k in perm]

    def update_task_creation_actor(self) -> torch.Tensor:
        """
        Train the task creation actor (#7 factorizer + #8 scorer) using completion
        credits accumulated in its buffer since the last call.

        Uses a separate optimizer from the main PPO policy.
        Returns the scalar creation loss (0.0 if no completed records or actor is None).
        """
        if self.task_creation_actor is None or self.creation_optimizer is None:
            return torch.tensor(0.0, device=self.device)

        loss = self.task_creation_actor.compute_creation_loss(
            scorer_loss_coef=self.lambda_creation_scorer,
            factorizer_loss_coef=self.lambda_creation_factorizer,
            kl_coef=self.lambda_creation_kl,
        )

        if loss.requires_grad:
            self.creation_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.task_creation_actor.parameters()), 0.5
            )
            self.creation_optimizer.step()

        return loss

    def update(self, memory: Memory, next_state_dict: Dict[str, np.ndarray] = None):
        """
        Update policy using PPO with GAE advantages.

        Args:
            memory: Memory buffer with experiences
            next_state_dict: Next state for bootstrapping (continuous tasks)
        """
        # Ensure policy is in train mode for gradient updates; policy_old stays in eval.
        self.policy.train()

        # Convert to tensors
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)

        # Convert state dicts to tensors
        state_dict_tensors = [
            self._state_dict_to_tensor(sd) for sd in memory.state_dicts
        ]

        # Populate episode_task_features for de-biasing.
        # During rollout, policy_old.select_action() is used rather than policy.forward(),
        # so self.policy.episode_task_features is never filled during collection.
        # Extract task features from stored state dicts here so the temporal_consistency
        # and state_delta debiasing losses have real data to work with.
        if self.use_debiasing:
            self.policy.episode_task_features = [
                sd['task_features'].detach() for sd in state_dict_tensors
            ]

        robot_mask_tensors = []
        for mask in memory.robot_masks:
            if mask is not None:
                robot_mask_tensors.append(
                    torch.tensor(mask, dtype=torch.bool).to(self.device)
                )
            else:
                robot_mask_tensors.append(None)

        # Compute state values for all states
        with torch.no_grad():
            _, values, _, _, _, _ = self.policy.evaluate_actions(
                state_dict_tensors,
                old_actions,
                robot_mask_tensors
            )
            values = values.squeeze(-1)

            # Get value of next state for bootstrapping
            if next_state_dict is not None:
                next_state_tensor = self._state_dict_to_tensor(next_state_dict)
                _, next_value, _, _, _, _ = self.policy.evaluate_actions(
                    [next_state_tensor],
                    torch.zeros(1, dtype=torch.long).to(self.device),
                    [robot_mask_tensors[-1] if robot_mask_tensors else None]
                )
                next_value = next_value.squeeze().to(self.device)
            else:
                next_value = torch.tensor(0.0).to(self.device)

        # Compute GAE advantages from raw rewards.
        # Advantages and returns are kept on the raw reward scale here.
        # The critic trains on raw-scale returns (correct value targets).
        # The actor receives separately normalized advantages (see below).
        raw_advantages = self._compute_gae(
            rewards_tensor,
            values,
            next_value,
            memory.is_terminals
        )

        # Critic targets: raw advantage + baseline (= discounted return estimate).
        # These are on the same scale as the rewards, which is what the critic should predict.
        returns = (raw_advantages + values).detach()

        # Normalize advantages for actor gradient only.
        # Prevents high-variance rewards from dominating gradient magnitude.
        # If the buffer has no meaningful signal (all rewards zero, adv_std below threshold),
        # skip scaling to avoid amplifying random critic-init noise by min_adv_std.
        adv_std = raw_advantages.std()
        if adv_std > self.min_adv_std:
            advantages = (raw_advantages - raw_advantages.mean()) / (adv_std + 1e-7)
        else:
            # No meaningful signal — just mean-center without scaling.
            # Using min_adv_std as a floor here would amplify noise 100-1000x.
            advantages = raw_advantages - raw_advantages.mean()

        self.logger.info(f"  Starting PPO update ({self.K_epochs} epochs)...")

        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):
            # Evaluate actions — raw_logits are live (gradient-connected) for de-biasing.
            # graph_emb_batch / fleet_emb_batch are reused by compute_ranking_loss to avoid
            # redundant GNN forward passes (one encode_context call per group member saved).
            logprobs, state_values, dist_entropy, raw_logits, graph_emb_batch, fleet_emb_batch = self.policy.evaluate_actions(
                state_dict_tensors,
                old_actions,
                robot_mask_tensors
            )

            state_values = state_values.squeeze(-1)

            # Build per-memory-index context cache for ranking loss.
            # graph_emb_batch[i] / fleet_emb_batch[i] correspond to state_dict_tensors[i].
            context_cache = {
                i: (graph_emb_batch[i], fleet_emb_batch[i])
                for i in range(graph_emb_batch.shape[0])
            }

            # PPO ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())
            clip_fraction = (torch.abs(ratios - 1.0) > self.eps_clip).float().mean()

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (fit to GAE-computed returns)
            critic_loss = 0.5 * self.MseLoss(state_values, returns)

            # Entropy bonus (exploration)
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            # De-biasing loss — pass live logits so gradients reach GNN encoders
            if self.use_debiasing:
                fresh_logits_list = list(raw_logits.unbind(0))
                debias_loss, debias_breakdown = self.policy.compute_debias_loss(
                    fresh_logits=fresh_logits_list
                )
            else:
                debias_loss = torch.tensor(0.0)
                debias_breakdown = {}

            # Ranking loss for task scorer (#6) — uses cached embeddings
            ranking_loss = self.compute_ranking_loss(
                memory, state_dict_tensors, advantages, context_cache=context_cache
            )
            lambda_ranking = 0.05

            # Total loss
            loss = (
                actor_loss
                + (self.critic_coef * critic_loss)
                + entropy_loss
                + debias_loss
                + (lambda_ranking * ranking_loss)
            )

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()), 0.5)
            self.optimizer.step()

            # Log losses for each epoch
            self.logger.info(
                f"    Epoch {epoch+1}/{self.K_epochs}: "
                f"Loss={loss.item():.4f} | "
                f"Actor={actor_loss.item():.4f} | "
                f"Critic={critic_loss.item():.4f} | "
                f"Entropy={entropy_loss.item():.4f} | "
                f"Ranking={ranking_loss.item():.4f}" +
                (f" | Debias={debias_loss.item():.4f}" if self.use_debiasing else "")
            )

            # Store last loss info
            if epoch == 0:
                self.last_loss_info = {
                    'total_loss': loss.item(),
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'ranking_loss': ranking_loss.item(),
                    'debias_loss': debias_loss.item() if self.use_debiasing else 0.0,
                    'clip_fraction': clip_fraction.item(),
                    'grad_norm': float(grad_norm),
                    # Advantage diagnostics: low adv_std means sparse/no rewards this rollout
                    'adv_std_raw': float(adv_std.item()),
                    'returns_mean': float(returns.mean().item()),
                    'returns_std': float(returns.std().item()),
                    **debias_breakdown
                }

        self.logger.info(f"  PPO update completed.")

        # Update task creation actor (#7/#8) from completion credits
        creation_loss = self.update_task_creation_actor()
        if creation_loss.item() != 0.0:
            self.logger.info(f"  Creation actor loss: {creation_loss.item():.4f}")
        self.last_loss_info['creation_loss'] = creation_loss.item()

        # Copy new weights to old policy and restore eval mode for next rollout
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

        # Reset episode tracking
        self.policy.reset_episode_tracking()

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        is_terminals: List[bool]
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE formula:
            A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
        where:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - terminal) - V(s_t)

        Args:
            rewards: Rewards [T]
            values: State values [T]
            next_value: Value of state after last timestep (for bootstrapping)
            is_terminals: Terminal flags [T]

        Returns:
            advantages: GAE advantages [T]
        """
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32).to(self.device)
        gae = 0

        # Compute GAE in reverse order
        for t in reversed(range(T)):
            if t == T - 1:
                # Bootstrap from next state
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - terminal) - V(s_t)
            terminal_mask = 0.0 if is_terminals[t] else 1.0
            delta = rewards[t] + self.gamma * next_value_t * terminal_mask - values[t]

            # GAE: A_t = δ_t + γλ * A_{t+1} * (1 - terminal)
            gae = delta + self.gamma * self.lambda_gae * gae * terminal_mask
            advantages[t] = gae

        return advantages

    def _state_dict_to_tensor(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy state dict to tensor dict, clamping extreme values."""
        tensor_dict = {}
        int64_keys = {"node_categorical", "edge_index", "edge_node_indices"}

        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                if key in int64_keys:
                    tensor_dict[key] = torch.tensor(value, dtype=torch.int64).to(self.device)
                else:
                    # Replace NaN/Inf before converting to tensor
                    value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
                    tensor_dict[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
            else:
                tensor_dict[key] = value

        return tensor_dict

    def save(self, filepath: str):
        """Save policy network."""
        payload = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(payload, filepath)

    def load(self, filepath: str):
        """Load policy network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_last_loss_info(self) -> Dict:
        """Get last training loss breakdown."""
        return getattr(self, 'last_loss_info', {})
