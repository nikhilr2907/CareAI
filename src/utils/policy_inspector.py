"""
Tools to inspect what the policy has learned.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class PolicyInspector:
    """Analyze trained policy weights, attention patterns, and decision-making."""

    def __init__(self, policy, device='cpu'):
        self.policy = policy
        self.device = device

    def get_weight_stats(self) -> Dict:
        """Analyze all weight matrices in the policy."""
        stats = {}
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                w = param.data.cpu().numpy()
                stats[name] = {
                    'shape': tuple(w.shape),
                    'mean': float(np.mean(w)),
                    'std': float(np.std(w)),
                    'min': float(np.min(w)),
                    'max': float(np.max(w)),
                    'norm': float(np.linalg.norm(w)),
                    'dead_ratio': float(np.mean(w == 0)),  # Frozen/dead weights
                }
        return stats

    def get_attention_weights(self, state_dict) -> Dict:
        """Extract attention weights from policy for a given state."""
        state_tensor = self.policy._state_dict_to_tensor(state_dict)

        self.policy.eval()
        with torch.no_grad():
            # Get attention outputs from the policy's attention layers
            # This requires access to the policy's internal attention modules
            graph_emb, fleet_emb = self.policy.encode_context(state_tensor)

            # Try to extract attention weights if available
            attn_dict = {}
            if hasattr(self.policy, 'attention_layers'):
                for i, attn_layer in enumerate(self.policy.attention_layers):
                    if hasattr(attn_layer, 'attn_weights'):
                        attn_dict[f'attention_{i}'] = attn_layer.attn_weights.cpu().numpy()

            return {
                'graph_embedding_norm': float(torch.norm(graph_emb).item()),
                'fleet_embedding_norm': float(torch.norm(fleet_emb).item()),
                'graph_embedding_mean': float(torch.mean(graph_emb).item()),
                'attention_weights': attn_dict,
            }

    def get_action_distribution(self, state_dict, num_robots: int) -> Dict:
        """Get predicted action probabilities for current state."""
        state_tensor = self.policy._state_dict_to_tensor(state_dict)
        robot_mask = np.ones(num_robots, dtype=bool)
        mask_tensor = torch.tensor(robot_mask, dtype=torch.bool).to(self.policy.device)

        self.policy.eval()
        with torch.no_grad():
            action_dist, value, _, _, _, _ = self.policy.evaluate_actions(
                [state_tensor],
                torch.arange(num_robots, dtype=torch.long).to(self.policy.device),
                [mask_tensor]
            )

            # Extract probabilities
            if hasattr(action_dist, 'probs'):
                probs = action_dist.probs[0].cpu().numpy()
            else:
                # Fallback if distribution is categorical
                probs = torch.softmax(action_dist, dim=-1)[0].cpu().numpy()

            return {
                'action_probabilities': {f'robot_{i}': float(p) for i, p in enumerate(probs)},
                'predicted_value': float(value[0].item()),
                'entropy': float(-np.sum(probs * np.log(probs + 1e-8))),
                'max_prob_robot': int(np.argmax(probs)),
                'probability_distribution': 'uniform' if np.std(probs) < 0.05 else 'focused',
            }

    def log_learning_state(self, logger, loss_info: Dict, iteration: int):
        """Log a comprehensive snapshot of what the policy has learned."""

        weight_stats = self.get_weight_stats()

        # Check for learning signals
        frozen_params = sum(1 for name, stats in weight_stats.items() if stats['dead_ratio'] > 0.5)
        total_params = len(weight_stats)

        avg_norm = np.mean([stats['norm'] for stats in weight_stats.values()])
        avg_mean = np.mean([abs(stats['mean']) for stats in weight_stats.values()])

        actor_loss = loss_info.get('actor_loss', 0)
        critic_loss = loss_info.get('critic_loss', 0)
        grad_norm = loss_info.get('grad_norm', 0)

        logger.info(
            f"[POLICY STATE iter={iteration}] "
            f"Weights: norm={avg_norm:.3f} mean={avg_mean:.3f} frozen={frozen_params}/{total_params} | "
            f"Loss: actor={actor_loss:.4f} critic={critic_loss:.2f} | "
            f"Grad: norm={grad_norm:.2f}"
        )

        # Warning flags
        if frozen_params > total_params * 0.5:
            logger.warning(f"  [WARNING] Many frozen weights ({frozen_params}/{total_params}) - learning has stalled!")

        if grad_norm > 10:
            logger.warning(f"  [WARNING] Exploding gradients (norm={grad_norm:.2f}) - reducing learning rate!")

        if actor_loss == 0.0:
            logger.warning(f"  [WARNING] Zero actor loss - policy not receiving advantage signal!")

        if critic_loss > 100:
            logger.warning(f"  [WARNING] High critic loss ({critic_loss:.2f}) - value function unstable!")
