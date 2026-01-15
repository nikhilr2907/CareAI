"""
De-biasing mechanisms for autoregressive task allocation.

Based on: "Autoregressive Policy Optimization for Constrained Allocation Tasks"
(Winkel et al., 2024, arXiv:2409.18735)

Problem: In autoregressive allocation, later decisions are biased by earlier ones
because the state changes after each assignment.

Solution: Multiple de-biasing strategies to reduce sequential dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class StateDeltaDebiasing(nn.Module):
    """
    Penalizes large state changes between sequential decisions.

    Intuition: If assigning Task 1 dramatically changes the state seen by Task 2,
    the decisions are highly dependent. We want more independence.
    """

    def __init__(self, lambda_debias=0.1):
        super().__init__()
        self.lambda_debias = lambda_debias

    def compute_loss(
        self,
        state_sequence: List[torch.Tensor],
        action_sequence: List[int]
    ) -> torch.Tensor:
        """
        Compute de-biasing loss from state changes.

        Args:
            state_sequence: List of states [s_0, s_1, ..., s_T]
            action_sequence: List of actions [a_0, a_1, ..., a_{T-1}]

        Returns:
            debias_loss: Scalar penalty for large state changes
        """
        if len(state_sequence) < 2:
            return torch.tensor(0.0)

        state_deltas = []

        for t in range(len(state_sequence) - 1):
            # Measure state change
            delta = torch.norm(state_sequence[t+1] - state_sequence[t])
            state_deltas.append(delta)

        # Penalize large state changes
        if state_deltas:
            mean_delta = torch.mean(torch.stack(state_deltas))
            debias_loss = self.lambda_debias * mean_delta
        else:
            debias_loss = torch.tensor(0.0)

        return debias_loss


class TemporalConsistencyDebiasing(nn.Module):
    """
    Encourages consistent action distributions across time steps.

    Intuition: The policy should give similar action probabilities for similar
    tasks, regardless of when they appear in the sequence.
    """

    def __init__(self, lambda_consistency=0.05):
        super().__init__()
        self.lambda_consistency = lambda_consistency

    def compute_loss(
        self,
        action_logits_sequence: List[torch.Tensor],
        task_similarity_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize inconsistent action distributions for similar tasks.

        Args:
            action_logits_sequence: List of action logits for each step
            task_similarity_matrix: [T, T] - pairwise task similarities

        Returns:
            consistency_loss: Scalar penalty for inconsistency
        """
        if len(action_logits_sequence) < 2:
            return torch.tensor(0.0)

        T = len(action_logits_sequence)
        consistency_penalties = []

        for i in range(T):
            for j in range(i+1, T):
                # If tasks i and j are similar...
                similarity = task_similarity_matrix[i, j]

                if similarity > 0.7:  # Threshold for "similar"
                    # ...their action distributions should be similar
                    logits_i = action_logits_sequence[i]
                    logits_j = action_logits_sequence[j]

                    # KL divergence between distributions
                    probs_i = F.softmax(logits_i, dim=-1)
                    probs_j = F.softmax(logits_j, dim=-1)

                    kl_div = F.kl_div(
                        probs_j.log(),
                        probs_i,
                        reduction='batchmean'
                    )

                    consistency_penalties.append(kl_div * similarity)

        if consistency_penalties:
            consistency_loss = self.lambda_consistency * torch.mean(
                torch.stack(consistency_penalties)
            )
        else:
            consistency_loss = torch.tensor(0.0)

        return consistency_loss


class PermutationInvarianceDebiasing(nn.Module):
    """
    Encourages policy to be invariant to task ordering.

    Intuition: The optimal allocation shouldn't change if we reorder tasks
    (assuming no deadlines).

    Implementation: Train on multiple random permutations of the same task set.
    """

    def __init__(self, lambda_permute=0.1):
        super().__init__()
        self.lambda_permute = lambda_permute

    def compute_loss(
        self,
        policy_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        original_assignments: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Compare policy outputs on different task orderings.

        Args:
            policy_outputs: List of (state, action_logits) for each permutation
            original_assignments: List of (task_id, robot_id) from canonical order

        Returns:
            permutation_loss: Penalty for order-dependent decisions
        """
        if len(policy_outputs) < 2:
            return torch.tensor(0.0)

        # Extract action distributions
        action_dists = [F.softmax(logits, dim=-1) for _, logits in policy_outputs]

        # Penalize high variance in action distributions across permutations
        action_dist_stack = torch.stack(action_dists)  # [num_perms, num_actions]
        action_variance = torch.var(action_dist_stack, dim=0).mean()

        permutation_loss = self.lambda_permute * action_variance

        return permutation_loss


class CausalMaskingDebiasing(nn.Module):
    """
    Masks future task information to prevent look-ahead bias.

    Intuition: When deciding on Task t, the model shouldn't "know" about
    tasks t+1, t+2, etc. (except their existence in the queue).
    """

    def __init__(self):
        super().__init__()

    def create_causal_mask(self, num_tasks: int) -> torch.Tensor:
        """
        Create causal attention mask for tasks.

        Args:
            num_tasks: Number of tasks in sequence

        Returns:
            mask: [num_tasks, num_tasks] - causal mask
        """
        # Lower triangular matrix (can only attend to current and past)
        mask = torch.tril(torch.ones(num_tasks, num_tasks))

        return mask.bool()


class ComprehensiveDebiasing(nn.Module):
    """
    Combines all de-biasing strategies.
    """

    def __init__(
        self,
        lambda_state_delta=0.1,
        lambda_consistency=0.05,
        lambda_permute=0.1,
        use_causal_mask=True
    ):
        super().__init__()

        self.state_delta_debias = StateDeltaDebiasing(lambda_state_delta)
        self.temporal_consistency_debias = TemporalConsistencyDebiasing(lambda_consistency)
        self.permutation_debias = PermutationInvarianceDebiasing(lambda_permute)
        self.causal_masking = CausalMaskingDebiasing() if use_causal_mask else None

        self.use_causal_mask = use_causal_mask

    def compute_total_debias_loss(
        self,
        state_sequence: List[torch.Tensor],
        action_sequence: List[int],
        action_logits_sequence: List[torch.Tensor],
        task_features_sequence: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total de-biasing loss.

        Args:
            state_sequence: List of states
            action_sequence: List of actions taken
            action_logits_sequence: List of action logits
            task_features_sequence: List of task feature vectors

        Returns:
            total_loss: Combined de-biasing loss
            loss_breakdown: Dictionary with individual loss components
        """
        # 1. State delta loss
        state_delta_loss = self.state_delta_debias.compute_loss(
            state_sequence,
            action_sequence
        )

        # 2. Temporal consistency loss
        if len(task_features_sequence) > 1:
            # Compute task similarity matrix
            task_similarity = self._compute_task_similarity(task_features_sequence)

            consistency_loss = self.temporal_consistency_debias.compute_loss(
                action_logits_sequence,
                task_similarity
            )
            similarity_threshold = 0.7
            upper_tri = torch.triu(task_similarity, diagonal=1)
            similar_pairs = int((upper_tri > similarity_threshold).sum().item())
            sim_mean = float(upper_tri[upper_tri > 0].mean().item()) if (upper_tri > 0).any() else 0.0
            sim_max = float(upper_tri.max().item()) if upper_tri.numel() > 0 else 0.0
        else:
            consistency_loss = torch.tensor(0.0)
            similarity_threshold = 0.7
            similar_pairs = 0
            sim_mean = 0.0
            sim_max = 0.0

        # 3. Total loss
        total_loss = state_delta_loss + consistency_loss

        loss_breakdown = {
            'state_delta': state_delta_loss.item(),
            'consistency': consistency_loss.item(),
            'total_debias': total_loss.item(),
            'debias_seq_len': len(task_features_sequence),
            'debias_sim_threshold': similarity_threshold,
            'debias_sim_pairs': similar_pairs,
            'debias_sim_mean': sim_mean,
            'debias_sim_max': sim_max
        }

        return total_loss, loss_breakdown

    def _compute_task_similarity(
        self,
        task_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pairwise task similarity matrix.

        Args:
            task_features: List of task feature vectors [T × feat_dim]

        Returns:
            similarity_matrix: [T, T] - cosine similarities
        """
        T = len(task_features)

        # Stack into matrix
        features_matrix = torch.stack(task_features)  # [T, feat_dim]

        # Normalize
        features_norm = F.normalize(features_matrix, p=2, dim=1)

        # Cosine similarity
        similarity_matrix = torch.mm(features_norm, features_norm.t())

        return similarity_matrix

    def get_causal_mask(self, num_tasks: int) -> torch.Tensor:
        """Get causal mask for attention."""
        if self.causal_masking:
            return self.causal_masking.create_causal_mask(num_tasks)
        return None


def test_debiasing():
    """Test de-biasing modules."""
    print("Testing De-biasing Mechanisms...")

    # Create dummy sequences
    state_dim = 193
    num_steps = 5
    num_actions = 6

    state_sequence = [torch.randn(state_dim) for _ in range(num_steps)]
    action_sequence = [0, 1, 2, 0, 5]  # Dummy actions
    action_logits_sequence = [torch.randn(num_actions) for _ in range(num_steps)]
    task_features_sequence = [torch.randn(12) for _ in range(num_steps)]

    # Test comprehensive debiasing
    debiaser = ComprehensiveDebiasing(
        lambda_state_delta=0.1,
        lambda_consistency=0.05
    )

    total_loss, breakdown = debiaser.compute_total_debias_loss(
        state_sequence,
        action_sequence,
        action_logits_sequence,
        task_features_sequence
    )

    print(f"\nTotal de-biasing loss: {total_loss.item():.4f}")
    print("\nLoss breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.4f}")

    # Test causal mask
    print("\nCausal mask (5 tasks):")
    mask = debiaser.get_causal_mask(5)
    print(mask.int())

    print("\n✓ De-biasing mechanisms working!")


if __name__ == '__main__':
    test_debiasing()
