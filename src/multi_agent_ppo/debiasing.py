"""De-biasing utilities for autoregressive task allocation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class StateDeltaDebiasing(nn.Module):
    """Penalize large state changes."""

    def __init__(self, lambda_debias=0.1):
        super().__init__()
        self.lambda_debias = lambda_debias

    def compute_loss(
        self,
        state_sequence: List[torch.Tensor],
        action_sequence: List[int],
        device=None,
    ) -> torch.Tensor:
        dev = (state_sequence[0].device if state_sequence else device) or torch.device("cpu")
        if len(state_sequence) < 2:
            return torch.zeros((), device=dev)

        state_deltas = [
            torch.norm(state_sequence[t + 1] - state_sequence[t])
            for t in range(len(state_sequence) - 1)
        ]
        mean_delta = torch.mean(torch.stack(state_deltas))
        return self.lambda_debias * mean_delta


class TemporalConsistencyDebiasing(nn.Module):
    """Penalize inconsistent action distributions."""

    def __init__(self, lambda_consistency=0.05):
        super().__init__()
        self.lambda_consistency = lambda_consistency

    def compute_loss(
        self,
        action_logits_sequence: List[torch.Tensor],
        task_similarity_matrix: torch.Tensor,
        device=None,
    ) -> torch.Tensor:
        dev = (action_logits_sequence[0].device if action_logits_sequence else device) or torch.device("cpu")
        if len(action_logits_sequence) < 2:
            return torch.zeros((), device=dev)

        T = len(action_logits_sequence)
        consistency_penalties = []

        for i in range(T):
            for j in range(i + 1, T):
                similarity = task_similarity_matrix[i, j]
                if similarity > 0.7:
                    logits_i = action_logits_sequence[i]
                    logits_j = action_logits_sequence[j]

                    probs_i = F.softmax(logits_i, dim=-1)
                    probs_j = F.softmax(logits_j, dim=-1)

                    kl_div = F.kl_div(probs_j.log(), probs_i, reduction="batchmean")
                    consistency_penalties.append(kl_div * similarity)

        if consistency_penalties:
            return self.lambda_consistency * torch.mean(torch.stack(consistency_penalties))
        return torch.zeros((), device=dev)


class PermutationInvarianceDebiasing(nn.Module):
    """Penalize order-sensitive outputs."""

    def __init__(self, lambda_permute=0.1):
        super().__init__()
        self.lambda_permute = lambda_permute

    def compute_loss(
        self,
        policy_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        original_assignments: List[Tuple[int, int]],
        device=None,
    ) -> torch.Tensor:
        dev = (policy_outputs[0][1].device if policy_outputs else device) or torch.device("cpu")
        if len(policy_outputs) < 2:
            return torch.zeros((), device=dev)

        action_dists = [F.softmax(logits, dim=-1) for _, logits in policy_outputs]
        action_dist_stack = torch.stack(action_dists)
        action_variance = torch.var(action_dist_stack, dim=0).mean()
        return self.lambda_permute * action_variance


class CausalMaskingDebiasing(nn.Module):
    """Build causal masks."""

    def __init__(self):
        super().__init__()

    def create_causal_mask(self, num_tasks: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(num_tasks, num_tasks))
        return mask.bool()


class ComprehensiveDebiasing(nn.Module):
    """Combine debiasing terms."""

    def __init__(
        self,
        lambda_state_delta=0.1,
        lambda_consistency=0.05,
        lambda_permute=0.1,
        use_causal_mask=True,
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
        task_features_sequence: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        dev = action_logits_sequence[0].device if action_logits_sequence else torch.device("cpu")

        state_delta_loss = self.state_delta_debias.compute_loss(
            state_sequence,
            action_sequence,
            device=dev,
        )

        if len(task_features_sequence) > 1:
            task_similarity = self._compute_task_similarity(task_features_sequence)
            consistency_loss = self.temporal_consistency_debias.compute_loss(
                action_logits_sequence,
                task_similarity,
                device=dev,
            )
            similarity_threshold = 0.7
            upper_tri = torch.triu(task_similarity, diagonal=1)
            similar_pairs = int((upper_tri > similarity_threshold).sum().item())
            sim_mean = float(upper_tri[upper_tri > 0].mean().item()) if (upper_tri > 0).any() else 0.0
            sim_max = float(upper_tri.max().item()) if upper_tri.numel() > 0 else 0.0
        else:
            consistency_loss = torch.zeros((), device=dev)
            similarity_threshold = 0.7
            similar_pairs = 0
            sim_mean = 0.0
            sim_max = 0.0

        total_loss = state_delta_loss + consistency_loss
        loss_breakdown = {
            "state_delta": state_delta_loss.item(),
            "consistency": consistency_loss.item(),
            "total_debias": total_loss.item(),
            "debias_seq_len": len(task_features_sequence),
            "debias_sim_threshold": similarity_threshold,
            "debias_sim_pairs": similar_pairs,
            "debias_sim_mean": sim_mean,
            "debias_sim_max": sim_max,
        }

        return total_loss, loss_breakdown

    def _compute_task_similarity(self, task_features: List[torch.Tensor]) -> torch.Tensor:
        features_matrix = torch.stack(task_features)
        features_norm = F.normalize(features_matrix, p=2, dim=1)
        return torch.mm(features_norm, features_norm.t())

    def get_causal_mask(self, num_tasks: int) -> torch.Tensor:
        if self.causal_masking:
            return self.causal_masking.create_causal_mask(num_tasks)
        return None
