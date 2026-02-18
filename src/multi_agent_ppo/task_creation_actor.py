"""
Task-creation actor components.

This module is intentionally decoupled from GAPO allocation policy code:
- Bayesian factorizer: state-conditional stochastic shortlist proposal
- Deterministic scorer: context-aware scoring of candidate tasks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TaskCreationRanking:
    """Outputs from task-creation ranking."""

    ranked_indices: torch.Tensor
    shortlist_indices: torch.Tensor
    scorer_scores: torch.Tensor
    factor_logits: torch.Tensor
    factor_kl: torch.Tensor


class BayesianCandidateFactorizer(nn.Module):
    """
    State-conditional Bayesian linear factorizer over candidate embeddings.

    Given state context s and candidate features x_i:
      - Encode x_i -> phi_i
      - Infer posterior q(w|s) = N(mu(s), diag(sigma^2(s)))
      - Sample w ~ q(w|s)
      - Logit_i = <w, phi_i>
    """

    def __init__(
        self,
        state_context_dim: int,
        candidate_feat_dim: int,
        embed_dim: int = 48,
        hidden_dim: int = 128,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.prior_sigma = float(max(prior_sigma, 1e-6))
        self.log_prior_var = float(2.0 * torch.log(torch.tensor(self.prior_sigma)).item())

        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_feat_dim),
            nn.Linear(candidate_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.LayerNorm(state_context_dim),
            nn.Linear(state_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * 2),
        )

    def _sample_weight(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        deterministic: bool,
    ) -> torch.Tensor:
        if deterministic:
            return mean
        noise = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_var) * noise

    def _kl_to_prior(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # KL[q || p] for diagonal Gaussian q against zero-mean isotropic Gaussian prior.
        prior_var = self.prior_sigma ** 2
        kl_per_dim = 0.5 * (
            (torch.exp(log_var) + mean.pow(2)) / prior_var - 1.0 - log_var + self.log_prior_var
        )
        return kl_per_dim.sum()

    def forward(
        self,
        state_context: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_context: [state_context_dim]
            candidate_features: [num_candidates, candidate_feat_dim]
            candidate_mask: [num_candidates] bool, True = feasible
            deterministic: use posterior mean (no sampling)
        """
        encoded_candidates = self.candidate_encoder(candidate_features)  # [N, E]

        posterior_params = self.posterior_net(state_context)  # [2E]
        mean, log_var = torch.chunk(posterior_params, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-8.0, max=4.0)

        sampled_weight = self._sample_weight(mean, log_var, deterministic=deterministic)  # [E]
        logits = torch.matmul(encoded_candidates, sampled_weight)  # [N]

        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, float("-inf"))

        return {
            "logits": logits,
            "kl": self._kl_to_prior(mean, log_var),
            "posterior_mean": mean,
            "posterior_log_var": log_var,
        }


class TaskCreationScorer(nn.Module):
    """Deterministic context-aware task scorer for shortlisted candidates."""

    def __init__(
        self,
        state_context_dim: int,
        candidate_feat_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = state_context_dim + candidate_feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state_context: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.dim() == 1:
            candidate_features = candidate_features.unsqueeze(0)
        context = state_context.unsqueeze(0).expand(candidate_features.shape[0], -1)
        scorer_input = torch.cat([candidate_features, context], dim=-1)
        return self.net(scorer_input).squeeze(-1)


class TaskCreationActor(nn.Module):
    """
    Two-stage task creation actor:
    1) Bayesian factorizer proposes a shortlist.
    2) Deterministic scorer ranks selected candidates.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        queue_feat_dim: int = 16,
        task_feat_dim: int = 15,
        factor_embed_dim: int = 48,
        factor_hidden_dim: int = 128,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.queue_norm = nn.LayerNorm(queue_feat_dim)
        self.state_context_dim = hidden_dim * 2 + queue_feat_dim

        self.factorizer = BayesianCandidateFactorizer(
            state_context_dim=self.state_context_dim,
            candidate_feat_dim=task_feat_dim,
            embed_dim=factor_embed_dim,
            hidden_dim=factor_hidden_dim,
            prior_sigma=prior_sigma,
        )
        self.scorer = TaskCreationScorer(
            state_context_dim=self.state_context_dim,
            candidate_feat_dim=task_feat_dim,
            hidden_dim=factor_hidden_dim,
        )

    def build_state_context(
        self,
        graph_embedding: torch.Tensor,
        fleet_embedding: torch.Tensor,
        queue_features: torch.Tensor,
    ) -> torch.Tensor:
        queue_features = self.queue_norm(queue_features)
        return torch.cat([graph_embedding, fleet_embedding, queue_features], dim=-1)

    def score_candidates(
        self,
        task_features: torch.Tensor,
        graph_embedding: torch.Tensor,
        fleet_embedding: torch.Tensor,
        queue_features: torch.Tensor,
    ) -> torch.Tensor:
        state_context = self.build_state_context(graph_embedding, fleet_embedding, queue_features)
        return self.scorer(state_context, task_features)

    @staticmethod
    def _sample_topk(
        logits: torch.Tensor,
        k: int,
        temperature: float,
        deterministic: bool,
    ) -> torch.Tensor:
        if logits.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)

        k = min(k, logits.shape[0])
        if deterministic:
            return torch.topk(logits, k=k, dim=0).indices

        tau = max(float(temperature), 1e-6)
        scaled = logits / tau
        gumbel = -torch.log(-torch.log(torch.rand_like(scaled).clamp(min=1e-8, max=1.0 - 1e-8)))
        return torch.topk(scaled + gumbel, k=k, dim=0).indices

    def _select_shortlist(
        self,
        factor_logits: torch.Tensor,
        shortlist_size: int,
        urgent_mask: Optional[torch.Tensor],
        candidate_mask: Optional[torch.Tensor],
        temperature: float,
        deterministic: bool,
    ) -> torch.Tensor:
        num_candidates = factor_logits.shape[0]
        shortlist_size = max(1, min(int(shortlist_size), num_candidates))

        feasible_mask = torch.isfinite(factor_logits)
        if candidate_mask is not None:
            feasible_mask = feasible_mask & candidate_mask

        urgent = torch.zeros_like(feasible_mask)
        if urgent_mask is not None:
            urgent = urgent_mask & feasible_mask

        urgent_indices = torch.nonzero(urgent, as_tuple=False).squeeze(-1)
        if urgent_indices.numel() >= shortlist_size:
            urgent_logits = factor_logits[urgent_indices]
            keep = self._sample_topk(
                urgent_logits, k=shortlist_size, temperature=temperature, deterministic=deterministic
            )
            return urgent_indices[keep]

        selected = []
        if urgent_indices.numel() > 0:
            selected.append(urgent_indices)

        remaining_k = shortlist_size - urgent_indices.numel()
        candidate_pool_mask = feasible_mask & ~urgent
        pool_indices = torch.nonzero(candidate_pool_mask, as_tuple=False).squeeze(-1)

        if remaining_k > 0 and pool_indices.numel() > 0:
            pool_logits = factor_logits[pool_indices]
            chosen = self._sample_topk(
                pool_logits, k=remaining_k, temperature=temperature, deterministic=deterministic
            )
            selected.append(pool_indices[chosen])

        if selected:
            return torch.cat(selected, dim=0)

        fallback = torch.nonzero(feasible_mask, as_tuple=False).squeeze(-1)
        if fallback.numel() == 0:
            return torch.arange(shortlist_size, device=factor_logits.device)
        if fallback.numel() <= shortlist_size:
            return fallback
        keep = self._sample_topk(
            factor_logits[fallback], k=shortlist_size, temperature=temperature, deterministic=deterministic
        )
        return fallback[keep]

    def rank_candidates(
        self,
        task_features: torch.Tensor,
        graph_embedding: torch.Tensor,
        fleet_embedding: torch.Tensor,
        queue_features: torch.Tensor,
        shortlist_size: int,
        temperature: float = 1.0,
        urgent_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> TaskCreationRanking:
        state_context = self.build_state_context(graph_embedding, fleet_embedding, queue_features)

        factor_out = self.factorizer(
            state_context=state_context,
            candidate_features=task_features,
            candidate_mask=candidate_mask,
            deterministic=deterministic,
        )
        factor_logits = factor_out["logits"]

        shortlist_indices = self._select_shortlist(
            factor_logits=factor_logits,
            shortlist_size=shortlist_size,
            urgent_mask=urgent_mask,
            candidate_mask=candidate_mask,
            temperature=temperature,
            deterministic=deterministic,
        )

        scorer_scores = self.scorer(state_context, task_features)
        if candidate_mask is not None:
            scorer_scores = scorer_scores.masked_fill(~candidate_mask, float("-inf"))

        shortlist_scores = scorer_scores[shortlist_indices]
        shortlist_order = torch.argsort(shortlist_scores, descending=True)
        ranked_shortlist = shortlist_indices[shortlist_order]

        all_indices = torch.arange(task_features.shape[0], device=task_features.device)
        remaining_mask = torch.ones_like(all_indices, dtype=torch.bool)
        remaining_mask[ranked_shortlist] = False
        if candidate_mask is not None:
            remaining_mask = remaining_mask & candidate_mask

        remaining_indices = all_indices[remaining_mask]
        if remaining_indices.numel() > 0:
            remaining_scores = scorer_scores[remaining_indices]
            remaining_order = torch.argsort(remaining_scores, descending=True)
            ranked_remaining = remaining_indices[remaining_order]
            ranked_indices = torch.cat([ranked_shortlist, ranked_remaining], dim=0)
        else:
            ranked_indices = ranked_shortlist

        if candidate_mask is not None:
            infeasible = all_indices[~candidate_mask]
            if infeasible.numel() > 0:
                ranked_indices = torch.cat([ranked_indices, infeasible], dim=0)

        return TaskCreationRanking(
            ranked_indices=ranked_indices,
            shortlist_indices=shortlist_indices,
            scorer_scores=scorer_scores,
            factor_logits=factor_logits,
            factor_kl=factor_out["kl"],
        )
