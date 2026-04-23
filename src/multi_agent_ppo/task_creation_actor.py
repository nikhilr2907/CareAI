
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from ..environment.graph_helpers import dijkstra_shortest_path


# Candidate feature dimension: matches Task.get_features() output (10 dims)
CANDIDATE_FEAT_DIM = 10

# Lightweight state summary dimension fed to both networks
STATE_SUMMARY_DIM = 8


@dataclass
class CandidateSpec:
    """Describes one SKU-at-node candidate without creating a Task object."""
    from_node_idx: int
    to_node_idx: int
    sku_id: Optional[str]      # None for node-level (no per-SKU inventory)
    category_id: float
    stock: float
    reorder: float
    max_stock: float
    rate: float                # Consumption rate (items / hour)
    time_to_stockout: float    # Hours until stockout at current rate
    features: np.ndarray       # [CANDIDATE_FEAT_DIM] matches Task.get_features()


@dataclass
class CreationRecord:
    """
    Snapshot stored at task-creation time, used later to compute training loss.

    state_summary and candidate_features are stored as numpy arrays.
    Fresh forward passes are run at loss-computation time to get gradients.
    """
    task_id: int
    state_summary: np.ndarray       # [STATE_SUMMARY_DIM]
    candidate_features: np.ndarray  # [num_candidates, CANDIDATE_FEAT_DIM] subsampled pool
    shortlist_indices: List[int]    # [k] indices into candidate_features used at creation
    chosen_in_shortlist: int        # index within shortlist that scorer selected


class BayesianCandidateFactorizer(nn.Module):
    """
    Thompson sampling over SKU candidate pool (#7).

    A posterior network maps the state summary s to (mu(s), log_sigma(s)) in the
    same space as the candidate feature vectors.  At each call:

        w ~ N(mu(s), sigma(s))          (Thompson sample; mu during eval)
        logit_i = phi_i @ w             (dot-product score per candidate)
        shortlist = Gumbel top-k(logits / temperature + gumbel_noise)

    KL(q(w|s) || N(0, prior_sigma^2 I)) is returned as a regularization term
    to be added to the training loss when backpropagating.
    """

    def __init__(
        self,
        state_dim: int = STATE_SUMMARY_DIM,
        candidate_feat_dim: int = CANDIDATE_FEAT_DIM,
        hidden_dim: int = 64,
        shortlist_k: int = 5,
        prior_sigma: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.candidate_feat_dim = candidate_feat_dim
        self.shortlist_k = shortlist_k
        self.prior_sigma = prior_sigma
        self.temperature = temperature

        # State -> posterior (mu, log_sigma) in candidate_feat_dim space
        self.posterior_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, candidate_feat_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, candidate_feat_dim)

    def forward(
        self,
        state_summary: torch.Tensor,       # [state_dim]
        candidate_features: torch.Tensor,  # [num_candidates, candidate_feat_dim]
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            shortlist_indices: [k] long tensor - indices into candidate_features
            logits: [num_candidates] - raw candidate scores before Gumbel noise
            kl_loss: scalar tensor - KL divergence term for regularization
        """
        h = self.posterior_net(state_summary)
        mu = self.mu_head(h)                              # [candidate_feat_dim]
        log_sigma = self.log_sigma_head(h).clamp(-4, 2)  # [candidate_feat_dim]
        sigma = torch.exp(log_sigma)

        # Thompson sampling: sample or use posterior mean
        if training:
            w = mu + sigma * torch.randn_like(mu)
        else:
            w = mu

        # KL(N(mu, sigma^2) || N(0, prior_sigma^2)) summed over dimensions
        prior_var = self.prior_sigma ** 2
        kl_loss = 0.5 * torch.sum(
            (mu ** 2 + sigma ** 2) / prior_var
            - 1.0
            - 2.0 * log_sigma
            + float(np.log(prior_var))
        )

        # Score all candidates: logit_i = phi_i @ w
        logits = candidate_features @ w  # [num_candidates]

        # Gumbel top-k for stochastic shortlisting
        k = min(self.shortlist_k, candidate_features.shape[0])
        if training:
            gumbel = -torch.log(
                -torch.log(torch.clamp(torch.rand_like(logits), min=1e-10))
            )
            perturbed = logits / self.temperature + gumbel
        else:
            perturbed = logits

        _, shortlist_indices = torch.topk(perturbed, k)
        return shortlist_indices, logits, kl_loss


class TaskCreationScorer(nn.Module):
    """
    Deterministic scorer over factorizer shortlist (#8).

    Concatenates state summary with each shortlisted candidate feature vector,
    runs an MLP to produce per-candidate scores, and selects the argmax.
    Exactly 1 candidate is picked per call and instantiated as a Task.
    """

    def __init__(
        self,
        state_dim: int = STATE_SUMMARY_DIM,
        candidate_feat_dim: int = CANDIDATE_FEAT_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + candidate_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state_summary: torch.Tensor,       # [state_dim]
        shortlist_features: torch.Tensor,  # [k, candidate_feat_dim]
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            scores: [k] per-candidate scalar scores
            selected_idx: int - argmax index into shortlist
        """
        k = shortlist_features.shape[0]
        state_exp = state_summary.unsqueeze(0).expand(k, -1)   # [k, state_dim]
        combined = torch.cat([state_exp, shortlist_features], dim=-1)
        scores = self.scorer(combined).squeeze(-1)              # [k]
        selected_idx = int(scores.argmax().item())
        return scores, selected_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_source_node(graph_state) -> Optional[int]:
    """Return index of the first storage node, or None if none exist."""
    for idx, node in enumerate(graph_state.nodes):
        if node.node_type == 'storage':
            return idx
    return None


def build_candidate_universe(
    graph_state,
    pending_tasks: list,
    current_time: float,
    source_node_idx: int,
    max_subsample: int = 30,
) -> List[CandidateSpec]:
    """
    Build the eligible SKU candidate universe from graph_state inventory.

    Eligibility (all conditions must hold):
        stock > reorder_point - not urgently low (urgent = deterministic replenishment)
        stock < max - has room to receive delivery
        rate > 0 - node actually consumes this SKU
        not already covered - no pending task already targets (to_node, sku_id)

    Supports both per-SKU inventory (node.sku_inventory dict) and node-level
    fallback (node.stock_level / node.max_stock).

    Returns up to max_subsample candidates (uniform random subsampled if larger).
    """
    covered = {(t.to_location_index, t.sku_id) for t in pending_tasks}

    candidates: List[CandidateSpec] = []
    for node_idx, node in enumerate(graph_state.nodes):
        if node_idx == source_node_idx:
            continue
        if not getattr(node, 'consumption_enabled', True):
            continue

        if node.sku_inventory:
            for sku_id, data in node.sku_inventory.items():
                stock = float(data.get('stock', 0.0))
                reorder = float(data.get('reorder', 0.0))
                max_stock = float(data.get('max', 0.0))
                rate = float(data.get('rate', 0.0))
                category = data.get('category', '')

                if stock <= reorder or rate <= 0 or max_stock <= 0 or stock >= max_stock:
                    continue
                if (node_idx, sku_id) in covered:
                    continue

                tts = stock / rate  # hours
                room = max_stock - stock
                num_items = max(1, int(room))
                cat_id = float(abs(hash(category)) % 100) if category else -1.0
                stock_ratio = stock / max_stock
                reorder_ratio = reorder / max_stock if max_stock > 0 else 0.0

                _, path_cost = dijkstra_shortest_path(
                    source_node_idx, node_idx, graph_state, len(graph_state.nodes)
                )

                feat = np.array([
                    float(node_idx),
                    path_cost,       # estimated_duration: dijkstra(hub → destination)
                    0.0,             # age (not in queue yet)
                    tts * 3600,      # time_to_deadline (proxy from tts)
                    0.0,             # queue_position (not in queue yet)
                    float(num_items),
                    tts,             # time_to_stockout (hours)
                    cat_id, stock_ratio, reorder_ratio,
                ], dtype=np.float32)

                candidates.append(CandidateSpec(
                    from_node_idx=source_node_idx,
                    to_node_idx=node_idx,
                    sku_id=sku_id,
                    category_id=cat_id,
                    stock=stock,
                    reorder=reorder,
                    max_stock=max_stock,
                    rate=rate,
                    time_to_stockout=tts,
                    features=feat,
                ))


    if not candidates:
        return []

    if len(candidates) > max_subsample:
        indices = np.random.choice(len(candidates), max_subsample, replace=False)
        candidates = [candidates[i] for i in indices]

    return candidates


def build_state_summary(
    pending_tasks: list,
    robots: list,
    graph_state,
    current_time: float,
    max_queue: int = 20,
) -> np.ndarray:
    """
    Build lightweight 8-dim state summary for factorizer and scorer input.

    Dims:
        0: queue fill ratio (num_pending / max_queue)
        1: fraction of robots with empty task queues (idle)
        2: fraction of current simulated hour elapsed
        3: mean pending task age, normalised by 5 minutes
        4: fraction of pending tasks with time_to_stockout < 1.0 hour (urgent)
        5: mean (stock_level / max_stock) across all nodes
        6: fraction of robots currently moving (from telemetry)
        7: queue fill ratio (duplicate of dim 0 for feature symmetry)
    """
    num_pending = len(pending_tasks)
    num_robots = max(len(robots), 1)

    queue_fill = num_pending / max(1, max_queue)

    robot_idle = sum(1 for r in robots if len(r.task_queue) == 0) / num_robots

    hour_frac = (current_time % 3600.0) / 3600.0

    if pending_tasks:
        avg_age = float(np.mean([current_time - t.arrival_time for t in pending_tasks]))
        avg_age_norm = avg_age / 300.0  # normalise by 5 minutes
        frac_urgent = sum(
            1 for t in pending_tasks
            if t.get_current_time_to_stockout() < 1.0
        ) / num_pending
    else:
        avg_age_norm = 0.0
        frac_urgent = 0.0

    stock_ratios = [
        node.stock_level / node.max_stock
        for node in graph_state.nodes
        if node.max_stock > 0
    ]
    mean_stock = float(np.mean(stock_ratios)) if stock_ratios else 1.0

    frac_moving = sum(
        1 for r in robots
        if r.telemetry is not None and getattr(r.telemetry, 'is_moving', False)
    ) / num_robots

    return np.array(
        [queue_fill, robot_idle, hour_frac, avg_age_norm,
         frac_urgent, mean_stock, frac_moving, queue_fill],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

class TaskCreationActor(nn.Module):
    """
    Container for BayesianCandidateFactorizer (#7) and TaskCreationScorer (#8).

    Called from GAPOTaskAssignmentEnv._generate_stochastic_tasks() to proactively
    create SKU replenishment tasks driven by inventory state, replacing blind
    random ad-hoc task generation.

    Full pipeline per firing:
        1. build_candidate_universe()   → eligible pool (filtered, subsampled ≤30)
        2. BayesianCandidateFactorizer  → Thompson-sampled shortlist (k=5)
        3. TaskCreationScorer           → pick 1 candidate
        4. Instantiate Task with live quantity safeguard
    """

    def __init__(
        self,
        state_dim: int = STATE_SUMMARY_DIM,
        candidate_feat_dim: int = CANDIDATE_FEAT_DIM,
        hidden_dim: int = 64,
        shortlist_k: int = 5,
        prior_sigma: float = 1.0,
        temperature: float = 1.0,
        candidate_subsample_size: int = 30,
        device: str = 'cpu',
    ):
        super().__init__()
        self.candidate_subsample_size = candidate_subsample_size
        self.device = device

        self.factorizer = BayesianCandidateFactorizer(
            state_dim=state_dim,
            candidate_feat_dim=candidate_feat_dim,
            hidden_dim=hidden_dim,
            shortlist_k=shortlist_k,
            prior_sigma=prior_sigma,
            temperature=temperature,
        )
        self.scorer = TaskCreationScorer(
            state_dim=state_dim,
            candidate_feat_dim=candidate_feat_dim,
            hidden_dim=hidden_dim,
        )

        # Training buffers: keyed by task_id
        # _pending_records holds creations waiting for completion outcome.
        # _completed_records accumulates (record, reward) pairs ready for the next loss step.
        self._pending_records: Dict[int, CreationRecord] = {}
        self._completed_records: List[Tuple[CreationRecord, float]] = []
        # Exponential moving average baseline for REINFORCE variance reduction
        self._reward_baseline: float = 0.0
        self._baseline_alpha: float = 0.05

    def create_task(
        self,
        graph_state,
        pending_tasks: list,
        robots: list,
        current_time: float,
        next_task_id: int,
        training: bool = False,
    ):
        """
        Run the full two-stage pipeline and return one new Task (or None).

        Returns:
            task: Task object, or None if no eligible candidates or quantity guard fails
            new_next_task_id: int
        """
        from ..environment.tasks.task_state import Task

        source_idx = _find_source_node(graph_state)
        if source_idx is None:
            return None, next_task_id

        candidates = build_candidate_universe(
            graph_state=graph_state,
            pending_tasks=pending_tasks,
            current_time=current_time,
            source_node_idx=source_idx,
            max_subsample=self.candidate_subsample_size,
        )
        if not candidates:
            return None, next_task_id

        state_np = build_state_summary(pending_tasks, robots, graph_state, current_time)
        state_t = torch.from_numpy(state_np).float().to(self.device)

        feat_np = np.stack([c.features for c in candidates], axis=0)  # [N, 15]
        feat_t = torch.from_numpy(feat_np).float().to(self.device)

        with torch.set_grad_enabled(training):
            shortlist_idx, _, _kl = self.factorizer(state_t, feat_t, training=training)
            shortlist_specs = [candidates[i] for i in shortlist_idx.tolist()]
            shortlist_feat = feat_t[shortlist_idx]
            _, chosen = self.scorer(state_t, shortlist_feat)

        selected = shortlist_specs[chosen]

        # Quantity safeguard: re-fetch live inventory immediately before creating Task
        node = graph_state.nodes[selected.to_node_idx]
        if (
            selected.sku_id
            and node.sku_inventory
            and selected.sku_id in node.sku_inventory
        ):
            data = node.sku_inventory[selected.sku_id]
            live_stock = float(data.get('stock', 0.0))
            max_stock = float(data.get('max', 0.0))
            reorder = float(data.get('reorder', 0.0))
        else:
            live_stock = node.stock_level
            max_stock = node.max_stock
            reorder = selected.reorder

        room = max_stock - live_stock
        if room <= 0:
            return None, next_task_id

        num_items = max(1, int(room))
        tts_seconds = selected.time_to_stockout * 3600
        deadline = current_time + max(tts_seconds, 300.0)  # At least 5 minutes

        task = Task(
            task_id=next_task_id,
            from_location_index=selected.from_node_idx,
            to_location_index=selected.to_node_idx,
            deadline=deadline,
            arrival_time=current_time,
            estimated_duration=selected.features[1],  # dijkstra(hub → destination) from candidate
            task_type='replenishment',
            num_items=num_items,
            sku_id=selected.sku_id,
            category_id=selected.category_id,
            source_stock_level=live_stock,
            time_to_stockout=selected.time_to_stockout,
            sku_stock_level=live_stock,
            sku_max_level=max_stock,
            reorder_point=reorder,
        )

        # Store creation context so the completion reward can be routed back for training
        self._pending_records[next_task_id] = CreationRecord(
            task_id=next_task_id,
            state_summary=state_np.copy(),
            candidate_features=feat_np.copy(),
            shortlist_indices=shortlist_idx.tolist(),
            chosen_in_shortlist=chosen,
        )

        return task, next_task_id + 1

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def record_completion(self, task_id: int, reward: float) -> None:
        """
        Route a completion reward back to the creation record for task_id.

        Called from the training loop when a task completes. If task_id is not in
        the pending buffer (e.g. created by the random fallback, or from a previous
        run), the call is silently ignored.
        """
        record = self._pending_records.pop(task_id, None)
        if record is not None:
            self._completed_records.append((record, reward))

    def compute_creation_loss(
        self,
        scorer_loss_coef: float = 1.0,
        factorizer_loss_coef: float = 0.1,
        kl_coef: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute REINFORCE training loss from completed creation records.

        For each completed record:
          - Re-runs factorizer forward (with gradients via reparameterisation) to get
            fresh logits and KL term.
          - Factorizer loss: -log_softmax(logits)[stored_shortlist_indices].mean() * (R - b)
          - Scorer loss:     -log_softmax(scores)[stored_chosen_idx]           * (R - b)
          - KL term:          KL(q(w|s) || N(0, I)) as regularisation
          where R is the completion reward and b is an EMA baseline.

        Clears the completed buffer after computing. Pending records are preserved.

        Returns:
            Scalar loss tensor (0.0 if no completed records).
        """
        if not self._completed_records:
            return torch.tensor(0.0, device=self.device)

        rewards = [r for _, r in self._completed_records]
        mean_r = float(np.mean(rewards))
        self._reward_baseline = (
            (1.0 - self._baseline_alpha) * self._reward_baseline
            + self._baseline_alpha * mean_r
        )

        scorer_losses: List[torch.Tensor] = []
        factorizer_losses: List[torch.Tensor] = []
        kl_losses: List[torch.Tensor] = []

        for record, reward in self._completed_records:
            R = reward - self._reward_baseline  # baseline-subtracted return

            state_t = torch.from_numpy(record.state_summary).float().to(self.device)
            feat_t = torch.from_numpy(record.candidate_features).float().to(self.device)

            # Re-run factorizer with gradients (reparameterisation through w)
            _shortlist_t, logits, kl_loss = self.factorizer(state_t, feat_t, training=True)

            # Factorizer REINFORCE: encourage high logits for the stored shortlist
            stored_idx = torch.tensor(
                record.shortlist_indices, dtype=torch.long, device=self.device
            )
            log_probs_all = F.log_softmax(logits, dim=0)        # [num_candidates]
            factorizer_log_prob = log_probs_all[stored_idx].mean()
            factorizer_losses.append(-factorizer_log_prob * R)
            kl_losses.append(kl_loss)

            # Re-run scorer on the stored shortlist features
            shortlist_feat = feat_t[stored_idx]                 # [k, feat_dim]
            scores, _ = self.scorer(state_t, shortlist_feat)
            log_probs_scorer = F.log_softmax(scores, dim=0)     # [k]
            chosen = min(record.chosen_in_shortlist, scores.shape[0] - 1)
            scorer_losses.append(-log_probs_scorer[chosen] * R)

        total = torch.tensor(0.0, device=self.device)
        if scorer_losses:
            total = total + scorer_loss_coef * torch.stack(scorer_losses).mean()
        if factorizer_losses:
            total = total + factorizer_loss_coef * torch.stack(factorizer_losses).mean()
        if kl_losses:
            total = total + kl_coef * torch.stack(kl_losses).mean()

        self._completed_records.clear()
        return total

    def clear_creation_buffer(self) -> None:
        """Clear both pending and completed buffers (e.g. at episode reset)."""
        self._pending_records.clear()
        self._completed_records.clear()
