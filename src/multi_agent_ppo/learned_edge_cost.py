import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Feature indices in EdgeTraversalRecord.to_feature_vector()
# [0] distance_m  — NOT fed to encoder; extracted in predict_cost() to convert seconds
# [1] corridor_width         ┐
# [2] num_robots_on_edge     │
# [3] people_count           │  These 8 congestion features are the encoder input
# [4] clutter_level          │  (indices 1–8 of the 9-element vector)
# [5] approaching_robot_count│
# [6] from_node_occupancy    │
# [7] to_node_occupancy      │
# [8] time_of_day            ┘
#
# Removed from original: same_direction_count, opposite_direction_count, stop_count.
# Direction is unknown at inference time; stop_count only exists post-traversal.

EDGE_COST_FEATURE_DIM = 9   # Full feature vector length (including distance_m at [0])
EDGE_COST_ENCODER_DIM = 8   # Congestion features fed to encoder (distance_m excluded)


class LearnedEdgeCostModel(nn.Module):
    """
    Predicts edge traversal time distribution given current conditions.

    Outputs both mean and log-variance so path planning can account for
    risk (high variance = unreliable edge).

    Architecture:
        Input features → MLP → (mean_time, log_variance)

    The model learns a delay factor relative to base traversal time:
        predicted_time = base_time * delay_factor
    where delay_factor >= 1.0 (can only be slower than free-flow).
    """

    def __init__(self, input_dim: int = EDGE_COST_FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        encoder_input_dim = input_dim - 1  # distance_m (index 0) excluded from encoder

        # Normalize congestion features before encoding (indices 1–8 span very different scales)
        self.input_norm = nn.LayerNorm(encoder_input_dim)

        # Shared feature encoder (congestion features only)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean head: predicts delay factor (via Softplus to ensure > 0)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Variance head: predicts log-variance of traversal time
        self.log_var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Init mean head bias to -3.0: softplus(-3) ≈ 0.049 → delay_factor starts near 1.05
        # (avoids the softplus(0) = 0.693 default which would start delay_factor at ~1.69)
        nn.init.constant_(self.mean_head[-1].bias, -3.0)
        # Initialize log_var head bias to small value (low initial uncertainty)
        nn.init.constant_(self.log_var_head[-1].bias, -2.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict delay factor distribution.

        Args:
            features: [batch, input_dim] edge features (distance_m at [0], congestion at [1:])

        Returns:
            delay_factor: [batch] predicted mean delay factor (>= 1.0, dimensionless)
            log_var: [batch] log-variance of delay factor, clamped to [-8, 4]

        The delay factor is scale-invariant: delay_factor = actual_time / base_time.
        To get traversal time in seconds: time = delay_factor * (distance_m / max_v_ms).
        Training on delay_factor rather than absolute seconds prevents long corridors
        from dominating the gradient signal.

        distance_m (index 0) is NOT passed to the encoder — it is only used in
        predict_cost() to convert the dimensionless delay_factor back to seconds.
        """
        # Congestion features only (exclude distance_m at index 0)
        congestion_features = features[:, 1:]  # [batch, input_dim-1]
        h = self.encoder(self.input_norm(congestion_features))

        # Predict delay factor >= 1.0 (corridor can only be slower than free-flow)
        raw_delay = self.mean_head(h).squeeze(-1)
        delay_factor = 1.0 + F.softplus(raw_delay)

        # Predicted log-variance of delay factor (clamped for numerical stability)
        log_var = self.log_var_head(h).squeeze(-1)
        log_var = torch.clamp(log_var, min=-8.0, max=4.0)

        return delay_factor, log_var

    def predict_cost(
        self,
        features: torch.Tensor,
        risk_sensitivity: float = 0.0
    ) -> torch.Tensor:
        """
        Predict path cost in seconds for Dijkstra, optionally risk-adjusted.

        Args:
            features: [batch, input_dim] or [num_edges, input_dim]
            risk_sensitivity: 0.0 = use mean only (risk-neutral)
                             >0.0 = penalize high-variance edges (risk-averse)
                             Typical: 0.5 - 1.0

        Returns:
            costs: [batch] or [num_edges] predicted traversal time in seconds
        """
        delay_factor, log_var = self.forward(features)

        # Convert delay factor back to seconds: time = delay_factor * (distance / max_v).
        # Current simulator/config convention uses 0.5 m/s uniformly across edges.
        distance = features[:, 0]
        base_time = distance / 0.5

        if risk_sensitivity > 0.0:
            # Risk-adjusted: penalise edges with high variance in delay
            std_dev = torch.exp(0.5 * log_var)
            return (delay_factor + risk_sensitivity * std_dev) * base_time
        else:
            return delay_factor * base_time

    def predict_numpy(
        self,
        features_np: np.ndarray,
        risk_sensitivity: float = 0.0
    ) -> np.ndarray:
        """
        Convenience method for use in Dijkstra (numpy in, numpy out).

        Args:
            features_np: [num_edges, input_dim] numpy array
            risk_sensitivity: risk aversion parameter

        Returns:
            costs: [num_edges] numpy array of predicted costs
        """
        with torch.no_grad():
            features = torch.tensor(features_np, dtype=torch.float32)
            costs = self.predict_cost(features, risk_sensitivity)
            return costs.numpy()


class HeuristicEdgeCostModel:
    """
    Fallback heuristic model used before enough data is collected.
    Mimics the LearnedEdgeCostModel interface but uses simple rules.
    """

    def predict_numpy(
        self,
        features_np: np.ndarray,
        risk_sensitivity: float = 0.0
    ) -> np.ndarray:
        """
        Heuristic cost prediction.

        Feature layout (matches EdgeTraversalRecord.to_feature_vector / build_edge_features):
            [0] distance_m
            [1] corridor_width
            [2] num_robots_on_edge
            [3] people_count
            [4] clutter_level
            [5] approaching_robot_count
            [6] from_node_occupancy
            [7] to_node_occupancy
            [8] time_of_day

        Args:
            features_np: [num_edges, 9] feature array
            risk_sensitivity: ignored in heuristic mode

        Returns:
            costs: [num_edges] predicted costs
        """
        distance = features_np[:, 0]
        num_robots = features_np[:, 2]
        people_count = features_np[:, 3]
        clutter = features_np[:, 4]
        approaching = features_np[:, 5]

        # Match the simulator/config baseline corridor speed used elsewhere.
        base_time = distance / 0.5

        # Simple additive penalties
        congestion = (num_robots + approaching) * 1.5
        people_penalty = people_count * 0.5
        clutter_penalty = clutter * 10.0

        return base_time + congestion + people_penalty + clutter_penalty


class EdgeCostTrainer:
    """
    Supervised trainer for LearnedEdgeCostModel.

    Collects EdgeTraversalRecords from simulation, maintains a replay buffer,
    and periodically trains the model on completed traversal data.

    Training loss: Gaussian negative log-likelihood
        NLL = 0.5 * (log_var + (actual - predicted_mean)^2 / exp(log_var))

    This jointly learns mean AND variance from data.
    """

    def __init__(
        self,
        model: LearnedEdgeCostModel,
        lr: float = 1e-3,
        max_buffer_size: int = 50000,
        min_train_samples: int = 100,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.max_buffer_size = max_buffer_size
        self.min_train_samples = min_train_samples

        # Replay buffer: list of (features_np, actual_time) tuples
        self.buffer: List[Tuple[np.ndarray, float]] = []

        # Training stats
        self.total_records_added = 0
        self.total_train_steps = 0
        self.recent_losses: List[float] = []

    def add_records_batch(self, records: list) -> int:
        """Add completed traversal records to the replay buffer. Returns count added."""
        count = 0
        for record in records:
            if record.actual_traversal_time <= 0 or record.base_traversal_time <= 0:
                continue
            self.buffer.append((record.to_feature_vector(), record.delay_factor))
            self.total_records_added += 1
            count += 1
            if len(self.buffer) > self.max_buffer_size:
                self.buffer.pop(0)
        return count

    def train_epoch(self, batch_size: int = 64, num_steps: int = 10) -> Optional[float]:
        """Train multiple steps. Returns average loss or None if not enough data."""
        if len(self.buffer) < self.min_train_samples:
            return None

        losses = []
        for _ in range(num_steps):
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
            features = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
            targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)

            delay_factor, log_var = self.model(features)
            variance = torch.exp(log_var)
            loss = (0.5 * (log_var + (targets - delay_factor) ** 2 / variance)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.total_train_steps += 1
            loss_val = loss.item()
            self.recent_losses.append(loss_val)
            if len(self.recent_losses) > 100:
                self.recent_losses.pop(0)
            losses.append(loss_val)

        return float(np.mean(losses))

    @property
    def is_ready(self) -> bool:
        """Whether enough data has been collected to use the learned model."""
        return len(self.buffer) >= self.min_train_samples

    @property
    def avg_recent_loss(self) -> float:
        """Average loss over recent training steps."""
        if not self.recent_losses:
            return float('inf')
        return np.mean(self.recent_losses)

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'buffer_size': len(self.buffer),
            'total_records': self.total_records_added,
            'total_train_steps': self.total_train_steps,
            'avg_recent_loss': self.avg_recent_loss,
            'is_ready': self.is_ready,
        }


class EdgeCostManager:
    """
    Manages the transition from heuristic to learned edge cost predictions.

    Starts with heuristic fallback, collects traversal data, and switches
    to the learned model once enough data is available.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        risk_sensitivity: float = 0.5,
        min_train_samples: int = 100,
        train_interval_records: int = 50,
        train_steps_per_interval: int = 10,
    ):
        self.model = LearnedEdgeCostModel(hidden_dim=hidden_dim)
        self.heuristic = HeuristicEdgeCostModel()
        self.trainer = EdgeCostTrainer(
            self.model, lr=lr, min_train_samples=min_train_samples
        )

        self.risk_sensitivity = risk_sensitivity
        self.train_interval_records = train_interval_records
        self.train_steps_per_interval = train_steps_per_interval
        self._records_since_last_train = 0
        self.using_learned_model = False

    def predict_edge_costs(self, features_np: np.ndarray) -> np.ndarray:
        """
        Predict edge costs for path planning.

        Automatically switches between heuristic and learned model.

        Args:
            features_np: [num_edges, 12] feature array

        Returns:
            costs: [num_edges] predicted costs
        """
        if self.using_learned_model and self.trainer.is_ready:
            return self.model.predict_numpy(features_np, self.risk_sensitivity)
        else:
            return self.heuristic.predict_numpy(features_np, self.risk_sensitivity)

    def add_traversal_records(self, records: list) -> None:
        """
        Add completed traversal records and optionally trigger training.
        """
        count = self.trainer.add_records_batch(records)
        self._records_since_last_train += count

        # Train periodically
        if (self._records_since_last_train >= self.train_interval_records
                and self.trainer.is_ready):
            self.trainer.train_epoch(
                num_steps=self.train_steps_per_interval
            )
            self._records_since_last_train = 0
            self.using_learned_model = True

    def get_stats(self) -> dict:
        """Get status information."""
        stats = self.trainer.get_stats()
        stats['using_learned_model'] = self.using_learned_model
        stats['risk_sensitivity'] = self.risk_sensitivity
        return stats

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing learned edge costs."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "trainer_buffer": self.trainer.buffer,
            "trainer_total_records_added": self.trainer.total_records_added,
            "trainer_total_train_steps": self.trainer.total_train_steps,
            "trainer_recent_losses": self.trainer.recent_losses,
            "records_since_last_train": self._records_since_last_train,
            "using_learned_model": self.using_learned_model,
            "risk_sensitivity": self.risk_sensitivity,
            "train_interval_records": self.train_interval_records,
            "train_steps_per_interval": self.train_steps_per_interval,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore learned edge cost checkpoint state."""
        if not state:
            return
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.trainer.buffer = state.get("trainer_buffer", [])
        self.trainer.total_records_added = state.get("trainer_total_records_added", 0)
        self.trainer.total_train_steps = state.get("trainer_total_train_steps", 0)
        self.trainer.recent_losses = state.get("trainer_recent_losses", [])
        self._records_since_last_train = state.get("records_since_last_train", 0)
        self.using_learned_model = state.get("using_learned_model", False)
        self.risk_sensitivity = state.get("risk_sensitivity", self.risk_sensitivity)
        self.train_interval_records = state.get("train_interval_records", self.train_interval_records)
        self.train_steps_per_interval = state.get("train_steps_per_interval", self.train_steps_per_interval)

    def build_edge_features(self, graph_state, all_robots=None) -> np.ndarray:
        """
        Build feature matrix for all edges in the graph.

        Args:
            graph_state: GraphState with edges
            all_robots: Optional list of all robots (for approaching count)

        Returns:
            features: [num_edges, 12] numpy array
        """
        features = []
        for edge_idx, edge in enumerate(graph_state.edges):
            # Count approaching robots (have this edge in their planned path but not on it)
            approaching = 0
            if all_robots:
                for robot in all_robots:
                    if hasattr(robot, 'planned_path') and robot.planned_path:
                        path = robot.planned_path
                        for i in range(len(path) - 1):
                            from_node = graph_state.nodes[path[i]].node_id
                            to_node = graph_state.nodes[path[i + 1]].node_id
                            if ((edge.from_node == from_node and edge.to_node == to_node) or
                                    (edge.from_node == to_node and edge.to_node == from_node)):
                                approaching += 1
                                break

            # Node occupancy at endpoints
            from_node_occ = 0
            to_node_occ = 0
            for node in graph_state.nodes:
                if node.node_id == edge.from_node:
                    from_node_occ = len(getattr(node, 'current_robot_ids', []))
                elif node.node_id == edge.to_node:
                    to_node_occ = len(getattr(node, 'current_robot_ids', []))

            current_time = getattr(graph_state, 'current_time', 0.0)
            time_of_day = (current_time % 86400.0) / 86400.0

            # 9 features matching EdgeTraversalRecord.to_feature_vector()
            # same_direction_count, opposite_direction_count, stop_count excluded:
            # direction is unknown at prediction time; stop_count is post-traversal only.
            features.append([
                edge.distance_m,
                edge.corridor_width,
                float(len(edge.active_robot_ids)),
                float(edge.people_count),
                edge.clutter_level,
                float(approaching),
                float(from_node_occ),
                float(to_node_occ),
                time_of_day,
            ])

        return np.array(features, dtype=np.float32)
