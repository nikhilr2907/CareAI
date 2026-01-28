"""
Learned Edge Cost Model for graph-based path planning.

Predicts edge traversal time (mean + variance) from real-time congestion features.
Trained supervised from completed edge traversal records, NOT from RL rewards.

The model replaces the hardcoded current_weight heuristic in HospitalEdge
with data-driven predictions that capture complex interactions between
congestion factors.

Training signal: Gaussian NLL on (predicted_mean, predicted_var) vs actual_time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Feature indices in EdgeTraversalRecord.to_feature_vector()
# [0] distance_m
# [1] corridor_width
# [2] num_robots_on_edge
# [3] same_direction_count
# [4] opposite_direction_count
# [5] people_count
# [6] clutter_level
# [7] approaching_robot_count
# [8] from_node_occupancy
# [9] to_node_occupancy
# [10] time_of_day
# [11] stop_count

EDGE_COST_FEATURE_DIM = 12


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

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

        # Initialize mean head bias to 0 (delay_factor starts at ~1.0 after softplus)
        nn.init.zeros_(self.mean_head[-1].bias)
        # Initialize log_var head bias to small value (low initial uncertainty)
        nn.init.constant_(self.log_var_head[-1].bias, -2.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict traversal time distribution.

        Args:
            features: [batch, input_dim] edge congestion features

        Returns:
            mean_time: [batch] predicted mean traversal time in seconds
            log_var: [batch] log-variance of traversal time
        """
        # Extract distance (first feature) for base time calculation
        distance = features[:, 0]  # distance_m

        # Use max_v_ms=1.0 as default (can't extract from features since it's not included)
        # Base traversal time = distance / max_speed
        base_time = distance  # distance_m / 1.0 m/s

        # Encode features
        h = self.encoder(features)

        # Predict delay factor >= 1.0
        raw_delay = self.mean_head(h).squeeze(-1)
        delay_factor = 1.0 + F.softplus(raw_delay)  # Minimum delay = 1.0x

        # Predicted mean traversal time
        mean_time = base_time * delay_factor

        # Predicted log-variance
        log_var = self.log_var_head(h).squeeze(-1)

        return mean_time, log_var

    def predict_cost(
        self,
        features: torch.Tensor,
        risk_sensitivity: float = 0.0
    ) -> torch.Tensor:
        """
        Predict path cost for Dijkstra, optionally risk-adjusted.

        Args:
            features: [batch, input_dim] or [num_edges, input_dim]
            risk_sensitivity: 0.0 = use mean only (risk-neutral)
                             >0.0 = penalize high-variance edges (risk-averse)
                             Typical: 0.5 - 1.0

        Returns:
            costs: [batch] or [num_edges] predicted traversal cost
        """
        mean_time, log_var = self.forward(features)

        if risk_sensitivity > 0.0:
            # Risk-adjusted cost: mean + λ * std_dev
            std_dev = torch.exp(0.5 * log_var)
            return mean_time + risk_sensitivity * std_dev
        else:
            return mean_time

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

        Args:
            features_np: [num_edges, 12] feature array
            risk_sensitivity: ignored in heuristic mode

        Returns:
            costs: [num_edges] predicted costs
        """
        distance = features_np[:, 0]
        num_robots = features_np[:, 2]
        same_dir = features_np[:, 3]
        opposite_dir = features_np[:, 4]
        people_count = features_np[:, 5]
        clutter = features_np[:, 6]

        base_time = distance / 1.0  # Assume 1 m/s

        # Simple additive penalties
        congestion = num_robots * 1.5 + opposite_dir * 2.0
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

    def add_traversal_record(self, record) -> None:
        """
        Add a completed traversal record to the replay buffer.

        Args:
            record: EdgeTraversalRecord with valid entry_time and exit_time
        """
        actual_time = record.actual_traversal_time
        if actual_time <= 0:
            return  # Invalid record

        features = record.to_feature_vector()
        self.buffer.append((features, actual_time))
        self.total_records_added += 1

        # Evict oldest if buffer full
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)

    def add_records_batch(self, records: list) -> int:
        """Add multiple records at once. Returns count of valid records added."""
        count = 0
        for record in records:
            if record.actual_traversal_time > 0:
                self.add_traversal_record(record)
                count += 1
        return count

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """
        Train one step on a random batch from the replay buffer.

        Returns:
            Loss value, or None if not enough data
        """
        if len(self.buffer) < self.min_train_samples:
            return None

        # Sample batch
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        features_list = [b[0] for b in batch]
        targets_list = [b[1] for b in batch]

        features = torch.tensor(np.array(features_list), dtype=torch.float32)
        targets = torch.tensor(targets_list, dtype=torch.float32)

        # Forward pass
        mean_time, log_var = self.model(features)

        # Gaussian NLL loss
        # NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
        variance = torch.exp(log_var)
        nll = 0.5 * (log_var + (targets - mean_time) ** 2 / variance)
        loss = nll.mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.total_train_steps += 1
        loss_val = loss.item()
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        return loss_val

    def train_epoch(self, batch_size: int = 64, num_steps: int = 10) -> Optional[float]:
        """
        Train multiple steps. Returns average loss or None if not enough data.
        """
        if len(self.buffer) < self.min_train_samples:
            return None

        losses = []
        for _ in range(num_steps):
            loss = self.train_step(batch_size)
            if loss is not None:
                losses.append(loss)

        return np.mean(losses) if losses else None

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
            # Count directional robots
            same_dir = 0
            opposite_dir = 0
            for rid, (prog, fidx, tidx) in edge.active_robot_progress.items():
                # We don't know the "query direction" here, so count total
                # The caller should handle direction if needed
                same_dir += 1  # Conservative: count all as same direction

            # Count approaching robots
            approaching = 0
            if all_robots:
                for robot in all_robots:
                    if hasattr(robot, 'planned_path') and robot.planned_path:
                        # Check if this edge is in the robot's future path
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

            # Time of day
            current_time = getattr(graph_state, 'current_time', 0.0)
            time_of_day = (current_time % 86400.0) / 86400.0

            features.append([
                edge.distance_m,
                edge.corridor_width,
                float(len(edge.active_robot_ids)),
                float(same_dir),
                float(opposite_dir),
                float(edge.people_count),
                edge.clutter_level,
                float(approaching),
                float(from_node_occ),
                float(to_node_occ),
                time_of_day,
                0.0,  # stop_count (real-time, not available at prediction time)
            ])

        return np.array(features, dtype=np.float32)
