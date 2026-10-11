import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature vector index 0 is distance_m (excluded from the encoder); indices 1-8 are congestion features. Direction/stop-count fields were dropped: unknown pre-traversal, or post-traversal only.
EDGE_COST_FEATURE_DIM = 9
EDGE_COST_ENCODER_DIM = 8


class LearnedEdgeCostModel(nn.Module):
    """Predicts edge traversal time as a mean delay factor and log-variance so path planning can account for risk."""

    def __init__(self, input_dim: int = EDGE_COST_FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        encoder_input_dim = input_dim - 1  # distance_m (index 0) excluded from encoder

        # Normalize congestion features before encoding (indices 1-8 span very different scales)
        self.input_norm = nn.LayerNorm(encoder_input_dim)

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

        self.log_var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Mean head bias set to -3.0 so softplus(-3)≈0.049 → delay_factor starts near 1.05 (default softplus(0)=0.693 would start near 1.69).
        mean_output = self.mean_head[-1]
        variance_output = self.log_var_head[-1]
        assert isinstance(mean_output, nn.Linear) and isinstance(variance_output, nn.Linear)
        assert mean_output.bias is not None and variance_output.bias is not None
        nn.init.constant_(mean_output.bias, -3.0)
        nn.init.constant_(variance_output.bias, -2.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the delay-factor distribution (mean >= 1.0, clamped log-variance) from congestion features, excluding distance_m."""
        # Congestion features only (exclude distance_m at index 0)
        congestion_features = features[:, 1:]
        h = self.encoder(self.input_norm(congestion_features))

        # Predict delay factor >= 1.0 (corridor can only be slower than free-flow)
        raw_delay = self.mean_head(h).squeeze(-1)
        delay_factor = 1.0 + F.softplus(raw_delay)

        log_var = self.log_var_head(h).squeeze(-1)
        log_var = torch.clamp(log_var, min=-8.0, max=4.0)

        return delay_factor, log_var

    def predict_cost(self, features: torch.Tensor, risk_sensitivity: float = 0.0) -> torch.Tensor:
        """Predict path cost in seconds for Dijkstra, optionally risk-adjusted."""
        delay_factor, log_var = self.forward(features)

        # time = delay_factor * (distance / max_v); sim/config convention uses 0.5 m/s uniformly.
        distance = features[:, 0]
        base_time = distance / 0.5

        if risk_sensitivity > 0.0:
            # Risk-adjusted: penalise edges with high variance in delay
            std_dev = torch.exp(0.5 * log_var)
            return (delay_factor + risk_sensitivity * std_dev) * base_time
        else:
            return delay_factor * base_time

    def predict_numpy(self, features_np: np.ndarray, risk_sensitivity: float = 0.0) -> np.ndarray:
        """Convenience method for use in Dijkstra (numpy in, numpy out)."""
        with torch.no_grad():
            features = torch.tensor(features_np, dtype=torch.float32)
            costs = self.predict_cost(features, risk_sensitivity)
            return costs.numpy()


class HeuristicEdgeCostModel:
    """Fallback heuristic model, matching the LearnedEdgeCostModel interface, used before enough data is collected."""

    def predict_numpy(self, features_np: np.ndarray, risk_sensitivity: float = 0.0) -> np.ndarray:
        """Predict traversal cost using simple additive congestion/people/clutter penalties."""
        distance = features_np[:, 0]
        num_robots = features_np[:, 2]
        people_count = features_np[:, 3]
        clutter = features_np[:, 4]
        approaching = features_np[:, 5]

        # Match the simulator/config baseline corridor speed used elsewhere.
        base_time = distance / 0.5

        congestion = (num_robots + approaching) * 1.5
        people_penalty = people_count * 0.5
        clutter_penalty = clutter * 10.0

        return base_time + congestion + people_penalty + clutter_penalty


class EdgeCostTrainer:
    """Supervised trainer for LearnedEdgeCostModel using a replay buffer and Gaussian NLL loss to jointly learn mean and variance."""

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

        self.buffer: List[Tuple[np.ndarray, float]] = []

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
            return float("inf")
        return float(np.mean(self.recent_losses))

    def get_stats(self) -> dict:
        return {
            "buffer_size": len(self.buffer),
            "total_records": self.total_records_added,
            "total_train_steps": self.total_train_steps,
            "avg_recent_loss": self.avg_recent_loss,
            "is_ready": self.is_ready,
        }


class EdgeCostManager:
    """Manages the transition from the heuristic to the learned edge-cost model as traversal data accumulates."""

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
        self.trainer = EdgeCostTrainer(self.model, lr=lr, min_train_samples=min_train_samples)

        self.risk_sensitivity = risk_sensitivity
        self.train_interval_records = train_interval_records
        self.train_steps_per_interval = train_steps_per_interval
        self._records_since_last_train = 0
        self.using_learned_model = False

    def predict_edge_costs(self, features_np: np.ndarray) -> np.ndarray:
        """Predict edge costs for path planning, automatically switching between the heuristic and learned model."""
        if self.using_learned_model and self.trainer.is_ready:
            return self.model.predict_numpy(features_np, self.risk_sensitivity)
        else:
            return self.heuristic.predict_numpy(features_np, self.risk_sensitivity)

    def add_traversal_records(self, records: list) -> None:
        """Add completed traversal records and optionally trigger training."""
        count = self.trainer.add_records_batch(records)
        self._records_since_last_train += count

        if self._records_since_last_train >= self.train_interval_records and self.trainer.is_ready:
            self.trainer.train_epoch(num_steps=self.train_steps_per_interval)
            self._records_since_last_train = 0
            self.using_learned_model = True

    def get_stats(self) -> dict:
        stats = self.trainer.get_stats()
        stats["using_learned_model"] = self.using_learned_model
        stats["risk_sensitivity"] = self.risk_sensitivity
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
        self.train_interval_records = state.get(
            "train_interval_records", self.train_interval_records
        )
        self.train_steps_per_interval = state.get(
            "train_steps_per_interval", self.train_steps_per_interval
        )

    def build_edge_features(self, graph_state, all_robots=None) -> np.ndarray:
        """Build the edge feature matrix for all edges in the graph, including robot approach/occupancy counts."""
        features = []
        for edge_idx, edge in enumerate(graph_state.edges):
            # Count approaching robots (have this edge in their planned path but not on it)
            approaching = 0
            if all_robots:
                for robot in all_robots:
                    if hasattr(robot, "planned_path") and robot.planned_path:
                        path = robot.planned_path
                        for i in range(len(path) - 1):
                            from_node = graph_state.nodes[path[i]].node_id
                            to_node = graph_state.nodes[path[i + 1]].node_id
                            if (edge.from_node == from_node and edge.to_node == to_node) or (
                                edge.from_node == to_node and edge.to_node == from_node
                            ):
                                approaching += 1
                                break

            from_node_occ = 0
            to_node_occ = 0
            for node in graph_state.nodes:
                if node.node_id == edge.from_node:
                    from_node_occ = len(getattr(node, "current_robot_ids", []))
                elif node.node_id == edge.to_node:
                    to_node_occ = len(getattr(node, "current_robot_ids", []))

            current_time = getattr(graph_state, "current_time", 0.0)
            time_of_day = (current_time % 86400.0) / 86400.0

            # Matches EdgeTraversalRecord.to_feature_vector(); direction/stop-count fields excluded (unknown pre-/post-traversal only).
            features.append(
                [
                    edge.distance_m,
                    edge.corridor_width,
                    float(len(edge.active_robot_ids)),
                    float(edge.people_count),
                    edge.clutter_level,
                    float(approaching),
                    float(from_node_occ),
                    float(to_node_occ),
                    time_of_day,
                ]
            )

        return np.array(features, dtype=np.float32)
