"""
Decision Logger — one CSV row per task assignment decision.

Captures demand context, robot state, and decision features at the moment
of assignment. When a policy is loaded, also records policy confidence and
heuristic comparison. When running heuristic only, captures ground-truth
context data for later policy validation.

Outcome columns (reward, deadline_met) are written as None at decision time
and back-filled in batch via queue_outcome() + flush_outcomes() once tasks
complete, avoiding per-completion file I/O.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_FIELDS = [
    # Identity
    "wall_time", "sim_time",
    # Task
    "task_id", "sku_id", "leg_type", "task_type",
    "from_node", "to_node", "num_items",
    # Demand / stock at decision time
    "stock_level", "stock_pct", "reorder_point", "par_level", "current_tts",
    # Task timing (allows offline reconstruction of age and time_to_deadline)
    "arrival_time", "current_deadline", "estimated_duration",
    # Task ranking
    "task_rank", "num_pending", "task_score",
    "second_task_id", "second_stock_pct", "score_margin",
    # Decision
    "action_robot", "heuristic_robot", "action_agreed", "source",
    # Policy confidence (null when heuristic only)
    "action_logprob", "action_entropy", "top_robot_prob",
    # Robot state at decision time
    "robot_battery", "robot_queue_depth", "robot_load",
    # Context
    "num_pending_tasks", "time_of_day_h", "priority_score",
    # Warmup (training only)
    "is_warmup", "used_heuristic",
    # Queue/fleet state at decision time (critic context)
    "q_num_pending", "q_avg_priority", "q_num_urgent", "q_oldest_age",
    "q_num_near_deadline", "q_main_pickups", "q_main_dropoffs",
    "q_overflow_pickups", "q_overflow_dropoffs", "q_avg_free_slots",
    "q_avg_overflow", "q_time_sin", "q_time_cos", "q_day_norm",
    "q_fleet_busy_ratio", "q_num_robots",
    # Per-robot assignment scores at decision time
    "robot_logits",
    # Outcomes (back-filled after task completion)
    "reward", "deadline_met",
]


class DecisionLogger:
    """
    Writes one CSV row per assignment decision across all scripts.

    Usage:
        dl = DecisionLogger(output_dir)
        dl.log(task, action, heuristic_action, env, sim_time)
        dl.close()
    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / "decisions.csv"
        self._fh = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=_FIELDS)
        self._writer.writeheader()
        self._outcome_buffer: dict[int, dict] = {}  # task_id -> {reward, deadline_met}

    def log(
        self,
        task,
        action: int,
        heuristic_action: int,
        env,
        sim_time: float,
        is_warmup: bool = False,
        used_heuristic: bool = False,
        action_logprob: Optional[float] = None,
        action_entropy: Optional[float] = None,
        top_robot_prob: Optional[float] = None,
        queue_features=None,
        robot_logits=None,
    ):
        robot = env.robots[action] if 0 <= action < len(env.robots) else None

        stock_level = getattr(task, "sku_stock_level_at_assign",
                     getattr(task, "current_sku_stock_level",
                     getattr(task, "sku_stock_level", None)))
        max_level = getattr(task, "sku_max_level", None)
        stock_pct = round(stock_level / max_level * 100, 1) if (stock_level is not None and max_level) else None

        # Ranking context — task's position in the pending queue and runner-up info
        pending = env.pending_tasks
        try:
            task_rank = pending.index(task)
        except ValueError:
            task_rank = None
        num_pending = len(pending)
        task_score = round(getattr(task, "learned_score", 0.0), 4)

        second_task_id = second_stock_pct = score_margin = None
        if num_pending >= 2:
            second = pending[1] if task_rank == 0 else pending[0]
            second_task_id = second.task_id
            s_stock = getattr(second, "sku_stock_level_at_assign",
                     getattr(second, "current_sku_stock_level",
                     getattr(second, "sku_stock_level", None)))
            s_max = getattr(second, "sku_max_level", None)
            second_stock_pct = round(s_stock / s_max * 100, 1) if (s_stock is not None and s_max) else None
            second_score = getattr(second, "learned_score", 0.0)
            score_margin = round(task_score - second_score, 4)

        self._writer.writerow({
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "sim_time": round(sim_time, 2),
            "task_id": task.task_id,
            "sku_id": getattr(task, "sku_id", None),
            "leg_type": getattr(task, "leg_type", None),
            "task_type": getattr(task, "task_type", None),
            "from_node": getattr(task, "from_location_index", None),
            "to_node": getattr(task, "to_location_index", None),
            "num_items": getattr(task, "num_items", None),
            "stock_level": stock_level,
            "stock_pct": stock_pct,
            "reorder_point": getattr(task, "reorder_point", None),
            "par_level": getattr(task, "par_level", None),
            "current_tts": round(task.get_current_time_to_stockout(), 2),
            "arrival_time": getattr(task, "arrival_time", None),
            "current_deadline": getattr(task, "current_deadline", getattr(task, "deadline", None)),
            "estimated_duration": getattr(task, "estimated_duration", None),
            "task_rank": task_rank,
            "num_pending": num_pending,
            "task_score": task_score,
            "second_task_id": second_task_id,
            "second_stock_pct": second_stock_pct,
            "score_margin": score_margin,
            "action_robot": action,
            "heuristic_robot": heuristic_action,
            "action_agreed": int(action == heuristic_action),
            "source": getattr(task, "source", "unknown"),
            "action_logprob": round(action_logprob, 4) if action_logprob is not None else None,
            "action_entropy": round(action_entropy, 4) if action_entropy is not None else None,
            "top_robot_prob": round(top_robot_prob, 4) if top_robot_prob is not None else None,
            "robot_battery": round(robot.telemetry.battery_level, 4) if (robot and robot.telemetry) else None,
            "robot_queue_depth": len(robot.task_queue) if robot else None,
            "robot_load": robot.telemetry.current_capacity if (robot and robot.telemetry) else None,
            "num_pending_tasks": len(env.pending_tasks),
            "time_of_day_h": round((sim_time % 86400) / 3600, 3),
            "priority_score": round(getattr(task, "learned_score", 0.0), 4),
            "is_warmup": int(is_warmup),
            "used_heuristic": int(used_heuristic),
            "q_num_pending":       round(float(queue_features[0]), 4) if queue_features is not None else None,
            "q_avg_priority":      round(float(queue_features[1]), 4) if queue_features is not None else None,
            "q_num_urgent":        round(float(queue_features[2]), 4) if queue_features is not None else None,
            "q_oldest_age":        round(float(queue_features[3]), 4) if queue_features is not None else None,
            "q_num_near_deadline": round(float(queue_features[4]), 4) if queue_features is not None else None,
            "q_main_pickups":      round(float(queue_features[5]), 4) if queue_features is not None else None,
            "q_main_dropoffs":     round(float(queue_features[6]), 4) if queue_features is not None else None,
            "q_overflow_pickups":  round(float(queue_features[7]), 4) if queue_features is not None else None,
            "q_overflow_dropoffs": round(float(queue_features[8]), 4) if queue_features is not None else None,
            "q_avg_free_slots":    round(float(queue_features[9]), 4) if queue_features is not None else None,
            "q_avg_overflow":      round(float(queue_features[10]), 4) if queue_features is not None else None,
            "q_time_sin":          round(float(queue_features[11]), 4) if queue_features is not None else None,
            "q_time_cos":          round(float(queue_features[12]), 4) if queue_features is not None else None,
            "q_day_norm":          round(float(queue_features[13]), 4) if queue_features is not None else None,
            "q_fleet_busy_ratio":  round(float(queue_features[14]), 4) if queue_features is not None else None,
            "q_num_robots":        round(float(queue_features[15]), 4) if queue_features is not None else None,
            "robot_logits": json.dumps([round(p, 4) for p in robot_logits]) if robot_logits is not None else None,
            "reward": None,
            "deadline_met": None,
        })
        self._fh.flush()

    def queue_outcome(self, task_id: int, reward: float, deadline_met: Optional[bool] = None):
        """Buffer an outcome for task_id. Flushed to disk in batch by flush_outcomes()."""
        self._outcome_buffer[task_id] = {
            "reward": round(reward, 4),
            "deadline_met": int(deadline_met) if deadline_met is not None else None,
        }

    def flush_outcomes(self):
        """Back-fill buffered outcomes into the CSV in a single read-write pass."""
        if not self._outcome_buffer:
            return
        import pandas as pd

        # Close the write handle before pandas reads the file; leaving it open
        # causes a stale file position after df.to_csv() overwrites the file,
        # which corrupts subsequent writerow() calls.
        self._fh.flush()
        self._fh.close()
        self._fh = None

        try:
            df = pd.read_csv(self._path)
            for task_id, outcomes in self._outcome_buffer.items():
                mask = df["task_id"] == task_id
                for col, val in outcomes.items():
                    df.loc[mask, col] = val
            df.to_csv(self._path, index=False)
            self._outcome_buffer.clear()
        except Exception as e:
            logger.warning(f"flush_outcomes: failed to back-fill outcomes — {e}")
        finally:
            # Reopen in append mode so subsequent writerow() calls go to the end
            self._fh = open(self._path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=_FIELDS)

    def close(self):
        self.flush_outcomes()
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path
