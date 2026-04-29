"""
Decision Logger — one CSV row per task assignment decision.

Captures demand context, robot state, and decision features at the moment
of assignment. When a policy is loaded, also records policy confidence and
heuristic comparison. When running heuristic only, captures ground-truth
context data for later policy validation.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


_FIELDS = [
    # Identity
    "wall_time", "sim_time",
    # Task
    "task_id", "sku_id", "leg_type", "task_type",
    "from_node", "to_node", "num_items",
    # Demand / stock at decision time
    "stock_level", "stock_pct", "reorder_point", "par_level",
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
        })
        self._fh.flush()

    def close(self):
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path
