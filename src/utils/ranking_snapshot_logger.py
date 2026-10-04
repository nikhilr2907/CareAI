"""
Ranking Snapshot Logger — one row per pending task per assignment decision.

Captures the full ranked queue state at each decision point so offline
analysis can compute Spearman correlation between learned scores and
urgency features (TTS, stock, age) across training.

Join to decisions.csv on assigned_task_id + sim_time to get the queue
context for a given assignment decision.
"""

import csv
from pathlib import Path


_FIELDS = [
    "assigned_task_id",   # task_id of the task being assigned — joins to decisions.csv
    "sim_time",
    "iteration",
    "task_id",            # task in the pending queue at this snapshot
    "rank",               # position in queue (0 = top, being assigned)
    "task_score",         # learned_score set by scorer
    "current_tts",        # live time-to-stockout in hours
    "stock_pct",          # current_stock / max_stock
    "age",                # sim_time - arrival_time
    "time_to_deadline",   # current_deadline - sim_time
    "num_items",
    "queue_size",         # total pending tasks at this snapshot
]


class RankingSnapshotLogger:
    """
    Appends one row per pending task at each assignment decision.

    Usage:
        rl = RankingSnapshotLogger(output_dir)
        rl.log_snapshot(assigned_task, pending_tasks, sim_time, iteration)
        rl.close()
    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / "ranking_snapshots.csv"
        self._fh = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=_FIELDS)
        self._writer.writeheader()

    def log_snapshot(self, assigned_task, pending_tasks, sim_time: float, iteration: int):
        """Log all pending tasks at the moment assigned_task is being assigned."""
        queue_size = len(pending_tasks)
        for rank, task in enumerate(pending_tasks):
            stock = task.get_current_sku_stock_level() if hasattr(task, 'get_current_sku_stock_level') else getattr(task, 'current_sku_stock_level', None)
            max_stock = getattr(task, 'sku_max_level', None)
            stock_pct = round(stock / max_stock * 100, 1) if (stock is not None and max_stock) else None
            arrival = getattr(task, 'arrival_time', None)
            deadline = getattr(task, 'current_deadline', getattr(task, 'deadline', None))

            self._writer.writerow({
                "assigned_task_id": assigned_task.task_id,
                "sim_time": round(sim_time, 2),
                "iteration": iteration,
                "task_id": task.task_id,
                "rank": rank,
                "task_score": round(getattr(task, 'learned_score', 0.0), 4),
                "current_tts": round(task.get_current_time_to_stockout(), 2),
                "stock_pct": stock_pct,
                "age": round(sim_time - arrival, 2) if arrival is not None else None,
                "time_to_deadline": round(deadline - sim_time, 2) if deadline is not None else None,
                "num_items": getattr(task, 'num_items', None),
                "queue_size": queue_size,
            })
        self._fh.flush()

    def close(self):
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path
