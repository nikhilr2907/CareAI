"""
Task-specific logging for GAPO training.
Tracks every assignment and completion with full metadata including SKU levels.
"""
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict


class TaskLogger:
    """Log individual task assignments and completions to text file."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.task_file = self.log_dir / "tasks.log"

        # Setup logger
        self.logger = logging.getLogger('GAPO_Tasks')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(self.task_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.task_metadata = {}  # task_id -> metadata
        # Track per-iteration stats
        self.iteration_assignments = defaultdict(list)  # iter -> [robot_ids]
        self.iteration_completions = defaultdict(lambda: {'on_time': 0, 'late': 0, 'robots': defaultdict(int)})  # iter -> stats

    def log_assignment(self, task, assigned_robot: int, assignment_reward: float,
                      iteration: int, sim_time: float, num_assignments: int, buffer_size: int):
        """Log task assignment."""
        task_id = task.task_id

        # Store metadata for completion later
        self.task_metadata[task_id] = {
            'iteration': iteration,
            'arrival_time': task.arrival_time,
            'deadline': task.deadline,
            'estimated_duration': task.estimated_duration,
            'from_location_idx': task.from_location_index,
            'to_location_idx': getattr(task, 'to_location_index', None),
            'assigned_robot': assigned_robot,
            'assignment_reward': assignment_reward,
            'sim_time_assigned': sim_time,
            'sku_id': getattr(task, 'sku_id', None),
            'sku_stock_level_at_assign': getattr(task, 'sku_stock_level', None),
            'sku_max_level': getattr(task, 'sku_max_level', None),
            'reorder_point': getattr(task, 'reorder_point', None),
            'par_level': getattr(task, 'par_level', None),
        }

        # Log assignment event
        feasibility = (task.deadline - sim_time) / max(task.estimated_duration, 1.0) - 1.0
        to_loc = getattr(task, 'to_location_index', '?')
        sku_id = getattr(task, 'sku_id', 'N/A')
        sku_stock = getattr(task, 'sku_stock_level', None)
        sku_max = getattr(task, 'sku_max_level', None)

        # Build SKU info string
        sku_info = f"sku_id={sku_id}"
        if sku_stock is not None and sku_max is not None:
            stock_pct = (sku_stock / sku_max * 100) if sku_max > 0 else 0
            sku_info += f" stock_level={sku_stock:.1f}/{sku_max:.1f}({stock_pct:.0f}%)"
            if hasattr(task, 'reorder_point') and task.reorder_point is not None:
                sku_info += f" reorder={task.reorder_point:.1f}"

        self.logger.info(
            f"ASSIGN task_id={task_id} iter={iteration} sim_time={sim_time:.1f}s "
            f"robot={assigned_robot} location={task.from_location_index}->{to_loc} "
            f"arrival={task.arrival_time:.1f}s deadline={task.deadline:.1f}s "
            f"est_duration={task.estimated_duration:.1f}s feasibility={feasibility:.2f} "
            f"assign_reward={assignment_reward:.4f} {sku_info} "
            f"(buffer={buffer_size} assignments={num_assignments})"
        )

        # Track assignment for iteration summary
        self.iteration_assignments[iteration].append(assigned_robot)

    def log_completion(self, task_id: int, completion_reward: float,
                      actual_completion_time: Optional[float], on_time: bool,
                      sim_time: float):
        """Log task completion."""
        if task_id not in self.task_metadata:
            # Task not in our log (maybe predates logging), but still log the completion
            self.logger.warning(
                f"COMPLT task_id={task_id} [NO METADATA] sim_time={sim_time:.1f}s "
                f"completion_reward={completion_reward:.4f} "
                f"actual_time={actual_completion_time:.1f}s if actual_completion_time else '?' "
                f"on_time={'YES' if on_time else 'LATE'}"
            )
            return

        meta = self.task_metadata[task_id]
        total_reward = meta['assignment_reward'] + completion_reward

        # Build SKU info string
        sku_info = f"sku_id={meta['sku_id'] or 'N/A'}"
        if meta['sku_stock_level_at_assign'] is not None and meta['sku_max_level'] is not None:
            stock_pct = (meta['sku_stock_level_at_assign'] / meta['sku_max_level'] * 100) if meta['sku_max_level'] > 0 else 0
            sku_info += f" stock_at_assign={meta['sku_stock_level_at_assign']:.1f}/{meta['sku_max_level']:.1f}({stock_pct:.0f}%)"
            if meta['reorder_point'] is not None:
                sku_info += f" reorder={meta['reorder_point']:.1f}"

        # Log completion event
        actual_time_str = f"{actual_completion_time:.1f}s" if actual_completion_time is not None else "?"
        self.logger.info(
            f"COMPLT task_id={task_id} iter={meta['iteration']} sim_time={sim_time:.1f}s "
            f"robot={meta['assigned_robot']} location={meta['from_location_idx']}->{meta['to_location_idx'] or '?'} "
            f"time_from_assign={sim_time - meta['sim_time_assigned']:.1f}s "
            f"assign_reward={meta['assignment_reward']:.4f} completion_reward={completion_reward:.4f} "
            f"total_reward={total_reward:.4f} actual_time={actual_time_str} "
            f"on_time={'YES' if on_time else 'LATE'} {sku_info}"
        )

        # Track completion for iteration summary
        iter_num = meta['iteration']
        robot = meta['assigned_robot']
        if on_time:
            self.iteration_completions[iter_num]['on_time'] += 1
        else:
            self.iteration_completions[iter_num]['late'] += 1
        self.iteration_completions[iter_num]['robots'][robot] += 1

        # Clean up
        del self.task_metadata[task_id]

    def log_iteration_summary(self, iteration: int):
        """Log summary of iteration completions and robot breakdown."""
        if iteration not in self.iteration_completions:
            return  # No completions in this iteration

        stats = self.iteration_completions[iteration]
        assignments = self.iteration_assignments[iteration]

        on_time = stats['on_time']
        late = stats['late']
        total_completed = on_time + late

        if total_completed == 0:
            return  # No completions logged

        on_time_pct = (on_time / total_completed * 100) if total_completed > 0 else 0
        total_assigned = len(assignments)
        assignment_to_completion_ratio = total_assigned / total_completed if total_completed > 0 else 0

        # Robot breakdown
        robot_breakdown = ", ".join(
            f"R{robot}:{count}"
            for robot, count in sorted(stats['robots'].items())
        )

        # Log summary
        self.logger.info(
            f"=== ITERATION {iteration} SUMMARY ===\n"
            f"  Assigned: {total_assigned} | Completed: {total_completed} | A:C Ratio: {assignment_to_completion_ratio:.2f}:1\n"
            f"  On-Time: {on_time}/{total_completed} ({on_time_pct:.1f}%) | Late: {late}/{total_completed}\n"
            f"  Robot Completion Breakdown: {robot_breakdown}\n"
            f"{'=' * 60}"
        )