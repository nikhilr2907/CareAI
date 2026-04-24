from dataclasses import dataclass
from pathlib import Path
from typing import List
from collections import defaultdict
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# Type hints for objects we'll receive
from src.environment.robot.robot_state import RobotState
from src.environment.graph.node import GraphNode
from src.environment.graph.edge import GraphEdge


class RealtimeAnalyticsCollector:
    """
    Collects and aggregates metrics from simulation/real deployment.

    Data flow:
    - Task completion data from TaskLogger.cost_data
    - RobotState/RobotTelemetry → record_robot_telemetry()
    - GraphNode → record_inventory_snapshot()
    - GraphEdge → record_corridor_state()

    Outputs:
    - metrics.json (raw events)
    - metrics_summary.csv (per-iteration aggregates)
    - plots_final/*.png (6 matplotlib plots)
    """

    def __init__(self,
                 output_dir: Path,
                 mode: str = 'sim',
                 snapshot_frequency: int = 50,
                 task_logger=None,
                 sku_logger=None):
        """
        Args:
            output_dir: Directory for metrics output (analytics/ subdir)
            mode: 'sim' or 'real' (for logging/configuration)
            snapshot_frequency: Generate plots every N iterations
            task_logger: Reference to TaskLogger for reading task completion data
            sku_logger: Reference to SKULogger for reading SKU event data
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.snapshot_frequency = snapshot_frequency
        self.task_logger = task_logger
        self.sku_logger = sku_logger

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.nodes = None  # Set by initialize()

        # Raw event streams (append-only)
        # Note: task completion data now read from task_logger.cost_data
        self.robot_samples = []    # {robot_id, status, battery, position, assigned_tasks}
        self.corridor_samples = []  # {edge, robot_count, clutter, people}

        # Iteration tracking
        self.current_iteration = 0
        self.iteration_summaries = defaultdict(dict)  # iteration → summary stats

        # Aggregated metrics (updated per _update_metrics())
        self.metrics = {
            'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0},
            'utilization': {'idle': 0, 'active': 0, 'waiting': 0},
            'replenishment_lag': {'mean': 0, 'max': 0},
            'cost': {'mean': 0, 'min': 0, 'max': 0},
            'a_c_ratio': 0.0,
        }

    # ========== RECORDING METHODS ==========

    def record_robot_telemetry(self, robots: List[RobotState], current_time: float):
        """
        Sample robot telemetry from RobotState + RobotTelemetry.
        Called periodically during training (e.g., every 10 steps).

        Args:
            robots: List of RobotState objects
            current_time: Current simulation/real time
        """
        for robot in robots:
            if robot.telemetry is not None:
                # Task-aware idle detection
                if len(robot.task_queue) == 0:
                    status = 'IDLE'
                elif robot.telemetry.velocity_ms > 0.1:
                    status = 'ACTIVE'
                else:
                    status = 'WAITING_AT_LOCATION'

                self.robot_samples.append({
                    'robot_id': robot.robot_id,
                    'status': status,
                    'battery': robot.battery_level,
                    'position': robot.current_position,
                    'assigned_tasks': len(robot.task_queue),
                    'timestamp': current_time,
                })

    def initialize(self, nodes: List[GraphNode]):
        """Store graph nodes for future metric modules. Call once after env creation."""
        self.nodes = nodes

    def record_corridor_state(self, edges: List[GraphEdge], current_time: float):
        """
        Sample corridor congestion from GraphEdge objects.
        Called periodically (e.g., every 50 steps).

        Args:
            edges: List of GraphEdge objects
            current_time: Current simulation/real time
        """
        for edge in edges:
            robot_count = len(edge.active_robot_ids) if hasattr(edge, 'active_robot_ids') else 0
            self.corridor_samples.append({
                'edge_id': f"{edge.from_node}->{edge.to_node}",
                'distance': edge.distance_m,
                'robot_count': robot_count,
                'clutter': getattr(edge, 'clutter_level', 0),
                'people': getattr(edge, 'people_count', 0),
                'timestamp': current_time,
            })

    # ========== METRICS COMPUTATION ==========

    def _update_metrics(self):
        """Recompute all aggregated metrics from raw event sources."""
        self._compute_latency()
        self._compute_utilization()
        self._compute_cost()
        self._compute_replenishment_lag()
        self._compute_ac_ratio()
        self._compute_sku_demand()
        self._compute_stockout_count()

    def _compute_latency(self):
        if not (self.task_logger and self.task_logger.cost_data):
            self.metrics['latency'] = {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0}
            return
        latencies = [e['latency'] for e in self.task_logger.cost_data if e.get('latency') is not None]
        if latencies:
            self.metrics['latency'] = {
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'mean': float(np.mean(latencies)),
            }

    def _compute_utilization(self):
        if not self.robot_samples:
            return
        statuses = defaultdict(int)
        for sample in self.robot_samples:
            statuses[sample['status']] += 1
        self.metrics['utilization'] = {
            'idle': int(statuses.get('IDLE', 0)),
            'active': int(statuses.get('ACTIVE', 0)),
            'waiting': int(statuses.get('WAITING_AT_LOCATION', 0)),
        }

    def _compute_cost(self):
        if not (self.task_logger and self.task_logger.cost_data):
            return
        data = self.task_logger.cost_data
        self.metrics['cost'] = {
            'mean':              float(np.mean([e['total_cost'] for e in data])),
            'min':               float(np.min([e['total_cost'] for e in data])),
            'max':               float(np.max([e['total_cost'] for e in data])),
            'mean_distance':     float(np.mean([e['distance_traveled'] * 0.5 for e in data])),
            'mean_robot_hours':  float(np.mean([e['robot_hours'] * 50.0 for e in data])),
            'mean_energy':       float(np.mean([e['energy_consumed'] * 0.10 for e in data])),
            'mean_cost_per_reward': float(np.mean([e['cost_per_reward'] for e in data])),
        }

    def _compute_replenishment_lag(self):
        if not (self.task_logger and self.task_logger.cost_data):
            return
        rep = [e for e in self.task_logger.cost_data
               if e.get('task_type') == 'replenishment' and e.get('leg_type') == 'dropoff']
        queue_waits, assign_to_exec, exec_to_done, totals = [], [], [], []
        for e in rep:
            arrival, assigned = e.get('arrival_time'), e.get('sim_time_assigned')
            exec_start, completion = e.get('execution_start_time'), e.get('sim_time')
            if arrival is not None and assigned is not None:
                queue_waits.append(assigned - arrival)
            if assigned is not None and exec_start is not None:
                assign_to_exec.append(exec_start - assigned)
            if exec_start is not None and completion is not None:
                exec_to_done.append(completion - exec_start)
            if arrival is not None and completion is not None:
                totals.append(completion - arrival)

        def _agg(vals):
            if not vals:
                return {'mean': 0, 'p95': 0, 'max': 0}
            return {
                'mean': float(np.mean(vals)),
                'p95': float(np.percentile(vals, 95)),
                'max': float(np.max(vals)),
            }
        self.metrics['replenishment_lag'] = {
            'queue_wait':     _agg(queue_waits),
            'assign_to_exec': _agg(assign_to_exec),
            'execution':      _agg(exec_to_done),
            'total':          _agg(totals),
        }

    def _compute_ac_ratio(self):
        if not self.task_logger:
            return
        total_assigned = sum(len(v) for v in self.task_logger.iteration_assignments.values())
        total_completed = sum(s['completed'] for s in self.task_logger.iteration_completions.values())
        self.metrics['a_c_ratio'] = total_assigned / total_completed if total_completed > 0 else 0.0

    def _compute_sku_demand(self):
        # TODO: time-series analysis (time-of-day variation, demand spikes) once snapshot_data list added
        if not self.sku_logger:
            return
        rates = self.sku_logger.get_rolling_demand_rates()
        if not rates:
            return
        sku_demand = {}
        for (node_idx, sku_id), rate in rates.items():
            snap = self.sku_logger.latest_snapshots.get((node_idx, sku_id))
            node_tag = snap['node_tag'] if snap else f"node_{node_idx}"
            sku_demand[f"{sku_id}@{node_tag}"] = rate
        self.metrics['sku_demand'] = sku_demand

    def _compute_stockout_count(self):
        if self.sku_logger:
            self.metrics['stockout_count'] = len(self.sku_logger.stockout_events)

    def increment_iteration(self):
        """Call at end of each training iteration to track progress."""
        self.current_iteration += 1

        # Store iteration summary
        task_count = len(self.task_logger.cost_data) if self.task_logger else 0
        self.iteration_summaries[self.current_iteration] = {
            'task_count': task_count,
            'robot_samples': len(self.robot_samples),
            'latency_p95': self.metrics['latency'].get('p95', 0),
        }

    # ========== OUTPUT GENERATION ==========

    def save_metrics_json(self):
        """Save all raw events and flow metrics to JSON for later analysis."""
        output = {
            'mode': self.mode,
            'iteration': self.current_iteration,
            'task_completions': self.task_logger.cost_data if self.task_logger else [],
            'robot_samples': self.robot_samples,
            'corridor_samples': self.corridor_samples,
            'metrics': self.metrics,
        }

        json_path = self.output_dir / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

    def save_metrics_csv(self):
        """Save iteration summaries to CSV."""
        csv_path = self.output_dir / 'metrics_summary.csv'

        if not self.iteration_summaries:
            return

        fieldnames = list(self.iteration_summaries[min(self.iteration_summaries.keys())].keys())
        fieldnames = ['iteration'] + fieldnames

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for iteration in sorted(self.iteration_summaries.keys()):
                row = {'iteration': iteration}
                row.update(self.iteration_summaries[iteration])
                writer.writerow(row)

    def plot_snapshot(self, final: bool = False):
        """
        Generate all 6 plots.

        Args:
            final: If True, save to plots_final/ with final naming
        """
        self._update_metrics()

        if final:
            plots_dir = self.output_dir / 'plots_final'
        else:
            plots_dir = self.output_dir / 'plots_iter'

        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate 6 plots
        self._plot_latency_percentiles(plots_dir, final)
        self._plot_robot_utilization(plots_dir, final)
        self._plot_replenishment_lag(plots_dir, final)
        self._plot_sku_demand_velocity(plots_dir, final)
        self._plot_cost_metrics(plots_dir, final)

    # ========== PLOT GENERATION ==========

    def _plot_latency_percentiles(self, output_dir: Path, final: bool = False):
        """Plot task latency p50, p95, p99."""
        if not self.task_logger or not self.task_logger.cost_data:
            return

        latencies = [e['latency'] for e in self.task_logger.cost_data if e.get('latency') is not None]
        if not latencies:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(self.metrics['latency']['p50'], color='green', linestyle='--',
                   label=f"P50: {self.metrics['latency']['p50']:.1f}s")
        ax.axvline(self.metrics['latency']['p95'], color='orange', linestyle='--',
                   label=f"P95: {self.metrics['latency']['p95']:.1f}s")
        ax.axvline(self.metrics['latency']['p99'], color='red', linestyle='--',
                   label=f"P99: {self.metrics['latency']['p99']:.1f}s")
        ax.set_xlabel('Latency (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Task Latency Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

        filename = '01_latency.png' if final else f'01_latency_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_robot_utilization(self, output_dir: Path, final: bool = False):
        """Plot robot status breakdown (IDLE, ACTIVE, WAITING)."""
        if not self.robot_samples:
            return

        statuses = defaultdict(int)
        for sample in self.robot_samples:
            statuses[sample['status']] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(statuses.keys())
        values = list(statuses.values())
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

        ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Sample Count')
        ax.set_title('Robot Utilization Status Distribution')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (label, value) in enumerate(zip(labels, values)):
            ax.text(i, value, str(value), ha='center', va='bottom', fontweight='bold')

        filename = '02_robot_utilization.png' if final else f'02_utilization_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_replenishment_lag(self, output_dir: Path, final: bool = False):
        """Plot replenishment pipeline breakdown — queue wait, assign→exec, execution."""
        lag = self.metrics.get('replenishment_lag', {})
        if not lag or lag.get('total', {}).get('mean', 0) == 0:
            return

        phases = ['Queue Wait', 'Assign→Exec', 'Execution']
        keys = ['queue_wait', 'assign_to_exec', 'execution']
        colors = ['#2E86AB', '#FFE66D', '#4ECDC4']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: mean per phase (stacked to show total)
        means = [lag.get(k, {}).get('mean', 0) for k in keys]
        bars = ax1.bar(phases, means, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Mean Duration (seconds)')
        ax1.set_title('Replenishment Pipeline — Mean Phase Duration', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.0f}s', ha='center', va='bottom', fontsize=10)

        # Right: p95 per phase
        p95s = [lag.get(k, {}).get('p95', 0) for k in keys]
        total = lag.get('total', {})
        bars2 = ax2.bar(phases, p95s, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('P95 Duration (seconds)')
        ax2.set_title(
            f'Replenishment Pipeline — P95  |  Total mean: {total.get("mean", 0):.0f}s  p95: {total.get("p95", 0):.0f}s',
            fontweight='bold'
        )
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, p95s):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.0f}s', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = '03_replenishment_lag.png' if final else f'03_lag_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_sku_demand_velocity(self, output_dir: Path, final: bool = False):
        """Plot SKU consumption rates."""
        if not self.metrics.get('sku_demand'):
            return

        sku_demand = self.metrics.get('sku_demand', {})

        if not sku_demand:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'SKU Demand Velocity\n(Insufficient data)',
                    ha='center', va='center', fontsize=12)
            ax.set_title('SKU Consumption Rates')
            ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            skus = list(sku_demand.keys())[:20]  # Top 20 SKUs
            rates = [sku_demand[sku] for sku in skus]

            ax.barh(skus, rates, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Consumption Rate (items/hour)')
            ax.set_title('SKU Demand Velocity (Top 20)')
            ax.grid(axis='x', alpha=0.3)

        filename = '04_sku_demand.png' if final else f'04_demand_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_cost_metrics(self, output_dir: Path, final: bool = False):
        """Plot cost breakdown — mean per-task cost split by component."""
        cost = self.metrics.get('cost', {})
        if not cost or cost.get('mean', 0) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: component breakdown (mean per task)
        components = ['Distance', 'Robot Hours', 'Energy']
        values = [
            cost.get('mean_distance', 0),
            cost.get('mean_robot_hours', 0),
            cost.get('mean_energy', 0),
        ]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax1.bar(components, values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Mean Cost Per Task ($)')
        ax1.set_title('Cost Breakdown by Component', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'${val:.2f}', ha='center', va='bottom', fontsize=10)

        # Right: summary stats
        labels = ['Mean', 'Min', 'Max']
        summary = [cost.get('mean', 0), cost.get('min', 0), cost.get('max', 0)]
        ax2.bar(labels, summary, color='#95E1D3', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Total Cost Per Task ($)')
        ax2.set_title(f'Cost Distribution  |  Cost/Reward: ${cost.get("mean_cost_per_reward", 0):.4f}',
                      fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, (label, val) in enumerate(zip(labels, summary)):
            ax2.text(i, val, f'${val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = '05_cost_metrics.png' if final else f'05_cost_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()


    def get_snapshot(self) -> dict:
        """Return current metrics for dashboard/logging."""
        task_count = len(self.task_logger.cost_data) if self.task_logger else 0
        return {
            'iteration': self.current_iteration,
            'metrics': self.metrics,
            'event_counts': {
                'tasks': task_count,
                'robot_samples': len(self.robot_samples),
                'sku_snapshots': len(self.sku_logger.latest_snapshots) if self.sku_logger else 0,
            }
        }
