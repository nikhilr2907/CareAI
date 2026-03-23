"""
Real-time analytics collector for training and deployment.
Consumes RobotState, RobotTelemetry, HospitalNode objects.
Task completion data read from TaskLogger.cost_data.
No data duplication - reads directly from environment objects.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# Type hints for objects we'll receive
from src.environment.robot.robot_state import RobotState
from src.environment.graph.node import HospitalNode
from src.environment.graph.edge import HospitalEdge


class RealtimeAnalyticsCollector:
    """
    Collects and aggregates metrics from simulation/real deployment.

    Data flow:
    - Task completion data from TaskLogger.cost_data
    - RobotState/RobotTelemetry → record_robot_telemetry()
    - HospitalNode → record_inventory_snapshot()
    - HospitalEdge → record_corridor_state()

    Outputs:
    - metrics.json (raw events)
    - metrics_summary.csv (per-iteration aggregates)
    - plots_final/*.png (6 matplotlib plots)
    """

    def __init__(self,
                 output_dir: Path,
                 mode: str = 'sim',
                 snapshot_frequency: int = 50,
                 task_logger=None):
        """
        Args:
            output_dir: Directory for metrics output (analytics/ subdir)
            mode: 'sim' or 'real' (for logging/configuration)
            snapshot_frequency: Generate plots every N iterations
            task_logger: Reference to TaskLogger for reading task completion data
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.snapshot_frequency = snapshot_frequency
        self.task_logger = task_logger

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ===== Flow Metrics Infrastructure =====
        self.nodes = None                    # Set by record_inventory_snapshot()
        self.metric_helpers = None           # Set by _initialize_metrics()
        self.metric_modules: Dict = {}       # {metric_name: metric_instance}
        self.flow_metrics_cache: Dict = {}   # Cached results from last compute()

        # Raw event streams (append-only)
        # Note: task completion data now read from task_logger.cost_data
        self.robot_samples = []    # {robot_id, status, battery, position, assigned_tasks}
        self.inventory_snapshots = []  # {location, sku, stock, max_stock}
        self.corridor_samples = []  # {edge, robot_count, clutter, people}

        # Iteration tracking
        self.current_iteration = 0
        self.iteration_summaries = defaultdict(dict)  # iteration → summary stats

        # Aggregated metrics (updated per _update_metrics())
        self.metrics = {
            'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0},
            'utilization': {'idle': 0, 'active': 0, 'waiting': 0},
            'replenishment_lag': {'mean': 0, 'max': 0},
            'sku_demand': {},  # sku → consumption_rate
            'cost': {'mean': 0, 'min': 0, 'max': 0},
            'failure_rate': 0.0,
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

    def record_inventory_snapshot(self, nodes: List[HospitalNode], current_time: float):
        """
        Sample inventory levels from HospitalNode objects.
        Called periodically during training (e.g., every 50 steps).

        On first call: store nodes and initialize flow metrics.

        Args:
            nodes: List of HospitalNode objects
            current_time: Current simulation/real time
        """
        # Initialize metrics on first call (capture nodes + floor layout)
        if self.nodes is None:
            self.nodes = nodes
            self._initialize_metrics()

        for node in nodes:
            # SKU-level inventory if available
            if hasattr(node, 'sku_inventory') and node.sku_inventory:
                for sku_id, sku_data in node.sku_inventory.items():
                    stock = sku_data.get('stock', 0)
                    max_stock = sku_data.get('max_stock', 1)
                    self.inventory_snapshots.append({
                        'location_id': node.node_id,
                        'sku_id': sku_id,
                        'stock': stock,
                        'max_stock': max_stock,
                        'stock_ratio': stock / max_stock if max_stock > 0 else 0,
                        'consumption_rate': node.consumption_rate,
                        'timestamp': current_time,
                    })

    def record_corridor_state(self, edges: List[HospitalEdge], current_time: float):
        """
        Sample corridor congestion from HospitalEdge objects.
        Called periodically (e.g., every 50 steps).

        Args:
            edges: List of HospitalEdge objects
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

    def record_cost_per_task(self, task_id: int, total_cost: float, total_reward: float):
        """
        Record cost breakdown for a task (from task_logger).

        Args:
            task_id: Task ID
            total_cost: Total cost in dollars
            total_reward: Total reward
        """
        # Store for later aggregation (cost computation happens in task_logger)
        self.task_events_with_cost = getattr(self, 'task_events_with_cost', {})
        self.task_events_with_cost[task_id] = {
            'cost': total_cost,
            'reward': total_reward,
            'cost_per_reward': total_cost / total_reward if total_reward > 0 else 0,
        }

    # ========== FLOW METRICS ORCHESTRATION ==========

    def _initialize_metrics(self):
        """Instantiate all flow metric modules (called on first inventory snapshot)."""
        try:
            from src.analytics.metrics import (
                MetricHelpers, ZoneLatencyMetric, RoutePerformanceMetric,
                TaskCompletionDistributionMetric, EfficiencyComparisonMetric,
            )
            self.metric_helpers = MetricHelpers(self.nodes)
            self.metric_modules = {
                'zone_latency': ZoneLatencyMetric(self.task_logger, self.nodes, self.metric_helpers),
                'route_performance': RoutePerformanceMetric(self.task_logger, self.nodes, self.metric_helpers),
                'task_completion_dist': TaskCompletionDistributionMetric(self.task_logger, self.nodes, self.metric_helpers),
                'efficiency_comparison': EfficiencyComparisonMetric(self.task_logger, self.nodes, self.metric_helpers),
            }
        except Exception:
            self.metric_modules = {}

    def compute_flow_metrics(self) -> Dict:
        """
        Orchestrate computation of all flow metrics.

        Returns:
            {
                'zone_latency': {...},
                'route_performance': {...},
                'task_completion_dist': {...},
                'efficiency_comparison': {...},
            }
        """
        if not self.metric_modules:
            return {}

        result = {}
        for name, metric in self.metric_modules.items():
            try:
                result[name] = metric.compute()
            except Exception:
                result[name] = {}

        self.flow_metrics_cache = result
        return result

    # ========== METRICS COMPUTATION ==========

    def _update_metrics(self):
        """Recompute aggregated metrics from raw events."""

        # 1. Task latency percentiles (from task_logger.cost_data)
        if self.task_logger and self.task_logger.cost_data:
            latencies = [e['latency'] for e in self.task_logger.cost_data if e.get('latency') is not None]
            if latencies:
                self.metrics['latency'] = {
                    'p50': float(np.percentile(latencies, 50)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                    'mean': float(np.mean(latencies)),
                }
        else:
            self.metrics['latency'] = {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0}

        # 2. Robot utilization breakdown
        if self.robot_samples:
            statuses = defaultdict(int)
            for sample in self.robot_samples:
                statuses[sample['status']] += 1
            total = sum(statuses.values())
            self.metrics['utilization'] = {
                'idle': int(statuses.get('IDLE', 0)),
                'active': int(statuses.get('ACTIVE', 0)),
                'waiting': int(statuses.get('WAITING_AT_LOCATION', 0)),
            }

        # 3. SKU demand (consumption rate)
        if self.inventory_snapshots:
            sku_stats = defaultdict(list)
            for snap in self.inventory_snapshots:
                sku_stats[snap['sku_id']].append(snap['consumption_rate'])
            self.metrics['sku_demand'] = {
                sku: float(np.mean(rates))
                for sku, rates in sku_stats.items()
            }

        # 4. Failure rate (no failure tracking in cost_data yet, set to 0)
        self.metrics['failure_rate'] = 0.0

    def increment_iteration(self):
        """Call at end of each training iteration to track progress."""
        self.current_iteration += 1

        # Store iteration summary
        task_count = len(self.task_logger.cost_data) if self.task_logger else 0
        self.iteration_summaries[self.current_iteration] = {
            'task_count': task_count,
            'robot_samples': len(self.robot_samples),
            'latency_p95': self.metrics['latency'].get('p95', 0),
            'failure_rate': self.metrics['failure_rate'],
        }

    # ========== OUTPUT GENERATION ==========

    def save_metrics_json(self):
        """Save all raw events and flow metrics to JSON for later analysis."""
        output = {
            'mode': self.mode,
            'iteration': self.current_iteration,
            'task_completions': self.task_logger.cost_data if self.task_logger else [],
            'robot_samples': self.robot_samples,
            'inventory_snapshots': self.inventory_snapshots,
            'corridor_samples': self.corridor_samples,
            'metrics': self.metrics,
            'flow_metrics': self.flow_metrics_cache,
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
        self._plot_failure_recovery(plots_dir, final)

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
        """Placeholder: Replenishment lag analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Replenishment Lag Analysis\n(Data source: task completion tracking)',
                ha='center', va='center', fontsize=12)
        ax.set_title('Replenishment Lag (Task Assignment to Completion)')
        ax.axis('off')

        filename = '03_replenishment_lag.png' if final else f'03_lag_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_sku_demand_velocity(self, output_dir: Path, final: bool = False):
        """Plot SKU consumption rates."""
        if not self.inventory_snapshots:
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
        """Plot cost analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        cost_data = self.metrics.get('cost', {})

        if cost_data and any(cost_data.values()):
            labels = list(cost_data.keys())
            values = list(cost_data.values())
            ax.bar(labels, values, alpha=0.8, edgecolor='black', color='#95E1D3')
            ax.set_ylabel('Cost ($)')
            ax.set_title('Cost Metrics')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Cost Metrics\n(Computed from task logger)',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Cost Metrics')
            ax.axis('off')

        filename = '05_cost_metrics.png' if final else f'05_cost_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_failure_recovery(self, output_dir: Path, final: bool = False):
        """Plot failure rate and recovery metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))

        failure_rate = self.metrics.get('failure_rate', 0)
        success_rate = 1.0 - failure_rate

        categories = ['Success', 'Failure']
        values = [success_rate * 100, failure_rate * 100]
        colors = ['#4ECDC4', '#FF6B6B']

        wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax.set_title(f'Robot Failure Rate ({failure_rate:.2%})')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        filename = '06_failure_recovery.png' if final else f'06_failure_iter{self.current_iteration}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def get_snapshot(self) -> dict:
        """Return current metrics for dashboard/logging."""
        task_count = len(self.task_logger.cost_data) if self.task_logger else 0
        return {
            'iteration': self.current_iteration,
            'metrics': self.metrics,
            'flow_metrics': self.flow_metrics_cache,
            'event_counts': {
                'tasks': task_count,
                'robot_samples': len(self.robot_samples),
                'inventory_snapshots': len(self.inventory_snapshots),
            }
        }
