from dataclasses import dataclass
from pathlib import Path
from typing import List
from collections import defaultdict, deque
import json
import numpy as np
import matplotlib.pyplot as plt

# Type hints for objects we'll receive
from src.environment.robot.robot_state import RobotState
from src.environment.graph.node import GraphNode


class RealtimeAnalyticsCollector:
    """
    Collects and aggregates derived metrics from simulation or real deployment.

    Data flow:
    - Task completion data from TaskLogger.cost_data
    - RobotState and RobotTelemetry via record_robot_telemetry()
    - SKU event and snapshot data from SKULogger

    Outputs:
    - metrics.json (derived metrics only)
    - plots/*.png and plots_final/*.png
    """

    def __init__(self,
                 output_dir: Path,
                 mode: str = 'sim',
                 snapshot_frequency: int = 50,
                 task_logger=None,
                 sku_logger=None,
                 robot_sample_logger=None):
        """
        Args:
            output_dir: Directory for metrics output (analytics/ subdir)
            mode: 'sim' or 'real' (for logging/configuration)
            snapshot_frequency: Generate plots every N iterations
            task_logger: Reference to TaskLogger for reading task completion data
            sku_logger: Reference to SKULogger for reading SKU event data
            robot_sample_logger: Reference to RobotSampleLogger for raw robot samples
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.snapshot_frequency = snapshot_frequency
        self.task_logger = task_logger
        self.sku_logger = sku_logger
        self.robot_sample_logger = robot_sample_logger

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.nodes = None  # Set by initialize()

        # Keep only a bounded recent window of robot samples per robot in memory.
        self.robot_samples = defaultdict(lambda: deque(maxlen=1000))

        # Iteration tracking
        self.current_iteration = 0

        # Aggregated metrics (updated per _update_metrics())
        self.metrics = {
            'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0},
            'utilization': {'idle': 0, 'active': 0, 'waiting': 0},
            'replenishment_lag': {'mean': 0, 'max': 0},
            'cost': {'mean': 0, 'min': 0, 'max': 0},
        }

    # ========== RECORDING METHODS ==========

    def record_robot_telemetry(self, robots: List[RobotState], current_time: float,
                               iteration: int = None, step: int = None):
        """
        Sample robot telemetry from RobotState + RobotTelemetry.
        Called periodically during training (e.g., every 10 steps).

        Args:
            robots: List of RobotState objects
            current_time: Current simulation/real time
        """
        if self.robot_sample_logger is None:
            raise RuntimeError("RealtimeAnalyticsCollector.record_robot_telemetry() requires robot_sample_logger")

        for robot in robots:
            if robot.telemetry is not None:
                sample = self.robot_sample_logger.log_sample(
                    robot, current_time, iteration=iteration, step=step
                )
                self.robot_samples[robot.robot_id].append(sample)

    def initialize(self, nodes: List[GraphNode]):
        """Store graph nodes for future metric modules. Call once after env creation."""
        self.nodes = nodes

    # ========== METRICS COMPUTATION ==========

    def _iter_robot_samples(self):
        for samples in self.robot_samples.values():
            for sample in samples:
                yield sample

    def _robot_sample_count(self) -> int:
        return sum(len(samples) for samples in self.robot_samples.values())

    def _update_metrics(self):
        """Recompute all aggregated metrics from raw event sources."""
        self._compute_latency()
        self._compute_utilization()
        self._compute_cost()
        self._compute_replenishment_lag()
        self._compute_sku_demand()
        self._compute_stockout_metrics()
        self._compute_zone_latency()

    def _compute_zone_latency(self):
        if not (self.nodes and self.task_logger):
            return
        from src.analytics.metrics import ZoneLatencyMetric, MetricHelpers
        self.metrics['zone_latency'] = ZoneLatencyMetric(
            self.task_logger, self.nodes, MetricHelpers(self.nodes)
        ).compute()

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
        for sample in self._iter_robot_samples():
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

    def _compute_stockout_metrics(self):
        if not self.sku_logger:
            return

        stockouts = self.sku_logger.stockout_events   # one dict per stockout onset
        restocks = self.sku_logger.restock_events      # one dict per robot dropoff restock

        if not stockouts:
            self.metrics['stockouts'] = {'total': 0}
            return

        # ------------------------------------------------------------------
        # Build a restock lookup: (node_idx, sku_id) → sorted list of sim_times
        # at which a restock was delivered. Sorted so we can find the first
        # restock *after* a given stockout onset with a simple list scan.
        # ------------------------------------------------------------------
        restock_times = defaultdict(list)
        for ev in restocks:
            restock_times[(ev['node_idx'], ev['sku_id'])].append(ev['sim_time'])
        for times in restock_times.values():
            times.sort()

        # ------------------------------------------------------------------
        # Helper: compute summary stats over a list of numeric values.
        # Returns zeros when empty so callers never need to guard.
        # ------------------------------------------------------------------
        def _agg(vals):
            if not vals:
                return {'mean': 0, 'p95': 0, 'max': 0, 'count': 0}
            arr = np.array(vals, dtype=float)
            return {
                'mean': round(float(arr.mean()), 2),
                'p95': round(float(np.percentile(arr, 95)), 2),
                'max': round(float(arr.max()), 2),
                'count': len(vals),
            }

        # ------------------------------------------------------------------
        # Level 1 — per (node_tag, sku_id) pair: finest granularity.
        # Identifies chronic offenders: a specific SKU at a specific shelf
        # that repeatedly runs out. Time-to-restock here tells us how quickly
        # the robot system responds to that exact location/item combination.
        # ------------------------------------------------------------------
        pair_counts = defaultdict(int)       # (node_tag, sku_id) -> total stockouts
        pair_ttr = defaultdict(list)         # (node_tag, sku_id) -> [gap_seconds, ...]

        for ev in stockouts:
            pair_key = (ev['node_tag'], ev['sku_id'])
            pair_counts[pair_key] += 1

            # Find the earliest restock for this (node_idx, sku_id) that arrived
            # at or after this stockout — that gap is the shelf's empty duration.
            lookup_key = (ev['node_idx'], ev['sku_id'])
            t_out = ev['sim_time']
            future = [t for t in restock_times[lookup_key] if t >= t_out]
            if future:
                pair_ttr[pair_key].append(min(future) - t_out)

        # Sort by stockout frequency descending so the worst offenders are first
        by_pair = {
            f"{sku_id}@{node_tag}": {
                'stockout_count': pair_counts[(node_tag, sku_id)],
                'time_to_restock': _agg(pair_ttr.get((node_tag, sku_id), [])),
            }
            for node_tag, sku_id in sorted(pair_counts, key=lambda k: -pair_counts[k])
        }

        # ------------------------------------------------------------------
        # Level 2 — per zone (spatial quadrant: SW/SE/NW/NE).
        # Aggregates all stockouts across all SKUs within each floor quadrant.
        # Reveals whether one area of the hospital is systematically under-
        # served by the robot fleet, regardless of which SKU is affected.
        # Only computed when nodes are available (initialize() was called).
        # ------------------------------------------------------------------
        by_zone = {}
        if self.nodes:
            from src.analytics.metrics import MetricHelpers
            helpers = MetricHelpers(self.nodes)

            zone_counts = defaultdict(int)   # zone_label -> total stockouts
            zone_ttr = defaultdict(list)     # zone_label -> [gap_seconds, ...]

            for ev in stockouts:
                # Map this node's physical position to a quadrant label
                zone_label = helpers.quadrant_label(
                    helpers.get_location_quadrant(ev['node_idx'])
                )
                zone_counts[zone_label] += 1

                lookup_key = (ev['node_idx'], ev['sku_id'])
                t_out = ev['sim_time']
                future = [t for t in restock_times[lookup_key] if t >= t_out]
                if future:
                    zone_ttr[zone_label].append(min(future) - t_out)

            by_zone = {
                zone: {
                    'stockout_count': zone_counts[zone],
                    'time_to_restock': _agg(zone_ttr.get(zone, [])),
                }
                for zone in sorted(zone_counts, key=lambda z: -zone_counts[z])
            }

        # ------------------------------------------------------------------
        # Level 3 — per category.
        # Groups by supply category (e.g. PPE, medication, linen). Answers:
        # which *type* of item runs out most, and how fast does the system
        # recover per category. Category is stored directly on each event
        # so no additional lookup is needed.
        # ------------------------------------------------------------------
        cat_counts = defaultdict(int)        # category -> total stockouts
        cat_ttr = defaultdict(list)          # category -> [gap_seconds, ...]

        for ev in stockouts:
            cat = ev.get('category') or 'unknown'
            cat_counts[cat] += 1

            lookup_key = (ev['node_idx'], ev['sku_id'])
            t_out = ev['sim_time']
            future = [t for t in restock_times[lookup_key] if t >= t_out]
            if future:
                cat_ttr[cat].append(min(future) - t_out)

        by_category = {
            cat: {
                'stockout_count': cat_counts[cat],
                'time_to_restock': _agg(cat_ttr.get(cat, [])),
            }
            for cat in sorted(cat_counts, key=lambda c: -cat_counts[c])
        }

        # ------------------------------------------------------------------
        # All three levels share the same shape per key:
        # { stockout_count: int, time_to_restock: {mean, p95, max, count} }
        # ------------------------------------------------------------------
        self.metrics['stockouts'] = {
            'total': len(stockouts),
            'by_pair': by_pair,
            'by_zone': by_zone,
            'by_category': by_category,
        }

    def increment_iteration(self):
        """Call at end of each training iteration to track progress."""
        self.current_iteration += 1

    # ========== OUTPUT GENERATION ==========

    def save_metrics_json(self):
        """Save derived analytics metrics to JSON."""
        output = {
            'mode': self.mode,
            'iteration': self.current_iteration,
            'metrics': self.metrics,
        }

        json_path = self.output_dir / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

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
        self._plot_zone_latency(plots_dir, final)

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
        for sample in self._iter_robot_samples():
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

    def _plot_zone_latency(self, output_dir: Path, final: bool = False):
        """Plot zone-latency summaries by destination quadrant."""
        zone_latency = self.metrics.get('zone_latency', {})
        by_zone = zone_latency.get('by_zone', {}) if zone_latency else {}
        if not by_zone:
            return

        zones = [z for z in ['SW', 'SE', 'NW', 'NE'] if by_zone.get(z, {}).get('count', 0) > 0]
        if not zones:
            return

        mean_vals = [by_zone[z].get('mean', 0.0) for z in zones]
        p95_vals = [by_zone[z].get('p95', 0.0) for z in zones]
        counts = [by_zone[z].get('count', 0) for z in zones]
        percent_loss = [by_zone[z].get('percent_loss', 0.0) for z in zones]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['#2E86AB', '#F18F01', '#A23B72', '#4ECDC4'][:len(zones)]

        bars1 = ax1.bar(zones, mean_vals, color=colors, alpha=0.85, edgecolor='black')
        ax1.set_ylabel('Mean Latency (seconds)')
        ax1.set_title('Zone Latency by Destination Quadrant — Mean', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val, count, loss in zip(bars1, mean_vals, counts, percent_loss):
            label = f'{val:.1f}s\nn={count}'
            if loss:
                label += f'\n+{loss:.1f}%'
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha='center',
                va='bottom',
                fontsize=9,
            )

        bars2 = ax2.bar(zones, p95_vals, color=colors, alpha=0.85, edgecolor='black')
        ax2.set_ylabel('P95 Latency (seconds)')
        ax2.set_title('Zone Latency by Destination Quadrant — P95', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar, val, count in zip(bars2, p95_vals, counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{val:.1f}s\nn={count}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

        # TODO: Add a by_route heatmap using zone_latency['by_route'] once we decide
        # whether the route-pair view should be shown as mean, p95, or both.
        plt.tight_layout()
        filename = '06_zone_latency.png' if final else f'06_zone_latency_iter{self.current_iteration}.png'
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
                'robot_samples': self._robot_sample_count(),
                'sku_snapshots': len(self.sku_logger.latest_snapshots) if self.sku_logger else 0,
            }
        }
