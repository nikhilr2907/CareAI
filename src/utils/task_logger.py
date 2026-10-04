"""
Task-specific logging for GAPO training.
Tracks every assignment and completion with full metadata including SKU levels.
"""
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class TaskLogger:
    """Log individual task assignments and completions to text file.

    Note: on_time/late tracking is informational only (diagnostic).
    Reward computation is driven solely by inventory health, not deadline compliance.
    """

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
        self.iteration_completions = defaultdict(lambda: {'completed': 0, 'robots': defaultdict(int)})  # iter -> stats
        # Track by source
        self.source_breakdown = defaultdict(lambda: {'assigned': 0, 'completed': 0})
        # Track cost metrics
        self.cost_data = []  # List of {'source', 'reward', 'distance', 'robot_hours', 'energy', 'total_cost', 'cost_per_reward'}

    def log_assignment(self, task, assigned_robot: int, assignment_reward: float,
                      iteration: int, sim_time: float, num_assignments: int, buffer_size: int):
        """Log task assignment."""
        task_id = task.task_id
        source = getattr(task, 'source', 'unknown')

        # Store metadata for completion later
        self.task_metadata[task_id] = {
            'iteration': iteration,
            'arrival_time': task.arrival_time,
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
            'source': source,
            'task_type': getattr(task, 'task_type', None),
            'leg_type': getattr(task, 'leg_type', None),
        }

        # Track source breakdown
        self.source_breakdown[source]['assigned'] += 1

        # Log assignment event
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
            f"arrival={task.arrival_time:.1f}s "
            f"assign_reward={assignment_reward:.4f} {sku_info} "
            f"(buffer={buffer_size} assignments={num_assignments})"
        )

        # Track assignment for iteration summary
        self.iteration_assignments[iteration].append(assigned_robot)

    def log_completion(self, task_id: int, completion_reward: float,
                      actual_completion_time: Optional[float],
                      sim_time: float,
                      distance_traveled: float = 0.0,
                      energy_consumed: float = 0.0):
        """Log task completion with cost breakdown (inventory-driven rewards + cost tracking)."""
        if task_id not in self.task_metadata:
            # Task not in our log (maybe predates logging), but still log the completion
            actual_time_str = f"{actual_completion_time:.1f}s" if actual_completion_time is not None else "?"
            self.logger.warning(
                f"COMPLT task_id={task_id} [NO METADATA] sim_time={sim_time:.1f}s "
                f"completion_reward={completion_reward:.4f} actual_time={actual_time_str}"
            )
            return

        meta = self.task_metadata[task_id]
        total_reward = meta['assignment_reward'] + completion_reward
        source = meta.get('source', 'unknown')

        # Compute cost metrics
        robot_hours = (sim_time - meta['sim_time_assigned']) / 3600.0 if sim_time >= meta['sim_time_assigned'] else 0.0

        # Cost unit rates (can be configured)
        distance_cost_unit = 0.5  # $ per meter
        robot_hours_cost_unit = 50.0  # $ per hour
        energy_cost_unit = 0.10  # $ per Wh

        distance_cost = distance_traveled * distance_cost_unit
        robot_hours_cost = robot_hours * robot_hours_cost_unit
        energy_cost = energy_consumed * energy_cost_unit
        total_cost = distance_cost + robot_hours_cost + energy_cost
        cost_per_reward = total_cost / total_reward if total_reward > 0 else 0.0

        # Build SKU info string
        sku_info = f"sku_id={meta['sku_id'] or 'N/A'}"
        if meta['sku_stock_level_at_assign'] is not None and meta['sku_max_level'] is not None:
            stock_pct = (meta['sku_stock_level_at_assign'] / meta['sku_max_level'] * 100) if meta['sku_max_level'] > 0 else 0
            sku_info += f" stock_at_assign={meta['sku_stock_level_at_assign']:.1f}/{meta['sku_max_level']:.1f}({stock_pct:.0f}%)"
            if meta['reorder_point'] is not None:
                sku_info += f" reorder={meta['reorder_point']:.1f}"

        # Log completion event with cost breakdown
        actual_time_str = f"{actual_completion_time:.1f}s" if actual_completion_time is not None else "?"
        self.logger.info(
            f"COMPLT task_id={task_id} iter={meta['iteration']} sim_time={sim_time:.1f}s "
            f"robot={meta['assigned_robot']} source={source} location={meta['from_location_idx']}->{meta['to_location_idx'] or '?'} "
            f"time_from_assign={sim_time - meta['sim_time_assigned']:.1f}s "
            f"assign_reward={meta['assignment_reward']:.4f} completion_reward={completion_reward:.4f} "
            f"total_reward={total_reward:.4f} actual_time={actual_time_str} "
            f"cost=[dist:{distance_cost:.2f} robohrs:{robot_hours_cost:.2f} energy:{energy_cost:.2f}] "
            f"total_cost={total_cost:.2f} cost_per_reward={cost_per_reward:.4f} {sku_info}"
        )

        # Track completion for iteration summary
        iter_num = meta['iteration']
        robot = meta['assigned_robot']
        self.iteration_completions[iter_num]['completed'] += 1
        self.iteration_completions[iter_num]['robots'][robot] += 1

        # Track source breakdown
        self.source_breakdown[source]['completed'] += 1

        # Track cost data for analytics
        self.cost_data.append({
            'task_id': task_id,
            'source': source,
            'robot': robot,
            'distance_traveled': distance_traveled,
            'robot_hours': robot_hours,
            'energy_consumed': energy_consumed,
            'total_cost': total_cost,
            'completion_reward': completion_reward,
            'total_reward': total_reward,
            'cost_per_reward': cost_per_reward,
            'sim_time': sim_time,
            'latency': actual_completion_time,
            'arrival_time': meta.get('arrival_time'),
            'from_location_idx': meta.get('from_location_idx'),
            'to_location_idx': meta.get('to_location_idx'),
            'task_type': meta.get('task_type'),
            'leg_type': meta.get('leg_type'),
            'iteration': meta.get('iteration'),
        })

        # Clean up
        del self.task_metadata[task_id]

    def log_iteration_summary(self, iteration: int):
        """Log summary of iteration completions and robot breakdown."""
        if iteration not in self.iteration_completions:
            return  # No completions in this iteration

        stats = self.iteration_completions[iteration]
        assignments = self.iteration_assignments[iteration]

        total_completed = stats['completed']

        if total_completed == 0:
            return  # No completions logged

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
            f"  Robot Completion Breakdown: {robot_breakdown}\n"
            f"{'=' * 60}"
        )

    def log_running_summary(self, iteration: int):
        """Log cumulative running summary across all iterations."""
        # Aggregate stats across all iterations up to current
        total_assigned_all = sum(len(self.iteration_assignments[i]) for i in range(iteration + 1))

        total_completed_all = 0
        robot_totals = defaultdict(int)

        for i in range(iteration + 1):
            if i in self.iteration_completions:
                stats = self.iteration_completions[i]
                total_completed_all += stats['completed']
                for robot, count in stats['robots'].items():
                    robot_totals[robot] += count

        assignment_to_completion_ratio = total_assigned_all / total_completed_all if total_completed_all > 0 else 0

        # Robot breakdown
        robot_breakdown = ", ".join(
            f"R{robot}:{count}"
            for robot, count in sorted(robot_totals.items())
        )

        # Source breakdown
        source_info = "\n".join(
            f"    {source}: {self.source_breakdown[source]['assigned']} assigned, "
            f"{self.source_breakdown[source]['completed']} completed"
            for source in sorted(self.source_breakdown.keys())
        )

        # Cost metrics
        if self.cost_data:
            avg_cost = np.mean([d['total_cost'] for d in self.cost_data])
            avg_reward = np.mean([d['total_reward'] for d in self.cost_data])
            avg_cost_per_reward = np.mean([d['cost_per_reward'] for d in self.cost_data])
            cost_info = f"  Avg Cost Per Task: ${avg_cost:.2f} | Avg Reward: {avg_reward:.2f} | Cost Per Reward: ${avg_cost_per_reward:.4f}\n"
        else:
            cost_info = "  Avg Cost Per Task: N/A (no cost data)\n"

        # Log running summary
        self.logger.info(
            f"=== RUNNING SUMMARY (Iterations 0-{iteration}) ===\n"
            f"  Total Assigned: {total_assigned_all} | Total Completed: {total_completed_all} | A:C Ratio: {assignment_to_completion_ratio:.2f}:1\n"
            f"  By Source:\n{source_info}\n"
            f"{cost_info}"
            f"  Robot Completion Breakdown: {robot_breakdown}\n"
            f"{'=' * 60}"
        )

    def plot_metrics(self, output_dir: Path):
        """Generate and save visualization plots from in-memory metrics."""
        output_dir = Path(output_dir)
        metrics_dir = output_dir / "metrics" if not output_dir.name.startswith('metrics') else output_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)

        max_iteration = max(self.iteration_completions.keys()) if self.iteration_completions else 0
        iterations = list(range(max_iteration + 1))

        # Plot 1: Source breakdown over iterations (stacked bar)
        self._plot_source_breakdown(iterations, metrics_dir)

        # Plot 2: Completion rate by source
        self._plot_completion_rates(metrics_dir)

        # Plot 3: Factoriser fallback rate
        self._plot_fallback_rate(iterations, metrics_dir)

        # Plot 4: A:C ratio trend + robot utilization
        self._plot_ac_ratio_and_robots(iterations, metrics_dir)

        # Plot 5: Cost metrics
        if self.cost_data:
            self._plot_cost_metrics(metrics_dir)

        self.logger.info(f"Metrics plots saved to {metrics_dir}")

    def _plot_source_breakdown(self, iterations, output_dir):
        """Plot source breakdown (deterministic/factoriser/fallback) over iterations."""
        sources = sorted(self.source_breakdown.keys())
        source_by_iter = {source: [] for source in sources}

        for it in iterations:
            for source in sources:
                # Count assignments in this iteration
                if it in self.iteration_assignments:
                    # Need to track per-source per-iteration; for now use cumulative
                    source_by_iter[source].append(self.source_breakdown[source]['assigned'])

        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(iterations))

        colors = {'deterministic': '#2E86AB', 'factoriser': '#A23B72',
                  'random_adhoc_fallback': '#F18F01', 'unknown': '#C73E1D'}

        for source in sources:
            counts = source_by_iter[source]
            color = colors.get(source, '#888888')
            ax.bar(iterations, counts, label=source, bottom=bottom, color=color, alpha=0.8)
            bottom += np.array(counts)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cumulative Assignments', fontsize=12)
        ax.set_title('Task Source Breakdown Over Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "01_source_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_completion_rates(self, output_dir):
        """Plot completion rate (completed/assigned) by source."""
        sources = sorted(self.source_breakdown.keys())
        assigned = []
        completed = []
        labels = []

        for source in sources:
            a = self.source_breakdown[source]['assigned']
            c = self.source_breakdown[source]['completed']
            if a > 0:
                rate = c / a
                assigned.append(a)
                completed.append(c)
                labels.append(f"{source}\n({c}/{a}={rate:.1%})")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(sources)]

        # Left: Counts
        x_pos = np.arange(len(labels))
        ax1.bar(x_pos, assigned, label='Assigned', color=colors, alpha=0.7)
        ax1.bar(x_pos, completed, label='Completed', color=colors, alpha=1.0)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Assigned vs Completed by Source', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Right: Completion rates
        rates = [c / a if a > 0 else 0 for a, c in zip(assigned, completed)]
        bars = ax2.bar(x_pos, rates, color=colors, alpha=0.8)
        ax2.set_ylabel('Completion Rate', fontsize=11)
        ax2.set_ylim([0, 1.1])
        ax2.set_title('Completion Rate by Source', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s for s in sources], fontsize=10)
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add percentage labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / "02_completion_rates.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_fallback_rate(self, iterations, output_dir):
        """Plot factoriser fallback rate: fallback / (fallback + factoriser)."""
        fallback_count = 0
        factoriser_count = 0
        fallback_rates = []

        for it in iterations:
            fallback_count += self.source_breakdown.get('random_adhoc_fallback', {}).get('assigned', 0) if it == 0 else 0
            factoriser_count += self.source_breakdown.get('factoriser', {}).get('assigned', 0) if it == 0 else 0

        # Compute overall fallback rate
        fallback_total = self.source_breakdown.get('random_adhoc_fallback', {}).get('assigned', 0)
        factoriser_total = self.source_breakdown.get('factoriser', {}).get('assigned', 0)
        total_attempts = fallback_total + factoriser_total

        fallback_rate = fallback_total / total_attempts if total_attempts > 0 else 0

        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Factoriser\nSuccess', 'Fallback to\nRandom']
        counts = [factoriser_total, fallback_total]
        colors_fb = ['#2E86AB', '#F18F01']

        bars = ax.bar(categories, counts, color=colors_fb, alpha=0.8, width=0.6)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Factoriser Success vs Fallback Rate ({fallback_rate:.1%} fallback)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "03_fallback_rate.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_ac_ratio_and_robots(self, iterations, output_dir):
        """Plot A:C ratio trend and final robot utilization breakdown."""
        ac_ratios = []
        robot_totals = defaultdict(int)

        for it in iterations:
            if it in self.iteration_completions:
                stats = self.iteration_completions[it]
                assigned = len(self.iteration_assignments.get(it, []))
                completed = stats['completed']

                ratio = assigned / completed if completed > 0 else 0
                ac_ratios.append(ratio)

                for robot, count in stats['robots'].items():
                    robot_totals[robot] += count
            else:
                ac_ratios.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: A:C ratio trend
        ax1.plot(iterations, ac_ratios, marker='o', linewidth=2, markersize=4, color='#2E86AB', alpha=0.8)
        ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Ideal (1:1)')
        ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Caution (1.5:1)')
        ax1.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Critical (2:1)')
        ax1.fill_between(iterations, ac_ratios, 1.0, where=np.array(ac_ratios) >= 1.0,
                         alpha=0.2, color='blue', label='Queue buildup')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('A:C Ratio', fontsize=11)
        ax1.set_title('Assignment:Completion Ratio Over Training', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right: Robot utilization breakdown
        robots = sorted(robot_totals.keys())
        completions = [robot_totals[r] for r in robots]
        colors_robot = plt.cm.Set3(np.linspace(0, 1, len(robots)))

        bars = ax2.bar([f'R{r}' for r in robots], completions, color=colors_robot, alpha=0.8)
        ax2.set_ylabel('Total Completions', fontsize=11)
        ax2.set_title('Robot Utilization Breakdown', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add count labels
        for bar, count in zip(bars, completions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "04_ac_ratio_and_robots.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_cost_metrics(self, output_dir):
        """Plot cost breakdown and cost efficiency metrics."""
        if not self.cost_data:
            return

        # Extract data by source
        source_costs = defaultdict(lambda: {'costs': [], 'rewards': [], 'cost_per_reward': []})
        for entry in self.cost_data:
            source = entry['source']
            source_costs[source]['costs'].append(entry['total_cost'])
            source_costs[source]['rewards'].append(entry['total_reward'])
            source_costs[source]['cost_per_reward'].append(entry['cost_per_reward'])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        sources = sorted(source_costs.keys())
        colors_cost = ['#2E86AB', '#A23B72', '#F18F01'][:len(sources)]

        # Plot 1: Average cost per source
        avg_costs = [np.mean(source_costs[s]['costs']) for s in sources]
        ax1.bar(sources, avg_costs, color=colors_cost, alpha=0.8)
        ax1.set_ylabel('Average Cost ($)', fontsize=11)
        ax1.set_title('Average Cost Per Task by Source', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, (src, cost) in enumerate(zip(sources, avg_costs)):
            ax1.text(i, cost, f'${cost:.2f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Cost vs Reward scatter (color by source)
        for source, color in zip(sources, colors_cost):
            ax2.scatter(source_costs[source]['rewards'], source_costs[source]['costs'],
                       label=source, alpha=0.6, s=50, color=color)
        ax2.set_xlabel('Total Reward', fontsize=11)
        ax2.set_ylabel('Total Cost ($)', fontsize=11)
        ax2.set_title('Cost vs Reward (Efficiency Frontier)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cost per reward unit (efficiency)
        avg_cost_per_reward = [np.mean(source_costs[s]['cost_per_reward']) for s in sources]
        bars = ax3.bar(sources, avg_cost_per_reward, color=colors_cost, alpha=0.8)
        ax3.set_ylabel('Cost Per Unit Reward ($/reward)', fontsize=11)
        ax3.set_title('Cost Efficiency by Source (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for bar, cpr in zip(bars, avg_cost_per_reward):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${cpr:.4f}', ha='center', va='bottom', fontsize=9)

        # Plot 4: Cost component breakdown (avg across all tasks)
        cost_components = defaultdict(float)
        component_counts = defaultdict(int)
        for entry in self.cost_data:
            distance_cost = entry['distance_traveled'] * 0.5
            energy_cost = entry['energy_consumed'] * 0.10
            robot_hours_cost = entry['robot_hours'] * 50.0

            cost_components['Distance'] += distance_cost
            cost_components['Robot Hours'] += robot_hours_cost
            cost_components['Energy'] += energy_cost
            component_counts['total'] += 1

        if component_counts['total'] > 0:
            for key in cost_components:
                cost_components[key] /= component_counts['total']

        components = list(cost_components.keys())
        values = list(cost_components.values())
        colors_comp = ['#2E86AB', '#A23B72', '#F18F01']

        wedges, texts, autotexts = ax4.pie(values, labels=components, autopct='%1.1f%%',
                                            colors=colors_comp, startangle=90)
        ax4.set_title('Average Cost Breakdown Across All Tasks', fontsize=12, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(output_dir / "05_cost_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()